"""为 Table 3 计算 C4'（处理臂 + LFS 目标联合搜索，val 最优重启）的完整指标列。

Table 3 的列集是 macro-F1 / macro-Recall / FPR(macro-OVR) / benign FPR / LFS，
而 valfit_treated_arm.json 只落了 macro-F1、benign FPR、LFS。这里从冻结权重
重新构造 12x12 混淆矩阵，补齐 macro-Recall 与 macro-OVR FPR，并顺带复算
C3、C4 以校验与已发表 Table 3 一致。

用法：python src/c4prime_table3.py
输出：results/c4prime_table3.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROBA = RES / "proba"

N = 12
LESION = [9, 10, 11]
MODEL_SEEDS = (42, 44, 46)
TREATED_PATTERNS = [
    "deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull_g1_bfull",
    "deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull",
]


def load(patterns, seed):
    for tmpl in patterns:
        base = PROBA / tmpl.format(seed=seed)
        files = {k: Path(f"{base}_{k}.npy") for k in ("test", "test_ytrue")}
        if all(p.exists() for p in files.values()):
            return {k: np.load(p) for k, p in files.items()}, base.name
    raise SystemExit(f"seed {seed} 找不到 proba")


def metrics(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(N), where=(tp + fn) > 0)
    # 一对余：TN_c = total - TP_c - FP_c - FN_c
    tn = cm.sum() - tp - fp - fn
    ovr = np.divide(fp, fp + tn, out=np.zeros(N), where=(fp + tn) > 0)
    row0 = cm[0].sum()
    return {"macro_f1": float(f1.mean()), "macro_recall": float(rec.mean()),
            "macro_ovr_fpr": float(ovr.mean()),
            "benign_fpr": float((row0 - cm[0, 0]) / row0),
            "LFS": float(f1[LESION].sum())}


def main():
    d = json.load(open(RES / "valfit_treated_arm.json", encoding="utf-8"))
    summary = {s["seed"]: s for s in d["summary"]}
    out = {"meta": {"script": "src/c4prime_table3.py",
                    "note": "C4' = treated arm, joint DE, objective LFS, val-best restart"},
           "per_seed": []}

    for seed in MODEL_SEEDS:
        td, name = load(TREATED_PATTERNS, seed)
        y = td["test_ytrue"].astype(np.int64)
        p = td["test"].astype(np.float64)
        best_opt = summary[seed]["lfsjoint_val_best_opt"]
        run = next(r for r in d["runs"]
                   if r["model_seed"] == seed and r["opt_seed"] == best_opt)
        rec = {"seed": seed, "proba": name, "opt_seed": best_opt}

        for tag, w in (("C3", np.ones(N)),
                       ("C4", None),
                       ("C4prime", np.asarray(run["weights"], dtype=np.float64))):
            if tag == "C4":
                continue
            pred = np.argmax(p * w, axis=1)
            cm = np.bincount(y * N + pred, minlength=N * N).reshape(N, N)
            rec[tag] = metrics(cm)

        out["per_seed"].append(rec)
        print(f"[seed {seed} opt {best_opt}] C4' {rec['C4prime']}", flush=True)
        print(f"[seed {seed}] C3 {rec['C3']}", flush=True)

    keys = ["macro_f1", "macro_recall", "macro_ovr_fpr", "benign_fpr", "LFS"]
    agg = {}
    for tag in ("C3", "C4prime"):
        agg[tag] = {}
        for k in keys:
            v = np.array([r[tag][k] for r in out["per_seed"]])
            agg[tag][k] = {"mean": round(float(v.mean()), 4),
                           "sd": round(float(v.std(ddof=1)), 4)}
    out["aggregate"] = agg
    json.dump(out, open(RES / "c4prime_table3.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("\n=== Table 3 rows (mean ± sd, n=3) ===")
    for tag in ("C3", "C4prime"):
        a = agg[tag]
        print(f"{tag:8s} macro-F1 {a['macro_f1']['mean']:.4f} ± {a['macro_f1']['sd']:.4f} | "
              f"macro-Rec {a['macro_recall']['mean']:.4f} ± {a['macro_recall']['sd']:.4f} | "
              f"FPR {a['macro_ovr_fpr']['mean']:.4f} ± {a['macro_ovr_fpr']['sd']:.4f} | "
              f"benign {a['benign_fpr']['mean']:.4f} ± {a['benign_fpr']['sd']:.4f} | "
              f"LFS {a['LFS']['mean']:.4f} ± {a['LFS']['sd']:.4f}")


if __name__ == "__main__":
    main()
