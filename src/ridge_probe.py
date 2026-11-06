"""脊线直测：在未处理臂上把 w_XSS 抬到边界上限，看 XSS F1 是否出现、
以及 Brute Force F1 是否等量下降。

目的：稿件原写 "the model emits no XSS positive for a weight to promote"，
但测量显示 XSS 在 w∈[0.1,10] 内是**可达**的（约 3550/756219 行满足
p_XSS/p_other > 0.01）。因此 XSS F1 = 0 的真实原因不是"没有正例可抬"，
而是"抬 XSS 会等量压低 Brute Force"——本脚本把这个权衡量化。

用法：python src/ridge_probe.py
输出：results/ridge_probe.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROBA = RES / "proba"
N = 12
BOT, BF, XSS = 9, 10, 11
LESION = [BOT, BF, XSS]
FACTORS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def metrics(p, y, w):
    pred = np.argmax(p * w, axis=1)
    cm = np.zeros((N, N), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
    row0 = cm[0].sum()
    return f1, float((row0 - cm[0, 0]) / row0)


def load_untreated(seed):
    base = PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p"
    p = np.load(f"{base}_test.npy")
    y = np.load(f"{base}_test_ytrue.npy")
    return p, y


def main():
    j = json.load(open(RES / "valfit_joint_3seed.json", encoding="utf-8"))
    best = {}
    for s in (42, 44, 46):
        rs = [r for r in j["runs"] if r["model_seed"] == s]
        b = max(rs, key=lambda r: r["val"]["LFS"])
        best[s] = (b["opt_seed"], np.asarray(b["weights"], dtype=float))

    out = {"note": "w_XSS scaled from the val-best joint vector; all other weights fixed",
           "factors": FACTORS, "per_seed": []}
    print(f"{'seed':>4} {'factor':>7} | {'w_XSS':>7} | {'XSS F1':>7} {'BF F1':>7} {'Bot F1':>7} "
          f"{'LFS':>7} {'dLFS':>8} {'benignFPR':>9}")
    print("-" * 82)
    for s in (42, 44, 46):
        p, y = load_untreated(s)
        os_, w0 = best[s]
        f1_0, _ = metrics(p, y, w0)
        base_lfs = float(f1_0[LESION].sum())
        rows = []
        for fac in FACTORS:
            w = w0.copy()
            w[XSS] = min(10.0, w0[XSS] * fac)
            f1, fpr = metrics(p, y, w)
            lfs = float(f1[LESION].sum())
            r = {"factor": fac, "w_XSS": round(float(w[XSS]), 4),
                 "xss_f1": round(float(f1[XSS]), 4), "bf_f1": round(float(f1[BF]), 4),
                 "bot_f1": round(float(f1[BOT]), 4), "LFS": round(lfs, 4),
                 "dLFS": round(lfs - base_lfs, 4), "benign_fpr": round(fpr, 6)}
            rows.append(r)
            print(f"{s:>4} {fac:>7.0f} | {w[XSS]:>7.3f} | {r['xss_f1']:>7.4f} {r['bf_f1']:>7.4f} "
                  f"{r['bot_f1']:>7.4f} {r['LFS']:>7.4f} {r['dLFS']:>+8.4f} {r['benign_fpr']:>9.6f}")
        out["per_seed"].append({"seed": s, "opt_seed": os_,
                                "base_xss_f1": round(float(f1_0[XSS]), 4),
                                "base_bf_f1": round(float(f1_0[BF]), 4),
                                "base_LFS": round(base_lfs, 4), "sweep": rows})
        print()

    # 汇总：w_XSS 推到上限时，XSS 与 BF 的变动
    agg = []
    for ps in out["per_seed"]:
        top = ps["sweep"][-1]
        agg.append({"seed": ps["seed"], "dXSS": round(top["xss_f1"] - ps["base_xss_f1"], 4),
                    "dBF": round(top["bf_f1"] - ps["base_bf_f1"], 4),
                    "dLFS": top["dLFS"]})
    out["aggregate"] = agg
    print("=== w_XSS 推到上限 10 时的变动 ===")
    for a in agg:
        print(f"  seed {a['seed']}: dXSS={a['dXSS']:+.4f}  dBF={a['dBF']:+.4f}  dLFS={a['dLFS']:+.4f}")
    json.dump(out, open(RES / "ridge_probe.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("\nwritten:", RES / "ridge_probe.json")


if __name__ == "__main__":
    main()
