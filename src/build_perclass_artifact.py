"""落盘按类分解 artifact：四条臂在三个种子上的逐类 F1。

臂的定义（全部 test 口径，权重拟合只用 val）：
  A3_argmax        未处理 + 均匀权重
  A3_calibrated    未处理 + DE(LFS 目标) 的 val-best  —— 校准臂
  C3_argmax        处理   + 均匀权重
  C4_published     处理   + 坐标上升(macro 目标)      —— 稿件 C4
  A0_lfsjoint      处理   + DE(LFS 目标) 的 val-best  —— 与校准臂目标一致

输出：results/valfit_joint_3seed_perclass.json
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


def load(patterns, seed):
    for t in patterns:
        b = PROBA / t.format(seed=seed)
        f = {k: Path(f"{b}_{k}.npy") for k in ("test", "test_ytrue")}
        if all(p.exists() for p in f.values()):
            return np.load(f["test"]), np.load(f["test_ytrue"])
    raise SystemExit(f"seed {seed}: 找不到 {patterns}")


TREATED = ["deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull_g1_bfull",
           "deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull"]
UNTREATED = ["deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p"]


def per_class(p, y, w):
    pred = np.argmax(p * w, axis=1)
    cm = np.zeros((N, N), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
    r0 = cm[0].sum()
    return {"Bot": round(float(f1[BOT]), 4), "BruteForce": round(float(f1[BF]), 4),
            "XSS": round(float(f1[XSS]), 4), "LFS": round(float(f1[LESION].sum()), 4),
            "macro_f1": round(float(f1.mean()), 4),
            "benign_fpr": round(float((r0 - cm[0, 0]) / r0), 6)}


def val_best(j, seed):
    rs = [r for r in j["runs"] if r["model_seed"] == seed]
    return np.asarray(max(rs, key=lambda r: r["val"]["LFS"])["weights"], dtype=float)


def main():
    jj = json.load(open(RES / "valfit_joint_3seed.json", encoding="utf-8"))
    ta = json.load(open(RES / "valfit_treated_arm.json", encoding="utf-8"))
    c4 = {a["seed"]: a for a in json.load(
        open(RES / "e17_calibration_summary.json", encoding="utf-8"))["c4"]
        if abs(a["tier_benign_cap"] - 0.008) < 1e-12}

    arms = {"A3_argmax": [], "A3_calibrated": [], "C3_argmax": [],
            "C4_published": [], "A0_lfsjoint": []}
    for seed in (42, 44, 46):
        pu, yu = load(UNTREATED, seed)
        pt, yt = load(TREATED, seed)
        wcal = val_best(jj, seed)
        wc4 = np.asarray(c4[seed]["w"], dtype=float)
        wtr = np.asarray(next(s for s in ta["runs"]
                              if s["model_seed"] == seed and s["opt_seed"] ==
                              next(x for x in ta["summary"] if x["seed"] == seed)["lfsjoint_val_best_opt"])
                         ["weights"], dtype=float)
        for name, p, y, w in (("A3_argmax", pu, yu, np.ones(N)),
                              ("A3_calibrated", pu, yu, wcal),
                              ("C3_argmax", pt, yt, np.ones(N)),
                              ("C4_published", pt, yt, wc4),
                              ("A0_lfsjoint", pt, yt, wtr)):
            d = per_class(p, y, w)
            d["seed"] = seed
            arms[name].append(d)

    means = {}
    for name, rows in arms.items():
        means[name] = {k: round(float(np.mean([r[k] for r in rows])), 4)
                       for k in ("Bot", "BruteForce", "XSS", "LFS", "macro_f1")}
        means[name]["benign_fpr"] = round(float(np.mean([r["benign_fpr"] for r in rows])), 6)

    # 匹配目标的对照（唯一公平的组间比较）
    matched = {
        "untreated_calibrated": means["A3_calibrated"],
        "treated_calibrated": means["A0_lfsjoint"],
        "delta_Bot": round(means["A0_lfsjoint"]["Bot"] - means["A3_calibrated"]["Bot"], 4),
        "delta_BruteForce": round(means["A0_lfsjoint"]["BruteForce"] - means["A3_calibrated"]["BruteForce"], 4),
        "delta_XSS": round(means["A0_lfsjoint"]["XSS"] - means["A3_calibrated"]["XSS"], 4),
        "delta_LFS": round(means["A0_lfsjoint"]["LFS"] - means["A3_calibrated"]["LFS"], 4),
    }
    base = round(float(np.mean([r["LFS"] for r in arms["A3_argmax"]])), 4)
    matched["untreated_baseline_LFS"] = base
    matched["treatment_gain"] = round(means["A0_lfsjoint"]["LFS"] - base, 4)
    matched["calibration_gain"] = round(means["A3_calibrated"]["LFS"] - base, 4)
    matched["recovery_fraction"] = round(matched["calibration_gain"] / matched["treatment_gain"], 4)
    matched["XSS_share_of_gap"] = round(matched["delta_XSS"] / matched["delta_LFS"], 4)

    out = {"note": "per-class F1 on the test split; weights fitted on validation only",
           "lesion_order": ["Bot", "WebAttack-BruteForce", "WebAttack-XSS"],
           "arms": arms, "means": means, "matched_objective_comparison": matched}
    json.dump(out, open(RES / "valfit_joint_3seed_perclass.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"{'arm':<16} {'Bot':>7} {'BF':>7} {'XSS':>7} {'LFS':>7} {'macro':>7} {'FPR':>8}")
    print("-" * 64)
    for name in ("A3_argmax", "A3_calibrated", "C3_argmax", "C4_published", "A0_lfsjoint"):
        m = means[name]
        print(f"{name:<16} {m['Bot']:>7.4f} {m['BruteForce']:>7.4f} {m['XSS']:>7.4f} "
              f"{m['LFS']:>7.4f} {m['macro_f1']:>7.4f} {m['benign_fpr']:>8.5f}")
    print()
    print("=== 匹配目标（DE + LFS，val-best of 4，3 种子） ===")
    for k in ("delta_Bot", "delta_BruteForce", "delta_XSS", "delta_LFS",
              "treatment_gain", "calibration_gain", "recovery_fraction", "XSS_share_of_gap"):
        print(f"  {k:<22} {matched[k]:+.4f}")
    print("\nwritten:", RES / "valfit_joint_3seed_perclass.json")


if __name__ == "__main__":
    main()
