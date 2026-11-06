"""在 val 上做联合搜索（差分进化），再在 test 上评估。

判定：可部署的推理期校准（val 拟合）能否在 benign FPR <= 0.008 内，
把未处理模型的 LFS 从 0.5791 拉近训练期处理的水平（+0.3950）。
"""
import json
import numpy as np
from scipy.optimize import differential_evolution

RES = "results/proba/"
BASE = RES + "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p"
CAP = 0.008
LESION = [9, 10, 11]


def lfs_and_fpr(p, y, w):
    pred = np.argmax(p * w, axis=1)
    f1s = []
    for c in LESION:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    fpr = np.mean(pred[y == 0] != 0)
    return float(sum(f1s)), float(fpr)


def macro_f1(p, y, w):
    pred = np.argmax(p * w, axis=1)
    out = []
    for c in range(12):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(out))


pv = np.load(BASE + "_val.npy")
yv = np.load(BASE + "_val_ytrue.npy")
pt = np.load(BASE + "_test.npy")
yt = np.load(BASE + "_test_ytrue.npy")

base_lfs, base_fpr = lfs_and_fpr(pt, yt, np.ones(12))
print(f"基线 test: LFS={base_lfs:.4f} FPR={base_fpr:.4f}")

BOUNDS = [(0.1, 10.0)] * 12
evals = {"n": 0}


def objective(w):
    evals["n"] += 1
    lfs, fpr = lfs_and_fpr(pv, yv, w)
    return -lfs + 1000.0 * max(0.0, fpr - CAP)


res = differential_evolution(
    objective, BOUNDS, maxiter=250, popsize=15, tol=1e-6,
    seed=0, polish=True, workers=1, updating="immediate",
)
w = res.x
val_lfs, val_fpr = lfs_and_fpr(pv, yv, w)
test_lfs, test_fpr = lfs_and_fpr(pt, yt, w)
test_macro = macro_f1(pt, yt, w)
print(f"val 拟合: val LFS={val_lfs:.4f} FPR={val_fpr:.4f}")
print(f"test 评估: LFS={test_lfs:.4f} ΔLFS={test_lfs-base_lfs:+.4f} "
      f"FPR={test_fpr:.4f} macro-F1={test_macro:.4f} 评估次数={evals['n']}")
print("权重:", [round(float(x), 4) for x in w])

json.dump({
    "protocol": "DE on val, objective LFS, hard cap benign FPR<=0.008, evaluated once on test",
    "data": BASE,
    "baseline_test": {"LFS": base_lfs, "benign_fpr": base_fpr},
    "val_fit": {"LFS": val_lfs, "benign_fpr": val_fpr},
    "test_eval": {"LFS": test_lfs, "delta_LFS": test_lfs - base_lfs,
                  "benign_fpr": test_fpr, "macro_f1": test_macro},
    "weights": [float(x) for x in w],
    "n_evals": evals["n"],
}, open("results/oracle_search_lfs_valfit.json", "w", encoding="utf-8"), indent=2)
print("written: results/oracle_search_lfs_valfit.json")
