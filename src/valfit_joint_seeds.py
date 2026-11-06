"""val 拟合联合搜索的优化器种子重复，用于估计该估计量的方差。

与 valfit_joint_search.py 完全同协议，仅改 differential_evolution 的 seed，
以回答：0.8035 这个可部署 LFS 是稳定值还是单次优化的运气。
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
        d = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2 * tp / d)
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
print(f"baseline test LFS={base_lfs:.4f} FPR={base_fpr:.4f}", flush=True)

BOUNDS = [(0.1, 10.0)] * 12
rows = []


def make_objective():
    def objective(w):
        lfs, fpr = lfs_and_fpr(pv, yv, w)
        return -lfs + 1000.0 * max(0.0, fpr - CAP)
    return objective


for s in (1, 2, 3):
    res = differential_evolution(
        make_objective(), BOUNDS, maxiter=250, popsize=15, tol=1e-6,
        seed=s, polish=True, workers=1, updating="immediate",
    )
    w = res.x
    v_lfs, v_fpr = lfs_and_fpr(pv, yv, w)
    t_lfs, t_fpr = lfs_and_fpr(pt, yt, w)
    t_macro = macro_f1(pt, yt, w)
    print(f"opt-seed {s}: val LFS={v_lfs:.4f} FPR={v_fpr:.4f} | "
          f"test LFS={t_lfs:.4f} dLFS={t_lfs - base_lfs:+.4f} "
          f"FPR={t_fpr:.4f} macro={t_macro:.4f}", flush=True)
    rows.append({"opt_seed": s, "val": {"LFS": v_lfs, "benign_fpr": v_fpr},
                 "test": {"LFS": t_lfs, "delta_LFS": t_lfs - base_lfs,
                          "benign_fpr": t_fpr, "macro_f1": t_macro},
                 "weights": [float(x) for x in w]})

t = [r["test"]["LFS"] for r in rows]
print(f"\ntest LFS across optimizer seeds 1-3: {[round(x, 4) for x in t]}")
print(f"mean={np.mean(t):.4f} sd={np.std(t, ddof=1):.4f} "
      f"min={min(t):.4f} max={max(t):.4f}")

json.dump({"protocol": "DE on val, LFS objective, cap 0.008, optimizer-seed repeats",
           "baseline_test": {"LFS": base_lfs, "benign_fpr": base_fpr},
           "runs": rows,
           "summary": {"mean": float(np.mean(t)), "sd": float(np.std(t, ddof=1)),
                       "min": float(min(t)), "max": float(max(t))}},
          open("results/oracle_search_lfs_valfit_seeds.json", "w",
               encoding="utf-8"), indent=2)
print("written: results/oracle_search_lfs_valfit_seeds.json")
