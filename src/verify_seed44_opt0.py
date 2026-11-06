"""Recompute seed-44 restart-0 from the stored weight vector, to test whether its
test LFS really equals seed-42's untreated baseline (0.579084) or whether that
value leaked in from the wrong array."""
import json
from pathlib import Path
import numpy as np

R = str(Path(__file__).resolve().parent.parent / "results")
P = R + "/proba"
LESION = [9, 10, 11]
N = 12


def load(seed, split):
    pre = f"{P}/deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p_{split}"
    return np.load(pre + ".npy"), np.load(pre + "_ytrue.npy")


def metrics(p, y, w=None):
    q = p if w is None else p * w
    pred = np.argmax(q, axis=1)
    cm = np.zeros((N, N), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    fpr = (cm[:, 0].sum() - cm[0, 0]) / (cm.sum() - cm[0, :].sum())
    return {
        "LFS": float(f1[LESION].sum()),
        "macro": float(f1.mean()),
        "fpr": float(fpr),
        "lesion_f1": [round(float(x), 4) for x in f1[LESION]],
        "support": [int(x) for x in cm.sum(1)[LESION]],
    }


j = json.load(open(R + "/valfit_joint_3seed.json"))
w44 = np.array(j["runs"][4]["weights"])  # model_seed 44, opt_seed 0
assert j["runs"][4]["model_seed"] == 44 and j["runs"][4]["opt_seed"] == 0

print("=== seed 42 A3 test (reference) ===")
p42, y42 = load(42, "test")
m42 = metrics(p42, y42)
print(m42)

print("\n=== seed 44 A3 test (its own baseline) ===")
p44, y44 = load(44, "test")
m44 = metrics(p44, y44)
print(m44)

print("\n=== seed 44 A3 test + opt0 weights (claimed 0.579084) ===")
m44w = metrics(p44, y44, w44)
print(m44w)

print("\n=== seed 46 A3 test ===")
p46, y46 = load(46, "test")
print(metrics(p46, y46))

print("\n=== cross-check: apply seed-44 weights to seed-42 array ===")
print(metrics(p42, y42, w44))

print("\n--- verdict ---")
print(f"seed44+opt0 recomputed LFS = {m44w['LFS']:.6f}; claimed 0.579084")
print(f"seed42 baseline LFS        = {m42['LFS']:.6f}")
print(f"arrays identical?          = {np.array_equal(p44, p42)}")
print(f"max|p44-p42|               = {np.abs(p44 - p42).max():.3e}")
print(f"ytrue identical?           = {np.array_equal(y44, y42)}")
print(f"argmax differs from own A3?= {not np.array_equal(np.argmax(p44*w44,1), np.argmax(p44,1))}")
print(f"n rows changed by weights  = {int((np.argmax(p44*w44,1)!=np.argmax(p44,1)).sum())}")
