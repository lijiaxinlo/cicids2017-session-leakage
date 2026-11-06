"""Independently recompute all 12 valfit_joint_3seed runs from the stored weight
vectors, and compare against the recorded JSON. Guards the 3-seed table before it
goes into the manuscript."""
import json
from pathlib import Path
import numpy as np

R = str(Path(__file__).resolve().parent.parent / "results")
P = R + "/proba"
LESION = [9, 10, 11]
N = 12

j = json.load(open(R + "/valfit_joint_3seed.json"))

cache = {}


def load(seed, split):
    k = (seed, split)
    if k not in cache:
        pre = f"{P}/deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p_{split}"
        cache[k] = (np.load(pre + ".npy"), np.load(pre + "_ytrue.npy"))
    return cache[k]


def metrics(p, y, w=None):
    q = p if w is None else p * w
    pred = np.argmax(q, axis=1)
    cm = np.zeros((N, N), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    benign_fpr = (cm[0, :].sum() - cm[0, 0]) / cm[0, :].sum()
    return float(f1[LESION].sum()), float(f1.mean()), float(benign_fpr), f1[LESION]


print(f"{'seed':>4} {'opt':>3} | {'LFS rec':>8} {'LFS json':>8} {'d':>8} | "
      f"{'mac rec':>8} {'mac json':>8} {'d':>8} | {'fpr rec':>8} {'fpr json':>8}")
print("-" * 108)
bad = 0
for r in j["runs"]:
    s, o = r["model_seed"], r["opt_seed"]
    w = np.array(r["weights"])
    p, y = load(s, "test")
    lfs, mac, fpr, f1s = metrics(p, y, w)
    dl, dm = lfs - r["test"]["LFS"], mac - r["test"]["macro_f1"]
    flag = "" if (abs(dl) < 5e-4 and abs(dm) < 5e-4) else "  <<< MISMATCH"
    if flag:
        bad += 1
    print(f"{s:>4} {o:>3} | {lfs:8.6f} {r['test']['LFS']:8.6f} {dl:+8.1e} | "
          f"{mac:8.6f} {r['test']['macro_f1']:8.6f} {dm:+8.1e} | "
          f"{fpr:8.6f} {r['test']['benign_fpr']:8.6f}{flag}")
    print(f"       lesion F1 = {[round(float(x),4) for x in f1s]}")

print("\n--- baselines (w = 1) ---")
for ps in j["per_seed"]:
    s = ps["model_seed"]
    p, y = load(s, "test")
    lfs, mac, fpr, f1s = metrics(p, y)
    print(f"seed {s}: LFS {lfs:.6f} (json {ps['a3_test']['LFS']:.6f})  "
          f"macro {mac:.6f} (json {ps['a3_test']['macro_f1']:.6f})  "
          f"fpr {fpr:.6f} (json {ps['a3_test']['benign_fpr']:.6f})  "
          f"lesion F1 {[round(float(x),4) for x in f1s]}")

print(f"\n{mis} mismatching run(s)" if (mis := bad) else "\nall 12 runs reproduce")
