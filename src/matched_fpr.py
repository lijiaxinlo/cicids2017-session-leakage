# -*- coding: utf-8 -*-
"""Matched-operating-point check for Table II.

Table II compares four representations at their own argmax, whose benign FPR
spans 0.00021--0.0333 (a factor of 158). This script re-scores each forest at a
*common* benign FPR using the paper's own decision rule (argmax of w_c p_c) with
only the benign weight moved, so representation and threshold are no longer
confounded.

Usage:
    python src/matched_fpr.py --views 78 19 --seeds 42
"""
import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import data_pipeline as dp
from pooled_view_rf import pool_to_19

ROOT = Path(__file__).resolve().parent.parent
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")


def score(proba, y_te, w0, n_classes):
    adj = proba
    if w0 != 1.0:
        adj = proba.copy()
        adj[:, 0] *= w0
    pred = adj.argmax(axis=1)
    cm = np.bincount(y_te * n_classes + pred,
                     minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    # Pr(yhat != BENIGN | y = BENIGN)
    fpr = float((cm[0].sum() - cm[0, 0]) / cm[0].sum())
    lfs = float(sum(f1[i] for i, c in enumerate(_CLASSES) if c in LESION))
    return fpr, lfs, float(f1.mean()), {c: float(f1[i]) for i, c in enumerate(_CLASSES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["78", "19"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--npz", default="results/feats_p2_convfeat.npz")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.00021, 0.0027, 0.008])
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--out", default="results/matched_fpr.json")
    args = ap.parse_args()

    global _CLASSES
    _CLASSES = dp.load_class_names()
    n_classes = len(_CLASSES)
    y = dp.load_y()
    tr = np.load(ROOT / "results/temporal/train_idx.npy")
    te = np.load(ROOT / "results/temporal/test_idx.npy")
    y_tr, y_te = y[tr].astype(np.int64), y[te].astype(np.int64)
    z = np.load(ROOT / args.npz)

    get = {
        "78": lambda: (np.ascontiguousarray(dp.load_X()[tr]),
                       np.ascontiguousarray(dp.load_X()[te])),
        "19": lambda: (np.ascontiguousarray(pool_to_19(dp.load_X()[tr])),
                       np.ascontiguousarray(pool_to_19(dp.load_X()[te]))),
        "128": lambda: (np.ascontiguousarray(z["train_ctx"]),
                        np.ascontiguousarray(z["test_ctx"])),
    }

    grid = np.unique(np.concatenate([np.geomspace(0.02, 0.9, 40),
                                     np.linspace(0.92, 1.08, 17),
                                     np.geomspace(1.1, 800, 90)]))

    out = {}
    for view in args.views:
        Xtr, Xte = get[view]()
        for seed in args.seeds:
            t0 = time.time()
            clf = RandomForestClassifier(n_estimators=args.n_estimators,
                                         max_depth=None, n_jobs=-1,
                                         random_state=seed, class_weight=None)
            clf.fit(Xtr, y_tr)
            proba = clf.predict_proba(Xte)
            f0, l0, m0, c0 = score(proba, y_te, 1.0, n_classes)
            row = {"view": view, "seed": seed, "fit_seconds": round(time.time() - t0, 1),
                   "argmax": {"fpr": f0, "lfs": l0, "macro_f1": m0, "bot": c0["Bot"]}}
            sweep = []
            for w0 in grid:
                f, l, m, c = score(proba, y_te, float(w0), n_classes)
                sweep.append((f, l, m, float(w0), c["Bot"]))
            for t in args.targets:
                cand = [s for s in sweep if s[0] <= t]
                row[f"fpr<={t}"] = (None if not cand else
                                    dict(zip(("fpr", "lfs", "macro_f1", "w0", "bot"),
                                             max(cand, key=lambda s: s[0]))))
            out[f"{view}_seed{seed}"] = row
            print(json.dumps(row, ensure_ascii=False), flush=True)
            del clf, proba

    p = ROOT / args.out
    prev = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    prev.update(out)
    p.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print("写出", p)


if __name__ == "__main__":
    main()
