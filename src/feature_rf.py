# -*- coding: utf-8 -*-
"""随机森林接在网络导出的表征上（2x2 的"同表征、换家族"格）。

配合 src/export_conv_feats.py 使用：那里导出 conv_max / ctx，这里把同一个随机森林
（100 棵、深度不限、无类别权重）接上去，其余一切与 Table II 的 Forest 行相同。

用法
----
    python src/feature_rf.py --npz results/feats_p2_convfeat.npz --key ctx
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import data_pipeline as dp
import eval_utils as eu

ROOT = Path(__file__).resolve().parent.parent
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--key", required=True, help="conv_max / ctx")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--split-dir", default="results/temporal")
    args = ap.parse_args()

    class_names = dp.load_class_names()
    y = dp.load_y()
    tr = np.load(ROOT / args.split_dir / "train_idx.npy")
    te = np.load(ROOT / args.split_dir / "test_idx.npy")
    z = np.load(ROOT / args.npz)
    X_tr = np.ascontiguousarray(z[f"train_{args.key}"])
    X_te = np.ascontiguousarray(z[f"test_{args.key}"])
    assert len(X_tr) == len(tr) and len(X_te) == len(te)
    y_tr, y_te = y[tr].astype(np.int64), y[te].astype(np.int64)
    print(f"{args.key}: train {X_tr.shape} test {X_te.shape}", flush=True)

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=None,
                                     n_jobs=-1, random_state=seed, class_weight=None)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        m = eu.compute_metrics(y_te, y_pred, class_names=class_names,
                               labels=list(range(len(class_names))))
        lfs = sum(m["per_class"][c]["f1"] for c in LESION)
        # 良性误报率按论文 §III-A 的定义：Pr(yhat != BENIGN | y = BENIGN) = fn/(fn+tp)。
        # 注意不能用 m["per_class"]["BENIGN"]["fpr"] —— eval_utils 那个是 one-vs-rest
        # 的 FP/(FP+TN)，对 BENIGN 而言等于 Pr(yhat = BENIGN | y != BENIGN)，方向相反。
        # 2026-10-24 修：与 fill_table2.py:51 的口径对齐。
        fpr = 1.0 - m["per_class"]["BENIGN"]["recall"]
        print(f"seed {seed}: macro-F1 {m['macro_f1']:.4f}  LFS {lfs:.4f}  "
              f"fpr {fpr:.6f}  ({time.time() - t0:.0f}s)", flush=True)
        runs.append({"seed": seed, "macro_f1": m["macro_f1"], "lesion_sum": lfs,
                     "benign_fpr": fpr,
                     "per_class_f1": {c: m["per_class"][c]["f1"] for c in class_names}})

    def agg(k):
        v = [r[k] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    summary = {k: dict(zip(("mean", "std"), agg(k)))
               for k in ("macro_f1", "lesion_sum", "benign_fpr")}
    per_class = {c: {"f1": dict(zip(("mean", "std"),
                                   (float(np.mean([r["per_class_f1"][c] for r in runs])),
                                    float(np.std([r["per_class_f1"][c] for r in runs])))))}
                 for c in class_names}
    out = {"model": "RandomForest", "features": args.key, "npz": args.npz,
           "split": args.split_dir, "seeds": args.seeds,
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "n_features": int(X_tr.shape[1]), "n_estimators": args.n_estimators,
           "class_weight": None, "timestamp": datetime.now().isoformat(timespec="seconds"),
           "summary": summary, "per_class_summary": per_class, "per_seed": runs}
    path = ROOT / "results" / f"rf_feat_{args.key}_metrics.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {path}")
    print(f"macro-F1 {summary['macro_f1']['mean']:.4f} +- {summary['macro_f1']['std']:.4f} | "
          f"LFS {summary['lesion_sum']['mean']:.4f} +- {summary['lesion_sum']['std']:.4f} | "
          f"fpr {summary['benign_fpr']['mean']:.6f}")


if __name__ == "__main__":
    main()
