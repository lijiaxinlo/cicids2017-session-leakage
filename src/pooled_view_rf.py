# -*- coding: utf-8 -*-
"""
2x2 的"树 x 19 步池化"格（第三轮审读 §五 第 1 格，决定性的一格）
=================================================================

论文的中心论断是"resampling 的大小是表征的属性"。原来只有对角线两格：

             78 维原始      19 步池化
    树        1.010          缺
    神经       缺           0.448

两者同时换了家族和表征，所以只支持"森林比序列网络好"，支持不了"差别来自表征"。
本脚本填"树 x 19 步池化"这一格：家族不变（同一个随机森林、同样的树数与深度），
只把输入从 78 维换成 19 步池化视图。

19 步视图的定义是**无权重**的：对 78 维做两次宽度 2 的最大池化（78 -> 39 -> 19），
与 conv 主干在位置上做的降维完全一致，只是用 max 代替学出来的卷积核。这样这一格
不依赖任何训练好的权重，"表征"这一个因素被单独换掉。

用法
----
    python src/pooled_view_rf.py                 # 无处理 + 处理，各 5 种子
    python src/pooled_view_rf.py --seeds 42      # 冒烟

产出
----
    results/rf_pooled19_p2_metrics.json         无处理（对应 Table II 的 Forest / none）
    results/rf_pooled19_p2_treat_metrics.json   与深度臂对齐的处理（Forest / resampled）
    results/runs/rf_pooled19_*_seed<seed>.json  逐种子记录
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
OUT_DIR = ROOT / "results"
RUN_DIR = OUT_DIR / "runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pool_to_19(X):
    """78 -> 39 -> 19，两次宽度 2、步长 2 的最大池化。

    与 conv 主干的位置降维逐位同构：MaxPool1d(2) 在长度为 39 的序列上取
    floor((39-2)/2)+1 = 19 个窗口，末位被丢弃 —— 也就是说模型的 19 步视图里
    不包含第 77、78 个原始特征。这里照抄这个行为，不修正。
    """
    n, f = X.shape
    assert f == 78, f"期望 78 维，得到 {f}"
    a = X.reshape(n, 39, 2).max(axis=2)           # 78 -> 39
    b = a[:, :38].reshape(n, 19, 2).max(axis=2)   # 39 -> 19（末位丢弃，同 MaxPool1d）
    assert b.shape[1] == 19, b.shape
    return b


def resample(tr, y, class_names, benign_target, lesion_target, seed=42):
    rng = np.random.default_rng(seed)
    benign_id = class_names.index("BENIGN")
    lesion_ids = [class_names.index(c) for c in LESION]
    keep = []
    for c in range(len(class_names)):
        idx_c = tr[y[tr] == c]
        if c == benign_id and benign_target and len(idx_c) > benign_target:
            idx_c = np.sort(rng.choice(idx_c, size=benign_target, replace=False))
        elif c in lesion_ids and lesion_target and len(idx_c) < lesion_target:
            extra = rng.choice(idx_c, size=lesion_target - len(idx_c), replace=True)
            idx_c = np.concatenate([idx_c, extra])
        keep.append(idx_c)
    return np.sort(np.concatenate(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--view", default="pooled19", choices=["pooled19", "raw"])
    ap.add_argument("--split", default="temporal", choices=["temporal", "grouped"])
    ap.add_argument("--benign-target", type=int, default=None)
    ap.add_argument("--lesion-target", type=int, default=None)
    args = ap.parse_args()

    class_names = dp.load_class_names()
    y = dp.load_y()
    tsd = ROOT / "results" / ("temporal" if args.split == "temporal" else "temporal_grouped")
    tr = np.load(tsd / "train_idx.npy")
    te = np.load(tsd / "test_idx.npy")
    X = dp.load_X(mmap=True)
    log(f"数据 X {X.shape} y {y.shape}；训练 {len(tr):,} / 测试 {len(te):,}")

    if args.benign_target or args.lesion_target:
        tr = resample(tr, y, class_names, args.benign_target, args.lesion_target)
        log(f"处理后训练集 {len(tr):,} 行")

    X_tr = np.ascontiguousarray(X[tr])
    X_te = np.ascontiguousarray(X[te])
    if args.view == "pooled19":
        X_tr, X_te = pool_to_19(X_tr), pool_to_19(X_te)
        log(f"19 步池化视图: train {X_tr.shape} test {X_te.shape}")
    y_tr, y_te = y[tr].astype(np.int64), y[te].astype(np.int64)

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=None,
                                     n_jobs=-1, random_state=seed, class_weight=None)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        m = eu.compute_metrics(y_te, y_pred, class_names=class_names,
                               labels=list(range(len(class_names))))
        m["seed"] = seed
        m["elapsed_sec"] = round(time.time() - t0, 1)
        lfs = sum(m["per_class"][c]["f1"] for c in LESION)
        # 良性误报率按论文 §III-A 定义：Pr(yhat != BENIGN | y = BENIGN) = fn/(fn+tp)。
        # eval_utils 的 per_class["BENIGN"]["fpr"] 是 one-vs-rest 的 FP/(FP+TN)，
        # 对 BENIGN 而言方向相反（攻击被判成 benign 的比例）。
        # 2026-10-24 修：与 fill_table2.py:51 对齐。
        m["benign_fpr"] = 1.0 - m["per_class"]["BENIGN"]["recall"]
        log(f"seed {seed}: macro-F1 {m['macro_f1']:.4f}  LFS {lfs:.4f}  "
            f"fpr {m['benign_fpr']:.6f}  ({m['elapsed_sec']}s)")
        runs.append((m, lfs))
        with open(RUN_DIR / f"rf_{args.view}_p2{'_treat' if args.benign_target else ''}"
                            f"_seed{seed}.json", "w", encoding="utf-8") as fp:
            json.dump({"metrics": {k: v for k, v in m.items()
                                   if not isinstance(v, dict)},
                       "per_class": m["per_class"],
                       "n_train": int(len(tr)), "n_test": int(len(te)),
                       "view": args.view, "split": args.split,
                       "benign_target": args.benign_target,
                       "lesion_target": args.lesion_target},
                      fp, ensure_ascii=False, indent=2)

    def agg(key):
        v = [m[key] for m, _ in runs]
        return float(np.mean(v)), float(np.std(v))          # ddof=0，全文统一

    summary = {}
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "benign_fpr"):
        summary[key] = dict(zip(("mean", "std"), agg(key)))
    lfs_v = [l for _, l in runs]
    summary["lesion_sum"] = {"mean": float(np.mean(lfs_v)),
                             "std": float(np.std(lfs_v)), "values": lfs_v}
    per_class = {}
    for c in class_names:
        per_class[c] = {
            k: {"mean": float(np.mean([m["per_class"][c][k] for m, _ in runs])),
                "std": float(np.std([m["per_class"][c][k] for m, _ in runs]))}
            for k in ("precision", "recall", "f1")}

    out = {
        "model": "RandomForest",
        "view": args.view,
        "split": args.split,
        "seeds": args.seeds,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(X_tr.shape[1]),
        "n_estimators": args.n_estimators,
        "class_weight": None,
        "treatment": {"benign_target": args.benign_target,
                      "lesion_target": args.lesion_target},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "per_class_summary": per_class,
        "per_seed": [{k: v for k, v in m.items() if not isinstance(v, dict)}
                     | {"lesion_sum": l} for m, l in runs],
    }
    suffix = "_treat" if args.benign_target else ""
    path = OUT_DIR / f"rf_{args.view}_p2{suffix}_metrics.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    log(f"写出 {path}")
    log(f"macro-F1 {summary['macro_f1']['mean']:.4f} +- {summary['macro_f1']['std']:.4f} | "
        f"LFS {summary['lesion_sum']['mean']:.4f} +- {summary['lesion_sum']['std']:.4f} | "
        f"fpr {summary['benign_fpr']['mean']:.6f}")


if __name__ == "__main__":
    main()
