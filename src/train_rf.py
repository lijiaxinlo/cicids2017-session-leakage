# -*- coding: utf-8 -*-
"""
随机森林 · 12 类口径（论文 Table 2 传统机器学习首行）
=====================================================
读 data_pipeline.py 产出的**固定划分**（results/splits/train_idx.npy / test_idx.npy），
5 个随机种子各跑一次，报告均值 ± 标准差。

划分只算一次、模型种子只影响模型本身 —— 这样方差只来自模型，对比才公平。

用法
----
    python src/train_rf.py                       # 5 种子全量
    python src/train_rf.py --seeds 42            # 单种子（冒烟）
    python src/train_rf.py --no-binary-check     # 跳过二分类自检
    python src/train_rf.py --class-weight none   # E13：无不平衡处理对照（产出 rf_12class_noweight_metrics.json）

产出
----
    results/rf_12class_metrics.json           含 per_seed 明细 + mean/std + 每类召回
    results/rf_12class_confusion_matrix.csv   与均值最接近的那一次的混淆矩阵
    results/rf_12class_binary_check.json      二分类自检（附属，不进主表）
    results/runs/rf12_seed<seed>.json         每次运行的完整记录
"""

import argparse
import gc
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import data_pipeline as dp
import eval_utils as eu

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"
RUN_DIR = OUT_DIR / "runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_info():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_one(X_tr, X_te, y_tr, y_te, seed, n_estimators, class_names, tag="rf",
            class_weight="balanced_subsample"):
    t0 = time.time()
    # sklearn 的 class_weight 需要真正的 None（不是字符串 "none"）
    cw = None if class_weight == "none" else class_weight
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
        class_weight=cw,
    )
    clf.fit(X_tr, y_tr)
    fit_sec = time.time() - t0
    y_pred = clf.predict(X_te)
    total_sec = time.time() - t0

    m = eu.compute_metrics(y_te, y_pred, class_names=class_names,
                           labels=list(range(len(class_names))))
    m["elapsed_fit_sec"] = round(fit_sec, 1)
    m["elapsed_total_sec"] = round(total_sec, 1)
    m["seed"] = seed
    return m, y_pred, clf


def main():
    ap = argparse.ArgumentParser(description="RF 12 类基线（固定划分 + 5 种子）")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--no-binary-check", action="store_true")
    ap.add_argument("--class-weight", type=str, default="balanced_subsample",
                    choices=["balanced_subsample", "none"],
                    help="类别权重方案；none 用于 E13 无不平衡处理对照（公平对称）")
    ap.add_argument("--split", default="random",
                    choices=["random", "temporal", "grouped"],
                    help="random = 原固定划分；temporal = 类内时序 purged 划分（返修意见 1）")
    ap.add_argument("--tag", type=str, default=None,
                    help="输出文件名后缀，区分返修新臂")
    ap.add_argument("--benign-target", type=int, default=None,
                    help="（可选）训练集 BENIGN 欠采样到该行数，与深度臂的处理对齐")
    ap.add_argument("--lesion-target", type=int, default=None,
                    help="（可选）Bot/BF/XSS 各自随机过采样到该行数（RF 不做 SMOTE）")
    args = ap.parse_args()

    cw_suffix = "_noweight" if args.class_weight == "none" else ""
    if args.split != "random":
        cw_suffix += f"_{args.split}"
    if args.benign_target or args.lesion_target:
        cw_suffix += "_treat"
    if args.tag:
        cw_suffix += f"_{args.tag}"
    cmd = " ".join(["python", "src/train_rf.py"] +
                   ([] if args.seeds == SEEDS else ["--seeds"] +
                    [str(s) for s in args.seeds]) +
                   ([] if args.class_weight == "balanced_subsample"
                    else ["--class-weight", args.class_weight]) +
                   ([] if args.split == "random" else ["--split", args.split]))
    t_start = time.time()

    class_names = dp.load_class_names()
    y = dp.load_y()
    if args.split in ("temporal", "grouped"):
        tsd = ROOT / "results" / ("temporal" if args.split == "temporal" else "temporal_grouped")
        tr = np.load(tsd / "train_idx.npy")
        te = np.load(tsd / "test_idx.npy")
    else:
        tr, te = dp.get_split(y)
    X = dp.load_X(mmap=True)
    log(f"数据: X {X.shape}  y {y.shape}；训练 {len(tr):,} / 测试 {len(te):,}")

    # 可选：与深度臂对齐的不平衡处理（BENIGN 欠采样 + 病灶类随机过采样）
    if args.benign_target or args.lesion_target:
        rng = np.random.default_rng(SEEDS[0])
        benign_id = class_names.index("BENIGN")
        lesion_ids = [class_names.index(c) for c in
                      ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")]
        keep = []
        for c in range(len(class_names)):
            idx_c = tr[y[tr] == c]
            if c == benign_id and args.benign_target and len(idx_c) > args.benign_target:
                idx_c = np.sort(rng.choice(idx_c, size=args.benign_target, replace=False))
            elif c in lesion_ids and args.lesion_target and len(idx_c) < args.lesion_target:
                extra = rng.choice(idx_c, size=args.lesion_target - len(idx_c), replace=True)
                idx_c = np.concatenate([idx_c, extra])
            keep.append(idx_c)
        tr = np.sort(np.concatenate(keep))
        log(f"处理后训练集 {len(tr):,} 行（BENIGN→{args.benign_target:,}，"
            f"病灶类→{args.lesion_target:,} 随机过采样）")

    # 固定划分已生成，这里只做一致性校验，绝不重新 split
    assert len(np.intersect1d(tr, te)) == 0

    X_tr = np.ascontiguousarray(X[tr])
    X_te = np.ascontiguousarray(X[te])
    y_tr, y_te = y[tr].astype(np.int64), y[te].astype(np.int64)
    log(f"已载入内存: X_tr {X_tr.shape} {X_tr.nbytes / 1e6:.0f} MB, "
        f"X_te {X_te.shape} {X_te.nbytes / 1e6:.0f} MB")

    runs = []
    preds = {}
    for seed in args.seeds:
        log(f"--- seed={seed} 训练中（{args.n_estimators} 棵树）---")
        m, y_pred, _ = run_one(X_tr, X_te, y_tr, y_te, seed,
                               args.n_estimators, class_names,
                               class_weight=args.class_weight)
        eu.print_report(f"RandomForest 12 类 · seed={seed}", m, class_names)
        runs.append(m)
        preds[seed] = y_pred

        rec = {
            "model": "RandomForest",
            "class_scheme": "12class",
            "command": cmd,
            "seed": seed,
            "n_estimators": args.n_estimators,
            "class_weight": args.class_weight,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_features": int(X.shape[1]),
            "split": (f"results/temporal/{{train,val,test}}_idx.npy (类内时序 purged, purge=200)"
                      if args.split == "temporal" else
                      f"results/temporal_grouped/{{train,val,test}}_idx.npy (组内随机, 同组同尺寸)"
                      if args.split == "grouped" else
                      "results/splits/{train,test}_idx.npy (test_size=0.3, seed=42, 只算一次)"),
            "split_mode": args.split,
            "treatment": {"benign_target": args.benign_target,
                          "lesion_target": args.lesion_target},
            "elapsed_sec": m["elapsed_total_sec"],
            "env": env_info(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "metrics": {k: m[k] for k in eu.SCALAR_KEYS},
            "per_class": m["per_class"],
        }
        with open(RUN_DIR / f"rf12{cw_suffix}_seed{seed}.json", "w", encoding="utf-8") as fp:
            json.dump(rec, fp, ensure_ascii=False, indent=2)
        gc.collect()

    # ---------- 聚合 ----------
    agg = eu.aggregate(runs)
    agg_cls = eu.aggregate_per_class(runs, class_names)

    mean_f1 = agg["macro_f1"]["mean"]
    closest = min(runs, key=lambda r: abs(r["macro_f1"] - mean_f1))
    log(f"macro-F1 均值 {mean_f1:.4f}；最接近均值的种子 = {closest['seed']} "
        f"({closest['macro_f1']:.4f})")

    eu.confusion_df(y_te, preds[closest["seed"]], class_names).to_csv(
        OUT_DIR / f"rf_12class{cw_suffix}_confusion_matrix.csv")

    result = {
        "model": "RandomForest",
        "class_scheme": "12class",
        "class_names": class_names,
        "command": cmd,
        "seeds": [r["seed"] for r in runs],
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "n_estimators": args.n_estimators,
        "class_weight": args.class_weight,
        "split_protocol": "固定划分 results/splits/，12 类分层，test_size=0.3，seed=42，只计算一次",
        "env": env_info(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_elapsed_sec": round(time.time() - t_start, 1),
        "summary": agg,
        "per_class_summary": agg_cls,
        "per_seed": [{"seed": r["seed"], "elapsed_sec": r["elapsed_total_sec"],
                      **{k: r[k] for k in eu.SCALAR_KEYS},
                      "per_class_recall": {c: r["per_class"][c]["recall"]
                                           for c in class_names}}
                     for r in runs],
        "confusion_matrix_from_seed": closest["seed"],
        "confusion_matrix_file": "results/rf_12class_confusion_matrix.csv",
    }
    with open(OUT_DIR / f"rf_12class{cw_suffix}_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)

    # ---------- 打印汇总表 ----------
    print(f"\n{'=' * 78}")
    print("  Table 2 首行 · RandomForest (12 classes, 5 seeds)")
    print(f"{'=' * 78}")
    print(f"  {'metric':<20}{'mean':>10}{'std':>10}   values")
    for k in eu.SCALAR_KEYS:
        vals = ", ".join(f"{v:.4f}" for v in agg[k]["values"])
        print(f"  {k:<20}{agg[k]['mean']:>10.4f}{agg[k]['std']:>10.4f}   [{vals}]")
    print(f"\n  {'class':<20}{'recall mean':>13}{'std':>9}{'support':>10}")
    for c in class_names:
        d = agg_cls[c]["recall"]
        print(f"  {c:<20}{d['mean']:>13.4f}{d['std']:>9.4f}{agg_cls[c]['support']:>10,}")
    print(f"{'=' * 78}")

    # ---------- 二分类自检（附属） ----------
    if not args.no_binary_check:
        log("二分类自检（BENIGN vs ATTACK，仅用于验证管线，不进主表）")
        yb_tr = (y_tr != 0).astype(np.int64)
        yb_te = (y_te != 0).astype(np.int64)
        seed = args.seeds[0]
        m_bin, _, _ = run_one(X_tr, X_te, yb_tr, yb_te, seed,
                              args.n_estimators, ["BENIGN", "ATTACK"], tag="rf_bin",
                              class_weight=args.class_weight)
        eu.print_report(f"RandomForest 二分类 · seed={seed}", m_bin,
                        ["BENIGN", "ATTACK"])
        with open(OUT_DIR / f"rf_12class{cw_suffix}_binary_check.json", "w", encoding="utf-8") as fp:
            json.dump({
                "note": "二分类仅作管线自检，属附属结果，不进 Table 2",
                "command": cmd, "seed": seed,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "n_estimators": args.n_estimators,
                "metrics": {k: m_bin[k] for k in eu.SCALAR_KEYS},
                "per_class": m_bin["per_class"],
                "env": env_info(),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }, fp, ensure_ascii=False, indent=2)

    log(f"全部完成，总耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
