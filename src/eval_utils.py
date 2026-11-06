# -*- coding: utf-8 -*-
"""
统一评价口径（RF 与所有深度模型共用）
=====================================
论文主指标 macro-F1，辅以 macro-Recall / macro-Precision / FPR / 各类别召回率。
Accuracy 仅作参考（不平衡场景下严重失真）。

FPR 定义：逐类计算 FP / (FP + TN) 后取宏平均（即 macro-averaged one-vs-rest FPR）。
二分类时该值退化为整体误报率。
"""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(y_true, y_pred, class_names=None, labels=None):
    """返回论文需要的全部指标（标量均为 float）。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n_cls = len(np.unique(np.concatenate([y_true, y_pred])))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = cm.shape[0]
    fprs, recs, precs, f1s, supports = [], [], [], [], []
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recs.append(r)
        precs.append(p)
        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        supports.append(int(cm[i, :].sum()))

    out = {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "fpr": float(np.mean(fprs)),
        "per_class": {},
        "n_classes": int(n_cls),
    }
    for i in range(n):
        name = class_names[i] if class_names is not None else f"class_{i}"
        out["per_class"][name] = {
            "precision": float(precs[i]),
            "recall": float(recs[i]),
            "f1": float(f1s[i]),
            "fpr": float(fprs[i]),
            "support": supports[i],
        }
    return out


def confusion_df(y_true, y_pred, class_names):
    import pandas as pd
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return pd.DataFrame(cm,
                        index=[f"true_{c}" for c in class_names],
                        columns=[f"pred_{c}" for c in class_names])


SCALAR_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "fpr"]


def aggregate(runs, keys=SCALAR_KEYS):
    """runs: list[dict] -> {key: {mean, std, values}}"""
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=np.float64)
        agg[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }
    return agg


def aggregate_per_class(runs, class_names):
    """每类 recall / f1 的 mean ± std（跨种子）。"""
    agg = {}
    for c in class_names:
        agg[c] = {}
        for k in ("precision", "recall", "f1"):
            vals = np.array([r["per_class"][c][k] for r in runs], dtype=np.float64)
            agg[c][k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        agg[c]["support"] = int(runs[0]["per_class"][c]["support"])
    return agg


def print_report(title, m, class_names=None):
    print(f"\n{'-' * 62}")
    print(f"  {title}")
    print(f"{'-' * 62}")
    print(f"  Accuracy         : {m['accuracy']:.4f}")
    print(f"  macro-Precision  : {m['macro_precision']:.4f}")
    print(f"  macro-Recall     : {m['macro_recall']:.4f}")
    print(f"  macro-F1         : {m['macro_f1']:.4f}   <== 主指标")
    print(f"  FPR (macro OVR)  : {m['fpr']:.6f}")
    if class_names:
        print(f"  {'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
        for c in class_names:
            d = m["per_class"][c]
            print(f"  {c:<20}{d['precision']:>10.4f}{d['recall']:>10.4f}"
                  f"{d['f1']:>10.4f}{d['support']:>10,}")
    print(f"{'-' * 62}")
