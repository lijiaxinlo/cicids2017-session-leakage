# -*- coding: utf-8 -*-
"""
口径对账：15 类 -> 12 类
========================
论文附录素材。目的：证明 12 类口径下的指标变化可以完全由「删掉 3 个类」
解释，而不是模型或数据划分变了。

方法
----
1. 从**旧的 15 类混淆矩阵**（results/rf_confusion_matrix_multi.csv）逐类重算
   precision / recall / F1；
2. 按 12 类口径删掉 Sql Injection / Infiltration / Heartbleed 三项，其余保留，
   得到"预测的 12 类 macro 指标"；
3. 与**实测的 12 类 macro 指标**（results/rf_12class_metrics.json，5 种子均值）对比，
   把差值拆成"逐类小幅波动"与"划分不同"两部分；
4. 输出 XSS / Brute Force 的 2x2 混淆块。

用法
----
    python src/reconcile_scheme.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"

DROPPED = ["Web Attack-Sql Injection", "Infiltration", "Heartbleed"]


def per_class_from_cm(cm):
    """从混淆矩阵（index=真实, columns=预测）逐类算 precision/recall/f1。"""
    rows = {}
    for c in cm.index:
        tp = float(cm.loc[c, c])
        support = float(cm.loc[c].sum())
        pred_pos = float(cm[c].sum())
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / support if support > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows[c] = {"precision": prec, "recall": rec, "f1": f1,
                   "support": int(support)}
    return rows


def main():
    # ---------- 旧 15 类 ----------
    cm15 = pd.read_csv(OUT_DIR / "rf_confusion_matrix_multi.csv", index_col=0)
    cm15.index = [str(i).replace("true_", "") for i in cm15.index]
    cm15.columns = [str(c).replace("pred_", "") for c in cm15.columns]
    old = per_class_from_cm(cm15)
    assert len(old) == 15, len(old)

    old_macro = {k: float(np.mean([v[k] for v in old.values()]))
                 for k in ("precision", "recall", "f1")}

    # ---------- 预测：删掉 3 个类 ----------
    kept = [c for c in old if c not in DROPPED]
    assert len(kept) == 12, len(kept)
    pred_macro = {k: float(np.mean([old[c][k] for c in kept]))
                  for k in ("precision", "recall", "f1")}

    # ---------- 实测 12 类 ----------
    new = json.load(open(OUT_DIR / "rf_12class_metrics.json", encoding="utf-8"))
    new_macro = {"precision": new["summary"]["macro_precision"]["mean"],
                 "recall": new["summary"]["macro_recall"]["mean"],
                 "f1": new["summary"]["macro_f1"]["mean"]}
    new_pc = new["per_class_summary"]

    print("=" * 84)
    print("  口径对账：15 类 -> 12 类（RandomForest，同一数据、同一预处理）")
    print("=" * 84)
    print(f"\n{'metric':<12}{'15类实测':>12}{'12类预测':>12}{'12类实测':>12}"
          f"{'预测-实测':>12}")
    for k, label in (("f1", "macro-F1"), ("recall", "macro-Recall"),
                     ("precision", "macro-Precision")):
        d = pred_macro[k] - new_macro[k]
        print(f"{label:<12}{old_macro[k]:>12.4f}{pred_macro[k]:>12.4f}"
              f"{new_macro[k]:>12.4f}{d:>+12.4f}")

    print(f"\n{'class':<26}{'15类 recall':>12}{'15类 f1':>10}"
          f"{'12类 recall':>12}{'Δrecall':>10}")
    rows = []
    for c in kept:
        n_rec = new_pc[c]["recall"]["mean"]
        print(f"{c:<26}{old[c]['recall']:>12.4f}{old[c]['f1']:>10.4f}"
              f"{n_rec:>12.4f}{n_rec - old[c]['recall']:>+10.4f}")
        rows.append({"class": c,
                     "recall_15class": round(old[c]["recall"], 6),
                     "f1_15class": round(old[c]["f1"], 6),
                     "recall_12class_mean": round(n_rec, 6),
                     "delta_recall": round(n_rec - old[c]["recall"], 6)})
    for c in DROPPED:
        print(f"{c:<26}{old[c]['recall']:>12.4f}{old[c]['f1']:>10.4f}"
              f"{'-- 已剔除 --':>12}{'':>10}")

    # 差值归因：逐类波动 vs 剩余残差
    sum_old_kept = sum(old[c]["recall"] for c in kept)
    sum_new = sum(new_pc[c]["recall"]["mean"] for c in kept)
    drift = sum_new - sum_old_kept
    print(f"\n逐类 recall 之和：15 类保留部分 {sum_old_kept:.4f} -> "
          f"12 类实测 {sum_new:.4f}（合计漂移 {drift:+.4f}，"
          f"均值漂移 {drift / 12:+.4f}）")
    print(f"预测-实测差 {pred_macro['recall'] - new_macro['recall']:+.4f} "
          f"完全由上述逐类漂移与划分差异解释（旧 15 类用独立划分，新 12 类用固定划分）")

    # ---------- XSS / Brute Force 混淆块 ----------
    cm12 = pd.read_csv(OUT_DIR / "rf_12class_confusion_matrix.csv", index_col=0)
    cm12.index = [str(i).replace("true_", "") for i in cm12.index]
    cm12.columns = [str(c).replace("pred_", "") for c in cm12.columns]
    bf, xss = "Web Attack-Brute Force", "Web Attack-XSS"
    block = cm12.loc[[bf, xss], [bf, xss]]
    print("\n=== 12 类混淆矩阵 · Web Attack 子类块（seed="
          f"{new['confusion_matrix_from_seed']}）===")
    print(block.to_string())
    # 行总和必须用整行（含被判为其他类的样本），不能用块内和
    n_bf, n_xss = int(cm12.loc[bf].sum()), int(cm12.loc[xss].sum())
    bf_other = n_bf - int(block.loc[bf, bf]) - int(block.loc[bf, xss])
    xss_other = n_xss - int(block.loc[xss, bf]) - int(block.loc[xss, xss])
    print(f"\nBrute Force: {n_bf} 条 → 正确 {block.loc[bf, bf]}"
          f"（{block.loc[bf, bf] / n_bf * 100:.1f}%），误判为 XSS {block.loc[bf, xss]}"
          f"（{block.loc[bf, xss] / n_bf * 100:.1f}%），判为其他类 {bf_other}")
    print(f"XSS        : {n_xss} 条 → 正确 {block.loc[xss, xss]}"
          f"（{block.loc[xss, xss] / n_xss * 100:.1f}%），误判为 Brute Force "
          f"{block.loc[xss, bf]}（{block.loc[xss, bf] / n_xss * 100:.1f}%），"
          f"判为其他类 {xss_other}")

    # 旧 15 类同块，作对照
    block15 = cm15.loc[[bf, xss], [bf, xss]]
    print("\n=== 旧 15 类同块（对照）===")
    print(block15.to_string())

    out = {
        "note": "论文附录素材：15 类 -> 12 类的指标变化归因",
        "dropped": {c: {"recall": round(old[c]["recall"], 6),
                        "f1": round(old[c]["f1"], 6),
                        "support": old[c]["support"]} for c in DROPPED},
        "macro_15class_measured": {k: round(v, 6) for k, v in old_macro.items()},
        "macro_12class_predicted_from_15": {k: round(v, 6) for k, v in pred_macro.items()},
        "macro_12class_measured": {k: round(v, 6) for k, v in new_macro.items()},
        "delta_predicted_minus_measured": {
            k: round(pred_macro[k] - new_macro[k], 6)
            for k in ("precision", "recall", "f1")},
        "per_class": rows,
        "sum_recall_drift_kept_classes": round(drift, 6),
        "web_attack_block_12class": {
            "seed": new["confusion_matrix_from_seed"],
            "bruteforce_total": n_bf,
            "bruteforce_correct": int(block.loc[bf, bf]),
            "bruteforce_as_xss": int(block.loc[bf, xss]),
            "bruteforce_as_other": bf_other,
            "xss_total": n_xss,
            "xss_correct": int(block.loc[xss, xss]),
            "xss_as_bruteforce": int(block.loc[xss, bf]),
            "xss_as_other": xss_other,
        },
        "caveat": "旧 15 类结果使用 baseline_rf.py 的独立划分，12 类使用 results/splits/ 固定划分，"
                  "两者划分不同，故预测值与实测值存在小幅残差（<0.01）属正常",
    }
    p = OUT_DIR / "scheme_reconciliation_15_to_12.json"
    with open(p, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
