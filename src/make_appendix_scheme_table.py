# -*- coding: utf-8 -*-
"""
附录：三种类别口径对照表（15 / 12 / 11 类）
=========================================
论文附录素材。正文只用 12 类；附录主动交代为什么不用 11 类（合并 Web Attack 子类
会消除本文要研究的类间混淆），比被审稿人问出来更稳妥。

数据来源（全部为实测值，不重新计算）
    15 类：results/rf_baseline_metrics.json 的 "multi"
    12 类：results/rf_12class_metrics.json 的 "summary"（5 种子 mean ± std）
    11 类：results/archive/11class/rf_11class_metrics.json 的 "summary"

用法
----
    python src/make_appendix_scheme_table.py
产出
----
    results/appendix_scheme_comparison.md
    results/appendix_scheme_comparison.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"
ARCHIVE = OUT_DIR / "archive" / "11class"


def load(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def main():
    rf15 = load(OUT_DIR / "rf_baseline_metrics.json")["multi"]
    rf12 = load(OUT_DIR / "rf_12class_metrics.json")["summary"]
    rf11 = load(ARCHIVE / "rf_11class_metrics.json")["summary"]

    rows = [
        {
            "scheme": "15-class (raw labels)",
            "n_classes": 15,
            "n_rows": 2520798,
            "dropped": "none",
            "seeds": 1,
            "macro_f1": {"mean": rf15["f1"], "std": None},
            "macro_recall": {"mean": rf15["recall"], "std": None},
            "macro_precision": {"mean": rf15["precision"], "std": None},
            "fpr": {"mean": rf15["fpr"], "std": None},
            "accuracy": {"mean": rf15["accuracy"], "std": None},
        },
        {
            "scheme": "12-class (adopted)",
            "n_classes": 12,
            "n_rows": 2520730,
            "dropped": "Web Attack-Sql Injection 21 / Infiltration 36 / Heartbleed 11",
            "seeds": len(rf12["macro_f1"]["values"]),
            "macro_f1": {"mean": rf12["macro_f1"]["mean"], "std": rf12["macro_f1"]["std"]},
            "macro_recall": {"mean": rf12["macro_recall"]["mean"], "std": rf12["macro_recall"]["std"]},
            "macro_precision": {"mean": rf12["macro_precision"]["mean"], "std": rf12["macro_precision"]["std"]},
            "fpr": {"mean": rf12["fpr"]["mean"], "std": rf12["fpr"]["std"]},
            "accuracy": {"mean": rf12["accuracy"]["mean"], "std": rf12["accuracy"]["std"]},
        },
        {
            "scheme": "11-class (rejected)",
            "n_classes": 11,
            "n_rows": 2520751,
            "dropped": "Sql Injection + Infiltration + Heartbleed, and merged "
                       "Brute Force / XSS / Sql Injection into a single Web Attack class",
            "seeds": len(rf11["macro_f1"]["values"]),
            "macro_f1": {"mean": rf11["macro_f1"]["mean"], "std": rf11["macro_f1"]["std"]},
            "macro_recall": {"mean": rf11["macro_recall"]["mean"], "std": rf11["macro_recall"]["std"]},
            "macro_precision": {"mean": rf11["macro_precision"]["mean"], "std": rf11["macro_precision"]["std"]},
            "fpr": {"mean": rf11["fpr"]["mean"], "std": rf11["fpr"]["std"]},
            "accuracy": {"mean": rf11["accuracy"]["mean"], "std": rf11["accuracy"]["std"]},
        },
    ]

    def fmt(d):
        if d["std"] is None:
            return f"{d['mean']:.4f}"
        return f"{d['mean']:.4f} ± {d['std']:.4f}"

    md = []
    md.append("# 附录 A：类别体系对照（RandomForest，同一数据与预处理）\n")
    md.append("正文采用 **12 类**。11 类方案在预实验中被否决，理由见文末。\n")
    md.append("| 口径 | 类数 | 样本数 | 剔除/合并 | 种子数 | macro-F1 | macro-Recall "
              "| macro-Precision | FPR | Accuracy |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['scheme']} | {r['n_classes']} | {r['n_rows']:,} | {r['dropped']} "
                  f"| {r['seeds']} | **{fmt(r['macro_f1'])}** | {fmt(r['macro_recall'])} "
                  f"| {fmt(r['macro_precision'])} | {fmt(r['fpr'])} | {fmt(r['accuracy'])} |")
    md.append("")
    md.append("> 15 类为单次运行（划分方式与 12 类不同），12/11 类为固定划分下的 5 种子均值 ± 标准差；"
              "三者不可逐位横比，只用于说明口径选择的影响。")
    md.append("")
    md.append("## 为什么不采用 11 类（合并 Web Attack 子类）\n")
    md.append("把 `Web Attack-Brute Force`、`Web Attack-XSS`、`Web Attack-Sql Injection` "
              "合并为单一 `Web Attack` 类后，macro-F1 由 0.8550（15 类）/ 0.9091（12 类）"
              "升至 0.9763。但该提升几乎全部来自「子类间混淆被合并消除」，而非检测能力变化："
              "15 类口径下 XSS 的召回仅 0.2959，其中 127/196 条被判为 Brute Force；"
              "合并后这些误判直接变成正确预测。本文的研究动机正是 Web Attack 子类间的"
              "特征高度重叠与严重不平衡，合并会消解研究问题本身，因此不采用该口径。")
    md.append("")
    md.append("12 类口径下 XSS 召回为 0.3051 ± 0.0043（196 条中约 60 条正确），"
              "129 条被判为 Brute Force，与 15 类口径下的混淆结构一致，问题被完整保留。")
    md.append("")
    md.append("英文表述（可直接用）：")
    md.append("")
    md.append("> *We deliberately keep the Web Attack sub-classes separate. Merging them into a "
              "single class inflates macro-F1 from 0.9091 to 0.9763 merely by absorbing the "
              "inter-sub-class confusion that this work aims to resolve, and therefore removes "
              "the very problem under study. Classes with fewer than 100 samples "
              "(Sql Injection, Infiltration, Heartbleed) are excluded by label, while all BENIGN "
              "records from their source capture files are retained.*")
    md.append("")

    out_md = OUT_DIR / "appendix_scheme_comparison.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json = OUT_DIR / "appendix_scheme_comparison.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump({
            "note": "论文附录 A：15/12/11 类口径对照，全部为实测值",
            "rows": rows,
            "adopted": "12-class",
            "rejected": "11-class（合并 Web Attack 子类会消除研究问题）",
        }, fp, ensure_ascii=False, indent=2)

    print("\n".join(md[:12]))
    print(f"\n已写入 {out_md}\n已写入 {out_json}")


if __name__ == "__main__":
    main()
