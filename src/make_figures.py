# -*- coding: utf-8 -*-
"""
生成论文图表
================================
当前实现：
    图 3  数据集类别分布（对数刻度）—— 论文中用来直观说明类别不平衡有多严重

用法：
    python make_figures.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

# Windows 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fig_label_distribution():
    """图 3：类别分布（对数刻度），直观呈现极端不平衡。"""
    dist = pd.read_csv(RES / "label_distribution.csv", index_col=0)
    s = dist["count"].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colors = ["#c0392b" if name != "BENIGN" else "#2c7fb8" for name in s.index]
    bars = ax.barh(range(len(s)), s.values, color=colors, height=0.7)

    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("样本数量（对数刻度）", fontsize=11)
    ax.set_title("CIC-IDS2017 类别分布（去重后 2,520,798 条）", fontsize=12)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    total = s.sum()
    for i, (name, v) in enumerate(zip(s.index, s.values)):
        ax.text(v * 1.15, i, f"{v:,} ({v / total * 100:.3f}%)",
                va="center", fontsize=7.5)

    ax.set_xlim(right=s.max() * 12)
    plt.tight_layout()
    out = RES / "fig3_label_distribution.png"
    plt.savefig(out)
    plt.close()
    print(f"已生成 {out}")


def fig_rf_performance():
    """对比二分类与多分类的表现，说明「二分类已饱和、多分类才是战场」。"""
    import json
    with open(RES / "rf_baseline_metrics.json", encoding="utf-8") as f:
        m = json.load(f)

    metrics = ["accuracy", "precision", "recall", "f1", "fpr"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
    bin_v = [m["binary"][k] for k in metrics]
    mul_v = [m["multi"][k] for k in metrics]

    x = range(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    ax.bar([i - w / 2 for i in x], bin_v, w, label="二分类（BENIGN vs ATTACK）",
           color="#2c7fb8")
    ax.bar([i + w / 2 for i in x], mul_v, w, label="多分类（全部攻击类型）",
           color="#c0392b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("指标值", fontsize=11)
    ax.set_title("随机森林基线：二分类 vs 多分类", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, (b, mm) in enumerate(zip(bin_v, mul_v)):
        ax.text(i - w / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, mm + 0.015, f"{mm:.3f}", ha="center", fontsize=7.5)

    plt.tight_layout()
    out = RES / "fig_rf_baseline.png"
    plt.savefig(out)
    plt.close()
    print(f"已生成 {out}")


if __name__ == "__main__":
    fig_label_distribution()
    try:
        fig_rf_performance()
    except FileNotFoundError:
        print("跳过 RF 对比图：先跑 baseline_rf.py --multi")
