"""Figure: the leak, measured without training (P0 vs P2), single panel.

Output: results/figures/fig_leakage.{pdf,png}  (IEEE single column, 3.45 in wide)
Data:   results/temporal/leakage_stats.json

单面板：横轴是对数刻度的"到最近同类训练流的行距中位数"，每个类两个点（P0 蓝、P2 橙），
连线表示 purge 把它推了多远。这一个面板同时说明两件事 —— P0 下每个类都是"几行之遥"，
P2 下每个类都在 10^3 行以外 —— 所以不再需要并排的第二个面板。

用 --two-panel 可以回到旧的两面板版本（(a) 占比条形 + (b) 中位行距）。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"

SHORT = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS Hulk",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "DoS GoldenEye": "GoldenEye",
    "FTP-Patator": "FTP-Patator",
    "DoS slowloris": "slowloris",
    "DoS Slowhttptest": "Slowhttptest",
    "SSH-Patator": "SSH-Patator",
    "Bot": "Bot",
    "Web Attack-Brute Force": "Web-BF",
    "Web Attack-XSS": "Web-XSS",
}
C_P0, C_P2 = "#2f5d8c", "#d1743a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--two-panel", action="store_true")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    stats = json.loads((RES / "temporal" / "leakage_stats.json").read_text(encoding="utf-8"))
    classes = list(stats["random"].keys())
    order = sorted(classes, key=lambda c: stats["random"][c]["median_gap_rows"])
    w10_p0 = [stats["random"][c]["pct_within_10_rows"] for c in order]
    med_p0 = [stats["random"][c]["median_gap_rows"] for c in order]
    med_p2 = [stats["temporal"][c]["median_gap_rows"] for c in order]
    y = np.arange(len(order))

    if args.two_panel:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(3.45, 3.30), sharey=True,
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.55})
        ax1.barh(y, w10_p0, height=0.68, color=C_P0, edgecolor="black", linewidth=0.35)
        ax1.set_xlim(0, 108)
        ax1.set_xlabel("share of test flows within 10 rows of a same-class "
                       "training flow  (%)", fontsize=6.0)
        for yi, v in zip(y, w10_p0):
            ax1.text(v + 1.5, yi, f"{v:.1f}", va="center", ha="left", fontsize=5.4)
        ax1.set_title("(a) P0: the standard 70/30 flow-level split", fontsize=6.4, pad=2)
        ax = ax2
        ax.set_title("(b) P0 against P2: the purge moves every class",
                     fontsize=6.4, pad=2)
    else:
        fig, ax = plt.subplots(figsize=(3.45, 1.72))

    ax.plot(med_p0, y, "o", ms=3.6, mfc=C_P0, mec="black", mew=0.35, zorder=3, label="P0")
    ax.plot(med_p2, y, "s", ms=3.6, mfc=C_P2, mec="black", mew=0.35, zorder=3, label="P2")
    for yi, a, b in zip(y, med_p0, med_p2):
        ax.plot([a, b], [yi, yi], color="0.6", lw=0.55, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.65, 1.6e6)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["1", "10", "$10^2$", "$10^3$", "$10^4$", "$10^5$"])
    ax.set_xlabel("median rows to the nearest same-class training flow", fontsize=6.2)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[c] for c in order], fontsize=6.2)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.legend(loc="lower right", fontsize=6.2, frameon=False, handlelength=0.9,
              handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    ax.grid(axis="x", lw=0.3, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_leakage.{ext}", dpi=600 if ext == "png" else None)
    print("wrote", OUT / "fig_leakage.pdf")


if __name__ == "__main__":
    main()
