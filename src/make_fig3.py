"""Figure 3: 12-class label distribution（对数横轴，按组着色，标注病灶组与不平衡比）。

数据源：results/label_distribution_12class.csv（清洗后全量 2,520,730 行）
输出：results/figures/fig3_label_distribution.png（300 dpi，投稿用）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

LESION = {"Bot", "Web Attack-Brute Force", "Web Attack-XSS"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
})

df = pd.read_csv(RES / "label_distribution_12class.csv")
df = df.sort_values("count", ascending=True).reset_index(drop=True)
total = df["count"].sum()

colors = []
for c in df["class"]:
    if c == "BENIGN":
        colors.append("#4C6EF5")          # benign: blue
    elif c in LESION:
        colors.append("#E8590C")          # lesion group: orange
    else:
        colors.append("#ADB5BD")          # other attacks: grey

fig, ax = plt.subplots(figsize=(6.6, 3.6))
y = np.arange(len(df))
bars = ax.barh(y, df["count"], color=colors, height=0.72, edgecolor="white", linewidth=0.5)

ax.set_yticks(y)
ax.set_yticklabels(df["class"], fontsize=8.5)
ax.set_xscale("log")
ax.set_xlim(300, 5e6)
ax.set_xlabel("Number of flows (log scale)")
ax.set_title("Class distribution after cleaning (2,520,730 flows, 12 classes)", fontsize=9.5, pad=8)

# 数值标注
for yi, (cnt, pct) in enumerate(zip(df["count"], df["ratio_pct"])):
    lab = f"{cnt:,}  ({pct:.3f}%)" if pct < 1 else f"{cnt:,}  ({pct:.2f}%)"
    ax.text(cnt * 1.15, yi, lab, va="center", ha="left", fontsize=7.2, color="#333333")

# 病灶组不用括注（避免与数值标签冲突），由图例颜色承担

# 不平衡比
b = df.loc[df["class"] == "BENIGN", "count"].iloc[0]
w = df["count"].min()
ax.text(0.985, 0.045, f"BENIGN : Web Attack-XSS = {b/w:,.0f} : 1",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF4E6", ec="#E8590C", lw=0.7))

ax.grid(axis="x", ls=":", lw=0.6, color="#CED4DA", alpha=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.legend(handles=[Patch(facecolor="#4C6EF5", label="BENIGN"),
                   Patch(facecolor="#ADB5BD", label="Other attack classes"),
                   Patch(facecolor="#E8590C", label="Lesion group: Bot, BF, XSS (0.16% of flows)")],
          loc="lower right", bbox_to_anchor=(1.0, 0.14), fontsize=7.4, frameon=False)

out = FIG / "fig3_label_distribution.png"
fig.savefig(out, dpi=300)
print("written:", out)
print(df[["class", "count", "ratio_pct"]].to_string(index=False))
