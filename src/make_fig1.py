"""Figure 1: 整体框架图（v3：按每框行数反算框高，注记移到框外空白带）。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures" / "fig1_framework.png"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.bbox": "tight"})

W, H = 100.0, 64.0
fig, ax = plt.subplots(figsize=(6.9, 4.4))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

C_MAIN, C_TREAT, C_OPT, C_GREY = "#4C6EF5", "#E8590C", "#2F9E44", "#495057"
F_MAIN, F_TREAT, F_OPT, F_GREY = "#EDF2FF", "#FFF4E6", "#EBFBEE", "#F1F3F5"

LINE_H = 1.95
BS = 6.0


def box(x, y, w, h, title, lines, fill, edge, dashed=False, tsize=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.35,rounding_size=0.8",
                                linewidth=1.3, edgecolor=edge, facecolor=fill,
                                linestyle=(0, (3.5, 2)) if dashed else "-", zorder=2))
    ax.text(x + w / 2, y + h - 1.2, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=edge, zorder=3, linespacing=1.2)
    body_top = y + h - 1.2 - (2.4 if "\n" in title else 1.4) - 3.0
    ax.text(x + w / 2, body_top, "\n".join(lines), ha="center", va="top",
            fontsize=BS, color="#212529", linespacing=1.55, zorder=3)
    need = 1.2 + (2.4 if "\n" in title else 1.4) + 3.0 + len(lines) * LINE_H + 0.8
    assert need <= h + 0.4, f"框高不足：{title!r} 需 {need:.1f}，实有 {h}"


def arrow(x1, y1, x2, y2, color=C_GREY, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=1))


# ---------- 上排五阶段 ----------
Y0, BH, BW, GAP = 33.0, 24.0, 17.0, 2.4
xs = [1.8 + i * (BW + GAP) for i in range(5)]

box(xs[0], Y0, BW, BH, "Input",
    ["78 CICFlowMeter", "features per flow", "", "one flow =", "one sample"], F_GREY, C_GREY)
box(xs[1], Y0, BW, BH, "Preprocessing",
    ["cleaning,", "de-duplication,", "standardisation", "", "scaler fit on", "fitting rows"], F_GREY, C_GREY)
box(xs[2], Y0, BW, BH, "Fixed split",
    ["drawn once,", "seed 42", "", "fit 1,588,059", "val 176,452", "test 756,219"], F_MAIN, C_MAIN)
box(xs[3], Y0, BW, BH, "Training-time\ntreatment",
    ["SMOTE, k = 5", "lesion classes only", "", "410–1,228 → 12,000", "", "BENIGN kept in full"],
    F_TREAT, C_TREAT)
box(xs[4], Y0, BW, BH, "Backbone",
    ["142,029 params", "", "Conv1D ×2", "→ BiLSTM", "→ attention", "→ FC → softmax"], F_MAIN, C_MAIN)

for i in range(4):
    arrow(xs[i] + BW + 0.3, Y0 + BH / 2, xs[i + 1] - 0.3, Y0 + BH / 2)

ax.text(xs[3] + BW / 2, Y0 - 1.0, "fitting set only", ha="center", va="top",
        fontsize=6.2, style="italic", color=C_TREAT)

# ---------- 输出分支 ----------
OY, OH, OW = 11.5, 12.5, 17.0
box(xs[3] + 1.2, OY, OW, OH, "C3 (reported)", ["argmax over", "12 logits"], F_MAIN, C_MAIN, tsize=7.0)
box(xs[4] + 0.3, OY, OW + 0.4, OH, "C4 (reported)",
    ["per-class weights", "benign FPR ≤ 0.008"], F_OPT, C_OPT, dashed=True, tsize=7.0)

cx = xs[4] + BW / 2
arrow(xs[4] + BW / 2 - 3.0, Y0 - 0.5, xs[3] + 1.2 + OW / 2 + 3.5, OY + OH + 0.4)
arrow(cx + 2.5, Y0 - 0.5, cx + 2.5, OY + OH + 0.4)
ax.text(cx + 3.2, (Y0 + OY + OH) / 2 - 0.5, "optional\nstage", ha="left", va="center",
        fontsize=6.0, style="italic", color=C_OPT, linespacing=1.4)

# ---------- held-out 注记 ----------
ax.plot([xs[2] + BW / 2, xs[2] + BW / 2, cx, cx],
        [Y0 + BH + 0.6, 60.6, 60.6, Y0 + BH + 0.6],
        ls=(0, (3, 2)), lw=1.0, color=C_GREY, zorder=1)
ax.text((xs[2] + cx) / 2, 61.2,
        "validation 176,452 — early-stopping monitor and calibration search set",
        ha="center", va="bottom", fontsize=6.4, style="italic", color=C_GREY)

ax.text(xs[3] + 1.2, OY - 2.1,
        "test 756,219 — forward pass only, weights frozen",
        ha="left", va="top", fontsize=6.4, style="italic", color=C_GREY)

# ---------- 底部泄漏控制 ----------
ax.text(W / 2, 1.6,
        "Resampling is applied to the fitting set only, after the validation split is drawn. "
        "Validation and test are never resampled\nand never used to fit any parameter; the test set "
        "is evaluated exactly once, after w is frozen.",
        ha="center", va="bottom", fontsize=6.3, color="#343A40", linespacing=1.5)

fig.savefig(OUT, dpi=300)
print("written:", OUT)
