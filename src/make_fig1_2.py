# -*- coding: utf-8 -*-
"""生成稿件 Figure 1（整体框架）与 Figure 2（逐层结构）
================================================================
用法：python src/make_fig1_2.py

产出：
  results/figures/fig1_framework.png   170 mm 宽（双栏）
  results/figures/fig2_backbone.png     85 mm 宽（单栏，高度自适应）
  300 dpi，英文标签，DejaVu Sans，全部字号 >= 7.0 pt

排版策略（v2，修掉首版的标题压字 / 文字出框 / 右侧裁切）：
  * 所有文字宽度用 matplotlib.textpath.TextPath 实测（单位 pt -> mm），
    再按盒内可用宽度贪心折行 —— 不靠肉眼估字符数，杜绝溢出。
  * 每个盒子内部用 stack 方式自上而下排字，行高按字号算，
    标题与正文各自占位，天然不重叠。
  * Figure 2 总高度由行数 / 行高 / 间隙反算后设定 figsize，杜绝底部裁切。

图形内容严格对齐 paper/manuscript_v0.3.md：
  §3.3 Fixed split and leakage control
  §3.4 Backbone architecture
  §3.5 Training-time imbalance treatment
  §3.6 Inference-time calibration family
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

PT2MM = 25.4 / 72.0

# ---------------------------------------------------------------- 配色（4 个色系以内）
C_MAIN = dict(fc="#E9EEF4", ec="#3C5A78")      # 主路径：灰蓝
C_TRAIN = dict(fc="#FCEBD5", ec="#C1711A")     # 训练期处理：橙
C_CAL = dict(fc="#E3F1E6", ec="#3D8A5B")       # 推理期校准：浅绿
C_NOTE = dict(fc="#F4F4F4", ec="#9A9A9A")      # 注记：中性灰
C_SUB = dict(fc="#FFFFFF", ec="#6C8AA6")       # 子块：白底细蓝框
C_HOLD = "#6E6E6E"                             # 留出集连线
INK = "#1A1A1A"
INK_SOFT = "#3A3A3A"
BLUE = "#3C5A78"
ORANGE = "#C1711A"
ORANGE_DK = "#8A4B08"
GREEN = "#2A6641"

FS = 7.0          # 正文字号（下限 7 pt）
FS_T = 7.5        # 小标题字号
LW = 0.9
LH = 1.28         # 行高倍数


# ---------------------------------------------------------------- 文字度量与折行
def tw(s, fs, bold=False, italic=False):
    """字符串宽度（mm），用字体度量实测。"""
    if not s.strip():
        return 0.0
    fp = FontProperties(family="DejaVu Sans", size=fs,
                        weight="bold" if bold else "normal",
                        style="italic" if italic else "normal")
    return TextPath((0, 0), s, prop=fp).get_extents().width * PT2MM


def wrap(s, fs, max_mm, bold=False, italic=False):
    """按可用宽度贪心折行；尊重原字符串里的换行符。"""
    out = []
    for part in s.split("\n"):
        cur = ""
        for w in part.split(" "):
            t = w if not cur else cur + " " + w
            if not cur or tw(t, fs, bold, italic) <= max_mm:
                cur = t
            else:
                out.append(cur)
                cur = w
        out.append(cur)
    return out


def nlines(s, fs, max_mm, bold=False, italic=False):
    return len(wrap(s, fs, max_mm, bold, italic))


def lh_mm(fs):
    return fs * LH * PT2MM


# ---------------------------------------------------------------- 绘图原子
def rbox(ax, x0, y0, x1, y1, fc, ec, ls="-", lw=LW, r=1.6, z=2):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, fs=FS, bold=False, italic=False, color=INK,
        ha="center", va="center", z=6):
    ax.text(x, y, s, fontsize=fs,
            fontweight="bold" if bold else "normal",
            fontstyle="italic" if italic else "normal",
            color=color, ha=ha, va=va, zorder=z)


def arrow(ax, p0, p1, color=BLUE, ls="-", lw=LW, rad=0.0, z=3, ms=6.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                        linewidth=lw, color=color, linestyle=ls, zorder=z,
                        shrinkA=0.8, shrinkB=0.8,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def polyline(ax, pts, color=C_HOLD, ls=(0, (1.8, 1.4)), lw=0.85, z=2.5):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
            linestyle=ls, linewidth=lw, zorder=z,
            solid_capstyle="butt", dash_capstyle="butt")


def stack(ax, x0, x1, y_top, items, pad=2.4, z=2, align="center"):
    """在已画好的盒内自上而下堆叠文字，返回最后一行底部的 y。

    items: list of (text, fs, bold, italic, color, gap_before)
    """
    maxw = (x1 - x0) - 2 * pad
    xc = (x0 + x1) / 2 if align == "center" else x0 + pad
    y = y_top - pad
    for (s, fs, bold, italic, color, gap) in items:
        y -= gap
        h = lh_mm(fs)
        for ln in wrap(s, fs, maxw, bold, italic):
            ax.text(xc, y - h / 2, ln, fontsize=fs,
                    fontweight="bold" if bold else "normal",
                    fontstyle="italic" if italic else "normal",
                    color=color, ha=align, va="center", zorder=z + 3,
                    linespacing=LH)
            y -= h
    return y


def need_h(x0, x1, items, pad=2.4):
    """盒内文字所需高度（mm）。"""
    maxw = (x1 - x0) - 2 * pad
    h = 2 * pad
    for (s, fs, bold, italic, color, gap) in items:
        h += gap + len(wrap(s, fs, maxw, bold, italic)) * lh_mm(fs)
    return h


def panel(ax, x0, y0, x1, y1, style, items, pad=2.4, lw=None, ls="-", r=1.6,
          z=2, align="center", name=""):
    """画盒子 + 堆叠文字，返回文字底部 y，并在溢出时告警。"""
    rbox(ax, x0, y0, x1, y1, fc=style["fc"], ec=style["ec"],
         lw=LW if lw is None else lw, ls=ls, r=r, z=z)
    y = stack(ax, x0, x1, y1, items, pad=pad, z=z, align=align)
    left = y - (y0 + pad)
    if left < -0.2:
        print(f"  [warn] panel '{name}' overflows by {-left:.1f} mm "
              f"(needs {need_h(x0, x1, items, pad):.1f} mm)")
    return y


def I(s, fs=FS, bold=False, italic=False, color=INK, gap=0.0):
    return (s, fs, bold, italic, color, gap)


# ================================================================ Figure 1
def fig1(path):
    W, H = 170.0, 124.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    TOP, BOT = 108.0, 52.0
    A0, A1 = 2.0, 30.0
    B0, B1 = 34.0, 64.0
    C0, C1 = 68.0, 104.0
    D0, D1 = 108.0, 133.0
    E0, E1 = 137.0, 167.0

    # ---------------- 第 1 列：输入 / 预处理
    panel(ax, A0, 88, A1, TOP, C_MAIN, [
        I("Input", FS_T, bold=True),
        I("78 CICFlowMeter\nfeatures per flow", gap=1.2),
        I("one flow = one sample,\nnot a packet sequence",
          FS, italic=True, color=INK_SOFT, gap=1.4),
    ], name="input")

    panel(ax, A0, 62, A1, 84, C_MAIN, [
        I("Preprocessing", FS_T, bold=True),
        I("cleaning,\nde-duplication,\nstandardisation", gap=1.2),
        I("scaler fitted on real\nfitting rows only",
          FS, italic=True, color=INK_SOFT, gap=1.4),
    ], name="preproc")
    arrow(ax, (16, 88), (16, 84))

    # ---------------- 第 2 列：固定划分
    panel(ax, B0, BOT, B1, TOP, C_MAIN, [
        I("Fixed split", FS_T, bold=True),
        I("drawn once, seed = 42,\nreused by every arm",
          FS, italic=True, color=INK_SOFT, gap=1.2),
    ], name="split-head")

    sub = [
        (90, 100, ["Training pool", "1,764,511"]),
        (78, 88, ["Fitting set", "1,588,059"]),
        (66, 76, ["Validation", "176,452  (10%)"]),
        (54, 64, ["Test", "756,219"]),
    ]
    for (sy0, sy1, lines) in sub:
        rbox(ax, B0 + 2, sy0, B1 - 2, sy1, fc="#FFFFFF", ec="#6C8AA6",
             lw=0.6, r=1.0, z=3)
        y = (sy0 + sy1) / 2 + (len(lines) - 1) * lh_mm(FS) / 2
        for ln in lines:
            txt(ax, (B0 + B1) / 2, y, ln, z=6)
            y -= lh_mm(FS)

    arrow(ax, (30, 73), (34, 73))

    # ---------------- 第 3 列：训练期处理（橙，核心贡献）
    panel(ax, C0, BOT, C1, TOP, C_TRAIN, [
        I("Training-time treatment", FS_T, bold=True, color=ORANGE_DK),
        I("fitting set only", FS, italic=True, color=ORANGE_DK, gap=0.8),
        I("SMOTE, k = 5", gap=1.8),
        I("lesion classes only:", gap=0.6),
        I("Bot, Brute Force, XSS", gap=0.6),
        I("410–1,228 → 12,000 each", gap=0.6),
        I("+33,436 synthetic rows", italic=True, gap=0.6),
    ], lw=1.3, name="treatment")

    # BENIGN 全量：视觉突出（白底 + 橙色实线 + 加粗）
    rbox(ax, C0 + 2.4, 70, C1 - 2.4, 78, fc="#FFFFFF", ec=ORANGE, lw=1.2,
         r=1.0, z=4)
    txt(ax, (C0 + C1) / 2, 75.4, "BENIGN kept in full", bold=True, z=6)
    txt(ax, (C0 + C1) / 2, 72.2, "(no undersampling)", italic=True, z=6)

    # focal loss：虚线 = 可选
    rbox(ax, C0 + 2.4, 56, C1 - 2.4, 66, fc="#FFFFFF", ec=ORANGE,
         ls=(0, (2.0, 1.4)), lw=0.9, r=1.0, z=4)
    txt(ax, (C0 + C1) / 2, 63.4, "focal loss, γ = 1", z=6)
    txt(ax, (C0 + C1) / 2, 59.4, "(optional)", italic=True, color=INK_SOFT, z=6)

    arrow(ax, (64, 78), (68, 78), color=ORANGE)

    # ---------------- 第 4 列：backbone（蓝）
    panel(ax, D0, BOT, D1, TOP, C_MAIN, [
        I("Backbone", FS_T, bold=True),
        I("142,029\nparameters", gap=1.0),
        I("Conv1D ×2\n+ GroupNorm", gap=1.8),
        I("→ BiLSTM\n64 × 2 directions", gap=0.6),
        I("→ additive\nattention", gap=0.6),
        I("→ FC →\nsoftmax", gap=0.6),
        I("→ 12 classes", gap=0.6),
        I("shared by all arms", italic=True, color=INK_SOFT, gap=1.8),
    ], name="backbone")
    arrow(ax, (104, 80), (108, 80))

    # ---------------- 第 5 列：两个输出臂（C4 在上，便于验证集直连）
    panel(ax, E0, 80, E1, TOP, C_CAL, [
        I("C4  (reported)", FS_T, bold=True, color=GREEN),
        I("per-class weights w_c", gap=1.2),
        I("benign FPR ≤ 0.008", gap=0.6),
        I("optional stage", italic=True, color=GREEN, gap=1.2),
    ], lw=1.3, ls=(0, (3.2, 1.6)), name="C4")

    panel(ax, E0, BOT, E1, 76, C_MAIN, [
        I("C3", FS_T, bold=True),
        I("argmax over\n12 logits", gap=1.2),
        I("no calibration", italic=True, color=INK_SOFT, gap=0.8),
    ], name="C3")

    arrow(ax, (133, 93), (137, 93), color=GREEN)
    arrow(ax, (133, 64), (137, 64))

    # ---------------- 留出集连线
    # 验证集：split 顶 → 上 → 右 → C4 顶（早停 + 校准搜索）
    polyline(ax, [(49, TOP), (49, 116), (152, 116), (152, TOP)])
    arrow(ax, (152, 112), (152, TOP), color=C_HOLD, lw=0.85, ms=5.0)
    txt(ax, 95, 119.6,
        "Validation 176,452 — early-stopping monitor and calibration search set",
        italic=True, color="#4A4A4A")

    # 测试集：split 底 → 下 → 右 → backbone 底（只做前向）
    polyline(ax, [(49, BOT), (49, 42), (120, 42), (120, BOT)])
    arrow(ax, (120, 46), (120, BOT), color=C_HOLD, lw=0.85, ms=5.0)
    txt(ax, 85, 45.2,
        "Test 756,219 — forward pass only, weights frozen",
        italic=True, color="#4A4A4A")

    # ---------------- 泄漏控制注记
    panel(ax, 40, 8, 167, 36, C_NOTE, [
        I("Leakage control", FS_T, bold=True, color=INK_SOFT),
        I("Resampling is applied to the fitting set only, and only after the "
          "validation split is drawn. Validation and test are never resampled "
          "and never used to fit any parameter; the test set is evaluated "
          "exactly once, after w_c is frozen. Four set-intersection assertions "
          "are written on every run (val ∩ test, fit ∩ test, val ⊂ pool, "
          "fit ∩ val).", gap=1.4, color="#2E2E2E"),
    ], lw=0.7, name="leakage")

    # ---------------- 图例
    panel(ax, 2, 8, 36, 36, C_NOTE, [
        I("Legend", FS_T, bold=True, color=INK_SOFT),
    ], lw=0.7, name="legend")
    arrow(ax, (5, 24.5), (13, 24.5), ms=5.5)
    txt(ax, 15, 24.5, "data flow", ha="left")
    rbox(ax, 5, 17.4, 13, 20.6, fc="#FFFFFF", ec=ORANGE, ls=(0, (2.0, 1.4)),
         lw=0.9, r=0.7)
    txt(ax, 15, 19.0, "optional", ha="left")
    polyline(ax, [(5, 13.0), (13, 13.0)])
    txt(ax, 15, 13.0, "held-out use", ha="left")

    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


# ================================================================ Figure 2
def fig2(path):
    W = 85.0
    BX0, BX1 = 2.0, 44.0
    SX = 68.0          # shape 列右对齐
    PX = 84.0          # params 列右对齐
    PAD, PADV, GAP = 2.2, 1.5, 2.9

    rows = [
        # (名称, shape 行, 参数量, 是否强调)
        ("Input  x", ["(B, 1, 78)"], "", False),
        ("Conv1d 1→64, k=3, p=1\n+ GroupNorm(1, 64) + ReLU",
         ["(B, 64, 78)"], "384", True),
        ("MaxPool1d(2)", ["(B, 64, 39)"], "", False),
        ("Conv1d 64→128, k=3, p=1\n+ GroupNorm(1, 128) + ReLU",
         ["(B, 128, 39)"], "24,960", True),
        ("MaxPool1d(2)  *", ["(B, 128, 19)"], "", False),
        ("transpose(1, 2)", ["(B, 19, 128)"], "", False),
        ("BiLSTM  128 → 64 × 2\n1 layer, bidirectional",
         ["(B, 19, 128)"], "99,328", True),
        ("Additive attention\nW: 128→64,  v: 64→1\nsoftmax over 19 steps",
         ["α (B, 19)", "ctx (B, 128)"], "8,321", True),
        ("Dropout  p = 0.4", ["(B, 128)"], "", False),
        ("Linear 128→64 + ReLU", ["(B, 64)"], "8,256", True),
        ("Dropout  p = 0.3", ["(B, 64)"], "", False),
        ("Linear 64→12", ["(B, 12)"], "780", True),
        ("Softmax", ["12 probabilities"], "", False),
    ]

    maxw = (BX1 - BX0) - 2 * PAD
    heights = []
    for (name, shape, params, emph) in rows:
        nl = max(nlines(name, FS, maxw), len(shape))
        heights.append(nl * lh_mm(FS) + 2 * PADV)

    body = sum(heights) + GAP * (len(rows) - 1)
    HEAD = 5.0          # 表头
    TOTAL_H = 13.0      # 总计块
    FOOT = 15.0         # 脚注
    MARG = 2.5
    H = MARG + HEAD + body + 4.0 + TOTAL_H + 3.0 + FOOT + MARG

    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # 表头
    yh = H - MARG - 1.2
    txt(ax, (BX0 + BX1) / 2, yh, "layer", bold=True, color="#555555")
    txt(ax, SX, yh, "output shape", bold=True, color="#555555", ha="right")
    txt(ax, PX, yh, "params", bold=True, color="#555555", ha="right")

    y = H - MARG - HEAD
    centers = []
    for idx, (name, shape, params, emph) in enumerate(rows):
        y1 = y
        y0 = y1 - heights[idx]
        if emph:
            rbox(ax, BX0, y0, BX1, y1, **C_MAIN, lw=1.0)
        else:
            rbox(ax, BX0, y0, BX1, y1, fc="#FFFFFF", ec="#8FA3B5", lw=0.8, r=1.2)
        cy = (y0 + y1) / 2
        for ln in wrap(name, FS, maxw):
            pass
        lines = wrap(name, FS, maxw)
        yy = cy + (len(lines) - 1) * lh_mm(FS) / 2
        for ln in lines:
            txt(ax, (BX0 + BX1) / 2, yy, ln)
            yy -= lh_mm(FS)
        sy = cy + (len(shape) - 1) * lh_mm(FS) / 2
        for ln in shape:
            txt(ax, SX, sy, ln, ha="right")
            sy -= lh_mm(FS)
        if params:
            txt(ax, PX, cy, params, ha="right", color=BLUE)
        centers.append((y0, y1))
        y = y0 - GAP

    # 竖向箭头
    for i in range(len(rows) - 1):
        a = centers[i][0]
        b = centers[i + 1][1]
        arrow(ax, (23, a), (23, b), lw=0.85, ms=5.0)

    # 总计块
    ty1 = centers[-1][0] - 4.0
    ty0 = ty1 - TOTAL_H
    rbox(ax, 2, ty0, 84, ty1, fc="#FFFFFF", ec=BLUE, lw=1.1, r=1.2)
    txt(ax, 43, ty1 - 4.2, "Total  142,029 trainable parameters",
        fs=FS_T, bold=True, color=BLUE)
    txt(ax, 43, ty1 - 9.0, "sum of the six parameterised blocks",
        italic=True, color="#555555")

    # 脚注
    fy = ty0 - 3.4
    for ln in wrap("* The second MaxPool1d(2) maps 39 → 19 by flooring, so the "
                   "final pooled position (raw features 77–78: Idle Max / Idle "
                   "Min) never reaches the recurrent stage. All three deep "
                   "baselines share this arrangement.", FS, W - 2 * MARG,
                   italic=True):
        txt(ax, MARG, fy, ln, italic=True, color="#4A4A4A", ha="left",
            va="top")
        fy -= lh_mm(FS) * 1.02

    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = os.path.join("results", "figures")
    os.makedirs(out, exist_ok=True)
    p1 = fig1(os.path.join(out, "fig1_framework.png"))
    p2 = fig2(os.path.join(out, "fig2_backbone.png"))
    for p in (p1, p2):
        print(f"{p}  {os.path.getsize(p) / 1024:.1f} KB")
