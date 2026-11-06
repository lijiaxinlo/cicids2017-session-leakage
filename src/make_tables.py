"""汇总出表出图：Table 2 / Table 3 / E7 四臂表 / 图 4 / 图 5 / 主方法混淆矩阵。

数据全部来自 results/ 下已落盘 json / npy，不重算任何训练。
输出：results/tables/*.md + results/figures/*.png
"""
import json
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figures"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")
SEEDS3 = (42, 44, 46)
plt.rcParams.update({"figure.dpi": 300, "font.size": 9})


def ms(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def load_main(seed):
    for pat in (f"deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull*.json",):
        for f in glob.glob(str(RES / pat)):
            if "aborted" in f:
                continue
            return json.load(open(f, encoding="utf-8"))
    return None


def main_row(seed):
    d = load_main(seed)
    t, pc = d["test_metrics"], d["test_per_class"]
    return (t["macro_f1"], t["macro_recall"], t["fpr"], t["benign_fpr"],
            sum(pc[n]["f1"] for n in LESION))


def table2():
    base = json.load(open(RES / "deep_baselines_12class_summary.json",
                          encoding="utf-8"))
    rows = []
    # RF（加权 / 无权）
    for name, fn in (("RF (balanced)", "rf_12class_metrics.json"),
                     ("RF (unweighted)", "rf_12class_noweight_metrics.json")):
        p = RES / fn
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        agg = d.get("summary") or {}
        pcs = d.get("per_class_summary") or {}
        try:
            lesion = sum(pcs[n]["f1"]["mean"] for n in LESION)
        except Exception:
            lesion = float("nan")
        rows.append([name,
                     (agg["macro_f1"]["mean"], agg["macro_f1"]["std"]),
                     (agg["macro_recall"]["mean"], agg["macro_recall"]["std"]),
                     (agg["fpr"]["mean"], agg["fpr"]["std"]),
                     (float("nan"), 0.0),
                     lesion, None])
    for key, label in (("cnn", "CNN"), ("cnn_lstm", "CNN-LSTM"),
                       ("lstm", "LSTM")):
        v = base["models"][key]
        m = v["metrics"]
        pc = v["per_class_f1"]
        try:
            lesion = sum(pc[n]["mean"] if isinstance(pc[n], dict) else pc[n]
                         for n in LESION)
        except Exception:
            lesion = float("nan")
        rows.append([label,
                     (m["macro_f1"]["mean"], m["macro_f1"]["std"]),
                     (m["macro_recall"]["mean"], m["macro_recall"]["std"]),
                     (m["fpr"]["mean"], m["fpr"]["std"]) if "fpr" in m
                     else (float("nan"), 0.0),
                     (m["benign_fpr"]["mean"], m["benign_fpr"]["std"])
                     if "benign_fpr" in m else (float("nan"), 0.0),
                     lesion, v["n_params"]["total"]])
    # 主方法 C3 / C4
    c3 = [main_row(s) for s in SEEDS3]
    c3 = [r for r in c3 if r]
    e17 = json.load(open(RES / "e17_calibration_summary.json", encoding="utf-8"))
    c4 = [r for r in e17["c4"] if r["tier_benign_cap"] == 0.008]
    rows.append(["Ours (C3, argmax)",
                 ms([r[0] for r in c3]), ms([r[1] for r in c3]),
                 ms([r[2] for r in c3]), ms([r[3] for r in c3]),
                 ms([r[4] for r in c3])[0], 142029])
    rows.append(["Ours (C4, calibrated)",
                 ms([r["test_macro_f1"] for r in c4]),
                 (float("nan"), 0.0),
                 (float("nan"), 0.0),
                 ms([r["test_benign_fpr"] for r in c4]),
                 ms([r["test_lesion_sum"] for r in c4])[0], 142029])

    def f(x, nd=4):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) \
            else f"{x[0]:.{nd}f}±{x[1]:.{nd}f}"

    md = ["# Table 2 主结果（12 类，固定划分；深度/RF 5 种子，主方法 n=3）", "",
          "| Model | macro-F1 | macro-Recall | FPR (macro-OVR) | benign FPR | "
          "Lesion F1 sum | Params |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {f(r[1])} | {f(r[2])} | {f(r[3])} | {f(r[4])} | "
                  f"{'—' if r[5] is None or (isinstance(r[5], float) and np.isnan(r[5])) else f'{r[5]:.4f}'} | "
                  f"{'—' if r[6] is None else format(r[6], ',')} |")
    (TAB / "table2_main_results.md").write_text("\n".join(md) + "\n",
                                                encoding="utf-8")
    print("\n".join(md))


def table3():
    arms = [("A0 (Full)", "*seed{s}_12class_g1_bfull*.json"),
            ("A1 (w/o Attention)", "*seed{s}*_a1.json"),
            ("A2 (w/o BiLSTM)", "*seed{s}*_a2.json"),
            ("A3 (w/o imbalance, CE)", "*seed{s}*_a3.json"),
            ("A4 (CE + resampling)", "*seed{s}*_a4.json"),
            ("A5 (focal, w/o SMOTE)", "*seed{s}*_a5.json")]
    md = ["# Table 3 消融（种子 42/44/46，mean±std）", "",
          "| Variant | macro-F1 | Lesion F1 sum | benign FPR | Δmacro | Δlesion |",
          "|---|---|---|---|---|---|"]
    got = {}
    for label, pat in arms:
        vals = []
        for s in SEEDS3:
            fs = [f for f in glob.glob(str(RES / pat.format(s=s)))
                  if "aborted" not in f]
            if not fs:
                continue
            d = json.load(open(fs[0], encoding="utf-8"))
            t, pc = d["test_metrics"], d["test_per_class"]
            vals.append((t["macro_f1"],
                         sum(pc[n]["f1"] for n in LESION), t["benign_fpr"]))
        if vals:
            got[label] = vals
    a0 = got.get("A0 (Full)")
    for label, vals in got.items():
        mf, lf, bf = ms([v[0] for v in vals]), ms([v[1] for v in vals]), \
            ms([v[2] for v in vals])
        dm = f"{mf[0]-ms([v[0] for v in a0])[0]:+.4f}" if a0 and label != "A0 (Full)" else "—"
        dl = f"{lf[0]-ms([v[1] for v in a0])[0]:+.4f}" if a0 and label != "A0 (Full)" else "—"
        md.append(f"| {label} | {mf[0]:.4f}±{mf[1]:.4f} | {lf[0]:.4f}±{lf[1]:.4f} | "
                  f"{bf[0]:.4f} | {dm} | {dl} |")
    (TAB / "table3_ablation.md").write_text("\n".join(md) + "\n",
                                            encoding="utf-8")
    print("\n".join(md))


def table_e7():
    src = json.load(open(RES / "e7_four_arms.json", encoding="utf-8"))
    md = ["# E7 比例敏感性（逐类上限 R；5 种子 mean±std，cap=0.008）", "",
          "| Tier | C3 macro | C4 macro | C3 lesion | C4 lesion | C4 bFPR | "
          "RF macro | RF+cal macro | RF lesion | RF+cal lesion |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for R in ("10", "100", "1000"):
        s = src["summary_cap0008"].get(R)
        if not s:
            continue
        md.append(f"| 1:{R} | " + " | ".join(
            f"{s[k][0]:.4f}±{s[k][1]:.4f}" for k in
            ("C3_macro", "C4_macro", "C3_lesion", "C4_lesion", "C4_bfpr",
             "RF_macro", "RFcal_macro", "RF_lesion", "RFcal_lesion")) + " |")
    (TAB / "table_e7_ratio.md").write_text("\n".join(md) + "\n",
                                           encoding="utf-8")
    print("\n".join(md))


def fig5():
    src = json.load(open(RES / "e7_four_arms.json", encoding="utf-8"))
    tiers = [10, 100, 1000]
    x = np.arange(len(tiers))
    labels = [f"1:{t}" for t in tiers]
    s = src["summary_cap0008"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, key_m, key_l, title in (
            (axes[0], "macro", None, "(a) 12-class macro-F1"),
            (axes[1], "lesion", None, "(b) Lesion-group F1 sum")):
        for arm, label, style in (
                ("C3", "Ours (argmax)", "o-"),
                ("C4", "Ours (calibrated)", "s--"),
                ("RF", "RF", "^-"),
                ("RFcal", "RF + cal", "d--")):
            ys = [s[str(t)][f"{arm}_{key_m}"][0] for t in tiers]
            es = [s[str(t)][f"{arm}_{key_m}"][1] for t in tiers]
            ax.errorbar(x, ys, yerr=es, fmt=style, capsize=3, label=label,
                        lw=1.4, ms=4)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Per-class imbalance cap (BENIGN : class)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_ratio_sensitivity.png")
    print("fig5 saved")


def fig4():
    """主结果对比（macro-F1 与病灶组 F1 和）。"""
    base = json.load(open(RES / "deep_baselines_12class_summary.json",
                          encoding="utf-8"))
    names, mf, lf = [], [], []
    for key, label in (("cnn", "CNN"), ("cnn_lstm", "CNN-LSTM"),
                       ("lstm", "LSTM")):
        v = base["models"][key]
        names.append(label)
        mf.append(v["metrics"]["macro_f1"]["mean"])
        lf.append(sum(v["per_class_f1"][n]["mean"] for n in LESION))
    c3 = [main_row(s) for s in SEEDS3]
    c3 = [r for r in c3 if r]
    names.append("Ours (C3)")
    mf.append(ms([r[0] for r in c3])[0])
    lf.append(ms([r[4] for r in c3])[0])
    e17 = json.load(open(RES / "e17_calibration_summary.json", encoding="utf-8"))
    c4 = [r for r in e17["c4"] if r["tier_benign_cap"] == 0.008]
    names.append("Ours (C4)")
    mf.append(ms([r["test_macro_f1"] for r in c4])[0])
    lf.append(ms([r["test_lesion_sum"] for r in c4])[0])
    tp = json.load(open(RES / "valfit_treated_arm.json", encoding="utf-8"))
    sel = tp["summary"]  # 每种子 val 最优重启 = C4'
    names.append("Ours (C4')")
    mf.append(float(np.mean([s["lfsjoint_macro"] for s in sel])))
    lf.append(float(np.mean([s["lfsjoint_LFS"] for s in sel])))

    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax[0].bar(x, mf, color=["#9bb", "#9bb", "#9bb", "#c66", "#c66", "#c33"])
    ax[0].set_xticks(x, names, rotation=15, fontsize=7)
    ax[0].set_title("(a) 12-class macro-F1", fontsize=9)
    ax[1].bar(x, lf, color=["#9bb", "#9bb", "#9bb", "#c66", "#c66", "#c33"])
    ax[1].set_xticks(x, names, rotation=15, fontsize=7)
    ax[1].set_title("(b) Lesion-group F1 sum", fontsize=9)
    for a in ax:
        a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_main_comparison.png")
    print("fig4 saved")


def cm_fig():
    """主方法（seed44）混淆矩阵（行归一化）。"""
    p = RES / "proba" / "deep_cnn_bilstm_attn_seed44_12class_g1_bfull_g1_bfull_test.npy"
    yp = RES / "proba" / "deep_cnn_bilstm_attn_seed44_12class_g1_bfull_g1_bfull_test_ytrue.npy"
    if not (p.exists() and yp.exists()):
        print("混淆矩阵：缺 proba")
        return
    pred = np.load(p).argmax(1)
    y = np.load(yp)
    names = json.load(open(RES / "deep_baselines_12class_summary.json",
                           encoding="utf-8"))["class_names"]
    k = len(names)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), names, rotation=90, fontsize=6)
    ax.set_yticks(range(k), names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Ours (C3, seed44) row-normalized confusion matrix",
                 fontsize=9)
    fig.colorbar(im, fraction=.046)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_confusion_ours.png")
    print("fig6 saved")


if __name__ == "__main__":
    table2()
    table3()
    table_e7()
    fig4()
    fig5()
    cm_fig()
