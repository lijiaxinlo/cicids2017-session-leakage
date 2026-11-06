# -*- coding: utf-8 -*-
"""返修汇总：把 P0/P1/P2 三套划分 × 处理/不处理 × 模型家族 × RF 臂摊成一张表。

    P0 = 全局随机 70/30（论文原划分）
    P1 = 每个 (会话,类) 组内随机 60/15/25（同组同尺寸，只打乱顺序）
    P2 = 每个 (会话,类) 组内时序 60/15/25 + purge=200

用法：python src/summarize_revision.py
输出：results/tables/table_revision.md
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN_DIRS = [ROOT / "results" / "runs", ROOT / "results"]
OUT = ROOT / "results" / "tables" / "table_revision.md"
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")


def load_deep():
    recs = {}
    for d in RUN_DIRS:
        for p in glob.glob(str(d / "*.json")):
            if p.endswith("_aborted.json"):
                continue
            try:
                r = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if "main_metrics" not in r or "test_metrics" not in r:
                continue
            recs[os.path.basename(p).replace(".json", "")] = r
    return recs


def load_rf():
    recs = {}
    for p in glob.glob(str(ROOT / "results" / "runs" / "rf12_*.json")):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        recs[os.path.basename(p).replace(".json", "")] = r
    return recs


def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def deep_row(recs, pred):
    sel = [r for t, r in recs.items() if pred(t, r)]
    if not sel:
        return None
    m = agg([r["main_metrics"]["macro_f1_12class"] for r in sel])
    l = agg([r["main_metrics"]["lesion_group_f1_sum"] for r in sel])
    f = agg([r["test_metrics"]["benign_fpr"] for r in sel])
    pc = {c: agg([r["main_metrics"]["lesion_group_f1"].get(c, 0.0) for r in sel])[0]
          for c in LESION}
    return {"n": len(sel), "seeds": sorted(r["seed"] for r in sel),
            "macro": m, "lfs": l, "fpr": f, "pc": pc}


def rf_row(recs, pred):
    sel = [r for t, r in recs.items() if pred(t, r)]
    if not sel:
        return None
    m = agg([r["metrics"]["macro_f1"] for r in sel])
    l = agg([sum(r["per_class"][c]["f1"] for c in LESION) for r in sel])
    # 良性误报率 = 1 - BENIGN 召回（不是 BENIGN 类那一行的 fpr 字段）
    f = agg([1.0 - r["per_class"]["BENIGN"]["recall"] for r in sel])
    pc = {c: agg([r["per_class"][c]["f1"] for r in sel])[0] for c in LESION}
    return {"n": len(sel), "seeds": sorted(r["seed"] for r in sel),
            "macro": m, "lfs": l, "fpr": f, "pc": pc}


def fmt(x, d=4):
    return f"{x[0]:.{d}f} ± {x[1]:.{d}f}" if x[1] > 1e-9 else f"{x[0]:.{d}f}"


def main():
    D, R = load_deep(), load_rf()
    rows = []

    def add(label, pred, kind="deep"):
        s = (rf_row(R, pred) if kind == "rf" else deep_row(D, pred))
        rows.append((label, s))
        if s:
            print(f"{label:<40} n={s['n']} {s['seeds']} macro={fmt(s['macro'])} "
                  f"LFS={fmt(s['lfs'])} FPR={fmt(s['fpr'])} "
                  f"Bot={s['pc']['Bot']:.3f} BF={s['pc']['Web Attack-Brute Force']:.3f} "
                  f"XSS={s['pc']['Web Attack-XSS']:.3f}")
        else:
            print(f"{label:<40} （无记录）")

    print("== P2 时序 ==")
    add("P2 deep untreated", lambda t, r: t.endswith("T_ce"))
    add("P2 deep treated", lambda t, r: t.endswith("T_trt_g1"))
    print("== P1 组内随机 ==")
    add("P1 deep untreated", lambda t, r: t.endswith("G_ce"))
    add("P1 deep treated", lambda t, r: t.endswith("G_trt"))
    print("== P0 全局随机 ==")
    add("P0 deep untreated (A3)", lambda t, r: t.endswith("_a3"))
    add("P0 deep treated (C3)", lambda t, r: "g1_bfull" in t and "sample" not in t)
    print("== P2 家族 ==")
    for mdl in ("cnn", "lstm", "cnn_lstm"):
        add(f"P2 {mdl} treated", lambda t, r, m=mdl: f"_{m}_seed" in t and t.endswith("T_trt_ng"))
    add("P2 cnn untreated", lambda t, r: "_cnn_seed" in t and t.endswith("T_ce_ng"))
    print("== P2 RF ==")
    add("P2 RF weighted", lambda t, r: t.startswith("rf12_temporal_p2_"), "rf")
    add("P2 RF untreated", lambda t, r: t.startswith("rf12_noweight_temporal_p2_"), "rf")
    add("P2 RF treated", lambda t, r: t.startswith("rf12_temporal_treat_p2treat_"), "rf")

    lines = ["# 返修新增臂汇总", "",
             "| 臂 | n | macro-F1 | LFS | benign FPR | Bot | BF | XSS |",
             "|---|---|---|---|---|---|---|---|"]
    for label, s in rows:
        if not s:
            lines.append(f"| {label} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(f"| {label} | {s['n']} | {fmt(s['macro'])} | {fmt(s['lfs'])} | "
                     f"{fmt(s['fpr'])} | {s['pc']['Bot']:.3f} | "
                     f"{s['pc']['Web Attack-Brute Force']:.3f} | "
                     f"{s['pc']['Web Attack-XSS']:.3f} |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
