# -*- coding: utf-8 -*-
"""
深度 baseline 聚合（12 类口径，Table 2 深度模型行）
=================================================
把 results/runs/deep_<model>_seed<seed>_12class.json 聚合为 mean ± std。

只认正式全量结果（文件名严格为 deep_<model>_seed<seed>_12class.json）；
冒烟（_sample*）、诊断（_bn/_ln/_diag*）一律排除，避免污染论文表格。

用法
----
    python src/summarize_deep.py
    python src/summarize_deep.py --require-seeds 5   # 种子不足 5 个时报错（正式出表用）

产出
----
    results/deep_baselines_12class_summary.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"
RUN_DIR = OUT_DIR / "runs"

MODELS = ["cnn", "lstm", "cnn_lstm"]
SCALARS = ["macro_f1", "macro_recall", "macro_precision", "fpr", "accuracy"]
PATTERN = re.compile(r"^deep_(?P<model>cnn_lstm|cnn|lstm)_seed(?P<seed>\d+)_12class\.json$")


def collect(model):
    runs = []
    for p in sorted(RUN_DIR.glob(f"deep_{model}_seed*_12class.json")):
        m = PATTERN.match(p.name)
        if not m or m.group("model") != model:
            continue
        d = json.load(open(p, encoding="utf-8"))
        d["_file"] = p.name
        runs.append(d)
    runs.sort(key=lambda d: d["seed"])
    return runs


def agg(values):
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "values": [float(x) for x in v]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-seeds", type=int, default=0,
                    help="要求每个模型的种子数达到该值，否则报错退出")
    args = ap.parse_args()

    class_names = json.load(open(OUT_DIR / "cache" / "class_names.json",
                                encoding="utf-8"))
    summary = {
        "class_scheme": "12class",
        "class_names": class_names,
        "note": "仅聚合正式全量结果 deep_<model>_seed<seed>_12class.json；"
                "冒烟与诊断结果不参与",
        "protocol": {
            "epochs_cap": 30, "patience": 5, "batch_size": 256,
            "lr": 1e-3, "norm": "gn", "loss": "CrossEntropyLoss",
            "early_stop_metric": "val macro-F1（验证集来自训练池，测试集不参与）",
            "best_epoch_rollback": True,
        },
        "models": {},
    }

    rows = []
    for model in MODELS:
        runs = collect(model)
        if not runs:
            print(f"[跳过] {model}: 无正式结果")
            continue
        if args.require_seeds and len(runs) < args.require_seeds:
            raise SystemExit(f"{model} 只有 {len(runs)} 个种子，"
                             f"要求 {args.require_seeds} 个，拒绝出表")

        # 出表前置条件：所有种子的泄漏检查必须通过，任一 false 即拒绝出表
        bad_leak = [r["_file"] for r in runs
                    if r.get("leakage_checks", {}).get("passed", None) is not True]
        if bad_leak:
            raise SystemExit(
                f"{model} 存在泄漏检查未通过或缺失的结果，拒绝出表（不得进入 Table 2）：\n  "
                + "\n  ".join(bad_leak))

        entry = {
            "n_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "files": [r["_file"] for r in runs],
            "n_params": runs[0]["n_params"],
            "config": runs[0]["config"],
            "metrics": {k: agg([r["test_metrics"][k] for r in runs])
                        for k in SCALARS},
            "per_class_recall": {
                c: agg([r["test_per_class"][c]["recall"] for r in runs])
                for c in class_names
            },
            "per_class_f1": {
                c: agg([r["test_per_class"][c]["f1"] for r in runs])
                for c in class_names
            },
            "support": {c: runs[0]["test_per_class"][c]["support"]
                        for c in class_names},
            "best_epoch": [r["best_epoch"] for r in runs],
            "epochs_run": [len(r["history"]) for r in runs],
            "timing": {
                "total_sec": agg([r["timing"]["total_sec"] for r in runs]),
                "epoch_train_sec_median": agg([
                    r["timing"].get("epoch_train_sec_median",
                                    r["timing"]["epoch_train_sec_avg"])
                    for r in runs]),
                "timing_anomalies": [len(r["timing"].get("timing_anomalies", []))
                                     for r in runs],
            },
            "peak_gpu_mb_max": max(r["peak_gpu_mb"] for r in runs),
            "leakage_checks_passed": all(
                r.get("leakage_checks", {}).get("passed", None) is True
                for r in runs),
        }
        summary["models"][model] = entry

        rows.append({
            "model": model,
            **{k: entry["metrics"][k] for k in SCALARS},
            "n_seeds": entry["n_seeds"],
        })

    out = OUT_DIR / "deep_baselines_12class_summary.json"
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    # ---------- 打印 Table 2 深度模型行 ----------
    print("=" * 96)
    print("  Table 2 · 深度模型行（12 类，mean ± std）")
    print("=" * 96)
    print(f"  {'model':<10}{'seeds':>6}{'macro-F1':>18}{'macro-Recall':>18}"
          f"{'macro-Precision':>20}{'FPR':>12}{'best_ep':>14}")
    for r in rows:
        print(f"  {r['model']:<10}{r['n_seeds']:>6}"
              f"{r['macro_f1']['mean']:>10.4f}±{r['macro_f1']['std']:<6.4f}"
              f"{r['macro_recall']['mean']:>10.4f}±{r['macro_recall']['std']:<6.4f}"
              f"{r['macro_precision']['mean']:>12.4f}±{r['macro_precision']['std']:<6.4f}"
              f"{r['fpr']['mean']:>12.6f}"
              f"{str(summary['models'][r['model']]['best_epoch']):>14}")

    print(f"\n  {'class':<26}" + "".join(f"{r['model']:>16}" for r in rows))
    for c in class_names:
        line = f"  {c:<26}"
        for r in rows:
            d = summary["models"][r["model"]]["per_class_recall"][c]
            line += f"{d['mean']:>9.4f}±{d['std']:<6.4f}"
        print(line)
    print("=" * 96)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
