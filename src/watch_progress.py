# -*- coding: utf-8 -*-
"""
批 1/2 训练进度看板（只读解析日志，不干扰训练）
=============================================
每 2 分钟重写一次 results/runs/batch1_progress.md，便于主理人不翻 10 个日志就能看进度。

规则：
  - 每跑完一个 epoch 一行：模型 | 种子 | epoch | val_macroF1 | 单轮耗时
  - 已跑完的种子，其最后一行加粗并标 ✅
  - 出现耗时异常的 epoch 标 ⚠
  - 全部预期种子跑完（或超过最大运行时长）后自动退出

用法
----
    python src/watch_progress.py                 # 单次生成
    python src/watch_progress.py --loop 120      # 每 120 秒刷新，自动退出
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "results" / "runs"
OUT = RUN_DIR / "batch1_progress.md"

MODELS = ["cnn", "cnn_lstm", "lstm"]
SEEDS = [42, 43, 44, 45, 46]
EXPECTED = len(MODELS) * len(SEEDS)

# [14:29:29] epoch 1/30 | train_loss 0.1245 | val_loss 0.0577 | gap -0.0668 |
#   val_macroF1 0.6634 | train 48.3s val 2.0s | peak 94 MB  <<< [耗时异常] ...
EPOCH_RE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2})\] epoch (?P<ep>\d+)/(?P<total>\d+) \| "
    r"train_loss (?P<tr>[\d.]+) \| val_loss (?P<va>[\d.]+) \| "
    r"gap (?P<gap>[-+][\d.]+) \| val_macroF1 (?P<f1>[\d.]+) \| "
    r"train (?P<sec>[\d.]+)s val (?P<vsec>[\d.]+)s"
)
ANOMALY_MARK = "[耗时异常]"


def parse_log(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            m = EPOCH_RE.match(line.strip())
            if m:
                rows.append({
                    "ts": m.group("ts"),
                    "epoch": int(m.group("ep")),
                    "total": int(m.group("total")),
                    "train_loss": float(m.group("tr")),
                    "val_loss": float(m.group("va")),
                    "gap": float(m.group("gap")),
                    "f1": float(m.group("f1")),
                    "sec": float(m.group("sec")),
                    "anomaly": ANOMALY_MARK in line,
                })
    return rows


def build():
    lines = ["# 批 1/2 训练进度", ""]
    lines.append(f"自动刷新（只读日志解析）｜更新时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append("图例：**粗体 ✅** = 该种子已完成（json 已落盘）；⚠ = 该 epoch 耗时异常（>1.5× 中位数）")
    lines.append("")

    done_models = {}
    for model in MODELS:
        done = 0
        for s in SEEDS:
            if (RUN_DIR / f"deep_{model}_seed{s}_12class.json").exists():
                done += 1
        done_models[model] = done

    total_done = sum(done_models.values())
    lines.append("## 完成度")
    lines.append("")
    lines.append("| 模型 | 已完成种子 |")
    lines.append("|---|---|")
    for model in MODELS:
        lines.append(f"| {model} | {done_models[model]}/5 |")
    lines.append(f"\n**合计 {total_done}/{EXPECTED}**\n")

    lines.append("## 逐 epoch 明细")
    lines.append("")
    lines.append("| 模型 | 种子 | epoch | val_macroF1 | 单轮耗时 | 状态 |")
    lines.append("|---|---|---|---|---|---|")
    for model in MODELS:
        for s in SEEDS:
            rows = parse_log(RUN_DIR / f"batch1_{model}_seed{s}.log")
            if not rows:
                continue
            finished = (RUN_DIR / f"deep_{model}_seed{s}_12class.json").exists()
            for i, r in enumerate(rows):
                last = i == len(rows) - 1
                flag = "⚠ 耗时异常" if r["anomaly"] else ""
                if finished and last:
                    lines.append(
                        f"| **{model}** | **{s}** | **{r['epoch']}/{r['total']}** | "
                        f"**{r['f1']:.4f}** | **{r['sec']:.1f}s** | **✅ 完成** {flag} |")
                else:
                    lines.append(f"| {model} | {s} | {r['epoch']}/{r['total']} | "
                                 f"{r['f1']:.4f} | {r['sec']:.1f}s | {flag} |")
    lines.append("")

    # 每种子小结
    lines.append("## 种子小结（最后一行 = 当前/最终）")
    lines.append("")
    lines.append("| 模型 | 种子 | 已跑 epoch | 当前 val_macroF1 | 最佳 val_macroF1 | 平均单轮耗时 | 完成 |")
    lines.append("|---|---|---|---|---|---|---|")
    for model in MODELS:
        for s in SEEDS:
            rows = parse_log(RUN_DIR / f"batch1_{model}_seed{s}.log")
            if not rows:
                continue
            secs = [r["sec"] for r in rows if not r["anomaly"]] or [r["sec"] for r in rows]
            lines.append(
                f"| {model} | {s} | {len(rows)} | {rows[-1]['f1']:.4f} | "
                f"{max(r['f1'] for r in rows):.4f} | {sum(secs) / len(secs):.1f}s | "
                f"{'✅' if (RUN_DIR / f'deep_{model}_seed{s}_12class.json').exists() else '—'} |")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    return total_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="刷新间隔秒数；0 = 只跑一次")
    ap.add_argument("--max-hours", type=float, default=12.0)
    args = ap.parse_args()

    t0 = time.time()
    while True:
        done = build()
        print(f"[{datetime.now():%H:%M:%S}] 进度已刷新：{done}/{EXPECTED} 种子完成 -> {OUT}",
              flush=True)
        if args.loop <= 0:
            break
        if done >= EXPECTED:
            print("全部种子完成，监控退出。", flush=True)
            break
        if time.time() - t0 > args.max_hours * 3600:
            print(f"已达最大运行时长 {args.max_hours}h，监控退出。", flush=True)
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
