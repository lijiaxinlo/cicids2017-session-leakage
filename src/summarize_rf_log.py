# -*- coding: utf-8 -*-
"""把 feature_rf.py 的逐种子日志汇总成 JSON（用于只跑了部分种子、提前停机的场合）。

feature_rf.py 只在 5 个种子全部跑完后才写 JSON；128 维表征上单种子约 20 分钟，
若只取前 k 个种子，用本脚本把日志里的行拼回同一结构的 summary。

日志行格式：
    seed 42: macro-F1 0.7946  LFS 1.0237  fpr 0.022758  (1228s)

用法
----
    python src/summarize_rf_log.py results/rf_feat_ctx.log --key ctx --out results/rf_feat_ctx_3seed.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAT = re.compile(
    r"seed\s+(\d+):\s+macro-F1\s+([\d.]+)\s+LFS\s+([\d.]+)\s+fpr\s+([\d.]+)\s+\(([\d.]+)s\)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = []
    for line in (ROOT / args.log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = PAT.search(line)
        if m:
            runs.append({"seed": int(m.group(1)),
                         "macro_f1": float(m.group(2)),
                         "lesion_sum": float(m.group(3)),
                         "benign_fpr": float(m.group(4)),
                         "fit_seconds": float(m.group(5))})
    if not runs:
        raise SystemExit("日志里没有种子结果行")

    def agg(k):
        v = [r[k] for r in runs]
        return float(np.mean(v)), float(np.std(v))   # ddof=0，与全文一致

    summary = {k: dict(zip(("mean", "std"), agg(k)))
               for k in ("macro_f1", "lesion_sum", "benign_fpr")}
    out = {"model": "RandomForest", "features": args.key,
           "source_log": args.log, "seeds": [r["seed"] for r in runs],
           "n_seeds": len(runs),
           "timestamp": datetime.now().isoformat(timespec="seconds"),
           "summary": summary, "per_seed": runs}
    p = ROOT / args.out
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(runs)} seeds -> {p}")
    print(f"macro-F1 {summary['macro_f1']['mean']:.4f} +- {summary['macro_f1']['std']:.4f} | "
          f"LFS {summary['lesion_sum']['mean']:.4f} +- {summary['lesion_sum']['std']:.4f} | "
          f"fpr {summary['benign_fpr']['mean']:.6f} +- {summary['benign_fpr']['std']:.6f}")


if __name__ == "__main__":
    main()
