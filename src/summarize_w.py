"""汇总类加权 CE 对照臂（W1/W2）并与 A3/A4 对照。

用法：python src/summarize_w.py
"""
import glob
import json
import statistics as st

import numpy as np


def load(tag):
    out = []
    files = []
    for pat in (f"results/deep_cnn_bilstm_attn_seed*_12class*_{tag}.json",
                f"results/runs/deep_cnn_bilstm_attn_seed*_12class*_{tag}.json"):
        files += glob.glob(pat)
    for f in sorted(files):
        d = json.load(open(f, encoding="utf-8"))
        if "main_metrics" not in d:
            print(f"  [skip] {f}: 无 main_metrics（aborted={d.get('aborted_reason')}）")
            continue
        pc = d["main_metrics"]["lesion_group_f1"]
        if any(r["seed"] == d["seed"] for r in out):   # results/ 与 results/runs/ 有副本
            continue
        out.append({
            "seed": d["seed"],
            "macro": d["main_metrics"]["macro_f1_12class"],
            "lfs": d["main_metrics"]["lesion_group_f1_sum"],
            "bfpr": d["test_metrics"]["benign_fpr"],
            "bot": pc.get("Bot", 0.0),
            "bf": pc.get("Web Attack-Brute Force", 0.0),
            "xss": pc.get("Web Attack-XSS", 0.0),
            "best_epoch": d.get("best_epoch"),
            "epochs_run": d.get("epochs_run"),
            "n_fit": d.get("n_fit_used"),
        })
    return out


def agg(rows):
    if not rows:
        return None
    f = lambda k: (st.mean(r[k] for r in rows), st.stdev(r[k] for r in rows) if len(rows) > 1 else 0.0)
    return {k: f(k) for k in ("macro", "lfs", "bfpr", "bot", "bf", "xss")}


if __name__ == "__main__":
    print(f"{'arm':<16}{'n':>3}  {'macro-F1':<16}{'LFS':<16}{'benign FPR':<16}{'Bot':<8}{'BF':<8}{'XSS':<8}")
    for name, tag in [("A3 (CE, none)", "a3"), ("A4 (CE+SMOTE)", "a4"),
                      ("W1 (1/sqrt n)", "w1_invsqrt"), ("W2 (1/n)", "w2_inv")]:
        rows = load(tag)
        a = agg(rows)
        if not a:
            print(f"{name:<16}{0:>3}  —")
            continue
        print(f"{name:<16}{len(rows):>3}  "
              f"{a['macro'][0]:.4f}±{a['macro'][1]:.4f}  "
              f"{a['lfs'][0]:.4f}±{a['lfs'][1]:.4f}  "
              f"{a['bfpr'][0]:.4f}±{a['bfpr'][1]:.4f}  "
              f"{a['bot'][0]:.3f}   {a['bf'][0]:.3f}   {a['xss'][0]:.3f}")
        for r in rows:
            print(f"      seed {r['seed']}: macro {r['macro']:.4f} | LFS {r['lfs']:.4f} "
                  f"| bFPR {r['bfpr']:.4f} | best_ep {r['best_epoch']} / {r['epochs_run']}")
