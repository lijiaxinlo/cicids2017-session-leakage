# -*- coding: utf-8 -*-
"""
"组内随机"对照划分：把"会话泄漏"与"会话内时间漂移"拆开。

返修意见 1 说随机划分可能因会话级泄漏而高估性能。要证明降低来自**时间**而不是
仅仅来自"换了类别比例"，需要第三个点：

    S0  全局随机 70/30            （论文原划分）
    S1  每个 (会话, 类) 组内随机 60/15/25  （与 S2 同组、同尺寸，只打乱顺序）
    S2  每个 (会话, 类) 组内时序 60/15/25 + purge（返修主协议）

S0 → S1 隔离"类别比例/组尺寸"的影响；S1 → S2 隔离"时间顺序"的影响。
若 S1 明显高于 S2，说明性能下降来自会话内的时间漂移（模型记住了同一战役里
相邻时刻的流），而不是类别构成变化。

用法：python src/grouped_split.py --build
输出：results/temporal_grouped/{train,val,test}_idx.npy
"""
import argparse
import json
from pathlib import Path

import numpy as np

import data_pipeline as dp

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "results" / "splits"
OUT = ROOT / "results" / "temporal_grouped"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC, VAL_FRAC, SEED = 0.60, 0.15, 20261023


def build():
    y = np.load(dp.CACHE_DIR / dp.Y_FILE)
    sid = np.load(SPLIT_DIR / "session_id.npy")
    classes = dp.load_class_names()
    rng = np.random.default_rng(SEED)

    tr, va, te = [], [], []
    for s in np.unique(sid):
        for c in range(len(classes)):
            idx = np.flatnonzero((sid == s) & (y == c))
            n = len(idx)
            if n == 0:
                continue
            idx = rng.permutation(idx)          # 唯一区别：不打乱 vs 打乱
            i_tr, i_va = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
            tr.append(idx[:i_tr]); va.append(idx[i_tr:i_va]); te.append(idx[i_va:])

    tr, va, te = np.sort(np.concatenate(tr)), np.sort(np.concatenate(va)), np.sort(np.concatenate(te))
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0
    np.save(OUT / "train_idx.npy", tr.astype(np.int64))
    np.save(OUT / "val_idx.npy", va.astype(np.int64))
    np.save(OUT / "test_idx.npy", te.astype(np.int64))
    (OUT / "manifest.json").write_text(json.dumps({
        "protocol": "per (session, class) random 0.60/0.15/rest, same group sizes as temporal split",
        "seed": SEED, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练 {len(tr):,} / 验证 {len(va):,} / 测试 {len(te):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    else:
        ap.print_help()
