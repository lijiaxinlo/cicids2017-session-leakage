# -*- coding: utf-8 -*-
"""
时序（purged）划分：训练/验证/测试在**时间上互不相邻**。
=========================================================

返修意见 1 的落点
----------------
审稿人指出：单一 flow-level 随机划分会让同一条连接（或相邻秒的 flow）同时
落进训练与测试，构成会话级泄漏。要求做 "session-disjoint **or** temporally
separated" 评估。

为什么不能做严格的 session-disjoint（这是数据事实，不是借口）
------------------------------------------------------------
`session_index.json` 显示：11 个攻击类**每一个都只出现在一个捕获会话里**
（Bot 只在周五上午，BF/XSS 只在周四上午 WebAttacks，DDoS 只在周五下午 DDos…），
只有 BENIGN 分布在全部 8 个会话。因此任何"留出一个会话"的划分都会让该会话
里的攻击类在训练集中彻底缺席 —— 12 类分层评估在该数据集上**不可识别**。

本模块给出的最强可行替代：**类内时序切分 + purge**
--------------------------------------------------
对每个 (会话, 类) 组，按文件内行号（CICFlowMeter 按流开始时间输出，行号单调
于时间）排序后切三段：

    [0, 0.60n) 训练   [0.60n, 0.75n) 验证   [0.75n, 1.0n) 测试

每个边界两侧各丢弃 `--purge` 行（默认 200），切断跨界连接的相邻 flow。

这样：
  * 每个类在训练/验证/测试里都有（可评估，且与随机划分的类别构成可比）；
  * 测试集的每一行在时间上都晚于训练集的任何一行（同会话内）；
  * 但"攻击战役身份"仍然共享 —— 这是该数据集无法消除的，论文里必须写明。

用法
----
    python src/temporal_split.py --build            # 生成划分 + 泄漏统计
    python src/temporal_split.py --check
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import data_pipeline as dp

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "results" / "splits"
OUT = ROOT / "results" / "temporal"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.60
VAL_FRAC = 0.15          # 验证占 15%，测试占 25%
PURGE = 200              # 边界两侧各丢弃的行数


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(purge=PURGE):
    y = np.load(dp.CACHE_DIR / dp.Y_FILE)
    sid = np.load(SPLIT_DIR / "session_id.npy")
    row = np.load(SPLIT_DIR / "row_in_file.npy")
    classes = dp.load_class_names()
    n_cls = len(classes)

    tr, va, te = [], [], []
    per_group = []
    for s in np.unique(sid):
        for c in range(n_cls):
            m = (sid == s) & (y == c)
            n = int(m.sum())
            if n == 0:
                continue
            idx = np.flatnonzero(m)
            idx = idx[np.argsort(row[idx], kind="stable")]
            i_tr = int(n * TRAIN_FRAC)
            i_va = int(n * (TRAIN_FRAC + VAL_FRAC))
            # purge：每个边界两侧各砍 purge 行（组太小时按比例退让）
            p = min(purge, max(0, n // 50))
            a_tr = idx[:max(0, i_tr - p)]
            a_va = idx[i_tr + p:i_va - p] if i_va - p > i_tr + p else idx[0:0]
            a_te = idx[i_va + p:]
            tr.append(a_tr); va.append(a_va); te.append(a_te)
            per_group.append({
                "session": int(s), "class": classes[c], "n": n,
                "train": int(len(a_tr)), "val": int(len(a_va)), "test": int(len(a_te)),
                "row_min": int(row[idx[0]]), "row_max": int(row[idx[-1]]),
            })

    tr = np.sort(np.concatenate(tr))
    va = np.sort(np.concatenate(va))
    te = np.sort(np.concatenate(te))

    # 硬断言：三集互斥
    assert len(np.intersect1d(tr, va)) == 0
    assert len(np.intersect1d(tr, te)) == 0
    assert len(np.intersect1d(va, te)) == 0

    np.save(OUT / "train_idx.npy", tr.astype(np.int64))
    np.save(OUT / "val_idx.npy", va.astype(np.int64))
    np.save(OUT / "test_idx.npy", te.astype(np.int64))

    # ---- 类别构成对照（必须与随机划分同量级，否则不能直接比） ----
    rand_tr = np.load(SPLIT_DIR / "train_idx.npy")
    rand_te = np.load(SPLIT_DIR / "test_idx.npy")
    comp = []
    for c in range(n_cls):
        comp.append({
            "class": classes[c],
            "temporal": {"train": int((y[tr] == c).sum()),
                         "val": int((y[va] == c).sum()),
                         "test": int((y[te] == c).sum())},
            "random": {"train": int((y[rand_tr] == c).sum()),
                       "test": int((y[rand_te] == c).sum())},
        })

    report = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": ("per (session, class) contiguous temporal split "
                     f"{TRAIN_FRAC:.2f}/{VAL_FRAC:.2f}/rest, purge={purge} rows per boundary"),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "composition": comp,
        "per_group": per_group,
    }
    (OUT / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    log(f"训练 {len(tr):,} / 验证 {len(va):,} / 测试 {len(te):,}（合计 {len(tr)+len(va)+len(te):,}）")
    for row_ in comp:
        t, r = row_["temporal"], row_["random"]
        log(f"  {row_['class']:<26} 时序 train {t['train']:>8,} val {t['val']:>7,} "
            f"test {t['test']:>7,} | 随机 train {r['train']:>8,} test {r['test']:>7,}")
    return report


def leakage_stats():
    """量化泄漏：每个测试 flow 到"同类训练 flow"在同一会话内的最近行距。

    随机划分下大量测试 flow 的训练邻居就在几行之内（同一条连接）；
    时序划分下这个距离被 purge 抬到数万行。
    """
    y = np.load(dp.CACHE_DIR / dp.Y_FILE)
    sid = np.load(SPLIT_DIR / "session_id.npy")
    row = np.load(SPLIT_DIR / "row_in_file.npy")
    classes = dp.load_class_names()

    protocols = {
        "random": (np.load(SPLIT_DIR / "train_idx.npy"), np.load(SPLIT_DIR / "test_idx.npy")),
        "temporal": (np.load(OUT / "train_idx.npy"), np.load(OUT / "test_idx.npy")),
    }
    res = {}
    for name, (tr, te) in protocols.items():
        per_cls = {}
        for c in range(len(classes)):
            ds = []
            for s in np.unique(sid):
                m_tr = tr[(sid[tr] == s) & (y[tr] == c)]
                m_te = te[(sid[te] == s) & (y[te] == c)]
                if len(m_tr) == 0 or len(m_te) == 0:
                    continue
                a = np.sort(row[m_tr])
                b = np.sort(row[m_te])
                pos = np.searchsorted(a, b)
                left = a[np.clip(pos - 1, 0, len(a) - 1)]
                right = a[np.clip(pos, 0, len(a) - 1)]
                ds.append(np.minimum(np.abs(b - left), np.abs(b - right)))
            if not ds:
                continue
            d = np.concatenate(ds)
            per_cls[classes[c]] = {
                "n_test": int(len(d)),
                "median_gap_rows": float(np.median(d)),
                "pct_within_10_rows": round(float((d <= 10).mean()) * 100, 2),
                "pct_within_100_rows": round(float((d <= 100).mean()) * 100, 2),
                "pct_within_1000_rows": round(float((d <= 1000).mean()) * 100, 2),
            }
        res[name] = per_cls
    (OUT / "leakage_stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    for name, per in res.items():
        log(f"== {name} ==")
        for c, v in per.items():
            log(f"  {c:<26} 中位行距 {v['median_gap_rows']:>10,.0f} | "
                f"≤10 行 {v['pct_within_10_rows']:>6.2f}% | "
                f"≤100 行 {v['pct_within_100_rows']:>6.2f}%")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--leak", action="store_true")
    ap.add_argument("--purge", type=int, default=PURGE)
    a = ap.parse_args()
    if a.build:
        build(a.purge)
    if a.leak:
        leakage_stats()
    if not (a.build or a.leak):
        ap.print_help()
