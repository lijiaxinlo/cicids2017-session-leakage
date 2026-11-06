# -*- coding: utf-8 -*-
"""
会话索引：给缓存里的每一行标上"来自哪个捕获文件"和"文件内第几行"。
=====================================================================

为什么需要
----------
返修意见 1 指出：单一 flow-level 随机划分会让同一条 TCP 连接（或同一秒内
的相邻 flow）同时落进训练与测试，属于会话级泄漏，会高估性能。要回答这个
问题，必须知道每行来自哪个捕获会话。

`data_pipeline.load_clean()` 在 `pd.concat` 之后直接 `drop_duplicates()`，
没有保留来源列，所以缓存里的 X/y 已经丢了会话信息。本模块**原样复刻**
`load_clean` + `apply_class_scheme` 的每一步，只在 concat 时多带两列
`_src` / `_row`，因此产出的行序与 `X_float32_12class.npy` 严格一致。

正确性保证
----------
产出的 `y_session.npy` 必须与缓存 `y_12class_int8.npy` **逐元素相等**，
否则会话索引就是错位的，本模块直接报错退出。校验通过才落盘。

产物（results/splits/）
    session_id.npy    int8   0..7，对应 SESSIONS 顺序
    row_in_file.npy   int32  该行在其源 CSV 中的原始行号（0-based，不含表头）
    file_len.npy      int64  每个源 CSV 的数据行数（按 SESSIONS 顺序）
    session_index.json       校验报告 + 每会话 x 每类计数

用法
----
    python src/session_index.py --build
    python src/session_index.py --check
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipeline as dp

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "results" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [
    "Friday-WorkingHours-Afternoon-DDos",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Morning",
    "Monday-WorkingHours",
    "Thursday-WorkingHours-Afternoon-Infilteration",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build():
    files = sorted(dp.RAW_DIR.glob("*.csv"))
    names = [f.stem.replace(".pcap_ISCX", "") for f in files]
    if names != SESSIONS:
        raise SystemExit(f"会话顺序与预期不符，不要继续。\n实际: {names}\n预期: {SESSIONS}")
    log(f"发现 {len(files)} 个会话文件")

    frames = []
    file_len = []
    for sid, f in enumerate(files):
        t0 = time.time()
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        file_len.append(len(df))
        df["_src"] = np.int8(sid)
        df["_row"] = np.arange(len(df), dtype=np.int32)
        frames.append(df)
        log(f"  [{sid}] {f.name}: {len(df):,} 行 ({time.time() - t0:.1f}s)")

    feat_cols = [c for c in frames[0].columns if c not in ("_src", "_row")]
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    log(f"合并: {len(df):,} 行 x {len(feat_cols)} 列")

    # 与 load_clean 一致：去重（等价于对原始列集 drop_duplicates，keep='first'），
    # 只是把 _src/_row 排除在比较列之外，从而保留**首次出现**那一行的来源。
    n0 = len(df)
    df = df.drop_duplicates(subset=feat_cols, keep="first")
    log(f"去重: {n0:,} -> {len(df):,}（移除 {n0 - len(df):,}）")
    gc.collect()

    n1 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feat_cols)
    log(f"清理 inf/NaN: {n1:,} -> {len(df):,}")
    gc.collect()

    lab = df["Label"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    for bad, good in dp.DASH_MAP.items():
        lab = lab.str.replace(bad, good, regex=False)
    lab = lab.str.replace("Web Attack - ", "Web Attack-", regex=False)
    df["Label"] = lab

    mapped = df["Label"].map(dp.LABEL_MAP)
    keep = mapped.notna().values
    df = df[keep]
    mapped = mapped[keep]
    log(f"12 类口径保留 {len(df):,} 行")

    y = mapped.map(dp.CLASS_TO_ID).to_numpy(np.int8)
    y_cache = np.load(dp.CACHE_DIR / dp.Y_FILE)

    if len(y) != len(y_cache) or not np.array_equal(y, y_cache):
        bad = np.flatnonzero(y != y_cache)[:5] if len(y) == len(y_cache) else "长度不同"
        raise SystemExit(
            f"复刻失败：y 与缓存不一致（len {len(y):,} vs {len(y_cache):,}），"
            f"首个不一致位置 {bad}。会话索引不可信，已中止。"
        )
    log("校验通过：y 与缓存逐元素相等")

    sid = df["_src"].to_numpy(np.int8)
    row = df["_row"].to_numpy(np.int32)

    np.save(SPLIT_DIR / "session_id.npy", sid)
    np.save(SPLIT_DIR / "row_in_file.npy", row)
    np.save(SPLIT_DIR / "file_len.npy", np.array(file_len, dtype=np.int64))

    # 每会话 x 每类计数 + 每类在各会话里的行号分位（判断时间结构）
    cls_names = dp.CLASS_NAMES
    table = {}
    for s, nm in enumerate(SESSIONS):
        m = sid == s
        cnt = {cls_names[c]: int((y[m] == c).sum()) for c in range(len(cls_names))
               if (y[m] == c).sum() > 0}
        table[nm] = {"rows": int(m.sum()), "classes": cnt}

    span = {}
    for c, cn in enumerate(cls_names):
        per = {}
        for s, nm in enumerate(SESSIONS):
            m = (sid == s) & (y == c)
            if m.sum() == 0:
                continue
            r = np.sort(row[m])
            q = np.percentile(r, [0, 5, 50, 95, 100]).astype(int).tolist()
            per[nm] = {"n": int(m.sum()), "row_min": q[0], "row_p05": q[1],
                       "row_med": q[2], "row_p95": q[3], "row_max": q[4],
                       "file_len": int(file_len[s]),
                       "span_frac": round((q[4] - q[0]) / file_len[s], 4)}
        span[cn] = per

    report = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verified_equal_to_cache": True,
        "n_rows": int(len(y)),
        "sessions": table,
        "class_row_span_per_session": span,
        "note": "row_in_file 是源 CSV 内的原始行号；CICFlowMeter 按流开始时间输出，"
                "故行号单调于时间，可作为会话内的时间轴。",
    }
    with open(SPLIT_DIR / "session_index.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log(f"已写出 {SPLIT_DIR/'session_id.npy'} 等 4 个文件")
    return report


def check():
    for nm in ("session_id.npy", "row_in_file.npy", "file_len.npy"):
        p = SPLIT_DIR / nm
        print(f"{nm}: {'OK' if p.exists() else '缺失'}")
    rep = json.loads((SPLIT_DIR / "session_index.json").read_text(encoding="utf-8"))
    print(json.dumps(rep["sessions"], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.check:
        check()
    else:
        ap.print_help()
