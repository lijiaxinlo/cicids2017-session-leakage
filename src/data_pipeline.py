# -*- coding: utf-8 -*-
"""
CIC-IDS2017 共用数据管线（12 类口径 · v2）
==========================================
论文所有模型（RF / CNN / LSTM / CNN-LSTM / 主方法）必须共用本模块产出的
同一份清洗结果与同一份训练/测试划分，否则任何对比都无效。

口径变更历史
------------
v1（11 类，已归档到 results/archive/11class/）：把 Web Attack 三个子类合并为
   一类，macro-F1 从 0.8550 冲到 0.9763。但这等于把论文要解决的问题
   （Web Attack 子类互相混淆）定义掉了，故废弃。
v2（12 类，当前）：**保留 Brute Force 与 XSS 分开**（二者混淆正是本文要解决的
   问题），只剔除三个统计上不可用的类：Web Attack-Sql Injection(21)、
   Infiltration(36)、Heartbleed(11)，共 68 行。按标签剔除、不按文件剔除，
   BENIGN 一行不动。

设计要点
--------
1. `load_clean()`    : 读 8 个 CSV -> 列名 strip -> 标签破折号归一 -> 去重
                       -> 去 inf/NaN -> 数值列 float32
2. `apply_class_scheme()` : 15 类 -> 12 类
3. `get_split()`     : 读或**只建一次**固定划分（12 类分层，test_size=0.3, seed=42），
                       之后所有模型、所有种子一律 np.load 这份划分
4. `build_arrays()`  : X 落盘为 float32 .npy（约 786 MB），后续 mmap 加载，
                       避免每次重读 844 MB CSV

缓存产物（results/cache/）
    X_float32_12class.npy   2,520,730 x 78 float32
    y_12class_int8.npy      2,520,730 int8（0..11，顺序见 class_names.json）
    feature_names.json / class_names.json / cache_manifest.json

划分产物（results/splits/）
    train_idx.npy / test_idx.npy / split_manifest.json

用法
----
    python src/data_pipeline.py --build          # 一次性生成缓存 + 划分 + 报告
    python src/data_pipeline.py --check          # 只校验已有缓存与划分
    python src/data_pipeline.py --dist           # 打印 12 类分布表
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results"
CACHE_DIR = OUT_DIR / "cache"
SPLIT_DIR = OUT_DIR / "splits"

for _d in (OUT_DIR, CACHE_DIR, SPLIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SCHEME_VERSION = "v2-12class"
RANDOM_STATE = 42
TEST_SIZE = 0.3

# 期望值（15 类清洗后 2,520,798，剔除 Sql Injection 21 + Infiltration 36 + Heartbleed 11）
EXPECTED_ROWS = 2_520_730
EXPECTED_N_FEATURES = 78
EXPECTED_N_CLASSES = 12
MIN_CLASS_COUNT = 600
EXPECTED_BENIGN = 2_095_057  # 与 15 类口径完全一致，证明"按标签剔除"没碰 BENIGN

X_FILE = f"X_float32_{EXPECTED_N_CLASSES}class.npy"
Y_FILE = f"y_{EXPECTED_N_CLASSES}class_int8.npy"

# ---------------------------------------------------------------- 标签规范化
# 本镜像的 CSV 在制作时把 en dash 破坏成了 U+FFFD（替换字符，原始字节
# \xef\xbf\xbd）。信息已丢失、换编码也恢复不了，只能按 dash 处理。
# 这段逻辑来自 baseline_rf.py，是踩过坑的，不要"优化"掉。
DASH_MAP = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\ufffd": "-"}

# ---------------------------------------------------------------- 12 类方案
# 类别顺序即标签整数编码（0..11）。顺序固定，落盘后不得改动。
CLASS_NAMES = [
    "BENIGN",
    "DoS Hulk",
    "DDoS",
    "PortScan",
    "DoS GoldenEye",
    "FTP-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "SSH-Patator",
    "Bot",
    "Web Attack-Brute Force",
    "Web Attack-XSS",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# 15 类（归一化后的字符串）-> 12 类映射；值为 None 表示按标签剔除
LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS Hulk",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "DoS GoldenEye": "DoS GoldenEye",
    "FTP-Patator": "FTP-Patator",
    "DoS slowloris": "DoS slowloris",
    "DoS Slowhttptest": "DoS Slowhttptest",
    "SSH-Patator": "SSH-Patator",
    "Bot": "Bot",
    # 保留：二者互相混淆正是本文要解决的问题
    "Web Attack-Brute Force": "Web Attack-Brute Force",
    "Web Attack-XSS": "Web Attack-XSS",
    # 剔除：样本量过少，单条误判就能让该类召回跳动 17–33 个百分点
    "Web Attack-Sql Injection": None,   # 21 条，测试集仅 6 条
    "Infiltration": None,               # 36 条
    "Heartbleed": None,                 # 11 条
}

DROPPED_LABELS = [k for k, v in LABEL_MAP.items() if v is None]
MERGED_LABELS = {k: v for k, v in LABEL_MAP.items() if v is not None and k != v}

DROP_REASON = {
    "Web Attack-Sql Injection": "仅 21 条（测试集 6 条），一条误判即令该类召回变动 16.7 个百分点",
    "Infiltration": "仅 36 条（测试集 11 条），无统计意义",
    "Heartbleed": "仅 11 条（测试集 3 条），无统计意义",
}
KEEP_REASON = {
    "Web Attack-Brute Force": "1,470 条（测试集 441 条），足以支撑逐类召回评估",
    "Web Attack-XSS": "652 条（测试集 196 条）；与 Brute Force 的混淆正是本文的立项问题，必须保留",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            b = fp.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 1. 加载清洗
def load_clean(verbose=True):
    """读 8 个 CSV，返回清洗后的 DataFrame（Label 已归一，数值列 float32）。

    去重在 float64 上进行，与 baseline_rf.py 完全一致（float32 去重可能因
    精度截断误合并不同行，故不提前降精度）。
    """
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"未在 {RAW_DIR} 找到 CSV，请先下载 CIC-IDS2017。")
    log(f"发现 {len(files)} 个 CSV 文件")

    frames, col_sets = [], {}
    for f in files:
        t0 = time.time()
        df = pd.read_csv(f, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]  # 列名带前导空格，必须 strip
        col_sets[f.name] = tuple(df.columns)
        frames.append(df)
        log(f"  读入 {f.name}: {df.shape[0]:,} 行 x {df.shape[1]} 列 "
            f"({time.time() - t0:.1f}s)")

    base = col_sets[files[0].name]
    for name, cols in col_sets.items():
        if cols != base:
            log(f"  [警告] 列结构与其他文件不一致: {name} "
                f"(缺失 {set(base) - set(cols)}, 多出 {set(cols) - set(base)})")

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    log(f"合并完成: {df.shape[0]:,} 行 x {df.shape[1]} 列")

    if "Label" not in df.columns:
        raise SystemExit(f"未找到 Label 列。现有列示例: {list(df.columns[:8])}")

    n0 = len(df)
    df = df.drop_duplicates()
    log(f"去重: {n0:,} -> {len(df):,}（移除 {n0 - len(df):,} 行）")
    gc.collect()

    n1 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    log(f"清理 inf/NaN: {n1:,} -> {len(df):,}（移除 {n1 - len(df):,} 行）")
    gc.collect()

    # 标签规范化：strip + 压缩空白 + 破折号归一
    lab = df["Label"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    for bad, good in DASH_MAP.items():
        lab = lab.str.replace(bad, good, regex=False)
    lab = lab.str.replace("Web Attack - ", "Web Attack-", regex=False)
    df["Label"] = lab

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    log(f"数值列已降为 float32（{len(num_cols)} 列）")
    gc.collect()
    return df


# ---------------------------------------------------------------- 2. 类别方案
def apply_class_scheme(df, verbose=True):
    """15 类 -> 12 类。返回 (df12, report)。df12 含 int8 列 'y'。"""
    dist15 = df["Label"].value_counts()
    if verbose:
        print("\n=== 原始 15 类分布 ===")
        tot = dist15.sum()
        for name, cnt in dist15.items():
            print(f"  {name:<28} {cnt:>10,}  ({cnt / tot * 100:7.4f}%)")

    unknown = sorted(set(dist15.index) - set(LABEL_MAP))
    if unknown:
        raise SystemExit(f"发现未在 LABEL_MAP 中定义的标签: {unknown}\n"
                         f"请先确认标签归一是否生效，不要盲目继续。")

    mapped = df["Label"].map(LABEL_MAP)
    n_drop = int(mapped.isna().sum())
    df12 = df[mapped.notna()].copy()
    df12["class_name"] = mapped[mapped.notna()].values

    if verbose:
        log(f"按标签剔除 {n_drop} 行: {DROPPED_LABELS}")
        log("保留 Web Attack-Brute Force 与 Web Attack-XSS 为独立两类")

    df12["y"] = df12["class_name"].map(CLASS_TO_ID).astype(np.int8)

    report = {
        "n_before": int(len(df)),
        "n_after": int(len(df12)),
        "n_dropped": n_drop,
        "dropped_labels": {k: int(dist15.get(k, 0)) for k in DROPPED_LABELS},
        "kept_web_labels": {k: int(dist15.get(k, 0)) for k in KEEP_REASON},
    }
    return df12, report


# ---------------------------------------------------------------- 3. 固定划分
def get_split(y=None, force=False):
    """返回 (train_idx, test_idx)。

    只建一次：若 results/splits/ 下已存在则直接 np.load，绝不重新 split。
    """
    tr_p = SPLIT_DIR / "train_idx.npy"
    te_p = SPLIT_DIR / "test_idx.npy"
    if tr_p.exists() and te_p.exists() and not force:
        tr = np.load(tr_p)
        te = np.load(te_p)
        log(f"复用已有划分: 训练 {len(tr):,} / 测试 {len(te):,}")
        return tr, te

    if y is None:
        y = np.load(CACHE_DIR / Y_FILE)
    t0 = time.time()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y,
                              random_state=RANDOM_STATE)
    tr, te = np.sort(tr), np.sort(te)
    np.save(tr_p, tr.astype(np.int64))
    np.save(te_p, te.astype(np.int64))
    log(f"已生成固定划分（{time.time() - t0:.1f}s）: 训练 {len(tr):,} / 测试 {len(te):,}")
    return tr, te


def write_split_manifest(y, tr, te, cmd=None):
    counts = []
    for i, name in enumerate(CLASS_NAMES):
        n_all = int((y == i).sum())
        n_tr = int((y[tr] == i).sum())
        n_te = int((y[te] == i).sum())
        counts.append({"id": i, "class": name, "total": n_all,
                       "train": n_tr, "test": n_te,
                       "ratio_pct": round(n_all / len(y) * 100, 4)})
    manifest = {
        "scheme": SCHEME_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": cmd or " ".join(sys.argv),
        "protocol": {
            "stratify_on": f"{len(CLASS_NAMES)}-class label",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "split_computed_once": True,
            "note": "此后所有模型、所有种子一律 np.load 本划分，禁止重新 split",
            "supersedes": "v1-11class（已归档 results/archive/11class/）",
        },
        "n_total": int(len(y)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "sha256": {
            "train_idx.npy": _sha256(SPLIT_DIR / "train_idx.npy"),
            "test_idx.npy": _sha256(SPLIT_DIR / "test_idx.npy"),
        },
        "per_class": counts,
    }
    with open(SPLIT_DIR / "split_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    return manifest


# ---------------------------------------------------------------- 4. 数组缓存
def build_arrays(df12, force=False):
    """X -> float32 .npy，y -> int8 .npy；返回 (X_path, y_path, feature_cols)。"""
    feature_cols = [c for c in df12.columns
                    if c not in ("Label", "class_name", "y")]
    xp = CACHE_DIR / X_FILE
    yp = CACHE_DIR / Y_FILE

    y = df12["y"].to_numpy(dtype=np.int8)
    if force or not yp.exists():
        np.save(yp, y)
        log(f"已写入 {yp} ({yp.stat().st_size / 1e6:.1f} MB)")
    else:
        log(f"{yp} 已存在，跳过")

    if force or not xp.exists():
        X = df12[feature_cols].to_numpy(dtype=np.float32)
        assert X.shape[1] == EXPECTED_N_FEATURES, X.shape
        np.save(xp, X)
        log(f"已写入 {xp} ({xp.stat().st_size / 1e6:.1f} MB)")
        del X
        gc.collect()
    else:
        log(f"{xp} 已存在，跳过")

    with open(CACHE_DIR / "feature_names.json", "w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp, ensure_ascii=False, indent=2)
    with open(CACHE_DIR / "class_names.json", "w", encoding="utf-8") as fp:
        json.dump(CLASS_NAMES, fp, ensure_ascii=False, indent=2)

    manifest = {
        "scheme": SCHEME_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(y)),
        "n_features": len(feature_cols),
        "n_classes": len(CLASS_NAMES),
        "dtype": "float32",
        "files": {xp.name: _sha256(xp), yp.name: _sha256(yp)},
    }
    with open(CACHE_DIR / "cache_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    return xp, yp, feature_cols


def load_X(mmap=True):
    return np.load(CACHE_DIR / X_FILE, mmap_mode="r" if mmap else None)


def load_y():
    return np.load(CACHE_DIR / Y_FILE)


def load_feature_names():
    with open(CACHE_DIR / "feature_names.json", encoding="utf-8") as fp:
        return json.load(fp)


def load_class_names():
    with open(CACHE_DIR / "class_names.json", encoding="utf-8") as fp:
        return json.load(fp)


def load_split_arrays(mmap=True):
    """一次性拿到 (X_train, X_test, y_train, y_test)。X 默认 mmap。"""
    X = load_X(mmap=mmap)
    y = load_y()
    tr, te = get_split(y)
    return X[tr], X[te], y[tr], y[te], tr, te


# ---------------------------------------------------------------- 校验
def check_all():
    """硬断言：类数、每类样本数、NaN/inf、行数。失败即抛异常。"""
    ok = True
    X = load_X()
    y = load_y()
    names = load_class_names()

    print("\n=== 断言（12 类口径）===")

    def _chk(cond, msg):
        nonlocal ok
        checks.append((bool(cond), msg))
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        ok = ok and bool(cond)

    checks = []
    _chk(len(names) == EXPECTED_N_CLASSES, f"类数 == {EXPECTED_N_CLASSES}（实际 {len(names)}）")
    _chk(X.shape[0] == len(y), f"X 行数 == y 行数（{X.shape[0]:,} vs {len(y):,}）")
    _chk(X.shape[1] == EXPECTED_N_FEATURES, f"特征数 == {EXPECTED_N_FEATURES}（实际 {X.shape[1]}）")
    _chk(X.shape[0] == EXPECTED_ROWS,
         f"总行数 == 预期 {EXPECTED_ROWS:,}（实际 {X.shape[0]:,}，"
         f"偏差 {X.shape[0] - EXPECTED_ROWS:+,}）")

    counts = np.bincount(y, minlength=len(names))
    _chk((counts > 0).all(), "无空类")
    _chk(counts.min() >= MIN_CLASS_COUNT,
         f"每类 >= {MIN_CLASS_COUNT}（最小类 {names[int(counts.argmin())]} = {counts.min():,}）")
    _chk(int(counts[CLASS_TO_ID["BENIGN"]]) == EXPECTED_BENIGN,
         f"BENIGN == {EXPECTED_BENIGN:,}（实际 {int(counts[CLASS_TO_ID['BENIGN']]):,}，"
         f"证明按标签剔除未碰 BENIGN）")

    n_bad = 0
    for s in range(0, X.shape[0], 200_000):
        blk = np.asarray(X[s:s + 200_000], dtype=np.float32)
        n_bad += int((~np.isfinite(blk)).sum())
    _chk(n_bad == 0, f"无 NaN/inf（发现 {n_bad} 个）")

    tr, te = get_split(y)
    _chk(len(np.intersect1d(tr, te)) == 0, "训练/测试索引无交集")
    _chk(len(tr) + len(te) == len(y), f"训练+测试 == 总数（{len(tr):,}+{len(te):,}）")
    _chk(abs(len(te) / len(y) - TEST_SIZE) < 1e-6, f"测试集比例 == {TEST_SIZE}")

    worst = 0.0
    for i in range(len(names)):
        r_all = counts[i] / counts.sum()
        r_te = (y[te] == i).sum() / len(te)
        worst = max(worst, abs(r_all - r_te))
    _chk(worst < 0.002, f"分层一致（各类测试集占比最大偏差 {worst:.5f}）")

    if not ok:
        raise SystemExit("断言失败，停止。请先查明原因，不要硬跑后续实验。")
    print("  --- 全部断言通过 ---")
    return counts


def print_distribution(y=None, tr=None, te=None):
    if y is None:
        y = load_y()
    if tr is None:
        tr, te = get_split(y)
    names = load_class_names()
    counts = np.bincount(y, minlength=len(names))
    tot = counts.sum()
    print(f"\n=== {len(names)} 类分布（Table 1 素材）===")
    print(f"{'类名':<26}{'样本数':>12}{'占比%':>10}{'训练数':>12}{'测试数':>12}")
    rows = []
    for i, name in enumerate(names):
        n_tr = int((y[tr] == i).sum())
        n_te = int((y[te] == i).sum())
        print(f"{name:<26}{counts[i]:>12,}{counts[i] / tot * 100:>10.4f}"
              f"{n_tr:>12,}{n_te:>12,}")
        rows.append({"id": i, "class": name, "count": int(counts[i]),
                     "ratio_pct": round(counts[i] / tot * 100, 4),
                     "train": n_tr, "test": n_te})
    print(f"{'合计':<26}{tot:>12,}{100.0:>10.4f}{len(tr):>12,}{len(te):>12,}")
    return rows


# ---------------------------------------------------------------- 主流程
def build(cmd=None):
    t0 = time.time()
    df = load_clean()
    df12, rep = apply_class_scheme(df)
    del df
    gc.collect()

    xp, yp, feature_cols = build_arrays(df12, force=True)
    y = df12["y"].to_numpy(dtype=np.int8)

    dist = (df12.groupby("class_name", observed=True).size()
            .reindex(CLASS_NAMES).rename("count").to_frame())
    dist["ratio_pct"] = (dist["count"] / dist["count"].sum() * 100).round(4)
    dist.index.name = "class"
    dist.to_csv(OUT_DIR / "label_distribution_12class.csv")

    scheme = {
        "scheme": SCHEME_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": cmd or " ".join(sys.argv),
        "n_classes": len(CLASS_NAMES),
        "class_order": CLASS_NAMES,
        "label_map": LABEL_MAP,
        "design_rationale": (
            "保留 Web Attack-Brute Force 与 Web Attack-XSS 为两个独立类别：二者的互相混淆"
            "（15 类口径下 XSS 召回仅 29.6%，127 条被判为 Brute Force）正是本文的立项问题，"
            "合并会消解研究动机。仅剔除样本量不足以支撑统计评估的三个类。"
        ),
        "drop": {
            "labels": DROPPED_LABELS,
            "counts": rep["dropped_labels"],
            "reasons": DROP_REASON,
            "mode": "按标签剔除，不按文件剔除；两个源 CSV 中的 BENIGN 行全部保留",
        },
        "keep": {"labels": list(KEEP_REASON), "counts": rep["kept_web_labels"],
                 "reasons": KEEP_REASON},
        "superseded_scheme": {
            "name": "v1-11class",
            "archive": "results/archive/11class/",
            "why_abandoned": "三合一使 macro-F1 从 0.8550 升至 0.9763，但混淆消失等于把待解决问题"
                             "定义掉了，且几乎不给主方法留提升空间",
        },
        "cleaning": {
            "n_raw_rows": 2_830_743,
            "n_after_dedup_minus_inf_nan": rep["n_before"],
            "n_after_class_scheme": rep["n_after"],
            "n_dropped_by_class_scheme": rep["n_dropped"],
            "n_features": len(feature_cols),
            "dtype": "float32",
        },
    }
    with open(OUT_DIR / "class_scheme_12.json", "w", encoding="utf-8") as fp:
        json.dump(scheme, fp, ensure_ascii=False, indent=2)

    del df12
    gc.collect()

    tr, te = get_split(y, force=True)
    write_split_manifest(y, tr, te, cmd=cmd)

    counts = check_all()
    print_distribution(y, tr, te)

    log(f"构建完成，总耗时 {time.time() - t0:.1f}s")
    log(f"缓存: {CACHE_DIR}")
    log(f"划分: {SPLIT_DIR}")
    return counts


def main():
    ap = argparse.ArgumentParser(description="CIC-IDS2017 共用数据管线（12 类）")
    ap.add_argument("--build", action="store_true", help="重建缓存 + 划分 + 报告")
    ap.add_argument("--check", action="store_true", help="只校验已有缓存与划分")
    ap.add_argument("--dist", action="store_true", help="打印 12 类分布表")
    args = ap.parse_args()

    if args.build:
        build()
    elif args.check:
        check_all()
        print_distribution()
    elif args.dist:
        print_distribution()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
