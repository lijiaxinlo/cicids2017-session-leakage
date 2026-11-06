# -*- coding: utf-8 -*-
"""
CIC-IDS2017 随机森林基线实验
================================
用途：论文 Table 2「传统机器学习」对照组的第一行，同时验证整条数据管线是否通畅。

流程：
    加载 8 个 CSV → 清洗（去重 / 去 inf / 去 NaN / 列名与标签规范化）
    → 分层划分训练测试集 → 训练 RandomForest → 输出 6 项指标 + 混淆矩阵

用法：
    python baseline_rf.py                    # 全量训练
    python baseline_rf.py --sample 300000    # 分层采样加速（首次验证用）
    python baseline_rf.py --multi            # 额外输出多分类结果

输出：
    控制台指标表
    results/rf_baseline_metrics.json     指标汇总
    results/rf_confusion_matrix.csv      二分类混淆矩阵
    results/label_distribution.csv       原始标签分布（论文 Table 1 素材）
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3

# 标签里的破折号需要规范化。
# 注意：本镜像的 CSV 在制作时把 en dash 破坏成了 U+FFFD（替换字符，
# 原始字节为 \xef\xbf\xbd），信息已丢失、换编码也恢复不了，只能按 dash 处理。
DASH_MAP = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\ufffd": "-"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- 数据加载
def load_raw():
    """读取全部 CSV 并合并。列名与标签做规范化。"""
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"未在 {RAW_DIR} 找到 CSV，请先下载 CIC-IDS2017。")

    log(f"发现 {len(files)} 个 CSV 文件")
    frames = []
    for f in files:
        t0 = time.time()
        df = pd.read_csv(f, low_memory=False)
        # 列名普遍带前导空格（如 ' Label'），必须 strip，否则取不到标签列
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
        log(f"  读入 {f.name}: {df.shape[0]:,} 行 × {df.shape[1]} 列 "
            f"({time.time() - t0:.1f}s)")

    df = pd.concat(frames, ignore_index=True)
    log(f"合并完成: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    return df


def clean(df):
    """去重、去无穷值、去缺失、标签规范化，并把数值列降为 float32 省内存。"""
    if "Label" not in df.columns:
        raise SystemExit(f"未找到 Label 列。现有列示例: {list(df.columns[:8])}")

    n0 = len(df)

    # 1) 去重（CIC-IDS2017 存在大量完全重复行）
    df = df.drop_duplicates()
    log(f"去重: {n0:,} -> {len(df):,}（移除 {n0 - len(df):,} 行）")

    # 2) 无穷值 -> NaN -> 删除
    n1 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    log(f"清理 inf/NaN: {n1:,} -> {len(df):,}（移除 {n1 - len(df):,} 行）")

    # 3) 标签规范化
    df["Label"] = (
        df["Label"].astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    # en dash / em dash 统一成 hyphen，避免同一类攻击被拆成两个标签
    for bad, good in DASH_MAP.items():
        df["Label"] = df["Label"].str.replace(bad, good, regex=False)
    # 'Web Attack - Brute Force' 等写法统一
    df["Label"] = df["Label"].str.replace("Web Attack - ", "Web Attack-", regex=False)

    # 4) 数值列降精度（float64 -> float32，内存减半）
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float32)
    log(f"数值列已降为 float32（{len(num_cols)} 列）")

    return df


# ---------------------------------------------------------------- 指标计算
def compute_metrics(y_true, y_pred, y_prob=None, labels=None):
    """计算论文需要的 6 项指标。macro 平均用于多分类。"""
    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    # FPR：逐类算 FP/(FP+TN) 后取平均（二分类即该类的 FPR）
    cm = confusion_matrix(y_true, y_pred)
    fprs = []
    for i in range(len(cm)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    fpr = float(np.mean(fprs))

    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                    average="macro")
        except Exception:
            auc = None

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "auc": None if auc is None else float(auc),
    }


def print_metrics(title, m):
    auc_str = "N/A" if m["auc"] is None else f"{m['auc']:.4f}"
    print(f"\n{'-' * 46}")
    print(f"  {title}")
    print(f"{'-' * 46}")
    print(f"  Accuracy   : {m['accuracy']:.4f}")
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  F1         : {m['f1']:.4f}")
    print(f"  FPR        : {m['fpr']:.4f}")
    print(f"  AUC        : {auc_str}")
    print(f"{'-' * 46}")


# ---------------------------------------------------------------- 主流程
def run(df, sample=None, do_multi=False):
    t_start = time.time()

    # 标签分布（论文 Table 1 素材，同时用来观察不平衡程度）
    dist = df["Label"].value_counts()
    dist.to_csv(OUT_DIR / "label_distribution.csv", header=["count"])
    print("\n=== 原始标签分布 ===")
    total = dist.sum()
    for name, cnt in dist.items():
        print(f"  {name:<32} {cnt:>10,}  ({cnt / total * 100:6.3f}%)")

    # 二分类标签：BENIGN vs ATTACK
    df = df.copy()
    df["binary_label"] = (df["Label"] != "BENIGN").astype(np.int8)

    feature_cols = [c for c in df.columns if c not in ("Label", "binary_label")]
    log(f"特征维度: {len(feature_cols)}")

    # 可选：分层采样加速
    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample,
                                 stratify=df["binary_label"],
                                 random_state=RANDOM_STATE)
        log(f"分层采样至 {len(df):,} 行")

    results = {}

    # ---------- 二分类 ----------
    X = df[feature_cols].values
    y = df["binary_label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log(f"二分类：训练 {len(X_tr):,} / 测试 {len(X_te):,}")

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=None, n_jobs=-1,
        random_state=RANDOM_STATE, class_weight="balanced_subsample")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    log(f"训练完成，耗时 {time.time() - t0:.1f}s")

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)

    m_bin = compute_metrics(y_te, y_pred, y_prob)
    print_metrics("RandomForest 二分类（BENIGN vs ATTACK）", m_bin)
    results["binary"] = m_bin

    cm = confusion_matrix(y_te, y_pred)
    pd.DataFrame(cm, index=["true_BENIGN", "true_ATTACK"],
                 columns=["pred_BENIGN", "pred_ATTACK"]
                 ).to_csv(OUT_DIR / "rf_confusion_matrix.csv")

    # ---------- 多分类（可选） ----------
    if do_multi:
        from sklearn.preprocessing import LabelEncoder
        # 剔除样本数过少的类别：stratify 要求每类至少 2 个样本
        vc = df["Label"].value_counts()
        rare = vc[vc < 5].index.tolist()
        if rare:
            log(f"多分类：剔除稀有类 {rare}（样本数 < 5，不足以分层划分）")
            df_m = df[~df["Label"].isin(rare)].copy()
        else:
            df_m = df
        X = df_m[feature_cols].values
        le = LabelEncoder()
        y_multi = le.fit_transform(df_m["Label"].values)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
            X, y_multi, test_size=TEST_SIZE, stratify=y_multi,
            random_state=RANDOM_STATE)
        clf2 = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced_subsample")
        log("多分类训练中……")
        clf2.fit(X_tr2, y_tr2)
        y_pred2 = clf2.predict(X_te2)
        m_multi = compute_metrics(y_te2, y_pred2)
        print_metrics("RandomForest 多分类（全部攻击类型）", m_multi)
        results["multi"] = m_multi

        pd.DataFrame(
            confusion_matrix(y_te2, y_pred2),
            index=[f"true_{c}" for c in le.classes_],
            columns=[f"pred_{c}" for c in le.classes_]
        ).to_csv(OUT_DIR / "rf_confusion_matrix_multi.csv")

    # ---------- 落盘 ----------
    results["meta"] = {
        "n_samples_used": int(len(df)),
        "n_features": len(feature_cols),
        "n_estimators": 100,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "sample": sample,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    with open(OUT_DIR / "rf_baseline_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    log(f"全部完成，总耗时 {time.time() - t_start:.1f}s")
    log(f"结果已写入 {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description="CIC-IDS2017 随机森林基线")
    ap.add_argument("--sample", type=int, default=None,
                    help="分层采样行数，用于快速验证（默认全量）")
    ap.add_argument("--multi", action="store_true", help="额外输出多分类结果")
    args = ap.parse_args()

    df = load_raw()
    df = clean(df)
    run(df, sample=args.sample, do_multi=args.multi)


if __name__ == "__main__":
    main()
