# -*- coding: utf-8 -*-
"""
不平衡处理模块（主方法，docs/method_design_v1.md §3）
====================================================
- FocalLoss                : 多分类 focal loss（§3.2），α 默认 uniform
- build_resampled_arrays   : BENIGN 随机欠采样 + 三类少数类 SMOTE（§3.1），
                              返回重采样数组与 resampling_audit

调用顺序（§3.5 L1/L3，顺序不能反）：
    1. train_test_split(fit_idx, val_idx)          # 先切验证集
    2. StandardScaler.partial_fit(真实 fit_idx)     # scaler 只看真实 fit
    3. transform
    4. build_resampled_arrays(...)                  # 欠采样 -> SMOTE
    5. val / test 永不重采样
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 12 类口径下的病灶组（docs §1.1：Bot / Web Attack-Brute Force / Web Attack-XSS）
LESION_CLASS_IDS = [9, 10, 11]
LESION_CLASS_NAMES = ["Bot", "Web Attack-Brute Force", "Web Attack-XSS"]


# ---------------------------------------------------------------- focal loss
class FocalLoss(nn.Module):
    """多分类 focal loss（logits 输入）。

    FL(p_t) = - alpha_y * (1 - p_t)^gamma * log(p_t),  p_t = softmax(logits)[y]

    alpha 传 None 或 "uniform" 表示全 1（设计推荐，理由见 §3.2：重采样与 α
    都在补偿类频次，叠加会重复补偿）。
    传 tensor 则按类别加权（inv_sqrt 口径由 build_alpha_inv_sqrt 生成，
    且必须基于**重采样前**的原始 fit 频次）。
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma 必须 >= 0")
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is None or (isinstance(alpha, str) and alpha == "uniform"):
            self.register_buffer("alpha", None)
        else:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", a)

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1)                  # (B, C)
        log_pt = log_pt.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_pt.exp()
        loss = -(1.0 - p_t).pow(self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_alpha_inv_sqrt(counts):
    """按**重采样前**的原始 fit 频次算 inv-sqrt α，归一化到均值 1（§3.2）。

    禁止用重采样后的频次——那会让未被过采样的饱和类（如 SSH-Patator）
    拿到比病灶类更高的权重（SSH 2.123 vs XSS 0.873，§3.2 的表）。
    """
    counts = np.asarray(counts, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"存在样本数为 0 的类，无法算 α：{counts}")
    raw = 1.0 / np.sqrt(counts)
    return raw / raw.mean()


def build_alpha_inv(counts):
    """按**重采样前**的原始 fit 频次算 1/n α，归一化到均值 1。

    教科书口径的 class weighting（等价于 sklearn 的 class_weight="balanced"）：
    权重与类频次成反比。用于回答"最简单的先验修正能不能替代重采样"。
    在 CIC-IDS2017 上 BENIGN:XSS ≈ 3219:1，故该口径的权重跨度约 3200 倍。
    """
    counts = np.asarray(counts, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"存在样本数为 0 的类，无法算 α：{counts}")
    raw = 1.0 / counts
    return raw / raw.mean()


# ---------------------------------------------------------------- 重采样
def build_resampled_arrays(X_fit_scaled, y_fit, real_fit_indices,
                           benign_target, minority_target,
                           minority_class_ids=LESION_CLASS_IDS,
                           benign_class_id=0, smote_k=5, seed=42,
                           verbose=True):
    """BENIGN 随机欠采样 + 少数类 SMOTE。输入必须是**已标准化**的 fit 集。

    参数
    ----
    X_fit_scaled : (n_fit, 78) float32  标准化后的 fit 集
    y_fit        : (n_fit,) int         fit 标签
    real_fit_indices : (n_fit,) int     每行对应的**全局行索引**（用于 L5 断言）
    benign_target : int | None          BENIGN 目标行数；None / "full" / >= 现量 则不欠采样
    minority_target : int               每个少数类的目标行数（SMOTE 后）
    minority_class_ids : list[int]      做 SMOTE 的类别
    smote_k : int                       SMOTE k_neighbors
    seed : int                          同时控制欠采样与 SMOTE（§3.5 L4）

    返回 dict：X_res / y_res / real_idx_res / synthetic_mask / audit / asserts
    """
    from imblearn.over_sampling import SMOTE

    y_fit = np.asarray(y_fit).astype(np.int64)
    real_fit_indices = np.asarray(real_fit_indices).astype(np.int64)
    n_classes = int(y_fit.max()) + 1
    counts = np.bincount(y_fit, minlength=n_classes)
    rng = np.random.default_rng(seed)

    # ---------- 1) BENIGN 随机欠采样（只动 fit，§3.1） ----------
    benign_ids = np.flatnonzero(y_fit == benign_class_id)
    n_benign_before = int(len(benign_ids))
    if benign_target is None or benign_target == "full" \
            or int(benign_target) >= n_benign_before:
        keep_benign = benign_ids
        benign_effective = n_benign_before
        benign_undersampled = False
    else:
        keep_benign = np.sort(rng.choice(benign_ids, size=int(benign_target),
                                         replace=False))
        benign_effective = int(len(keep_benign))
        benign_undersampled = True
    other_ids = np.flatnonzero(y_fit != benign_class_id)
    sel = np.sort(np.concatenate([keep_benign, other_ids]))

    X_u = X_fit_scaled[sel]
    y_u = y_fit[sel]
    idx_u = real_fit_indices[sel]
    counts_u = np.bincount(y_u, minlength=n_classes)

    # ---------- 2) 少数类 SMOTE（k=5，只对病灶三类，绝不对 BENIGN） ----------
    minority = sorted(int(c) for c in minority_class_ids)
    counts_before = {int(c): int(counts_u[c]) for c in minority}
    min_count = min(counts_before.values())
    k_eff = min(int(smote_k), min_count - 1)
    if k_eff < 1:
        raise SystemExit(f"少数类样本数不足以做 SMOTE（最小 {min_count}，k 必须 >=1）。"
                         f"冒烟请增大 --sample 或降低 smote_k。")
    sampling_strategy = {c: int(minority_target) for c in minority
                         if counts_u[c] < int(minority_target)}
    n_synth = 0
    if sampling_strategy:
        sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_eff,
                   random_state=seed)
        X_u, y_u = sm.fit_resample(X_u, y_u)
        # SMOTE 追加的合成样本在尾部；真实行保持原有相对顺序
        n_before = int(counts_u.sum())
        n_synth = int(len(X_u) - n_before)
    synthetic_mask = np.zeros(len(X_u), dtype=bool)
    if n_synth:
        synthetic_mask[len(X_u) - n_synth:] = True
    # 真实行保留其全局索引；合成行记 -1（不参与 L5 的 ⊆ fit_idx 检验）
    real_idx_res = (np.concatenate([idx_u, np.full(n_synth, -1, dtype=idx_u.dtype)])
                    if n_synth else idx_u)
    y_res = y_u.astype(np.int64)
    counts_after = np.bincount(y_res, minlength=n_classes)

    audit = {
        "seed": int(seed),
        "benign_class_id": int(benign_class_id),
        "benign_before": n_benign_before,
        "benign_after": benign_effective,
        "benign_undersampled": benign_undersampled,
        "benign_target": (None if benign_target in (None, "full")
                          else int(benign_target)),
        "minority_class_ids": minority,
        "smote_k_effective": int(k_eff),
        "minority": {str(c): {
            "before": counts_before[c],
            "after": int(counts_after[c]),
            "synthetic": int(counts_after[c] - counts_before[c]),
        } for c in minority},
        "n_fit_before": int(len(y_fit)),
        "n_fit_after": int(len(y_res)),
        "n_synthetic": n_synth,
        "n_real_used": int(len(y_res) - n_synth),
        "counts_after": {int(c): int(counts_after[c]) for c in range(n_classes)},
        "smote_sampling_strategy": {str(k): int(v)
                                    for k, v in sampling_strategy.items()},
    }

    # ---------- L5 断言：真实行 ⊆ fit_idx，数量自洽 ----------
    asserts = {}
    asserts["L5_count_selfconsistent"] = bool(
        audit["n_real_used"] + audit["n_synthetic"] == len(y_res))
    asserts["L5_real_subset_of_fit"] = bool(
        np.isin(real_idx_res[~synthetic_mask], real_fit_indices).all())
    if not all(asserts.values()):
        raise SystemExit(f"重采样 L5 断言失败：{asserts}，停止，不进入训练。")

    if verbose:
        print(f"[重采样] BENIGN {n_benign_before:,} -> {benign_effective:,}"
              f"（{'欠采样' if benign_undersampled else '未动'}）；"
              f"SMOTE 合成 {n_synth:,} 行；"
              f"fit 总量 {audit['n_fit_before']:,} -> {audit['n_fit_after']:,}；"
              f"k_eff={k_eff}", flush=True)

    return {
        "X_res": np.ascontiguousarray(X_u, dtype=np.float32),
        "y_res": y_res,
        "real_idx_res": real_idx_res,
        "synthetic_mask": synthetic_mask,
        "audit": audit,
        "asserts": asserts,
    }
