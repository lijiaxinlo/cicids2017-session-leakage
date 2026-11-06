# -*- coding: utf-8 -*-
"""
主方法训练脚本（阶段二 · Phase 2a）
==================================
CNN-BiLSTM-Attention + 混合采样（BENIGN 欠采样 + 三类 SMOTE）+ focal loss。

与 baseline 的关系（docs/method_design_v1.md §4.2.1 零调参口径）：
    lr=1e-3 / wd=1e-4 / batch=256 / dropout=0.3 / epochs 上限 30 /
    patience=5（val macro-F1）/ GroupNorm / 输入 (B,1,78) —— 与三个 baseline 完全一致。
    仅有的方法特有旋钮：focal γ、BENIGN 目标量（§4.2.2，6 组，val 上选）。

数据泄漏硬约束（§3.5，全部写成 assert，任一失败立即终止）：
    L1 重采样只在 fit_idx 切出 val 之后做（顺序写死在本脚本流程里）
    L2 val / test 永不重采样（resample 前后对 val/test 标签做校验和比对）
    L3 StandardScaler 只在真实 fit_idx 上 fit，先于 SMOTE
    L4 SMOTE random_state = seed
    L5 真实行 ⊆ fit_idx，n_real + n_synthetic == n_res（在 imbalance.py 内断言）
    L6 测试集只在训练结束后评估一次

用法
----
    # 冒烟（30 万训练池 / 2 epoch / γ=2 / BENIGN 目标 30 万）
    python src/train_main.py --seed 42 --epochs 2 --sample 300000

    # 全量正式（E4）
    python src/train_main.py --seed 42 --epochs 30 --patience 5

    # 消融臂（S0-L0 / S1-L0）
    python src/train_main.py --loss ce --sampler none    # S0-L0（= 深度 baseline 等价）
    python src/train_main.py --loss ce                   # S1-L0
"""

import argparse
import gc
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import data_pipeline as dp
import eval_utils as eu
import imbalance as im
import models as M
import models_adv as MA

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"
RUN_DIR = OUT_DIR / "runs"
CACHE_DIR = OUT_DIR / "cache"
PROBA_DIR = OUT_DIR / "proba"
RUN_DIR.mkdir(parents=True, exist_ok=True)
PROBA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK = 200_000
DEFAULT_GAMMA = 2.0
DEFAULT_BENIGN_TARGET = 300_000
# v4.3 门禁：唯一中止条件 = 3-epoch 滑动平均超阈（benign FPR > 0.05，macro-OVR > 0.02）
FPR_DIVERGED_MA = 0.05
FPR_DIVERGED_MA_MACRO = 0.02
DEFAULT_MINORITY_TARGET = 12_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_info():
    import torch as _t
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "imbalanced_learn": __import__("imblearn").__version__,
        "torch": _t.__version__,
        "cuda_available": _t.cuda.is_available(),
        "device": _t.cuda.get_device_name(0) if _t.cuda.is_available() else "cpu",
    }


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler_chunked(X, idx, scaler):
    for s in range(0, len(idx), CHUNK):
        blk = np.asarray(X[idx[s:s + CHUNK]], dtype=np.float64)
        scaler.partial_fit(blk)
    return scaler


def transform_chunked(X, idx, scaler, dtype=np.float32):
    out = np.empty((len(idx), X.shape[1]), dtype=dtype)
    for s in range(0, len(idx), CHUNK):
        blk = np.asarray(X[idx[s:s + CHUNK]], dtype=np.float64)
        out[s:s + CHUNK] = scaler.transform(blk).astype(dtype)
    return out


def make_loader(Xa, ya, batch_size, shuffle):
    x = torch.from_numpy(np.ascontiguousarray(Xa).reshape(-1, 1, Xa.shape[1]))
    y = torch.from_numpy(np.ascontiguousarray(ya).astype(np.int64))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


def benign_fpr(y_true, y_pred, benign_id=0):
    """良性误报率 P(pred != BENIGN | y = BENIGN)（IDS 部署口径，§3.3.1）。"""
    m = y_true == benign_id
    n = int(m.sum())
    if n == 0:
        return 0.0
    return float((y_pred[m] != benign_id).sum() / n)


@torch.no_grad()
def evaluate(model, loader, device, collect_pred=False, collect_proba=False):
    model.eval()
    total_loss, n = 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    preds, trues, probs = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = model(xb)
        total_loss += float(crit(out, yb))
        n += yb.numel()
        preds.append(out.argmax(1).cpu().numpy())
        if collect_proba:
            probs.append(torch.softmax(out, dim=1).float().cpu().numpy())
        if collect_pred:
            trues.append(yb.cpu().numpy())
    res = {"loss": total_loss / max(n, 1), "y_pred": np.concatenate(preds)}
    if collect_pred:
        res["y_true"] = np.concatenate(trues)
    if collect_proba:
        res["y_prob"] = np.concatenate(probs)
    return res


def main():
    ap = argparse.ArgumentParser(description="主方法训练（CNN-BiLSTM-Attention + 不平衡处理）")
    ap.add_argument("--model", default="cnn_bilstm_attn",
                    choices=list(MA.ADV_MODELS) + list(M.MODELS),
                    help="cnn_bilstm_attn = 主方法；cnn/lstm/cnn_lstm = 家族对照（走 models.py）")
    ap.add_argument("--split", default="random",
                    choices=["random", "temporal", "grouped"],
                    help="random=原固定划分；temporal=类内时序 purged；grouped=同组同尺寸但组内随机")
    ap.add_argument("--pool", default="attn", choices=["attn", "mean"],
                    help="池化方式：attn=主方法；mean=消融 A1（单变量，其余全不动）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--sample", type=int, default=None,
                    help="训练池分层抽样行数（冒烟用），测试集保持完整")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--tag", type=str, default=None)
    # ---- 方法特有旋钮（§4.2.2）----
    ap.add_argument("--loss", default="focal", choices=["focal", "ce"],
                    help="focal = 主方法；ce = 消融 S?L0 臂")
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                    help="focal γ（仅 --loss focal 时生效）")
    ap.add_argument("--alpha-mode", default="uniform",
                    choices=["uniform", "inv_sqrt", "inv"],
                    help="α 口径；inv_sqrt/inv 必须基于重采样前的原始 fit 频次（§3.2）")
    ap.add_argument("--sampler", default="smote", choices=["smote", "none"],
                    help="smote = 混合采样（主方法）；none = 不重采样（消融 S0 臂）")
    ap.add_argument("--benign-target", default=str(DEFAULT_BENIGN_TARGET),
                    help="BENIGN 欠采样目标行数，或 'full'（不欠采样）")
    ap.add_argument("--minority-target", type=int, default=DEFAULT_MINORITY_TARGET,
                    help="三类少数类 SMOTE 目标行数（各类相同，保持 BF/XSS 对等）")
    ap.add_argument("--smote-k", type=int, default=5)
    ap.add_argument("--tier-ratio", type=float, default=None,
                    help="（E7/D11）逐类不平衡上限扫描：BENIGN 不动，每个攻击类欠采样至"
                         " <= floor(n_benign_fit/R)。默认 None 关闭，不影响任何既有行为")
    # ---- FPR 门禁（§3.3.2：连续 2 epoch 超硬上限×2 即中断）----
    ap.add_argument("--fpr-abort-macro", type=float, default=0.01,
                    help="val macro-OVR FPR 中断阈值（= 硬上限 0.005 × 2）")
    ap.add_argument("--fpr-abort-benign", type=float, default=0.016,
                    help="val 良性误报率中断阈值（= 硬上限 0.008 × 2）")
    ap.add_argument("--diverged-ma-benign", type=float, default=FPR_DIVERGED_MA,
                    help="3-epoch 滑动平均 benign FPR 的中止阈值；P2（时序）协议下"
                         "常态即 0.08–0.14，需显式抬高才能跑完")
    ap.add_argument("--diverged-ma-macro", type=float, default=FPR_DIVERGED_MA_MACRO,
                    help="3-epoch 滑动平均 macro-OVR FPR 的中止阈值")
    ap.add_argument("--save-proba", action="store_true",
                    help="保存 val/test 逐类概率 float32（λ / w_c 校准用；val 可搜，test 在 λ 定前不读）")
    ap.add_argument("--save-ckpt", action="store_true")
    args = ap.parse_args()

    # 解析 benign_target
    if str(args.benign_target).lower() in ("full", "none"):
        benign_target = "full"
    else:
        benign_target = int(args.benign_target)

    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件名：默认配置（focal+smote+γ2+B30万+uniform）产出干净名字，非默认才加后缀
    # v4.3 修复双拼 bug：--tag 与自动 γ/benign 后缀只拼一次（tag 优先，取代自动后缀）
    auto_tag = ""
    if args.loss == "focal" and args.gamma != DEFAULT_GAMMA:
        auto_tag += f"g{args.gamma:g}"
    if benign_target != DEFAULT_BENIGN_TARGET:
        auto_tag += ("_" if auto_tag else "") + \
            ("full" if benign_target == "full" else str(benign_target))
    tag = args.tag if args.tag else auto_tag
    suffix = ""
    if args.sample:
        suffix += f"_sample{args.sample}"
    if args.loss != "focal":
        suffix += f"_{args.loss}"
    if args.sampler != "smote":
        suffix += f"_{args.sampler}"
    if args.loss == "focal" and args.alpha_mode != "uniform":
        suffix += f"_{args.alpha_mode}"
    if args.minority_target != DEFAULT_MINORITY_TARGET:
        suffix += f"_m{args.minority_target}"
    if args.tier_ratio:
        suffix += f"_tier{args.tier_ratio:g}"
    if args.pool != "attn":
        suffix += f"_pool{args.pool}"
    if args.model != "cnn_bilstm_attn":
        suffix += f"_{args.model}"
    if tag:
        suffix += f"_{tag}"
    name = f"deep_{args.model}_seed{args.seed}_12class{suffix}"

    cmd = " ".join(["python", "src/train_main.py", "--model", args.model,
                    "--seed", str(args.seed), "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                    "--loss", args.loss, "--sampler", args.sampler,
                    "--gamma", str(args.gamma), "--alpha-mode", args.alpha_mode,
                    "--benign-target", str(args.benign_target),
                    "--minority-target", str(args.minority_target),
                    "--patience", str(args.patience)]
                   + (["--sample", str(args.sample)] if args.sample else [])
                   + (["--tag", args.tag] if args.tag else []))

    t_start = time.time()
    class_names = dp.load_class_names()
    n_classes = len(class_names)
    y = dp.load_y()
    if args.split in ("temporal", "grouped"):
        # 返修意见 1：类内时序 purged 划分（train / val / test 时间上不相邻）
        # grouped = 同组同尺寸但组内随机，用于把"时间漂移"与"类别构成"拆开
        tsd = ROOT / "results" / ("temporal" if args.split == "temporal" else "temporal_grouped")
        tr_all = np.load(tsd / "train_idx.npy")
        val_fixed = np.load(tsd / "val_idx.npy")
        te = np.load(tsd / "test_idx.npy")
        log(f"[split={args.split}] 训练 {len(tr_all):,} / 验证 {len(val_fixed):,} / "
            f"测试 {len(te):,}")
    else:
        tr_all, te = dp.get_split(y)          # 固定划分，只读
        val_fixed = None
    X = dp.load_X(mmap=True)
    log(f"模型={args.model} seed={args.seed} device={device} loss={args.loss} "
        f"sampler={args.sampler} γ={args.gamma} α={args.alpha_mode} "
        f"benign_target={benign_target} minority_target={args.minority_target}")

    # ---------- 训练池可选分层抽样（冒烟；测试集保持完整） ----------
    if args.sample and args.sample < len(tr_all):
        tr_sel, _ = train_test_split(tr_all, train_size=args.sample,
                                     stratify=y[tr_all], random_state=args.seed)
        tr_sel = np.sort(tr_sel)
        log(f"训练池分层抽样至 {len(tr_sel):,} 行")
    else:
        tr_sel = tr_all

    # ---------- L1 第一步：先切验证集（重采样必须在切分之后） ----------
    if val_fixed is not None:
        fit_idx, val_idx = np.sort(tr_all), np.sort(val_fixed)
    else:
        fit_idx, val_idx = train_test_split(tr_sel, test_size=args.val_frac,
                                            stratify=y[tr_sel], random_state=args.seed)
        fit_idx, val_idx = np.sort(fit_idx), np.sort(val_idx)

    # ---------- E7（v4.4 / 决策 D11）：逐类不平衡上限扫描 ----------
    # 规则：BENIGN 全量不动；每个攻击类独立欠采样至 <= floor(n_benign_fit / R)，
    # 本就低于上限的类保持原样；验证集/测试集一行不动。
    # 动机：自然比例下 BENIGN 已是 4.9:1 多数类，欠采样 BENIGN 只会让数据更平衡，
    # 无法构造"更极端"的不平衡；只有压缩攻击类才能单调加剧逐类稀缺性。
    tier_info = None
    if args.tier_ratio:
        benign_id = class_names.index("BENIGN")
        n_benign = int((y[fit_idx] == benign_id).sum())
        cap = max(1, int(n_benign // args.tier_ratio))
        rng = np.random.default_rng(args.seed)
        keep, per_class = [], {}
        n_before_total = int(len(fit_idx))
        for c in range(n_classes):
            idx_c = np.sort(fit_idx[y[fit_idx] == c])
            if c == benign_id or len(idx_c) <= cap:
                keep.append(idx_c)
                per_class[class_names[c]] = {"n_before": int(len(idx_c)),
                                             "n_after": int(len(idx_c))}
            else:
                sel = rng.choice(len(idx_c), size=cap, replace=False)
                keep.append(np.sort(idx_c[sel]))
                per_class[class_names[c]] = {"n_before": int(len(idx_c)),
                                             "n_after": int(cap)}
        fit_idx = np.sort(np.concatenate(keep))
        tier_info = {
            "tier_ratio": args.tier_ratio,
            "benign_fit": n_benign,
            "attack_cap_per_class": cap,
            "n_fit_before": n_before_total,
            "n_fit_after": int(len(fit_idx)),
            "per_class": per_class,
            "rule": ("BENIGN 全量不动；每个攻击类欠采样至 <= floor(n_benign_fit/R)，"
                     "低于上限的类保持原样；val/test 一行不动；"
                     "欠采样在 val 切分之后、SMOTE 之前（L1 顺序不变）"),
            "subsample_rng": f"np.random.default_rng(seed={args.seed})，按类别升序依次抽样",
        }
        log(f"[E7 逐类上限] 1:{args.tier_ratio:g} cap={cap:,}/类 | "
            f"fit {n_before_total:,} -> {len(fit_idx):,} | "
            f"BENIGN {n_benign:,} 不动")

    # ---------- 泄漏断言（§3.5 L1/L2/L6 前置部分） ----------
    pool = np.zeros(len(y), dtype=bool)
    pool[tr_all] = True
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[te] = True
    leakage = {
        "val_in_test": int(test_mask[val_idx].sum()),
        "fit_in_test": int(test_mask[fit_idx].sum()),
        "val_outside_train_pool": int((~pool[val_idx]).sum()),
        "fit_val_overlap": int(len(np.intersect1d(fit_idx, val_idx))),
    }
    assert leakage["val_in_test"] == 0, "验证集落入测试集，数据泄漏！"
    assert leakage["fit_in_test"] == 0, "训练集落入测试集，数据泄漏！"
    assert leakage["fit_val_overlap"] == 0, "训练/验证有交集！"
    if args.split in ("temporal", "grouped"):
        # 时序协议下 val 是独立切出的一段，本就不在 train 池内 —— 断言改为互斥即可
        assert leakage["val_outside_train_pool"] == len(val_idx), \
            "时序协议下 val 必须完全独立于 train"
    else:
        assert leakage["val_outside_train_pool"] == 0, "验证集不在训练池内！"
        if not args.sample:  # L2：全量时行数必须与固定划分吻合
            assert len(val_idx) == 176_452, len(val_idx)
            assert len(te) == 756_219, len(te)
    y_fit_raw, y_val_raw = y[fit_idx], y[val_idx]
    y_te_raw = y[te]
    val_counts_before = np.bincount(y_val_raw, minlength=n_classes).tolist()
    test_counts_before = np.bincount(y_te_raw, minlength=n_classes).tolist()
    log(f"训练 {len(fit_idx):,} / 验证 {len(val_idx):,} / 测试 {len(te):,}")
    log(f"[泄漏检查·切分] val∩test=0, fit∩test=0, val⊂train_pool=True, fit∩val=0")

    # ---------- L3 第二步：StandardScaler 只在真实 fit_idx 上 fit（先于 SMOTE） ----------
    scaler = StandardScaler()
    t0 = time.time()
    fit_scaler_chunked(X, fit_idx, scaler)
    scaler_path = CACHE_DIR / (f"scaler_main_sample{args.sample}.npz" if args.sample
                               else "scaler_main.npz")
    np.savez(scaler_path, mean=scaler.mean_.astype(np.float64),
             scale=scaler.scale_.astype(np.float64),
             n_samples_fit=np.array([len(fit_idx)]))
    log(f"StandardScaler 仅在真实 fit 集 fit（{len(fit_idx):,} 行），"
        f"{time.time() - t0:.1f}s -> {scaler_path.name}")

    X_fit = transform_chunked(X, fit_idx, scaler)
    X_val = transform_chunked(X, val_idx, scaler)
    X_te = transform_chunked(X, te, scaler)
    y_fit, y_val, y_te = (y_fit_raw.astype(np.int64), y_val_raw.astype(np.int64),
                          y_te_raw.astype(np.int64))

    # ---------- L1 第二步：重采样（只在 fit 上；val/test 不碰） ----------
    if args.sampler == "smote":
        real_fit_indices = np.asarray(fit_idx).astype(np.int64)
        res = im.build_resampled_arrays(
            X_fit, y_fit, real_fit_indices,
            benign_target=benign_target,
            minority_target=args.minority_target,
            minority_class_ids=im.LESION_CLASS_IDS,
            benign_class_id=0, smote_k=args.smote_k, seed=args.seed)
        X_fit_used, y_fit_used = res["X_res"], res["y_res"]
        resampling_audit = res["audit"]
        resampling_audit["asserts"] = res["asserts"]
        # L2：resample 之后 val/test 的逐类分布必须与之前一致
        val_counts_after = np.bincount(y_val, minlength=n_classes).tolist()
        test_counts_after = np.bincount(y_te, minlength=n_classes).tolist()
        resampling_audit["val_untouched"] = bool(val_counts_after == val_counts_before)
        resampling_audit["test_untouched"] = bool(test_counts_after == test_counts_before)
        assert resampling_audit["val_untouched"], "val 分布在重采样后改变！"
        assert resampling_audit["test_untouched"], "test 分布在重采样后改变！"
        resampling_audit["leakage_assert_passed"] = True
        log(f"[泄漏检查·重采样] val/test 分布逐类比对一致，真实行 ⊆ fit_idx，"
            f"n_real={resampling_audit['n_real_used']:,} + "
            f"n_synth={resampling_audit['n_synthetic']:,} = {len(y_fit_used):,}")
    else:
        X_fit_used, y_fit_used = X_fit, y_fit
        resampling_audit = {"sampler": "none", "n_fit_after": int(len(y_fit)),
                            "leakage_assert_passed": True}
    gc.collect()

    # ---------- 模型与损失 ----------
    if args.model in MA.ADV_MODELS:
        model = MA.build_adv_model(args.model, n_features=X.shape[1],
                                   n_classes=n_classes, dropout=args.dropout,
                                   pool=args.pool).to(device)
    else:
        # 家族对照：cnn / lstm / cnn_lstm 走 models.py，norm 固定 gn 与全文口径一致
        model = M.build_model(args.model, n_features=X.shape[1],
                              n_classes=n_classes, dropout=args.dropout,
                              norm="gn").to(device)
    nparam = (MA.count_params(model) if args.model in MA.ADV_MODELS
              else M.count_params(model))
    log(f"模型参数量: {nparam}")

    if args.loss == "focal":
        if args.alpha_mode == "uniform":
            alpha = None
        else:
            # inv_sqrt / inv 口径必须用**重采样前**的原始 fit 频次（§3.2）
            counts_orig = np.bincount(y_fit, minlength=n_classes)
            _builder = (im.build_alpha_inv_sqrt if args.alpha_mode == "inv_sqrt"
                        else im.build_alpha_inv)
            alpha = torch.tensor(_builder(counts_orig),
                                 dtype=torch.float32, device=device)
            log(f"α({args.alpha_mode}, 原始频次): "
                f"{ {c: round(float(a), 3) for c, a in zip(range(n_classes), alpha.tolist())} }")
        crit = im.FocalLoss(gamma=args.gamma, alpha=alpha)
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    dl_fit = make_loader(X_fit_used, y_fit_used, args.batch_size, shuffle=True)
    dl_val = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    dl_te = make_loader(X_te, y_te, args.batch_size, shuffle=False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # ---------- 训练循环 ----------
    history, train_secs, timing_anomalies = [], [], []
    best_f1, best_epoch, best_state = -1.0, -1, None
    bad_epochs = 0
    benign_fpr_hist, macro_fpr_hist, aborted_reason = [], [], None

    for ep in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        run_loss, seen = 0.0, 0
        for xb, yb in dl_fit:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            run_loss += float(loss) * yb.numel()
            seen += yb.numel()
        train_loss = run_loss / max(seen, 1)
        train_sec = time.time() - t_ep

        val = evaluate(model, dl_val, device)
        vm = eu.compute_metrics(y_val, val["y_pred"], class_names=class_names,
                                labels=list(range(n_classes)))
        vb_fpr = benign_fpr(y_val, val["y_pred"])
        peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if device.type == "cuda" else 0.0)

        # 耗时异常检测
        train_secs.append(train_sec)
        med_prev = float(np.median(train_secs[:-1])) if len(train_secs) > 2 else None
        anomaly = bool(med_prev is not None and train_sec > 1.5 * med_prev)
        if anomaly:
            timing_anomalies.append({"epoch": ep, "train_sec": round(train_sec, 1),
                                     "median_prev": round(med_prev, 1),
                                     "ratio": round(train_sec / med_prev, 2)})

        # v4.3 门禁：逐 epoch 审计 flag（macro>0.01 或 benign>0.016），纯记录、不中止
        gate_flag = bool(vm["fpr"] > args.fpr_abort_macro
                         or vb_fpr > args.fpr_abort_benign)

        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val["loss"], 6),
            "gap_val_minus_train": round(val["loss"] - train_loss, 6),
            "val_macro_f1": round(vm["macro_f1"], 6),
            "val_macro_recall": round(vm["macro_recall"], 6),
            "val_fpr": round(vm["fpr"], 6),
            "val_benign_fpr": round(vb_fpr, 6),
            "gate_flag": gate_flag,
            "train_sec": round(train_sec, 1),
            "val_sec": round(time.time() - t_ep - train_sec, 1),
            "peak_gpu_mb": round(peak_mb, 1),
            "timing_anomaly": anomaly,
        })
        log(f"epoch {ep}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | val_macroF1 {vm['macro_f1']:.4f} | "
            f"val_fpr {vm['fpr']:.5f} | val_benign_fpr {vb_fpr:.5f} | "
            f"train {train_sec:.1f}s | peak {peak_mb:.0f} MB"
            + ("  <<< [耗时异常]" if anomaly else ""))

        # ---------- FPR 门禁 v4.3（§3.3.2）：仅保留病理性中止 ----------
        # 3-epoch 滑动平均（从 epoch 3 起，窗口含当前 epoch）：
        #   val_benign_fpr > 0.05 或 val_fpr(macro) > 0.02 -> aborted_reason="fpr_diverged"
        # gate_flag 越限只审计不中止。
        benign_fpr_hist.append(vb_fpr)
        macro_fpr_hist.append(vm["fpr"])
        if len(benign_fpr_hist) >= 3:
            ma_benign = sum(benign_fpr_hist[-3:]) / 3.0
            ma_macro = sum(macro_fpr_hist[-3:]) / 3.0
            # D12（E7 压力档）：--tier-ratio 下 macro FPR 只审计不中止。
            # 理由：macro-OVR FPR 的 0.02 病理性阈值是在自然比例训练动态下调定的；
            # E7 是刻意构造的极端稀缺压力测试，macro FPR 退化本身就是被测量的结果，
            # 用它中止训练会恰好抹掉待测信号。部署口径 benign FPR 的中止条件保留。
            macro_diverged = (args.tier_ratio is None
                              and ma_macro > args.diverged_ma_macro)
            if ma_benign > args.diverged_ma_benign or macro_diverged:
                aborted_reason = "fpr_diverged"
                log(f"  [FPR 门禁] 病理性发散：3-epoch 滑动平均 benign FPR "
                    f"{ma_benign:.5f} > {args.diverged_ma_benign} 或 macro FPR "
                    f"{ma_macro:.5f} > {args.diverged_ma_macro}（窗口 benign "
                    f"{[round(x, 5) for x in benign_fpr_hist[-3:]]} / macro "
                    f"{[round(x, 5) for x in macro_fpr_hist[-3:]]}）—— "
                    f"训练中断，不产出测试集结果")
                break
        elif gate_flag:
            log(f"  [FPR 门禁] epoch {ep} 越限（仅审计）：macro {vm['fpr']:.5f} / "
                f"benign {vb_fpr:.5f}")

        # 早停 + best 回滚（val macro-F1）
        if vm["macro_f1"] > best_f1 + 1e-6:
            best_f1, best_epoch = vm["macro_f1"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_per_class = {c: vm["per_class"][c] for c in class_names}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"早停：验证集 macro-F1 连续 {args.patience} 轮无提升")
                break

    early_stopped = (aborted_reason is None and len(history) < args.epochs)

    # ---------- L6：测试集只评估一次（aborted 时不产出测试结果） ----------
    rec = None
    if aborted_reason is None:
        if best_state is not None:
            model.load_state_dict(best_state)
        t_te = time.time()
        test = evaluate(model, dl_te, device, collect_pred=True,
                        collect_proba=args.save_proba)
        test_sec = time.time() - t_te
        tm = eu.compute_metrics(test["y_true"], test["y_pred"],
                                class_names=class_names,
                                labels=list(range(n_classes)))
        te_bfpr = benign_fpr(test["y_true"], test["y_pred"])
        # 双主指标（team-lead 定稿口径）：① 12 类 macro-F1 ② 病灶组三类 F1 之和
        lesion_ids = im.LESION_CLASS_IDS
        lesion_sum = float(sum(tm["per_class"][class_names[c]]["f1"] for c in lesion_ids))
        eu.print_report(f"{args.model} · seed={args.seed} · TEST (best epoch {best_epoch})",
                        tm, class_names)
        print(f"  双主指标: 12类macro-F1={tm['macro_f1']:.4f} | "
              f"病灶组F1之和(Bot+BF+XSS)={lesion_sum:.4f} | 良性误报率={te_bfpr:.6f}")

        peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if device.type == "cuda" else 0.0)
        total_sec = time.time() - t_start

        rec = {
            "model": args.model,
            "class_scheme": "12class",
            "command": cmd,
            "seed": args.seed,
            "device": str(device),
            "n_params": nparam,
            "config": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "dropout": args.dropout, "patience": args.patience,
                "val_frac": args.val_frac, "norm": "gn",
                "loss": args.loss, "gamma": args.gamma if args.loss == "focal" else None,
                "alpha_mode": args.alpha_mode if args.loss == "focal" else None,
                "sampler": args.sampler,
                "benign_target": benign_target,
                "minority_target": args.minority_target if args.sampler == "smote" else None,
                "smote_k": args.smote_k,
                "optimizer": "Adam", "input_shape": "(B, 1, 78)",
                "fpr_gate": {"audit_macro": args.fpr_abort_macro,
                             "audit_benign": args.fpr_abort_benign,
                             "gate_semantics": "v4.3_audit_pathological",
                             "diverged_ma3_benign": args.diverged_ma_benign,
                             "diverged_ma3_macro": args.diverged_ma_macro,
                             "rule": "gate_flag 仅逐 epoch 审计；唯一中止条件 = "
                                     "3-epoch 滑动平均 val benign FPR > 0.05 或 "
                                     "val macro FPR > 0.02（fpr_diverged）"},
                "sample_train": args.sample,
                "tier": tier_info,
                "pool": args.pool,
                "model_name": args.model,
            },
            "leakage_checks": {**leakage, "passed": True},
            "resampling_audit": resampling_audit,
            "n_fit_real": int(len(fit_idx)),
            "n_fit_used": int(len(y_fit_used)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(te)),
            "n_features": int(X.shape[1]),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "early_stopped": bool(early_stopped),
            "best_val_macro_f1": round(float(best_f1), 6),
            "val_per_class_at_best": best_val_per_class,
            "history": history,
            "main_metrics": {
                "macro_f1_12class": round(tm["macro_f1"], 6),
                "lesion_group_f1_sum": round(lesion_sum, 6),
                "lesion_group_f1": {class_names[c]: round(
                    tm["per_class"][class_names[c]]["f1"], 6) for c in lesion_ids},
            },
            "test_metrics": {**{k: tm[k] for k in eu.SCALAR_KEYS},
                             "benign_fpr": round(te_bfpr, 6)},
            "test_per_class": tm["per_class"],
            "timing": {
                "epoch_train_sec_median": round(float(np.median(
                    [h["train_sec"] for h in history])), 1),
                "total_sec": round(total_sec, 1),
                "test_eval_sec": round(test_sec, 1),
                "timing_anomalies": timing_anomalies,
            },
            "peak_gpu_mb": round(peak_mb, 1),
            "env": env_info(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        for p in (OUT_DIR / f"{name}.json", RUN_DIR / f"{name}.json"):
            with open(p, "w", encoding="utf-8") as fp:
                json.dump(rec, fp, ensure_ascii=False, indent=2)
        eu.confusion_df(test["y_true"], test["y_pred"], class_names).to_csv(
            OUT_DIR / f"{name}_confusion_matrix.csv")

        # ---------- 概率落盘（val 可搜 λ/w_c；test 在确定前不得读取） ----------
        if args.save_proba:
            np.save(PROBA_DIR / f"{name}_val.npy",
                    evaluate(model, dl_val, device, collect_proba=True)["y_prob"]
                    .astype(np.float32))
            np.save(PROBA_DIR / f"{name}_test.npy", test["y_prob"].astype(np.float32))
            np.save(PROBA_DIR / f"{name}_val_ytrue.npy", y_val.astype(np.int8))
            np.save(PROBA_DIR / f"{name}_test_ytrue.npy", test["y_true"].astype(np.int8))
            log(f"已保存 val/test 概率矩阵 -> {PROBA_DIR}/{name}_{{val,test}}.npy "
                f"(float32；val 用于搜 λ，test 在 λ 确定前不得读取)")

        # ---------- 最优权重落盘（与 train_deep.py 相同格式：纯 state_dict） ----------
        if args.save_ckpt and best_state is not None:
            p = OUT_DIR / f"{name}_best.pt"
            torch.save(best_state, p)
            rec["artifacts"] = {"best_ckpt": str(p.relative_to(ROOT))}
            with open(OUT_DIR / f"{name}.json", "w", encoding="utf-8") as fp:
                json.dump(rec, fp, ensure_ascii=False, indent=2)
            with open(RUN_DIR / f"{name}.json", "w", encoding="utf-8") as fp:
                json.dump(rec, fp, ensure_ascii=False, indent=2)
            log(f"已保存 best-epoch 权重 -> {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
    else:
        total_sec = time.time() - t_start
        aborted = {
            "model": args.model, "class_scheme": "12class", "command": cmd,
            "seed": args.seed, "aborted_reason": aborted_reason,
            "gate_semantics": "v4.3_audit_pathological",
            "benign_fpr_ma3": round(sum(benign_fpr_hist[-3:]) / 3.0, 6),
            "benign_fpr_window": [round(x, 6) for x in benign_fpr_hist[-3:]],
            "macro_fpr_ma3": round(sum(macro_fpr_hist[-3:]) / 3.0, 6),
            "macro_fpr_window": [round(x, 6) for x in macro_fpr_hist[-3:]],
            "history": history, "epochs_run": len(history),
            "note": "病理性发散门禁触发（3-epoch 滑动平均 benign FPR > 0.05），"
                    "测试集未评估、无结果产出（§3.3.2 v4.3）",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(RUN_DIR / f"{name}_aborted.json", "w", encoding="utf-8") as fp:
            json.dump(aborted, fp, ensure_ascii=False, indent=2)
        log(f"训练中止（{aborted_reason}），总耗时 {total_sec:.1f}s，"
            f"记录 -> {name}_aborted.json")

    del X_fit, X_val, X_te, X_fit_used, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"完成，总耗时 {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
