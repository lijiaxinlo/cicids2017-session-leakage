# -*- coding: utf-8 -*-
"""
深度 baseline 训练脚本（阶段一 · 12 类口径）
==========================================
读 data_pipeline.py 的固定划分 -> 训练池内部再切 10% 验证集（早停用，绝不碰测试集）
-> 只在训练集上 fit StandardScaler -> 训练 -> 测试集只评估一次。

归一化统一说明：三个深度 baseline（CNN / LSTM / CNN-LSTM）一律使用 **GroupNorm**，
不用 BatchNorm。原因是 BatchNorm 在预实验中出现间歇性 loss 背离（running stats
在长 epoch 下与权重错位，val macro-F1 曾从 0.70 崩到 0.20 后自愈）。GN 逐样本归一化、
无 running stats、长度无关，实测精度不降。--norm bn/ln 仅供诊断对照，不得用于正式结果。

用法
----
    # 冒烟（分层抽样 30 万，2 epoch）
    python src/train_deep.py --model cnn  --seed 42 --epochs 2 --sample 300000
    python src/train_deep.py --model lstm --seed 42 --epochs 2 --sample 300000
    python src/train_deep.py --model cnn_lstm --seed 42 --epochs 2 --sample 300000

    # 全量正式训练（30 epoch 上限 + patience=5 + best-epoch 回滚）
    python src/train_deep.py --model cnn --seed 42 --epochs 30 --patience 5

    # BN/LN 诊断对照（不用于正式结果）
    python src/train_deep.py --model cnn_lstm --seed 42 --epochs 5 --norm bn --tag diag

产出
----
    results/deep_<model>_seed<seed>_12class.json        完整记录（含 per-epoch 历史）
    results/runs/deep_<model>_seed<seed>_12class.json   同上
    results/deep_<model>_seed<seed>_12class_confusion_matrix.csv
    results/cache/scaler.npz                            全量；冒烟为 scaler_sample<N>.npz
"""

import argparse
import gc
import json
import os
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
import models as M

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results"
RUN_DIR = OUT_DIR / "runs"
CACHE_DIR = OUT_DIR / "cache"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRAC = 0.1
CHUNK = 200_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_info():
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return info


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler_chunked(X, idx, scaler):
    """用 partial_fit 分块拟合 StandardScaler，避免把全量转成 float64 撑爆内存。"""
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
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


@torch.no_grad()
def evaluate(model, loader, device, n_classes, collect_pred=False,
             collect_proba=False):
    model.eval()
    total_loss, n = 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    preds, trues, probs = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        out = model(xb)
        total_loss += float(crit(out, yb))
        n += yb.numel()
        p = out.argmax(1)
        preds.append(p.cpu().numpy())
        if collect_proba:
            probs.append(torch.softmax(out, dim=1).float().cpu().numpy())
        if collect_pred:
            trues.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds)
    res = {"loss": total_loss / max(n, 1), "y_pred": y_pred}
    if collect_pred:
        res["y_true"] = np.concatenate(trues)
    if collect_proba:
        res["y_prob"] = np.concatenate(probs)
    return res


def main():
    ap = argparse.ArgumentParser(description="深度 baseline 训练（CNN/LSTM/CNN-LSTM）")
    ap.add_argument("--model", required=True, choices=list(M.MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sample", type=int, default=None,
                    help="训练集分层抽样行数（冒烟用），测试集仍用完整测试集")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--norm", type=str, default="gn",
                    choices=["bn", "gn", "ln", "none"],
                    help="卷积段归一化方式。默认 gn（GroupNorm）：三个深度 baseline "
                         "统一使用，无 running stats，避免 BatchNorm 的间歇性 loss 背离；"
                         "bn/ln 仅供诊断对照")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--save-proba", action="store_true",
                    help="保存测试集逐类概率 softmax(logits) 为 float16 .npy"
                         "（供阶段二推理期先验校正 λ 搜索使用；不影响训练与指标）")
    ap.add_argument("--save-val-proba", action="store_true",
                    help="（P1 新增）保存 best 权重下 val 集逐类概率 float32 + val ytrue "
                         "到 results/proba/{name}_val.npy / _val_ytrue.npy"
                         "（供 C2(val) 公平校准对照；默认关闭，不影响任何既有行为）")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="保存 best-epoch 权重 .pt（便于不重训就重算推理结果）")
    ap.add_argument("--tag", type=str, default=None, help="输出文件后缀标记")
    args = ap.parse_args()

    torch.set_num_threads(args.num_threads)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件名带口径标记：全量默认配置产出 deep_<model>_seed<seed>_12class.json
    suffix = f"_sample{args.sample}" if args.sample else ""
    if args.norm != "gn":
        suffix += f"_{args.norm}"
    if args.tag:
        suffix += f"_{args.tag}"
    name = f"deep_{args.model}_seed{args.seed}_12class{suffix}"
    cmd = " ".join(["python", "src/train_deep.py", "--model", args.model,
                    "--seed", str(args.seed), "--epochs", str(args.epochs)] +
                   (["--sample", str(args.sample)] if args.sample else []) +
                   ["--batch-size", str(args.batch_size), "--norm", args.norm,
                    "--patience", str(args.patience)] +
                   (["--save-proba"] if args.save_proba else []) +
                   (["--save-val-proba"] if args.save_val_proba else []) +
                   (["--save-ckpt"] if args.save_ckpt else []))

    t_start = time.time()
    class_names = dp.load_class_names()
    y = dp.load_y()
    tr_all, te = dp.get_split(y)
    X = dp.load_X(mmap=True)
    log(f"模型={args.model} seed={args.seed} device={device} "
        f"batch={args.batch_size} lr={args.lr}")
    log(f"固定划分: 训练池 {len(tr_all):,} / 测试 {len(te):,}")

    # ---------- 训练集可选分层抽样（只动训练池，测试集保持完整） ----------
    if args.sample and args.sample < len(tr_all):
        tr_sel, _ = train_test_split(
            tr_all, train_size=args.sample, stratify=y[tr_all],
            random_state=args.seed)
        tr_sel = np.sort(tr_sel)
        log(f"训练池分层抽样至 {len(tr_sel):,} 行"
            f"（stratify={len(class_names)} 类，seed={args.seed}）")
    else:
        tr_sel = tr_all

    # ---------- 训练集内部再切验证集（早停监控验证集 macro-F1） ----------
    fit_idx, val_idx = train_test_split(
        tr_sel, test_size=args.val_frac, stratify=y[tr_sel],
        random_state=args.seed)
    fit_idx, val_idx = np.sort(fit_idx), np.sort(val_idx)

    # 数据泄漏硬断言：验证集必须完全落在训练池内，且与测试集不相交
    pool = np.zeros(len(y), dtype=bool)
    pool[tr_all] = True
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[te] = True
    n_val_in_test = int(test_mask[val_idx].sum())
    n_fit_in_test = int(test_mask[fit_idx].sum())
    n_val_outside_pool = int((~pool[val_idx]).sum())
    assert n_val_in_test == 0, f"验证集有 {n_val_in_test} 条落在测试集，数据泄漏！"
    assert n_fit_in_test == 0, f"训练集有 {n_fit_in_test} 条落在测试集，数据泄漏！"
    assert n_val_outside_pool == 0, f"验证集有 {n_val_outside_pool} 条不在训练池内！"
    assert len(np.intersect1d(fit_idx, val_idx)) == 0, "训练/验证有交集"
    log(f"训练 {len(fit_idx):,} / 验证 {len(val_idx):,} / 测试 {len(te):,}")
    log(f"[泄漏检查] val∩test=0, fit∩test=0, val⊂train_pool=True, "
        f"fit∩val=0 —— 测试集全程不参与训练与早停")

    # ---------- 归一化：只在训练集 fit ----------
    scaler = StandardScaler()
    t0 = time.time()
    fit_scaler_chunked(X, fit_idx, scaler)
    scaler_path = CACHE_DIR / (f"scaler_sample{args.sample}.npz" if args.sample
                               else "scaler.npz")
    np.savez(scaler_path, mean=scaler.mean_.astype(np.float64),
             scale=scaler.scale_.astype(np.float64),
             n_samples_fit=np.array([len(fit_idx)]))
    log(f"StandardScaler 仅在训练集 fit（{len(fit_idx):,} 行），"
        f"{time.time() - t0:.1f}s，已存 {scaler_path}")
    assert len(fit_idx) < len(y), "scaler 不应接触测试集"

    X_fit = transform_chunked(X, fit_idx, scaler)
    X_val = transform_chunked(X, val_idx, scaler)
    X_te = transform_chunked(X, te, scaler)
    y_fit, y_val, y_te = y[fit_idx].astype(np.int64), y[val_idx].astype(np.int64), y[te].astype(np.int64)
    log(f"标准化完成: X_fit {X_fit.shape} {X_fit.nbytes / 1e6:.0f} MB, "
        f"X_val {X_val.nbytes / 1e6:.0f} MB, X_te {X_te.nbytes / 1e6:.0f} MB")

    dl_fit = make_loader(X_fit, y_fit, args.batch_size, shuffle=True)
    dl_val = make_loader(X_val, y_val, args.batch_size, shuffle=False)
    dl_te = make_loader(X_te, y_te, args.batch_size, shuffle=False)

    # ---------- 模型 ----------
    model = M.build_model(args.model, n_features=X.shape[1],
                          n_classes=len(class_names), dropout=args.dropout,
                          norm=args.norm).to(device)
    nparam = M.count_params(model)
    log(f"模型参数量: {nparam}")
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    history = []
    train_secs = []
    timing_anomalies = []
    best_f1, best_epoch, best_state = -1.0, -1, None
    bad_epochs = 0

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

        t_v = time.time()
        val = evaluate(model, dl_val, device, len(class_names))
        val_sec = time.time() - t_v
        vm = eu.compute_metrics(y_val, val["y_pred"], class_names=class_names,
                                labels=list(range(len(class_names))))
        peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if device.type == "cuda" else 0.0)

        # 耗时异常检测：与前几轮中位数比较，>1.5x 视为异常（机器负载污染）
        train_secs.append(train_sec)
        med_prev = float(np.median(train_secs[:-1])) if len(train_secs) > 2 else None
        anomaly = bool(med_prev is not None and train_sec > 1.5 * med_prev)
        if anomaly:
            timing_anomalies.append({"epoch": ep, "train_sec": round(train_sec, 1),
                                     "median_prev": round(med_prev, 1),
                                     "ratio": round(train_sec / med_prev, 2)})

        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val["loss"], 6),
            "gap_val_minus_train": round(val["loss"] - train_loss, 6),
            "val_macro_f1": round(vm["macro_f1"], 6),
            "val_macro_recall": round(vm["macro_recall"], 6),
            "train_sec": round(train_sec, 1),
            "val_sec": round(val_sec, 1),
            "peak_gpu_mb": round(peak_mb, 1),
            "timing_anomaly": anomaly,
        })
        flag = ""
        if anomaly:
            flag = (f"  <<< [耗时异常] {train_sec:.1f}s 为中位数 "
                    f"{med_prev:.1f}s 的 {train_sec / med_prev:.2f}x，"
                    f"疑机器负载污染，该轮耗时不可用于排期")
        log(f"epoch {ep}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | gap {val['loss'] - train_loss:+.4f} | "
            f"val_macroF1 {vm['macro_f1']:.4f} | "
            f"train {train_sec:.1f}s val {val_sec:.1f}s | peak {peak_mb:.0f} MB{flag}")

        if vm["macro_f1"] > best_f1 + 1e-6:
            best_f1, best_epoch = vm["macro_f1"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"早停：验证集 macro-F1 连续 {args.patience} 轮无提升")
                break

    # ---------- 用最佳权重在测试集评估一次 ----------
    if best_state is not None:
        model.load_state_dict(best_state)
    t_te = time.time()
    test = evaluate(model, dl_te, device, len(class_names), collect_pred=True,
                    collect_proba=args.save_proba)
    test_sec = time.time() - t_te
    tm = eu.compute_metrics(test["y_true"], test["y_pred"],
                            class_names=class_names,
                            labels=list(range(len(class_names))))
    eu.print_report(f"{args.model} · seed={args.seed} · TEST (best epoch {best_epoch})",
                    tm, class_names)

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
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "norm": args.norm,
            "norm_note": "三个深度 baseline 统一使用 GroupNorm（无 running stats），"
                         "BatchNorm 在预实验中存在间歇性 loss 背离，仅作诊断对照",
            "patience": args.patience,
            "val_frac": args.val_frac,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "input_shape": "(B, 1, 78)",
            "sample_train": args.sample,
        },
        "leakage_checks": {
            "val_in_test": n_val_in_test,
            "fit_in_test": n_fit_in_test,
            "val_outside_train_pool": n_val_outside_pool,
            "fit_val_overlap": int(len(np.intersect1d(fit_idx, val_idx))),
            "passed": True,
        },
        "n_fit": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(te)),
        "n_features": int(X.shape[1]),
        "split_protocol": "固定划分 results/splits/；验证集从训练池内切 10%（seed=种子）",
        "scaler_path": str(scaler_path.relative_to(ROOT)),
        "scaler_fit_on": "仅训练集（fit_idx）",
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(best_f1), 6),
        "history": history,
        "test_metrics": {k: tm[k] for k in eu.SCALAR_KEYS},
        "test_per_class": tm["per_class"],
        "timing": {
            "epoch_train_sec_avg": round(float(np.mean([h["train_sec"] for h in history])), 1),
            "epoch_train_sec_median": round(float(np.median([h["train_sec"] for h in history])), 1),
            "total_sec": round(total_sec, 1),
            "test_eval_sec": round(test_sec, 1),
            "timing_anomalies": timing_anomalies,
            "anomaly_rule": "单轮训练耗时 > 前几轮中位数 x1.5 即记为异常，不用于排期",
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

    # ---------- 可选：保存测试集逐类概率与 best 权重（纯附加，不影响训练与指标）----------
    artifact_paths = {}
    if args.save_proba:
        p = OUT_DIR / f"{name}_test_proba.npy"
        np.save(p, test["y_prob"].astype(np.float16))
        artifact_paths["test_proba"] = str(p.relative_to(ROOT))
        np.save(OUT_DIR / f"{name}_test_ytrue.npy", test["y_true"].astype(np.int8))
        artifact_paths["test_ytrue"] = str(
            (OUT_DIR / f"{name}_test_ytrue.npy").relative_to(ROOT))
        log(f"已保存测试集概率 {test['y_prob'].shape} float16 -> {p.name} "
            f"({p.stat().st_size / 1e6:.1f} MB)")
    # ---------- 可选（P1）：best 权重下的 val 概率落盘（C2(val) 公平校准用） ----------
    if args.save_val_proba:
        proba_dir = OUT_DIR / "proba"
        proba_dir.mkdir(exist_ok=True)
        val_best = evaluate(model, dl_val, device, len(class_names),
                            collect_proba=True)
        pv = proba_dir / f"{name}_val.npy"
        np.save(pv, val_best["y_prob"].astype(np.float32))
        np.save(proba_dir / f"{name}_val_ytrue.npy", y_val.astype(np.int8))
        artifact_paths["val_proba"] = str(pv.relative_to(ROOT))
        artifact_paths["val_ytrue"] = str(
            (proba_dir / f"{name}_val_ytrue.npy").relative_to(ROOT))
        log(f"已保存 val 概率 {val_best['y_prob'].shape} float32 -> {pv.name} "
            f"(best epoch {best_epoch} 权重；val 定义 = 训练池内 {args.val_frac:.0%} "
            f"分层切分 random_state={args.seed}，与早停监控集一致)")

    if args.save_ckpt and best_state is not None:
        p = OUT_DIR / f"{name}_best.pt"
        torch.save(best_state, p)
        artifact_paths["best_ckpt"] = str(p.relative_to(ROOT))
        log(f"已保存 best-epoch 权重 -> {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
    if artifact_paths:
        rec["artifacts"] = artifact_paths
        for p in (OUT_DIR / f"{name}.json", RUN_DIR / f"{name}.json"):
            with open(p, "w", encoding="utf-8") as fp:
                json.dump(rec, fp, ensure_ascii=False, indent=2)
    log(f"完成，总耗时 {total_sec:.1f}s，峰值显存 {peak_mb:.0f} MB -> {name}.json")

    del X_fit, X_val, X_te, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
