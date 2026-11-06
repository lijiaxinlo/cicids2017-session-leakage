# -*- coding: utf-8 -*-
"""
把训练好的网络的**内部表征**导出来，供"换家族、不换表征"的对照使用。

论文的中心论断涉及"表征"，而原来的 2x2 只填了对角线（森林接 78 维原始、
序列网络接 19 步池化），两者同时换了家族和表征。本脚本导出网络自己算出的两个
表征，让森林直接接上去：

  conv_max : 卷积主干输出 (B, 128, 19) 沿 19 步取通道内最大值 -> (B, 128)
             —— "19 步池化视图"的逐通道汇总，是网络看到的、池化之后的东西
  ctx      : BiLSTM + attention 之后的 128 维上下文向量
             —— 分类头 fc1 的实际输入，即网络用来做决策的那个向量

森林接在 ctx 上等价于"把 2 层 MLP 头换成森林，其余一字不动"。

用法
----
    python src/export_conv_feats.py --ckpt results/deep_..._best.pt --out results/feats_p2_convfeat.npz

产出
----
    <out> : train_conv_max, test_conv_max, train_ctx, test_ctx (float32)
            行序与 results/temporal/{train,test}_idx.npy 一致
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import data_pipeline as dp
from models_adv import build_adv_model

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-dir", default="results/temporal")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = dp.load_X(mmap=True)
    tr = np.load(ROOT / args.split_dir / "train_idx.npy")
    te = np.load(ROOT / args.split_dir / "test_idx.npy")
    sc = np.load(CACHE / "scaler_main.npz")
    mean, scale = sc["mean"], sc["scale"]

    model = build_adv_model("cnn_bilstm_attn", pool="attn").to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location=device))
    model.eval()

    out = {}
    for tag, idx in (("train", tr), ("test", te)):
        n = len(idx)
        conv_max = np.empty((n, 128), dtype=np.float32)
        ctx = np.empty((n, 128), dtype=np.float32)
        for s in range(0, n, args.batch_size):
            sel = idx[s:s + args.batch_size]
            xb = (np.asarray(X[sel], dtype=np.float64) - mean) / scale
            xb = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32))
            xb = xb.unsqueeze(1).to(device)
            with torch.no_grad():
                z = model.conv(xb)                       # (B,128,19)
                conv_max[s:s + len(sel)] = z.max(dim=2).values.cpu().numpy()
                h, _ = model.lstm(z.transpose(1, 2))     # (B,19,128)
                c, _ = model.attention(h)                # (B,128)
                ctx[s:s + len(sel)] = c.cpu().numpy()
        out[f"{tag}_conv_max"] = conv_max
        out[f"{tag}_ctx"] = ctx
        print(f"{tag}: {n:,} 行 -> conv_max {conv_max.shape} ctx {ctx.shape}", flush=True)

    np.savez_compressed(ROOT / args.out, **out)
    print("写出", args.out)


if __name__ == "__main__":
    main()
