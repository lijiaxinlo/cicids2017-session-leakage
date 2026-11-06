"""E6：导出主方法 attention 权重（P1/P3 可解释性分析用）。

用 seed44 的 best 权重（seed42 的 run 早于 --save-ckpt 落盘功能，
无 ckpt；seed44 同为 γ=1/BENIGN full 正式配置，分析等价）。
测试集前向一次，存 alpha (756219, 19) float16。

输出：results/attn_alpha_seed44.npy
      results/attn_alpha_seed44_meta.json（逐类平均 alpha、与 pred 的一致性核对）
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import data_pipeline as dp
import models_adv as MA

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
CKPT = OUT / "deep_cnn_bilstm_attn_seed44_12class_g1_bfull_g1_bfull_best.pt"
SCALER = OUT / "cache" / "scaler_main.npz"
BATCH = 4096


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def main():
    class_names = dp.load_class_names()
    y = dp.load_y()
    tr, te = dp.get_split(y)
    X = dp.load_X(mmap=True)
    sc = np.load(SCALER)
    mean, scale = sc["mean"], sc["scale"]
    log(f"scaler: {SCALER.name}（fit 于 {int(sc['n_samples_fit'][0]):,} 行）")

    model = MA.build_adv_model("cnn_bilstm_attn", pool="attn")
    state = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    log(f"ckpt 载入 {CKPT.name} | 参数 {MA.count_params(model)}")

    n = len(te)
    alphas = np.empty((n, 19), dtype=np.float16)
    preds = np.empty(n, dtype=np.int8)
    for s in range(0, n, BATCH):
        idx = te[s:s + BATCH]
        xb = ((X[idx] - mean) / scale).astype(np.float32)
        xb = torch.from_numpy(xb).unsqueeze(1).to(dev)   # (B,1,78)
        logits, alpha = model(xb, return_alpha=True)
        alphas[s:s + len(idx)] = alpha.float().cpu().numpy().astype(np.float16)
        preds[s:s + len(idx)] = logits.argmax(1).cpu().numpy().astype(np.int8)
        if s % (BATCH * 40) == 0:
            log(f"  {s + len(idx):,}/{n:,}")

    np.save(OUT / "attn_alpha_seed44.npy", alphas)
    # 逐类平均 alpha（可解释性素材）
    per_class = {}
    for c, name in enumerate(class_names):
        m = y[te] == c
        if m.sum():
            per_class[name] = {
                "n": int(m.sum()),
                "mean_alpha": np.round(
                    alphas[m].astype(np.float32).mean(0), 4).tolist(),
                "argmax_step": int(alphas[m].astype(np.float32).mean(0).argmax()),
            }
    meta = {"ckpt": CKPT.name, "seed": 44,
            "note": "seed42 无 ckpt（早于落盘功能），用 seed44 等价正式配置",
            "alpha_shape": list(alphas.shape), "dtype": "float16",
            "steps": 19, "per_class_mean_alpha": per_class,
            "pred_dist": np.bincount(preds.astype(np.int64),
                                     minlength=len(class_names)).tolist()}
    with open(OUT / "attn_alpha_seed44_meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    log(f"完成 -> attn_alpha_seed44.npy {alphas.shape}；"
        f"逐类平均 alpha 已存 meta")


if __name__ == "__main__":
    main()
