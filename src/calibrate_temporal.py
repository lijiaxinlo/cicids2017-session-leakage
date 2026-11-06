# -*- coding: utf-8 -*-
"""
P2（时序协议）处理臂的推理期校准：C4（坐标上升，macro 目标）与 C4′（联合搜索，病灶目标）。
=====================================================================================

协议与 P1 完全一致，只换 val/test 概率矩阵：
  * 权重 w ∈ [0.1, 10]^12，pred = argmax_c w_c p_c
  * 硬约束：val benign FPR <= 0.008
  * C4  = 坐标上升（21 点 log 网格，≤3 轮），目标 val macro-F1 —— 预注册口径
  * C4′ = 差分进化，目标 val 病灶组 F1 之和 —— 与论文所问的量对齐
  * w 冻结后才碰 test（结构上保证）

用法：python src/calibrate_temporal.py
输出：results/temporal/calibration.json
"""
import argparse
import json
import os
from pathlib import Path

# 线程数限制：本脚本与训练并发时曾因 BLAS 抢占把单轮训练拖到 3800s
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
from scipy.optimize import differential_evolution

import calibrate_lambda as CL

ROOT = Path(__file__).resolve().parent.parent
PROBA = ROOT / "results" / "proba"
OUT = ROOT / "results" / "temporal" / "calibration.json"
CAP = 0.008
LESION = [9, 10, 11]
SEEDS = [42, 44, 46]
TAG = "T_trt_nogate"


def lesion_sum(y, pred):
    s = 0.0
    for c in LESION:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        d = 2 * tp + fp + fn
        s += 0.0 if d == 0 else 2 * tp / d
    return float(s)


def fit_joint(val_p, y_val, cap=CAP, seed=0):
    """差分进化，目标 = val 病灶和；约束 = val benign FPR <= cap（超约束即罚）。"""
    def obj(w):
        pred = np.argmax(val_p * w, axis=1)
        fpr = float(np.mean(pred[y_val == 0] != 0))
        if fpr > cap:
            return 10.0 + (fpr - cap) * 100.0
        return -lesion_sum(y_val, pred)

    best = None
    for s in range(1):                      # 单次重启：P2 的 val 是 P1 的 2.1 倍，
        r = differential_evolution(         # 且与训练并发时墙钟成本过高
            obj, [(0.1, 10.0)] * 12, seed=seed + s, maxiter=60,
            popsize=10, tol=1e-6, polish=False, init="sobol")
        if best is None or r.fun < best.fun:
            best = r
    w = best.x
    pred = np.argmax(val_p * w, axis=1)
    return w, {"val_benign_fpr": round(float(np.mean(pred[y_val == 0] != 0)), 6),
               "val_macro_f1": round(CL.macro_f1(y_val, pred), 6),
               "val_lesion_sum": round(lesion_sum(y_val, pred), 6),
               "objective": float(best.fun)}


def eval_frozen(w, p, y):
    pred = np.argmax(p * w, axis=1)
    out = {"test_macro_f1": round(CL.macro_f1(y, pred), 6),
           "test_lesion_sum": round(lesion_sum(y, pred), 6),
           "test_benign_fpr": round(float(np.mean(pred[y == 0] != 0)), 6)}
    # 逐类 F1：2tp / (2tp + fp + fn)。注意分母必须含两倍的 tp，
    # 早期版本写成 2tp/(tp+fp+fn)，会把 F1 抬高（0.239 显示成 0.272）。
    for tag, c in (("bot", 9), ("bf", 10), ("xss", 11)):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        d = 2 * tp + fp + fn
        out[f"test_{tag}"] = round(float(2 * tp / d) if d else 0.0, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["c4", "c4p", "both"],
                    help="只跑坐标上升(C4)、只跑联合搜索(C4')、或两者")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default=TAG, help="概率矩阵后缀（= 训练时的 --tag）")
    a = ap.parse_args()
    out = Path(a.out)

    # 已有结果续跑：按种子增量写盘，中断不丢
    res = {}
    if out.exists():
        try:
            res = json.loads(out.read_text(encoding="utf-8"))
            res.pop("agg", None)
        except Exception:
            res = {}

    for s in SEEDS:
        base = PROBA / f"deep_cnn_bilstm_attn_seed{s}_12class_{a.tag}"
        vp = np.load(f"{base}_val.npy"); vy = np.load(f"{base}_val_ytrue.npy")
        tp = np.load(f"{base}_test.npy"); ty = np.load(f"{base}_test_ytrue.npy")
        print(f"seed {s}: val {vp.shape} test {tp.shape}", flush=True)
        rec = res.setdefault(str(s), {})

        if a.mode in ("c4", "both") and "C4" not in rec:
            w4, info4 = CL.calibrate(vp, vy, CAP)          # C4：坐标上升
            t4 = eval_frozen(w4, tp, ty)
            print(f"  C4  val {info4.get('final_val_benign_fpr')} -> test {t4}", flush=True)
            rec["C4"] = {"w": [round(float(x), 4) for x in w4], "val": info4, "test": t4}
            out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

        if a.mode in ("c4p", "both") and "C4p" not in rec:
            wj, infoj = fit_joint(vp, vy)                  # C4′：联合搜索
            tj = eval_frozen(wj, tp, ty)
            print(f"  C4' val {infoj['val_benign_fpr']} -> test {tj}", flush=True)
            rec["C4p"] = {"w": [round(float(x), 4) for x in wj], "val": infoj, "test": tj}
            out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    agg = {}
    for k in ("C4", "C4p"):
        have = [s for s in SEEDS if k in res.get(str(s), {})]
        if len(have) < len(SEEDS):
            print(f"[跳过 {k} 聚合] 只有 {len(have)}/{len(SEEDS)} 个种子")
            continue
        for m in ("test_macro_f1", "test_lesion_sum", "test_benign_fpr",
                  "test_bot", "test_bf", "test_xss"):
            v = np.array([res[str(s)][k]["test"][m] for s in SEEDS])
            agg[f"{k}_{m}"] = [round(float(v.mean()), 4), round(float(v.std()), 4)]
    res["agg"] = agg
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n== 3 种子均值 ==")
    for k, v in agg.items():
        print(f"  {k:<28} {v[0]} ± {v[1]}")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
