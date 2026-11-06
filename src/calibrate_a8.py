"""A8 校准臂（v4 §6.1.2f 必做，0 GPU）。

对象：A3（主架构 + 纯 CE + 无重采样）的 seed42 模型概率。
问题：'少数类不被预测' 是模型没学到，还是决策边界偏了？
做法：val 搜逐类权重 w_c（硬约束 benign FPR <= 0.008），test 冻结后报。
对比：A3 argmax vs A3+w。若增益 > 0.10 → 校准必须作为 Table 2 的一行；
若毫无增益 → 说明模型本身学不到（转预案 A）。

输出：results/a8_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import calibrate_lambda as CL  # noqa: E402

PROBA = ROOT / "results" / "proba"
OUT = ROOT / "results"
CAP = 0.008
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")


def main():
    vp = PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_val.npy"
    vy = PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_val_ytrue.npy"
    tp = PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_test.npy"
    ty = PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_test_ytrue.npy"
    miss = [str(p) for p in (vp, vy, tp, ty) if not p.exists()]
    if miss:
        raise SystemExit(f"缺输入: {miss}")

    y_val, val_p = np.load(vy), np.load(vp).astype(np.float32)
    y_te, test_p = np.load(ty), np.load(tp).astype(np.float32)

    pred = test_p.argmax(1)
    argmax_m = {"macro_f1": CL.macro_f1(y_te, pred),
                "lesion_sum": CL.lesion_sum(y_te, pred),
                "benign_fpr": CL.benign_fpr(y_te, pred)}
    w, info = CL.calibrate(val_p, y_val, CAP)
    cal_m = CL.eval_frozen(w, test_p, y_te)

    rec = {"purpose": "A8 校准臂：A3（CE+无重采样）模型 + 逐类权重校准",
           "cap_benign_fpr": CAP, "seed": 42,
           "argmax": argmax_m,
           "calibrated": cal_m,
           "w": np.asarray(w).round(4).tolist(),
           "val_at_best": {"macro_f1": info["final_val_macro_f1"],
                           "benign_fpr": info["final_val_benign_fpr"]},
           "delta_macro": cal_m["test_macro_f1"] - argmax_m["macro_f1"],
           "delta_lesion": cal_m["test_lesion_sum"] - argmax_m["lesion_sum"]}
    with open(OUT / "a8_calibration.json", "w", encoding="utf-8") as fp:
        json.dump(rec, fp, ensure_ascii=False, indent=2)
    print(json.dumps(rec, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
