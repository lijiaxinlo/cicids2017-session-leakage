# -*- coding: utf-8 -*-
"""
时序协议下的**分会话**评估：把测试集按捕获会话切开，看结论是否只在某一个会话里成立。

返修意见 1 的后半句是 "validate the main findings across additional datasets or
independent scenarios"。本数据集里 11 个攻击类各自只出现在一个捕获会话（见
session_index.json），因此"留一会话"式的外部验证不可识别；可行的独立情景是
**两个时间方向**（正序 / 倒序）与**分会话分解**。本脚本做后者。

输入：results/proba/deep_cnn_bilstm_attn_seed*_12class_T_trt_nogate_{val,test}*.npy
      results/temporal/calibration.json（C4 冻结权重）
输出：results/temporal/per_session.json
"""
import json
from pathlib import Path

import numpy as np

import data_pipeline as dp
from session_index import SESSIONS

ROOT = Path(__file__).resolve().parent.parent
PROBA = ROOT / "results" / "proba"
TEM = ROOT / "results" / "temporal"
SPLITS = ROOT / "results" / "splits"
LESION = [9, 10, 11]
SEEDS = [42, 44, 46]
TAG = "T_trt_g1"


def f1_binary(y, pred, c):
    tp = np.sum((pred == c) & (y == c))
    fp = np.sum((pred == c) & (y != c))
    fn = np.sum((pred != c) & (y == c))
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else float(2 * tp / d)


def macro_f1(y, pred, n_cls=12):
    """只在测试集里真实出现过的类上取平均（分会话后很多类为空）。"""
    present = [c for c in range(n_cls) if np.any(y == c)]
    return float(np.mean([f1_binary(y, pred, c) for c in present])), present


def main():
    sid = np.load(SPLITS / "session_id.npy")
    calib = json.loads((TEM / "calibration_g1_fixed.json").read_text(encoding="utf-8"))

    te_idx = np.load(TEM / "test_idx.npy")
    te_sid = sid[te_idx]

    out = {"seeds": {}, "note": "macro 只在会话内真实出现的类上取平均"}
    for s in SEEDS:
        base = PROBA / f"deep_cnn_bilstm_attn_seed{s}_12class_{TAG}"
        p = np.load(f"{base}_test.npy")
        yt = np.load(f"{base}_test_ytrue.npy")
        w = np.array(calib[str(s)]["C4"]["w"], dtype=np.float64)
        pred = np.argmax(p * w, axis=1)

        rec = {"overall": {"macro": round(macro_f1(yt, pred)[0], 4),
                           "lfs": round(sum(f1_binary(yt, pred, c) for c in LESION), 4),
                           "benign_fpr": round(float(np.mean(pred[yt == 0] != 0)), 5),
                           "n": int(len(yt))},
               "by_session": {}}
        for sv in np.unique(te_sid):
            m = te_sid == sv
            ys, ps = yt[m], pred[m]
            lab = SESSIONS[int(sv)]
            rec["by_session"][lab] = {
                "n": int(m.sum()),
                "classes_present": [int(c) for c in np.unique(ys)],
                "macro": round(macro_f1(ys, ps)[0], 4),
                "lfs": round(sum(f1_binary(ys, ps, c) for c in LESION), 4),
                "bot": round(f1_binary(ys, ps, 9), 4),
                "bf": round(f1_binary(ys, ps, 10), 4),
                "xss": round(f1_binary(ys, ps, 11), 4),
                "benign_fpr": (round(float(np.mean(ps[ys == 0] != 0)), 5)
                               if np.any(ys == 0) else None),
            }
        out["seeds"][str(s)] = rec
        print(f"seed {s}: overall {rec['overall']}")
        for k, v in rec["by_session"].items():
            print(f"   {k:<46} n={v['n']:>7,} macro={v['macro']:.4f} "
                  f"lfs={v['lfs']:.4f} fpr={v['benign_fpr']}")

    (TEM / "per_session.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\n写出 {TEM / 'per_session.json'}")


if __name__ == "__main__":
    main()
