#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
E17（v4.3 §4.3.3）：约束前沿逐类权重校准（0 GPU，纯后处理）。

模型：pred = argmax_c( w_c * p_c )，w_c > 0
搜索：坐标上升，逐类在 exp(linspace(log 0.1, log 10, 21)) 网格上扫，最多 3 轮
目标：val 12 类 macro-F1
硬约束：校准后 val benign FPR <= 档位值（每步先判约束，超约束候选跳过；
        整条网格都超则沿扩展网格升 w_0 强制收敛，并打 infeasible/enforced 标记）
约束前沿：benign 约束档 ∈ {0.003, 0.005, 0.008, 0.012, 0.016, 0.020} 各跑一次，
          w 固定后一次性算 test（benign FPR / macro-F1 / 病灶和），禁止用 test 调 w。

臂：
  C4（主方法，val 口径）：results/proba/deep_cnn_bilstm_attn_seed{42,44,46}_12class_g1_bfull*_val.npy
      （历史双拼名 _g1_bfull_g1_bfull 用 glob 兼容；seed42 为旧单拼名）
  C2 test-oracle（纯 CE，只有 test 段 -> test 搜 w 的 oracle 上限，单独标注）：
      results/deep_{cnn,lstm,cnn_lstm}_seed{42..46}_12class_test_proba.npy
  C2 val（P1 公平对照，val 搜 w -> test 一次性报告，与 C4 同口径可直接相减）：
      cnn_lstm: results/proba/deep_cnn_lstm_seed{42..46}_12class_val.npy（--save-val-proba 补跑）
      rf:       results/proba/rf_12class_seed{42..46}_val.npy（rf_val_proba.py 补跑，
                fit=90% 训练池 + class_weight=none，与深度侧 fit/val 协议一致）

用法：
  python src/calibrate_lambda.py              # 全量（C4 + C2-oracle + C2-val）
  python src/calibrate_lambda.py --only-c2val # 复用已有 C4/C2-oracle，只补 C2-val

纪律：test proba 在 w 冻结前不读取（代码结构上先锁 w 再碰 test）。
输出：results/e17_calibration_summary.json + results/e17_calibration_summary.md
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROBA = RESULTS / "proba"
N_CLASSES = 12
BENIGN_ID = 0
LESION_IDS = [9, 10, 11]          # Bot / Web Attack-Brute Force / Web Attack-XSS
TIERS = [0.003, 0.005, 0.008, 0.012, 0.016, 0.020]
GRID = np.exp(np.linspace(np.log(0.1), np.log(10.0), 21))   # 0.1 .. 10，含 1.0
MAX_ROUNDS = 3
SEEDS_MAIN = [42, 44, 46]
SEEDS_CE = [42, 43, 44, 45, 46]
SUMMARY_JSON = RESULTS / "e17_calibration_summary.json"
SUMMARY_MD = RESULTS / "e17_calibration_summary.md"
METRIC_KEYS = ["final_val_benign_fpr", "final_val_macro_f1",
               "test_benign_fpr", "test_macro_f1", "test_lesion_sum"]


def benign_fpr(y_true, y_pred):
    m = y_true == BENIGN_ID
    return float((y_pred[m] != BENIGN_ID).mean())


def macro_f1(y_true, y_pred):
    # 向量化 macro-F1（bincount 混淆矩阵），等价于 sklearn macro/zero_division=0，
    # 但快 ~10 倍（坐标上升每档位最多 756 次候选评估，必须快）
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(yt * N_CLASSES + yp,
                     minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom_p = tp + fp
    denom_r = tp + fn
    prec = np.divide(tp, denom_p, out=np.zeros_like(tp), where=denom_p > 0)
    rec = np.divide(tp, denom_r, out=np.zeros_like(tp), where=denom_r > 0)
    denom_f = prec + rec
    f1 = np.divide(2 * prec * rec, denom_f, out=np.zeros_like(tp),
                   where=denom_f > 0)
    return float(f1.mean())


def lesion_sum(y_true, y_pred):
    # 与 macro_f1 同一套混淆矩阵，取病灶三类 F1 之和
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    cm = np.bincount(yt * N_CLASSES + yp,
                     minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    s = 0.0
    for c in LESION_IDS:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        s += 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(s)


def calibrate(val_p, y_val, cap):
    """坐标上升：目标 val macro-F1，硬约束 val benign FPR <= cap。
    返回 (w, info)。info 含 rounds / infeasible / enforced_w0 标记。"""
    w = np.ones(N_CLASSES, dtype=np.float64)
    pred = np.argmax(val_p * w, axis=1)
    cur_f1 = macro_f1(y_val, pred)
    info = {"rounds": 0, "infeasible_classes": [],
            "initial_val_benign_fpr": round(benign_fpr(y_val, pred), 6)}
    for rnd in range(1, MAX_ROUNDS + 1):
        changed = False
        for c in range(N_CLASSES):
            best_f1, best_w = -1.0, None
            all_violate = True
            for g in GRID:
                w_try = w.copy()
                w_try[c] = g
                pred = np.argmax(val_p * w_try, axis=1)
                bf = benign_fpr(y_val, pred)
                if bf <= cap:
                    all_violate = False
                    f1 = macro_f1(y_val, pred)
                    if f1 > best_f1 + 1e-12:      # 平手取先扫到的（网格升序=小 w）
                        best_f1, best_w = f1, g
            if all_violate:
                # 整条网格都超约束：网格内无解，保持当前 w 并记 infeasible
                # （benign 类的强制收敛由 calibrate 末尾的 enforcement 段处理）
                info["infeasible_classes"].append({"class": c, "tier_cap": cap})
                continue
            if best_f1 > cur_f1 + 1e-9 and best_w is not None:
                w[c] = best_w
                cur_f1 = best_f1
                changed = True
        info["rounds"] = rnd
        if not changed:
            break
    # ---------- 约束强制段（文档兜底"取满足约束的最大 w"） ----------
    # 坐标上升可能在网格内找不到既满足约束又提升 F1 的候选（argmax FPR 已超档位时）。
    # benign FPR 对 w_0 单调不增 -> 沿扩展网格（1..10^4）向上扫 w_0，
    # 取满足约束的最小 w_0（对 F1 伤害最小）；连 10^4 都不满足则记 infeasible。
    pred = np.argmax(val_p * w, axis=1)
    if benign_fpr(y_val, pred) > cap:
        enforced = None
        for w0 in np.geomspace(max(w[0], 1.0), 1e4, 41):
            w_try = w.copy()
            w_try[0] = w0
            if benign_fpr(y_val, np.argmax(val_p * w_try, axis=1)) <= cap:
                enforced = float(w0)
                break
        if enforced is not None:
            w[0] = enforced
            info["enforced_w0"] = round(enforced, 2)
        else:
            info["constraint_unreachable"] = True
    pred = np.argmax(val_p * w, axis=1)
    info["final_val_benign_fpr"] = round(benign_fpr(y_val, pred), 6)
    info["final_val_macro_f1"] = round(macro_f1(y_val, pred), 6)
    info["w"] = [round(float(x), 4) for x in w]
    return w, info


def eval_frozen(w, test_p, y_test):
    """w 冻结后一次性算 test 指标（本函数不得在 w 确定前调用）。"""
    pred = np.argmax(test_p * w, axis=1)
    return {"test_benign_fpr": round(benign_fpr(y_test, pred), 6),
            "test_macro_f1": round(macro_f1(y_test, pred), 6),
            "test_lesion_sum": round(lesion_sum(y_test, pred), 6)}


def load_main_seed(seed):
    """按历史命名（seed42 单拼 g1_bfull；seed44/46 双拼 g1_bfull_g1_bfull）glob 定位。"""
    pats_val = sorted(glob.glob(str(
        PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull*_val.npy")))
    if len(pats_val) != 1:
        sys.exit(f"[E17] seed{seed} val proba 定位失败（{len(pats_val)} 个匹配），中止不猜测")
    base = pats_val[0][: -len("_val.npy")]
    files = {"val_p": f"{base}_val.npy", "val_y": f"{base}_val_ytrue.npy",
             "test_p": f"{base}_test.npy", "test_y": f"{base}_test_ytrue.npy"}
    for k, p in files.items():
        if not Path(p).exists():
            sys.exit(f"[E17] seed{seed} 缺 {k}: {p} —— 如实上报，不猜测")
    return {k: np.load(p) for k, p in files.items()}


def load_deep_val(seed, model="cnn_lstm"):
    """P1：cnn_lstm val proba（results/proba/，--save-val-proba 补跑）+ test proba（results/ 根，float16）。"""
    vp = PROBA / f"deep_{model}_seed{seed}_12class_val.npy"
    vy = PROBA / f"deep_{model}_seed{seed}_12class_val_ytrue.npy"
    tp = RESULTS / f"deep_{model}_seed{seed}_12class_test_proba.npy"
    ty = RESULTS / f"deep_{model}_seed{seed}_12class_test_ytrue.npy"
    missing = [str(p) for p in (vp, vy, tp, ty) if not p.exists()]
    if missing:
        return None, missing
    return {"val_p": np.load(vp), "val_y": np.load(vy),
            "test_p": np.load(tp).astype(np.float32), "test_y": np.load(ty)}, None


def load_rf_val(seed):
    """P1：RF-cal val/test proba（rf_val_proba.py 补跑，fit=90% 训练池）。"""
    vp = PROBA / f"rf_12class_seed{seed}_val.npy"
    vy = PROBA / f"rf_12class_seed{seed}_val_ytrue.npy"
    tp = PROBA / f"rf_12class_seed{seed}_test.npy"
    ty = PROBA / f"rf_12class_seed{seed}_test_ytrue.npy"
    missing = [str(p) for p in (vp, vy, tp, ty) if not p.exists()]
    if missing:
        return None, missing
    return {"val_p": np.load(vp), "val_y": np.load(vy),
            "test_p": np.load(tp), "test_y": np.load(ty)}, None


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": round(float(a.mean()), 6), "std": round(float(a.std(ddof=1)), 6)}


def run_c2_val(out):
    """P1 公平 C2(val)：val 搜 w -> test 冻结评估，与 C4 同口径可直接相减。"""
    rows = []
    for model, loader, seeds in [("cnn_lstm", load_deep_val, SEEDS_CE),
                                 ("rf", load_rf_val, SEEDS_CE)]:
        for seed in seeds:
            d, missing = loader(seed)
            if d is None:
                out["notes"].append(f"C2(val) 缺 {model} seed{seed} 输入，跳过: "
                                    f"{missing[0]}")
                continue
            val_p = d["val_p"].astype(np.float32)
            y_val = d["val_y"]
            test_p, y_test = d["test_p"], d["test_y"]
            print(f"[C2val] {model} seed{seed}: val {val_p.shape} / "
                  f"test {test_p.shape}")
            for cap in TIERS:
                t1 = time.time()
                w, info = calibrate(val_p, y_val, cap)
                m = eval_frozen(w, test_p, y_test)          # w 已冻结
                rows.append({"model": model, "seed": seed,
                             "tier_benign_cap": cap, "protocol": "val",
                             "argmax_val_benign_fpr": info.pop(
                                 "initial_val_benign_fpr"),
                             **info, **m, "sec": round(time.time() - t1, 1)})
            print(f"  cap=0.008: test benign "
                  f"{[r['test_benign_fpr'] for r in rows if r['model'] == model and r['seed'] == seed and r['tier_benign_cap'] == 0.008][0]:.4f}")
    return rows


def build_summary(c4_rows, c2_rows, c2v_rows, ce_models):
    summary = {"c4": {}, "c2": {}, "c2_val": {}}
    for cap in TIERS:
        rows = [r for r in c4_rows if r["tier_benign_cap"] == cap]
        if rows:
            summary["c4"][str(cap)] = {k: mean_std([r[k] for r in rows])
                                       for k in METRIC_KEYS}
    for model in ce_models:
        summary["c2"][model] = {}
        for cap in TIERS:
            rows = [r for r in c2_rows if r["model"] == model
                    and r["tier_benign_cap"] == cap]
            if rows:
                summary["c2"][model][str(cap)] = {k: mean_std([r[k] for r in rows])
                                                  for k in METRIC_KEYS}
    for model in ("cnn_lstm", "rf"):
        summary["c2_val"][model] = {}
        for cap in TIERS:
            rows = [r for r in c2v_rows if r["model"] == model
                    and r["tier_benign_cap"] == cap]
            if rows:
                summary["c2_val"][model][str(cap)] = {k: mean_std([r[k] for r in rows])
                                                      for k in METRIC_KEYS}
    # ---------- 公平判定：C4(val) - C2(val)（架构师 §4.3.3 阈值 0.03） ----------
    verdict = {}
    for cap in TIERS:
        c4 = summary["c4"].get(str(cap), {}).get("test_macro_f1", {}).get("mean")
        for model in ("cnn_lstm", "rf"):
            c2v = summary["c2_val"].get(model, {}).get(
                str(cap), {}).get("test_macro_f1", {}).get("mean")
            if c4 is not None and c2v is not None:
                verdict[f"cap{cap}_{model}"] = {
                    "c4_val_test_macro_f1": c4, "c2_val_test_macro_f1": c2v,
                    "delta": round(c4 - c2v, 6),
                    "contribution2_holds(>=0.03)": bool(c4 - c2v >= 0.03)}
    summary["c4_minus_c2val_verdict"] = verdict
    return summary


def write_md(out, summary, c4_rows, ce_models, t0):
    lines = ["# E17 约束前沿校准（v4.3 §4.3.3，含 P1 公平 C2(val) 对照）", "",
             "模型：`pred = argmax_c(w_c·p_c)`；坐标上升（网格 0.1–10，21 点，≤3 轮）；"
             "目标 = val 12 类 macro-F1；硬约束 = val benign FPR ≤ 档位。"
             "test 指标在 w 冻结后一次性计算。", "",
             "## C4 主方法（SMOTE+focal，γ=1，BENIGN=full；val 口径，3 种子 mean±std）",
             "", "| 约束档 | val benign FPR(实测) | val macro-F1 | test benign FPR | "
             "test macro-F1 | test 病灶和 |", "|---|---|---|---|---|---|"]
    for cap in TIERS:
        s = summary["c4"].get(str(cap))
        if s:
            lines.append(
                f"| ≤{cap} | {s['final_val_benign_fpr']['mean']:.4f}±{s['final_val_benign_fpr']['std']:.4f} "
                f"| {s['final_val_macro_f1']['mean']:.4f}±{s['final_val_macro_f1']['std']:.4f} "
                f"| {s['test_benign_fpr']['mean']:.4f}±{s['test_benign_fpr']['std']:.4f} "
                f"| {s['test_macro_f1']['mean']:.4f}±{s['test_macro_f1']['std']:.4f} "
                f"| {s['test_lesion_sum']['mean']:.4f}±{s['test_lesion_sum']['std']:.4f} |")
    lines += ["", "逐种子（C4）：", "", "| seed | 约束档 | val benign FPR | val F1 | "
              "test benign FPR | test F1 | 病灶和 |", "|---|---|---|---|---|---|---|"]
    for r in c4_rows:
        lines.append(f"| {r['seed']} | ≤{r['tier_benign_cap']} | "
                     f"{r['final_val_benign_fpr']:.4f} | {r['final_val_macro_f1']:.4f} | "
                     f"{r['test_benign_fpr']:.4f} | {r['test_macro_f1']:.4f} | "
                     f"{r['test_lesion_sum']:.4f} |")
    lines += ["", "## C2 纯 CE 对照（**test-oracle 口径**，单独标注，不与 val 口径混报）",
              "", "| 模型 | 约束档 | oracle benign FPR(实测) | oracle F1 | 备注 |",
              "|---|---|---|---|---|"]
    for model in ce_models:
        for cap in TIERS:
            s = summary["c2"].get(model, {}).get(str(cap))
            if s:
                lines.append(f"| {model} | ≤{cap} | "
                             f"{s['final_val_benign_fpr']['mean']:.4f} | "
                             f"{s['final_val_macro_f1']['mean']:.4f} | "
                             f"test 段搜索（oracle） |")
    lines += ["", "## C2 val 口径（**P1 公平对照**：val 搜 w -> test 冻结评估，与 C4 同口径）",
              "", "| 模型 | 约束档 | val benign FPR(实测) | val macro-F1 | "
              "test benign FPR | test macro-F1 | test 病灶和 |",
              "|---|---|---|---|---|---|---|"]
    for model in ("cnn_lstm", "rf"):
        for cap in TIERS:
            s = summary["c2_val"].get(model, {}).get(str(cap))
            if s:
                lines.append(
                    f"| {model} | ≤{cap} | "
                    f"{s['final_val_benign_fpr']['mean']:.4f}±{s['final_val_benign_fpr']['std']:.4f} "
                    f"| {s['final_val_macro_f1']['mean']:.4f}±{s['final_val_macro_f1']['std']:.4f} "
                    f"| {s['test_benign_fpr']['mean']:.4f}±{s['test_benign_fpr']['std']:.4f} "
                    f"| {s['test_macro_f1']['mean']:.4f}±{s['test_macro_f1']['std']:.4f} "
                    f"| {s['test_lesion_sum']['mean']:.4f}±{s['test_lesion_sum']['std']:.4f} |")
    lines += ["", "### 公平判定：C4(val) − C2(val)（贡献②阈值 ≥ 0.03）", "",
              "| 约束档 | 对照 | C4 test F1 | C2(val) test F1 | Δ | 贡献②成立 |",
              "|---|---|---|---|---|---|"]
    for key, v in summary.get("c4_minus_c2val_verdict", {}).items():
        cap, model = key.replace("cap", "", 1).split("_", 1)
        lines.append(f"| ≤{cap} | {model} | {v['c4_val_test_macro_f1']:.4f} | "
                     f"{v['c2_val_test_macro_f1']:.4f} | {v['delta']:+.4f} | "
                     f"{'是' if v['contribution2_holds(>=0.03)'] else '否'} |")
    lines += ["", "## 注记"]
    for n in out["notes"]:
        lines.append(f"- {n}")
    lines.append("- C2(val) cnn_lstm 的 val proba 由 --save-val-proba 补跑"
                 "（确定性逐位验证通过）；rf 为 RF-cal 臂（fit=90% 训练池、"
                 "class_weight=none，与深度侧 fit/val 协议一致），非已报告 RF 行")
    lines.append("- 逐种子 w 见 JSON（results/e17_calibration_summary.json）")
    lines.append(f"- 生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}，"
                 f"总耗时 {time.time() - t0:.0f}s")
    with open(SUMMARY_MD, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    t0 = time.time()
    only_c2val = "--only-c2val" in sys.argv
    out = {"protocol": "E17 v4.3 §4.3.3 约束前沿校准（含 P1 公平 C2(val) 对照）",
           "model": "pred=argmax(w*p), 坐标上升 3 轮, 网格 exp(linspace(log0.1,log10,21))",
           "objective": "val 12类 macro-F1，硬约束 val benign FPR <= 档位",
           "tiers": TIERS, "seeds_main": SEEDS_MAIN, "c4": [], "c2": [],
           "c2_val": [], "notes": []}
    ce_models = ["cnn", "lstm", "cnn_lstm"]

    c4_rows, c2_rows = [], []
    prev = None
    if only_c2val and SUMMARY_JSON.exists():
        with open(SUMMARY_JSON, encoding="utf-8") as fp:
            prev = json.load(fp)
    if only_c2val and prev and prev.get("c4") and prev.get("c2"):
        out["c4"], out["c2"] = prev["c4"], prev["c2"]
        out["notes"] = list(prev.get("notes", []))
        c4_rows, c2_rows = prev["c4"], prev["c2"]
        print("[reuse] --only-c2val：复用已有 C4 / C2-oracle 结果，不重算")
    else:
        # ---------- C4：主方法 3 种子（val 搜 -> test 一次性报告） ----------
        for seed in SEEDS_MAIN:
            d = load_main_seed(seed)
            y_val, val_p = d["val_y"], d["val_p"].astype(np.float32)
            test_p, y_test = d["test_p"].astype(np.float32), d["test_y"]
            print(f"[C4] seed{seed}: val {val_p.shape} / test {test_p.shape}")
            for cap in TIERS:
                t1 = time.time()
                w, info = calibrate(val_p, y_val, cap)
                test_m = eval_frozen(w, test_p.astype(np.float32), y_test)
                row = {"seed": seed, "tier_benign_cap": cap,
                       "argmax_val_benign_fpr": info.pop("initial_val_benign_fpr"),
                       **info, **test_m, "sec": round(time.time() - t1, 1)}
                c4_rows.append(row)
                print(f"  cap={cap}: val benign {info['final_val_benign_fpr']:.4f} "
                      f"val F1 {info['final_val_macro_f1']:.4f} -> test benign "
                      f"{test_m['test_benign_fpr']:.4f} test F1 "
                      f"{test_m['test_macro_f1']:.4f} 病灶和 "
                      f"{test_m['test_lesion_sum']:.4f} ({row['sec']}s)")
        out["c4"] = c4_rows

        # ---------- C2：纯 CE 对照（只有 test 段 -> test-oracle 口径，单独标注） ----------
        for model in ce_models:
            for seed in SEEDS_CE:
                tp = RESULTS / f"deep_{model}_seed{seed}_12class_test_proba.npy"
                ty = RESULTS / f"deep_{model}_seed{seed}_12class_test_ytrue.npy"
                if not (tp.exists() and ty.exists()):
                    out["notes"].append(f"C2 缺 {model} seed{seed} test proba，跳过")
                    continue
                test_p = np.load(tp).astype(np.float32)
                y_test = np.load(ty)
                print(f"[C2] {model} seed{seed}: test {test_p.shape} "
                      f"(test-oracle 口径)")
                for cap in TIERS:
                    w, info = calibrate(test_p, y_test, cap)
                    m = eval_frozen(w, test_p, y_test)
                    c2_rows.append({"model": model, "seed": seed,
                                    "tier_benign_cap": cap, "oracle": "test",
                                    **info, **m})
        out["c2"] = c2_rows
        out["notes"].append("C2(test-oracle) 全部为 test 段搜索（纯 CE proba 只有 "
                            "test 段），不得与 val 口径混报")

    # ---------- P1 公平 C2(val)：val 搜 -> test 冻结 ----------
    c2v_rows = run_c2_val(out)
    out["c2_val"] = c2v_rows

    summary = build_summary(c4_rows, c2_rows, c2v_rows, ce_models)
    out["summary_mean_std"] = summary

    with open(SUMMARY_JSON, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    write_md(out, summary, c4_rows, ce_models, t0)
    print(f"\n[E17] 完成，总耗时 {time.time() - t0:.0f}s -> "
          f"results/e17_calibration_summary.{{json,md}}")


if __name__ == "__main__":
    main()
