#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""真正的 oracle 上界搜索（任务 A）+ C2 口径差异量化（任务 B）。

背景：稿件 Table 6 的两行 "test (oracle)" 实际是 E17 坐标上升从均匀权重出发的单坐标局部最优
（U-oracle_seed42: rounds=1, 仅 BENIGN 由 1.0 -> 约束强制段抬高, ΔLFS = 0.0000）。
本脚本在 test 段 proba 上做**穷尽式**搜索，检验"预算内回收 0.0000"这一结论是否只是搜索算法的产物。

模型：pred = argmax_c ( w_c * p_c )，w ∈ [0.1, 10]^12
目标：LFS = F1(Bot) + F1(Web Attack-Brute Force) + F1(Web Attack-XSS)   （类 9/10/11）
硬约束：benign FPR <= 0.008

任务 A（test 段 = oracle 口径，合法用于定界）：
  A1 随机搜索      : 对数均匀 [0.1,10] 采 20,000 组，只留可行解，报最优
  A2 多起点坐标上升: E17 同网格 (21 点) / ≤3 轮，50 个随机起点
  A3 差分进化      : scipy differential_evolution, 12 维, bounds [0.1,10], maxiter=250, polish
  A4 成对坐标上升  : 66 个坐标对 × 21×21 联合网格（直接检验审稿人"XSS 与 BF 必须同时升降"的论断）
  另报无约束最优（A1 忽略 FPR），用于对照稿件 +0.2247 / FPR 0.0289

任务 B（C2 口径）：val 拟合 vs test 拟合，同 E17 协议（21 点网格、≤3 轮、目标 macro-F1、
  val/test benign FPR <= 0.008），量化 test-oracle 优势。

加速：p 为 softmax 输出且 w ∈ [0.1,10]，故 p_c < p_max/101 的类在任何 w 下都不可能成为 argmax
（严格不等式：0.1*p_max > 10*p_max/101）。把这些"无歧义"样本的贡献固化为混淆矩阵常数，
只对 ~48k 条歧义样本逐候选算 argmax，单次评估从 ~70 ms 降到 ~1.5 ms。
脚本启动时用精确全矩阵评估器校验快路径（含边界权重），差异 >1e-12 直接报错退出。

用法：python src/oracle_search_lfs.py
输出：results/oracle_search_lfs.json + results/oracle_search_lfs.md
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROBA = RES / "proba"

N = 12
BENIGN = 0
LESION = [9, 10, 11]
CAP = 0.008
GRID = np.exp(np.linspace(np.log(0.1), np.log(10.0), 21))   # 与 E17 完全相同
MAX_ROUNDS = 3
W_LO, W_HI = 0.1, 10.0
REL_THRESH = 1.0 / 101.0        # 无歧义判定阈值（见文件头证明）

BASE_LFS = 0.579084             # 已核对的 argmax 基线
SEED_RANDOM = 20240501
SEED_STARTS = 20240502
SEED_DE = 20240503
N_RANDOM = 20000
N_STARTS = 50
DE_MAXITER = 250

CLASS_NAMES = json.load(open(RES / "class_scheme_12.json", encoding="utf-8"))["class_order"]


# ---------------------------------------------------------------- 精确评估器
def cm_of(y, pred):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    return np.bincount(y * N + pred, minlength=N * N).reshape(N, N).astype(np.float64)


def metrics_from_cm(cm):
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros(N), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(N), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(N), where=(prec + rec) > 0)
    row0 = cm[BENIGN].sum()
    fpr = float((row0 - cm[BENIGN, BENIGN]) / row0) if row0 > 0 else 0.0
    return {"macro_f1": float(f1.mean()),
            "LFS": float(f1[LESION].sum()),
            "benign_fpr": fpr,
            "f1_per_class": [float(v) for v in f1]}


class Evaluator:
    """快路径评估器：无歧义样本固化为常数混淆矩阵，只对歧义样本逐候选 argmax。"""

    def __init__(self, p, y, name=""):
        self.name = name
        self.p = np.ascontiguousarray(p, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        mx = self.p.max(1, keepdims=True)
        keep = self.p >= mx * REL_THRESH
        k = keep.sum(1)
        fixed = k == 1
        amb = ~fixed
        pred_fixed = np.argmax(np.where(keep[fixed], self.p[fixed], -1.0), axis=1)
        self.fixed_cm = cm_of(self.y[fixed], pred_fixed)
        self.p_amb = np.ascontiguousarray(self.p[amb])
        self.y_amb12 = (self.y[amb] * N).astype(np.int32)
        self.amb_idx = np.where(amb)[0]
        self.n_fixed = int(fixed.sum())
        self.n_amb = int(amb.sum())
        self.n_benign = int((self.y == BENIGN).sum())
        self._buf = np.empty_like(self.p_amb)

    def cm(self, w):
        np.multiply(self.p_amb, w.astype(np.float32), out=self._buf)
        pred_amb = np.argmax(self._buf, axis=1).astype(np.int32)
        add = np.bincount(self.y_amb12 + pred_amb,
                          minlength=N * N).reshape(N, N).astype(np.float64)
        return self.fixed_cm + add

    def metrics(self, w):
        return metrics_from_cm(self.cm(w))

    def fpr(self, w):
        cm = self.cm(w)
        row0 = cm[BENIGN].sum()
        return float((row0 - cm[BENIGN, BENIGN]) / row0)

    # --- 精确全矩阵评估（校验 / 收尾用）
    def exact_metrics(self, w):
        pred = np.argmax(self.p.astype(np.float64) * w, axis=1)
        return metrics_from_cm(cm_of(self.y, pred))

    # --- BENIGN 权重扫描（约束强制段，精确）
    def scan_fpr_w0(self, w, w0_grid):
        """固定其余 11 维，精确计算各 w0 下的 benign FPR（只用 benign 行）。"""
        w = np.asarray(w, dtype=np.float64).copy()
        others = [c for c in range(N) if c != BENIGN]
        score_other = (self.p.astype(np.float64)[:, others] * w[others]).max(1)
        ben = self.y == BENIGN
        p0b = self.p[ben, BENIGN].astype(np.float64)
        so_b = score_other[ben]
        return np.array([float((w0 * p0b < so_b).mean()) for w0 in w0_grid])


# ---------------------------------------------------------------- 搜索组件
def force_feasible(ev, w, cap=CAP):
    """约束强制段：只抬 w0，找满足 FPR<=cap 的最小 w0（与 E17 同策略）。"""
    info = {}
    if ev.fpr(w) <= cap:
        return w, info
    w0_grid = np.geomspace(max(w[BENIGN], 1.0), 1e4, 61)
    fprs = ev.scan_fpr_w0(w, w0_grid)
    hit = np.where(fprs <= cap)[0]
    if len(hit) == 0:
        info["constraint_unreachable"] = True
        return w, info
    w2 = w.copy()
    w2[BENIGN] = float(w0_grid[hit[0]])
    info["enforced_w0"] = round(float(w2[BENIGN]), 4)
    return w2, info


def coord_ascent(ev, w0, objective, cap, max_rounds=MAX_ROUNDS, grid=GRID):
    """E17 同构坐标上升：逐类扫网格，先判约束，取目标改进最大者。"""
    w = np.asarray(w0, dtype=np.float64).copy()
    m = ev.metrics(w)
    cur = m[objective]
    info = {"rounds": 0, "infeasible_classes": 0}
    for rnd in range(1, max_rounds + 1):
        changed = False
        for c in range(N):
            best_v, best_g, all_violate = -np.inf, None, True
            for g in grid:
                if abs(g - w[c]) < 1e-15:
                    all_violate = False
                    continue
                wt = w.copy()
                wt[c] = g
                m = ev.metrics(wt)
                if cap is not None and m["benign_fpr"] > cap:
                    continue
                all_violate = False
                if m[objective] > best_v + 1e-12:
                    best_v, best_g = m[objective], g
            if all_violate:
                info["infeasible_classes"] += 1
                continue
            if best_g is not None and best_v > cur + 1e-9:
                w[c] = best_g
                cur = best_v
                changed = True
        info["rounds"] = rnd
        if not changed:
            break
    if cap is not None:
        w, extra = force_feasible(ev, w, cap)
        info.update(extra)
    info["fit_objective_value"] = round(float(ev.metrics(w)[objective]), 6)
    return w, info


def pair_ascent(ev, w0, objective, cap, max_rounds=2, grid=GRID):
    """成对坐标上升：66 个坐标对，21x21 联合网格（能走对角方向）。"""
    w = np.asarray(w0, dtype=np.float64).copy()
    cur = ev.metrics(w)[objective]
    pairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    info = {"rounds": 0, "n_pairs": len(pairs)}
    for rnd in range(1, max_rounds + 1):
        changed = False
        for a, b in pairs:
            best_v, best_ga, best_gb = -np.inf, None, None
            wa0, wb0 = w[a], w[b]
            for ga in grid:
                for gb in grid:
                    if abs(ga - wa0) < 1e-15 and abs(gb - wb0) < 1e-15:
                        continue
                    wt = w.copy()
                    wt[a], wt[b] = ga, gb
                    m = ev.metrics(wt)
                    if cap is not None and m["benign_fpr"] > cap:
                        continue
                    if m[objective] > best_v + 1e-12:
                        best_v, best_ga, best_gb = m[objective], ga, gb
            if best_ga is not None and best_v > cur + 1e-9:
                w[a], w[b] = best_ga, best_gb
                cur = best_v
                changed = True
        info["rounds"] = rnd
        if not changed:
            break
    if cap is not None:
        w, extra = force_feasible(ev, w, cap)
        info.update(extra)
    info["fit_objective_value"] = round(float(ev.metrics(w)[objective]), 6)
    return w, info


def w_info(ev, w, base_lfs=BASE_LFS):
    m = ev.metrics(w)
    nonunit = [{"class": int(c), "name": CLASS_NAMES[c], "w": round(float(w[c]), 4)}
               for c in range(N) if abs(w[c] - 1.0) > 1e-9]
    return {"LFS": round(m["LFS"], 6),
            "delta_LFS": round(m["LFS"] - base_lfs, 6),
            "benign_fpr": round(m["benign_fpr"], 6),
            "macro_f1": round(m["macro_f1"], 6),
            "feasible": bool(m["benign_fpr"] <= CAP),
            "n_nonunit_w": len(nonunit),
            "nonunit_w": nonunit,
            "w": [round(float(x), 4) for x in w]}


# ---------------------------------------------------------------- 任务 A
def task_a(ev, t0):
    out = {"baseline": None, "random_search": None, "random_unconstrained": None,
           "multistart_coord": None, "differential_evolution": None,
           "pair_ascent": None}
    base = ev.metrics(np.ones(N))
    out["baseline"] = {"LFS": round(base["LFS"], 6), "benign_fpr": round(base["benign_fpr"], 6),
                       "macro_f1": round(base["macro_f1"], 6),
                       "note": "argmax w=1；FPR>CAP 说明基线本身不可行，任何可行解都必须抬 w0"}
    print(f"[{time.time()-t0:6.0f}s] baseline argmax: LFS={base['LFS']:.6f} "
          f"FPR={base['benign_fpr']:.6f} macroF1={base['macro_f1']:.6f}", flush=True)

    # --- A1 随机搜索
    rng = np.random.default_rng(SEED_RANDOM)
    W = 10.0 ** rng.uniform(np.log10(W_LO), np.log10(W_HI), size=(N_RANDOM, N))
    best_feas, best_free = None, None
    n_feas = 0
    t = time.time()
    for w in W:
        m = ev.metrics(w)
        if best_free is None or m["LFS"] > best_free[0]:
            best_free = (m["LFS"], w.copy(), m["benign_fpr"])
        if m["benign_fpr"] <= CAP:
            n_feas += 1
            if best_feas is None or m["LFS"] > best_feas[0]:
                best_feas = (m["LFS"], w.copy(), m["benign_fpr"])
    print(f"[{time.time()-t0:6.0f}s] A1 random: {N_RANDOM} 组, 可行 {n_feas} "
          f"({n_feas/N_RANDOM:.1%}), 用时 {time.time()-t:.0f}s; "
          f"best feasible LFS={best_feas[0]:.4f} FPR={best_feas[2]:.4f}; "
          f"best unconstrained LFS={best_free[0]:.4f} FPR={best_free[2]:.4f}", flush=True)
    out["random_search"] = {**w_info(ev, best_feas[1]),
                            "n_samples": N_RANDOM, "n_feasible": int(n_feas),
                            "seed": SEED_RANDOM, "elapsed_sec": round(time.time() - t, 1)}
    out["random_unconstrained"] = w_info(ev, best_free[1])

    # --- A2 多起点坐标上升
    rng = np.random.default_rng(SEED_STARTS)
    starts = 10.0 ** rng.uniform(np.log10(W_LO), np.log10(W_HI), size=(N_STARTS, N))
    t = time.time()
    runs = []
    for i, s in enumerate(starts):
        w_s, _ = force_feasible(ev, s.copy(), CAP)
        w, info = coord_ascent(ev, w_s, "LFS", CAP)
        m = ev.metrics(w)
        runs.append({"start_idx": i, "LFS": m["LFS"], "benign_fpr": m["benign_fpr"],
                     "rounds": info["rounds"], "w": w.copy()})
    runs_sorted = sorted(runs, key=lambda r: -r["LFS"])
    best = runs_sorted[0]
    lfs_all = np.array([r["LFS"] for r in runs])
    print(f"[{time.time()-t0:6.0f}s] A2 multistart: {N_STARTS} 起点, 用时 {time.time()-t:.0f}s; "
          f"best LFS={best['LFS']:.4f} FPR={best['benign_fpr']:.4f}; "
          f"LFS 分布 min/median/max = {lfs_all.min():.4f}/{np.median(lfs_all):.4f}/{lfs_all.max():.4f}",
          flush=True)
    out["multistart_coord"] = {**w_info(ev, best["w"]),
                               "n_starts": N_STARTS, "seed": SEED_STARTS,
                               "LFS_min": round(float(lfs_all.min()), 6),
                               "LFS_median": round(float(np.median(lfs_all)), 6),
                               "LFS_max": round(float(lfs_all.max()), 6),
                               "n_distinct_LFS": int(len(np.unique(np.round(lfs_all, 6)))),
                               "elapsed_sec": round(time.time() - t, 1)}
    best_ms_w = best["w"]

    # --- A3 差分进化
    t = time.time()
    calls = {"n": 0}

    def neg_obj(x):
        calls["n"] += 1
        m = ev.metrics(x)
        viol = max(0.0, m["benign_fpr"] - CAP)
        return -(m["LFS"] - 1000.0 * viol)

    de = differential_evolution(neg_obj, bounds=[(W_LO, W_HI)] * N, maxiter=DE_MAXITER,
                               popsize=15, tol=1e-8, mutation=(0.5, 1.0), recombination=0.7,
                               polish=True, seed=SEED_DE, workers=1, updating="immediate",
                               init="latinhypercube")
    w_de, _ = force_feasible(ev, np.asarray(de.x, dtype=np.float64), CAP)
    print(f"[{time.time()-t0:6.0f}s] A3 DE: {calls['n']} 次评估, 用时 {time.time()-t:.0f}s; "
          f"raw best obj={de.fun:.6f} -> LFS={ev.metrics(w_de)['LFS']:.4f} "
          f"FPR={ev.metrics(w_de)['benign_fpr']:.4f}", flush=True)
    out["differential_evolution"] = {**w_info(ev, w_de),
                                     "maxiter": DE_MAXITER, "popsize": 15,
                                     "n_eval": calls["n"], "seed": SEED_DE,
                                     "raw_de_fun": round(float(de.fun), 6),
                                     "elapsed_sec": round(time.time() - t, 1)}

    # --- A4 成对坐标上升（从 A2 最优出发，也可看作对单坐标搜索的结构性反驳）
    t = time.time()
    w_pair, info_pair = pair_ascent(ev, best_ms_w.copy(), "LFS", CAP)
    print(f"[{time.time()-t0:6.0f}s] A4 pair-ascent: 用时 {time.time()-t:.0f}s; "
          f"LFS={ev.metrics(w_pair)['LFS']:.4f} FPR={ev.metrics(w_pair)['benign_fpr']:.4f} "
          f"rounds={info_pair['rounds']}", flush=True)
    out["pair_ascent"] = {**w_info(ev, w_pair), "protocol": "66 pairs x 21x21 grid, <=2 rounds",
                          "rounds": info_pair["rounds"], "elapsed_sec": round(time.time() - t, 1)}
    return out


# ---------------------------------------------------------------- 任务 B
def task_b(ev_val, ev_test, t0):
    """C2 口径：val 拟合 vs test 拟合（同 E17 协议，目标 macro-F1）。"""
    base_val = ev_val.metrics(np.ones(N))
    base_test = ev_test.metrics(np.ones(N))

    # val 拟合 -> test 报告（合法口径）
    w_val, info_val = coord_ascent(ev_val, np.ones(N), "macro_f1", CAP)
    # test 拟合 -> test 报告（oracle 口径）
    w_test, info_test = coord_ascent(ev_test, np.ones(N), "macro_f1", CAP)

    def rep(w):
        return w_info(ev_test, w)

    diff = np.asarray(w_val) - np.asarray(w_test)
    out = {
        "protocol": "E17 同构：21 点网格 exp(linspace(log0.1,log10,21))、<=3 轮坐标上升、"
                    "目标 macro-F1、硬约束 benign FPR<=0.008、拟合集上做约束强制段",
        "argmax_val": {"macro_f1": round(base_val["macro_f1"], 6),
                       "LFS": round(base_val["LFS"], 6),
                       "benign_fpr": round(base_val["benign_fpr"], 6)},
        "argmax_test": {"macro_f1": round(base_test["macro_f1"], 6),
                        "LFS": round(base_test["LFS"], 6),
                        "benign_fpr": round(base_test["benign_fpr"], 6)},
        "val_fit": {**rep(w_val), "fit_info": info_val},
        "test_fit": {**rep(w_test), "fit_info": info_test},
        "gap_test_minus_val_fit": {
            "macro_f1": round(rep(w_test)["macro_f1"] - rep(w_val)["macro_f1"], 6),
            "LFS": round(rep(w_test)["LFS"] - rep(w_val)["LFS"], 6),
            "benign_fpr": round(rep(w_test)["benign_fpr"] - rep(w_val)["benign_fpr"], 6)},
        "weight_diff": {
            "n_coords_diff": int((np.abs(diff) > 1e-9).sum()),
            "per_class": [{"class": c, "name": CLASS_NAMES[c],
                           "w_val_fit": round(float(w_val[c]), 4),
                           "w_test_fit": round(float(w_test[c]), 4),
                           "diff": round(float(diff[c]), 4)}
                          for c in range(N) if abs(diff[c]) > 1e-9]},
    }
    print(f"[{time.time()-t0:6.0f}s] B: val-fit test macroF1={out['val_fit']['macro_f1']:.4f} "
          f"LFS={out['val_fit']['LFS']:.4f} FPR={out['val_fit']['benign_fpr']:.4f} | "
          f"test-fit test macroF1={out['test_fit']['macro_f1']:.4f} "
          f"LFS={out['test_fit']['LFS']:.4f} FPR={out['test_fit']['benign_fpr']:.4f} | "
          f"gap macro={out['gap_test_minus_val_fit']['macro_f1']:+.4f}", flush=True)
    return out


# ---------------------------------------------------------------- 校验
def validate_fast_path(ev, rng):
    cands = [np.ones(N)]
    for _ in range(6):
        cands.append(10.0 ** rng.uniform(np.log10(W_LO), np.log10(W_HI), size=N))
    cands.append(np.where(np.arange(N) % 2 == 0, W_HI, W_LO))
    cands.append(np.where(np.arange(N) % 2 == 0, W_LO, W_HI))
    worst = 0.0
    for w in cands:
        a = ev.metrics(w)
        b = ev.exact_metrics(w)
        for k in ("LFS", "benign_fpr", "macro_f1"):
            worst = max(worst, abs(a[k] - b[k]))
    return worst


def main():
    t0 = time.time()
    p_te = np.load(PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_test.npy")
    y_te = np.load(PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_test_ytrue.npy")
    p_va = np.load(PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_val.npy")
    y_va = np.load(PROBA / "deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_val_ytrue.npy")

    ev_te = Evaluator(p_te, y_te, "test")
    ev_va = Evaluator(p_va, y_va, "val")
    print(f"data: test {ev_te.p.shape} (无歧义 {ev_te.n_fixed}, 歧义 {ev_te.n_amb}) | "
          f"val {ev_va.p.shape} (无歧义 {ev_va.n_fixed}, 歧义 {ev_va.n_amb})", flush=True)

    rng = np.random.default_rng(7)
    for ev in (ev_te, ev_va):
        worst = validate_fast_path(ev, rng)
        print(f"validate fast path [{ev.name}]: max|diff| = {worst:.3e}", flush=True)
        if worst > 1e-9:
            raise SystemExit(f"快路径校验失败（{ev.name}, {worst}），停止。")

    base = ev_te.metrics(np.ones(N))
    if abs(base["LFS"] - BASE_LFS) > 1e-4:
        raise SystemExit(f"基线 LFS 复现失败: {base['LFS']} != {BASE_LFS}")
    print(f"基线复现 OK: LFS={base['LFS']:.6f} (= {BASE_LFS}), "
          f"benign FPR={base['benign_fpr']:.6f}, macroF1={base['macro_f1']:.6f}", flush=True)

    res = {"meta": {"script": "src/oracle_search_lfs.py",
                    "data": {"test": "results/proba/deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_test.npy",
                             "val": "results/proba/deep_cnn_bilstm_attn_seed42_12class_ce_none_a3p_val.npy"},
                    "n_test": int(len(y_te)), "n_val": int(len(y_va)),
                    "class_order": CLASS_NAMES, "cap_benign_fpr": CAP,
                    "w_range": [W_LO, W_HI], "grid_points": int(len(GRID)),
                    "objective": "LFS = F1[9]+F1[10]+F1[11]"},
           "task_a": task_a(ev_te, t0),
           "task_b": task_b(ev_va, ev_te, t0)}
    res["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(RES / "oracle_search_lfs.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    # ---------------- markdown
    A, B = res["task_a"], res["task_b"]
    L = ["# 真正的 oracle 上界搜索 + C2 口径差异", "",
         f"数据：`{res['meta']['data']['test']}`（test 段 n={res['meta']['n_test']:,}，"
         f"未处理模型，纯 CE，无重采样）；val 段 n={res['meta']['n_val']:,}。",
         f"模型 `pred = argmax_c(w_c·p_c)`，`w ∈ [{W_LO},{W_HI}]^12`；目标 LFS = F1(Bot)+F1(BruteForce)+F1(XSS)；"
         f"硬约束 benign FPR ≤ {CAP}。", "",
         "## 0. 基线复现（w ≡ 1）", "",
         "| 指标 | test | val |", "|---|---|---|",
         f"| LFS | {A['baseline']['LFS']:.4f} | {B['argmax_val']['LFS']:.4f} |",
         f"| macro-F1 | {A['baseline']['macro_f1']:.4f} | {B['argmax_val']['macro_f1']:.4f} |",
         f"| benign FPR | {A['baseline']['benign_fpr']:.4f} | {B['argmax_val']['benign_fpr']:.4f} |",
         "",
         f"基线 FPR {A['baseline']['benign_fpr']:.4f} > {CAP}，即**均匀权重本身不可行**；"
         "任何可行解都必须抬高 BENIGN 权重（这也是 E17 那一步 `enforced_w0` 的全部内容）。", "",
         "## 1. 任务 A：搜索到的可行最优（test 段 = oracle 口径）", "",
         "| 搜索 | 最优 LFS | ΔLFS vs 0.5791 | benign FPR | 非单位权重 | 非单位权重明细 | 耗时 |",
         "|---|---|---|---|---|---|---|"]
    def wdesc(r):
        if r["n_nonunit_w"] == 0:
            return "无（全 1）"
        return "; ".join(f"{d['name']}={d['w']}" for d in r["nonunit_w"])
    rows = [("A1 随机搜索 (20,000 组)", A["random_search"]),
            ("A2 多起点坐标上升 (50 起点)", A["multistart_coord"]),
            ("A3 差分进化 (maxiter=250)", A["differential_evolution"]),
            ("A4 成对坐标上升 (66 对, 21×21)", A["pair_ascent"])]
    for name, r in rows:
        L.append(f"| {name} | {r['LFS']:.4f} | {r['delta_LFS']:+.4f} | {r['benign_fpr']:.4f} | "
                 f"{r['n_nonunit_w']} | {wdesc(r)} | {r.get('elapsed_sec','-')} s |")
    u = A["random_unconstrained"]
    L += ["", f"**无约束最优（随机搜索，忽略 FPR 约束）**：LFS = {u['LFS']:.4f}"
              f"（ΔLFS = {u['delta_LFS']:+.4f}），benign FPR = {u['benign_fpr']:.4f}。"
              "对照稿件现有 +0.2247 / FPR 0.0289。", "",
          f"A2 多起点 LFS 分布：min {A['multistart_coord']['LFS_min']:.4f} / "
          f"median {A['multistart_coord']['LFS_median']:.4f} / max {A['multistart_coord']['LFS_max']:.4f}；"
          f"不同终点值 {A['multistart_coord']['n_distinct_LFS']} 个。", "",
          "## 2. 任务 B：C2 口径（val 拟合 vs test 拟合，目标 macro-F1，协议同 E17）", "",
          "| 口径 | test macro-F1 | test LFS | test benign FPR | 非单位权重 | 权重明细 |",
          "|---|---|---|---|---|---|"]
    for nm, key in (("val 拟合（合法口径）", "val_fit"), ("test 拟合（oracle 口径）", "test_fit")):
        r = B[key]
        L.append(f"| {nm} | {r['macro_f1']:.4f} | {r['LFS']:.4f} | {r['benign_fpr']:.4f} | "
                 f"{r['n_nonunit_w']} | {wdesc(r)} |")
    g = B["gap_test_minus_val_fit"]
    L += ["", f"口径差（test 拟合 − val 拟合）：macro-F1 {g['macro_f1']:+.4f}，"
              f"LFS {g['LFS']:+.4f}，benign FPR {g['benign_fpr']:+.4f}；"
              f"权重差异坐标数 {B['weight_diff']['n_coords_diff']}。", "",
          f"总耗时 {res['meta']['elapsed_sec']} s。", ""]
    (RES / "oracle_search_lfs.md").write_text("\n".join(L), encoding="utf-8")
    print("written:", RES / "oracle_search_lfs.json", "and", RES / "oracle_search_lfs.md", flush=True)


if __name__ == "__main__":
    main()
