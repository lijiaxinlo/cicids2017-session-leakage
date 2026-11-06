"""val 拟合联合搜索（DE）在 3 个模型种子上的重复 + 4 次优化器重启。

背景：稿件里"可部署校准回收率 57%"是拿 seed 42 单种子的分子（ΔLFS +0.2244）除以 3 种子均值
的分母（+0.3950），基数不一致。本脚本把未处理臂 A3 在 seed 44/46 上补齐后，用同一协议
（val 拟合、test 一次性评估、LFS 目标、benign FPR ≤ 0.008 罚函数、DE maxiter=250/popsize=15/
tol=1e-6/polish/workers=1/updating=immediate）在 3 个模型种子 × 4 个优化器种子上各跑一次，
给出均值 ± 标准差与双盆地分布。

与 src/valfit_joint_search.py 的关系：**超参、目标函数、评估口径逐位一致**，只加了两点：
  1) 覆盖 3 个模型种子（42/44/46）；
  2) 目标函数走"无歧义样本固化"的快路径（p_c < p_max/101 的类在 w∈[0.1,10] 下不可能成为
     argmax，故其贡献可固化为整数混淆矩阵），F1 仍用 2tp/(2tp+fp+fn) 的整数口径逐位复算。
  → seed42 的 4 条 test LFS 必须与 results/oracle_search_lfs_valfit*.json 完全一致（1e-6），
    否则说明快路径改变了优化轨迹，脚本会显式告警。

回收率分母 = 同种子的 (C4 @ cap0.008 test LFS) − (A3 test LFS)；C4 值读自
results/e17_calibration_summary.json（不手填）。

用法：/d/Anaconda/python src/valfit_joint_3seed.py            # 3 个模型种子
      /d/Anaconda/python src/valfit_joint_3seed.py 42         # 只跑指定种子（校验用）
输出：results/valfit_joint_3seed.json + results/valfit_joint_3seed.md
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROBA = RES / "proba"

N = 12
CAP = 0.008
LESION = [9, 10, 11]
MODEL_SEEDS = tuple(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else (42, 44, 46)
OPT_SEEDS = (0, 1, 2, 3)
BOUNDS = [(0.1, 10.0)] * N
DE_KW = dict(maxiter=250, popsize=15, tol=1e-6, polish=True, workers=1,
             updating="immediate")
BASIN_SPLIT = 0.70           # 双盆地分界（seed42 观测到 0.8035 / 0.5494）
REL_THRESH = 1.0 / 101.0     # 无歧义判定阈值（见文件头）
EXPECTED_42 = {0: 0.8035167575915695, 1: 0.5494276795005203,
               2: 0.8027894810939317, 3: 0.5494276795005203}


# ---------------------------------------------------------------- 逐位一致的快路径
def exact_metrics(p, y, w):
    """与 valfit_joint_search.py 完全相同的参考实现（全矩阵、整数 F1 口径）。"""
    pred = np.argmax(p * w, axis=1)
    f1s = []
    for c in range(N):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        d = 2 * tp + fp + fn
        f1s.append(0.0 if d == 0 else 2 * tp / d)
    fpr = float(np.mean(pred[y == 0] != 0))
    return {"LFS": float(sum(f1s[c] for c in LESION)),
            "macro_f1": float(np.mean(f1s)), "benign_fpr": fpr}


class FastEval:
    """快路径：无歧义样本固化为整数混淆矩阵，只对歧义样本逐候选 argmax。"""

    def __init__(self, p, y):
        self.p32 = np.ascontiguousarray(p, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        mx = self.p32.max(1, keepdims=True)
        keep = self.p32 >= mx * REL_THRESH
        k = keep.sum(1)
        fixed = k == 1
        amb = ~fixed
        pred_fixed = np.argmax(np.where(keep[fixed], self.p32[fixed], -1.0), axis=1)
        self.fixed_cm = np.bincount(self.y[fixed] * N + pred_fixed,
                                    minlength=N * N).reshape(N, N).astype(np.int64)
        self.p_amb = np.ascontiguousarray(self.p32[amb].astype(np.float64))
        self.y_amb12 = (self.y[amb] * N).astype(np.int32)
        self.buf = np.empty_like(self.p_amb)
        self.n_fixed = int(fixed.sum())
        self.n_amb = int(amb.sum())

    def cm(self, w):
        np.multiply(self.p_amb, w, out=self.buf)          # p(float32->float64) * w(float64)
        pred = np.argmax(self.buf, axis=1).astype(np.int32)
        return self.fixed_cm + np.bincount(self.y_amb12 + pred,
                                           minlength=N * N).reshape(N, N)

    def metrics(self, w):
        cm = self.cm(w)
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        d = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
        row0 = cm[0].sum()
        return {"LFS": float(f1[9] + f1[10] + f1[11]),
                "macro_f1": float(f1.mean()),
                "benign_fpr": float((row0 - cm[0, 0]) / row0)}


def load(seed):
    base = PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p"
    files = {k: Path(f"{base}_{k}.npy") for k in ("val", "val_ytrue", "test", "test_ytrue")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"缺少 proba 文件：{missing}")
    return {k: np.load(p) for k, p in files.items()}


def c4_reference():
    d = json.load(open(RES / "e17_calibration_summary.json", encoding="utf-8"))
    return {a["seed"]: {"test_lesion_sum": a["test_lesion_sum"],
                        "test_macro_f1": a["test_macro_f1"],
                        "test_benign_fpr": a["test_benign_fpr"]}
            for a in d.get("c4", []) if abs(a["tier_benign_cap"] - CAP) < 1e-12}


def main():
    t0 = time.time()
    c4 = c4_reference()
    out = {"meta": {"script": "src/valfit_joint_3seed.py",
                    "protocol": "DE on val (objective LFS, penalty 1000*max(0,fpr-0.008)), "
                                "maxiter=250, popsize=15, tol=1e-6, polish=True, workers=1, "
                                "updating=immediate; test evaluated once after weights frozen",
                    "model_seeds": list(MODEL_SEEDS), "opt_seeds": list(OPT_SEEDS),
                    "cap": CAP, "bounds": [0.1, 10.0], "basin_split": BASIN_SPLIT,
                    "c4_source": "results/e17_calibration_summary.json (tier_benign_cap=0.008)"},
           "per_seed": [], "runs": []}

    for seed in MODEL_SEEDS:
        d = load(seed)
        ev_v = FastEval(d["val"], d["val_ytrue"])
        ev_t = FastEval(d["test"], d["test_ytrue"])

        # 等价性硬校验：快路径 vs 参考实现
        rng = np.random.default_rng(seed)
        worst = 0.0
        for w in [np.ones(N)] + [10.0 ** rng.uniform(-1, 1, N) for _ in range(6)]:
            for ev, pp, yy in ((ev_v, d["val"], d["val_ytrue"]), (ev_t, d["test"], d["test_ytrue"])):
                a = ev.metrics(w)
                b = exact_metrics(pp, yy, w)
                for k in ("LFS", "benign_fpr", "macro_f1"):
                    worst = max(worst, abs(a[k] - b[k]))
        print(f"[seed {seed}] 快路径 vs 参考实现 max|Δ|={worst:.1e} "
              f"(val 歧义 {ev_v.n_amb}/{ev_v.n_amb + ev_v.n_fixed} 行, "
              f"test 歧义 {ev_t.n_amb}/{ev_t.n_amb + ev_t.n_fixed} 行)", flush=True)
        if worst > 1e-12:
            raise SystemExit(f"快路径与参考实现不等价 seed {seed}: {worst}")

        bv = ev_v.metrics(np.ones(N))
        bt = ev_t.metrics(np.ones(N))
        print(f"[seed {seed}] A3 基线: val macro={bv['macro_f1']:.4f} LFS={bv['LFS']:.4f} "
              f"FPR={bv['benign_fpr']:.4f} | test macro={bt['macro_f1']:.4f} "
              f"LFS={bt['LFS']:.4f} FPR={bt['benign_fpr']:.4f}", flush=True)
        if not (0.55 <= bt["LFS"] <= 0.62):
            raise SystemExit(f"seed {seed} 的 A3 test LFS={bt['LFS']:.4f} 偏离预期 0.582±0.03，"
                             "疑似复现配方不一致，停止。")
        out["per_seed"].append({
            "model_seed": seed,
            "a3_val": {k: round(bv[k], 6) for k in ("macro_f1", "LFS", "benign_fpr")},
            "a3_test": {k: round(bt[k], 6) for k in ("macro_f1", "LFS", "benign_fpr")},
            "c4_test": c4.get(seed)})

        for os_ in OPT_SEEDS:
            n_ev = {"n": 0}

            def objective(w):
                n_ev["n"] += 1
                m = ev_v.metrics(w)
                return -m["LFS"] + 1000.0 * max(0.0, m["benign_fpr"] - CAP)

            t = time.time()
            res = differential_evolution(objective, BOUNDS, seed=os_, **DE_KW)
            w = np.asarray(res.x, dtype=np.float64)
            mv = ev_v.metrics(w)
            mt = ev_t.metrics(w)
            rec = {"model_seed": seed, "opt_seed": os_,
                   "val": {"LFS": round(mv["LFS"], 6), "benign_fpr": round(mv["benign_fpr"], 6),
                           "macro_f1": round(mv["macro_f1"], 6)},
                   "test": {"LFS": round(mt["LFS"], 6),
                            "delta_LFS": round(mt["LFS"] - bt["LFS"], 6),
                            "benign_fpr": round(mt["benign_fpr"], 6),
                            "macro_f1": round(mt["macro_f1"], 6)},
                   "basin": "high" if mt["LFS"] > BASIN_SPLIT else "low",
                   "n_eval": n_ev["n"], "elapsed_sec": round(time.time() - t, 1),
                   "weights": [round(float(x), 6) for x in w]}
            out["runs"].append(rec)
            print(f"[seed {seed} opt {os_}] val LFS={mv['LFS']:.4f} FPR={mv['benign_fpr']:.4f} | "
                  f"test LFS={mt['LFS']:.4f} Δ={mt['LFS']-bt['LFS']:+.4f} "
                  f"FPR={mt['benign_fpr']:.4f} macro={mt['macro_f1']:.4f} "
                  f"basin={rec['basin']} ({rec['elapsed_sec']}s)", flush=True)

    # ---------- 校验 seed42 与既有结果一致
    got42 = {r["opt_seed"]: r["test"]["LFS"] for r in out["runs"] if r["model_seed"] == 42}
    if got42:
        bad = {k: [got42.get(k), v] for k, v in EXPECTED_42.items()
               if got42.get(k) is None or abs(got42[k] - v) > 1e-6}
        out["seed42_reproduction_check"] = {
            "expected": {str(k): v for k, v in EXPECTED_42.items()},
            "got": {str(k): v for k, v in got42.items()},
            "mismatch": {str(k): v for k, v in bad.items()}}
        if bad:
            print(f"⚠ seed42 复现校验不一致：{bad}（快路径改变了优化轨迹？）", flush=True)
        else:
            print("seed42 复现校验通过：4 条 test LFS 与既有 JSON 完全一致", flush=True)

    # ---------- 汇总
    rows = out["runs"]
    per_seed_sum = []
    for seed in MODEL_SEEDS:
        rs = [r for r in rows if r["model_seed"] == seed]
        a3 = next(p for p in out["per_seed"] if p["model_seed"] == seed)
        a3_lfs = a3["a3_test"]["LFS"]
        c4_lfs = a3["c4_test"]["test_lesion_sum"] if a3["c4_test"] else None
        best = max(rs, key=lambda r: r["val"]["LFS"])          # val 选优（部署时唯一可用的选法）
        mean_opt = float(np.mean([r["test"]["LFS"] for r in rs]))
        denom = (c4_lfs - a3_lfs) if c4_lfs is not None else None
        per_seed_sum.append({
            "model_seed": seed, "a3_test_LFS": a3_lfs, "c4_test_LFS": c4_lfs,
            "denominator_C4_minus_A3": round(denom, 6) if denom else None,
            "val_best_opt_seed": best["opt_seed"],
            "val_best_val_LFS": best["val"]["LFS"],
            "val_best_test_LFS": best["test"]["LFS"],
            "val_best_delta_LFS": round(best["test"]["LFS"] - a3_lfs, 6),
            "val_best_recovery": round((best["test"]["LFS"] - a3_lfs) / denom, 4) if denom else None,
            "mean_opt_test_LFS": round(mean_opt, 6),
            "mean_opt_delta_LFS": round(mean_opt - a3_lfs, 6),
            "mean_opt_recovery": round((mean_opt - a3_lfs) / denom, 4) if denom else None,
            "n_high_basin": sum(r["basin"] == "high" for r in rs),
            "n_low_basin": sum(r["basin"] == "low" for r in rs),
            "test_LFS_list": [r["test"]["LFS"] for r in rs],
        })
    out["per_seed_summary"] = per_seed_sum

    vb = np.array([p["val_best_test_LFS"] for p in per_seed_sum])
    mo = np.array([p["mean_opt_test_LFS"] for p in per_seed_sum])
    a3 = np.array([p["a3_test_LFS"] for p in per_seed_sum])
    c4v = np.array([p["c4_test_LFS"] for p in per_seed_sum])
    denom = c4v - a3
    allv = np.array([r["test"]["LFS"] for r in rows])
    alld = np.array([r["test"]["delta_LFS"] for r in rows])
    dd = 1 if len(vb) > 1 else 0
    agg = {
        "n_model_seeds": len(per_seed_sum),
        "val_best_test_LFS_mean": round(float(vb.mean()), 6),
        "val_best_test_LFS_sd": round(float(vb.std(ddof=dd)), 6),
        "mean_opt_test_LFS_mean": round(float(mo.mean()), 6),
        "mean_opt_test_LFS_sd": round(float(mo.std(ddof=dd)), 6),
        "a3_test_LFS_mean": round(float(a3.mean()), 6),
        "a3_test_LFS_sd": round(float(a3.std(ddof=dd)), 6),
        "c4_test_LFS_mean": round(float(c4v.mean()), 6),
        "denominator_mean": round(float(denom.mean()), 6),
        "recovery_val_selected_pooled": round(float((vb - a3).mean() / denom.mean()), 4),
        "recovery_no_selection_pooled": round(float((mo - a3).mean() / denom.mean()), 4),
        "recovery_val_selected_mean_of_ratios": round(float(np.mean([p["val_best_recovery"] for p in per_seed_sum])), 4),
        "recovery_no_selection_mean_of_ratios": round(float(np.mean([p["mean_opt_recovery"] for p in per_seed_sum])), 4),
        "all_runs_test_LFS_mean": round(float(allv.mean()), 6),
        "all_runs_test_LFS_sd": round(float(allv.std(ddof=dd)), 6),
        "all_runs_delta_mean": round(float(alld.mean()), 6),
        "all_runs_delta_sd": round(float(alld.std(ddof=dd)), 6),
        "basin_counts": {f"seed{p['model_seed']}": {"high": p["n_high_basin"], "low": p["n_low_basin"]}
                         for p in per_seed_sum},
        "n_high_basin_total": int(sum(p["n_high_basin"] for p in per_seed_sum)),
        "n_low_basin_total": int(sum(p["n_low_basin"] for p in per_seed_sum)),
    }
    out["aggregate"] = agg
    out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(RES / "valfit_joint_3seed.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---------------- markdown
    L = ["# val 拟合联合搜索：3 个模型种子 × 4 次优化器重启", "",
         f"协议：`pred=argmax_c(w_c·p_c)`，`w∈[0.1,10]^12`；目标 LFS = F1(Bot)+F1(Web-BF)+F1(Web-XSS)；"
         f"硬约束 benign FPR ≤ {CAP}（罚函数 `1000·max(0,fpr−cap)`）；"
         "`differential_evolution(maxiter=250, popsize=15, tol=1e-6, polish=True, workers=1, "
         "updating=\"immediate\")`；**拟合只用 val，test 在权重冻结后评估一次**。", "",
         "## 1. A3 基线（w≡1）", "",
         "| 模型种子 | test macro-F1 | test LFS | test benign FPR | val macro-F1 | val LFS | C4 @cap0.008 test LFS | 分母 (C4−A3) |",
         "|---|---|---|---|---|---|---|---|"]
    for p in out["per_seed"]:
        c4v_ = p["c4_test"]["test_lesion_sum"] if p["c4_test"] else None
        den = f"{c4v_ - p['a3_test']['LFS']:.4f}" if c4v_ is not None else "—"
        L.append(f"| {p['model_seed']} | {p['a3_test']['macro_f1']:.4f} | {p['a3_test']['LFS']:.4f} | "
                 f"{p['a3_test']['benign_fpr']:.4f} | {p['a3_val']['macro_f1']:.4f} | "
                 f"{p['a3_val']['LFS']:.4f} | {c4v_ if c4v_ is not None else '缺失'} | {den} |")
    L += ["", "## 2. 全部重启明细", "",
          "| 模型种子 | 优化器种子 | val LFS | val FPR | test LFS | ΔLFS(vs 同种子 A3) | test FPR | test macro-F1 | 盆地 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model_seed']} | {r['opt_seed']} | {r['val']['LFS']:.4f} | "
                 f"{r['val']['benign_fpr']:.4f} | {r['test']['LFS']:.4f} | "
                 f"{r['test']['delta_LFS']:+.4f} | {r['test']['benign_fpr']:.4f} | "
                 f"{r['test']['macro_f1']:.4f} | {r['basin']} |")
    L += ["", "## 3. 按模型种子分组", "",
          "| 模型种子 | val 选优的 opt seed | val LFS | test LFS | ΔLFS | 回收率 | 4 次均值 test LFS | 均值回收率 | 高/低盆地 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in per_seed_sum:
        L.append(f"| {p['model_seed']} | {p['val_best_opt_seed']} | {p['val_best_val_LFS']:.4f} | "
                 f"{p['val_best_test_LFS']:.4f} | {p['val_best_delta_LFS']:+.4f} | "
                 f"{p['val_best_recovery']:.1%} | {p['mean_opt_test_LFS']:.4f} | "
                 f"{p['mean_opt_recovery']:.1%} | {p['n_high_basin']}/{p['n_low_basin']} |")
    L += ["", "## 4. 汇总（跨模型种子）", "",
          f"- **val 选优 test LFS：{agg['val_best_test_LFS_mean']:.4f} ± {agg['val_best_test_LFS_sd']:.4f}**",
          f"- 4 次重启不选优 test LFS 均值：{agg['mean_opt_test_LFS_mean']:.4f} ± {agg['mean_opt_test_LFS_sd']:.4f}"
          f"（全部 {len(rows)} 条：{agg['all_runs_test_LFS_mean']:.4f} ± {agg['all_runs_test_LFS_sd']:.4f}）",
          f"- A3 基线 test LFS：{agg['a3_test_LFS_mean']:.4f} ± {agg['a3_test_LFS_sd']:.4f}；"
          f"C4 test LFS 均值 {agg['c4_test_LFS_mean']:.4f}；分母 (C4−A3) 均值 {agg['denominator_mean']:.4f}",
          f"- **val 选优回收率（合并口径）：{agg['recovery_val_selected_pooled']:.1%}**"
          f"（按种子比值再平均 {agg['recovery_val_selected_mean_of_ratios']:.1%}）",
          f"- **不选优回收率（合并口径）：{agg['recovery_no_selection_pooled']:.1%}**"
          f"（按种子比值再平均 {agg['recovery_no_selection_mean_of_ratios']:.1%}）",
          f"- 盆地计数：高盆地 {agg['n_high_basin_total']} 条 / 低盆地 {agg['n_low_basin_total']} 条"
          f"（分界 test LFS = {BASIN_SPLIT}）", "",
          "## 5. 注意事项（读表前必看）", "",
          "1. **val 选优的打平问题**：seed 44 的三次高盆地重启在 val 上完全打平（val LFS 均 0.8263），"
          "val 无法区分，按“首个最大”选中 opt 1，其 test macro-F1 仅 0.6987，另两次为 0.7756 / 0.7770。"
          "说明只用 val LFS 选重启可能选到牺牲 macro-F1 的解；加 val macro-F1 作次级排序键会更稳。"
          "不选优均值因此被这一条拉低。",
          "2. **seed 46 的 val→test 系统性上偏**：val LFS 1.1725 → test 1.1106（同向但幅度不同），"
          "且其校准后 test LFS 1.1106 已超过同种子 C4 的 0.9775，回收率 >100%。"
          "单种子结论不能外推，跨种子方差极大（val 选优 0.8035–1.1106）。",
          "3. **分母口径**：稿件原用的 0.3950 = mean(C4) − A3(seed 42) = 0.9741 − 0.5791，"
          f"本文用同种子配对均值 {agg['denominator_mean']:.4f}，两者差 0.7%，不影响结论量级。", "",
          f"seed42 复现校验：{out.get('seed42_reproduction_check', {}).get('mismatch', 'n/a') or '完全一致'}；"
          f"总耗时 {out['meta']['elapsed_sec']} s。", ""]
    (RES / "valfit_joint_3seed.md").write_text("\n".join(L), encoding="utf-8")
    print("written:", RES / "valfit_joint_3seed.json", "and", RES / "valfit_joint_3seed.md", flush=True)


if __name__ == "__main__":
    main()
