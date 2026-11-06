"""处理臂（C4）用同一 LFS 目标做联合搜索，检验"校准在 Bot+BF 上领先 +0.0677"
是机制差异还是目标函数差异（第五轮评审 §3）。

同时落盘按类分解 artifact（评审 P1-3），并测量 max p_XSS / 可达性（评审 P1-2）。

协议与 src/valfit_joint_3seed.py 逐位一致：DE on val, objective LFS,
penalty 1000*max(0,fpr-0.008), maxiter=250, popsize=15, tol=1e-6, polish=True,
workers=1, updating="immediate"；test 只在权重冻结后评估一次。

用法：
  python src/valfit_treated_arm.py probe   # 只做完整性校验 + 按类分解 + XSS 可达性
  python src/valfit_treated_arm.py         # 追加 3 种子 × 4 重启的 DE
输出：results/valfit_treated_arm.json / .md / results/valfit_joint_3seed_perclass.json
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
BOT, BF, XSS = 9, 10, 11
MODEL_SEEDS = (42, 44, 46)
OPT_SEEDS = (0, 1, 2, 3)
BOUNDS = [(0.1, 10.0)] * N
DE_KW = dict(maxiter=250, popsize=15, tol=1e-6, polish=True, workers=1,
             updating="immediate")
REL_THRESH = 1.0 / 101.0
PROBE_ONLY = len(sys.argv) > 1 and sys.argv[1] == "probe"


class FastEval:
    def __init__(self, p, y):
        self.p32 = np.ascontiguousarray(p, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        mx = self.p32.max(1, keepdims=True)
        keep = self.p32 >= mx * REL_THRESH
        fixed = keep.sum(1) == 1
        amb = ~fixed
        pred_fixed = np.argmax(np.where(keep[fixed], self.p32[fixed], -1.0), axis=1)
        self.fixed_cm = np.bincount(self.y[fixed] * N + pred_fixed,
                                    minlength=N * N).reshape(N, N).astype(np.int64)
        self.p_amb = np.ascontiguousarray(self.p32[amb].astype(np.float64))
        self.y_amb12 = (self.y[amb] * N).astype(np.int32)
        self.buf = np.empty_like(self.p_amb)
        self.n_amb = int(amb.sum())

    def cm(self, w):
        np.multiply(self.p_amb, w, out=self.buf)
        pred = np.argmax(self.buf, axis=1).astype(np.int32)
        return self.fixed_cm + np.bincount(self.y_amb12 + pred,
                                           minlength=N * N).reshape(N, N)

    def full(self, w):
        cm = self.cm(w)
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        d = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
        row0 = cm[0].sum()
        return {"LFS": float(f1[LESION].sum()), "macro_f1": float(f1.mean()),
                "benign_fpr": float((row0 - cm[0, 0]) / row0),
                "f1": f1, "support": cm.sum(1)}


def load(prefix_candidates, seed):
    """按候选命名模式加载（proba 目录里 seed42 与 44/46 的命名不一致）。"""
    for tmpl in prefix_candidates:
        base = PROBA / tmpl.format(seed=seed)
        files = {k: Path(f"{base}_{k}.npy") for k in ("val", "val_ytrue", "test", "test_ytrue")}
        if all(p.exists() for p in files.values()):
            return {k: np.load(p) for k, p in files.items()}, str(base.name)
    raise SystemExit(f"seed {seed} 找不到 proba：{prefix_candidates}")


TREATED_PATTERNS = [
    "deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull_g1_bfull",
    "deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull",
]
UNTREATED_PATTERNS = ["deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p"]


def c4_weights():
    d = json.load(open(RES / "e17_calibration_summary.json", encoding="utf-8"))
    return {a["seed"]: a for a in d.get("c4", [])
            if abs(a["tier_benign_cap"] - CAP) < 1e-12}


def f1_dict(m):
    return {"Bot": round(float(m["f1"][BOT]), 4),
            "BruteForce": round(float(m["f1"][BF]), 4),
            "XSS": round(float(m["f1"][XSS]), 4),
            "LFS": round(m["LFS"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "benign_fpr": round(m["benign_fpr"], 6)}


def xss_reachability(p, y):
    """XSS 能否被任何 w∈[0.1,10]^12 选中：需 p_xss/max_{c≠xss} p_c > 1/100。"""
    px = p[:, XSS].astype(np.float64)
    rest = p.copy()
    rest[:, XSS] = -1.0
    mx_other = rest.max(1).astype(np.float64)
    ratio = px / np.maximum(mx_other, 1e-12)
    return {"max_p_xss": float(px.max()),
            "max_p_xss_argmax_is_xss": bool(np.argmax(px) == XSS),
            "max_ratio_xss_over_other": float(ratio.max()),
            "n_rows_reachable": int((ratio > 1.0 / 100.0).sum()),
            "n_rows": int(len(p)),
            "n_xss_true": int((y == XSS).sum()),
            "max_ratio_on_true_xss": float(ratio[y == XSS].max()) if (y == XSS).any() else None}


def main():
    t0 = time.time()
    c4 = c4_weights()
    out = {"meta": {"script": "src/valfit_treated_arm.py", "cap": CAP,
                    "bounds": [0.1, 10.0], "objective": "LFS",
                    "protocol": "identical to valfit_joint_3seed.py; fit on val only"},
           "integrity": [], "per_class": {"untreated": [], "treated": []},
           "xss_reachability": {}, "runs": [], "per_seed": []}

    for seed in MODEL_SEEDS:
        td, tname = load(TREATED_PATTERNS, seed)
        ud, _ = load(UNTREATED_PATTERNS, seed)
        ev_v = FastEval(td["val"], td["val_ytrue"])
        ev_t = FastEval(td["test"], td["test_ytrue"])

        # ---- 完整性校验：C4 权重作用在处理臂 proba 上必须复现已发表的 C4
        w4 = np.asarray(c4[seed]["w"], dtype=np.float64)
        m4 = ev_t.full(w4)
        pub = c4[seed]
        d_lfs = m4["LFS"] - pub["test_lesion_sum"]
        d_mac = m4["macro_f1"] - pub["test_macro_f1"]
        d_fpr = m4["benign_fpr"] - pub["test_benign_fpr"]
        print(f"[seed {seed}] proba={tname}", flush=True)
        print(f"[seed {seed}] C4 复现: LFS {m4['LFS']:.6f} vs {pub['test_lesion_sum']:.6f} "
              f"(Δ{d_lfs:+.2e}) | macro {m4['macro_f1']:.6f} vs {pub['test_macro_f1']:.6f} "
              f"(Δ{d_mac:+.2e}) | FPR {m4['benign_fpr']:.6f} vs {pub['test_benign_fpr']:.6f}", flush=True)
        out["integrity"].append({"seed": seed, "proba": tname,
                                 "d_LFS": d_lfs, "d_macro": d_mac, "d_fpr": d_fpr})
        if abs(d_lfs) > 1e-4 or abs(d_mac) > 1e-4:
            raise SystemExit(f"seed {seed} 的 C4 无法复现，proba 命名或权重对应错误；停止。")

        # ---- 按类分解（4 条臂）
        m_c3 = ev_t.full(np.ones(N))                      # 处理臂 argmax = C3
        m_c4 = m4                                          # 处理臂 + macro 目标坐标上升 = C4
        ev_ut = FastEval(ud["test"], ud["test_ytrue"])
        m_a3 = ev_ut.full(np.ones(N))
        out["per_class"]["treated"].append({"seed": seed, "C3_argmax": f1_dict(m_c3),
                                            "C4_published": f1_dict(m_c4)})
        out["per_class"]["untreated"].append({"seed": seed, "A3_argmax": f1_dict(m_a3)})
        print(f"[seed {seed}] C3 {f1_dict(m_c3)}", flush=True)
        print(f"[seed {seed}] C4 {f1_dict(m_c4)}", flush=True)
        print(f"[seed {seed}] A3 {f1_dict(m_a3)}", flush=True)

        out["xss_reachability"][str(seed)] = {
            "treated": xss_reachability(td["test"], td["test_ytrue"]),
            "untreated": xss_reachability(ud["test"], ud["test_ytrue"])}
        print(f"[seed {seed}] XSS 可达性(treated): {out['xss_reachability'][str(seed)]['treated']}", flush=True)

        out["per_seed"].append({"seed": seed, "c4_LFS": m4["LFS"], "c3_LFS": m_c3["LFS"],
                                "c4_macro": m4["macro_f1"], "c3_macro": m_c3["macro_f1"]})

        if PROBE_ONLY:
            continue

        # ---- 处理臂上做 12 次 LFS 目标 DE
        base_lfs = m_c3["LFS"]
        for os_ in OPT_SEEDS:
            def objective(w):
                m = ev_v.full(w)
                return -m["LFS"] + 1000.0 * max(0.0, m["benign_fpr"] - CAP)

            t = time.time()
            res = differential_evolution(objective, BOUNDS, seed=os_, **DE_KW)
            w = np.asarray(res.x, dtype=np.float64)
            mv = ev_v.full(w)
            mt = ev_t.full(w)
            rec = {"model_seed": seed, "opt_seed": os_,
                   "val": {"LFS": round(mv["LFS"], 6), "benign_fpr": round(mv["benign_fpr"], 6),
                           "macro_f1": round(mv["macro_f1"], 6)},
                   "test": {"LFS": round(mt["LFS"], 6), "delta_LFS": round(mt["LFS"] - base_lfs, 6),
                            "benign_fpr": round(mt["benign_fpr"], 6),
                            "macro_f1": round(mt["macro_f1"], 6)},
                   "test_perclass": f1_dict(mt),
                   "elapsed_sec": round(time.time() - t, 1),
                   "weights": [round(float(x), 6) for x in w]}
            out["runs"].append(rec)
            print(f"[seed {seed} opt {os_}] val LFS={mv['LFS']:.4f} FPR={mv['benign_fpr']:.4f} | "
                  f"test LFS={mt['LFS']:.4f} Δ={mt['LFS']-base_lfs:+.4f} FPR={mt['benign_fpr']:.4f} "
                  f"macro={mt['macro_f1']:.4f} | Bot {mt['f1'][BOT]:.4f} BF {mt['f1'][BF]:.4f} "
                  f"XSS {mt['f1'][XSS]:.4f} ({rec['elapsed_sec']}s)", flush=True)

    if not PROBE_ONLY:
        summ = []
        for seed in MODEL_SEEDS:
            rs = [r for r in out["runs"] if r["model_seed"] == seed]
            best = max(rs, key=lambda r: r["val"]["LFS"])
            ps = next(p for p in out["per_class"]["treated"] if p["seed"] == seed)
            summ.append({"seed": seed,
                         "C4_LFS": ps["C4_published"]["LFS"],
                         "C4_Bot": ps["C4_published"]["Bot"],
                         "C4_BF": ps["C4_published"]["BruteForce"],
                         "C4_XSS": ps["C4_published"]["XSS"],
                         "lfsjoint_val_best_opt": best["opt_seed"],
                         "lfsjoint_LFS": best["test"]["LFS"],
                         "lfsjoint_Bot": best["test_perclass"]["Bot"],
                         "lfsjoint_BF": best["test_perclass"]["BruteForce"],
                         "lfsjoint_XSS": best["test_perclass"]["XSS"],
                         "lfsjoint_macro": best["test"]["macro_f1"],
                         "lfsjoint_fpr": best["test"]["benign_fpr"]})
        out["summary"] = summ
        a = np.array([[s["C4_Bot"] + s["C4_BF"], s["lfsjoint_Bot"] + s["lfsjoint_BF"],
                       s["C4_XSS"], s["lfsjoint_XSS"]] for s in summ])
        out["aggregate"] = {
            "C4_BotBF_mean": round(float(a[:, 0].mean()), 4),
            "lfsjoint_BotBF_mean": round(float(a[:, 1].mean()), 4),
            "BotBF_diff_mean": round(float((a[:, 1] - a[:, 0]).mean()), 4),
            "BotBF_diff_per_seed": [round(float(x), 4) for x in (a[:, 1] - a[:, 0])],
            "C4_XSS_mean": round(float(a[:, 2].mean()), 4),
            "lfsjoint_XSS_mean": round(float(a[:, 3].mean()), 4),
            "C4_LFS_mean": round(float(np.mean([s["C4_LFS"] for s in summ])), 4),
            "lfsjoint_LFS_mean": round(float(np.mean([s["lfsjoint_LFS"] for s in summ])), 4),
            "n_lfsjoint_beats_C4": int(sum(s["lfsjoint_LFS"] > s["C4_LFS"] for s in summ))}
        out["meta"]["elapsed_sec"] = round(time.time() - t0, 1)

    json.dump(out, open(RES / "valfit_treated_arm.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("written:", RES / "valfit_treated_arm.json", flush=True)

    if not PROBE_ONLY:
        g = out["aggregate"]
        print("\n=== 结论 ===")
        print(f"C4      Bot+BF = {g['C4_BotBF_mean']:.4f}  XSS = {g['C4_XSS_mean']:.4f}  LFS = {g['C4_LFS_mean']:.4f}")
        print(f"C4+LFS联合 Bot+BF = {g['lfsjoint_BotBF_mean']:.4f}  XSS = {g['lfsjoint_XSS_mean']:.4f}  LFS = {g['lfsjoint_LFS_mean']:.4f}")
        print(f"Bot+BF 差（处理臂重新按 LFS 优化后 − C4）= {g['BotBF_diff_mean']:+.4f}  逐种子 {g['BotBF_diff_per_seed']}")
        print(f"LFS 联合解超过 C4 的种子数：{g['n_lfsjoint_beats_C4']}/3")


if __name__ == "__main__":
    main()
