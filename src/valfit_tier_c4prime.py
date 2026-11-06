"""在 E7 的三个比例档位上，用与 C4' 完全相同的协议重拟合处理臂，
检验"报告配置（C4'）在 1:1000 下是否仍然稳定"。

背景：Table 8 / §5.4 的校准列用的是 C4（坐标上升 + macro 目标），
而稿件报告的方法是 C4'（DE + LFS 目标）。评审指出鲁棒性实验没有覆盖报告方法。

协议与 src/valfit_treated_arm.py 一致：DE on val，目标 LFS，
benign FPR <= 0.008 罚函数，maxiter=250, popsize=15, tol=1e-6,
polish=True, workers=1, updating=immediate；test 在权重冻结后评估一次。
每 (tier, seed) 做 2 次重启，按 validation LFS 取最优。

用法：python src/valfit_tier_c4prime.py
输出：results/valfit_tier_c4prime.json / .md
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
BOT, BF, XSS = 9, 10, 11
LESION = [BOT, BF, XSS]
SEEDS = (42, 43, 44, 45, 46)
# R=10 在本数据上是 no-op（§4.4），其自然比例参考点即主实验的 C4'（1.2078，三种子），
# 因此这里只重拟合 R=100 与 R=1000 两档，5 个种子（与 Table 8 的种子集一致）。
TIERS = (100, 1000)
# 处理臂在主实验中 12/12 次重启均未落入保守盆地（§5.3），故这里每格 1 次重启。
OPT_SEEDS = (0,)
BOUNDS = [(0.1, 10.0)] * N
DE_KW = dict(maxiter=250, popsize=15, tol=1e-6, polish=True, workers=1,
             updating="immediate")
REL_THRESH = 1.0 / 101.0


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
                "benign_fpr": float((row0 - cm[0, 0]) / row0), "f1": f1}


def load_tier(seed, tier):
    base = PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_tier{tier}_e7_r{tier}"
    files = {k: Path(f"{base}_{k}.npy") for k in ("val", "val_ytrue", "test", "test_ytrue")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"缺文件: {missing}")
    return {k: np.load(p) for k, p in files.items()}


def main():
    t0 = time.time()
    tiers = TIERS
    merge = False
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--tiers" and i + 1 < len(argv):
            tiers = tuple(int(x) for x in argv[i + 1].split(","))
        elif a == "--merge":
            merge = True

    out = {"meta": {"script": "src/valfit_tier_c4prime.py", "cap": CAP,
                    "bounds": [0.1, 10.0], "objective": "LFS",
                    "restarts_per_cell": len(OPT_SEEDS),
                    "note": "同 C4' 协议（DE+LFS）在三个比例档位上重拟合处理臂"},
           "runs": [], "per_tier": {}}

    if merge and (RES / "valfit_tier_c4prime.json").exists():
        prev = json.loads((RES / "valfit_tier_c4prime.json").read_text(encoding="utf-8"))
        out["per_tier"] = {k: v for k, v in prev["per_tier"].items() if int(k) not in tiers}
        out["runs"] = [r for r in prev["runs"] if r["tier"] not in tiers]
        out["meta"] = dict(prev["meta"])
        out["meta"]["restarts_per_cell"] = len(OPT_SEEDS)
        out["meta"]["note"] = ("同 C4' 协议（DE+LFS）在三个比例档位上重拟合处理臂；"
                               "R=10 档为后补（§5.4 的重启对齐比较需要同协议的自然比例点），"
                               "其余两档沿用首次运行结果")
        print(f"merge: kept tiers {sorted(out['per_tier'])}, re-running {list(tiers)}", flush=True)

    for tier in tiers:
        cells = []
        for seed in SEEDS:
            d = load_tier(seed, tier)
            ev_v = FastEval(d["val"], d["val_ytrue"])
            ev_t = FastEval(d["test"], d["test_ytrue"])
            base_t = ev_t.full(np.ones(N))
            best = None
            for os_ in OPT_SEEDS:
                def objective(w):
                    m = ev_v.full(w)
                    return -m["LFS"] + 1000.0 * max(0.0, m["benign_fpr"] - CAP)

                r = differential_evolution(objective, BOUNDS, seed=os_, **DE_KW)
                w = np.clip(r.x, 0.1, 10.0)
                mv = ev_v.full(w)
                mt = ev_t.full(w)
                rec = {"tier": tier, "seed": seed, "opt_seed": os_,
                       "val": {"LFS": round(mv["LFS"], 4), "benign_fpr": round(mv["benign_fpr"], 6)},
                       "test": {"LFS": round(mt["LFS"], 4), "macro_f1": round(mt["macro_f1"], 4),
                                "benign_fpr": round(mt["benign_fpr"], 6),
                                "delta_LFS": round(mt["LFS"] - base_t["LFS"], 4)},
                       "w": [round(float(x), 4) for x in w]}
                out["runs"].append(rec)
                if best is None or rec["val"]["LFS"] > best["val"]["LFS"]:
                    best = rec
            cells.append({"seed": seed, "argmax": {
                "LFS": round(base_t["LFS"], 4), "macro_f1": round(base_t["macro_f1"], 4),
                "benign_fpr": round(base_t["benign_fpr"], 6)}, "selected": best})
            print(f"tier {tier} seed {seed}: argmax LFS {base_t['LFS']:.4f} -> "
                  f"refit {best['test']['LFS']:.4f} (val {best['val']['LFS']:.4f}, "
                  f"fpr {best['test']['benign_fpr']:.4f})", flush=True)
        ls = [c["selected"]["test"]["LFS"] for c in cells]
        ms = [c["selected"]["test"]["macro_f1"] for c in cells]
        fs = [c["selected"]["test"]["benign_fpr"] for c in cells]
        m3 = [c["selected"]["test"]["LFS"] for c in cells if c["seed"] in (42, 44, 46)]
        out["per_tier"][str(tier)] = {
            "n": len(cells), "cells": cells,
            "mean_LFS_3seed_common": round(float(np.mean(m3)), 4),
            "mean_LFS": round(float(np.mean(ls)), 4),
            "sd_LFS": round(float(np.std(ls, ddof=1)), 4),
            "mean_macro": round(float(np.mean(ms)), 4),
            "sd_macro": round(float(np.std(ms, ddof=1)), 4),
            "mean_bfpr": round(float(np.mean(fs)), 6),
            "sd_bfpr": round(float(np.std(fs, ddof=1)), 6)}
        print(f"== tier {tier}: LFS {np.mean(ls):.4f} +/- {np.std(ls, ddof=1):.4f}, "
              f"macro {np.mean(ms):.4f}, bFPR {np.mean(fs):.5f}", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    (RES / "valfit_tier_c4prime.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# C4' 协议在比例档位上的重拟合（{len(SEEDS)} seeds x {len(OPT_SEEDS)} restart）", "",
             f"耗时 {out['elapsed_sec']} s；cap 0.008；目标 LFS；DE 参数同 valfit_treated_arm.py", "",
             "| tier | C3 argmax LFS | C4' refit LFS | Δ | macro-F1 | benign FPR |",
             "|---|---|---|---|---|---|"]
    for tier in sorted(int(k) for k in out["per_tier"]):
        p = out["per_tier"][str(tier)]
        base = float(np.mean([c["argmax"]["LFS"] for c in p["cells"]]))
        lines.append(f"| 1:{tier} | {base:.4f} | {p['mean_LFS']:.4f} ± {p['sd_LFS']:.4f} | "
                     f"{p['mean_LFS']-base:+.4f} | {p['mean_macro']:.4f} ± {p['sd_macro']:.4f} | "
                     f"{p['mean_bfpr']:.5f} |")
    (RES / "valfit_tier_c4prime.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
