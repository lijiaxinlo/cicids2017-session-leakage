"""直接以 F1(Web Attack-XSS) 为唯一目标做联合搜索，判定未处理臂上 XSS 是否可达。

背景：稿件原写 "a per-class weight ... cannot create a maximum where none exists"，
但权重乘概率后取 argmax 当然会改变 argmax。所以需要实测：
  若以 F1(XSS) 为目标能在边界内把 XSS 抬起来 → "不可达"是错的，
     真实原因是抬 XSS 会等量压低 Brute Force（脊线）；
  若连直接优化都得到 0 → XSS 在 w∈[0.1,10] 内确实不可达，
     但理由必须写成概率结构，而不是"乘法不能创造最大值"。

协议同 valfit_joint_3seed.py：DE on val, maxiter=250, popsize=15, tol=1e-6,
polish=True, workers=1, updating=immediate, benign FPR ≤ 0.008 罚函数。
每种子 3 次重启。

用法：python src/xss_reach_test.py
输出：results/xss_reach_test.json
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
CAP = 0.008
BOT, BF, XSS = 9, 10, 11
LESION = [BOT, BF, XSS]
BOUNDS = [(0.1, 10.0)] * N
DE_KW = dict(maxiter=250, popsize=15, tol=1e-6, polish=True, workers=1,
             updating="immediate")
REL = 1.0 / 101.0


class Fast:
    def __init__(self, p, y):
        self.p32 = np.ascontiguousarray(p, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        mx = self.p32.max(1, keepdims=True)
        keep = self.p32 >= mx * REL
        fixed = keep.sum(1) == 1
        amb = ~fixed
        pf = np.argmax(np.where(keep[fixed], self.p32[fixed], -1.0), axis=1)
        self.fcm = np.bincount(self.y[fixed] * N + pf, minlength=N * N).reshape(N, N).astype(np.int64)
        self.pa = np.ascontiguousarray(self.p32[amb].astype(np.float64))
        self.ya = (self.y[amb] * N).astype(np.int32)
        self.buf = np.empty_like(self.pa)

    def cm(self, w):
        np.multiply(self.pa, w, out=self.buf)
        pred = np.argmax(self.buf, axis=1).astype(np.int32)
        return self.fcm + np.bincount(self.ya + pred, minlength=N * N).reshape(N, N)

    def m(self, w):
        cm = self.cm(w)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        d = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, d, out=np.zeros(N), where=d > 0)
        r0 = cm[0].sum()
        return f1, float((r0 - cm[0, 0]) / r0)


def load(seed):
    b = PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p"
    return (np.load(f"{b}_val.npy"), np.load(f"{b}_val_ytrue.npy"),
            np.load(f"{b}_test.npy"), np.load(f"{b}_test_ytrue.npy"))


def main():
    t0 = time.time()
    out = {"objective": "F1(Web Attack-XSS) only, cap 0.008", "per_seed": []}
    for seed in (42, 44, 46):
        pv, yv, pt, yt = load(seed)
        ev, et = Fast(pv, yv), Fast(pt, yt)
        rows = []
        for os_ in (0, 1, 2):
            def obj(w):
                f1, fpr = ev.m(w)
                return -f1[XSS] + 1000.0 * max(0.0, fpr - CAP)

            r = differential_evolution(obj, BOUNDS, seed=os_, **DE_KW)
            w = np.asarray(r.x, dtype=float)
            fv, fprv = ev.m(w)
            ft, fprt = et.m(w)
            rec = {"opt_seed": os_,
                   "val_xss_f1": round(float(fv[XSS]), 4),
                   "val_fpr": round(fprv, 6),
                   "test_xss_f1": round(float(ft[XSS]), 4),
                   "test_bf_f1": round(float(ft[BF]), 4),
                   "test_bot_f1": round(float(ft[BOT]), 4),
                   "test_LFS": round(float(ft[LESION].sum()), 4),
                   "test_fpr": round(fprt, 6),
                   "weights": [round(float(x), 4) for x in w]}
            rows.append(rec)
            print(f"[seed {seed} opt {os_}] val F1(XSS)={rec['val_xss_f1']:.4f} FPR={fprv:.4f} | "
                  f"test XSS={rec['test_xss_f1']:.4f} BF={rec['test_bf_f1']:.4f} "
                  f"Bot={rec['test_bot_f1']:.4f} LFS={rec['test_LFS']:.4f} FPR={fprt:.4f}",
                  flush=True)
        best = max(rows, key=lambda x: x["val_xss_f1"])
        out["per_seed"].append({"seed": seed, "runs": rows, "val_best": best})
        print(f"[seed {seed}] val-best test XSS F1 = {best['test_xss_f1']:.4f} "
              f"(BF {best['test_bf_f1']:.4f}, LFS {best['test_LFS']:.4f})", flush=True)
        print()

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(RES / "xss_reach_test.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print("written:", RES / "xss_reach_test.json", f"({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
