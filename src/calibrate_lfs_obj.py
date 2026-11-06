"""回应模拟审稿意见 M1：把校准目标从 macro-F1 换成病灶组 F1 和（LFS）后重做替代检验。

审稿人指认：E17 的目标函数是 val macro-F1，而结论却关于 LFS——一个能抬 LFS 却压 macro 的
权重会被目标函数主动拒绝。因此"校准不能替代处理"可能是搜索设计的产物。本脚本直接检验该指控。

协议与 calibrate_lambda.py 完全一致（同一网格 exp(linspace(log0.1,log10,21))、坐标上升 3 轮、
同一约束强制段、test 在 w 冻结后一次性评估），**只改目标函数**。

臂：
  T-val  主方法(C3)种子 42/44/46，cap=0.008，目标 LFS，val 拟合 -> test 报告
  T-orc  主方法种子 42，目标 LFS，直接在 test 上拟合（上界，非法但用于定界）
  U-val  未处理(A3, 纯 CE)种子 42，cap=0.008，目标 LFS，val 拟合 -> test 报告
  U-orc  未处理种子 42，目标 LFS，test 拟合（上界）
  U-free 未处理种子 42，目标 LFS，**无 FPR 约束**，val 拟合 -> test 报告

输出：results/calib_lfs_objective.json + .md
"""
import glob
import json
import time
from pathlib import Path

import numpy as np

import calibrate_lambda as CL

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PROBA = RES / "proba"
CAP = 0.008
OBJ = CL.lesion_sum          # 唯一改动：目标函数
OBJ_NAME = "LFS"
N = CL.N_CLASSES


def calibrate_obj(val_p, y_val, cap, objective):
    """与 CL.calibrate 同构，但目标函数可换。cap=None 表示无约束。"""
    w = np.ones(N, dtype=np.float64)
    pred = np.argmax(val_p * w, axis=1)
    cur = objective(y_val, pred)
    info = {"rounds": 0, "infeasible_classes": []}
    for rnd in range(1, CL.MAX_ROUNDS + 1):
        changed = False
        for c in range(N):
            best_v, best_w = -1.0, None
            all_violate = True
            for g in CL.GRID:
                w_try = w.copy()
                w_try[c] = g
                pred = np.argmax(val_p * w_try, axis=1)
                if cap is not None and CL.benign_fpr(y_val, pred) > cap:
                    continue
                all_violate = False
                v = objective(y_val, pred)
                if v > best_v + 1e-12:
                    best_v, best_w = v, g
            if all_violate:
                info["infeasible_classes"].append({"class": c, "tier_cap": cap})
                continue
            if best_v > cur + 1e-9 and best_w is not None:
                w[c] = best_w
                cur = best_v
                changed = True
        info["rounds"] = rnd
        if not changed:
            break
    # 约束强制段：benign FPR 对 w0 单调不增
    if cap is not None:
        pred = np.argmax(val_p * w, axis=1)
        if CL.benign_fpr(y_val, pred) > cap:
            enforced = None
            for w0 in np.geomspace(max(w[0], 1.0), 1e4, 41):
                w_try = w.copy()
                w_try[0] = w0
                if CL.benign_fpr(y_val, np.argmax(val_p * w_try, axis=1)) <= cap:
                    enforced = float(w0)
                    break
            if enforced is not None:
                w[0] = enforced
                info["enforced_w0"] = round(enforced, 2)
            else:
                info["constraint_unreachable"] = True
    pred = np.argmax(val_p * w, axis=1)
    info["final_val_benign_fpr"] = round(CL.benign_fpr(y_val, pred), 6)
    info["final_val_LFS"] = round(CL.lesion_sum(y_val, pred), 6)
    info["final_val_macro_f1"] = round(CL.macro_f1(y_val, pred), 6)
    info["n_nonunit_w"] = int(sum(abs(x - 1.0) > 1e-9 for x in w))
    info["w"] = [round(float(x), 4) for x in w]
    return w, info


def report(w, p, y):
    pred = np.argmax(p * w, axis=1)
    return {"benign_fpr": round(CL.benign_fpr(y, pred), 6),
            "macro_f1": round(CL.macro_f1(y, pred), 6),
            "LFS": round(CL.lesion_sum(y, pred), 6)}


def load_treated(seed):
    hits = sorted(glob.glob(str(PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_g1_bfull*_val.npy")))
    base = hits[0][:-len("_val.npy")]
    return {"val_p": np.load(f"{base}_val.npy"), "val_y": np.load(f"{base}_val_ytrue.npy"),
            "test_p": np.load(f"{base}_test.npy"), "test_y": np.load(f"{base}_test_ytrue.npy")}


def load_untreated(seed=42):
    base = PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_ce_none_a3p"
    return {"val_p": np.load(f"{base}_val.npy"), "val_y": np.load(f"{base}_val_ytrue.npy"),
            "test_p": np.load(f"{base}_test.npy"), "test_y": np.load(f"{base}_test_ytrue.npy")}


def main():
    out = {"protocol": "objective = lesion-group F1 sum (LFS); grid/rounds/constraint identical to E17",
           "cap": CAP, "objective": OBJ_NAME, "arms": []}
    t0 = time.time()

    def add(name, d, cap, fit_on):
        vp, vy = (d["val_p"], d["val_y"]) if fit_on == "val" else (d["test_p"], d["test_y"])
        w, info = calibrate_obj(vp, vy, cap, OBJ)
        base = report(np.ones(N), d["test_p"], d["test_y"])
        got = report(w, d["test_p"], d["test_y"])
        rec = {"arm": name, "cap": cap, "fit_on": fit_on, "seed": d.get("seed"),
               "argmax_test": base, "calibrated_test": got,
               "delta_LFS": round(got["LFS"] - base["LFS"], 6),
               "delta_macro": round(got["macro_f1"] - base["macro_f1"], 6),
               "fit": info}
        out["arms"].append(rec)
        print(f"[{time.time()-t0:6.0f}s] {name}: LFS {base['LFS']:.4f}->{got['LFS']:.4f} "
              f"(d={rec['delta_LFS']:+.4f}) macro {base['macro_f1']:.4f}->{got['macro_f1']:.4f} "
              f"benignFPR {got['benign_fpr']:.4f} nonunit_w={info['n_nonunit_w']} rounds={info['rounds']}",
              flush=True)

    # 主方法（已处理）
    for s in (42, 44, 46):
        d = load_treated(s); d["seed"] = s
        add(f"T-val_seed{s}", d, CAP, "val")
    d = load_treated(42); d["seed"] = 42
    add("T-oracle_seed42", d, CAP, "test")

    # 未处理（A3，纯 CE）
    u = load_untreated(42); u["seed"] = 42
    add("U-val_seed42", u, CAP, "val")
    add("U-oracle_seed42", u, CAP, "test")
    add("U-free_seed42", u, None, "val")

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(RES / "calib_lfs_objective.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    # md
    L = ["# 校准目标换成 LFS 后的替代检验（回应审稿意见 M1）", "",
         f"协议：网格 exp(linspace(log0.1,log10,21))、坐标上升 ≤3 轮、约束强制段与 E17 完全一致；"
         f"**唯一改动 = 目标函数由 macro-F1 换成病灶组 F1 和**。cap = {CAP}（U-free 无约束）。", "",
         "| 臂 | 拟合集 | cap | argmax LFS | 校准后 LFS | ΔLFS | argmax macro | 校准后 macro | test benign FPR | 非单位权重数 | 轮数 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["arms"]:
        L.append(f"| {r['arm']} | {r['fit_on']} | {r['cap']} | {r['argmax_test']['LFS']:.4f} | "
                 f"{r['calibrated_test']['LFS']:.4f} | {r['delta_LFS']:+.4f} | "
                 f"{r['argmax_test']['macro_f1']:.4f} | {r['calibrated_test']['macro_f1']:.4f} | "
                 f"{r['calibrated_test']['benign_fpr']:.4f} | {r['fit']['n_nonunit_w']} | {r['fit']['rounds']} |")
    L += ["", f"耗时 {out['elapsed_sec']} s。", ""]
    (RES / "calib_lfs_objective.md").write_text("\n".join(L), encoding="utf-8")
    print("written:", RES / "calib_lfs_objective.md")


if __name__ == "__main__":
    main()
