"""E7 四臂校准（v4.4 协议）：每档 (RF, RF+cal, C3, C4)。

复用 calibrate_lambda.py 的 calibrate()/eval_frozen()（零改动）。
- C3 = 主方法 argmax；C4 = 主方法 val 搜 w -> test 冻结
- RF  = RF argmax；RF+cal = 同套 val 搜 w
输入命名：
  主方法  results/proba/deep_cnn_bilstm_attn_seed{s}_12class_tier{R}_e7_r{R}_{val,val_ytrue,test,test_ytrue}.npy
  RF     results/proba/e7_rf_R{R}_seed{s}_{val,val_ytrue,test,test_ytrue}.npy
输出：results/e7_four_arms.json + results/e7_four_arms.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_lambda as CL  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROBA = ROOT / "results" / "proba"
OUT = ROOT / "results"
TIERS = [10, 100, 1000]
SEEDS = [42, 43, 44, 45, 46]
CAPS = [0.008]  # 四臂表只需硬上限档；六档前沿由 E17 在自然比例上给出
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")


def main_p(seed, R, kind):
    return PROBA / f"deep_cnn_bilstm_attn_seed{seed}_12class_tier{R}_e7_r{R}_{kind}.npy"


def rf_p(seed, R, kind):
    return PROBA / f"e7_rf_R{R}_seed{seed}_{kind}.npy"


def load(seed, R):
    out = {}
    for arm, fn in (("main", main_p), ("rf", rf_p)):
        ps = {k: fn(seed, R, k) for k in
              ("val", "val_ytrue", "test", "test_ytrue")}
        if not all(p.exists() for p in ps.values()):
            return None
        out[arm] = {k: np.load(p) for k, p in ps.items()}
    return out


def arms_for(d, cap):
    """返回四臂指标：argmax 与 val 校准后（w 冻结再碰 test）。"""
    res = {}
    for arm in ("main", "rf"):
        a = d[arm]
        y_val, vp = a["val_ytrue"], a["val"].astype(np.float32)
        tp, y_te = a["test"].astype(np.float32), a["test_ytrue"]
        # argmax 臂
        pred = tp.argmax(1)
        arg = {"test_macro_f1": CL.macro_f1(y_te, pred),
               "test_lesion_sum": CL.lesion_sum(y_te, pred),
               "test_benign_fpr": CL.benign_fpr(y_te, pred)}
        # 校准臂（val 搜 -> test 冻结）
        w, info = CL.calibrate(vp, y_val, cap)
        cal = CL.eval_frozen(w, tp, y_te)
        cal["val_benign_fpr"] = info["final_val_benign_fpr"]
        cal["val_macro_f1"] = info["final_val_macro_f1"]
        res[arm] = {"argmax": arg, "cal": cal}
    return res


def mean_std(rows, path):
    v = []
    for r in rows:
        cur = r
        for k in path:
            cur = cur[k]
        v.append(cur)
    a = np.asarray(v, float)
    return round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4)


def main():
    out = {"protocol": "E7 四臂（v4.4）：每档 RF / RF+cal / C3 / C4；校准一律 val 搜、test 冻结后报",
           "tiers": {}, "caps": CAPS}
    for R in TIERS:
        tier_rows = []
        for s in SEEDS:
            d = load(s, R)
            if d is None:
                print(f"[缺输入] R={R} seed={s}")
                continue
            row = {"seed": s, "R": R}
            for cap in CAPS:
                row[str(cap)] = arms_for(d, cap)
            tier_rows.append(row)
            m = row["0.008"]
            print(f"R={R} seed{s} cap=0.008: C3 {m['main']['argmax']['test_macro_f1']:.4f}"
                  f" -> C4 {m['main']['cal']['test_macro_f1']:.4f} | "
                  f"RF {m['rf']['argmax']['test_macro_f1']:.4f} -> "
                  f"RF+cal {m['rf']['cal']['test_macro_f1']:.4f}")
        out["tiers"][str(R)] = tier_rows

    # 汇总（cap=0.008 主档）
    md = ["# E7 四臂结果（v4.4）", "",
          "每档 5 种子 mean±std；校准 cap=0.008（硬上限档）。", "",
          "| 档位 | C3 macro | C4 macro | C3 病灶 | C4 病灶 | C4 benignFPR "
          "| RF macro | RF+cal macro | RF 病灶 | RF+cal 病灶 | RF+cal benignFPR |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for R in TIERS:
        rows = out["tiers"][str(R)]
        if not rows:
            continue
        vals = {}
        for key, path in {
            "C3_macro": ("0.008", "main", "argmax", "test_macro_f1"),
            "C4_macro": ("0.008", "main", "cal", "test_macro_f1"),
            "C3_lesion": ("0.008", "main", "argmax", "test_lesion_sum"),
            "C4_lesion": ("0.008", "main", "cal", "test_lesion_sum"),
            "C4_bfpr": ("0.008", "main", "cal", "test_benign_fpr"),
            "RF_macro": ("0.008", "rf", "argmax", "test_macro_f1"),
            "RFcal_macro": ("0.008", "rf", "cal", "test_macro_f1"),
            "RF_lesion": ("0.008", "rf", "argmax", "test_lesion_sum"),
            "RFcal_lesion": ("0.008", "rf", "cal", "test_lesion_sum"),
            "RFcal_bfpr": ("0.008", "rf", "cal", "test_benign_fpr"),
        }.items():
            vals[key] = mean_std(rows, list(path))
        summary[str(R)] = vals
        md.append(f"| R={R} | " + " | ".join(
            f"{vals[k][0]:.4f}±{vals[k][1]:.4f}" for k in
            ("C3_macro", "C4_macro", "C3_lesion", "C4_lesion", "C4_bfpr",
             "RF_macro", "RFcal_macro", "RF_lesion", "RFcal_lesion",
             "RFcal_bfpr")) + " |")
    out["summary_cap0008"] = summary
    with open(OUT / "e7_four_arms.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    (OUT / "e7_four_arms.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
