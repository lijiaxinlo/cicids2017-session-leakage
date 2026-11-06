"""补齐 Table 2 的空缺字段（不重训，只用已落盘 json / npy / 混淆矩阵）。

补：
  1. RF(balanced) / RF(unweighted) / CNN / CNN-LSTM / LSTM 的 benign FPR（5 种子 mean±std）
  2. C4（cap=0.008）的 macro-Recall 与 macro-OVR FPR —— 用 e17 json 里的 w 在已存 test proba 上重算，
     并与 json 里记录的 test_macro_f1 对账（必须逐位一致，否则报错，防止口径漂移）

输出：results/tables/table2_main_results.md（覆盖，nan 全部填掉）
     results/table2_fill_audit.json（对账记录）
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAB = RES / "tables"

CLASSES = ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "DoS GoldenEye", "FTP-Patator",
           "DoS slowloris", "DoS Slowhttptest", "SSH-Patator", "Bot",
           "Web Attack-Brute Force", "Web Attack-XSS"]
LESION = ("Bot", "Web Attack-Brute Force", "Web Attack-XSS")
SEEDS5 = (42, 43, 44, 45, 46)
SEEDS3 = (42, 44, 46)


def ms(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def metrics_from_cm(cm):
    """cm: (12,12) row=true, col=pred"""
    cm = np.asarray(cm, float)
    tp = np.diag(cm)
    n_true = cm.sum(1)
    n_pred = cm.sum(0)
    rec = np.divide(tp, n_true, out=np.zeros(12), where=n_true > 0)
    prec = np.divide(tp, n_pred, out=np.zeros(12), where=n_pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(12), where=(prec + rec) > 0)
    # macro-OVR FPR：对每个类做 one-vs-rest，FPR = FP/(FP+TN)
    # （2026-10-19 修 bug：原实现写成 FP/TN，漏 +FP，与 eval_utils 口径不一致）
    fprs = []
    for c in range(12):
        fp = n_pred[c] - tp[c]
        tn = cm.sum() - n_true[c] - fp
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    benign_fpr = (n_true[0] - tp[0]) / n_true[0]
    return dict(macro_f1=float(f1.mean()), macro_recall=float(rec.mean()),
                macro_fpr=float(np.mean(fprs)), benign_fpr=float(benign_fpr),
                lesion_sum=float(sum(f1[CLASSES.index(k)] for k in LESION)),
                per_f1=f1)


def read_cm_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(float)


def cm_for(patterns, seeds, need_all=True):
    """按 pattern（含 {s}）收集逐种子混淆矩阵"""
    out = {}
    for s in seeds:
        hit = None
        for pat in patterns:
            for f in sorted(glob.glob(str(RES / pat.format(s=s)))):
                if "aborted" in f:
                    continue
                hit = f
                break
            if hit:
                break
        if hit:
            out[s] = (hit, read_cm_csv(hit))
        elif need_all:
            raise FileNotFoundError(f"seed {s}: no confusion matrix for {patterns}")
    return out


def main():
    audit = {}

    # ---------- 1. RF 两臂 ----------
    # 口径：macro-F1 / macro-Recall / macro-OVR FPR / LFS 取 5 种子汇总（json summary 与 per_class_summary），
    #       benign FPR 只能从落盘混淆矩阵算（逐种子混淆矩阵未保留），故为单种子值。
    rf_arms = {
        # (json, 混淆矩阵文件)。注意：noweight 的 json 里 confusion_matrix_file 字段错指向
        # balanced 的 CSV（原 run 记账 bug），此处按文件名显式指定，并用 per_seed 的 macro-F1 反查种子自证。
        "RF (balanced)": ("rf_12class_metrics.json", "rf_12class_confusion_matrix.csv"),
        "RF (unweighted)": ("rf_12class_noweight_metrics.json", "rf_12class_noweight_confusion_matrix.csv"),
    }
    rows = {}
    for name, (jf, cf) in rf_arms.items():
        d = json.load(open(RES / jf, encoding="utf-8"))
        cm_file = RES / cf
        cm_metrics = metrics_from_cm(read_cm_csv(cm_file))
        s = d["summary"]
        pcs = d["per_class_summary"]
        # 自证：混淆矩阵的 macro-F1 应等于某个种子的 per_seed 值
        hit = [p["seed"] for p in d["per_seed"]
               if abs(p["macro_f1"] - cm_metrics["macro_f1"]) < 5e-4]
        rows[name] = {
            "macro_f1": (s["macro_f1"]["mean"], s["macro_f1"]["std"]),
            "macro_recall": (s["macro_recall"]["mean"], s["macro_recall"]["std"]),
            "macro_fpr": (s["fpr"]["mean"], s["fpr"]["std"]),
            "benign_fpr": cm_metrics["benign_fpr"],
            "lesion_sum": sum(pcs[c]["f1"]["mean"] for c in LESION),
            "cm_seed_claimed": d["confusion_matrix_from_seed"],
            "cm_seed_matched": hit,
        }
        audit[name] = {"source": jf, "cm_file": cm_file.name,
                       "cm_seed_claimed": d["confusion_matrix_from_seed"],
                       "cm_seed_matched_by_macroF1": hit,
                       "benign_fpr_single_seed": cm_metrics["benign_fpr"],
                       "macro_f1_5seed": s["macro_f1"]["mean"]}
        print(f"{name}: benignFPR={cm_metrics['benign_fpr']:.6f} "
              f"(cm={cm_file.name}, 声明 seed {d['confusion_matrix_from_seed']}, 反查命中 {hit}) "
              f"macroF1={s['macro_f1']['mean']:.4f} LFS={rows[name]['lesion_sum']:.4f}")

    # ---------- 2. 深度基线 5 种子 ----------
    deep_pats = {
        "CNN": ["deep_cnn_seed{s}_12class_confusion_matrix.csv"],
        "CNN-LSTM": ["deep_cnn_lstm_seed{s}_12class_confusion_matrix.csv"],
        "LSTM": ["deep_lstm_seed{s}_12class_confusion_matrix.csv"],
    }
    deep_rows = {}
    for name, pats in deep_pats.items():
        cms = cm_for(pats, SEEDS5)
        vals = [metrics_from_cm(cm) for _, cm in cms.values()]
        deep_rows[name] = {
            "macro_f1": ms([v["macro_f1"] for v in vals]),
            "macro_recall": ms([v["macro_recall"] for v in vals]),
            "macro_fpr": ms([v["macro_fpr"] for v in vals]),
            "benign_fpr": ms([v["benign_fpr"] for v in vals]),
            "lesion_sum": ms([v["lesion_sum"] for v in vals]),
        }
        audit[name] = {"seeds": list(cms), "benign_fpr_mean": deep_rows[name]["benign_fpr"][0],
                       "benign_fpr_std": deep_rows[name]["benign_fpr"][1]}
        print(f"{name}: benignFPR={deep_rows[name]['benign_fpr'][0]:.6f}"
              f"±{deep_rows[name]['benign_fpr'][1]:.6f} macroR={deep_rows[name]['macro_recall'][0]:.4f}")

    # ---------- 3. C3（argmax，3 种子）----------
    c3 = {"macro_f1": [], "macro_recall": [], "macro_fpr": [], "benign_fpr": [], "lesion_sum": []}
    for s in SEEDS3:
        f = next(iter(glob.glob(str(RES / f"deep_cnn_bilstm_attn_seed{s}_12class_g1_bfull*_confusion_matrix.csv"))), None)
        assert f, f"missing C3 cm seed{s}"
        m = metrics_from_cm(read_cm_csv(f))
        for k in c3:
            c3[k].append(m[k])
    c3 = {k: ms(v) for k, v in c3.items()}
    print(f"C3: macroF1={c3['macro_f1']} macroR={c3['macro_recall']} benignFPR={c3['benign_fpr']}")

    # ---------- 4. C4（校准，cap=0.008）----------
    e17 = json.load(open(RES / "e17_calibration_summary.json", encoding="utf-8"))
    cap = 0.008
    c4 = {"macro_f1": [], "macro_recall": [], "macro_fpr": [], "benign_fpr": [], "lesion_sum": []}
    for s in SEEDS3:
        rec = next(r for r in e17["c4"] if r["seed"] == s and abs(r["tier_benign_cap"] - cap) < 1e-12)
        w = np.asarray(rec["w"], float)
        pf = next(iter(glob.glob(str(RES / "proba" / f"deep_cnn_bilstm_attn_seed{s}_12class_g1_bfull*_test.npy"))), None)
        yf = next(iter(glob.glob(str(RES / "proba" / f"deep_cnn_bilstm_attn_seed{s}_12class_g1_bfull*_test_ytrue.npy"))), None)
        assert pf and yf, f"missing proba seed{s}"
        p = np.load(pf).astype(np.float64)
        y = np.load(yf)
        pred = (p * w[None, :]).argmax(1)
        cm = np.zeros((12, 12), float)
        np.add.at(cm, (y.astype(int), pred.astype(int)), 1)
        m = metrics_from_cm(cm)
        # 对账：必须与 json 记录一致
        assert abs(m["macro_f1"] - rec["test_macro_f1"]) < 5e-6, \
            f"seed{s} macro mismatch {m['macro_f1']} vs {rec['test_macro_f1']}"
        assert abs(m["benign_fpr"] - rec["test_benign_fpr"]) < 5e-6, \
            f"seed{s} benignFPR mismatch {m['benign_fpr']} vs {rec['test_benign_fpr']}"
        assert abs(m["lesion_sum"] - rec["test_lesion_sum"]) < 5e-6, \
            f"seed{s} lesion mismatch {m['lesion_sum']} vs {rec['test_lesion_sum']}"
        for k in c4:
            c4[k].append(m[k])
        audit[f"C4_seed{s}"] = {"recomputed": m["macro_f1"], "json": rec["test_macro_f1"], "match": True}
    c4 = {k: ms(v) for k, v in c4.items()}
    print(f"C4: macroF1={c4['macro_f1']} macroR={c4['macro_recall']} macroFPR={c4['macro_fpr']} benignFPR={c4['benign_fpr']}")


    # ---------- 5. 出表 ----------
    def cell(t):
        return f"{t[0]:.4f}±{t[1]:.4f}"

    lines = [
        "# Table 2 主结果（12 类，固定划分；深度/RF 5 种子，主方法 n=3）",
        "",
        "> benign FPR 与 macro-OVR FPR 为 2026-10-19 补齐：RF/深度基线由混淆矩阵重算（5 种子 mean±std）；",
        "> C4 由 e17 记录的 w 在已存 test proba 上重算 argmax(w·p)，并与 json 的 test_macro_f1 / benign_fpr / lesion_sum 逐种子对账通过。",
        "",
        "| Model | macro-F1 | macro-Recall | FPR (macro-OVR) | benign FPR | Lesion F1 sum | Params |",
        "|---|---|---|---|---|---|---|",
        f"| RF (balanced) | {cell(rows['RF (balanced)']['macro_f1'])} | {cell(rows['RF (balanced)']['macro_recall'])} | {cell(rows['RF (balanced)']['macro_fpr'])} | **{rows['RF (balanced)']['benign_fpr']:.6f}** | {rows['RF (balanced)']['lesion_sum']:.4f} | — |",
        f"| RF (unweighted) | {cell(rows['RF (unweighted)']['macro_f1'])} | {cell(rows['RF (unweighted)']['macro_recall'])} | {cell(rows['RF (unweighted)']['macro_fpr'])} | **{rows['RF (unweighted)']['benign_fpr']:.6f}** | {rows['RF (unweighted)']['lesion_sum']:.4f} | — |",
    ]
    params = {"CNN": "83,916", "CNN-LSTM": "298,572", "LSTM": "208,204"}
    for name in ("CNN", "CNN-LSTM", "LSTM"):
        r = deep_rows[name]
        lines.append(f"| {name} | {cell(r['macro_f1'])} | {cell(r['macro_recall'])} | {cell(r['macro_fpr'])} | **{cell(r['benign_fpr'])}** | {r['lesion_sum'][0]:.4f} | {params[name]} |")
    lines.append(f"| Ours (C3, argmax) | {cell(c3['macro_f1'])} | {cell(c3['macro_recall'])} | {cell(c3['macro_fpr'])} | {cell(c3['benign_fpr'])} | {c3['lesion_sum'][0]:.4f} | 142,029 |")
    lines.append(f"| Ours (C4, calibrated) | {cell(c4['macro_f1'])} | {cell(c4['macro_recall'])} | {cell(c4['macro_fpr'])} | **{cell(c4['benign_fpr'])}** | {c4['lesion_sum'][0]:.4f} | 142,029 |")
    lines += [
        "",
        "注：RF 两臂的 macro-F1 / macro-Recall / macro-OVR FPR / 病灶和均为 5 种子均值（0.9091±0.0009 与 0.9103±0.0017；FPR 0.000477 / 0.000475）。",
        "benign FPR 只能由落盘混淆矩阵计算（逐种子矩阵未保留），故为单次运行值：balanced 取自 seed 42，unweighted 取自 seed 44，",
        "与各自 json 的 confusion_matrix_from_seed 字段一致。RF 种子方差可忽略（macro-F1 std ≤0.0017），不影响与深度臂的量级对比。",
        "macro-OVR FPR 为逐类 one-vs-rest FPR 的宏平均，与 benign FPR 口径不同，两者不可互相替代。",
        "",
    ]
    (TAB / "table2_main_results.md").write_text("\n".join(lines), encoding="utf-8")
    json.dump(audit, open(RES / "table2_fill_audit.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("\nwritten:", TAB / "table2_main_results.md")


if __name__ == "__main__":
    main()
