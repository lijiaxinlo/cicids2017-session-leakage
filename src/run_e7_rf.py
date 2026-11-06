"""E7 比例敏感性（v4.4 / 决策 D11）—— RF 臂。

三档逐类不平衡上限：BENIGN 全量不动，每个攻击类欠采样至 <= floor(n_benign_fit/R)，
R ∈ {10, 100, 1000}；低于上限的类保持原样；验证集/测试集一行不动。
（动机见 train_main.py --tier-ratio 注释：自然比例下 BENIGN 已是多数类，
 只有压缩攻击类才能单调加剧逐类稀缺性。）

每档每种子产出：
  - RF 臂 = 90% fit 模型在固定测试集上的 argmax
  - RF+cal 臂 = 同一模型的 val/test 概率落盘，供 calibrate_lambda 同套校准
val 切分与深度侧逐位一致：训练池内分层 10%，random_state=seed。
输出：results/proba/e7_rf_R{10,100,1000}_seed{s}_{val,val_ytrue,test,test_ytrue}.npy
      results/e7_rf_tiers.json（逐档逐种子指标 + 逐类行数审计）
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import data_pipeline as dp
import eval_utils as eu

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
PROBA = OUT / "proba"
PROBA.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
TIERS = [10, 100, 1000]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cap_fit(fit_idx, y, benign_id, cap, rng):
    keep, audit = [], {}
    for c in sorted(set(y[fit_idx].tolist())):
        idx_c = np.sort(fit_idx[y[fit_idx] == c])
        if c == benign_id or len(idx_c) <= cap:
            keep.append(idx_c)
            audit[int(c)] = [int(len(idx_c)), int(len(idx_c))]
        else:
            sel = rng.choice(len(idx_c), size=cap, replace=False)
            keep.append(np.sort(idx_c[sel]))
            audit[int(c)] = [int(len(idx_c)), int(cap)]
    return np.sort(np.concatenate(keep)), audit


def main():
    class_names = dp.load_class_names()
    benign_id = class_names.index("BENIGN")
    y = dp.load_y()
    tr, te = dp.get_split(y)
    X = dp.load_X(mmap=True)
    te_mask = np.zeros(len(y), dtype=bool)
    te_mask[te] = True
    log(f"数据 X {X.shape}；训练池 {len(tr):,} / 测试 {len(te):,}")

    out = {"protocol": "E7 v4.4/D11 逐类不平衡上限扫描（RF 臂）",
           "rule": "BENIGN 不动；攻击类欠采样至 <= floor(n_benign_fit/R)；"
                   "低于上限的类保持原样；val/test 不动",
           "class_weight": "balanced_subsample", "n_estimators": 100,
           "val_protocol": "train-pool 分层 10%，random_state=seed（与深度侧一致）",
           "sklearn": sklearn.__version__, "python": platform.python_version(),
           "tiers": {}}

    for R in TIERS:
        rows = []
        for s in SEEDS:
            vp = PROBA / f"e7_rf_R{R}_seed{s}_val.npy"
            tp = PROBA / f"e7_rf_R{R}_seed{s}_test.npy"
            if vp.exists() and tp.exists():
                log(f"R={R} seed{s}: 已存在，跳过")
                continue
            fit_all, val_idx = train_test_split(
                tr, test_size=0.1, stratify=y[tr], random_state=s)
            fit_all, val_idx = np.sort(fit_all), np.sort(val_idx)
            assert te_mask[val_idx].sum() == 0

            n_benign = int((y[fit_all] == benign_id).sum())
            cap = max(1, n_benign // R)
            rng = np.random.default_rng(s)
            fit_idx, audit = cap_fit(fit_all, y, benign_id, cap, rng)

            t0 = time.time()
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=None, n_jobs=-1,
                random_state=s, class_weight="balanced_subsample")
            clf.fit(X[fit_idx], y[fit_idx])
            p_val = clf.predict_proba(X[val_idx]).astype(np.float32)
            p_test = clf.predict_proba(X[te]).astype(np.float32)
            np.save(vp, p_val)
            np.save(PROBA / f"e7_rf_R{R}_seed{s}_val_ytrue.npy",
                    y[val_idx].astype(np.int8))
            np.save(tp, p_test)
            np.save(PROBA / f"e7_rf_R{R}_seed{s}_test_ytrue.npy",
                    y[te].astype(np.int8))

            tm = eu.compute_metrics(y[te], p_test.argmax(1),
                                    class_names=class_names,
                                    labels=list(range(len(class_names))))
            pc = tm["per_class"]
            lesion = sum(pc[n]["f1"] for n in
                         ("Bot", "Web Attack-Brute Force", "Web Attack-XSS"))
            # benign FPR（部署口径）：eu.compute_metrics 不返回该字段，手算
            y_te = y[te]
            pred_te = p_test.argmax(1)
            m_b = y_te == benign_id
            benign_fpr = float((pred_te[m_b] != benign_id).sum() / m_b.sum())
            row = {"seed": s, "R": R, "n_benign": n_benign, "cap": cap,
                   "n_fit_before": int(len(fit_all)), "n_fit_after": int(len(fit_idx)),
                   "audit_per_class": audit,
                   "test_macro_f1": round(float(tm["macro_f1"]), 6),
                   "test_lesion_sum": round(float(lesion), 6),
                   "test_fpr": round(float(tm["fpr"]), 6),
                   "test_benign_fpr": round(benign_fpr, 6),
                   "sec": round(time.time() - t0, 1)}
            rows.append(row)
            log(f"R={R} seed{s}: fit {len(fit_all):,}->{len(fit_idx):,} "
                f"(cap {cap:,}) | test macroF1 {row['test_macro_f1']:.4f} "
                f"病灶和 {row['test_lesion_sum']:.4f} "
                f"benignFPR {row['test_benign_fpr']:.4f} | {row['sec']:.0f}s")
        out["tiers"][str(R)] = rows

    with open(OUT / "e7_rf_tiers.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    log("RF 臂全部完成 -> e7_rf_tiers.json")


if __name__ == "__main__":
    main()
