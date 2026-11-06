"""P1 收尾：RF val proba 落盘（C2(val) 公平校准对照用）。

独立脚本，不改动 train_rf.py 与任何既有结果文件。
val 口径与深度侧完全一致：训练池内 10% 分层切分，random_state=种子
（train_deep.py L215-218：train_test_split(tr, test_size=0.1,
stratify=y[tr], random_state=seed)，然后 np.sort）。
RF 超参与 train_rf.py 默认一致：n_estimators=100、class_weight=balanced_subsample。
fit 只用 90% —— 与深度侧 C2/C4 口径对称（val 不进入任何模型的训练集），
因此这些 val 概率是合法的样本外概率，可直接用于校准搜索。

输出：results/proba/rf_12class_seed{s}_val.npy（float32, (176452, 12)）
      results/proba/rf_12class_seed{s}_val_ytrue.npy（int8）
      results/rf_val_proba_meta.json（协议与逐种子记录）
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    class_names = dp.load_class_names()
    y = dp.load_y()
    tr, te = dp.get_split(y)
    X = dp.load_X(mmap=True)
    te_mask = np.zeros(len(y), dtype=bool)
    te_mask[te] = True
    log(f"数据: X {X.shape}；训练池 {len(tr):,} / 测试 {len(te):,}")

    recs = []
    for s in SEEDS:
        out_p = PROBA / f"rf_12class_seed{s}_val.npy"
        out_t = PROBA / f"rf_12class_seed{s}_test.npy"
        if out_p.exists() and out_t.exists():
            log(f"seed{s}: 已存在，跳过")
            continue
        # 与深度侧逐位一致的 val 切分
        fit_idx, val_idx = train_test_split(
            tr, test_size=0.1, stratify=y[tr], random_state=s)
        fit_idx, val_idx = np.sort(fit_idx), np.sort(val_idx)
        assert te_mask[val_idx].sum() == 0, "val 落入测试集，泄漏！"
        assert len(np.intersect1d(fit_idx, val_idx)) == 0, "fit/val 有交集"

        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1,
            random_state=s, class_weight="balanced_subsample")
        clf.fit(X[fit_idx], y[fit_idx])
        fit_sec = time.time() - t0
        assert list(clf.classes_) == list(range(len(class_names)))

        p_val = clf.predict_proba(X[val_idx]).astype(np.float32)
        np.save(out_p, p_val)
        np.save(PROBA / f"rf_12class_seed{s}_val_ytrue.npy",
                y[val_idx].astype(np.int8))
        # test 概率：同一 90% 模型在固定测试集上的样本外预测
        #（w 在 val 上冻结之前不得读取 test——由 calibrate_lambda.py 的
        #  calibrate()->eval_frozen() 顺序保证，本脚本只负责落盘）
        p_test = clf.predict_proba(X[te]).astype(np.float32)
        np.save(PROBA / f"rf_12class_seed{s}_test.npy", p_test)
        np.save(PROBA / f"rf_12class_seed{s}_test_ytrue.npy",
                y[te].astype(np.int8))

        vm = eu.compute_metrics(y[val_idx], p_val.argmax(1),
                                class_names=class_names,
                                labels=list(range(len(class_names))))
        sec = time.time() - t0
        log(f"seed{s}: fit {len(fit_idx):,} / val {len(val_idx):,} | "
            f"val macroF1 {vm['macro_f1']:.4f} | fit {fit_sec:.0f}s 总 {sec:.0f}s")
        recs.append({"seed": s, "n_fit": int(len(fit_idx)),
                     "n_val": int(len(val_idx)),
                     "val_macro_f1": round(float(vm["macro_f1"]), 6),
                     "fit_sec": round(fit_sec, 1),
                     "total_sec": round(sec, 1)})
        del clf

    meta = {
        "purpose": "C2(val) 公平校准对照（RF 臂）",
        "class_weight": "balanced_subsample",
        "n_estimators": 100,
        "val_protocol": ("train-pool 内分层 10% 切分，random_state=seed，"
                         "与 train_deep.py L215-218 逐位一致；fit 只用 90%"),
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "per_seed": recs,
    }
    with open(OUT / "rf_val_proba_meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    log("全部完成 -> rf_val_proba_meta.json")


if __name__ == "__main__":
    main()
