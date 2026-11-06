"""落盘六配置选优扫描（E3）的 artifact —— 评审第九轮 P1-3。

§4.3 报告的 2.03% / 2.30% / 0.7948 等数字此前只存在于运行日志与设计文档里，
`results/hpo_main_method.json` 不存在。本脚本从 `results/runs/*_aborted.json`
（被门禁中止的 5 组，含逐 epoch history）与 `results/*_g1_bfull.json`
（唯一幸存组）重建这张表，只转录已记录的值，不推算。

用法：python src/build_hpo_artifact.py
输出：results/hpo_main_method.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RUNS = RES / "runs"

# (gamma, benign_target) -> 源文件；六组 = 3 个 gamma × 2 个 BENIGN 目标
ABORTED = {
    (1.0, "300000"): "deep_cnn_bilstm_attn_seed42_12class_g1_aborted.json",
    (2.0, "300000"): "deep_cnn_bilstm_attn_seed42_12class_aborted.json",
    (3.0, "300000"): "deep_cnn_bilstm_attn_seed42_12class_g3_aborted.json",
    (2.0, "full"): "deep_cnn_bilstm_attn_seed42_12class_bfull_bfull_aborted.json",
    (3.0, "full"): "deep_cnn_bilstm_attn_seed42_12class_g3_bfull_g3_bfull_aborted.json",
}
SURVIVOR = ("g1_bfull", RES / "deep_cnn_bilstm_attn_seed42_12class_g1_bfull.json")


def parse_cmd(cmd):
    """从命令行取 gamma 与 benign-target，避免依赖文件名。"""
    g = bt = None
    toks = cmd.split()
    for i, t in enumerate(toks):
        if t == "--gamma":
            g = float(toks[i + 1])
        if t == "--benign-target":
            bt = toks[i + 1]
    return g, bt


def main():
    rows = []
    for key, fname in ABORTED.items():
        p = RUNS / fname
        d = json.load(open(p, encoding="utf-8"))
        g, bt = parse_cmd(d["command"])
        assert (g, bt) == key, f"{(g, bt)} != {key} for {fname}"
        h = d["history"]
        best_e = max(h, key=lambda e: e["val_macro_f1"])
        rows.append({
            "gamma": g,
            "benign_target": bt,
            "status": "aborted",
            "aborted_reason": d["aborted_reason"],
            "epochs_run": d["epochs_run"],
            "best_epoch": best_e["epoch"],
            "val_macro_f1_best": round(max(e["val_macro_f1"] for e in h), 6),
            "val_macro_f1_last": round(h[-1]["val_macro_f1"], 6),
            "val_benign_fpr_last": round(h[-1]["val_benign_fpr"], 6),
            "val_benign_fpr_at_best_epoch": round(best_e["val_benign_fpr"], 6),
            "val_benign_fpr_max": round(max(e["val_benign_fpr"] for e in h), 6),
            "val_benign_fpr_series": [round(e["val_benign_fpr"], 6) for e in h],
            "test_macro_f1": None,
            "test_lesion_sum": None,
            "timestamp": d["timestamp"],
            "source": f"results/runs/{fname}",
        })

    d = json.load(open(SURVIVOR[1], encoding="utf-8"))
    g, bt = parse_cmd(d["command"])
    h = d.get("history", [])
    best_e = max(h, key=lambda e: e["val_macro_f1"]) if h else None
    rows.append({
        "gamma": g,
        "benign_target": bt,
        "status": "survived",
        "aborted_reason": None,
        "epochs_run": len(h) if h else d.get("best_epoch"),
        "best_epoch": best_e["epoch"] if best_e else d.get("best_epoch"),
        "val_macro_f1_best": round(max(e["val_macro_f1"] for e in h), 6) if h else d.get("val_macro_f1"),
        "val_macro_f1_last": round(h[-1]["val_macro_f1"], 6) if h else None,
        "val_benign_fpr_last": round(h[-1]["val_benign_fpr"], 6) if h else None,
        "val_benign_fpr_at_best_epoch": round(best_e["val_benign_fpr"], 6) if best_e else None,
        "val_benign_fpr_max": round(max(e["val_benign_fpr"] for e in h), 6) if h else None,
        "val_benign_fpr_series": [round(e["val_benign_fpr"], 6) for e in h],
        "test_macro_f1": (round(d["test_metrics"]["macro_f1"], 6)
                          if d.get("test_metrics") else None),
        "test_lesion_sum": (round(d["main_metrics"]["lesion_group_f1_sum"], 6)
                            if d.get("main_metrics") else None),
        "timestamp": d.get("timestamp"),
        "source": str(SURVIVOR[1].relative_to(ROOT)).replace("\\", "/"),
    })

    rows.sort(key=lambda r: (r["benign_target"] != "full", r["gamma"]))
    ab = [r["val_benign_fpr_last"] for r in rows if r["status"] == "aborted"]
    out = {
        "meta": {
            "script": "src/build_hpo_artifact.py",
            "purpose": "E3 six-configuration selection sweep behind §4.3 (gamma x BENIGN target)",
            "seed": 42,
            "n_configs": len(rows),
            "n_aborted": sum(r["status"] == "aborted" for r in rows),
            "n_survived": sum(r["status"] == "survived" for r in rows),
            "abort_bfpr_min": round(min(ab), 6),
            "abort_bfpr_max": round(max(ab), 6),
            "note": "aborted runs produced no test-set result by protocol (§3.3.2 v4.2); "
                    "val_benign_fpr_last is the value at the epoch the gate fired, "
                    "val_benign_fpr_at_best_epoch is the value at the epoch whose weights "
                    "would have been kept. For the survivor the two differ (2.05% at epoch 6, "
                    "1.16% at epoch 11), because early stopping kept epoch 6, not the last epoch.",
        },
        "configs": rows,
    }
    json.dump(out, open(RES / "hpo_main_method.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"{'gamma':>6} {'benign':>7} {'status':>9} {'epochs':>7} "
          f"{'valMacro':>9} {'valBFPR':>9}")
    for r in rows:
        print(f"{r['gamma']:>6.1f} {r['benign_target']:>7} {r['status']:>9} "
              f"{r['epochs_run']:>7} {r['val_macro_f1_best']:>9.4f} "
              f"{(r['val_benign_fpr_last'] or 0):>9.4f}")
    print(f"\nabort bFPR range: {out['meta']['abort_bfpr_min']:.4f} – "
          f"{out['meta']['abort_bfpr_max']:.4f}")
    print("written:", RES / "hpo_main_method.json")


if __name__ == "__main__":
    main()
