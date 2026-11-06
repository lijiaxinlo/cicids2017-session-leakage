#!/usr/bin/env bash
# E3 六旋钮选优驱动：γ∈{1,2,3} × BENIGN∈{300000,full} × seed42，全量 fit
# 断点可续：已存在结果 json 的组自动跳过
set -u
V="${PY:-python}"
cd "$(dirname "$0")/.." || exit 1
LOGDIR="results/runs"
MAIN_LOG="$LOGDIR/e3_driver.log"

run_cfg () {
  local tag="$1"; shift
  local expect_json="$1"; shift
  if [[ -f "results/$expect_json" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $tag (已存在 results/$expect_json)" | tee -a "$MAIN_LOG"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $tag -> $expect_json" | tee -a "$MAIN_LOG"
  "$V" src/train_main.py --seed 42 --save-proba "$@" \
      > "$LOGDIR/e3_${tag}.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] FAIL $tag (exit=$rc) 详见 results/runs/e3_${tag}.log" | tee -a "$MAIN_LOG"
  else
    echo "[$(date +%H:%M:%S)] DONE  $tag" | tee -a "$MAIN_LOG"
  fi
  return $rc
}

echo "===== E3 启动 $(date) =====" | tee -a "$MAIN_LOG"
# 1) γ=2, BENIGN=300000（默认配置）
run_cfg g2_b300k "deep_cnn_bilstm_attn_seed42_12class.json"
# 2) γ=1, BENIGN=300000
run_cfg g1_b300k "deep_cnn_bilstm_attn_seed42_12class_g1.json" --gamma 1
# 3) γ=3, BENIGN=300000
run_cfg g3_b300k "deep_cnn_bilstm_attn_seed42_12class_g3.json" --gamma 3
# 4) γ=1, BENIGN=full
run_cfg g1_bfull "deep_cnn_bilstm_attn_seed42_12class_g1_bfull.json" --gamma 1 --benign-target full
# 5) γ=2, BENIGN=full
run_cfg g2_bfull "deep_cnn_bilstm_attn_seed42_12class_bfull.json" --benign-target full
# 6) γ=3, BENIGN=full
run_cfg g3_bfull "deep_cnn_bilstm_attn_seed42_12class_g3_bfull.json" --gamma 3 --benign-target full
echo "===== E3 全部结束 $(date) =====" | tee -a "$MAIN_LOG"
