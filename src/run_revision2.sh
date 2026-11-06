#!/bin/bash
# 返修实验驱动（串行；配置统一为论文报告的 C3 口径：focal γ=1、α=uniform、
# SMOTE 病灶类→12000、BENIGN 不欠采样，避免"协议 × 超参"混淆）
#
#   stage rf      : P2 下 RF 三臂（加权 / 不处理 / 与深度臂同处理）
#   stage trt_g1  : P2 下处理臂（与 C3 同配置）—— 协议对比的主臂
#   stage family  : P2 下家族网格 cnn/lstm/cnn_lstm × 3 种子（闸门关闭）
#   stage fam_ce  : P2 下 cnn 不处理 × 3 种子（架构 × 处理的分解）
#   stage calib4p : P2 下 C4′（联合搜索，病灶目标）
set -u
V="${PY:-python}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
GATE="--fpr-abort-benign 1.0 --fpr-abort-macro 1.0 --diverged-ma-benign 1.0 --diverged-ma-macro 1.0"
TRT="--loss focal --gamma 1.0 --alpha-mode uniform --sampler smote --benign-target full --minority-target 12000"
STAGE="${1:-all}"

run () {  # run <model> <seed> <tag> <extra...>
  local M="$1"; local S="$2"; local T="$3"; shift 3
  echo "===== $M seed $S tag $T ====="
  "$V" src/train_main.py --model "$M" --seed "$S" --epochs 30 --batch-size 256 \
    --lr 0.001 --patience 5 --tag "$T" "$@" 2>&1 | tail -16
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "rf" ]; then
  echo "===== rf: 加权（= 论文口径 balanced_subsample） ====="
  "$V" src/train_rf.py --split temporal --tag p2 2>&1 | tail -10
  echo "===== rf: 不处理 ====="
  "$V" src/train_rf.py --split temporal --class-weight none --tag p2 2>&1 | tail -10
  echo "===== rf: 与深度臂同处理（BENIGN 30万 + 病灶类 1.2万随机过采样） ====="
  "$V" src/train_rf.py --split temporal --benign-target 300000 --lesion-target 12000 --tag p2treat 2>&1 | tail -10
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "trt_g1" ]; then
  for S in 42 44 46; do
    run cnn_bilstm_attn "$S" T_trt_g1 --split temporal --save-proba $TRT $GATE
  done
  echo "===== calib4 on T_trt_g1 ====="
  ( cd src && "$V" calibrate_temporal.py --mode c4 --tag T_trt_g1 \
      --out ../results/temporal/calibration_g1.json ) 2>&1
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "family" ]; then
  for M in cnn lstm cnn_lstm; do
    for S in 42 44 46; do
      run "$M" "$S" T_trt_ng --split temporal $TRT $GATE
    done
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "fam_ce" ]; then
  for S in 42 44 46; do
    run cnn "$S" T_ce_ng --split temporal --loss ce --sampler none \
        --benign-target full $GATE
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "calib4p" ]; then
  echo "===== calib4p ====="
  ( cd src && "$V" calibrate_temporal.py --mode c4p ) 2>&1
fi

echo "ALLDONE"
