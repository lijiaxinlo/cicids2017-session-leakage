#!/bin/bash
# 返修实验驱动
# 阶段 1：时序（purged）协议下的主臂 —— 回答意见 1
# 阶段 2：家族 x 处理网格（原划分）—— 回答意见 2
# 每个臂 3 种子（42/44/46），与全文一致；不开 per-model 调参。
V="${PY:-python}"
STAGE="${1:-all}"

run () {  # run <model> <seed> <tag> <extra args...>
  local M="$1"; local S="$2"; local T="$3"; shift 3
  echo "===== $M seed $S tag $T ====="
  "$V" src/train_main.py --model "$M" --seed "$S" --epochs 30 --batch-size 256 \
    --lr 0.001 --patience 5 --tag "$T" "$@" 2>&1 | tail -18
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "temporal" ]; then
  for S in 42 44 46; do
    # 未处理（CE，无重采样）—— 时序协议下的 A3 等价臂
    run cnn_bilstm_attn $S T_ce --split temporal --loss ce --sampler none --benign-target full
  done
  for S in 42 44 46; do
    # 处理臂（focal γ=2 + SMOTE 300k/12k）—— 时序协议下的 C3 等价臂
    run cnn_bilstm_attn $S T_trt --split temporal --save-proba
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "family" ]; then
  for M in cnn lstm cnn_lstm; do
    for S in 42 44 46; do
      # 同一处理，换骨干；超参不重调
      run $M $S trt --save-proba
    done
  done
fi
