#!/bin/bash
# 第三轮：组内随机对照划分（P1），把"时间邻近"与"训练集变小/构成变化"拆开
# P0 全局随机 → P1 组内随机（同组同尺寸）→ P2 组内时序，三点连线定位性能下降的来源
#   stage p1_ce  : P1 未处理 × 3 种子
#   stage p1_trt : P1 处理（与 C3 同配置）× 3 种子
# 注意：旧版本此脚本的 G_trt 循环漏了 $TRT，会跑成"未处理"，已修。
set -u
V="${PY:-python}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
GATE="--fpr-abort-benign 1.0 --fpr-abort-macro 1.0 --diverged-ma-benign 1.0 --diverged-ma-macro 1.0"
TRT="--loss focal --gamma 1.0 --alpha-mode uniform --sampler smote --benign-target full --minority-target 12000"
STAGE="${1:-all}"

run () {  # run <seed> <tag> <extra...>
  local S="$1"; local T="$2"; shift 2
  echo "===== grouped seed $S tag $T ====="
  "$V" src/train_main.py --model cnn_bilstm_attn --seed "$S" --epochs 30 \
    --batch-size 256 --lr 0.001 --patience 5 --tag "$T" --split grouped "$@" 2>&1 | tail -16
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "p1_ce" ]; then
  for S in 42 44 46; do
    run "$S" P1_ce --loss ce --sampler none --benign-target full $GATE
  done
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "p1_trt" ]; then
  for S in 42 44 46; do
    run "$S" P1_trt $TRT $GATE
  done
fi
echo "ALLDONE"
