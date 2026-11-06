#!/bin/bash
# 类加权 CE 对照臂（无重采样）：W1 = α∝1/√n（论文 §3.2 口径），W2 = α∝1/n（教科书 balanced）
V="${PY:-python}"
MODE="$1"   # inv_sqrt | inv
TAG="$2"
for S in 42 44 46; do
  echo "===== $TAG seed $S ====="
  "$V" src/train_main.py --model cnn_bilstm_attn --seed $S --epochs 30 \
    --batch-size 256 --lr 0.001 --loss focal --sampler none --gamma 0.0 \
    --alpha-mode $MODE --benign-target full --minority-target 12000 \
    --patience 5 --tag $TAG 2>&1 | tail -25
done
