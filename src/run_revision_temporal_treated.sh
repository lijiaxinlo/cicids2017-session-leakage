#!/bin/bash
# 时序协议下的处理臂（闸门关闭版）
# 说明：v4.2 的 FPR 闸门是为 P1（随机划分）设计的，阈值 0.05 是"相对 P1 常态"的
# 发散判据。P2 下 val benign FPR 从第 1 轮就在 0.08–0.14，闸门会删掉全部三个种子，
# 那样反而看不到数字。故显式关闭闸门（--fpr-abort-* 设 1.0），结果如实报告，
# 并在论文里写明"审计规则本身在 P2 下会拦下所有运行"。
V="${PY:-python}"
for S in 42 44 46; do
  echo "===== T_trt_nogate seed $S ====="
  "$V" src/train_main.py --model cnn_bilstm_attn --seed $S --epochs 30 \
    --batch-size 256 --lr 0.001 --patience 5 --split temporal --save-proba \
    --fpr-abort-benign 1.0 --fpr-abort-macro 1.0 --diverged-ma-benign 1.0 --diverged-ma-macro 1.0 --tag T_trt_nogate 2>&1 | tail -20
done
echo ALLDONE
