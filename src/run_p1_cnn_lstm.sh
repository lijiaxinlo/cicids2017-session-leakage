#!/usr/bin/env bash
# P1：CNN-LSTM seeds 43-46 val proba 补跑（seed42 已单独完成并通过确定性验证）
# 断点可续：val proba 已存在的种子自动跳过；每种子跑完即做逐位确定性核对
set -u
V="${PY:-python}"
cd "$(dirname "$0")/.." || exit 1
LOG="results/runs/p1_cnn_lstm_valproba_driver.log"

for S in 43 44 45 46; do
  if [[ -f "results/proba/deep_cnn_lstm_seed${S}_12class_val.npy" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP seed$S（val proba 已存在）" | tee -a "$LOG"
    continue
  fi
  echo "[$(date +%H:%M:%S)] START cnn_lstm seed$S" | tee -a "$LOG"
  "$V" src/train_deep.py --model cnn_lstm --seed "$S" --epochs 30 \
      --batch-size 256 --norm gn --patience 5 --save-val-proba \
      > "results/runs/p1_cnn_lstm_seed${S}.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$(date +%H:%M:%S)] FAIL seed$S (exit=$rc)" | tee -a "$LOG"
    exit $rc
  fi
  # 逐位确定性核对（best_val_macro_f1 必须与备份记录一致）
  "$V" - <<PYEOF | tee -a "$LOG"
import json
old = json.load(open('results/archive/backup_before_p1/deep_cnn_lstm_seed${S}_12class.json', encoding='utf-8'))
new = json.load(open('results/deep_cnn_lstm_seed${S}_12class.json', encoding='utf-8'))
ok = old['best_val_macro_f1'] == new['best_val_macro_f1'] and old['best_epoch'] == new['best_epoch']
print(f"seed${S} 确定性核对: old={old['best_val_macro_f1']} new={new['best_val_macro_f1']} best_epoch {old['best_epoch']}vs{new['best_epoch']} -> {'PASS' if ok else 'FAIL'}")
if not ok:
    raise SystemExit(2)
PYEOF
  [[ $? -ne 0 ]] && { echo "[$(date +%H:%M:%S)] ABORT: seed$S 确定性核对失败" | tee -a "$LOG"; exit 2; }
  echo "[$(date +%H:%M:%S)] DONE  seed$S" | tee -a "$LOG"
done
echo "===== P1 深度侧全部结束 $(date) =====" | tee -a "$LOG"
