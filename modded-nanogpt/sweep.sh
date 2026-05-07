#!/bin/bash
set -e
cd /root/modded-nanogpt
source /root/.venv-loco/bin/activate
N_STEPS=${N_STEPS:-10}
LOG_DIR=${LOG_DIR:-sweep_logs}
mkdir -p $LOG_DIR
for K in 1 4 16 64 256 1024 4096 16384; do
    LOG=$LOG_DIR/k${K}_n${N_STEPS}.log
    echo "=== K=$K N_STEPS=$N_STEPS ===" | tee -a $LOG
    LOCO_STEPS=$K LOCO_LR=0.0001 LR=0.001 SEED=42 \
      N_STEPS=$N_STEPS STAGED=1 SIGN=1 \
      torchrun --standalone --nproc_per_node=8 train_t3.py 2>&1 | tee -a $LOG
done
echo "DONE"
