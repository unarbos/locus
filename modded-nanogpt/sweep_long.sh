#!/bin/bash
set -e
cd /root/modded-nanogpt
source /root/.venv-loco/bin/activate
LOG_DIR=${LOG_DIR:-sweep_long_logs}
mkdir -p $LOG_DIR

run_config() {
    local NAME=$1
    local STAGED=$2
    local K=$3
    local LOG=$LOG_DIR/${NAME}.log
    echo "=== $NAME staged=$STAGED K=$K ===" | tee -a $LOG
    LOCO_STEPS=$K LOCO_LR=0.0001 LR=0.001 SEED=42 \
      N_STEPS=1000 STAGED=$STAGED SIGN=1 LOG_EVERY=20 \
      torchrun --standalone --nproc_per_node=8 train_t3.py 2>&1 | tee -a $LOG
}

run_config baseline 0 0
run_config k1 1 1
run_config k4 1 4
run_config k16 1 16
run_config k64 1 64
run_config k256 1 256

echo "DONE"
