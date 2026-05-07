#!/bin/bash
set -e
cd /root/modded-nanogpt
source /root/.venv-loco/bin/activate
LOG_DIR=${LOG_DIR:-batch_sweep_logs}
mkdir -p $LOG_DIR

run() {
    local NAME=$1
    local STAGED=$2
    local K=$3
    local LR=$4
    local LOCO_LR=$5
    local BATCH=$6
    local N_STEPS=$7
    local LOG=$LOG_DIR/${NAME}.log
    echo "=== $NAME B=$BATCH LR=$LR LOCO_LR=$LOCO_LR K=$K N=$N_STEPS ===" | tee -a $LOG
    LOCO_STEPS=$K LOCO_LR=$LOCO_LR LR=$LR SEED=42 BATCH_SEQ=$BATCH \
      N_STEPS=$N_STEPS STAGED=$STAGED SIGN=1 LOG_EVERY=20 \
      torchrun --standalone --nproc_per_node=8 train_t3.py 2>&1 | tee -a $LOG
}

# Tuning sweep: 200 steps
N_TUNE=200
for B in 4 8 16 32; do
    for LR in 3e-4 1e-3 3e-3; do
        run tune_b${B}_baseline_lr${LR} 0 0 $LR 0 $B $N_TUNE
    done
    for LR in 3e-4 1e-3 3e-3; do
        for LL in 1e-5 1e-4; do
            run tune_b${B}_locoK4_lr${LR}_ll${LL} 1 4 $LR $LL $B $N_TUNE
        done
    done
done

echo "TUNING DONE"
