#!/bin/bash
set -e
cd /root/modded-nanogpt
source /root/.venv-loco/bin/activate
LOG_DIR=${LOG_DIR:-final_logs}
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
    echo "=== $NAME B=$BATCH K=$K LR=$LR LL=$LOCO_LR N=$N_STEPS ===" | tee -a $LOG
    LOCO_STEPS=$K LOCO_LR=$LOCO_LR LR=$LR SEED=42 BATCH_SEQ=$BATCH \
      N_STEPS=$N_STEPS STAGED=$STAGED SIGN=1 LOG_EVERY=20 \
      torchrun --standalone --nproc_per_node=8 train_t3.py 2>&1 | tee -a $LOG
}

N=500
# batch 8: tuned LR=5.81e-4 (baseline), LR=4.03e-4 LOCO_LR=1.19e-7 (loco)
run b8_baseline 0 0 5.81e-4 0 8 $N
for K in 1 4 16 64 256 1024; do
    run b8_locoK${K} 1 $K 4.03e-4 1.19e-7 8 $N
done

# batch 32: tuned LR=4.13e-4 (baseline), LR=4.03e-4 LOCO_LR=1.19e-7 (loco)
run b32_baseline 0 0 4.13e-4 0 32 $N
for K in 1 4 16 64 256; do
    run b32_locoK${K} 1 $K 4.03e-4 1.19e-7 32 $N
done

# batch 128: tuned LR=7.95e-4 (baseline), LR=7.95e-4 LOCO_LR=3.59e-4 (loco)
run b128_baseline 0 0 7.95e-4 0 128 $N
for K in 1 4 16 64; do
    run b128_locoK${K} 1 $K 7.95e-4 3.59e-4 128 $N
done

echo "DONE"
