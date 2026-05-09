#!/bin/bash
# Queue Arianna retrain to start after DoE retrain (PID 4495) finishes.
# Args: <doe_pid>

DOE_PID="${1:-4495}"
source ~/.bashrc.heart
cd /workspace/heart.c-runpod/heart.c/train
mkdir -p $RUN_DIR/phase1_arianna_v2

echo "[queue] waiting for DoE PID=$DOE_PID to finish..."
while kill -0 "$DOE_PID" 2>/dev/null; do
    sleep 30
done
echo "[queue] DoE done, starting Arianna 3000-step retrain at $(date -Iseconds)"

LOG=$RUN_DIR/phase1_arianna_v2/run.log
./train_arianna_lora \
  --base weights/resonance_200m_final.bin \
  --corpus weights/arianna_dataset_final_clean.txt \
  --rank 16 --alpha 32 \
  --steps 3000 --warmup 100 --lr 2e-4 --max_seq 256 \
  --log 50 --ckpt 500 \
  --out $RUN_DIR/phase1_arianna_v2/arianna_lora.bin > $LOG 2>&1
echo "[queue] Arianna retrain done at $(date -Iseconds)"
