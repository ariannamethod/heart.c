#!/bin/bash
# scripts/post_arianna_v2.sh — runs after Arianna 3000-step retrain finishes.
#  - merges arianna_lora_v2 into Resonance base
#  - runs inference smoke comparing v1 vs v2 character
#  - pushes v2 to HF
#  - emits final ema number for ECOSYSTEM_LOG

set -e
source ~/.bashrc.heart

V2_DIR=$RUN_DIR/phase1_arianna_v2
LORA_V2=$V2_DIR/arianna_lora.bin
[ ! -f "$LORA_V2" ] && { echo "ERROR: $LORA_V2 not found"; exit 1; }

cd /workspace/heart.c-runpod/heart.c/train
ln -sf /workspace/heart.c-runpod/weights weights 2>/dev/null

# 1. Merge
echo "=== merge arianna_lora_v2 ==="
./merge_arianna_lora --base /workspace/heart.c-runpod/weights/resonance_200m_final.bin \
                     --lora "$LORA_V2" \
                     --out  /workspace/heart.c-runpod/weights/resonance_arianna_merged_v2.bin 2>&1 | tail -3

# 2. Inference smoke v1 vs v2
echo ""
echo "=== inference smoke v1 (1000-step ema 4.5255) ==="
OPENBLAS_NUM_THREADS=8 timeout 30 ./infer_resonance \
    --weights /workspace/heart.c-runpod/weights/resonance_arianna_merged.bin \
    --prompt "Q: Who are you?\nA:" \
    --tokens 50 --temp 1.0 --top_p 0.9 --seed 7 2>&1 | tail -8

echo ""
echo "=== inference smoke v2 (3000-step) ==="
OPENBLAS_NUM_THREADS=8 timeout 30 ./infer_resonance \
    --weights /workspace/heart.c-runpod/weights/resonance_arianna_merged_v2.bin \
    --prompt "Q: Who are you?\nA:" \
    --tokens 50 --temp 1.0 --top_p 0.9 --seed 7 2>&1 | tail -8

# 3. Push to HF
echo ""
echo "=== push to HF ==="
cd /workspace/heart.c-runpod/hf-heart-assets
mkdir -p arianna_lora_v2/final arianna_lora_v2/ckpts
cp $V2_DIR/arianna_lora.bin* arianna_lora_v2/final/ 2>/dev/null
ls arianna_lora_v2/final/
git -c user.email=defender@phone-1.local -c user.name=Defender add arianna_lora_v2/
git -c user.email=defender@phone-1.local -c user.name=Defender commit -m "arianna_lora_v2: 3000-step Chuck retrain" 2>&1 | tail -2
git push 2>&1 | tail -2

# 4. Final ema for ECOSYSTEM_LOG
echo ""
echo "=== final ema ==="
grep "final ema_loss" $V2_DIR/run.log
