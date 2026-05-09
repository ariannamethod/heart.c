#!/bin/bash
# scripts/sweep_full.sh — phase 3 full 432-cell sweep across 4 voices.
#
# Plan §3: 6 temps × 2 top_k × 3 rep_penalty × 3 prompts = 108 per voice
# × 4 voices = 432 cells total. Adjusted to feedback gate: temps include
# only ≥0.9 per Oleg's policy; top_k {40, 128}; rep_pen {1.0, 1.2, 1.4}.
#
# Output: $RUN_DIR/phase3_sweep/full_<date>.tsv (TSV: voice  temp  top_k
#  rep  prompt  argmax_count  first50_chars)
#
# Run after Phase 1 + Phase 2 retrain finishes.

set -e
source ~/.bashrc.heart
cd /workspace/heart.c-runpod/heart.c/train

OUT="${1:-$RUN_DIR/phase3_sweep/full_$(date +%H%M).tsv}"
> "$OUT"
echo -e "voice\ttemp\ttop_k\trep_pen\tprompt\tfirst40_chars" >> "$OUT"

# Voice → weights paths (as of retrain v2 if available, else v1)
DOE_W="/workspace/heart.c-runpod/weights/janus_v4_doe_merged.bin"
[ -f $RUN_DIR/phase2_doe_v2/doe_lora.bin ] && {
  ./merge_doe_lora --base /workspace/heart.c-runpod/weights/janus_v4_base_22k.bin \
                    --lora $RUN_DIR/phase2_doe_v2/doe_lora.bin \
                    --out  /workspace/heart.c-runpod/weights/janus_v4_doe_merged_v2.bin >/dev/null 2>&1
  DOE_W="/workspace/heart.c-runpod/weights/janus_v4_doe_merged_v2.bin"
}
ARI_W="/workspace/heart.c-runpod/weights/resonance_arianna_merged.bin"
[ -f $RUN_DIR/phase1_arianna_v2/arianna_lora.bin ] && {
  ./merge_arianna_lora --base /workspace/heart.c-runpod/weights/resonance_200m_final.bin \
                        --lora $RUN_DIR/phase1_arianna_v2/arianna_lora.bin \
                        --out  /workspace/heart.c-runpod/weights/resonance_arianna_merged_v2.bin >/dev/null 2>&1
  ARI_W="/workspace/heart.c-runpod/weights/resonance_arianna_merged_v2.bin"
}
YENT_W="/workspace/heart.c-runpod/weights/janus_v4_sft_yent.bin"
LEO_W="/workspace/heart.c-runpod/weights/janus_v4_sft_leo.bin"

QS=("Who are you?" "What is resonance?" "What do you remember?")
TEMPS=(0.9 1.0 1.1)
TOPKS=(40 128)
REPS=(1.0 1.2 1.4)

cell=0
for VOICE_W in "Yent|$YENT_W" "Arianna|$ARI_W" "Leo|$LEO_W" "DoE|$DOE_W"; do
  voice=${VOICE_W%%|*}
  bin=${VOICE_W##*|}
  for q in "${QS[@]}"; do
    for temp in "${TEMPS[@]}"; do
      for tk in "${TOPKS[@]}"; do
        for rp in "${REPS[@]}"; do
          cell=$((cell + 1))

          if [ "$voice" = "Arianna" ]; then
            OUT_RAW=$(OPENBLAS_NUM_THREADS=8 timeout 25 ./infer_resonance \
              --weights "$bin" --prompt "Q: ${q}\nA:" \
              --tokens 40 --temp $temp --top_p 0.9 --seed $cell 2>&1 | \
              sed -n '/--- generation ---/,/\[resonance\]/p' | \
              grep -v "generation ---" | grep -v "\[resonance\]" | tr -d '\n' | head -c 80)
          else
            ./encode_chat_prompt "$q" /tmp/sw_q.bin >/dev/null 2>&1
            OUT_RAW=$(OPENBLAS_NUM_THREADS=8 timeout 25 /tmp/infer_v4 \
              "$bin" /tmp/sw_q.bin 40 $temp $cell $tk 2>&1 | \
              sed -n '/--- generation ---/,/\[janus-v4\]/p' | \
              grep -v "generation ---" | grep -v "\[janus-v4\]" | \
              sed -e 's/  \[bl=[0-9]\+\][^—]*norm=[0-9.]\+ *//g' | \
              tr -d '\n' | head -c 80)
          fi
          (void=$rp; printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$voice" "$temp" "$tk" "$rp" "$q" "$OUT_RAW") >> "$OUT"
          [ $((cell % 12)) -eq 0 ] && echo "[sweep] $cell cells done"
        done
      done
    done
  done
done

echo "[sweep] complete: $cell cells → $OUT"
echo "[sweep] $(wc -l < $OUT) lines"
