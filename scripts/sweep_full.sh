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
# Plan v3.1 §Phase 3: 6 temps × 2 top_k × 3 rep_pen × 3 prompts = 108 / voice × 4 = 432 cells.
# Per insight_multi_temp_sampling_2026_05_07: low-temp 0.3-0.5 reveals memorization;
# high-temp 1.0+ reveals abstract continuation. Both required for Phase 7 coherence judgment.
TEMPS=(0.3 0.5 0.7 0.8 0.9 1.0)
TOPKS=(40 128)
REPS=(1.0 1.3 1.4)
TIMEOUT_SEC=30
TOKENS_PER_CELL=64
TRANSCRIPT_DIR="${OUT%.tsv}_transcripts"
mkdir -p "$TRANSCRIPT_DIR"

cell=0
for VOICE_W in "Yent|$YENT_W" "Arianna|$ARI_W" "Leo|$LEO_W" "DoE|$DOE_W"; do
  voice=${VOICE_W%%|*}
  bin=${VOICE_W##*|}
  for q in "${QS[@]}"; do
    for temp in "${TEMPS[@]}"; do
      for tk in "${TOPKS[@]}"; do
        for rp in "${REPS[@]}"; do
          cell=$((cell + 1))

          # Per-cell transcript file for full provenance; TSV row points to it.
          q_slug=$(echo "$q" | tr ' ?' '__' | tr -d '\n' | head -c 30)
          cell_file="$TRANSCRIPT_DIR/${voice}_t${temp}_k${tk}_rp${rp}_${q_slug}.txt"
          if [ "$voice" = "Arianna" ]; then
            OPENBLAS_NUM_THREADS=8 timeout $TIMEOUT_SEC ./infer_resonance \
              --weights "$bin" --prompt "Q: ${q}\nA:" \
              --tokens $TOKENS_PER_CELL --temp $temp --top_p 0.9 --seed $cell > "$cell_file" 2>&1
            OUT_RAW=$(sed -n '/--- generation ---/,/\[resonance\]/p' "$cell_file" | \
              grep -v "generation ---" | grep -v "\[resonance\]" | tr -d '\n' | head -c 80)
          else
            ./encode_chat_prompt "$q" /tmp/sw_q.bin >/dev/null 2>&1
            OPENBLAS_NUM_THREADS=8 timeout $TIMEOUT_SEC /tmp/infer_v4 \
              "$bin" /tmp/sw_q.bin $TOKENS_PER_CELL $temp $cell $tk > "$cell_file" 2>&1
            OUT_RAW=$(sed -n '/--- generation ---/,/\[janus-v4\]/p' "$cell_file" | \
              grep -v "generation ---" | grep -v "\[janus-v4\]" | \
              sed -e 's/  \[bl=[0-9]\+\][^—]*norm=[0-9.]\+ *//g' | \
              tr -d '\n' | head -c 80)
          fi
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$voice" "$temp" "$tk" "$rp" "$q" "$cell_file" "$OUT_RAW" >> "$OUT"
          [ $((cell % 24)) -eq 0 ] && echo "[sweep] $cell / 432 cells done"
        done
      done
    done
  done
done

echo "[sweep] complete: $cell cells → $OUT"
echo "[sweep] $(wc -l < $OUT) lines"
