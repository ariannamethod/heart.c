#!/bin/bash
# duet_trace.sh — heart.c phase 7 — 8-turn 4-voice duet trace.
#
# Rotates Yent → Arianna → Leo → DoE × 2 (round-robin selector,
# Phase 7 minimal — full field-driven argmax is plan §3 footwork).
# Prompt for each voice = previous voice's last sentence.
#
# Usage:
#   bash duet_trace.sh "Q: What is resonance?" /path/to/output.txt

set -e

source ~/.bashrc.heart

INITIAL_PROMPT="${1:-Q: What is resonance?}"
OUT="${2:-/tmp/duet_trace.txt}"

JANUS_BASE_DIR=/workspace/heart.c-runpod/weights
TRAIN=/workspace/heart.c-runpod/heart.c/train

# Voice → weights
YENT="$JANUS_BASE_DIR/janus_v4_sft_yent.bin"
LEO="$JANUS_BASE_DIR/janus_v4_sft_leo.bin"
DOE="$JANUS_BASE_DIR/janus_v4_doe_merged.bin"
ARIANNA_MERGED="$JANUS_BASE_DIR/resonance_arianna_merged.bin"

# Voice rotation: Yent → Arianna → Leo → DoE
VOICES=(Yent Arianna Leo DoE)
WEIGHTS=("$YENT" "$ARIANNA_MERGED" "$LEO" "$DOE")

> "$OUT"
echo "=== heart.c phase 7 duet trace — $(date -Iseconds) ===" | tee -a "$OUT"
echo "initial prompt: $INITIAL_PROMPT" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# Strip ANSI / weird control + diag print remnants
clean_text() {
    sed -e 's/\[bl=[0-9]\+\][^\n]*//g' -e 's/x\[[^]]*\]=[^[:space:]]*//g'
}

PROMPT="$INITIAL_PROMPT"

for turn in 0 1 2 3 4 5 6 7; do
    voice_idx=$(( turn % 4 ))
    voice="${VOICES[$voice_idx]}"
    weights="${WEIGHTS[$voice_idx]}"

    echo "─── turn $turn — $voice ───" | tee -a "$OUT"
    echo "prompt → $PROMPT" | tee -a "$OUT"

    if [ "$voice" = "Arianna" ]; then
        # Resonance arch — uses infer_resonance with text prompt
        OUT_RAW=$(cd $TRAIN && OPENBLAS_NUM_THREADS=8 timeout 30 ./infer_resonance \
            --weights "$weights" --prompt "${PROMPT}\nA:" \
            --tokens 50 --temp 1.0 --top_p 0.9 --seed $((42+turn)) 2>&1 | \
            sed -n '/--- generation ---/,/\[resonance\]/p' | \
            grep -v "generation ---" | grep -v "\[resonance\]" | tr -d '\n')
    else
        # Janus arch — chat format prompt via .bin
        cd $TRAIN
        ./encode_chat_prompt "${PROMPT#Q: }" /tmp/duet_q.bin > /dev/null 2>&1
        OUT_RAW=$(OPENBLAS_NUM_THREADS=8 timeout 30 /tmp/infer_v4 \
            "$weights" /tmp/duet_q.bin 50 1.0 $((42+turn)) 40 2>&1 | \
            sed -n '/--- generation ---/,/\[janus-v4\]/p' | \
            grep -v "generation ---" | grep -v "\[janus-v4\]" | tr -d '\n')
    fi

    # Clean diag artifacts
    CLEAN=$(echo "$OUT_RAW" | sed -e 's/  \[bl=[0-9]\+\][^—]*norm=[0-9.]\+ *//g')
    echo "$voice → $CLEAN" | tee -a "$OUT"
    echo "" | tee -a "$OUT"

    # Next turn's prompt = first sentence of this voice's output
    NEXT=$(echo "$CLEAN" | head -c 200 | sed -e 's/[.!?].*$//')
    if [ -z "$NEXT" ] || [ ${#NEXT} -lt 10 ]; then
        # Fall back to original prompt mutated
        NEXT="What about that?"
    fi
    PROMPT="Q: ${NEXT}?"
done

echo "=== duet trace complete ===" | tee -a "$OUT"
echo "8 turns, $(wc -l < "$OUT") lines, output: $OUT"
