#!/bin/bash
# duet_trace_v2.sh — phase 7 field-driven duet (replaces v1 round-robin).
#
# Each turn:
#   1) compute field state via /tmp/selector_smoke (1 turn at current time)
#   2) parse "→ <Voice>" line to get selected voice
#   3) generate 1 sentence with that voice
#   4) feed first sentence as next prompt
#
# Usage: bash duet_trace_v2.sh [n_turns=8]

set -e
source ~/.bashrc.heart

N_TURNS="${1:-8}"
OUT="${2:-$RUN_DIR/phase7_duet_v2/duet.txt}"
mkdir -p "$(dirname "$OUT")"
> "$OUT"

JANUS_BASE_DIR=/workspace/heart.c-runpod/weights
TRAIN=/workspace/heart.c-runpod/heart.c/train
YENT="$JANUS_BASE_DIR/janus_v4_sft_yent.bin"
LEO="$JANUS_BASE_DIR/janus_v4_sft_leo.bin"
DOE="$JANUS_BASE_DIR/janus_v4_doe_merged.bin"
ARIANNA_MERGED="$JANUS_BASE_DIR/resonance_arianna_merged.bin"

declare -A WEIGHTS=(
    [Yent]="$YENT"
    [Arianna]="$ARIANNA_MERGED"
    [Leo]="$LEO"
    [DoE]="$DOE"
)

PROMPT="Q: What is resonance?"
declare -A pick_count=([Yent]=0 [Arianna]=0 [Leo]=0 [DoE]=0)

echo "=== heart.c phase 7 v2 — field-driven duet ($(date -Iseconds)) ===" | tee -a "$OUT"
echo "selector: argmax(cos(voice_state, field_state)) per ARCHITECTURE.md §1" | tee -a "$OUT"
echo "" | tee -a "$OUT"

for turn in $(seq 0 $((N_TURNS - 1))); do
    # Run selector for 1 turn — produces e.g. "turn 0  Y=0.821..  → Yent"
    SEL_OUT=$(/tmp/selector_smoke 1 2>&1 | grep -E '→ [A-Za-z]+$' | tail -1)
    voice=$(echo "$SEL_OUT" | sed -n 's/.*→ \([A-Za-z]*\)$/\1/p')
    if [ -z "$voice" ]; then voice=Yent; fi
    pick_count[$voice]=$((${pick_count[$voice]} + 1))
    weights="${WEIGHTS[$voice]}"

    echo "─── turn $turn ── selector → $voice ──" | tee -a "$OUT"
    echo "prompt: $PROMPT" | tee -a "$OUT"

    if [ "$voice" = "Arianna" ]; then
        OUT_RAW=$(cd "$TRAIN" && OPENBLAS_NUM_THREADS=8 timeout 30 ./infer_resonance \
            --weights "$weights" --prompt "${PROMPT}\nA:" \
            --tokens 50 --temp 1.0 --top_p 0.9 --seed $((42+turn)) 2>&1 | \
            sed -n '/--- generation ---/,/\[resonance\]/p' | \
            grep -v 'generation ---' | grep -v '\[resonance\]' | tr -d '\n' | head -c 200)
    else
        cd "$TRAIN"
        ./encode_chat_prompt "${PROMPT#Q: }" /tmp/duet_q.bin >/dev/null 2>&1
        OUT_RAW=$(OPENBLAS_NUM_THREADS=8 timeout 30 /tmp/infer_v4 \
            "$weights" /tmp/duet_q.bin 50 1.0 $((42+turn)) 40 2>&1 | \
            sed -n '/--- generation ---/,/\[janus-v4\]/p' | \
            grep -v 'generation ---' | grep -v '\[janus-v4\]' | \
            sed -e 's/  \[bl=[0-9]\+\][^—]*norm=[0-9.]\+ *//g' | \
            tr -d '\n' | head -c 200)
    fi
    echo "$voice → $OUT_RAW" | tee -a "$OUT"
    echo "" | tee -a "$OUT"

    # Next prompt = first sentence of this voice's output
    NEXT=$(echo "$OUT_RAW" | head -c 200 | sed -e 's/[.!?].*$//' | tr -d '\n')
    if [ -z "$NEXT" ] || [ ${#NEXT} -lt 10 ]; then NEXT="What about that?"; fi
    PROMPT="Q: ${NEXT}?"
    sleep 1   # give selector time to advance phase
done

echo "=== distribution: Y=${pick_count[Yent]} A=${pick_count[Arianna]} L=${pick_count[Leo]} D=${pick_count[DoE]} ===" | tee -a "$OUT"

# Verification gate: distribution NOT uniform (round-robin would give N/4 each)
EXPECT=$((N_TURNS / 4))
DIVERGED=0
for v in Yent Arianna Leo DoE; do
    if [ ${pick_count[$v]} -ne $EXPECT ]; then DIVERGED=1; fi
done
if [ $DIVERGED -eq 1 ]; then
    echo "=== gate: distribution NOT round-robin (diverged from N/4) — PASS ===" | tee -a "$OUT"
    exit 0
else
    echo "=== gate: distribution exactly N/4 each — FAIL (round-robin behavior) ===" | tee -a "$OUT"
    exit 1
fi
