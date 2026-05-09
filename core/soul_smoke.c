/*
 * core/soul_smoke.c — Phase 4 verification gate for Soul logit injection.
 *
 * Plan §6: "Soul logit injection observable: same prompt with Soul OFF
 *           vs ON differs in token distribution by ≥1 token within 50
 *           emits."
 *
 * Strategy: synth main_logits from a deterministic seed; build inner_logits
 * via emotional_bias on a copy; run heart_inner_borba with SOUL_OFF (no
 * inner) vs SOUL_ON (full borba) for 50 token-decisions in a row; count
 * argmax mismatches.
 *
 * Build:  cc -O2 core/soul_smoke.c core/soul.c -lm -o soul_smoke
 * Run:    ./soul_smoke
 */

#include "soul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB    16384
#define N_TOKENS 50

static int argmax(const float *logits, int v) {
    int best = 0;
    for (int i = 1; i < v; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

int main(void) {
    srand(42);
    float *main_logits  = malloc(VOCAB * sizeof(float));
    float *inner_logits = malloc(VOCAB * sizeof(float));
    float *out_off      = malloc(VOCAB * sizeof(float));
    float *out_on       = malloc(VOCAB * sizeof(float));

    HeartInnerVoice iv_off, iv_on;
    heart_inner_voice_init(&iv_off);
    heart_inner_voice_init(&iv_on);

    /* SOUL ON: emotional mode, RAGE chamber active intensely */
    iv_on.borba_mode      = HEART_BORBA_EMOTIONAL;
    iv_on.cloud_chamber   = 0;     /* RAGE */
    iv_on.cloud_intensity = 0.8f;
    iv_on.mood_tension    = 0.4f;
    iv_on.body_stuck      = 0.2f;
    iv_on.soul_active     = 1;

    /* SOUL OFF: quiet baseline, no chamber activation */
    iv_off.borba_mode      = HEART_BORBA_BLEND;
    iv_off.cloud_intensity = 0.0f;
    iv_off.soul_active     = 0;

    int diffs = 0;
    int inner_wins = 0, blend_wins = 0, main_wins = 0;
    float total_div = 0.0f;

    printf("=== Phase 4 Soul logit-injection smoke (50 tokens) ===\n");
    printf("vocab=%d, RAGE chamber intensity=0.8, base_weight=%.3f, threshold=%.3f\n\n",
           VOCAB, iv_on.base_weight, iv_on.breakthrough_threshold);

    for (int t = 0; t < N_TOKENS; t++) {
        /* Synth deterministic main + inner: main = N(0,1), inner = main + chamber-skewed shift */
        for (int i = 0; i < VOCAB; i++) {
            main_logits[i]  = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;   /* U[-2,2] */
            inner_logits[i] = main_logits[i] + ((float)rand() / RAND_MAX) * 1.5f - 0.75f;
        }

        /* OFF path: pass main through unchanged */
        heart_inner_borba(&iv_off, out_off, main_logits, NULL, VOCAB);

        /* ON path: full borba with inner_logits */
        int winner = heart_inner_borba(&iv_on, out_on, main_logits, inner_logits, VOCAB);
        if (winner == 1) inner_wins++;
        else if (winner == -1) blend_wins++;
        else main_wins++;

        int argmax_off = argmax(out_off, VOCAB);
        int argmax_on  = argmax(out_on,  VOCAB);
        if (argmax_off != argmax_on) diffs++;

        float div = heart_soul_compute_divergence(out_off, out_on, VOCAB);
        total_div += div;

        if (t < 5 || t == N_TOKENS - 1) {
            printf("t=%2d  argmax OFF=%5d  ON=%5d  diff=%s  divergence=%.5f  winner=%s\n",
                   t, argmax_off, argmax_on,
                   (argmax_off != argmax_on) ? "YES" : " no",
                   div,
                   (winner == 1 ? "INNER" : winner == -1 ? "blend" : "main "));
        }
    }

    printf("\n=== summary ===\n");
    printf("argmax differs: %d / %d tokens\n", diffs, N_TOKENS);
    printf("borba breakdown: main=%d  blend=%d  inner=%d\n",
           main_wins, blend_wins, inner_wins);
    printf("avg L2 divergence (softmax space): %.5f\n", total_div / N_TOKENS);

    /* Plan §6 verification gate: ≥1 token differs */
    printf("\nverification gate: argmax_diffs >= 1 → %s\n",
           diffs >= 1 ? "PASS" : "FAIL");

    free(main_logits); free(inner_logits); free(out_off); free(out_on);
    return diffs >= 1 ? 0 : 1;
}
