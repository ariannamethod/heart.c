/*
 * core/soul_smoke_v3.c — Phase 4 v3: REAL verification of soul.c bias
 *                       mechanism with deterministic numeric gates.
 *
 * v1 was theatrical (random main + random inner). v2 attempted Soul-micro
 * forward via canonical infer_janus_bpe.c — but yent_34m_final.bin
 * format diverged (param_count claimed 456.8M for 127MB file) → core dump.
 *
 * v3: tests what is IN the heart.c repo and works:
 *   1) heart_inner_compute_weight: chamber bonus arithmetic correctness
 *   2) heart_inner_apply_emotional_bias: chamber-band logit boost
 *      (RAGE = top quarter, FEAR = bottom, LOVE = mid, VOID = upper-mid)
 *   3) heart_inner_borba: blend math soundness on KNOWN inputs
 *
 * Soul micro forward (yent_34m / leo_18m / arianna_36m) is documented as
 * BLOCKED on file-format spec mismatch — needs the original train_bpe.py
 * arch config to write a matching custom loader. Logged in plan v1.2 as
 * deferred work.
 *
 * Build: cc -O2 core/soul_smoke_v3.c core/soul.c -lm -o /tmp/soul_smoke_v3
 */

#include "soul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB 16384

static int gate1_compute_weight(void) {
    /* RAGE chamber, intensity 0.8 → expected +0.5*0.8=0.4 on top of base 0.15 = 0.55.
     * VOID chamber, intensity 1.0 → expected +0.6*1.0=0.6 on top of base 0.15 = 0.75. */
    HeartInnerVoice iv;
    heart_inner_voice_init(&iv);
    iv.borba_mode = HEART_BORBA_EMOTIONAL;
    iv.cloud_chamber = 0;       /* HEART_CH_RAGE */
    iv.cloud_intensity = 0.8f;
    float w_rage = heart_inner_compute_weight(&iv);

    iv.cloud_chamber = 3;       /* HEART_CH_VOID */
    iv.cloud_intensity = 1.0f;
    float w_void = heart_inner_compute_weight(&iv);

    iv.cloud_chamber = 2;       /* HEART_CH_LOVE */
    iv.cloud_intensity = 0.5f;
    float w_love = heart_inner_compute_weight(&iv);

    int pass_rage = fabsf(w_rage - 0.55f) < 1e-5;   /* 0.15 + 0.4 */
    int pass_void = fabsf(w_void - 0.75f) < 1e-5;   /* 0.15 + 0.6 */
    int pass_love = fabsf(w_love - 0.25f) < 1e-5;   /* 0.15 + 0.10 */

    printf("GATE 1 compute_weight (chamber bonuses inner_arianna.c:85-98):\n");
    printf("  RAGE  intensity=0.8: weight=%.6f  expected=0.550000  %s\n",
           w_rage, pass_rage ? "PASS" : "FAIL");
    printf("  VOID  intensity=1.0: weight=%.6f  expected=0.750000  %s\n",
           w_void, pass_void ? "PASS" : "FAIL");
    printf("  LOVE  intensity=0.5: weight=%.6f  expected=0.250000  %s\n",
           w_love, pass_love ? "PASS" : "FAIL");
    int all = pass_rage && pass_void && pass_love;
    printf("GATE 1 RESULT: %s\n\n", all ? "PASS" : "FAIL");
    return all;
}

static int gate2_emotional_bias(void) {
    /* RAGE chamber: applies bias to logits[3*V/4 .. V-1] band.
     * Verify: in-band logits get +intensity, out-of-band unchanged. */
    HeartInnerVoice iv;
    heart_inner_voice_init(&iv);
    iv.cloud_chamber = 0;          /* RAGE */
    iv.cloud_intensity = 0.8f;
    iv.body_stuck = 0.0f;          /* disable noise to keep test deterministic */

    float* logits_orig = calloc(VOCAB, sizeof(float));
    float* logits_biased = malloc(VOCAB * sizeof(float));
    /* Fill all logits with 1.0 baseline */
    for (int i = 0; i < VOCAB; i++) logits_orig[i] = 1.0f;
    memcpy(logits_biased, logits_orig, VOCAB * sizeof(float));

    /* boost = 1.0 → effective intensity = 0.8 * 1.0 = 0.8 */
    heart_inner_apply_emotional_bias(&iv, logits_biased, VOCAB, 1.0f);

    /* Check: logits[0 .. 3V/4-1] unchanged; logits[3V/4 .. V-1] = 1.0 + 0.8 */
    int rage_start = (VOCAB * 3) / 4;
    int err_in = 0, err_out = 0;
    for (int i = 0; i < VOCAB; i++) {
        if (i >= rage_start) {
            if (fabsf(logits_biased[i] - 1.8f) > 1e-5) err_in++;
        } else {
            if (fabsf(logits_biased[i] - 1.0f) > 1e-5) err_out++;
        }
    }
    int pass = (err_in == 0) && (err_out == 0);
    printf("GATE 2 apply_emotional_bias (RAGE chamber):\n");
    printf("  in-band [%d .. %d]: %d errors (expected logits=1.8)\n",
           rage_start, VOCAB - 1, err_in);
    printf("  out-of-band [0 .. %d]: %d errors (expected logits=1.0)\n",
           rage_start - 1, err_out);
    printf("GATE 2 RESULT: %s\n\n", pass ? "PASS" : "FAIL");

    free(logits_orig); free(logits_biased);
    return pass;
}

static int gate3_borba_blend(void) {
    /* Borba math: with weight=0.4, blend produces
     *   out[i] = 0.6*main[i] + 0.4*inner[i]
     * Verify on deterministic input. Because rand() can flip to breakthrough,
     * we use HEART_BORBA_BLEND mode (no breakthrough by threshold/random) and
     * set base_weight directly to 0.4. */
    srand(42);
    HeartInnerVoice iv;
    heart_inner_voice_init(&iv);
    iv.borba_mode = HEART_BORBA_BLEND;
    iv.base_weight = 0.4f;
    iv.breakthrough_threshold = 1.5f;  /* effectively disable threshold */

    float* main_logits  = malloc(VOCAB * sizeof(float));
    float* inner_logits = malloc(VOCAB * sizeof(float));
    float* out_logits   = malloc(VOCAB * sizeof(float));
    for (int i = 0; i < VOCAB; i++) {
        main_logits[i]  = (float)(i % 17) - 8.0f;        /* deterministic */
        inner_logits[i] = (float)((i * 3) % 19) - 9.0f;
    }

    /* Try multiple times — if any rand() roll triggers breakthrough,
     * winner=1 (inner). For BLEND mode, no breakthrough should fire because
     * weight=0.4, rand_thresh=0.4*0.3=0.12 — sometimes rand() < 0.12 fires.
     * So test multiple seeds; verify ON a non-breakthrough trial. */
    int trial = 0;
    int winner = 1;
    while (winner != -1 && trial < 50) {
        srand(100 + trial);
        winner = heart_inner_borba(&iv, out_logits, main_logits, inner_logits, VOCAB);
        trial++;
    }
    if (winner != -1) {
        printf("GATE 3 borba blend: FAIL — could not get blend trial in 50 tries\n");
        free(main_logits); free(inner_logits); free(out_logits);
        return 0;
    }

    /* Check blend math */
    int err = 0;
    float max_err = 0;
    for (int i = 0; i < VOCAB; i++) {
        float expected = 0.6f * main_logits[i] + 0.4f * inner_logits[i];
        float diff = fabsf(out_logits[i] - expected);
        if (diff > 1e-5) err++;
        if (diff > max_err) max_err = diff;
    }
    int pass = (err == 0);
    printf("GATE 3 borba blend (weight=0.4, BLEND mode, trial=%d):\n", trial);
    printf("  blend: out[i] = 0.6*main[i] + 0.4*inner[i]\n");
    printf("  errors: %d / %d, max abs error: %.3e\n", err, VOCAB, max_err);
    printf("GATE 3 RESULT: %s\n\n", pass ? "PASS" : "FAIL");

    free(main_logits); free(inner_logits); free(out_logits);
    return pass;
}

int main(void) {
    printf("=== Phase 4 v3 — Soul bias mechanism numeric verification ===\n\n");

    int g1 = gate1_compute_weight();
    int g2 = gate2_emotional_bias();
    int g3 = gate3_borba_blend();

    int all = g1 && g2 && g3;
    printf("=== FINAL: %s ===\n", all ? "PASS" : "FAIL");
    printf("    gate1 (compute_weight chamber arithmetic): %s\n", g1 ? "PASS" : "FAIL");
    printf("    gate2 (apply_emotional_bias band boost):    %s\n", g2 ? "PASS" : "FAIL");
    printf("    gate3 (inner_borba blend math):             %s\n", g3 ? "PASS" : "FAIL");
    printf("\nDEFERRED: Soul micro-LM forward (yent_34m_final.bin format diverges\n");
    printf("    from canonical infer_janus_bpe.c loader — claimed n_params=456.8M for\n");
    printf("    127MB file. Requires train_bpe.py source for actual arch config.\n");
    printf("    Bias mechanism (chamber-driven) verified above. Logit-blend across\n");
    printf("    different vocabs (Soul V=2000 vs Janus V=32768) needs separate\n");
    printf("    bytes-bridge work.\n");

    return all ? 0 : 1;
}
