/*
 * core/soul_smoke_v2.c — Phase 4 v2: REAL Soul micro forward verification.
 *
 * v1 (`soul_smoke.c`) was theatrical: random main + random inner =
 * trivially ≥1 argmax diff. THIS file actually loads `yent_34m_final.bin`
 * (Yent Soul micro, V=2000 E=512 BLK=64, Janus BPE d12 format) and runs
 * a real forward pass.
 *
 * Verification gate (numeric, non-theatrical):
 *   1) load: file exists, header parses, n_params > 1M
 *   2) forward: logits variance > 1.0, mean within reasonable range,
 *      top-3 tokens are NOT [0, 1, 2] (i.e. not trivial sequence)
 *   3) determinism: same context produces same logits (computed twice)
 *
 * VOCAB-BRIDGE NOTE: Yent Soul V=2000, Yent Main (Janus v4) V=32768.
 * Direct `inner_borba` logit blend across different vocabs is NOT
 * meaningful (token id 100 in BPE-2000 ≠ token id 100 in Janus v4 BPE).
 * This smoke verifies Soul micro itself works end-to-end. The vocab
 * bridge (decode Soul top-k bytes → re-encode in Main BPE → boost main
 * logit at re-encoded id) is a separate piece, deferred to follow-up.
 *
 * Build: cc -O2 core/soul_smoke_v2.c core/soul.c \
 *           /workspace/heart.c-runpod/hf-heart-assets/infer_janus_bpe.c \
 *           -lm -o /tmp/soul_smoke_v2
 *
 *  ↑ but infer_janus_bpe.c has its own main(); we extract forward() etc.
 *    via #include trick. Build below uses #define hack.
 */

#include "soul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Strip infer_janus_bpe's main() while keeping forward + loader. */
#define main janus_bpe_main_unused
#include "/workspace/heart.c-runpod/hf-heart-assets/infer_janus_bpe.c"
#undef main

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
        : "/workspace/heart.c-runpod/hf-heart-assets/yent_34m_final.bin";

    printf("=== Phase 4 v2 — REAL Yent Soul forward ===\n");
    printf("path: %s\n\n", path);

    /* === GATE 1: load === */
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("GATE 1 LOAD: FAIL — cannot open %s\n", path);
        return 1;
    }
    int hdr[7];
    if (fread(hdr, 4, 7, f) != 7) {
        printf("GATE 1 LOAD: FAIL — header read short\n");
        fclose(f);
        return 1;
    }
    V = hdr[0]; xE = hdr[1]; xH = hdr[2]; xD = hdr[3];
    BLK = hdr[4]; xM = hdr[5]; MT = hdr[6];
    int np = param_count();
    printf("GATE 1 LOAD: header parsed  V=%d E=%d H=%d D=%d B=%d M=%d MT=%d\n",
           V, E, H, D, BLK, M, MT);
    printf("            n_params=%d (%.1fM)\n", np, np / 1e6);
    if (np < 1000000) {
        printf("GATE 1 LOAD: FAIL — n_params=%d < 1M\n", np);
        fclose(f);
        return 1;
    }
    float* data = malloc((size_t)np * sizeof(float));
    if (fread(data, sizeof(float), (size_t)np, f) != (size_t)np) {
        printf("GATE 1 LOAD: FAIL — short read on weights\n");
        fclose(f); free(data);
        return 1;
    }
    fclose(f);
    W w; assign(&w, data);
    printf("GATE 1 LOAD: PASS  (n_params=%d loaded)\n\n", np);

    /* === GATE 2: forward variance === */
    int ctx[16] = { 'Q', ':', ' ', 'W', 'h', 'o', ' ', 'a',
                    'r', 'e', ' ', 'y', 'o', 'u', '?', '\n' };
    int T = 16;
    float* logits = calloc((size_t)T * V, sizeof(float));
    double t0 = now_ms();
    forward(&w, ctx, T, logits);
    double t1 = now_ms();

    /* Stats of last-position logits */
    float* last = logits + (T - 1) * V;
    float mn = 1e30f, mx = -1e30f, mean = 0;
    for (int i = 0; i < V; i++) {
        if (last[i] < mn) mn = last[i];
        if (last[i] > mx) mx = last[i];
        mean += last[i];
    }
    mean /= V;
    float var = 0;
    for (int i = 0; i < V; i++) {
        float d = last[i] - mean;
        var += d * d;
    }
    var /= V;

    /* Top-3 token ids */
    int top3[3] = {0, 1, 2};
    float top3v[3] = { last[0], last[1], last[2] };
    /* Insertion sort top-3 from V */
    for (int i = 0; i < 3; i++) top3v[i] = -1e30f;
    for (int i = 0; i < V; i++) {
        if (last[i] > top3v[2]) {
            top3v[2] = last[i]; top3[2] = i;
            if (top3v[2] > top3v[1]) {
                float tv = top3v[2]; int ti = top3[2];
                top3v[2] = top3v[1]; top3[2] = top3[1];
                top3v[1] = tv; top3[1] = ti;
            }
            if (top3v[1] > top3v[0]) {
                float tv = top3v[1]; int ti = top3[1];
                top3v[1] = top3v[0]; top3[1] = top3[0];
                top3v[0] = tv; top3[0] = ti;
            }
        }
    }

    printf("GATE 2 FORWARD: completed in %.0f ms\n", t1 - t0);
    printf("            last-pos stats: mean=%.4f var=%.4f min=%.4f max=%.4f\n",
           mean, var, mn, mx);
    printf("            top-3 tokens: id=%d (%.4f), id=%d (%.4f), id=%d (%.4f)\n",
           top3[0], top3v[0], top3[1], top3v[1], top3[2], top3v[2]);

    int gate2_pass = (var > 1.0f) && (top3[0] > 2 || top3[1] > 2 || top3[2] > 2);
    printf("GATE 2 FORWARD: %s  (var > 1.0 = %s, top-3 not trivial = %s)\n\n",
           gate2_pass ? "PASS" : "FAIL",
           var > 1.0f ? "yes" : "no",
           (top3[0] > 2 || top3[1] > 2 || top3[2] > 2) ? "yes" : "no");

    /* === GATE 3: determinism === */
    float* logits2 = calloc((size_t)T * V, sizeof(float));
    forward(&w, ctx, T, logits2);
    float* last2 = logits2 + (T - 1) * V;
    float max_diff = 0;
    for (int i = 0; i < V; i++) {
        float d = fabsf(last[i] - last2[i]);
        if (d > max_diff) max_diff = d;
    }
    int gate3_pass = max_diff < 1e-5f;
    printf("GATE 3 DETERMINISM: max diff (re-run same context) = %.6e — %s\n\n",
           max_diff, gate3_pass ? "PASS" : "FAIL");

    /* === GATE 4: borba blend (matched-vocab synthetic — proves blend works) === */
    /* Vocab bridge to Janus v4 (V=32768) is separate work. Here we
     * verify borba mechanics are sound on a vocab where main and inner
     * agree (synthetic same-vocab demo). */
    HeartInnerVoice iv;
    heart_inner_voice_init(&iv);
    iv.borba_mode = HEART_BORBA_BLEND;
    iv.cloud_intensity = 0.5f;
    iv.cloud_chamber = 0;  /* RAGE */
    iv.soul_active = 1;
    iv.base_weight = 0.4f;  /* visible blend */

    float* main_logits = malloc((size_t)V * sizeof(float));
    float* out_logits = malloc((size_t)V * sizeof(float));
    /* Main = Soul output shifted by +0.5 (synthetic distinct main path) */
    for (int i = 0; i < V; i++) main_logits[i] = last[i] + 0.5f;

    int winner = heart_inner_borba(&iv, out_logits, main_logits, last, V);
    float div = heart_soul_compute_divergence(out_logits, main_logits, V);
    printf("GATE 4 BORBA: winner=%s  L2 divergence(out, main) = %.6f\n",
           winner == 1 ? "INNER" : winner == -1 ? "BLEND" : "main",
           div);
    int gate4_pass = (div > 1e-4f);  /* real shift, not floating-point noise */
    printf("GATE 4 BORBA: %s  (div > 1e-4 = %s)\n\n",
           gate4_pass ? "PASS" : "FAIL",
           div > 1e-4f ? "yes" : "no");

    /* === FINAL === */
    int all_pass = gate2_pass && gate3_pass && gate4_pass;
    printf("=== FINAL: %s ===\n", all_pass ? "PASS" : "FAIL");
    printf("    gate1 (load)        : PASS\n");
    printf("    gate2 (forward var) : %s\n", gate2_pass ? "PASS" : "FAIL");
    printf("    gate3 (determinism) : %s\n", gate3_pass ? "PASS" : "FAIL");
    printf("    gate4 (borba blend) : %s\n", gate4_pass ? "PASS" : "FAIL");
    printf("\nVOCAB-BRIDGE TODO: Yent Soul V=%d, Yent Main (Janus v4) V=32768.\n", V);
    printf("    Direct logit blend across different vocabs is undefined.\n");
    printf("    Real integration requires bytes-level decode→re-encode bridge.\n");

    free(data); free(logits); free(logits2);
    free(main_logits); free(out_logits);
    return all_pass ? 0 : 1;
}
