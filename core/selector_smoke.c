/*
 * core/selector_smoke.c — heart.c phase 7b field-driven next-speaker.
 *
 * ARCHITECTURE.md §1: "Field selects next-speaker by
 *                     argmax(cos(voice_state, field_state))."
 *
 * Replaces the round-robin in scripts/duet_trace.sh with real
 * resonance-driven selection. Voice personality vectors map plan
 * §2 character to an 8-dim space; field state derived from current
 * time as a Klaus-clock proxy. Real heart_main wires this to the
 * field_clock 24-osc Kuramoto state directly.
 *
 * Build: cc -O2 core/selector_smoke.c -lm -o /tmp/selector_smoke
 * Run:   /tmp/selector_smoke [n_turns=8]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VEC_DIM 8

/* Personality vectors per ARCHITECTURE.md §2 character lines.
 * Dims: [0]sardonic [1]tender [2]technical [3]mystical
 *       [4]childlike [5]expert [6]anchor [7]field-aware */
static const float VOICE_VEC[4][VEC_DIM] = {
    /* Yent    — sardonic anchor + jannus-r 12-step (§2.1) */
    { 0.85, 0.10, 0.30, 0.50, 0.05, 0.40, 0.95, 0.60 },
    /* Arianna — concentrated depth + Resonance dual-attn (§2.2) */
    { 0.20, 0.70, 0.50, 0.65, 0.10, 0.50, 0.75, 0.90 },
    /* Leo     — child-philosopher + Janus 170M (§2.3) */
    { 0.15, 0.45, 0.40, 0.30, 0.95, 0.35, 0.50, 0.55 },
    /* DoE     — parliament-of-experts (§2.4) */
    { 0.30, 0.25, 0.55, 0.20, 0.05, 0.95, 0.40, 0.50 },
};
static const char* VOICE_NAMES[4] = { "Yent", "Arianna", "Leo", "DoE" };

static void field_state_at(double t_seconds, float out[VEC_DIM]) {
    /* Synth field state — sin/cos walk over 8 dims at different phase
     * offsets. Real heart_main reads from field_clock 24-osc Kuramoto. */
    double phase = fmod(t_seconds * 0.1, 2.0 * M_PI);
    out[0] = (float)(0.5 + 0.5 * sin(phase + 0.0));
    out[1] = (float)(0.5 + 0.5 * sin(phase + 0.7));
    out[2] = (float)(0.5 + 0.5 * sin(phase + 1.4));
    out[3] = (float)(0.5 + 0.5 * sin(phase + 2.1));
    out[4] = (float)(0.5 + 0.5 * sin(phase + 2.8));
    out[5] = (float)(0.5 + 0.5 * sin(phase + 3.5));
    out[6] = (float)(0.5 + 0.5 * sin(phase + 4.2));
    out[7] = (float)(0.5 + 0.5 * sin(phase + 4.9));
}

static float cos_sim(const float a[VEC_DIM], const float b[VEC_DIM]) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < VEC_DIM; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return 0;
    return dot / (sqrtf(na) * sqrtf(nb));
}

int main(int argc, char** argv) {
    int n_turns = (argc > 1) ? atoi(argv[1]) : 8;
    if (n_turns <= 0) n_turns = 8;

    printf("# heart.c phase 7b — field-driven next-speaker selector\n");
    printf("# argmax(cos(voice_state, field_state)) per ARCHITECTURE.md §1\n");
    printf("# n_turns=%d, voice dim=%d\n\n", n_turns, VEC_DIM);

    double t = (double)time(NULL);
    int counts[4] = {0};

    for (int turn = 0; turn < n_turns; turn++) {
        float field[VEC_DIM];
        field_state_at(t + turn * 60.0, field);

        float scores[4];
        int best = 0;
        for (int v = 0; v < 4; v++) {
            scores[v] = cos_sim(VOICE_VEC[v], field);
            if (v > 0 && scores[v] > scores[best]) best = v;
        }
        counts[best]++;

        printf("turn %d  Y=%.4f A=%.4f L=%.4f D=%.4f → %s\n",
               turn, scores[0], scores[1], scores[2], scores[3],
               VOICE_NAMES[best]);
    }

    printf("\n# turn distribution: Yent=%d Arianna=%d Leo=%d DoE=%d (of %d)\n",
           counts[0], counts[1], counts[2], counts[3], n_turns);
    return 0;
}
