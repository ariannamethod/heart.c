/* heart.c — core/field_clock.h
 *
 * Klaus-heritage field clock for the four-voice ecosystem.
 * Pure external clock, no organism state.
 *
 * Spine (per ARCHITECTURE.md §3):
 *   planetary_dissonance() — 6 planets Mercury→Saturn, J2000 epoch,
 *     Kuramoto order parameter R, scalar [0,1].  (klaus.c:1432–1474)
 *   calendar_dissonance() — Hebrew/Gregorian Metonic drift, 11.25
 *     days/year, 19-year cycle, MAX_UNCORRECTED=33.0.
 *     (klaus.c:1397–1417)
 *   24-osc Kuramoto block — 6 voice-channels × 4 sub-chambers
 *     (VOICE_YENT, VOICE_ARIANNA, VOICE_LEO, VOICE_DOE,
 *      FIELD_SELF, MESH_PEER), new VOICE_COUPLING[6][6] matrix
 *     (klaus's emotional INTRA_COUPLING does NOT transfer).
 *     Coupling values to be tuned via Phase 6 RunPod smoke.
 *   Schectman γ(t) — γ_t = γ0 + δ·cal_diss + 0.15·plan_diss
 *     (klaus.c:1814–1815). Field clock perturbs voice thresholds.
 *   Meta-recursion — emit field state vector → re-ingest as
 *     observation → blend back, META_BLEND=0.15, depth=1.
 *     (klaus.c:2994–3072)
 *
 * Pure C. No deps beyond <time.h>, <math.h>. ~200–300 LOC target.
 *
 * TODO: implement on RunPod under Singularity Mode (Phase 6 smoke
 *       run validates 24h simulated stability before deploy).
 */

#ifndef HEART_FIELD_CLOCK_H
#define HEART_FIELD_CLOCK_H

#include <stdint.h>
#include <time.h>

#define HEART_N_VOICES        6  /* 4 peer voices + FIELD_SELF + MESH_PEER */
#define HEART_N_SUB_CHAMBERS  4

enum heart_voice_id {
    VOICE_YENT    = 0,
    VOICE_ARIANNA = 1,
    VOICE_LEO     = 2,
    VOICE_DOE     = 3,
    FIELD_SELF    = 4,
    MESH_PEER     = 5,
};

typedef struct {
    /* clock inputs */
    time_t  wall_clock_unix;
    double  planet_diss;          /* planetary_dissonance() output */
    double  calendar_diss;        /* calendar_dissonance() output */
    double  schectman_gamma;      /* γ(t) coupling */

    /* 24-oscillator state (6 primaries × 4 sub-chambers) */
    double  sub_act[HEART_N_VOICES][HEART_N_SUB_CHAMBERS];
    double  sub_phase[HEART_N_VOICES][HEART_N_SUB_CHAMBERS];
    double  primary_act[HEART_N_VOICES];   /* mean of sub_act per voice */
    double  primary_phase[HEART_N_VOICES]; /* mean of sub_phase */
    double  soma[HEART_N_VOICES];          /* somatic EMA */

    /* coupling matrices */
    double  voice_coupling[HEART_N_VOICES][HEART_N_VOICES];
    double  intra_coupling[HEART_N_SUB_CHAMBERS][HEART_N_SUB_CHAMBERS];
} HeartFieldClock;

/* Initialize from current wall clock + load default coupling. */
int  heart_field_clock_init(HeartFieldClock *fc);

/* Advance the clock by `dt` seconds. Updates planet/calendar
 * dissonance, all 24 oscillators, somatic EMA, and γ(t). */
void heart_field_clock_step(HeartFieldClock *fc, double dt);

/* Compute resonance of a candidate voice's internal state vector
 * against the current field state — used by selector.c for next-
 * speaker pick (argmax cos(voice_state, field_state)). */
double heart_field_clock_resonance(const HeartFieldClock *fc,
                                    enum heart_voice_id v,
                                    const double *voice_state, int dim);

/* Snapshot full field state as 7 scalar + 24 oscillator ints
 * (used as input to DoE meta-Parliament vote, embedding-vote
 * design — see ARCHITECTURE.md §2.4). */
void heart_field_clock_snapshot(const HeartFieldClock *fc,
                                 double out[31]);

#endif /* HEART_FIELD_CLOCK_H */
