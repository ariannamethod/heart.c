/* heart.c — core/parliament.h
 *
 * DoE-heritage meta-FieldLayer Parliament for embedding-vote design
 * (ARCHITECTURE.md §2.4 v1.1 — RAM-safe sequential-activation).
 *
 * Lifted from doe.c:
 *   parliament_elect()       lines 1833–1877 — variable-k vote
 *   notorch_step()           lines 1886–1922 — Hebbian online plast.
 *   try_apoptosis()          lines 1972–1985 — expert death rule
 *
 * IMPORTANT: input domain in heart.c is per-CANDIDATE embedding-
 * plus-field (32 SPA + 7 scalars + 24 osc = 63 floats), NOT
 * per-token hidden-state. Row-projection 63→16 lives in this layer.
 */
#ifndef HEART_PARLIAMENT_H
#define HEART_PARLIAMENT_H

#include <stdint.h>

#define HEART_PARL_VOTE_DIM     63   /* 32 SPA + 7 scalar + 24 osc */
#define HEART_PARL_PROJ_DIM     16
#define HEART_PARL_MIN_EXPERTS  2
#define HEART_PARL_MAX_EXPERTS  16
#define HEART_PARL_VITALITY_LO  0.1f
#define HEART_PARL_DEATH_STREAK 8

typedef struct {
    int   alive;
    float w_vote[HEART_PARL_PROJ_DIM];
    float vitality;
    int   low_vitality_streak;
    int   tokens_seen;
    int   age;
    /* Hebbian LoRA pair — adapter on the projection */
    float lora_A[HEART_PARL_PROJ_DIM * HEART_PARL_VOTE_DIM];
    float lora_B[HEART_PARL_PROJ_DIM];
} HeartParlExpert;

typedef struct {
    HeartParlExpert experts[HEART_PARL_MAX_EXPERTS];
    int n_alive;
    float consensus_ema;            /* 0.9*old + 0.1*new */
    float lr;                       /* 0.01 */
    float decay;                    /* 0.999 */
    float projection[HEART_PARL_PROJ_DIM * HEART_PARL_VOTE_DIM]; /* deterministic seed */
} HeartParliament;

int   heart_parliament_init(HeartParliament *p, uint32_t seed);

/* Vote on N candidates, return chosen index + softmax weights.
 * Inputs: per-candidate vote vectors of HEART_PARL_VOTE_DIM. */
int   heart_parliament_elect(HeartParliament *p,
                              const float *cand_vecs,
                              int n_cands,
                              float *out_weights);

/* Hebbian update post-commit, signaled by prophecy_debt sign. */
void  heart_parliament_notorch_step(HeartParliament *p,
                                     const float *winning_vec,
                                     float prophecy_debt);

/* Expert death cycle (every 10 commits). */
void  heart_parliament_try_apoptosis(HeartParliament *p);

#endif /* HEART_PARLIAMENT_H */
