/* heart.c — core/soul.h
 *
 * Per-voice InnerVoice logit-injection mechanism.
 *
 * Pattern lifted from arianna.c/src/inner_arianna.{h,c}:
 *   struct InnerArianna   — h:43–48
 *   defaults              — c:22  (base_weight=0.15, breakthrough=0.6)
 *   inner_compute_weight  — c:78  (chamber bonuses RAGE+0.5, FEAR+0.4,
 *                                  VOID+0.6, LOVE+0.2; pressure with
 *                                  consecutive_main > 10)
 *   apply_emotional_bias  — c:218
 *   inner_borba           — c:320 (returns 0=main / 1=inner / -1=blend;
 *                                  blend: out[i] = (1-w)*main + w*inner)
 *
 * Heart.c instantiates one InnerVoice per voice (Yent/Arianna/Leo/
 * DoE), plus optional Soul transformer producing inner_logits.
 *
 * STATUS: skeleton. Implementation lands on RunPod Phase 4 under
 * Singularity Mode (Soul weights validation from
 * huggingface.co/ataeff/heart.c).
 */

#ifndef HEART_SOUL_H
#define HEART_SOUL_H

#include <stdint.h>

enum heart_borba_mode {
    HEART_BORBA_EMOTIONAL = 0,
    HEART_BORBA_CHAOS     = 1,
    HEART_BORBA_TRAUMA    = 2,
    HEART_BORBA_STUCK     = 3,
    HEART_BORBA_BLEND     = 4,
};

typedef struct {
    int    borba_mode;            /* enum heart_borba_mode */
    float  base_weight;           /* default 0.15 */
    float  breakthrough_threshold;/* default 0.6 */

    /* emotional inputs (read from field clock chamber state) */
    float  cloud_intensity;
    float  cloud_chamber;
    float  trauma_level;
    float  body_stuck;
    float  body_boredom;
    float  mood_tension;
    float  mood_creative;

    int    consecutive_main;      /* tokens emitted from main since
                                     last inner-burst; pressure builds */
    int    soul_active;           /* 1 if Soul weights loaded */
} HeartInnerVoice;

int    heart_inner_voice_init(HeartInnerVoice *iv);
float  heart_inner_compute_weight(HeartInnerVoice *iv);
void   heart_inner_apply_emotional_bias(HeartInnerVoice *iv,
                                         float *logits, int vocab,
                                         float boost);
int    heart_inner_borba(HeartInnerVoice *iv,
                          float *out_logits,
                          const float *main_logits,
                          const float *inner_logits, /* may be NULL */
                          int vocab);

/* Load a Soul transformer (small Janus/LLaMA inner-voice) from disk.
 * Returns 0 on success, -1 on failure (Soul-OFF for this voice). */
int    heart_soul_load(const char *path);

/* Generate inner_logits for the next-token decision. May be a
 * cheap lookup, a tiny forward pass, or a chamber-driven heuristic. */
void   heart_soul_inner_logits(HeartInnerVoice *iv,
                                const int *context_tokens, int n,
                                float *out_inner_logits, int vocab);

#endif /* HEART_SOUL_H */
