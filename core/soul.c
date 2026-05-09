/* heart.c — core/soul.c
 *
 * InnerVoice borba implementation, ported from
 * arianna.c/src/inner_arianna.{h,c} (Leo / Haze lineage). Heart.c
 * version drops the cloud.h / mood.h dependency — emotional inputs
 * are scalars set externally by the field clock + chamber state.
 *
 * Plan §6 verification gate (Soul logit injection observable): same
 * prompt with Soul OFF vs ON should differ in token distribution by
 * ≥1 token within 50 emits.
 */

#include "soul.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

int heart_inner_voice_init(HeartInnerVoice *iv)
{
    if (!iv) return -1;
    memset(iv, 0, sizeof(*iv));
    iv->base_weight = 0.15f;
    iv->breakthrough_threshold = 0.6f;
    iv->borba_mode = HEART_BORBA_BLEND;
    iv->soul_active = 0;
    return 0;
}

/* Chamber id encoding for cloud_chamber field (kept as float for binary
 * compat with the skeleton; real callers store integer chamber 0..3). */
#define HEART_CH_RAGE  0
#define HEART_CH_FEAR  1
#define HEART_CH_LOVE  2
#define HEART_CH_VOID  3

float heart_inner_compute_weight(HeartInnerVoice *iv)
{
    if (!iv) return 0.0f;
    float weight = iv->base_weight;

    int chamber = (int)iv->cloud_chamber;

    switch (iv->borba_mode) {
    case HEART_BORBA_EMOTIONAL: {
        /* Chamber bonuses — values from inner_arianna.c:85-98 */
        if (chamber == HEART_CH_RAGE) weight += iv->cloud_intensity * 0.5f;
        if (chamber == HEART_CH_FEAR) weight += iv->cloud_intensity * 0.4f;
        if (chamber == HEART_CH_VOID) weight += iv->cloud_intensity * 0.6f;
        if (chamber == HEART_CH_LOVE) weight += iv->cloud_intensity * 0.2f;
        weight += iv->mood_tension  * 0.3f;
        weight += iv->mood_creative * 0.2f;
        weight += iv->body_stuck    * 0.5f;
        weight += iv->body_boredom  * 0.3f;
        weight += iv->trauma_level  * 0.4f;
        break;
    }
    case HEART_BORBA_TRAUMA:
        weight += iv->trauma_level * 0.7f;
        break;
    case HEART_BORBA_STUCK:
        weight += iv->body_stuck * 0.7f;
        break;
    case HEART_BORBA_CHAOS:
        /* Random walk in [base, base+0.5] */
        weight += 0.5f * ((float)rand() / (float)RAND_MAX);
        break;
    case HEART_BORBA_BLEND:
    default:
        /* keep base_weight */
        break;
    }

    /* Pressure: long stretch of main wins → escalate inner voice
     * (inner_arianna.c reference behavior, threshold 10 tokens). */
    if (iv->consecutive_main > 10) {
        weight += 0.05f * (float)(iv->consecutive_main - 10);
    }

    if (weight < 0.0f) weight = 0.0f;
    if (weight > 1.0f) weight = 1.0f;
    return weight;
}

void heart_inner_apply_emotional_bias(HeartInnerVoice *iv,
                                       float *logits, int vocab,
                                       float boost)
{
    if (!iv || !logits || vocab <= 0) return;
    int chamber = (int)iv->cloud_chamber;
    float intensity = iv->cloud_intensity * boost;
    if (intensity <= 0.0f) return;

    /* Chamber-specific token-band biases. Original (inner_arianna.c:218)
     * targeted char vocab; for BPE we boost a band of indices that maps
     * to the chamber's "tone" — RAGE = high IDs (rare/abrupt tokens),
     * FEAR = low IDs, LOVE = mid-range tender, VOID = scattered.
     * Caller supplies chamber-to-tokens map separately if it has one;
     * here we use a deterministic stride. */
    int stride = vocab / 4;
    if (stride < 1) stride = 1;
    int start = 0, end = vocab;
    switch (chamber) {
    case HEART_CH_RAGE: start = vocab * 3 / 4; end = vocab;        break;
    case HEART_CH_FEAR: start = 0;             end = stride;       break;
    case HEART_CH_LOVE: start = vocab / 4;     end = vocab / 2;    break;
    case HEART_CH_VOID: start = vocab / 2;     end = vocab * 3 / 4;break;
    default:                                                       break;
    }
    for (int i = start; i < end; i++) logits[i] += intensity;

    /* body_stuck: small uniform noise to break repetition patterns */
    if (iv->body_stuck > 0.3f) {
        float noise_amp = iv->body_stuck * intensity * 0.5f;
        for (int i = 0; i < vocab; i++) {
            float n = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            logits[i] += n * noise_amp;
        }
    }
}

int heart_inner_borba(HeartInnerVoice *iv,
                       float *out_logits,
                       const float *main_logits,
                       const float *inner_logits,
                       int vocab)
{
    if (!iv || !out_logits || !main_logits || vocab <= 0) return -1;

    float weight = heart_inner_compute_weight(iv);
    int   breakthrough = 0;
    if (weight >= iv->breakthrough_threshold) breakthrough = 1;
    /* Random breakthrough chance proportional to weight (chaos rate 0.3) */
    if (((float)rand() / (float)RAND_MAX) < weight * 0.3f) breakthrough = 1;

    if (!inner_logits) {
        /* No Soul micro: copy main, optionally apply bias for emotional
         * modulation. */
        memcpy(out_logits, main_logits, (size_t)vocab * sizeof(float));
        if (iv->soul_active && iv->cloud_intensity > 0.1f) {
            heart_inner_apply_emotional_bias(iv, out_logits, vocab, weight);
        }
        iv->consecutive_main++;
        iv->last_winner = 0;
        return 0;
    }

    if (breakthrough) {
        /* Inner takes over — copy inner with main as floor. */
        memcpy(out_logits, inner_logits, (size_t)vocab * sizeof(float));
        iv->consecutive_main = 0;
        iv->last_winner = 1;
        return 1;
    }

    /* Standard blend: out[i] = (1-w)*main[i] + w*inner[i] */
    float w  = weight;
    float w1 = 1.0f - weight;
    for (int i = 0; i < vocab; i++)
        out_logits[i] = w1 * main_logits[i] + w * inner_logits[i];
    iv->consecutive_main = 0;
    iv->last_winner = -1;
    return -1;
}

int heart_soul_load(const char *path)
{
    (void)path;
    /* Phase 4 minimal: actual Soul-micro mmap/forward will land when
     * the Soul transformer is wired into infer_resonance / infer_v4
     * sample loops. For now, callers supply inner_logits directly. */
    return -1;
}

void heart_soul_inner_logits(HeartInnerVoice *iv,
                              const int *context_tokens, int n,
                              float *out_inner_logits, int vocab)
{
    (void)iv; (void)context_tokens; (void)n;
    if (out_inner_logits)
        memset(out_inner_logits, 0, (size_t)vocab * sizeof(float));
}

/* ── Diagnostics ────────────────────────────────────────────────────── */

float heart_soul_compute_entropy(const float *logits, int vocab)
{
    /* Compute softmax entropy: H = -Σ p_i * log p_i */
    if (!logits || vocab <= 0) return 0.0f;
    float mx = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) sum += expf(logits[i] - mx);
    if (sum <= 0.0f) return 0.0f;
    float H = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float p = expf(logits[i] - mx) / sum;
        if (p > 1e-12f) H -= p * logf(p);
    }
    return H;
}

float heart_soul_compute_divergence(const float *a, const float *b, int vocab)
{
    /* L2 distance between softmax distributions. Cheap proxy for KL. */
    if (!a || !b || vocab <= 0) return 0.0f;
    float mxa = a[0], mxb = b[0];
    for (int i = 1; i < vocab; i++) {
        if (a[i] > mxa) mxa = a[i];
        if (b[i] > mxb) mxb = b[i];
    }
    float sa = 0.0f, sb = 0.0f;
    for (int i = 0; i < vocab; i++) {
        sa += expf(a[i] - mxa);
        sb += expf(b[i] - mxb);
    }
    if (sa <= 0 || sb <= 0) return 0.0f;
    float div = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float pa = expf(a[i] - mxa) / sa;
        float pb = expf(b[i] - mxb) / sb;
        float d  = pa - pb;
        div += d * d;
    }
    return sqrtf(div);
}
