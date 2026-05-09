/* heart.c — voices/leo/leo_anchors.h
 *
 * Leo's 325-word child-voice anchor lexicon, verbatim from
 * neoleo/leo.c:962–1078. Grouped by 6 chambers:
 *   FEAR / LOVE / RAGE / VOID / FLOW / COMPLEX
 *
 * Used at decode time as logit-bias on tokens whose decoded bytes
 * overlap any anchor. Sample words: bunny, teddy, cookie, cocoa,
 * lullaby, puddle, monster, closet, underbed.
 *
 * STATUS: skeleton — full table copied verbatim from neoleo at
 * Phase 0 vendor step.
 */
#ifndef HEART_LEO_ANCHORS_H
#define HEART_LEO_ANCHORS_H

#define HEART_LEO_N_ANCHORS 325

typedef struct {
    const char *word;
    int chamber;   /* 0=FEAR 1=LOVE 2=RAGE 3=VOID 4=FLOW 5=COMPLEX */
} HeartLeoAnchor;

extern const HeartLeoAnchor HEART_LEO_ANCHORS[HEART_LEO_N_ANCHORS];

#endif
