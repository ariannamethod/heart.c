/*
 * infer_resonance.c — minimal Resonance 200M inference driver.
 *
 * Wraps resonance.aml/tools/resonance_forward.h (which already has
 * resonance_load / resonance_generate functions). Used for Phase 3
 * sweep + Phase 7 duet trace, since amlc is not deployed on this pod.
 *
 * Build (against resonance forward header on pod):
 *   cc -O2 -DUSE_BLAS \
 *      -I/workspace/heart.c-runpod/notorch \
 *      -I/workspace/heart.c-runpod/resonance.aml/tools \
 *      infer_resonance.c \
 *      /workspace/heart.c-runpod/notorch/notorch.c \
 *      /workspace/heart.c-runpod/notorch/gguf.c \
 *      -lm -lopenblas -o infer_resonance
 *
 * Run: ./infer_resonance --weights resonance_arianna_merged.bin \
 *                         --prompt "Q: Who are you?\nA:" \
 *                         --tokens 100 --temp 0.7 --top_p 0.9
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

/* The resonance_forward.h header defines static functions and globals.
 * We include it directly — it's a header-as-translation-unit pattern. */
#include "resonance_forward.h"

int main(int argc, char** argv) {
    const char* weights = "weights/resonance_arianna_merged.bin";
    const char* prompt  = "Q: Who are you?\nA:";
    int   max_tokens    = 100;
    float temp          = 0.7f;
    float top_p         = 0.9f;
    unsigned seed       = (unsigned)time(NULL);

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--weights") && i+1<argc) weights    = argv[++i];
        else if (!strcmp(argv[i], "--prompt")  && i+1<argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--tokens")  && i+1<argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp")    && i+1<argc) temp       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top_p")   && i+1<argc) top_p      = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed")    && i+1<argc) seed       = (unsigned)atoi(argv[++i]);
    }
    srand(seed);
    fprintf(stderr, "[infer_resonance] weights=%s temp=%.2f top_p=%.2f tokens=%d seed=%u\n",
            weights, temp, top_p, max_tokens, seed);

    ResonanceCtx ctx = {0};
    if (resonance_load(&ctx, weights)) return 1;
    resonance_generate(&ctx, prompt, max_tokens, temp, top_p);
    resonance_free(&ctx);
    return 0;
}
