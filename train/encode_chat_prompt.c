/*
 * encode_chat_prompt.c — encode a question into Janus v4 chat-format
 * tokens and write as .bin for infer_v4 to read directly.
 *
 * Format produced (matches chain_dialogue.py canonical):
 *   int32 n;
 *   int32 tokens[n] = [USER_START, <BPE-encoded question>, USER_END, ASST_START]
 *
 * Build:
 *   cc -O2 -I/tmp/include -I/workspace/heart.c-runpod/notorch \
 *      encode_chat_prompt.c \
 *      /workspace/heart.c-runpod/notorch/notorch.c \
 *      /workspace/heart.c-runpod/notorch/gguf.c \
 *      -lm -lopenblas -o encode_chat_prompt
 *
 * Run:
 *   ./encode_chat_prompt "Who are you?" /tmp/yent_prompt.bin
 */

#include "notorch.h"
#include "/workspace/heart.c-runpod/dario/janus_v4_bpe_merges.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define JANNUS_BOS         32759
#define JANNUS_USER_START  32760
#define JANNUS_USER_END    32761
#define JANNUS_ASST_START  32762

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <question text> <out.bin>\n", argv[0]);
        return 1;
    }
    const char* q = argv[1];
    const char* out_path = argv[2];

    nt_bpe bpe = {0};
    nt_bpe_init(&bpe, janus_v4_bpe_merges, JANUS_V4_BPE_MERGES);

    int q_toks[2048];
    int qn = nt_bpe_encode(&bpe, q, (int)strlen(q), q_toks, 2048);
    if (qn < 0) { fprintf(stderr, "encode failed\n"); return 1; }
    fprintf(stderr, "[encode] %d question tokens\n", qn);

    /* Wrap: [BOS, USER_START, q..., USER_END, ASST_START]
     * Per yent.aml/README.md:79,124. */
    int total = qn + 4;
    int* full = malloc((size_t)total * sizeof(int));
    full[0] = JANNUS_BOS;
    full[1] = JANNUS_USER_START;
    memcpy(full + 2, q_toks, (size_t)qn * sizeof(int));
    full[qn + 2] = JANNUS_USER_END;
    full[qn + 3] = JANNUS_ASST_START;

    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", out_path); return 1; }
    int32_t n32 = (int32_t)total;
    fwrite(&n32, 4, 1, f);
    fwrite(full, 4, total, f);
    fclose(f);
    fprintf(stderr, "[encode] wrote %d tokens to %s (USER_START + Q + USER_END + ASST_START)\n",
            total, out_path);
    free(full);
    return 0;
}
