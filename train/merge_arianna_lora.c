/*
 * merge_arianna_lora.c — Merge Arianna LoRA (ARLR format) into Resonance 200M
 *                       base (RS02 format).
 *
 * Recipe: W_merged[i,j] = W_base[i,j] + scale * sum_r A[i,r] * B[r,j]
 * Where:
 *   - A is [out_dim, rank], B is [rank, in_dim]
 *   - scale = alpha / rank
 *   - LoRA targets: 7 projections per block (wq, wk, wv, wo, mlp_gate,
 *                                            mlp_up, mlp_down)
 *
 * Output: same RS02 format as input base, with merged weights.
 * The resonance.aml inference path reads RS02 directly — no code changes
 * needed downstream.
 *
 * Build: cc -O2 merge_arianna_lora.c -o merge_arianna_lora -lm
 * Run:   ./merge_arianna_lora --base resonance_200m_final.bin \
 *                              --lora arianna_lora.bin \
 *                              --out  resonance_arianna_merged.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#define NUM_LORA_PROJ 7
#define LP_Q     0
#define LP_K     1
#define LP_V     2
#define LP_O     3
#define LP_GATE  4
#define LP_UP    5
#define LP_DOWN  6

typedef struct {
    int V, E, H, D, B, M, T, R;
    int n_merges;
} ResonanceCfg;

/* In-memory weight pointers — same layout as resonance_forward.h:71 Weights */
typedef struct {
    float* tok_emb;       /* [V, E] */
    struct {
        float *wr_a, *wr_b, *gate;
        float *norm1, *wq, *wk, *wv, *wo;
        float *norm2, *mlp_gate, *mlp_up, *mlp_down;
    } b[64];               /* up to 64 blocks */
    float* norm_f;
    float* out_head;
} Weights;

static void assign(Weights* w, float* p, ResonanceCfg* c) {
    int V = c->V, E = c->E, B = c->B, H = c->H, R = c->R, T = c->T, M = c->M;
    w->tok_emb = p; p += (size_t)V * E;
    for (int i = 0; i < B; i++) {
        w->b[i].wr_a     = p; p += (size_t)H * E * R;
        w->b[i].wr_b     = p; p += (size_t)H * R * T;
        w->b[i].gate     = p; p += H;
        w->b[i].norm1    = p; p += E;
        w->b[i].wq       = p; p += (size_t)E * E;
        w->b[i].wk       = p; p += (size_t)E * E;
        w->b[i].wv       = p; p += (size_t)E * E;
        w->b[i].wo       = p; p += (size_t)E * E;
        w->b[i].norm2    = p; p += E;
        w->b[i].mlp_gate = p; p += (size_t)M * E;
        w->b[i].mlp_up   = p; p += (size_t)M * E;
        w->b[i].mlp_down = p; p += (size_t)E * M;
    }
    w->norm_f   = p; p += E;
    w->out_head = p; p += (size_t)V * E;
}

/* Apply W += scale * A @ B  in-place to a base weight tensor.
 *   W is [out_dim, in_dim]
 *   A is [out_dim, rank]
 *   B is [rank, in_dim]
 */
static void apply_lora(float* W, const float* A, const float* B,
                       int out_dim, int in_dim, int rank, float scale) {
    /* delta[i, j] = sum_r A[i, r] * B[r, j] */
    for (int i = 0; i < out_dim; i++) {
        for (int r = 0; r < rank; r++) {
            float a = A[i * rank + r] * scale;
            if (a == 0.0f) continue;  /* common: A starts at 0 */
            const float* B_r = B + r * in_dim;
            float* W_i = W + i * in_dim;
            for (int j = 0; j < in_dim; j++) {
                W_i[j] += a * B_r[j];
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* base_path = NULL;
    const char* lora_path = NULL;
    const char* out_path  = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--base") && i+1 < argc) base_path = argv[++i];
        else if (!strcmp(argv[i], "--lora") && i+1 < argc) lora_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i+1 < argc) out_path = argv[++i];
    }
    if (!base_path || !lora_path || !out_path) {
        fprintf(stderr, "usage: %s --base <RS02.bin> --lora <ARLR.bin> --out <merged.bin>\n",
                argv[0]);
        return 1;
    }

    /* ── Read base (RS02) into memory ── */
    FILE* f = fopen(base_path, "rb");
    if (!f) { fprintf(stderr, "[base] cannot open %s\n", base_path); return 1; }

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return 1; }
    if (magic != 0x52533032) {
        fprintf(stderr, "[base] bad magic 0x%08x (need RS02)\n", magic);
        fclose(f); return 1;
    }
    int hdr[9];
    if (fread(hdr, 4, 9, f) != 9) { fclose(f); return 1; }
    ResonanceCfg cfg = {
        .E = hdr[0], .B = hdr[1], .T = hdr[2], .H = hdr[3],
        .D = hdr[4], .R = hdr[5], .M = hdr[6], .V = hdr[7]
    };
    fprintf(stderr, "[base] V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
            cfg.V, cfg.E, cfg.H, cfg.D, cfg.B, cfg.M, cfg.T, cfg.R);

    uint32_t n_merges_u32;
    if (fread(&n_merges_u32, 4, 1, f) != 1) { fclose(f); return 1; }
    cfg.n_merges = (int)n_merges_u32;
    int* merges_buf = malloc((size_t)cfg.n_merges * 3 * sizeof(int));
    if (fread(merges_buf, 12, cfg.n_merges, f) != (size_t)cfg.n_merges) {
        fclose(f); free(merges_buf); return 1;
    }
    fprintf(stderr, "[base] BPE merges: %d\n", cfg.n_merges);

    size_t np = 2 * (size_t)cfg.V * cfg.E + 1L * cfg.E;
    np += (size_t)cfg.B * (
        cfg.E + 3L * cfg.E * cfg.E
        + (size_t)cfg.H * cfg.E * cfg.R
        + (size_t)cfg.H * cfg.R * cfg.T
        + cfg.H + (size_t)cfg.E * cfg.E + cfg.E
        + 3L * cfg.M * cfg.E
    );
    float* blob = malloc(np * sizeof(float));
    if (fread(blob, sizeof(float), np, f) != np) {
        fclose(f); free(blob); free(merges_buf); return 1;
    }
    fclose(f);
    fprintf(stderr, "[base] %.1fM weights loaded\n", np / 1e6);

    Weights w = {0};
    assign(&w, blob, &cfg);

    /* ── Read LoRA (ARLR) and apply ── */
    f = fopen(lora_path, "rb");
    if (!f) { fprintf(stderr, "[lora] cannot open %s\n", lora_path); return 1; }
    uint32_t lora_magic;
    if (fread(&lora_magic, 4, 1, f) != 1) { fclose(f); return 1; }
    if (lora_magic != 0x524C5241) {
        fprintf(stderr, "[lora] bad magic 0x%08x (need ARLR)\n", lora_magic);
        fclose(f); return 1;
    }
    int32_t rank, lora_B, lora_E, lora_M;
    float alpha;
    fread(&rank, 4, 1, f);
    fread(&alpha, 4, 1, f);
    fread(&lora_B, 4, 1, f);
    fread(&lora_E, 4, 1, f);
    fread(&lora_M, 4, 1, f);
    if (lora_B != cfg.B || lora_E != cfg.E || lora_M != cfg.M) {
        fprintf(stderr, "[lora] dim mismatch: B=%d/%d E=%d/%d M=%d/%d\n",
                lora_B, cfg.B, lora_E, cfg.E, lora_M, cfg.M);
        fclose(f); return 1;
    }
    float scale = alpha / (float)rank;
    fprintf(stderr, "[lora] rank=%d alpha=%.2f scale=%.4f → applying to %d blocks × %d projections\n",
            rank, alpha, scale, cfg.B, NUM_LORA_PROJ);

    int proj_dims[NUM_LORA_PROJ][2] = {
        { cfg.E, cfg.E }, { cfg.E, cfg.E }, { cfg.E, cfg.E }, { cfg.E, cfg.E },
        { cfg.M, cfg.E }, { cfg.M, cfg.E }, { cfg.E, cfg.M },
    };

    for (int bl = 0; bl < cfg.B; bl++) {
        for (int p = 0; p < NUM_LORA_PROJ; p++) {
            int out_dim = proj_dims[p][0];
            int in_dim  = proj_dims[p][1];
            float* A = malloc((size_t)out_dim * rank * sizeof(float));
            float* B = malloc((size_t)rank * in_dim * sizeof(float));
            fread(A, sizeof(float), (size_t)out_dim * rank, f);
            fread(B, sizeof(float), (size_t)rank * in_dim, f);

            float* W = NULL;
            switch (p) {
                case LP_Q:    W = w.b[bl].wq;       break;
                case LP_K:    W = w.b[bl].wk;       break;
                case LP_V:    W = w.b[bl].wv;       break;
                case LP_O:    W = w.b[bl].wo;       break;
                case LP_GATE: W = w.b[bl].mlp_gate; break;
                case LP_UP:   W = w.b[bl].mlp_up;   break;
                case LP_DOWN: W = w.b[bl].mlp_down; break;
            }
            apply_lora(W, A, B, out_dim, in_dim, rank, scale);
            free(A); free(B);
        }
        if ((bl + 1) % 5 == 0)
            fprintf(stderr, "[merge] %d/%d blocks merged\n", bl + 1, cfg.B);
    }
    fclose(f);
    fprintf(stderr, "[merge] done\n");

    /* ── Write merged base (RS02) ── */
    f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "[out] cannot open %s\n", out_path); return 1; }
    fwrite(&magic, 4, 1, f);
    fwrite(hdr, 4, 9, f);
    fwrite(&n_merges_u32, 4, 1, f);
    fwrite(merges_buf, 12, cfg.n_merges, f);
    fwrite(blob, sizeof(float), np, f);
    fclose(f);
    fprintf(stderr, "[out] merged RS02 written → %s\n", out_path);

    free(blob); free(merges_buf);
    return 0;
}
