/*
 * merge_doe_lora.c — Merge DoE LoRA (DJLR format) into Janus v4 177M
 *                    base (JANU format).
 *
 * Recipe: W_merged[i,j] = W_base[i,j] + scale * sum_r A[i,r] * B[r,j]
 * LoRA targets: 7 projections per block (cq, ck, cv, cproj, wg, wu, wd).
 * Frozen: wr_a, wr_b, gate, wvr, wj.
 *
 * Output: same JANU format (256-byte header + raw fp32 weights), so
 * dario/infer_v4 reads directly. No code changes downstream.
 *
 * Build: cc -O2 merge_doe_lora.c -o merge_doe_lora -lm
 * Run:   ./merge_doe_lora --base janus_v4_base_22k.bin \
 *                          --lora doe_lora.bin \
 *                          --out  janus_v4_doe_merged.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NUM_LORA_PROJ 7
#define LP_Q     0   /* cq */
#define LP_K     1   /* ck */
#define LP_V     2   /* cv */
#define LP_O     3   /* cproj */
#define LP_GATE  4   /* wg (mlp) */
#define LP_UP    5   /* wu (mlp) */
#define LP_DOWN  6   /* wd (mlp) */

typedef struct {
    int V, E, H, D, B, M, T, R;
    long n_params;
} JanusCfg;

typedef struct {
    float *resid_l, *x0_l, *smear_l, *backout_l;
    float* wte;
    struct {
        float *wr_a, *wr_b, *gate;
        float *cq, *ck, *cv, *wvr, *wj, *cproj;
        float *wg, *wu, *wd;
    } b[64];
    float* head;
    float* smear_g;
} JWeights;

static void assign(JWeights* w, float* p, JanusCfg* c) {
    int V = c->V, E = c->E, B = c->B, H = c->H, R = c->R, T = c->T, M = c->M;
    w->resid_l = p; p += B;
    w->x0_l = p; p += B;
    w->smear_l = p; p += 1;
    w->backout_l = p; p += 1;
    w->wte = p; p += (size_t)V * E;
    for (int i = 0; i < B; i++) {
        w->b[i].wr_a  = p; p += (size_t)H * E * R;
        w->b[i].wr_b  = p; p += (size_t)H * R * T;
        w->b[i].gate  = p; p += (size_t)H * 3;
        w->b[i].cq    = p; p += (size_t)E * E;
        w->b[i].ck    = p; p += (size_t)E * E;
        w->b[i].cv    = p; p += (size_t)E * E;
        w->b[i].wvr   = p; p += (size_t)E * E;
        w->b[i].wj    = p; p += (size_t)E * E;
        w->b[i].cproj = p; p += (size_t)E * E;
        w->b[i].wg    = p; p += (size_t)M * E;
        w->b[i].wu    = p; p += (size_t)M * E;
        w->b[i].wd    = p; p += (size_t)E * M;
    }
    w->head = p; p += (size_t)V * E;
    w->smear_g = p; p += 24;
}

static void apply_lora(float* W, const float* A, const float* B,
                       int out_dim, int in_dim, int rank, float scale) {
    for (int i = 0; i < out_dim; i++) {
        for (int r = 0; r < rank; r++) {
            float a = A[i * rank + r] * scale;
            if (a == 0.0f) continue;
            const float* B_r = B + r * in_dim;
            float* W_i = W + i * in_dim;
            for (int j = 0; j < in_dim; j++) W_i[j] += a * B_r[j];
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
        fprintf(stderr, "usage: %s --base <JANU.bin> --lora <DJLR.bin> --out <merged.bin>\n",
                argv[0]);
        return 1;
    }

    /* Read JANU base (256-byte header + weights) */
    FILE* f = fopen(base_path, "rb");
    if (!f) { fprintf(stderr, "[base] open %s failed\n", base_path); return 1; }

    int magic_buf[2];
    if (fread(magic_buf, 4, 2, f) != 2) { fclose(f); return 1; }
    if (magic_buf[0] != 0x4A414E55) {
        fprintf(stderr, "[base] bad magic 0x%08x (need JANU)\n", magic_buf[0]);
        fclose(f); return 1;
    }
    int hdr[8];
    fread(hdr, 4, 8, f);
    JanusCfg cfg = {
        .V = hdr[0], .E = hdr[1], .H = hdr[2], .D = hdr[3],
        .B = hdr[4], .M = hdr[5], .T = hdr[6], .n_params = hdr[7]
    };
    long fixed = 2L*cfg.B + 2 + 24 + 2L*cfg.V*cfg.E
                 + (long)cfg.B*(6L*cfg.E*cfg.E + 3*cfg.H + 3L*cfg.M*cfg.E);
    cfg.R = (int)((cfg.n_params - fixed) / ((long)cfg.B * cfg.H * (cfg.E + cfg.T)));

    fprintf(stderr, "[base] V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d n_params=%ld\n",
            cfg.V, cfg.E, cfg.H, cfg.D, cfg.B, cfg.M, cfg.T, cfg.R, cfg.n_params);

    /* Read 256-byte header bytes verbatim (we'll write them back unchanged) */
    fseek(f, 0, SEEK_SET);
    unsigned char header[256];
    fread(header, 1, 256, f);

    float* blob = malloc((size_t)cfg.n_params * sizeof(float));
    if (fread(blob, sizeof(float), cfg.n_params, f) != (size_t)cfg.n_params) {
        fclose(f); free(blob); return 1;
    }
    fclose(f);
    fprintf(stderr, "[base] %.1fM weights loaded\n", cfg.n_params / 1e6);

    JWeights w = {0};
    assign(&w, blob, &cfg);

    /* Read DJLR LoRA */
    f = fopen(lora_path, "rb");
    if (!f) { fprintf(stderr, "[lora] open %s failed\n", lora_path); return 1; }
    uint32_t lora_magic;
    fread(&lora_magic, 4, 1, f);
    if (lora_magic != 0x524C4A44) {
        fprintf(stderr, "[lora] bad magic 0x%08x (need DJLR)\n", lora_magic);
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
        fprintf(stderr, "[lora] dim mismatch\n"); fclose(f); return 1;
    }
    float scale = alpha / (float)rank;
    fprintf(stderr, "[lora] rank=%d alpha=%.2f scale=%.4f\n", rank, alpha, scale);

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
                case LP_Q:    W = w.b[bl].cq;    break;
                case LP_K:    W = w.b[bl].ck;    break;
                case LP_V:    W = w.b[bl].cv;    break;
                case LP_O:    W = w.b[bl].cproj; break;
                case LP_GATE: W = w.b[bl].wg;    break;
                case LP_UP:   W = w.b[bl].wu;    break;
                case LP_DOWN: W = w.b[bl].wd;    break;
            }
            apply_lora(W, A, B, out_dim, in_dim, rank, scale);
            free(A); free(B);
        }
        if ((bl + 1) % 5 == 0)
            fprintf(stderr, "[merge] %d/%d blocks merged\n", bl + 1, cfg.B);
    }
    fclose(f);

    /* Write merged JANU */
    f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "[out] open %s failed\n", out_path); return 1; }
    fwrite(header, 1, 256, f);
    fwrite(blob, sizeof(float), cfg.n_params, f);
    fclose(f);
    fprintf(stderr, "[out] merged JANU written → %s\n", out_path);

    free(blob);
    return 0;
}
