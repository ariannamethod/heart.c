/*
 * train_doe_lora.c — LoRA SFT for Janus v4 177M on DoE personality corpus.
 *
 * Heart.c plan v1.1 §2.4 / §12 Phase 2.
 *
 * Target: Janus v4 (V=32759 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64),
 *         JANU magic raw .bin format from huggingface.co/ataeff/janus4.
 * Recipe: yent-style LoRA r=16 α=32 on cq/ck/cv/cproj + wg/wu/wd, 1000 steps,
 *         "### Question: ... ### Answer: ..." format, masked CE on answer.
 * Optimizer: Chuck (CLAUDE.md ban on diagonal baselines).
 * Tokenizer: dario/janus_v4_bpe_merges.h (32503 merges, tiktoken-derived).
 *
 * Forward chain ports dario/infer_v4.c:130 prefill_batch into tape ops:
 *   embed → norm → for B blocks: resid_lambda blend → norm → QKV/Vr →
 *   RoPE + QK-norm → 3-way gate (content + RRPRAM-lowrank + Echo) →
 *   cproj + residual → norm → SwiGLU MLP + residual → backout.
 *
 * Skipped vs canonical inference (LoRA character-preserving):
 *   - smear (smear_l ≈ 0.32, minor effect, complex per-token gate)
 *   - softcap (15·tanh(x/15)) — sampling-time only
 *
 * Build: make -C train train_doe_lora
 * Run:   ./train_doe_lora \
 *          --base   weights/janus_v4_base_22k.bin \
 *          --corpus weights/personality_sft.txt \
 *          --rank   16 --steps 1000 --lr 2e-4 --max_seq 256 \
 *          --out    checkpoints/doe_lora.bin
 */

#include "notorch.h"
#ifdef USE_CUDA
  #include "notorch_cuda.h"
#endif
/* Embed BPE merges directly — janus_v4_bpe_merges.h in dario repo.
 * Resolved via -I$(DARIO_DIR) in train/Makefile. */
#include "janus_v4_bpe_merges.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>

/* ── JANU header → Config ───────────────────────────────────────────────── */

typedef struct {
    int V, E, H, D, B, M, T, R;
    long n_params;
} JanusConfig;

/* Per-block tape indices */
typedef struct {
    int wr_a_idx;       /* [H, E, R] frozen */
    int wr_b_idx;       /* [H, R, T] frozen */
    int wr_combined_idx;/* [H*R*(E+T)] frozen — for nt_rrpram_lowrank_attention */
    int gate_idx;       /* [H, 3] raw frozen */
    int gs0_exp_idx;    /* [E] sigmoid-broadcast: softmax(gate[h,0..2])[0] per head dim */
    int gs1_exp_idx;    /* [E] same for index 1 */
    int gs2_exp_idx;    /* [E] same for index 2 */
    int cq_idx, ck_idx, cv_idx, wvr_idx, wj_idx, cproj_idx;  /* [E,E] each */
    int wg_idx, wu_idx;  /* [M, E] */
    int wd_idx;          /* [E, M] */
} BlockBase;

typedef struct {
    int resid_l_idx;    /* [B] residual lambda per layer */
    int x0_l_idx;       /* [B] x0 lambda per layer */
    int smear_l_idx;    /* [1] frozen tape param (smear applied pre-tape in make_embedding_lookup) */
    int backout_l_idx;  /* [1] */
    int wte_idx;        /* [V, E] */
    int head_idx;       /* [V, E] */
    int smear_g_idx;    /* [24] frozen tape param */
    /* Cached scalar values (read once at load) for use in tape forward */
    float* resid_l_data;   /* [B] */
    float* x0_l_data;      /* [B] */
    float  smear_l_val;    /* scalar — applied at embed (post-norm, pre-x0-capture) */
    float  smear_g_data[24]; /* [24] raw, used in pre-tape make_embedding_lookup */
    float  backout_l_val;
    BlockBase* blocks;
} JanusBase;

/* ── LoRA — 7 projections per layer ────────────────────────────────────── */

#define NUM_LORA_PROJ 7
#define LP_Q     0   /* cq    [E, E] */
#define LP_K     1   /* ck    [E, E] */
#define LP_V     2   /* cv    [E, E] */
#define LP_O     3   /* cproj [E, E] */
#define LP_GATE  4   /* wg    [M, E] */
#define LP_UP    5   /* wu    [M, E] */
#define LP_DOWN  6   /* wd    [E, M] */

typedef struct {
    int A_idx;   /* [out_dim, rank] */
    int B_idx;   /* [rank, in_dim] */
} LoRABlock;

typedef struct {
    int   rank;
    float alpha;
    float scale;
    LoRABlock (*blocks)[NUM_LORA_PROJ];   /* [B][7] */
} LoRA;

/* ── Args ─────────────────────────────────────────────────────────────────── */

typedef struct {
    const char* base_path;
    const char* corpus_path;
    const char* out_path;
    int   rank;
    float alpha;
    float lr;
    int   warmup;
    int   steps;
    int   max_seq;
    int   log_every;
    int   ckpt_every;
    int   seed;
} Args;

static void args_default(Args* a) {
    a->base_path   = "weights/janus_v4_base_22k.bin";
    a->corpus_path = "weights/personality_sft.txt";
    a->out_path    = "checkpoints/doe_lora.bin";
    a->rank        = 16;
    a->alpha       = 32.0f;
    a->lr          = 2e-4f;
    a->warmup      = 50;
    a->steps       = 1000;
    a->max_seq     = 256;
    a->log_every   = 10;
    a->ckpt_every  = 100;
    a->seed        = 42;
}

static int args_parse(int argc, char** argv, Args* a) {
    args_default(a);
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--base")    && i+1<argc) a->base_path   = argv[++i];
        else if (!strcmp(argv[i], "--corpus")  && i+1<argc) a->corpus_path = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i+1<argc) a->out_path    = argv[++i];
        else if (!strcmp(argv[i], "--rank")    && i+1<argc) a->rank        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha")   && i+1<argc) a->alpha       = atof(argv[++i]);
        else if (!strcmp(argv[i], "--lr")      && i+1<argc) a->lr          = atof(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")  && i+1<argc) a->warmup      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps")   && i+1<argc) a->steps       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max_seq") && i+1<argc) a->max_seq     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--log")     && i+1<argc) a->log_every   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ckpt")    && i+1<argc) a->ckpt_every  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")    && i+1<argc) a->seed        = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    return 0;
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static nt_tensor* tensor_from_floats(const float* src, int* shape, int ndim) {
    nt_tensor* t = nt_tensor_new_shape(shape, ndim);
    if (!t) return NULL;
    int n = 1; for (int i = 0; i < ndim; i++) n *= shape[i];
    memcpy(t->data, src, (size_t)n * sizeof(float));
    return t;
}

static int load_frozen_param(const float* data, int* shape, int ndim) {
    nt_tensor* t = tensor_from_floats(data, shape, ndim);
    if (!t) return -1;
    int idx = nt_tape_param(t);
    nt_tape_freeze_param(idx);
    nt_tape_no_decay(idx);
    return idx;
}

/* ── JANU loader ─────────────────────────────────────────────────────── */

static int janus_load(const char* path, JanusConfig* cfg, JanusBase* base) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[load] cannot open %s\n", path); return 1; }

    int magic_buf[2];
    if (fread(magic_buf, 4, 2, f) != 2) { fclose(f); return 1; }
    if (magic_buf[0] != 0x4A414E55) {
        fprintf(stderr, "[load] bad magic 0x%08x (expected 'JANU' = 0x4A414E55)\n",
                magic_buf[0]);
        fclose(f); return 1;
    }
    int hdr[8];
    if (fread(hdr, 4, 8, f) != 8) { fclose(f); return 1; }
    cfg->V = hdr[0]; cfg->E = hdr[1]; cfg->H = hdr[2]; cfg->D = hdr[3];
    cfg->B = hdr[4]; cfg->M = hdr[5]; cfg->T = hdr[6];
    cfg->n_params = hdr[7];

    /* Derive R: n_params = 66 + 2*V*E + B*(6*E*E + H*R*(E+T) + 3*H + 3*M*E)
     *   66 = 2B + 2 + 24 (resid_l + x0_l + smear_l + backout_l + smear_g);
     *   (66 only valid for B=20). Generic: 2B + 2 + 24 = 2*B+26. */
    long fixed_prefix = 2L*cfg->B + 2 + 24;
    long fixed = fixed_prefix + 2L*cfg->V*cfg->E
                 + (long)cfg->B*(6L*cfg->E*cfg->E + 3*cfg->H + 3L*cfg->M*cfg->E);
    cfg->R = (int)((cfg->n_params - fixed)
                   / ((long)cfg->B * cfg->H * (cfg->E + cfg->T)));

    fprintf(stderr, "[load] JANU v%d V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d n_params=%ld\n",
            magic_buf[1], cfg->V, cfg->E, cfg->H, cfg->D, cfg->B, cfg->M, cfg->T,
            cfg->R, cfg->n_params);

    if (cfg->H * cfg->D != cfg->E) {
        fprintf(stderr, "[load] H*D (%d) != E (%d)\n", cfg->H * cfg->D, cfg->E);
        fclose(f); return 1;
    }

    /* Weights start at offset 256 (256-byte header) */
    fseek(f, 256, SEEK_SET);
    float* blob = malloc((size_t)cfg->n_params * sizeof(float));
    if (!blob) { fclose(f); return 1; }
    size_t got = fread(blob, sizeof(float), cfg->n_params, f);
    fclose(f);
    if (got != (size_t)cfg->n_params) {
        fprintf(stderr, "[load] short read: %zu/%ld\n", got, cfg->n_params);
        free(blob); return 1;
    }
    fprintf(stderr, "[load] %.1fM weights read\n", cfg->n_params / 1e6);

    int E = cfg->E, B = cfg->B, H = cfg->H, R = cfg->R, T = cfg->T, M = cfg->M, V = cfg->V;
    int D = cfg->D;
    float* p = blob;

    /* Prefix scalars (per assign() in infer_v4.c:84) */
    {
        int sh[1] = {B};
        base->resid_l_idx = load_frozen_param(p, sh, 1);
        base->resid_l_data = malloc(B * sizeof(float));
        memcpy(base->resid_l_data, p, B * sizeof(float));
        p += B;
    }
    {
        int sh[1] = {B};
        base->x0_l_idx = load_frozen_param(p, sh, 1);
        base->x0_l_data = malloc(B * sizeof(float));
        memcpy(base->x0_l_data, p, B * sizeof(float));
        p += B;
    }
    {
        int sh[1] = {1};
        base->smear_l_idx = load_frozen_param(p, sh, 1);
        base->smear_l_val = p[0];
        p += 1;
    }
    {
        int sh[1] = {1};
        base->backout_l_idx = load_frozen_param(p, sh, 1);
        base->backout_l_val = p[0];
        p += 1;
    }
    /* wte [V, E] */
    {
        int sh[2] = {V, E};
        base->wte_idx = load_frozen_param(p, sh, 2);
        p += (size_t)V * E;
    }

    base->blocks = calloc(B, sizeof(BlockBase));
    for (int bl = 0; bl < B; bl++) {
        BlockBase* blk = &base->blocks[bl];

        /* wr_a [H, E, R] */
        const float* wr_a_src = p;
        { int sh[3] = {H, E, R}; blk->wr_a_idx = load_frozen_param(p, sh, 3); p += (size_t)H*E*R; }
        /* wr_b [H, R, T] */
        const float* wr_b_src = p;
        { int sh[3] = {H, R, T}; blk->wr_b_idx = load_frozen_param(p, sh, 3); p += (size_t)H*R*T; }
        /* gate [H, 3] raw */
        const float* gate_src = p;
        { int sh[2] = {H, 3}; blk->gate_idx = load_frozen_param(p, sh, 2); p += (size_t)H*3; }

        /* Pre-bake wr_combined for nt_rrpram_lowrank_attention:
         * concat Wr_a (H*E*R) + Wr_b (H*R*T) → flat */
        {
            size_t a_len = (size_t)H * E * R;
            size_t b_len = (size_t)H * R * T;
            float* wr_c = malloc((a_len + b_len) * sizeof(float));
            memcpy(wr_c, wr_a_src, a_len * sizeof(float));
            memcpy(wr_c + a_len, wr_b_src, b_len * sizeof(float));
            int sh[1] = {(int)(a_len + b_len)};
            blk->wr_combined_idx = load_frozen_param(wr_c, sh, 1);
            free(wr_c);
        }

        /* Pre-bake 3-way gate softmax broadcasts.
         * For each head h: gs[h] = softmax(gate[h, 0..2]). Then expand to [E]:
         *   gs0_exp[h*D + d] = gs[h, 0]   for all d.
         * This lets us do nt_mul(content, gs0_tiled) etc. — same pattern as
         * Resonance trainer's per-head sigmoid blend. */
        {
            float* gs0 = malloc(E * sizeof(float));
            float* gs1 = malloc(E * sizeof(float));
            float* gs2 = malloc(E * sizeof(float));
            for (int h = 0; h < H; h++) {
                float g[3] = { gate_src[h*3], gate_src[h*3+1], gate_src[h*3+2] };
                float mx = g[0]; if (g[1] > mx) mx = g[1]; if (g[2] > mx) mx = g[2];
                float e0 = expf(g[0]-mx), e1 = expf(g[1]-mx), e2 = expf(g[2]-mx);
                float s = e0 + e1 + e2;
                e0 /= s; e1 /= s; e2 /= s;
                for (int d = 0; d < D; d++) {
                    gs0[h*D + d] = e0;
                    gs1[h*D + d] = e1;
                    gs2[h*D + d] = e2;
                }
            }
            int sh[1] = {E};
            blk->gs0_exp_idx = load_frozen_param(gs0, sh, 1);
            blk->gs1_exp_idx = load_frozen_param(gs1, sh, 1);
            blk->gs2_exp_idx = load_frozen_param(gs2, sh, 1);
            free(gs0); free(gs1); free(gs2);
        }

        /* cq, ck, cv, wvr, wj, cproj — all [E, E] */
        { int sh[2] = {E,E}; blk->cq_idx    = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->ck_idx    = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->cv_idx    = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->wvr_idx   = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->wj_idx    = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->cproj_idx = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        /* mlp wg, wu [M, E]; wd [E, M] */
        { int sh[2] = {M,E}; blk->wg_idx    = load_frozen_param(p, sh, 2); p += (size_t)M*E; }
        { int sh[2] = {M,E}; blk->wu_idx    = load_frozen_param(p, sh, 2); p += (size_t)M*E; }
        { int sh[2] = {E,M}; blk->wd_idx    = load_frozen_param(p, sh, 2); p += (size_t)E*M; }
    }

    /* head [V, E] */
    {
        int sh[2] = {V, E};
        base->head_idx = load_frozen_param(p, sh, 2);
        p += (size_t)V * E;
    }
    /* smear_g [24] */
    {
        int sh[1] = {24};
        base->smear_g_idx = load_frozen_param(p, sh, 1);
        memcpy(base->smear_g_data, p, 24 * sizeof(float));
        p += 24;
    }

    if ((long)(p - blob) != cfg->n_params) {
        fprintf(stderr, "[load] walk mismatch: p-blob=%ld vs n_params=%ld\n",
                (long)(p - blob), cfg->n_params);
        free(blob); return 1;
    }
    free(blob);
    fprintf(stderr, "[load] frozen base loaded onto tape (%d blocks)\n", B);
    return 0;
}

/* ── LoRA init ─────────────────────────────────────────────────────── */

static LoRA* lora_init(JanusConfig* cfg, int rank, float alpha) {
    LoRA* lr = calloc(1, sizeof(LoRA));
    lr->rank  = rank;
    lr->alpha = alpha;
    lr->scale = alpha / (float)rank;
    lr->blocks = calloc(cfg->B, sizeof(lr->blocks[0]));

    int proj_dims[NUM_LORA_PROJ][2] = {
        { cfg->E, cfg->E },   /* Q  cq */
        { cfg->E, cfg->E },   /* K  ck */
        { cfg->E, cfg->E },   /* V  cv */
        { cfg->E, cfg->E },   /* O  cproj */
        { cfg->M, cfg->E },   /* GATE wg */
        { cfg->M, cfg->E },   /* UP   wu */
        { cfg->E, cfg->M },   /* DOWN wd */
    };

    long total = 0;
    for (int bl = 0; bl < cfg->B; bl++) {
        for (int p = 0; p < NUM_LORA_PROJ; p++) {
            int out_dim = proj_dims[p][0];
            int in_dim  = proj_dims[p][1];
            int A_shape[2] = { out_dim, rank };
            int B_shape[2] = { rank,    in_dim };
            nt_tensor* A = nt_tensor_new_shape(A_shape, 2);
            nt_tensor_fill(A, 0.0f);
            nt_tensor* B = nt_tensor_new_shape(B_shape, 2);
            nt_tensor_xavier(B, in_dim, rank);
            lr->blocks[bl][p].A_idx = nt_tape_param(A);
            lr->blocks[bl][p].B_idx = nt_tape_param(B);
            total += (long)out_dim * rank + (long)rank * in_dim;
        }
    }
    fprintf(stderr, "[lora] rank=%d alpha=%.1f scale=%.4f → %ld trainable floats (%.2f MB)\n",
            rank, alpha, lr->scale, total, total * 4.0 / 1024.0 / 1024.0);
    return lr;
}

static int lora_apply(LoRA* lr, int layer, int proj, int x_idx, int y_idx, int T) {
    LoRABlock* blk = &lr->blocks[layer][proj];
    int b_x_idx    = nt_seq_linear(blk->B_idx, x_idx, T);
    int scaled_idx = nt_scale(b_x_idx, lr->scale);
    int delta_idx  = nt_seq_linear(blk->A_idx, scaled_idx, T);
    return nt_add(y_idx, delta_idx);
}

static int proj_with_lora(LoRA* lr, int layer, int proj,
                          int w_idx, int x_idx, int T) {
    int y_idx = nt_seq_linear(w_idx, x_idx, T);
    return lora_apply(lr, layer, proj, x_idx, y_idx, T);
}

/* ── Per-step seq utilities (same pattern as train_arianna_lora) ───── */

static int tile_per_seq(int src_idx, int T, int E) {
    nt_tape* tp = nt_tape_get();
    nt_tensor* src = tp->entries[src_idx].output;
    int shape[2] = {T, E};
    nt_tensor* tiled = nt_tensor_new_shape(shape, 2);
    for (int t = 0; t < T; t++)
        memcpy(tiled->data + (size_t)t * E, src->data, (size_t)E * sizeof(float));
    int idx = nt_tape_param(tiled);
    nt_tape_freeze_param(idx);
    nt_tape_no_decay(idx);
    return idx;
}

/* Embed + norm + smear pre-tape (matches canonical Janus prefill_batch in
 * dario/infer_v4.c:135-157):
 *   x = norm(wte[tokens])             — RMSnorm BEFORE smear
 *   if smear_l > 1e-6:
 *     for p in 1..T-1:
 *       gate[p] = smear_l * sigmoid(smear_g[0..23] @ x[p, :24])
 *       x[p] += gate[p] * x[p-1]
 * Smear/wte both frozen — pre-tape is fine, no gradient flow needed back.
 * Returns frozen [T,E] tape param; trainer must NOT re-apply seq_rmsnorm. */
static int make_embedding_lookup(int wte_idx, int* tokens, int T, int E,
                                  float smear_l, const float* smear_g) {
    nt_tape* tp = nt_tape_get();
    nt_tensor* W = tp->entries[wte_idx].output;
    int shape[2] = {T, E};
    nt_tensor* h = nt_tensor_new_shape(shape, 2);
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        memcpy(h->data + (size_t)t * E, W->data + (size_t)tok * E, (size_t)E * sizeof(float));
    }
    /* RMSnorm each position (non-parametric, eps=1e-5 per nanochat). */
    for (int t = 0; t < T; t++) {
        float* row = h->data + (size_t)t * E;
        float ss = 0.0f;
        for (int e = 0; e < E; e++) ss += row[e] * row[e];
        float inv = 1.0f / sqrtf(ss / (float)E + 1e-5f);
        for (int e = 0; e < E; e++) row[e] *= inv;
    }
    /* Smear (skip if smear_l < threshold per canonical). */
    if (smear_l > 1e-6f) {
        for (int p = 1; p < T; p++) {
            float* xp  = h->data + (size_t)p     * E;
            float* xpm = h->data + (size_t)(p-1) * E;
            float dot = 0.0f;
            for (int d = 0; d < 24; d++) dot += smear_g[d] * xp[d];
            float gate = smear_l / (1.0f + expf(-dot));
            for (int e = 0; e < E; e++) xp[e] += gate * xpm[e];
        }
    }
    int idx = nt_tape_param(h);
    nt_tape_freeze_param(idx);
    nt_tape_no_decay(idx);
    return idx;
}

static int make_targets(int* tokens, int T_input) {
    int shape[1] = {T_input};
    nt_tensor* t = nt_tensor_new_shape(shape, 1);
    for (int i = 0; i < T_input; i++) t->data[i] = (float)tokens[i + 1];
    int idx = nt_tape_param(t);
    nt_tape_freeze_param(idx);
    nt_tape_no_decay(idx);
    return idx;
}

static int make_loss_mask(int lq, int T_input) {
    int shape[1] = {T_input};
    nt_tensor* m = nt_tensor_new_shape(shape, 1);
    for (int i = 0; i < T_input; i++)
        m->data[i] = (i >= lq - 1) ? 1.0f : 0.0f;
    int idx = nt_tape_param(m);
    nt_tape_freeze_param(idx);
    nt_tape_no_decay(idx);
    return idx;
}

/* ── Forward (Janus v4 prefill, tape-aware) ─────────────────────────── */

static int forward(JanusBase* base, LoRA* lr, JanusConfig* cfg,
                   int* tokens, int T_input) {
    int E = cfg->E, B = cfg->B, H = cfg->H, D = cfg->D;
    int backout_layer = B / 2;

    /* Embed + norm + smear pre-tape (canonical Janus prefill_batch
     * dario/infer_v4.c:135-157). smear/wte/smear_g all frozen — pre-tape
     * compute is fine, no gradient flows back through them.
     * x0 captured AFTER smear per canonical infer_v4.c:156. */
    int h = make_embedding_lookup(base->wte_idx, tokens, T_input, E,
                                   base->smear_l_val, base->smear_g_data);
    int x0 = h;

    int backout_buf = -1;

    for (int bl = 0; bl < B; bl++) {
        BlockBase* blk = &base->blocks[bl];

        /* Residual blend: x = resid_l[bl] * x + x0_l[bl] * x0 */
        float rl  = base->resid_l_data[bl];
        float x0l = base->x0_l_data[bl];
        int h_scaled  = nt_scale(h,  rl);
        int x0_scaled = nt_scale(x0, x0l);
        h = nt_add(h_scaled, x0_scaled);

        /* Pre-attention norm (non-parametric) */
        int xn = nt_seq_rmsnorm(h, -1, T_input, E);

        /* Q/K/V/Vr projections; LoRA on Q/K/V; Vr frozen-only */
        int q  = proj_with_lora(lr, bl, LP_Q, blk->cq_idx, xn, T_input);
        int k  = proj_with_lora(lr, bl, LP_K, blk->ck_idx, xn, T_input);
        int v  = proj_with_lora(lr, bl, LP_V, blk->cv_idx, xn, T_input);
        int vr = nt_seq_linear(blk->wvr_idx, xn, T_input);

        /* Strike 4: split-half RoPE via notorch patch + per-head QK-norm.
         * Janus v4 split-half (infer_v4.c:35-49), base 100000. */
        q = nt_rope_split_half_freq(q, T_input, D, 100000.0f);
        k = nt_rope_split_half_freq(k, T_input, D, 100000.0f);
        q = nt_seq_rmsnorm(q, -1, T_input * H, D); q = nt_scale(q, 1.2f);
        k = nt_seq_rmsnorm(k, -1, T_input * H, D); k = nt_scale(k, 1.2f);

        /* Content attention: full MHA (Janus has KV=H, no GQA) */
        int content = nt_mh_causal_attention(q, k, v, T_input, D);

        /* RRPRAM low-rank attention (canonical Janus broadcast pattern with sc=1/sqrt(D),
         * per dario/infer_v4.c:218-249). Per-position op was function-class different
         * and caused training plateau near ln(V) = uniform-distribution loss. */
        int rrpram = nt_rrpram_broadcast_attention(blk->wr_combined_idx,
                                                    xn, vr, T_input, E, H, D);

        /* Echo: e = wj @ xn (acts as third value path, no attention pooling) */
        int echo = nt_seq_linear(blk->wj_idx, xn, T_input);

        /* 3-way gate blend per head:
         *   blend = gs0 * content + gs1 * rrpram + gs2 * echo
         * gs{0,1,2}_exp are pre-baked [E] frozen tape params; tile to [T,E]. */
        int gs0_t = tile_per_seq(blk->gs0_exp_idx, T_input, E);
        int gs1_t = tile_per_seq(blk->gs1_exp_idx, T_input, E);
        int gs2_t = tile_per_seq(blk->gs2_exp_idx, T_input, E);
        int g0c   = nt_mul(content, gs0_t);
        int g1r   = nt_mul(rrpram,  gs1_t);
        int g2e   = nt_mul(echo,    gs2_t);
        int blend = nt_add(g0c, nt_add(g1r, g2e));

        /* Output projection + residual */
        int o = proj_with_lora(lr, bl, LP_O, blk->cproj_idx, blend, T_input);
        h = nt_add(h, o);

        /* Cache mid-layer for backout */
        if (bl == backout_layer) backout_buf = h;

        /* MLP block: norm → SwiGLU → residual */
        int xn2      = nt_seq_rmsnorm(h, -1, T_input, E);
        int gate_lin = proj_with_lora(lr, bl, LP_GATE, blk->wg_idx, xn2, T_input);
        int up_lin   = proj_with_lora(lr, bl, LP_UP,   blk->wu_idx, xn2, T_input);
        int swi      = nt_swiglu(gate_lin, up_lin);
        int down     = proj_with_lora(lr, bl, LP_DOWN, blk->wd_idx, swi, T_input);
        h = nt_add(h, down);
    }

    /* Backout: x -= backout_l * backout_buf */
    if (backout_buf >= 0) {
        int neg_bo = nt_scale(backout_buf, -base->backout_l_val);
        h = nt_add(h, neg_bo);
    }

    /* Final non-parametric norm + head; skip softcap during training */
    int hn     = nt_seq_rmsnorm(h, -1, T_input, E);
    int logits = nt_seq_linear(base->head_idx, hn, T_input);
    return logits;
}

/* ── Tape truncation ─────────────────────────────────────────────────── */

static void tape_drop_temporaries(int keep_count) {
    nt_tape* tp = nt_tape_get();
    while (tp->count > keep_count) {
        nt_tape_entry* e = &tp->entries[tp->count - 1];
        if (e->output) { nt_tensor_free(e->output); e->output = NULL; }
        if (e->grad)   { nt_tensor_free(e->grad);   e->grad   = NULL; }
        tp->count--;
    }
}

/* ── Dataset (DoE personality_sft.txt — alternating Q:/A: like Arianna) ── */

typedef struct {
    char**  q_texts;
    char**  a_texts;
    int     n_pairs;
} DoeCorpus;

/* Parse <human>q\n<ai>a\n<human>q'... format per janus.doe/m.c:929-960.
 * Tags are STRIPPED — only content between tags becomes tokens.
 * mask=0 on human-segment tokens, mask=1 on ai-segment tokens. */
static DoeCorpus* corpus_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[corpus] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = 0;
    fclose(f);

    int n_pairs = 0;
    for (long i = 0; i + 7 <= sz; i++) {
        if ((i == 0 || buf[i-1] == '\n') && memcmp(buf + i, "<human>", 7) == 0) n_pairs++;
    }

    DoeCorpus* c = calloc(1, sizeof(DoeCorpus));
    c->q_texts = calloc(n_pairs > 0 ? n_pairs : 1, sizeof(char*));
    c->a_texts = calloc(n_pairs > 0 ? n_pairs : 1, sizeof(char*));
    c->n_pairs = 0;

    long i = 0;
    while (i < sz) {
        while (i + 7 <= sz && !((i == 0 || buf[i-1] == '\n') && memcmp(buf + i, "<human>", 7) == 0))
            i++;
        if (i + 7 > sz) break;
        long q_start = i + 7;
        long q_end = q_start;
        while (q_end + 4 <= sz && !((buf[q_end-1] == '\n' || q_end == q_start) &&
                                     memcmp(buf + q_end, "<ai>", 4) == 0))
            q_end++;
        if (q_end + 4 > sz) break;
        long a_start = q_end + 4;
        long a_end = a_start;
        while (a_end + 7 <= sz && !((buf[a_end-1] == '\n') && memcmp(buf + a_end, "<human>", 7) == 0))
            a_end++;
        if (a_end + 7 > sz) a_end = sz;

        size_t q_len = (size_t)(q_end - q_start);
        size_t a_len = (size_t)(a_end - a_start);
        while (q_len > 0 && (buf[q_start + q_len - 1] == '\n' ||
                              buf[q_start + q_len - 1] == ' '  ||
                              buf[q_start + q_len - 1] == '\t')) q_len--;
        while (a_len > 0 && (buf[a_start + a_len - 1] == '\n' ||
                              buf[a_start + a_len - 1] == ' '  ||
                              buf[a_start + a_len - 1] == '\t')) a_len--;
        if (q_len > 0 && a_len > 0) {
            c->q_texts[c->n_pairs] = malloc(q_len + 1);
            memcpy(c->q_texts[c->n_pairs], buf + q_start, q_len);
            c->q_texts[c->n_pairs][q_len] = 0;
            c->a_texts[c->n_pairs] = malloc(a_len + 1);
            memcpy(c->a_texts[c->n_pairs], buf + a_start, a_len);
            c->a_texts[c->n_pairs][a_len] = 0;
            c->n_pairs++;
        }
        i = a_end;
    }
    free(buf);
    fprintf(stderr, "[corpus] %d <human>/<ai> pairs parsed from %s\n", c->n_pairs, path);
    return c;
}

static void corpus_free(DoeCorpus* c) {
    if (!c) return;
    for (int i = 0; i < c->n_pairs; i++) { free(c->q_texts[i]); free(c->a_texts[i]); }
    free(c->q_texts); free(c->a_texts); free(c);
}

/* DoE personality format per janus.doe/m.c:929-960: tags stripped, segments
 * concatenated. q_text encoded with trailing "\n", a_text encoded with
 * trailing "\n". Mask=0 on q_tokens, mask=1 on a_tokens (handled by caller
 * via lq boundary). */
static int* encode_pair(nt_bpe* bpe, const char* q, const char* a,
                        uint32_t* lq_out, uint32_t* la_out, int max_total) {
    char qbuf[8192], abuf[8192];
    snprintf(qbuf, sizeof(qbuf), "%s\n", q);
    snprintf(abuf, sizeof(abuf), "%s\n", a);

    int* q_toks = malloc((size_t)max_total * sizeof(int));
    int* a_toks = malloc((size_t)max_total * sizeof(int));
    int qn = nt_bpe_encode(bpe, qbuf, (int)strlen(qbuf), q_toks, max_total);
    int an = nt_bpe_encode(bpe, abuf, (int)strlen(abuf), a_toks, max_total);
    if (qn <= 0 || an <= 0) { free(q_toks); free(a_toks); return NULL; }
    if (qn + an > max_total) {
        an = max_total - qn;
        if (an < 2) { free(q_toks); free(a_toks); return NULL; }
    }
    int* out = malloc((size_t)(qn + an) * sizeof(int));
    memcpy(out, q_toks, (size_t)qn * sizeof(int));
    memcpy(out + qn, a_toks, (size_t)an * sizeof(int));
    *lq_out = (uint32_t)qn;
    *la_out = (uint32_t)an;
    free(q_toks); free(a_toks);
    return out;
}

/* ── Save LoRA ───────────────────────────────────────────────────────── */

static int lora_save(LoRA* lr, JanusConfig* cfg, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[save] cannot open %s\n", path); return 1; }
    /* Magic: "DJLR" = 0x524C4A44 LE — DoE Janus LoRA format */
    uint32_t magic = 0x524C4A44;
    fwrite(&magic, 4, 1, f);
    int32_t r = lr->rank, b = cfg->B, e = cfg->E, m = cfg->M;
    fwrite(&r, 4, 1, f);
    fwrite(&lr->alpha, 4, 1, f);
    fwrite(&b, 4, 1, f);
    fwrite(&e, 4, 1, f);
    fwrite(&m, 4, 1, f);

    int proj_dims[NUM_LORA_PROJ][2] = {
        { cfg->E, cfg->E }, { cfg->E, cfg->E }, { cfg->E, cfg->E }, { cfg->E, cfg->E },
        { cfg->M, cfg->E }, { cfg->M, cfg->E }, { cfg->E, cfg->M },
    };
    nt_tape* tp = nt_tape_get();
    for (int bl = 0; bl < cfg->B; bl++) {
        for (int p = 0; p < NUM_LORA_PROJ; p++) {
            nt_tensor* A = tp->entries[lr->blocks[bl][p].A_idx].output;
            nt_tensor* B = tp->entries[lr->blocks[bl][p].B_idx].output;
            int out_dim = proj_dims[p][0];
            int in_dim  = proj_dims[p][1];
            fwrite(A->data, sizeof(float), (size_t)out_dim * lr->rank, f);
            fwrite(B->data, sizeof(float), (size_t)lr->rank * in_dim, f);
        }
    }
    fclose(f);
    fprintf(stderr, "[save] LoRA written → %s\n", path);
    return 0;
}

/* ── Main ────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    Args A;
    if (args_parse(argc, argv, &A)) return 1;
    srand((unsigned)A.seed);
    nt_seed((uint64_t)A.seed);

    printf("=== DoE LoRA SFT (Janus v4 177M, Chuck) ===\n");
    printf("base:   %s\ncorpus: %s\nout:    %s\n",
           A.base_path, A.corpus_path, A.out_path);
    printf("rank=%d alpha=%.1f lr=%.2e warmup=%d steps=%d max_seq=%d seed=%d\n\n",
           A.rank, A.alpha, A.lr, A.warmup, A.steps, A.max_seq, A.seed);

    double t0 = now_ms();

    nt_tape_start();
#ifdef USE_CUDA
    if (gpu_init() != 0) {
        fprintf(stderr, "[main] gpu_init failed — falling back to CPU\n");
    } else {
        nt_set_gpu_mode(1);
        fprintf(stderr, "[main] CUDA enabled (cublas + tf32)\n");
    }
#endif

    JanusConfig cfg = {0};
    JanusBase   base = {0};
    if (janus_load(A.base_path, &cfg, &base)) return 1;

    LoRA* lr = lora_init(&cfg, A.rank, A.alpha);

    nt_bpe bpe = {0};
    nt_bpe_init(&bpe, janus_v4_bpe_merges, JANUS_V4_BPE_MERGES);
    fprintf(stderr, "[bpe] vocab=%d merges=%d\n", bpe.vocab_size, bpe.n_merges);

    DoeCorpus* corpus = corpus_load(A.corpus_path);
    if (!corpus || corpus->n_pairs == 0) {
        fprintf(stderr, "[main] empty corpus\n"); return 1;
    }

    int n_init_entries = nt_tape_get()->count;
    printf("[init] elapsed %.0f ms, tape entries after init: %d\n\n",
           now_ms() - t0, n_init_entries);

    nt_schedule sched = nt_schedule_cosine(A.lr, A.warmup, A.steps, A.lr * 0.1f);

    nt_train_mode(1);
    double train_t0 = now_ms();
    float ema_loss = 0.0f;

    for (int step = 0; step < A.steps; step++) {
        int idx = rand() % corpus->n_pairs;
        uint32_t lq = 0, la = 0;
        int* tokens = encode_pair(&bpe, corpus->q_texts[idx], corpus->a_texts[idx],
                                   &lq, &la, A.max_seq);
        if (!tokens) continue;
        int T = (int)(lq + la);
        int T_input = T - 1;
        if (T_input < 4 || (int)la < 2) { free(tokens); continue; }

        int logits_idx  = forward(&base, lr, &cfg, tokens, T_input);
        int targets_idx = make_targets(tokens, T_input);
        int mask_idx    = make_loss_mask((int)lq, T_input);
        int loss_idx    = nt_seq_cross_entropy_masked(logits_idx, targets_idx, mask_idx,
                                                       T_input, cfg.V);

        float loss = nt_tape_get()->entries[loss_idx].output->data[0];

        if (step == 0) {
            nt_tensor* lt = nt_tape_get()->entries[logits_idx].output;
            float* L = lt->data + (size_t)(T_input - 1) * cfg.V;
            float mean = 0, var = 0;
            for (int v = 0; v < cfg.V; v++) mean += L[v];
            mean /= cfg.V;
            for (int v = 0; v < cfg.V; v++) var += (L[v] - mean) * (L[v] - mean);
            var /= cfg.V;
            fprintf(stderr, "[diag] step0 logits[last-pos]: mean=%.4f var=%.4f\n",
                    mean, var);
        }

        nt_tape_backward(loss_idx);
        float lr_now = nt_schedule_get_lr(&sched);
        nt_tape_chuck_step(lr_now, loss);

        ema_loss = (step == 0) ? loss : (0.95f * ema_loss + 0.05f * loss);

        if (step % A.log_every == 0) {
            double el = (now_ms() - train_t0) / 1000.0;
            double tps = (step + 1) * (double)T_input / el;
            printf("step %4d/%d | T=%3d lq=%u la=%u | loss %.4f (ema %.4f) | lr %.2e | %.0fs %.0f tok/s\n",
                   step, A.steps, T_input, lq, la, loss, ema_loss, lr_now, el, tps);
            fflush(stdout);
        }
        if (step > 0 && step % A.ckpt_every == 0) {
            char ckpt[512];
            snprintf(ckpt, sizeof(ckpt), "%s.step%d", A.out_path, step);
            lora_save(lr, &cfg, ckpt);
        }

        free(tokens);
        tape_drop_temporaries(n_init_entries);
    }

    lora_save(lr, &cfg, A.out_path);

    char meta[512];
    snprintf(meta, sizeof(meta), "%s.meta", A.out_path);
    FILE* mf = fopen(meta, "w");
    if (mf) {
        fprintf(mf, "steps=%d\n", A.steps);
        fprintf(mf, "ema_loss=%.4f\n", ema_loss);
        fprintf(mf, "rank=%d\n", A.rank);
        fprintf(mf, "alpha=%.2f\n", A.alpha);
        fprintf(mf, "wall_seconds=%.0f\n", (now_ms() - train_t0) / 1000.0);
        fclose(mf);
    }

    corpus_free(corpus);
    free(base.resid_l_data);
    free(base.x0_l_data);

    printf("\n=== done. final ema_loss=%.4f, wall=%.0fs ===\n",
           ema_loss, (now_ms() - train_t0) / 1000.0);
    return 0;
}
