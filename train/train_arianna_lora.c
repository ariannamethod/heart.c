/*
 * train_arianna_lora.c — LoRA SFT for Resonance 200M on Arianna identity corpus.
 *
 * Target: Resonance 200M (RS02 magic, dual attention: Content + RRPRAM low-rank,
 *         parametric RMSNorm, sigmoid per-head gate, embedded BPE).
 * Recipe: heart.c plan v1.1 §2.2 — yent-style LoRA r=16 α=32, masked CE on
 *         "### Question: ... ### Answer: ..." form, Chuck optimizer (notorch).
 *
 * Build: make -C train train_arianna_lora
 * Run:   ./train_arianna_lora \
 *          --base   weights/resonance_200m_final.bin \
 *          --corpus weights/arianna_dataset_final_clean.txt \
 *          --rank   16 \
 *          --alpha  32 \
 *          --steps  1000 \
 *          --lr     2e-4 \
 *          --warmup 50 \
 *          --out    checkpoints/arianna_lora.bin
 *
 * Architecture (per resonance.aml/tools/resonance_forward.h:127–247):
 *   embed → for L blocks:
 *     rmsnorm_p(norm1) → xn
 *     q = wq @ xn                               [E]
 *     k = wk @ xn                               [E]
 *     v = wv @ xn                               [E]
 *     q,k = rope_even_odd(q,k, pos, D)
 *     content[h] = softmax(q[h] @ K^T / √D) @ V[h]      per head
 *     rrpram[h]  = softmax(((xn @ Wr_a[h]) @ Wr_b[h]) / √D) @ V[h]   per head
 *     blend[h]   = sigmoid(gate[h]) * content[h] + (1 - sigmoid(gate[h])) * rrpram[h]
 *     o = wo @ blend → residual
 *     rmsnorm_p(norm2) → xn2
 *     swiglu(mlp_gate @ xn2, mlp_up @ xn2) @ mlp_down → residual
 *   rmsnorm_p(norm_f) → out_head → logits
 *
 * LoRA targets: 7 projections per layer — wq, wk, wv, wo, mlp_gate, mlp_up, mlp_down.
 *               wr_a / wr_b / gate / norm1 / norm2 frozen (preserve dual-attention character).
 *
 * Optimizer: Chuck (nt_tape_chuck_step) — per CLAUDE.md ban on classical
 *            diagonal optimizer baselines.
 */

#include "notorch.h"
#ifdef USE_CUDA
  #include "notorch_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdint.h>

/* ── RS02 header → Config ───────────────────────────────────────────────── */

typedef struct {
    int V, E, H, D, B, M, T, R;
    int n_merges;
    int (*merges)[2];   /* (a, b) BPE merge table, size n_merges */
} ResonanceConfig;

/* ── Per-block tape indices (frozen base) ─────────────────────────────────── */

typedef struct {
    int wr_a_idx;            /* [H, E, R] */
    int wr_b_idx;            /* [H, R, T] */
    int gate_idx;            /* [H] (raw) */
    int g_expanded_idx;      /* [E]  pre-computed sigmoid(gate[h]) repeated */
    int g_one_minus_idx;     /* [E]  pre-computed (1 - sigmoid(gate[h])) repeated */
    int norm1_idx;           /* [E] */
    int wq_idx, wk_idx, wv_idx, wo_idx;   /* [E, E] each */
    int norm2_idx;           /* [E] */
    int wgate_idx, wup_idx;  /* [M, E] each */
    int wdown_idx;           /* [E, M] */
    /* RRPRAM combined buffer: [H, E*R + R*T] for nt_rrpram_lowrank_attention */
    int wr_combined_idx;
} BlockBase;

typedef struct {
    int tok_emb_idx;        /* [V, E] */
    int norm_f_idx;          /* [E] */
    int out_head_idx;        /* [V, E] */
    BlockBase* blocks;       /* [B] */
} ResonanceBase;

/* ── LoRA — 7 projections per layer ────────────────────────────────────── */

#define NUM_LORA_PROJ 7
#define LP_Q     0
#define LP_K     1
#define LP_V     2
#define LP_O     3
#define LP_GATE  4   /* MLP gate (mlp_gate) */
#define LP_UP    5   /* MLP up (mlp_up) */
#define LP_DOWN  6   /* MLP down (mlp_down) */

typedef struct {
    int A_idx;   /* [out_dim, rank] */
    int B_idx;   /* [rank, in_dim] */
} LoRABlock;

typedef struct {
    int   rank;
    float alpha;
    float scale;        /* alpha / rank */
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
    a->base_path   = "weights/resonance_200m_final.bin";
    a->corpus_path = "weights/arianna_dataset_final_clean.txt";
    a->out_path    = "checkpoints/arianna_lora.bin";
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

/* ── RS02 loader ─────────────────────────────────────────────────────────── */

/* Reads RS02 magic + 8-int config + n_merges + merges + raw fp32 weights blob.
 * Pushes each tensor to notorch tape as a frozen param.
 * Pre-computes sigmoid(gate) expanded to [E] for the per-head blend. */
static int resonance_load(const char* path, ResonanceConfig* cfg, ResonanceBase* base) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[load] cannot open %s\n", path); return 1; }

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return 1; }
    if (magic != 0x52533032) {
        fprintf(stderr, "[load] bad magic 0x%08x (expected 'RS02' = 0x52533032)\n", magic);
        fclose(f); return 1;
    }

    /* RS02 header is 9 int32s — first 8 are configs (E,B,T,H,D,R,M,V),
     * the 9th is reserved/unused (resonance_forward.h:264 reads 9, uses 8). */
    int hdr[9];
    if (fread(hdr, 4, 9, f) != 9) { fclose(f); return 1; }
    cfg->E = hdr[0]; cfg->B = hdr[1]; cfg->T = hdr[2]; cfg->H = hdr[3];
    cfg->D = hdr[4]; cfg->R = hdr[5]; cfg->M = hdr[6]; cfg->V = hdr[7];

    fprintf(stderr, "[load] V=%d E=%d H=%d D=%d B=%d M=%d T=%d R=%d\n",
            cfg->V, cfg->E, cfg->H, cfg->D, cfg->B, cfg->M, cfg->T, cfg->R);

    /* Sanity */
    if (cfg->H * cfg->D != cfg->E) {
        fprintf(stderr, "[load] H*D (%d) != E (%d)\n", cfg->H * cfg->D, cfg->E);
        fclose(f); return 1;
    }

    /* BPE merges */
    uint32_t n_merges_u32;
    if (fread(&n_merges_u32, 4, 1, f) != 1) { fclose(f); return 1; }
    cfg->n_merges = (int)n_merges_u32;
    cfg->merges = malloc((size_t)cfg->n_merges * 2 * sizeof(int));
    for (int i = 0; i < cfg->n_merges; i++) {
        int triple[3];
        if (fread(triple, 4, 3, f) != 3) { fclose(f); return 1; }
        cfg->merges[i][0] = triple[0];
        cfg->merges[i][1] = triple[1];
    }
    fprintf(stderr, "[load] BPE merges loaded: %d\n", cfg->n_merges);

    /* Total raw fp32 floats — matches resonance_load() in resonance_forward.h:289 */
    size_t np = 2 * (size_t)cfg->V * cfg->E + 1L * cfg->E;
    np += (size_t)cfg->B * (
        cfg->E + 3L * cfg->E * cfg->E
        + (size_t)cfg->H * cfg->E * cfg->R
        + (size_t)cfg->H * cfg->R * cfg->T
        + cfg->H + (size_t)cfg->E * cfg->E + cfg->E
        + 3L * cfg->M * cfg->E
    );
    float* blob = malloc(np * sizeof(float));
    if (!blob) { fclose(f); return 1; }
    size_t got = fread(blob, sizeof(float), np, f);
    fclose(f);
    if (got != np) {
        fprintf(stderr, "[load] short read: %zu/%zu floats\n", got, np);
        free(blob); return 1;
    }
    fprintf(stderr, "[load] %.1fM weights read\n", np / 1e6);

    /* ── Walk the blob in the SAME order as assign() in resonance_forward.h:94 ── */
    float* p = blob;
    int E = cfg->E, B = cfg->B, H = cfg->H, R = cfg->R, T = cfg->T, M = cfg->M, V = cfg->V;
    int D = cfg->D;

    /* tok_emb [V, E] */
    {
        int sh[2] = {V, E};
        base->tok_emb_idx = load_frozen_param(p, sh, 2);
        p += (size_t)V * E;
    }

    base->blocks = calloc(B, sizeof(BlockBase));

    for (int bl = 0; bl < B; bl++) {
        BlockBase* blk = &base->blocks[bl];

        /* Direct Parameters first: wr_a, wr_b, gate */
        {
            int sh[3] = {H, E, R};
            blk->wr_a_idx = load_frozen_param(p, sh, 3);
            p += (size_t)H * E * R;
        }
        {
            int sh[3] = {H, R, T};
            blk->wr_b_idx = load_frozen_param(p, sh, 3);
            p += (size_t)H * R * T;
        }
        /* gate raw [H] — also build pre-computed expanded forms */
        const float* gate_raw = p;
        {
            int sh[1] = {H};
            blk->gate_idx = load_frozen_param(p, sh, 1);
            p += H;
        }

        /* Pre-compute sigmoid(gate[h]) expanded to [E] (g per-head, repeated D dims).
         * This bakes the sigmoid into a frozen tape param so the tape graph stays
         * differentiable on the LoRA path without backpropping through gate. */
        {
            float* g_exp = malloc((size_t)E * sizeof(float));
            float* g_inv = malloc((size_t)E * sizeof(float));
            for (int h = 0; h < H; h++) {
                float g = 1.0f / (1.0f + expf(-gate_raw[h]));
                for (int d = 0; d < D; d++) {
                    g_exp[h * D + d] = g;
                    g_inv[h * D + d] = 1.0f - g;
                }
            }
            int sh[1] = {E};
            blk->g_expanded_idx  = load_frozen_param(g_exp, sh, 1);
            blk->g_one_minus_idx = load_frozen_param(g_inv, sh, 1);
            free(g_exp); free(g_inv);
        }

        /* Build wr_combined for nt_rrpram_lowrank_attention.
         * Spec (notorch.h:404): "wr_combined holds Wr_a (size H*E*R) followed by
         * Wr_b (size H*R*T_r), so total length = H*R*(E+T_r). Rank derived from
         * tensor length: R = len / (H * (E + T))". So we just concatenate. */
        {
            size_t a_len = (size_t)H * E * R;
            size_t b_len = (size_t)H * R * T;
            float* wr_combined = malloc((a_len + b_len) * sizeof(float));
            /* wr_a is currently at p - (H*E*R + H*R*T + H) — already loaded. We need
             * to re-read from the original blob slice. Simpler: keep pointers. */
            /* Re-walk: blob start of this block's wr_a was at (p - H - H*R*T - H*E*R). */
            const float* a_src = p - (size_t)H - (size_t)H*R*T - (size_t)H*E*R;
            const float* b_src = p - (size_t)H - (size_t)H*R*T;
            memcpy(wr_combined, a_src, a_len * sizeof(float));
            memcpy(wr_combined + a_len, b_src, b_len * sizeof(float));
            int sh[1] = {(int)(a_len + b_len)};
            blk->wr_combined_idx = load_frozen_param(wr_combined, sh, 1);
            free(wr_combined);
        }

        /* Sub-Module weights in init order: norm1, wq, wk, wv, wo, norm2,
         * mlp_gate, mlp_up, mlp_down */
        { int sh[1] = {E};   blk->norm1_idx   = load_frozen_param(p, sh, 1); p += E; }
        { int sh[2] = {E,E}; blk->wq_idx      = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->wk_idx      = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->wv_idx      = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[2] = {E,E}; blk->wo_idx      = load_frozen_param(p, sh, 2); p += (size_t)E*E; }
        { int sh[1] = {E};   blk->norm2_idx   = load_frozen_param(p, sh, 1); p += E; }
        { int sh[2] = {M,E}; blk->wgate_idx   = load_frozen_param(p, sh, 2); p += (size_t)M*E; }
        { int sh[2] = {M,E}; blk->wup_idx     = load_frozen_param(p, sh, 2); p += (size_t)M*E; }
        { int sh[2] = {E,M}; blk->wdown_idx   = load_frozen_param(p, sh, 2); p += (size_t)E*M; }
    }

    { int sh[1] = {E};   base->norm_f_idx   = load_frozen_param(p, sh, 1); p += E; }
    { int sh[2] = {V,E}; base->out_head_idx = load_frozen_param(p, sh, 2); p += (size_t)V*E; }

    if ((size_t)(p - blob) != np) {
        fprintf(stderr, "[load] walk mismatch: p-blob=%zu vs np=%zu\n",
                (size_t)(p - blob), np);
        free(blob); return 1;
    }
    free(blob);

    fprintf(stderr, "[load] frozen base loaded onto tape\n");
    return 0;
}

/* ── LoRA init ─────────────────────────────────────────────────────────── */

static LoRA* lora_init(ResonanceConfig* cfg, int rank, float alpha) {
    LoRA* lr = calloc(1, sizeof(LoRA));
    lr->rank  = rank;
    lr->alpha = alpha;
    lr->scale = alpha / (float)rank;
    lr->blocks = calloc(cfg->B, sizeof(lr->blocks[0]));

    /* Per-projection (out, in) shapes for Resonance:
     *   wq, wk, wv, wo   : [E, E]
     *   mlp_gate, mlp_up : [M, E]
     *   mlp_down         : [E, M]
     */
    int proj_dims[NUM_LORA_PROJ][2] = {
        { cfg->E, cfg->E },   /* Q  */
        { cfg->E, cfg->E },   /* K  */
        { cfg->E, cfg->E },   /* V  */
        { cfg->E, cfg->E },   /* O  */
        { cfg->M, cfg->E },   /* GATE */
        { cfg->M, cfg->E },   /* UP   */
        { cfg->E, cfg->M },   /* DOWN */
    };

    long total = 0;
    for (int bl = 0; bl < cfg->B; bl++) {
        for (int p = 0; p < NUM_LORA_PROJ; p++) {
            int out_dim = proj_dims[p][0];
            int in_dim  = proj_dims[p][1];
            int A_shape[2] = { out_dim, rank };
            int B_shape[2] = { rank,    in_dim };
            nt_tensor* A = nt_tensor_new_shape(A_shape, 2);
            nt_tensor_fill(A, 0.0f);                       /* A starts zero — LoRA convention */
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

/* y = y + scale * A @ (B @ x) — matches train_doe_coder.c:205 lora_apply */
static int lora_apply(LoRA* lr, int layer, int proj, int x_idx, int y_idx, int T) {
    LoRABlock* blk = &lr->blocks[layer][proj];
    int b_x_idx    = nt_seq_linear(blk->B_idx, x_idx, T);
    int scaled_idx = nt_scale(b_x_idx, lr->scale);
    int delta_idx  = nt_seq_linear(blk->A_idx, scaled_idx, T);
    return nt_add(y_idx, delta_idx);
}

/* y = w @ x + LoRA delta, no bias path needed (Resonance has no projection biases) */
static int proj_with_lora(LoRA* lr, int layer, int proj,
                          int w_idx, int x_idx, int T) {
    int y_idx = nt_seq_linear(w_idx, x_idx, T);
    return lora_apply(lr, layer, proj, x_idx, y_idx, T);
}

/* ── Per-step seq utilities ──────────────────────────────────────────── */

/* Tile [E] vector to [T, E] sequence — used for per-head gate broadcast. */
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

/* Embedding lookup — gather rows of tok_emb at token positions */
static int make_embedding_lookup(int tok_emb_idx, int* tokens, int T, int E) {
    nt_tape* tp = nt_tape_get();
    nt_tensor* W = tp->entries[tok_emb_idx].output;
    int shape[2] = {T, E};
    nt_tensor* h = nt_tensor_new_shape(shape, 2);
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        memcpy(h->data + (size_t)t * E, W->data + (size_t)tok * E, (size_t)E * sizeof(float));
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
    for (int i = 0; i < T_input; i++) {
        /* tokens[0..lq-1] = question prompt → mask to 0 (don't train on Q).
         * tokens[lq..T-1] = answer → mask to 1. After shift by 1 for next-token
         * prediction, position i predicts tokens[i+1], so we want mask[i] = 1
         * iff tokens[i+1] is in the answer span: i+1 >= lq → i >= lq - 1. */
        m->data[i] = (i >= lq - 1) ? 1.0f : 0.0f;
    }
    int idx = nt_tape_param(m);
    nt_tape_freeze_param(idx);
    nt_tape_no_decay(idx);
    return idx;
}

/* ── Forward (Resonance dual attention with LoRA) ─────────────────────── */

static int forward(ResonanceBase* base, LoRA* lr, ResonanceConfig* cfg,
                   int* tokens, int T_input) {
    int E = cfg->E, M = cfg->M, B = cfg->B, H = cfg->H, D = cfg->D, T = cfg->T;
    (void)T;

    int h = make_embedding_lookup(base->tok_emb_idx, tokens, T_input, E);

    for (int bl = 0; bl < B; bl++) {
        BlockBase* blk = &base->blocks[bl];

        /* === norm1 → Q/K/V === */
        int xn = nt_seq_rmsnorm(h, blk->norm1_idx, T_input, E);

        int q = proj_with_lora(lr, bl, LP_Q, blk->wq_idx, xn, T_input);
        int k = proj_with_lora(lr, bl, LP_K, blk->wk_idx, xn, T_input);
        int v = proj_with_lora(lr, bl, LP_V, blk->wv_idx, xn, T_input);

        /* RoPE — Resonance uses base 10000 (resonance_forward.h:56) */
        q = nt_rope_freq(q, T_input, D, 10000.0f);
        k = nt_rope_freq(k, T_input, D, 10000.0f);

        /* Content attention: full MHA (Resonance has KV = H, no GQA).
         * Use nt_mh_causal_attention — it has a GPU path; nt_gqa_causal_attention
         * is CPU-only AND doesn't ensure_cpu its inputs, so reads stale CPU data
         * when GPU mode is on (notorch.c:3127). */
        int content = nt_mh_causal_attention(q, k, v, T_input, D);

        /* RRPRAM low-rank attention (broadcast pattern, canonical Janus per
         * dario/infer_v4.c:218-249): mid[h,r] = sum_t sum_e x[t,e]*wr_a[h,e,r]
         * is layer-broadcast across all positions. Per-position op
         * nt_rrpram_lowrank_attention is function-class different and causes
         * training plateau near ln(V) = uniform-distribution loss. */
        int rrpram = nt_rrpram_broadcast_attention(blk->wr_combined_idx,
                                                    xn, v, T_input, E, H, D);

        /* Per-head sigmoid gate blend: pre-computed g_expanded[E] on tape.
         * blend = g_tiled * content + (1-g_tiled) * rrpram */
        int g_tiled   = tile_per_seq(blk->g_expanded_idx,  T_input, E);
        int inv_tiled = tile_per_seq(blk->g_one_minus_idx, T_input, E);
        int gated_c   = nt_mul(content, g_tiled);
        int gated_r   = nt_mul(rrpram,  inv_tiled);
        int blend     = nt_add(gated_c, gated_r);

        /* WO + LoRA + residual */
        int o = proj_with_lora(lr, bl, LP_O, blk->wo_idx, blend, T_input);
        h = nt_add(h, o);

        /* === norm2 → SwiGLU MLP === */
        int xn2 = nt_seq_rmsnorm(h, blk->norm2_idx, T_input, E);

        int gate_lin = proj_with_lora(lr, bl, LP_GATE, blk->wgate_idx, xn2, T_input);
        int up_lin   = proj_with_lora(lr, bl, LP_UP,   blk->wup_idx,   xn2, T_input);
        int swi      = nt_swiglu(gate_lin, up_lin);
        int down     = proj_with_lora(lr, bl, LP_DOWN, blk->wdown_idx, swi, T_input);
        h = nt_add(h, down);
        (void)M;
    }

    /* Final norm + head */
    int hn     = nt_seq_rmsnorm(h, base->norm_f_idx, T_input, E);
    int logits = nt_seq_linear(base->out_head_idx, hn, T_input);
    return logits;
}

/* ── Tape truncation ─────────────────────────────────────────────────── */

static void tape_drop_temporaries(int keep_count) {
    nt_tape* tp = nt_tape_get();
    while (tp->count > keep_count) {
        nt_tape_entry* e = &tp->entries[tp->count - 1];
        if (e->output) {
            nt_tensor_free(e->output);
            e->output = NULL;
        }
        if (e->grad) {
            nt_tensor_free(e->grad);
            e->grad = NULL;
        }
        tp->count--;
    }
}

/* ── Dataset: on-the-fly Q/A from text corpus ───────────────────────── */

typedef struct {
    char**  q_texts;
    char**  a_texts;
    int     n_pairs;
} ArianaCorpus;

/* Parse text into Q:/A: pairs.
 *   Format: alternating "Q: ...\nA: ...\nQ: ...\nA: ..."
 *   Each Q/A may span multiple lines until the next Q: or A: marker. */
static ArianaCorpus* corpus_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[corpus] cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    buf[sz] = 0;
    fclose(f);

    /* First pass: count pairs */
    int n_pairs = 0;
    for (long i = 0; i < sz - 1; i++) {
        if (i == 0 || buf[i-1] == '\n') {
            if (buf[i] == 'Q' && buf[i+1] == ':') n_pairs++;
        }
    }

    ArianaCorpus* c = calloc(1, sizeof(ArianaCorpus));
    c->q_texts = calloc(n_pairs, sizeof(char*));
    c->a_texts = calloc(n_pairs, sizeof(char*));
    c->n_pairs = 0;

    /* Second pass: extract */
    long i = 0;
    while (i < sz) {
        /* find next Q: at line start */
        while (i < sz && !(buf[i] == 'Q' && (i+1 < sz && buf[i+1] == ':') &&
                           (i == 0 || buf[i-1] == '\n')))
            i++;
        if (i >= sz) break;
        long q_start = i + 2;  /* skip 'Q:' */
        while (q_start < sz && (buf[q_start] == ' ' || buf[q_start] == '\t')) q_start++;
        long q_end = q_start;
        /* find 'A:' at line start after Q */
        while (q_end < sz && !(buf[q_end] == 'A' && (q_end+1 < sz && buf[q_end+1] == ':') &&
                               (q_end == 0 || buf[q_end-1] == '\n')))
            q_end++;
        if (q_end >= sz) break;
        long a_start = q_end + 2;
        while (a_start < sz && (buf[a_start] == ' ' || buf[a_start] == '\t')) a_start++;
        long a_end = a_start;
        /* find next 'Q:' at line start after A */
        while (a_end < sz && !(buf[a_end] == 'Q' && (a_end+1 < sz && buf[a_end+1] == ':') &&
                               (a_end == 0 || buf[a_end-1] == '\n')))
            a_end++;

        size_t q_len = (size_t)(q_end - q_start);
        size_t a_len = (size_t)(a_end - a_start);
        /* Trim trailing whitespace/newlines */
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
    fprintf(stderr, "[corpus] %d Q/A pairs parsed from %s\n", c->n_pairs, path);
    return c;
}

static void corpus_free(ArianaCorpus* c) {
    if (!c) return;
    for (int i = 0; i < c->n_pairs; i++) { free(c->q_texts[i]); free(c->a_texts[i]); }
    free(c->q_texts); free(c->a_texts); free(c);
}

/* Encode "### Question: <q>\n### Answer: <a>" via Resonance BPE.
 * Returns malloc'd token array; *lq_out = #tokens in Q+marker, *la_out = #tokens in A.
 * Caller frees. */
static int* encode_pair(nt_bpe* bpe, const char* q, const char* a,
                        uint32_t* lq_out, uint32_t* la_out, int max_total) {
    char qbuf[8192], abuf[8192];
    snprintf(qbuf, sizeof(qbuf), "### Question: %s\n### Answer: ", q);
    snprintf(abuf, sizeof(abuf), "%s", a);

    int* q_toks = malloc((size_t)max_total * sizeof(int));
    int* a_toks = malloc((size_t)max_total * sizeof(int));
    int qn = nt_bpe_encode(bpe, qbuf, (int)strlen(qbuf), q_toks, max_total);
    int an = nt_bpe_encode(bpe, abuf, (int)strlen(abuf), a_toks, max_total);
    if (qn <= 0 || an <= 0) { free(q_toks); free(a_toks); return NULL; }
    if (qn + an > max_total) {
        /* truncate answer side first */
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

/* Format (heart.c arianna_lora_v1):
 *   magic "ARLR"           (uint32 = 0x524C5241 little-endian = 'A','R','L','R')
 *   rank   (int32)
 *   alpha  (float32)
 *   B      (int32) — n_layers
 *   E,M    (int32, int32) — Resonance dims (for sanity check on load)
 *   per-layer × NUM_LORA_PROJ:
 *     A_data (out * rank floats), B_data (rank * in floats) */
static int lora_save(LoRA* lr, ResonanceConfig* cfg, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[save] cannot open %s\n", path); return 1; }
    uint32_t magic = 0x524C5241;
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

    printf("=== Arianna LoRA SFT (Resonance 200M, Chuck) ===\n");
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

    ResonanceConfig cfg = {0};
    ResonanceBase   base = {0};
    if (resonance_load(A.base_path, &cfg, &base)) return 1;

    LoRA* lr = lora_init(&cfg, A.rank, A.alpha);

    nt_bpe bpe = {0};
    nt_bpe_init(&bpe, cfg.merges, cfg.n_merges);
    fprintf(stderr, "[bpe] vocab=%d merges=%d\n", bpe.vocab_size, bpe.n_merges);
    if (bpe.vocab_size != cfg.V) {
        fprintf(stderr, "[bpe] WARN: bpe.vocab_size=%d vs cfg.V=%d\n", bpe.vocab_size, cfg.V);
    }

    ArianaCorpus* corpus = corpus_load(A.corpus_path);
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
            /* Diag: variance of logits at last position to confirm forward is alive */
            nt_tensor* lt = nt_tape_get()->entries[logits_idx].output;
            float* L = lt->data + (size_t)(T_input - 1) * cfg.V;
            float mean = 0, var = 0, mn = 1e30f, mx = -1e30f;
            for (int v = 0; v < cfg.V; v++) {
                mean += L[v];
                if (L[v] < mn) mn = L[v];
                if (L[v] > mx) mx = L[v];
            }
            mean /= cfg.V;
            for (int v = 0; v < cfg.V; v++) var += (L[v] - mean) * (L[v] - mean);
            var /= cfg.V;
            fprintf(stderr, "[diag] step0 logits[last-pos]: mean=%.4f var=%.4f min=%.4f max=%.4f\n",
                    mean, var, mn, mx);
            /* Also diag tok_emb norm — sanity-check the base loader */
            nt_tensor* te = nt_tape_get()->entries[base.tok_emb_idx].output;
            float emb_var = 0, emb_mean = 0;
            int n_emb = te->len < 100000 ? te->len : 100000;
            for (int i = 0; i < n_emb; i++) emb_mean += te->data[i];
            emb_mean /= n_emb;
            for (int i = 0; i < n_emb; i++) emb_var += (te->data[i] - emb_mean) * (te->data[i] - emb_mean);
            emb_var /= n_emb;
            fprintf(stderr, "[diag] tok_emb (first %d floats): mean=%.6f var=%.6f\n",
                    n_emb, emb_mean, emb_var);
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
    free(cfg.merges);

    printf("\n=== done. final ema_loss=%.4f, wall=%.0fs ===\n",
           ema_loss, (now_ms() - train_t0) / 1000.0);
    return 0;
}
