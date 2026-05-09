/*
 * train_doe_lora.c — LoRA SFT for Janus-base (Qwen 2.5-0.5B GGUF) on DoE
 *                    personality corpus.
 *
 * Lifted from DoE.coder/train/train_doe_coder.c (recipe-source-of-record per
 * heart.c plan v1.1 BLOCKER C3). Two surgical changes:
 *   1. nt_tape_adamw_step → nt_tape_chuck_step  (CLAUDE.md ban on diagonal
 *      optimizer baselines; default = Chuck per memory/feedback_adam_ban_*.md)
 *   2. gpu_init() + nt_set_gpu_mode(1) at main entry for A100 SXM acceleration.
 *
 * Target: Qwen 2.5-0.5B base GGUF (E=896, L=24, H=14, KV=2, FFN=4864, V=151936)
 * Recipe: yent — LoRA r=16 α=32, 1000 steps, "### Question: ... ### Answer: ..."
 *         format, loss masked to answer tokens.
 *
 * Build: make -C train train_doe_lora
 * Run:   ./train/train_doe_lora \
 *          --base   weights/qwen25_05b_base_q8_0.gguf \
 *          --data   data/doe_pure.bin \
 *          --rank   16 \
 *          --steps  1000 \
 *          --lr     2e-4 \
 *          --out    checkpoints/doe_lora.bin
 */

#include "notorch.h"
#include "gguf.h"
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

// ── Model dims (filled at load time, defaults match Qwen 2.5-Coder-0.5B) ──────

typedef struct {
    int E;          // embed_dim       — 896
    int L;          // n_layers        — 24
    int H;          // n_heads         — 14
    int KV;         // n_kv_heads      —  2
    int HD;         // head_dim        — 64
    int Q_DIM;      // H * HD          — 896
    int KVD;        // KV * HD         — 128
    int FFN;        // ffn_dim         — 4864
    int V;          // vocab           — 151936
    float rope_base;
    float rms_eps;
    int   has_output_weight;
} Config;

typedef struct {
    int tok_emb_idx;
    int out_norm_idx;
    int out_weight_idx;
    struct {
        int attn_norm_idx;
        int wq_idx, wk_idx, wv_idx, wo_idx;
        int q_bias_idx, k_bias_idx, v_bias_idx;     // -1 if absent
        int ffn_norm_idx;
        int wgate_idx, wup_idx, wdown_idx;
    } *layers;
} BaseModel;

#define NUM_LORA_PROJ 7   // q, k, v, o, gate, up, down
#define LP_Q    0
#define LP_K    1
#define LP_V    2
#define LP_O    3
#define LP_GATE 4
#define LP_UP   5
#define LP_DOWN 6

typedef struct {
    int A_idx;   // [out, r]
    int B_idx;   // [r, in]
} LoRABlock;

typedef struct {
    int rank;
    float scale;             // α / rank
    LoRABlock (*blocks)[NUM_LORA_PROJ];   // [L][7]
} LoRA;

// ── Dataset reader ────────────────────────────────────────────────────────────

typedef struct {
    uint32_t n_examples;
    uint32_t vocab_size;
    uint32_t max_len;
    uint8_t* arena;
    size_t   arena_bytes;
    size_t*  offsets;
} Dataset;

static Dataset* dataset_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s: failed\n", path); return NULL; }
    Dataset* ds = (Dataset*)calloc(1, sizeof(Dataset));
    fread(&ds->n_examples, 4, 1, f);
    fread(&ds->vocab_size, 4, 1, f);
    fread(&ds->max_len,    4, 1, f);
    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    ds->arena_bytes = (size_t)(end - start);
    fseek(f, start, SEEK_SET);
    ds->arena = (uint8_t*)malloc(ds->arena_bytes);
    fread(ds->arena, 1, ds->arena_bytes, f);
    fclose(f);
    ds->offsets = (size_t*)calloc(ds->n_examples, sizeof(size_t));
    size_t off = 0;
    for (uint32_t i = 0; i < ds->n_examples; i++) {
        ds->offsets[i] = off;
        uint32_t lq, la;
        memcpy(&lq, ds->arena + off, 4);
        memcpy(&la, ds->arena + off + 4, 4);
        off += 8 + (lq + la) * 4;
    }
    printf("[dataset] %u examples, vocab=%u, max_len=%u, arena=%.1f MB\n",
           ds->n_examples, ds->vocab_size, ds->max_len, ds->arena_bytes / 1024.0 / 1024.0);
    return ds;
}

static void dataset_get(Dataset* ds, uint32_t idx, int** tokens_out, uint32_t* lq_out, uint32_t* la_out) {
    uint8_t* p = ds->arena + ds->offsets[idx];
    memcpy(lq_out, p, 4);
    memcpy(la_out, p + 4, 4);
    *tokens_out = (int*)(p + 8);
}

static void dataset_free(Dataset* ds) {
    if (!ds) return;
    free(ds->arena); free(ds->offsets); free(ds);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static double now_ms(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static nt_tensor* tensor_from_floats(const float* src, int* shape, int ndim) {
    nt_tensor* t = nt_tensor_new_shape(shape, ndim);
    if (!t) return NULL;
    memcpy(t->data, src, t->len * sizeof(float));
    return t;
}

// Embedding lookup [T, E] from token IDs, frozen on tape.
static int make_embedding_lookup(BaseModel* bm, Config* cfg, int* tokens, int T) {
    nt_tape* tp = nt_tape_get();
    nt_tensor* wte = tp->entries[bm->tok_emb_idx].output;   // [V, E]
    int sh[2] = { T, cfg->E };
    nt_tensor* emb = nt_tensor_new_shape(sh, 2);
    if (!emb) return -1;
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= cfg->V) tok = 0;
        memcpy(emb->data + t * cfg->E,
               wte->data + tok * cfg->E,
               cfg->E * sizeof(float));
    }
    int idx = nt_tape_param(emb);
    nt_tape_freeze_param(idx);
    return idx;
}

// Broadcast bias [dim] → [T, dim] (frozen on tape, fresh per step).
static int make_tiled_bias(int bias_idx, int T, int dim) {
    if (bias_idx < 0) return -1;
    nt_tape* tp = nt_tape_get();
    nt_tensor* b = tp->entries[bias_idx].output;
    int sh[2] = { T, dim };
    nt_tensor* tiled = nt_tensor_new_shape(sh, 2);
    if (!tiled) return -1;
    for (int t = 0; t < T; t++)
        memcpy(tiled->data + t * dim, b->data, dim * sizeof(float));
    int idx = nt_tape_param(tiled);
    nt_tape_freeze_param(idx);
    return idx;
}

// targets[t] = tokens[t+1] for t in 0..T_input-1, as float tensor for cross_entropy.
static int make_targets(int* tokens, int T_input) {
    int sh[1] = { T_input };
    nt_tensor* t = nt_tensor_new_shape(sh, 1);
    if (!t) return -1;
    for (int i = 0; i < T_input; i++) t->data[i] = (float)tokens[i + 1];
    int idx = nt_tape_param(t);
    nt_tape_freeze_param(idx);
    return idx;
}

// mask[i] = 1 if input position i predicts an answer token, else 0.
// answer tokens are at original positions [lq, lq+la-1].
// position i predicts tokens[i+1], so we want i+1 >= lq → i >= lq-1.
static int make_loss_mask(int lq, int T_input) {
    int sh[1] = { T_input };
    nt_tensor* m = nt_tensor_new_shape(sh, 1);
    if (!m) return -1;
    for (int i = 0; i < T_input; i++) m->data[i] = (i >= lq - 1) ? 1.0f : 0.0f;
    int idx = nt_tape_param(m);
    nt_tape_freeze_param(idx);
    return idx;
}

// LoRA composite delta on top of an existing y_idx:
//   y' = y + scale * A · (B · x)
static int lora_apply(LoRA* lr, int layer, int proj, int x_idx, int y_idx, int T) {
    LoRABlock* blk = &lr->blocks[layer][proj];
    int b_x_idx     = nt_seq_linear(blk->B_idx, x_idx, T);
    int scaled_idx  = nt_scale(b_x_idx, lr->scale);
    int delta_idx   = nt_seq_linear(blk->A_idx, scaled_idx, T);
    return nt_add(y_idx, delta_idx);
}

// Linear with optional bias add, then LoRA delta.
static int proj_with_bias_lora(LoRA* lr, int layer, int proj,
                                int w_idx, int x_idx, int bias_idx,
                                int T, int dim) {
    int y_idx = nt_seq_linear(w_idx, x_idx, T);
    if (bias_idx >= 0) {
        int tb = make_tiled_bias(bias_idx, T, dim);
        y_idx = nt_add(y_idx, tb);
    }
    return lora_apply(lr, layer, proj, x_idx, y_idx, T);
}

// ── Forward ───────────────────────────────────────────────────────────────────

static int forward(BaseModel* bm, LoRA* lr, Config* cfg, int* tokens, int T_input) {
    int E = cfg->E, FFN = cfg->FFN;
    int Q_DIM = cfg->Q_DIM, KVD = cfg->KVD;

    int h = make_embedding_lookup(bm, cfg, tokens, T_input);

    for (int l = 0; l < cfg->L; l++) {
        int xn = nt_seq_rmsnorm(h, bm->layers[l].attn_norm_idx, T_input, E);

        int q = proj_with_bias_lora(lr, l, LP_Q, bm->layers[l].wq_idx, xn,
                                     bm->layers[l].q_bias_idx, T_input, Q_DIM);
        int k = proj_with_bias_lora(lr, l, LP_K, bm->layers[l].wk_idx, xn,
                                     bm->layers[l].k_bias_idx, T_input, KVD);
        int v = proj_with_bias_lora(lr, l, LP_V, bm->layers[l].wv_idx, xn,
                                     bm->layers[l].v_bias_idx, T_input, KVD);

        q = nt_rope_freq(q, T_input, cfg->HD, cfg->rope_base);
        k = nt_rope_freq(k, T_input, cfg->HD, cfg->rope_base);

        int attn = nt_gqa_causal_attention(q, k, v, T_input, cfg->HD, cfg->H, cfg->KV);

        int o_lin = nt_seq_linear(bm->layers[l].wo_idx, attn, T_input);
        int o     = lora_apply(lr, l, LP_O, attn, o_lin, T_input);
        h         = nt_add(h, o);

        int xn2 = nt_seq_rmsnorm(h, bm->layers[l].ffn_norm_idx, T_input, E);

        int gate_lin = nt_seq_linear(bm->layers[l].wgate_idx, xn2, T_input);
        int gate     = lora_apply(lr, l, LP_GATE, xn2, gate_lin, T_input);

        int up_lin = nt_seq_linear(bm->layers[l].wup_idx, xn2, T_input);
        int up     = lora_apply(lr, l, LP_UP, xn2, up_lin, T_input);

        int swi  = nt_swiglu(gate, up);
        int down_lin = nt_seq_linear(bm->layers[l].wdown_idx, swi, T_input);
        int down     = lora_apply(lr, l, LP_DOWN, swi, down_lin, T_input);

        h = nt_add(h, down);
    }

    int hn = nt_seq_rmsnorm(h, bm->out_norm_idx, T_input, E);
    int logits = nt_seq_linear(bm->out_weight_idx, hn, T_input);
    return logits;
}

// ── Base model load ───────────────────────────────────────────────────────────

static BaseModel* base_load(gguf_file* gf, Config* cfg) {
    cfg->L  = gf->n_layers;
    cfg->E  = gf->embed_dim;
    cfg->H  = gf->n_heads;
    cfg->KV = gf->n_kv_heads;
    cfg->FFN = gf->ffn_dim;
    cfg->rope_base = gf->rope_freq_base;
    cfg->rms_eps   = gf->rms_eps;

    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti < 0) { fprintf(stderr, "missing blk.0.attn_q.weight\n"); return NULL; }
    cfg->Q_DIM = (int)gf->tensors[ti].shape[1];
    cfg->HD    = cfg->Q_DIM / cfg->H;
    cfg->KVD   = cfg->HD * cfg->KV;

    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti < 0) { fprintf(stderr, "missing token_embd.weight\n"); return NULL; }
    cfg->V = (int)gf->tensors[ti].shape[1];

    int has_out = gguf_find_tensor(gf, "output.weight") >= 0;
    cfg->has_output_weight = has_out;

    printf("[base] E=%d L=%d H=%d KV=%d HD=%d FFN=%d V=%d rope=%.0f rms_eps=%.1e (out=%s)\n",
           cfg->E, cfg->L, cfg->H, cfg->KV, cfg->HD, cfg->FFN, cfg->V,
           cfg->rope_base, cfg->rms_eps, has_out ? "untied" : "tied");

    BaseModel* bm = (BaseModel*)calloc(1, sizeof(BaseModel));
    bm->layers = calloc(cfg->L, sizeof(bm->layers[0]));

    #define LOAD_AS_PARAM(name, target_idx, sh0, sh1) do { \
        int _ti = gguf_find_tensor(gf, name); \
        if (_ti < 0) { fprintf(stderr, "missing %s\n", name); return NULL; } \
        float* _data = gguf_dequant(gf, _ti); \
        if (!_data) { fprintf(stderr, "dequant %s failed\n", name); return NULL; } \
        int _shape[2] = { sh0, sh1 }; \
        nt_tensor* _t = tensor_from_floats(_data, _shape, (sh1 > 0 ? 2 : 1)); \
        if (!_t) { fprintf(stderr, "tensor %s [%d,%d] failed\n", name, sh0, sh1); free(_data); return NULL; } \
        free(_data); \
        target_idx = nt_tape_param(_t); \
        nt_tape_no_decay(target_idx); \
        nt_tape_freeze_param(target_idx); \
    } while (0)

    LOAD_AS_PARAM("token_embd.weight",  bm->tok_emb_idx,    cfg->V, cfg->E);
    LOAD_AS_PARAM("output_norm.weight", bm->out_norm_idx,   cfg->E, 0);
    if (has_out) {
        LOAD_AS_PARAM("output.weight",   bm->out_weight_idx, cfg->V, cfg->E);
    } else {
        bm->out_weight_idx = bm->tok_emb_idx;
    }

    char nbuf[128];
    for (int l = 0; l < cfg->L; l++) {
        #define L_LOAD(field, fmt, sh0, sh1) do { \
            snprintf(nbuf, sizeof(nbuf), fmt, l); \
            LOAD_AS_PARAM(nbuf, bm->layers[l].field, sh0, sh1); \
        } while (0)

        L_LOAD(attn_norm_idx, "blk.%d.attn_norm.weight", cfg->E, 0);
        L_LOAD(wq_idx, "blk.%d.attn_q.weight", cfg->Q_DIM, cfg->E);
        L_LOAD(wk_idx, "blk.%d.attn_k.weight", cfg->KVD,   cfg->E);
        L_LOAD(wv_idx, "blk.%d.attn_v.weight", cfg->KVD,   cfg->E);
        L_LOAD(wo_idx, "blk.%d.attn_output.weight", cfg->E, cfg->Q_DIM);

        bm->layers[l].q_bias_idx = -1;
        bm->layers[l].k_bias_idx = -1;
        bm->layers[l].v_bias_idx = -1;
        snprintf(nbuf, sizeof(nbuf), "blk.%d.attn_q.bias", l);
        if (gguf_find_tensor(gf, nbuf) >= 0) L_LOAD(q_bias_idx, "blk.%d.attn_q.bias", cfg->Q_DIM, 0);
        snprintf(nbuf, sizeof(nbuf), "blk.%d.attn_k.bias", l);
        if (gguf_find_tensor(gf, nbuf) >= 0) L_LOAD(k_bias_idx, "blk.%d.attn_k.bias", cfg->KVD, 0);
        snprintf(nbuf, sizeof(nbuf), "blk.%d.attn_v.bias", l);
        if (gguf_find_tensor(gf, nbuf) >= 0) L_LOAD(v_bias_idx, "blk.%d.attn_v.bias", cfg->KVD, 0);

        L_LOAD(ffn_norm_idx, "blk.%d.ffn_norm.weight", cfg->E, 0);
        L_LOAD(wgate_idx, "blk.%d.ffn_gate.weight", cfg->FFN, cfg->E);
        L_LOAD(wup_idx,   "blk.%d.ffn_up.weight",   cfg->FFN, cfg->E);
        L_LOAD(wdown_idx, "blk.%d.ffn_down.weight", cfg->E,   cfg->FFN);
        #undef L_LOAD
    }
    #undef LOAD_AS_PARAM

    nt_tape* tp = nt_tape_get();
    printf("[base] loaded %d frozen params on tape\n", tp->n_params);
    return bm;
}

// ── LoRA init ─────────────────────────────────────────────────────────────────

static LoRA* lora_init(Config* cfg, int rank, float alpha) {
    LoRA* lr = (LoRA*)calloc(1, sizeof(LoRA));
    lr->rank = rank;
    lr->scale = alpha / (float)rank;
    lr->blocks = calloc(cfg->L, sizeof(lr->blocks[0]));

    int proj_dims[NUM_LORA_PROJ][2] = {
        { cfg->Q_DIM, cfg->E },
        { cfg->KVD,   cfg->E },
        { cfg->KVD,   cfg->E },
        { cfg->E,     cfg->Q_DIM },
        { cfg->FFN,   cfg->E },
        { cfg->FFN,   cfg->E },
        { cfg->E,     cfg->FFN },
    };

    long total = 0;
    for (int l = 0; l < cfg->L; l++) {
        for (int p = 0; p < NUM_LORA_PROJ; p++) {
            int out_dim = proj_dims[p][0];
            int in_dim  = proj_dims[p][1];
            int A_shape[2] = { out_dim, rank };
            int B_shape[2] = { rank,    in_dim };
            nt_tensor* A = nt_tensor_new_shape(A_shape, 2);
            nt_tensor_fill(A, 0.0f);
            nt_tensor* B = nt_tensor_new_shape(B_shape, 2);
            nt_tensor_xavier(B, in_dim, rank);
            lr->blocks[l][p].A_idx = nt_tape_param(A);
            lr->blocks[l][p].B_idx = nt_tape_param(B);
            total += A->len + B->len;
        }
    }
    printf("[lora] rank=%d alpha=%.1f scale=%.4f → %ld trainable params (%.2f MB)\n",
           rank, alpha, lr->scale, total, total * 4.0 / 1024.0 / 1024.0);
    return lr;
}

// ── Save .lora ────────────────────────────────────────────────────────────────

static int lora_save(LoRA* lr, Config* cfg, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "save %s: failed\n", path); return 1; }
    uint32_t magic = 0x41524F4C, version = 1, r32 = (uint32_t)lr->rank;
    float alpha = lr->scale * (float)lr->rank;
    uint32_t L = (uint32_t)cfg->L, np = NUM_LORA_PROJ;
    uint64_t fp = 0, reserved = 0;
    fwrite(&magic, 4, 1, f); fwrite(&version, 4, 1, f);
    fwrite(&r32, 4, 1, f); fwrite(&alpha, 4, 1, f);
    fwrite(&L, 4, 1, f); fwrite(&np, 4, 1, f);
    fwrite(&fp, 8, 1, f); fwrite(&reserved, 8, 1, f);

    nt_tape* tp = nt_tape_get();
    long total = 0;
    for (uint32_t l = 0; l < L; l++) {
        for (uint32_t p = 0; p < np; p++) {
            nt_tensor* A = tp->entries[lr->blocks[l][p].A_idx].output;
            nt_tensor* B = tp->entries[lr->blocks[l][p].B_idx].output;
            uint32_t out = (uint32_t)A->shape[0];
            uint32_t in  = (uint32_t)B->shape[1];
            fwrite(&out, 4, 1, f); fwrite(&in, 4, 1, f);
            fwrite(A->data, sizeof(float), A->len, f);
            fwrite(B->data, sizeof(float), B->len, f);
            total += A->len + B->len;
        }
    }
    fclose(f);
    printf("[lora] saved %s (%ld floats, %.2f MB)\n", path, total, total * 4.0 / 1024.0 / 1024.0);
    return 0;
}

// ── Tape cleanup: drop intermediate entries between steps, keep params ────────

static void tape_drop_temporaries(int keep_count) {
    nt_tape* tp = nt_tape_get();
    for (int i = keep_count; i < tp->count; i++) {
        if (tp->entries[i].output) {
            nt_tensor_free(tp->entries[i].output);
            tp->entries[i].output = NULL;
        }
        if (tp->entries[i].grad) {
            nt_tensor_free(tp->entries[i].grad);
            tp->entries[i].grad = NULL;
        }
    }
    tp->count = keep_count;
    // Also clear grads on params for next step
    for (int i = 0; i < keep_count; i++) {
        if (tp->entries[i].grad) {
            nt_tensor_free(tp->entries[i].grad);
            tp->entries[i].grad = NULL;
        }
    }
}

// ── Args ──────────────────────────────────────────────────────────────────────

typedef struct {
    const char* base_path;
    const char* data_path;
    const char* out_path;
    int   rank;
    int   steps;
    int   log_every;
    int   ckpt_every;
    float lr;
    float alpha;
    int   seed;
    int   max_seq;
} Args;

static void args_default(Args* a) {
    a->base_path  = "weights/doe-coder-base.gguf";
    a->data_path  = "data/doe_pure.bin";
    a->out_path   = "checkpoints/doe_coder_lora.bin";
    a->rank       = 16;
    a->steps      = 1000;
    a->log_every  = 1;
    a->ckpt_every = 100;
    a->lr         = 2e-4f;
    a->alpha      = 32.0f;
    a->seed       = 42;
    a->max_seq    = 512;
}

static int args_parse(int argc, char** argv, Args* a) {
    args_default(a);
    for (int i = 1; i < argc; i++) {
        #define ARG_STR(flag, target) if (!strcmp(argv[i], flag) && i+1 < argc) { target = argv[++i]; continue; }
        #define ARG_INT(flag, target) if (!strcmp(argv[i], flag) && i+1 < argc) { target = atoi(argv[++i]); continue; }
        #define ARG_F32(flag, target) if (!strcmp(argv[i], flag) && i+1 < argc) { target = (float)atof(argv[++i]); continue; }
        ARG_STR("--base",      a->base_path);
        ARG_STR("--data",      a->data_path);
        ARG_STR("--out",       a->out_path);
        ARG_INT("--rank",      a->rank);
        ARG_INT("--steps",     a->steps);
        ARG_INT("--log",       a->log_every);
        ARG_INT("--ckpt",      a->ckpt_every);
        ARG_INT("--seed",      a->seed);
        ARG_INT("--max-seq",   a->max_seq);
        ARG_F32("--lr",        a->lr);
        ARG_F32("--alpha",     a->alpha);
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("usage: %s [--base path] [--data path] [--out path] [--rank N] [--steps N] [--lr F] [--alpha F] [--max-seq N]\n", argv[0]);
            return 1;
        }
        fprintf(stderr, "unknown arg: %s\n", argv[i]);
        return 1;
    }
    return 0;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args A;
    if (args_parse(argc, argv, &A)) return 1;
    srand((unsigned)A.seed);

    printf("=== DoE.coder LoRA SFT ===\n");
    printf("base: %s\ndata: %s\nout:  %s\n", A.base_path, A.data_path, A.out_path);
    printf("rank=%d alpha=%.1f lr=%.2e steps=%d max_seq=%d seed=%d\n\n",
           A.rank, A.alpha, A.lr, A.steps, A.max_seq, A.seed);

    double t0 = now_ms();

    gguf_file* gf = gguf_open(A.base_path);
    if (!gf) { fprintf(stderr, "gguf_open failed\n"); return 1; }

    nt_tape_start();
#ifdef USE_CUDA
    if (gpu_init() != 0) {
        fprintf(stderr, "[main] gpu_init failed — falling back to CPU\n");
    } else {
        nt_set_gpu_mode(1);
        fprintf(stderr, "[main] CUDA enabled (cublas + tf32)\n");
    }
#endif

    Config cfg = {0};
    BaseModel* bm = base_load(gf, &cfg);
    if (!bm) { gguf_close(gf); return 1; }

    LoRA* lr = lora_init(&cfg, A.rank, A.alpha);

    Dataset* ds = dataset_load(A.data_path);
    if (!ds) return 1;

    if ((int)ds->vocab_size > cfg.V) {
        fprintf(stderr, "vocab mismatch: dataset=%u > model=%d (fatal)\n", ds->vocab_size, cfg.V);
        return 1;
    }
    if ((int)ds->vocab_size != cfg.V) {
        printf("[init] vocab note: dataset=%u, model=%d (subset, OK)\n", ds->vocab_size, cfg.V);
    }

    // Snapshot tape count after init — everything below is "params + lora".
    // After each step, truncate tape to this count (drops forward intermediates).
    int n_init_entries = nt_tape_get()->count;
    printf("[init] elapsed %.0f ms, tape entries after init: %d\n\n", now_ms() - t0, n_init_entries);

    nt_schedule sched = nt_schedule_cosine(A.lr, /*warmup*/50, A.steps, A.lr * 0.1f);

    nt_train_mode(1);
    double train_t0 = now_ms();
    float ema_loss = 0;

    for (int step = 0; step < A.steps; step++) {
        uint32_t idx = (uint32_t)rand() % ds->n_examples;
        int* tokens; uint32_t lq, la;
        dataset_get(ds, idx, &tokens, &lq, &la);
        int T = (int)(lq + la);
        if (T > A.max_seq) {
            // truncate from the right — answer side
            la = (uint32_t)(A.max_seq - lq);
            T  = A.max_seq;
        }
        int T_input = T - 1;
        if (T_input < 4 || (int)la < 2) {
            // skip degenerate
            continue;
        }

        int logits_idx  = forward(bm, lr, &cfg, tokens, T_input);
        int targets_idx = make_targets(tokens, T_input);
        int mask_idx    = make_loss_mask((int)lq, T_input);
        int loss_idx    = nt_seq_cross_entropy_masked(logits_idx, targets_idx, mask_idx,
                                                       T_input, cfg.V);

        float loss = nt_tape_get()->entries[loss_idx].output->data[0];

        nt_tape_backward(loss_idx);
        float lr_now = nt_schedule_get_lr(&sched);
        nt_tape_chuck_step(lr_now, loss);

        ema_loss = (step == 0) ? loss : (0.95f * ema_loss + 0.05f * loss);

        if (step % A.log_every == 0) {
            double elapsed = (now_ms() - train_t0) / 1000.0;
            double tok_per_s = (step + 1) * (double)T_input / elapsed;
            printf("step %4d/%d | T=%3d lq=%u la=%u | loss %.4f (ema %.4f) | lr %.2e | %.0fs %.0f tok/s\n",
                   step, A.steps, T_input, lq, la, loss, ema_loss, lr_now, elapsed, tok_per_s);
            fflush(stdout);
        }
        if (step > 0 && step % A.ckpt_every == 0) {
            char ckpt[512];
            snprintf(ckpt, sizeof(ckpt), "%s.step%d", A.out_path, step);
            lora_save(lr, &cfg, ckpt);
        }

        // Drop intermediates, keep params
        tape_drop_temporaries(n_init_entries);
    }

    lora_save(lr, &cfg, A.out_path);
    printf("\n=== done in %.1f s ===\n", (now_ms() - train_t0) / 1000.0);

    dataset_free(ds);
    free(bm->layers); free(bm);
    free(lr->blocks); free(lr);
    nt_tape_destroy();
    gguf_close(gf);
    return 0;
}
