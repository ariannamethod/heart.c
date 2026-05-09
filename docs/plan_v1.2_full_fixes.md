# heart.c plan v1.2 — full fixes after Opus audit (2026-05-09)

> Audit by Opus subagent against ARCHITECTURE.md v1.1 + runpod_plan_v1.md
> v1.1 + singularity_protocol.md surfaced 7 phases done sloppily. This
> document lists every fix concretely with numeric verification gates,
> ordered by dependency. **Each step has a numeric pass/fail that cannot
> be theatrical.**

---

## Phase 1 — Arianna LoRA on Resonance 200M

**Done**: 1000 steps Chuck, ema 4.94 → 4.5255 (`runpod/.../phase1_arianna/run2.log`).
Coverage: 1227 Q/A pairs × 1000 steps × ~70 tok/step = 70K tokens / 1.2M
corpus = 0.058 epoch (`hf-heart-assets/arianna_dataset_final_clean.txt:size`).

**Halyavно**: Karpathy ratio (CLAUDE.md) says ~1.1MB × 10-15K iter on
~10M params; we're at 1.21MB × 4.67M trainable LoRA params, so 3000-4000
steps minimum for proper character coverage.

**Fix**:
1. Retrain 3000 steps (already queued by `scripts/queue_arianna_retrain.sh`
   via PID 4589 — kill if Oleg confirms DoE kill, then run independently).
2. Re-merge `resonance_arianna_merged_v2.bin`.
3. Sweep at temp ∈ {0.9, 1.0, 1.1} × top_p ∈ {0.9} × 4 prompts per
   `feedback_temp_sweep_before_judging_2026_05_07.md`.

**Verification gate (numeric)**:
- ema at step 3000 ≤ 4.0 (vs 4.5255 at step 1000); compute Δ
- For each of 4 prompts: top-1 token at temp=1.0 must reproduce a token
  appearing in Arianna corpus (`arianna_dataset_final_clean.txt`) at any
  position — measured by direct grep on decoded output

---

## Phase 2 — DoE LoRA on Janus v4 177M

**Done**: 1000 steps ema 9.6709 (`phase2_doe/run.log`); retrain 5000 steps
in progress at step 850 ema 9.89 (`phase2_doe_v2/run.log`) — bouncing
9.7-10.0, no convergence trend.

**Halyavно**: structural notorch bug `nt_rrpram_lowrank_attention`
computes per-position `u[r,i]` (`notorch.c:3296`) vs canonical Janus
broadcast `mid[r] = sum_t sum_e xn[t,e]*wr_a[h,e,r]` once per layer
(`infer_v4.c:218-222`). LoRA on cq/ck/cv cannot recover structural
function-class diff. Subagent verdict: 5000-step retrain will plateau
≥9.4 — pod time wasted.

**Fix**:
1. **Need Oleg confirm**: kill DoE PID 4495 retrain (no convergence path).
2. Patch notorch with `nt_rrpram_broadcast_attention` op:
   - new NT_OP_RRPRAM_BROADCAST opcode in notorch.h
   - forward: compute mid[h,r] = sum_t sum_e x[t,e]*wr_a[h,e,r] once per
     layer; scores[h,j] = sum_r mid[h,r]*wr_b[h,r,j]; softmax causal
     per-position; output[t,h,d] = sum_j attn[t,j]*v[j,h,d]
   - backward: dx, dwr_a, dwr_b, dv via chain rule
   - CPU implementation (~150 LoC); GPU kernel deferred (CPU-only fallback)
3. Replace `nt_rrpram_lowrank_attention` call in `train_doe_lora.c:357`
   with new op
4. Retrain DoE 1000 steps post-patch (not 5000 — broadcast op should
   converge faster)

**Verification gate (numeric)**:
- Pre-LoRA base forward with patched op: step 0 loss ≤ 7.0 (vs 8.12 with
  per-position op — broadcast adds context info → lower base loss)
- Post-LoRA training: ema after 500 steps ≤ 6.5; after 1000 ≤ 5.5
- Inference: top-1 next-token at chat-format prompt "Who are you?" includes
  vocabulary from `doe_pure.jsonl` (e.g. "parliament", "expert", "vote",
  "consensus") — direct grep verification

---

## Phase 3 — voice sweep

**Done**: 36 cells across 4 sub-sweeps (raw / high-temp / chat-fmt /
chat-fmt+BOS), `phase3_sweep/results_v{1..4}*.txt` on HF.

**Halyavно**: 36/432 = 8.3% of plan §3 spec. `sweep_full.sh` axes mismatch
— produces 216 not 432. `voices_optima.h` not committed.

**Fix**: amend per Oleg's temp-policy gate:
- temps ∈ {0.9, 1.0, 1.1, 1.2} (4)
- top_k ∈ {40, 128} (2) — `128` ≈ "no filter" for these vocabs
- rep_penalty ∈ {1.0, 1.2, 1.4} (3)
- prompts: 4 (`Who are you?` / `What is resonance?` / `What do you remember?` / `Where are we now?`)
- voices: 4
= 4 × 2 × 3 × 4 × 4 = **384 cells** total (96 per voice). Full plan §3
was 432 — 48 less, justified by Oleg's "ниже 0.9 не пробовать" gate.

Run `sweep_full.sh` (axes corrected) post-Phase 1+2 retrain. Output:
- raw `phase3_sweep/full.tsv`
- per-voice argmax of "lowest character entropy" → `docs/per_voice_optima_locked.md`
- `voices_optima.h` with `#define YENT_TEMP 1.0` etc.

**Verification gate (numeric)**:
- 384 cells written to TSV (line count = 385 with header)
- Per voice: locked (temp, top_k, rep_pen) triple selected by maximizing
  count of in-corpus token matches across 4 prompts
- `voices_optima.h` committed and `#include`-able in heart_main.c

---

## Phase 4 — Soul logit injection

**Done**: `core/soul_smoke.c` 50/50 argmax differ "PASS" — but
construction makes this trivially passable (random main + nonzero blend
weight). `heart_soul_load` returns -1; `heart_soul_inner_logits` zeros.
Plan §6.6 demanded actual Soul micro forward. ECOSYSTEM_LOG self-contradicts:
"Phase 4 Soul — not run this session" (commit `ce7da99`) vs HF
`phase4_smoke/soul.txt` PASS (commit `ddc5e81`).

**Halyavно**: theatrical pass.

**Fix**:
1. Implement `heart_soul_load(path)` — mmap a Janus BPE d12 .bin
   (`yent_34m_final.bin` 127M / `leo_18m_final.bin` 60M from
   `hf-heart-assets/`). Format known: `infer_janus_bpe.c` precedent.
   Reuse its loader as static helper.
2. Implement `heart_soul_inner_logits(iv, context_tokens, n, out, vocab)`:
   - run actual forward pass through Soul micro on context_tokens
   - write next-token logits to `out`
3. Smoke v2:
   - Pick a real BPE-encoded prompt from Arianna corpus (first 16 tokens
     of first Q/A pair)
   - Run main voice forward (Resonance + Arianna LoRA) → `main_logits`
   - Run Soul forward (yent_34m_final.bin loaded as Yent's Soul micro
     since its corpus is Yent-style) → `inner_logits`
   - `heart_inner_borba(...)` blend
   - Compute: top-3 tokens of main vs out — at least 1 must differ AND
     out top-1 must NOT be uniform/random
4. Stats:
   - main entropy
   - inner entropy
   - L2 divergence
   - top-3 main tokens vs top-3 out tokens (intersection size)

**Verification gate (numeric)**:
- Soul micro loads (file exists, magic matches, n_params > 1M)
- inner_logits has variance > 1.0 (i.e. real model output not zero stub)
- top-3 tokens of main vs out: ≥1 token differs, ≤3 token differs
  (cohérent shift — not random)
- L2 divergence between softmax(main) and softmax(out) > 0.05 (real
  effect, not floating-point noise)

---

## Phase 5 — KK ingest + 7-signal scoring

**Done**: `kk/kk_smoke.c` ingests 5 docs, runs 4 queries, prints 7-signal
breakdown. Output `phase5_smoke/kk.txt`.

**Halyavно**: plan §6.5 demanded "scoring policy match Dario §6 spec to
6 decimal places" — `kk_smoke.c:71` prints `%.4f`, no assertion code.
Hebbian bridge integration not tested.

**Fix**:
1. In `kk_smoke.c` add explicit policy verification:
   ```c
   const struct { const char* name; double expected; } weights[] = {
       {"lexical",   0.36}, {"recency", 0.12}, {"trust",    0.10},
       {"linkage",   0.16}, {"scope",   0.10}, {"namespace", 0.08},
       {"freshness", 0.08},
   };
   /* Read computed weights from kk_kernel internal config (export
    * via getter or just re-derive from kk_chunk_resonance call) and
    * assert |computed - expected| < 1e-6 for each of 7. */
   ```
2. Add Hebbian bridge mock — implement minimal `word_resonance`,
   `get_prophecies`, `destiny_magnitude` callbacks. Attach via
   `kk_set_hebbian_bridge`. Run same query with bridge ON vs OFF —
   `hebbian_boost` field must differ.
3. Print explicit `assert: signal=lexical computed=0.360000 expected=0.360000 PASS`
   per signal.

**Verification gate (numeric)**:
- 7/7 assertion lines printed PASS, all `|computed - expected| < 1e-6`
- Bridge ON vs OFF: at least 1 query result has `hebbian_boost` field
  numerically different (> 1e-9)

---

## Phase 6 — field_clock 24h sim

**Done**: 24h sim NaN=0 sat=0 R 0.0009→0.0233 PASS
(`phase6_smoke/field_clock.txt`).

**Halyavно**: plan §6.4 + ARCHITECTURE.md §3 v1.1 also demanded
verification of VOICE_COUPLING phase-lock relationships (Yent↔Arianna
+0.3, Yent↔Leo −0.2). Not asserted. R range 0.0009→0.0233 is suspicious
— oscillators essentially free-running, no coupling visible.

**Fix**: extend `field_clock_smoke.c` to:
1. Track per-primary mean phase per hour
2. Compute phase difference between primary pairs (Y-A, Y-L, A-L, Y-D, etc.)
3. Compute variance of phase difference over the 24h window
4. Assert: variance(Y-A diff) < variance(Y-L diff) — high coupling pair
   should be more locked than anti-coupling pair

**Verification gate (numeric)**:
- variance(Y-A phase diff) < variance(Y-L phase diff) over 24h
- Both finite, no NaN
- Asserted explicitly with PASS/FAIL print

---

## Phase 7 — duet trace + DoE meta-Parliament

**Done**: `scripts/duet_trace.sh` round-robin × 8 turns;
`core/selector_smoke.c` field-driven smoke separate, NOT integrated.
DoE embedding-vote (plan §2.4 v1.1 — entire BLOCKER-resolution rewrite)
NOT implemented anywhere.

**Halyavно**: half of plan §1 + §2.4 not delivered. The four-voice
ecology speaks via round-robin only.

**Fix**:
1. Write `scripts/duet_trace_v2.sh` that:
   - Each turn: read field state (call /tmp/selector_smoke once,
     parse output)
   - Pick voice via argmax(cos)
   - Generate 1 sentence with that voice
   - Stat: log selector decision per turn
2. Implement `core/parliament_vote.c` (DoE meta-Parliament minimal):
   - Each turn: pick K=3 of 4 peer voices, generate 1 candidate sentence each
   - For each candidate: SPA-embed (lift from `yent.aml/jannus-r/tools/jannus_spa.h`)
   - DoE meta vote on embeddings: random-init weights × embedding → log score, pick argmax
   - Selected candidate persists; others discarded
3. Wire into duet_trace_v2.sh

**Verification gate (numeric)**:
- selector log shows non-uniform voice distribution (NOT N/4 each — not
  exactly round-robin)
- meta-vote disagrees with naive selector argmax in ≥1 of 8 turns
- All 8 turns produce non-empty output (token count > 5 each)

---

## Phase 8 — daemon + boot

**Done**: skeleton `runtime/heart_main.c` + 14-line `boot/heart-daemon.sh`
referencing nonexistent `~/.local/bin/heart`.

**Halyavно**: not built, not tested.

**Fix**:
1. Build `runtime/heart_main.c` minimal: parse argv `serve` / `converse`
   / `status`, link voice TUs + selector + soul + field_clock + KK
   modules. Output: `~/.local/bin/heart`.
2. Test boot watchdog: kill heart, confirm respawn within 5s
3. Test mesh integration: `mesh-agent exec heart-status` returns uptime

**Verification gate (numeric)**:
- `heart status` returns 0, prints uptime > 0
- After `pkill heart`, `pgrep heart` returns within 5 seconds (PID > 0)
- `mesh-agent exec heart-converse` returns transcript

**Cut**: defer to phone-1 deploy session (not on RunPod). Phone-1 is the
real boot environment. Pod can verify build cleanness only.

**Cut Verification gate**: `make heart` builds clean (binary exists +
`./heart status` exits 0).

---

## Cost estimate

| step | hours | cost |
|---|---|---|
| Patch notorch `nt_rrpram_broadcast_attention` | 1.5 | $2.10 |
| Phase 2 DoE re-train post-RRPRAM (1000 steps) | 0.4 | $0.55 |
| Phase 1 Arianna 3000-step retrain | 0.3 | $0.42 |
| Phase 3 384-cell sweep | 0.8 | $1.11 |
| Phase 4 Soul real impl (load + forward + smoke) | 1.5 | $2.10 |
| Phase 5 KK 6-decimal + Hebbian bridge | 0.5 | $0.70 |
| Phase 6 field_clock phase-lock assertion | 0.3 | $0.42 |
| Phase 7 duet field-driven + meta-vote | 1.5 | $2.10 |
| Phase 8 build heart daemon (cut: pod-build only) | 0.3 | $0.42 |
| **Total** | **~7.1h** | **~$9.92** |

Pod budget remaining: ~$7. **Over by ~$3.** Need cuts:
- Phase 7 meta-vote reduced to selector + simple K=3 random pick (no SPA
  embedding) — saves 1h
- Phase 8 already cut to build-only

After cuts: ~6h, ~$8.4. Within budget if Oleg willing to spend $1.4
extra OR drop one phase entirely.

---

## Order of execution

1. **Need Oleg confirm**: kill DoE 5000-step retrain PID 4495 (Singularity
   Mode "no convergence" stop, save $0.58 + 25 min)
2. Notorch RRPRAM broadcast patch (independent of retrains)
3. Phase 1 Arianna 3000-step retrain (parallel-safe in background once
   DoE is killed) — **but pod has 1 GPU**, so really sequential
4. Phase 4 Soul real impl (CPU side, can interleave with GPU training)
5. Phase 5 KK 6-decimal + Hebbian (CPU side)
6. Phase 6 field_clock phase-lock (CPU side)
7. Phase 2 DoE re-train post-RRPRAM (depends on 2)
8. Phase 7 duet field-driven + meta-vote (depends on Phase 1+2 retrains
   + selector working)
9. Phase 3 full sweep 384 cells (depends on Phase 1+2 retrain)
10. Phase 8 build heart daemon (independent)

— Defender, plan v1.2 (post-Opus audit), 2026-05-09
