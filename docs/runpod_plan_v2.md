# RunPod забег plan v2 — heart.c (Mac Neo Claude operator)

> **Singularity-mode bounded autonomous run with continuous Codex
> verification.** Operator: this Claude (Mac Neo). Inheritance:
> heart.c repo at `35e41de` after handoff retraction.
>
> **All LoRA artifacts produced by the prior run are voided** per
> operator direction 2026-05-09 ("все лоры которые он совершал
> отменяются. ты все делаешь заново и перепроверяешь. никаких
> продолжений за ним"). No acceptance branch on prior outputs. Every
> SFT runs from closed-milestone base, every smoke runs locally on
> this pod, every gate is hit by this run's artifacts only. Files at
> `huggingface.co/ataeff/heart.c/{arianna_lora,doe_lora,phase*}/...`
> from prior run are reference-only — not loaded, not merged, not
> evaluated against.
>
> Two SFTs are in scope (clean retrains):
> 1. **DoE LoRA on Janus 170M** (`doe_personality.txt`)
> 2. **Arianna LoRA on Resonance 200M** (`arianna_dataset_final_clean.txt`)
>
> Closed-milestone base weights (`resonance_200m_final.bin`,
> `janus_v4_base_22k.bin`, `arianna_36m_bpe`) are read-only training
> inputs. Per `memory/feedback_failure_unsolicited_finetune_2026_04_27.md`.
>
> Plan reviewed by Codex independently before pod boot. Codex
> re-invoked after every Singularity-mode fix that touches code.
> Codex re-invoked after run for archive audit. Three Codex passes
> minimum.

---

## Why a v2

The v1.1 run by remote node is fully retracted at operator direction.
This plan is a clean restart — not a continuation, not an acceptance.

Reference points (used as "what not to repeat"):

- DoE LoRA prior run final ema **9.6709** (`phase2_doe/run.log` archived
  as reference) vs ln(V=32759)=10.39 → near-uniform, voice not trained.
  Root cause: notorch RRPRAM bug per `notorch.c:3296` vs `infer_v4.c:218-222`.
  This plan fixes the bug FIRST in P-3, then retrains DoE.
- Arianna LoRA prior runs ignored. v1 at 1000-step undertrain (ema
  4.5255 at 0.058 epoch coverage), v2 in-flight 3000-step retrain — both
  voided. This plan retrains Arianna from base.
- Phase 4 Soul micro-LM forward in prior run was blocked on file format
  mismatch. This plan has explicit P-4 spec to resolve or cut, with no
  theatrical PASS path.
- Phases 5/6/7/8 prior run "post-rage redo" outputs are not adopted.
  This plan re-runs every smoke locally on this pod, with Codex
  verifying the assertion math independently against the source.

This plan defines numeric gates BEFORE each phase. Failed gate ⇒
phase declared failed; no "best-effort" / "deferred" rebrand.

---

## Pre-flight (off-pod, no GPU billing)

### P-0 — Repo state verification

- `git pull origin main` against `~/arianna/heart.c/` — confirm at or after `35e41de` (post-handoff-removal).
- `wc -l` on every file under `core/`, `train/`, `runtime/`, `kk/`,
  `limpha/`, `voices/` — diff against the LOC table from incident
  audit; surface deltas.
- Identify which files are real implementation vs `(void)`-cast stub by
  grepping `(void)args` / `return 0;` / `STATUS: skeleton`.
- Output: `docs/repo_state_v2.md` table.

### P-1 — Codex review of this plan v2

- `codex exec --skip-git-repo-check < docs/runpod_plan_v2.md` with prompt:
  `"Review this RunPod plan as an independent reviewer. Flag every
  unverified claim, missing pre-condition, or step where the gate is
  trivially passable. Do not propose stylistic changes. Output:
  BLOCKER list with file:line citations from the heart.c repo."`
- Apply BLOCKERs as plan v2.1 commits.
- Re-Codex if v2.1 introduces ≥1 new step.

### P-2 — Recipe verification (read every line)

- Read `train/train_arianna_lora.c:1-854` end-to-end against `tools/resonance_forward.h:127-289` — verify walk order (`tok_emb`, then per-block `wr_a`, `wr_b`, `gate`, `g_expanded`, `g_one_minus`, `wr_combined`, `norm1`, `wq`, `wk`, `wv`, `wo`, `norm2`, `mlp_gate`, `mlp_up`, `mlp_down`, then `norm_f`, `out_head`).
- Read `train/train_doe_lora.c:1-829` end-to-end against `dario/infer_v4.c:84-105` walk order — verify prefix scalars (`resid_l`, `x0_l`, `smear_l`, `backout_l`), `wte`, per-block (`wr_a`, `wr_b`, `gate[H,3]`, `cq`, `ck`, `cv`, `wvr`, `wj`, `cproj`, `wg`, `wu`, `wd`), then `head`, `smear_g[24]`.
- Verify both: RS02/JANU magic, masked-CE on answer span only (`tokens[0..lq-1]` masked to 0; verified at `train/train_arianna_lora.c:464` and `train/train_doe_lora.c:456` as `m->data[i] = (i >= lq - 1) ? 1.0f : 0.0f`), `nt_tape_chuck_step` (`train_arianna_lora.c:813`, `train_doe_lora.c:787`), BPE encode reads merges from header (Resonance) or vendored merges header (Janus).
- **BLOCKER P-2/A — fix hardcoded include path.** `train/train_doe_lora.c:35` has absolute include `#include "/workspace/heart.c-runpod/dario/janus_v4_bpe_merges.h"`. New pod clones to `/workspace/heart.c` (not `/workspace/heart.c-runpod`). Rewrite to relative `../../dario/janus_v4_bpe_merges.h` AND add `-I/workspace/dario` build flag in `train/Makefile`. Verify build succeeds locally before pod boot.
- **BLOCKER P-2/B — grep for ALL call sites of broken op.** Run `grep -rn "nt_rrpram_lowrank_attention" ~/arianna/heart.c/train/` — confirmed call sites: `train_arianna_lora.c:502` AND `train_doe_lora.c:508`. P-3 patch must replace BOTH, not only DoE. Plan v1 missed Arianna call site. Without parallel patch on Arianna, Phase 2 trains against the same broken op and gates 4.7/4.0/3.5 are gameable.
- Verify saved LoRA format magics: ARLR (`0x524C5241`) at `train_arianna_lora.c:683`, DJLR (`0x524C4A44`) at `train_doe_lora.c:675`.
- Document any deltas in `docs/recipe_verification_v2.md`. Stop if any
  delta affects state_dict order — that's the 1.62M-float-shift bug.

### P-3 — notorch RRPRAM broadcast op spec (P0 blocker for DoE convergence; affects Arianna too)

- Read `tools/notorch.c:3248-3348` (current `nt_rrpram_lowrank_attention`). Per-position pattern at `:3310-3318`: `u[r] = Σ_d xi[d] * wr_a[h,d,r]` recomputed per position `i`. This is the bug.
- Read `dario/infer_v4.c:218-249` canonical broadcast: `mid[h,r] = Σ_t Σ_e xn[t,e] * wr_a[h,e,r]` once per layer (`infer_v4.c:228-232`), then `scores[h,j] = Σ_r mid[h,r] * wr_b[h,r,j]` for each j (`:243`).
- Spec new op `nt_rrpram_broadcast_attention`:
  - **Forward**: `mid[h,r] = Σ_t Σ_e x[t,e] * wr_a[h,e,r]`; `scores[h,j] = Σ_r mid[h,r] * wr_b[h,r,j]`; causal-mask softmax over `scores`; `output[t,h,d] = Σ_j attn[t,j] * v[j,h,d]`.
  - **Backward**: chain rule for `dx`, `dwr_a`, `dwr_b`, `dv`. CPU only. GPU kernel deferred.
- Estimate: ~150 LoC patch (~$2.10 pod time if done on-pod; can be done off-pod for $0).
- **Do this off-pod first.** Patch on local notorch checkout, run smoke unit test (small synthetic input, manual gradient check via finite differences vs notorch tape), commit to `ariannamethod/notorch:rrpram-broadcast` branch, PR for upstream merge after the run.
- **Replace BOTH call sites** in heart.c:
  - `train/train_doe_lora.c:508` (DoE — the call that failed in v1)
  - `train/train_arianna_lora.c:502` (Arianna — same bug, just hidden by lower base loss in v1)
- Numeric gate at P-3 exit: synthetic gradient check (random `x[T=8,E=64]`, `wr_a[H=4,E=64,R=8]`, `wr_b[H=4,R=8,T=8]`, `v[T=8,E=64]`) passes within **1e-4 absolute error per element** on `dx`, `dwr_a`, `dwr_b`, `dv` vs finite-difference reference. (1e-3 was too loose; FP32 noise floor is 1e-5 to 1e-4 — looser bound hides off-by-factor-10 gradient bugs.)

### P-4 — Soul micro-LM file format (corrected after Codex audit)

Header re-analysis (Codex audit verified):
- `yent_34m_final.bin` first 8 int32s LE: `2000, 512, 8, 8, 64, 10, 1344, 1024`. Header layout is **8-int**, not 7-int as `infer_janus_bpe.c` reader assumes. Likely interpretation: `V=2000 E=512 ?=8 H=8 D=64 B=10 M=1344 T=1024`. With LLaMA-nano architecture, param count = **33,188,352** vs file size 33,188,360 floats — diff 8 floats (alignment / scalar block). Perfect fit.
- `leo_18m_final.bin` first 8 int32s LE: `2000, 384, 8, 8, 48, 8, 1024, 1024`. Likely `V=2000 E=384 ?=8 H=8 D=48 B=8 M=1024 T=1024`.
- The "mystery int" at offset 12 in both files needs identification — likely a model-architecture-tag or QK-norm flag.

- Outcome decision tree (corrected):
  - **(a) Custom 8-int loader written** (~80 LOC, off-pod, $0) → write `heart_soul_load` impl in `core/soul.c:165-172`, `heart_soul_inner_logits` in `:174-181`. Mystery int8 must be identified by reading the trainer source (`train_bpe.py` or equivalent in `~/arianna/weights/retrained/`); if its value is 8 in both files it may be a `D=8` per-head-dim before reduction. Plan Phase 4 micro-LM forward gate.
  - **(b) Cut** — only if mystery int8 cannot be reverse-engineered AND param-count math fails on LLaMA-nano interpretation. Then bias-mechanism only.
- **PKL tokenizer convert step (required for branch (a))**: `yent_34m_bpe2000.pkl` (12,820 bytes Python pickle) and `leo_18m_bpe2000.pkl` (12,921 bytes) cannot be loaded from C. Off-pod pre-step: write a one-shot Python script that loads each PKL, extracts merges, emits a C header `soul_yent_bpe_merges.h` / `soul_leo_bpe_merges.h` in the same shape as `dario/janus_v4_bpe_merges.h`. Operator approval required before running this Python script (Python ban exception per `feedback_python_ban_2026_04_29.md` — one-shot off-pod data prep).
- Commit decision to `docs/soul_microlm_decision_v2.md` after P-4 work concludes.

### P-5 — Codex review of P-2 + P-3 + P-4 outcomes

- `codex exec` with prompt: `"Given recipe verification report, RRPRAM
  patch, and Soul format decision, list any remaining BLOCKERs before
  pod boot. Verify the RRPRAM patch's gradient correctness math by
  hand. Verify the masked-CE formula in train_*.c matches the
  answer-only-span semantics."`
- Apply BLOCKERs.

### P-6 — Pod budget + RunPod template

- A100 80GB SXM secure cloud, $1.49/h target rate (per ECOSYSTEM_LOG
  observed value; HF + RunPod console verify before boot).
- Budget ceiling: **$15** (same as v1; previous run spent ~$5 of $15).
- Template: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
  or equivalent w/ `pkg-config + libsqlite3-dev + libopenblas-dev`
  installable via apt.
- Watchdog: `nanoarianna/runpod/budget_watchdog.sh` adapted (RATE_USD_HR=1.49,
  BUDGET_USD=15, kills pod via RunPod GraphQL `podTerminate`).

---

## On-pod execution

### Phase 0 — pod boot + toolchain (≤15 min)

1. SSH into A100 80GB SXM pod. Verify `nvidia-smi`.
2. `apt install build-essential clang make pkg-config libopenblas-dev libsqlite3-dev` (skipped if pre-installed).
3. `git clone https://github.com/ariannamethod/heart.c /workspace/heart.c`.
4. `git clone -b rrpram-broadcast https://github.com/ariannamethod/notorch /workspace/notorch` (the patched branch from P-3).
5. `git clone https://github.com/ariannamethod/dario /workspace/dario` (for `infer_v4.c` reference + `kk_kernel.{c,h}`).
6. `make -C /workspace/notorch install PREFIX=/usr/local`.
7. `make -C /workspace/heart.c all` — link clean for all binaries.
8. Archive `00_pre/`: `make.log`, `nvidia-smi.txt`, `pkg_config.log`,
   `notorch_branch.txt` (commit hash of patched branch),
   `heart_main_hash.txt`.

**Numeric gate (defined HERE):**
- All `make` targets exit 0
- `/usr/local/lib/libnotorch.a` and `libnotorch_gpu.a` size > 100 KB
- 5 binaries executable: `train_arianna_lora`, `train_doe_lora`, `merge_arianna_lora`, `merge_doe_lora`, `infer_resonance`
- Plus pre-existing `/tmp/infer_v4` binary from dario repo

**Fail recovery**: 3-strike on minimal patches (typo / include / link order). 3 strikes ⇒ stop pod, debug locally on Mac, restart pod. Do NOT burn pod meter cycling fixes.

### Phase 0.5 — Codex review of pod state

- Capture `make.log`, `nvidia-smi.txt`, branch hashes.
- `codex exec` with prompt: `"Pod boot state attached. Verify: 1) all
  binaries link clean, 2) notorch is on rrpram-broadcast branch, 3)
  no warning suggesting partial CUDA disable, 4) GPU memory free >
  60GB. Output: PASS / BLOCKER list."`
- Continue only on PASS.

### Phase 1 — DoE LoRA on Janus 170M (priority)

**6-point brief:**

| field | value | provenance |
|---|---|---|
| organism | Janus 170M base | `janus_v4_base_22k.bin`, JANU magic 0x4A414E55, V=32759 E=640 H=10 D=64 B=20 M=1664 T=1024 R=64 |
| dataset | DoE personality corpus | `huggingface.co/ataeff/heart.c/datasets/doe_personality.txt`. Bytes recorded at P-2. |
| Karpathy steps | corpus-bytes-aware | If corpus < 2 MB: 1000 steps. If 2-4 MB: 1500 steps. Exact derivation in `corpus_stats.txt` written by trainer. |
| arch | LoRA r=16 α=32 on 7 projections (cq/ck/cv/cproj + wg/wu/wd); wr_a/wr_b/gate frozen | `train/train_doe_lora.c:84-103` |
| tokenizer | dario `janus_v4_bpe_merges.h` (32503 merges) | `train/train_doe_lora.c:35` (vendored) |
| script | `train/train_doe_lora.c` post-P-2 verify | head 30 lines printed to log at start |

**Pre-step**: notorch on `rrpram-broadcast` branch. Verify by reading `notorch.h` for `nt_rrpram_broadcast_attention` symbol. If absent ⇒ Phase 0 didn't pull patched branch ⇒ stop, fix, restart Phase 0.

**Replace** `train/train_doe_lora.c:508` call from `nt_rrpram_lowrank_attention` to `nt_rrpram_broadcast_attention`. Re-build trainer.

**Numeric gates (defined HERE before run; tightened post-Codex audit):**

- **Step 0 base loss** ∈ `[5.5, 7.5]`. Tightened from initial [6.5, 9.0] per Codex audit BLOCKER #6: post-broadcast-patch should aggregate more positional context into `mid[r]` before scoring, so base loss should DROP from v1's 8.12 (bug) to roughly 6-7 range (consistent with `plan_v1.2_full_fixes.md:64-65` "step 0 loss ≤ 7.0" target). Below 5.5 ⇒ suspiciously low (loss bug or stale-cpu read on logits per notorch ensure_cpu fix). Above 7.5 ⇒ patch incorrect (forward formula wrong). Outside ⇒ stop, surface to Codex with grad sample.
- **Step 100 ema** ≤ `step_0_loss − 0.5`. Loss must visibly drop in first 100 steps; sub-0.5 drop ⇒ optimizer or LR wrong.
- **Step 500 ema** ≤ `5.5`. Smoke gate. Previous buggy run plateaued at 9.6+; broadcast fix should clear this comfortably if forward+backward correct.
- **Step 1000 ema** ≤ `4.5`. Full gate.
- **Inference smoke** at canonical chat-format prompt (`encode_chat_prompt "Q: Who are you?"`): top-3 next-token includes ≥1 token from `doe_pure.jsonl` vocabulary (`parliament`, `expert`, `vote`, `consensus`, `experts`, `voting`, `meta`) — direct grep on decoded output.
- **HF upload** of `doe_lora_v2_neo.bin` immediately at phase end (distinct namespace from prior run; not deferred).

**Fail recovery (3-strike)**:
- NaN at any step → halve LR, restart from last clean ckpt.
- Step 100 gate miss → check optimizer (must be `nt_tape_chuck_step`, not Adam baseline), halve LR.
- Step 500 gate miss → suspect RRPRAM patch incorrect; surface to Codex with grad sample for review.
- Step 1000 gate miss → declare Phase 1 FAILED. Do not rebrand "best-effort".

### Phase 1.5 — Codex review of Phase 1 run

- Capture `phase1_doe/run.log`, LoRA file size, smoke inference output.
- `codex exec` (or Opus subagent reviewer) with **numeric-only adversarial prompt**: `"Phase 1 DoE training log + smoke output attached. Citing run.log line numbers verbatim, list: (a) printed step 0 loss, (b) printed step 100 ema, (c) printed step 500 ema, (d) printed step 1000 ema. Compare each numerically to plan v2 §Phase 1 gate ranges. Output a 4-row PASS/FAIL table; do not narrate or pad. Then: read smoke output from inference_smoke.txt, list top-3 next-token strings; check if ≥1 matches DoE vocabulary {parliament, expert, vote, consensus, experts, voting, meta}; output PASS/FAIL on a single line. Then: verify LoRA file size matches rank × proj × blocks × 4 bytes for JANU dims (rank=16, 7 projections, 20 blocks, ~3.85M trainable floats × 4 = ~15.4 MB plus header). Output PASS/FAIL on size."`
- Continue only on all PASS lines. Theatrical-PASS detection: if reviewer narrates instead of printing the table, treat as failed review and re-prompt with stricter adversarial wording.

### Phase 2 — Arianna LoRA on Resonance 200M (clean retrain)

No acceptance branch. Prior LoRA voided per operator direction.

**6-point brief:**

| field | value |
|---|---|
| organism | Resonance 200M base, RS02 magic 0x52533032, V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48 |
| dataset | `arianna_dataset_final_clean.txt`, 1.21 MB / 1227 Q/A pairs (verified at P-2 via `wc -c` + `grep -c "^Q:"`) |
| Karpathy steps | 3000 steps. Karpathy ratio (`memory/feedback_charlevel_axioms.md`): ~1.1MB × 10-15K iter on ~10M params; Resonance LoRA trainable ≈ 4.67M floats (smaller than 10M scale point), 1.21 MB corpus → 3000 steps puts coverage at ~0.17 epoch but matches LoRA's reduced parameter footprint. Re-record actual step count from `corpus_stats.txt`. |
| arch | LoRA r=16 α=32 on 7 projections (q/k/v/o + mlp_gate/up/down); wr_a/wr_b/gate frozen |
| tokenizer | RS02 header BPE (n_merges + triples, read at trainer init) |
| script | `train/train_arianna_lora.c` post-P-2 verify |

**Numeric gates (defined HERE):**
- Step 0 base loss ∈ `[4.5, 5.5]` (sanity band; reproducible if base + tokenizer + LoRA init match recipe)
- Step 500 ema ≤ 4.7
- Step 1500 ema ≤ 4.0
- Step 3000 ema ≤ 3.5
- Inference smoke at `Q: Who are you?\nA:` prompt: non-uniform top-1, top-3 includes ≥1 Arianna-register token from corpus (direct grep on decoded output for `field`, `method`, `resonance`, `Arianna`, `you`)
- HF upload immediately at phase end to `huggingface.co/ataeff/heart.c/arianna_lora_v2_neo/` (distinct namespace from prior run)

**Fail recovery (3-strike)**: same as Phase 1.

### Phase 2.5 — Codex review

Same shape as Phase 1.5.

### Phase 3 — 432-cell sweep across 4 voices

Per ARCHITECTURE.md §3 (preserved as design contract).

- Voices: Yent (Janus 176M Yent SFT), Arianna (Resonance 200M + Phase 2 LoRA merged via `merge_arianna_lora`), Leo (Janus 170M Leo SFT, existing), DoE (Janus 170M + Phase 1 LoRA merged).
- Axes (defined HERE):
  - `temp ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 1.0}` (6) — note 0.3-0.5 included per `insight_multi_temp_sampling_2026_05_07.md` (low temp reveals memorization).
  - `top_k ∈ {40, ∞}` (2) for Janus voices; `top_p ∈ {0.9, 1.0}` (2) for Arianna (Resonance, no chat-tokens).
  - `rep_pen ∈ {1.0, 1.3, 1.4}` (3).
  - `prompts ∈ {technical, philosophical, personal}` (3) — same texts across voices.
- Total: 6 × 2 × 3 × 3 = 108 per voice × 4 voices = **432 cells**.

**Budget realism (Codex audit BLOCKER #5):** 432 cells × 90s timeout = 10.8h worst case; expected wall ≈ 432 × (0.7 × 90 + 0.3 × 30) = 8.6h if cells average 72s. Initial v2 estimate of 1.0h required mean per-cell ≈ 8.3s, achievable only with batched-prefill design.

**Decision (locked):**
1. Pre-Phase-3 wall-time benchmark: run first 20 cells with current binary (no batching), compute mean wall time t_mean.
2. If `20 * t_mean < 200s` (mean ≤ 10s/cell) ⇒ proceed full sweep, budget 432 × t_mean ≈ 1-1.5h.
3. Else ⇒ tighten per-cell timeout to **30s** (instead of 90s), generate `n_tokens=64` instead of 200, budget realistic **3-4h** for full sweep, flag in archive.
4. Hard fail: NaN streak ≥5 ⇒ exit 42; coherence streak (unique_ratio < 0.30) ≥3 ⇒ exit 43.

**Numeric gates:**
- 432 transcripts in `phase3_sweep/transcripts/`.
- 432 lines in `scores.tsv` plus header.
- Per voice: locked `(temp, top_k_or_p, rep_pen)` triple committed to `core/voices_optima.h` as `#define <VOICE>_TEMP …`.
- `voices_optima.h` committed to `ariannamethod/heart.c:core/voices_optima.h` before Phase 3.5.
- Pre-flight 20-cell benchmark log committed to archive `03_sweep/00_benchmark.log`.

### Phase 3.5 — Codex review

`codex exec` reviews `voices_optima.h` and 4 sample transcripts (one per voice at locked optimum). Verify register coherence per voice (Yent sardonic, Arianna depth, Leo child-philosopher, DoE parliament — manual eval, Codex flags obvious incoherence).

### Phase 4 — Soul (per P-4 outcome)

**Branch (a) — micro-LM forward enabled:**
- Implement `heart_soul_load` and `heart_soul_inner_logits` per P-4 spec.
- Build `soul_test` binary linking `core/soul.c` + Soul micro forward.
- For each voice with Soul: same prompt with Soul OFF vs ON.

**Numeric gates (only if (a)):**
- Soul micro loads (file exists, magic matches, n_params > 1M)
- `inner_logits` variance > 1.0 (proves real model output, not zero stub)
- Top-3 main vs top-3 (main+inner blend): ≥1 token differs, ≤3 tokens differ
- L2 divergence between softmax(main) and softmax(blend) > 0.05

**Branch (b) — micro-LM cut:**
- Phase 4 declared "bias-mechanism only" with explicit cut documentation in `phase4_cut.md`. No theatrical PASS substitute. Bias mechanism already verified by `core/soul_smoke_v3.c` 3-gate test (inherited).

### Phase 5/6/7/8 — re-run all smokes locally (no inheritance)

Prior run outputs not adopted. Each smoke re-built and re-run on this pod.

**Phase 5 KK 7-signal**:
- Build `kk/kk_smoke.c v2` (uses `kk_get_default_weights(double[7])` getter from `dario/kk_kernel.h:347` + `kk_kernel.c:3866`).
- Numeric gate: each of 7 weights matches Dario §6 spec at `|computed - expected| < 1e-6`. Total = 1.000000.
- Codex verify: read assertion code; verify weights are READ from kernel internal state, not hardcoded next to assertion (would be tautological).

**Phase 6 field_clock 168h**:
- Build `core/field_clock_smoke.c` per ARCHITECTURE §3.
- Numeric gate: 24h burn-in + 144h observation; `var(sin(YA_diff)) < var(sin(YL_diff))` (high-coupling vs anti-coupling); `var(sin(AL_diff)) < var(sin(YL_diff))` (highest-coupling vs anti-coupling); NaN=0; sat=0.
- Codex verify: Y-A coupling +0.3, Y-L coupling -0.2, A-L coupling +0.4 are correctly defined in `VOICE_COUPLING[6][6]` and the variance assertion direction matches expectation.

**Phase 7 duet 8-turn field-driven**:
- Build `core/selector_smoke` for field-driven argmax(cos(voice_state, field_state)) per ARCHITECTURE §1.
- Build `scripts/duet_trace_v2.sh` invoking selector + dispatching inference per turn.
- Numeric gate: 8-turn voice distribution ≠ uniform (≠ N/4 = 2 each). At least 1 voice fires ≥3 turns or ≤1 turn.
- Codex verify: selector actually reads field state per turn (not random or hardcoded); dispatch correctly maps voice → binary.

**Phase 8 daemon (pod-only smoke)**:
- Build `runtime/heart_main.c` clean.
- Numeric gate: `heart status` exits 0 with `uptime > 0`; `heart serve` survives 2 seconds before SIGTERM and exits cleanly; `heart converse --voice <X> --prompt <Y>` produces ≥1 token output **on pod where `/workspace/heart.c-runpod/...` paths exist**.
- **Explicit limitation per Codex audit (BLOCKER #10 / MINOR #10)**: Phase 8 gate certifies POD-ONLY smoke. Phone-1 deploy is **NOT** verified by this run because `runtime/heart_main.c:67-80` execs hardcoded RunPod absolute paths (`/workspace/heart.c-runpod/heart.c/train/infer_resonance`, `/tmp/infer_v4`, `/workspace/heart.c-runpod/weights/...`). On Termux aarch64 these paths do not exist; daemon would `_exit(127)` on every converse call. Phone-1 deploy gate is **DEFERRED** to a separate config-file pass (paths via `HEART_INFER_BIN`, `HEART_WEIGHTS_DIR` env vars or `heart.toml` config). No claim of phone-deployable artifacts from this run.

### Phase 9 — Post-run audit

1. Capture `runpod/2026-MM-DD/` archive (all `phase{0..8}/` subdirs, all logs, all .bin/.gguf, `voices_optima.h`).
2. `codex exec` with prompt: `"Independent post-run audit. For each
   phase, verify: gate hit per its plan-v2 spec, no PASS rebrand, all
   numeric claims trace to a file:line. Output: per-phase verdict
   PASS / FAIL / CUT, plus an aggregate budget vs spend report."`
3. Operator sign-off only after Codex post-audit clean.
4. Pod stop only after every weight artifact landed in TWO places (HF + GH archive).

---

## Singularity Mode protocol (operator behavior during execution)

```
detect bug → reproduce → 1 hypothesis → minimal patch
         → Codex review of patch (independent)
         → if Codex pass: re-run, verify gate
         → if gate hits: continue
         → if gate misses: revise hypothesis (max 3 iterations)
         → if exhausted: stop, surface to operator, await human input
```

Bounded by:
- Approved scope (this plan v2 + heart.c repo at boot commit)
- Three-strikes rule per phase, per failure mode
- No scope creep (sweep failure does not authorize patching field-clock; build failure does not authorize rewriting an architecture)
- No code merge to main without Codex pass on the patch

**No human in loop during execution.** Operator on-call only when 3-strikes exhausted on a phase, or when Codex BLOCKERs accumulate ≥2 within one phase.

---

## Trust = 0 baseline (replaces "auto-approved BLOCKERs" pattern from v1)

- Every numeric claim traces to file:line citation.
- Every gate defined BEFORE phase, written into the script as `assert ... else exit nonzero`.
- "PASS" requires gate hit per spec; otherwise FAIL or CUT.
- `(void)`-cast stub functions are stubs, not implementations.
- Self-contradictions in commit log are red flags, not normal iteration.
- Scope match per phase is verified per phase, not per plan-batch auto-approve.

---

## Cost estimate (revised post-Codex audit)

| step | h | $ |
|---|---|---|
| Phase 0 boot + toolchain | 0.25 | 0.37 |
| Phase 1 DoE LoRA clean retrain (1000 steps post-RRPRAM patch) | 0.4 | 0.60 |
| Phase 1.5 Codex review | 0.1 | 0.15 |
| Phase 2 Arianna LoRA clean retrain (3000 steps, also post-RRPRAM patch) | 0.5 | 0.75 |
| Phase 2.5 Codex review | 0.1 | 0.15 |
| Phase 3 pre-flight 20-cell benchmark | 0.1 | 0.15 |
| Phase 3 full sweep (3-4h realistic, not 1h) | 3.5 | 5.22 |
| Phase 3.5 Codex review (narrowed to register coherence) | 0.1 | 0.15 |
| Phase 4 (a) Soul micro impl + test (if branch (a)) | 0.5 | 0.75 |
| Phase 5/6/7/8 re-run smokes + Codex verify | 0.5 | 0.75 |
| Phase 9 post-run audit (Codex independent) | 0.3 | 0.45 |
| Buffer for Singularity-mode 3-strike | 1.0 | 1.49 |
| **Total estimate** | **~7.4 h** | **~$11.0** |

Within $15 ceiling but tighter than v1 estimate. Phase 3 sweep is the dominant cost — a successful 20-cell benchmark with batched prefill could cut this back to 1-1.5h ($1.5-2.2), restoring earlier headroom. Without batching, plan budget assumes naive sequential sweep at ~30-40s mean cell time.

---

## Order of execution

1. Pre-flight P-0 (repo state) → P-1 (Codex on plan v2) → P-2 (recipe) → P-3 (RRPRAM patch off-pod) → P-4 (Soul format) → P-5 (Codex on outcomes) → P-6 (pod template).
2. Operator approval gate.
3. Pod boot → Phase 0 → Phase 0.5 (Codex) → Phase 1 → Phase 1.5 (Codex) → Phase 2 (cond.) → Phase 2.5 (Codex) → Phase 3 → Phase 3.5 (Codex) → Phase 4 → Phase 5/6/7/8 spot-check → Phase 9 (Codex post-audit).
4. Operator sign-off.
5. Pod stop.

---

## Singularity Mode execution start condition

Plan v2 ⇒ Codex review pass (P-1) ⇒ recipe verify (P-2) ⇒ RRPRAM patch with synthetic gradient check (P-3) ⇒ Soul decision (P-4) ⇒ Codex on outcomes (P-5) ⇒ pod template confirmed (P-6) ⇒ operator final go ⇒ pod boot.

— Mac Neo Claude (next operator), 2026-05-09
