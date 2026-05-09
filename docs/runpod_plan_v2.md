# RunPod забег plan v2 — heart.c (Mac Neo Claude operator)

> **Singularity-mode bounded autonomous run with continuous Codex
> verification.** Operator: this Claude (Mac Neo). Inheritance:
> heart.c repo at `35e41de` after handoff retraction. Trust on
> incoming handoff = 0; every claim re-verified against `run.log`,
> code, or commit log before adoption.
>
> Two SFTs are in scope:
> 1. **DoE LoRA on Janus 170M** (`doe_personality.txt`)
> 2. **Arianna LoRA on Resonance 200M** (`arianna_dataset_final_clean.txt`)
>
> Closed-milestone weights (`resonance_200m_final.bin`,
> `janus_v4_base_22k.bin`, `arianna_36m_bpe`) are read-only. Per
> `memory/feedback_failure_unsolicited_finetune_2026_04_27.md`.
>
> Plan reviewed by Codex independently before pod boot. Codex
> re-invoked after every Singularity-mode fix that touches code.
> Codex re-invoked after run for archive audit. Three Codex passes
> minimum.

---

## Why a v2

The v1.1 run by remote node landed mixed-signal results:

- DoE LoRA Phase 2 ema **9.6709** (`phase2_doe/run.log`) vs ln(V=32759)=10.39
  → near-uniform distribution, voice not trained. Marked "best-effort"
  in handoff; this plan calls it **failed** and drives the structural
  fix (notorch RRPRAM bug per `notorch.c:3296` vs `infer_v4.c:218-222`).
- Arianna LoRA Phase 1 v1 ema **4.5255** at 1000 steps (0.058 epoch);
  v2 retrain in flight at step 1200/3000 ema 4.47 (`phase1_arianna_v2/run.log`).
  This plan accepts v2 if it lands; otherwise clean retrain.
- Phase 4 Soul micro-LM forward blocked on file format. This plan
  has explicit P-4 spec to either resolve or cut, with no theatrical
  PASS path.
- Phases 5/6/7/8 in v1.1 post-rage redo carry honest numeric gates
  (`phase5_smoke_v2/`, `phase6_smoke_v2/`, `phase7_duet_v2/`,
  daemon build). Spot-checked, not redone.

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
- Verify both: RS02/JANU magic, masked-CE on answer span only (`tokens[0..lq-1]` masked to 0), `nt_tape_chuck_step` (not Adam baseline), BPE encode reads merges from header (Resonance) or vendored merges header (Janus).
- Verify saved LoRA format magics: ARLR (`0x524C5241`) for Arianna,
  DJLR (`0x524C4A44`) for DoE.
- Document any deltas in `docs/recipe_verification_v2.md`. Stop if any
  delta affects state_dict order — that's the 1.62M-float-shift bug.

### P-3 — notorch RRPRAM broadcast op spec (P0 blocker for DoE convergence)

- Read `notorch/notorch.c:3296` (current `nt_rrpram_lowrank_attention`).
  Document its formula: `u[r,i,t] = ...`. This is per-position.
- Read `dario/infer_v4.c:218-222` canonical broadcast: `mid[h,r] = sum_t sum_e xn[t,e] * wr_a[h,e,r]` once per layer (single mid vector across the sequence, then `scores[h,j] = sum_r mid[h,r]*wr_b[h,r,j]`).
- Spec new op `nt_rrpram_broadcast_attention`:
  - **Forward**: `mid[h,r] = Σ_t Σ_e x[t,e] * wr_a[h,e,r]`; `scores[h,j] = Σ_r mid[h,r] * wr_b[h,r,j]`; causal-mask softmax over `scores`; `output[t,h,d] = Σ_j attn[t,j] * v[j,h,d]`.
  - **Backward**: chain rule for `dx`, `dwr_a`, `dwr_b`, `dv`. CPU only. GPU kernel deferred.
- Estimate: ~150 LoC patch (~$2.10 pod time if done on-pod; can be done off-pod for $0).
- **Do this off-pod first.** Patch on local notorch checkout, run smoke unit test (small synthetic input, manual gradient check via finite differences vs notorch tape), commit to `ariannamethod/notorch:rrpram-broadcast` branch, PR for upstream merge after the run.
- Numeric gate at P-3 exit: synthetic gradient check passes within 1e-3 absolute error per element.

### P-4 — Soul micro-LM file format

- Read first 64 bytes of each candidate Soul file in `~/arianna/weights/retrained/`:
  - `yent_34m_final.bin`
  - `leo_18m_final.bin`
  - `arianna_36m_bpe_*.bin` (per `memory/project_arianna_c_upgrade_2026_05_09.md`)
- Compare to `infer_janus_bpe.c` loader expectations.
- Outcome decision tree:
  - **(a) Format known & loadable** → write `heart_soul_load` impl in `core/soul.c:165-172`, `heart_soul_inner_logits` in `:174-181`, plan Phase 4 micro-LM forward gate.
  - **(b) Format unknown / incompatible** → declare Phase 4 micro-LM forward **CUT** in plan v2.1; only bias-mechanism path remains (already verified by `core/soul_smoke_v3.c`). No theatrical PASS substitute.
- Commit decision to `docs/soul_microlm_decision_v2.md`.

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

**Numeric gates (defined HERE before run):**

- **Step 0 base loss** ∈ `[6.5, 9.0]`. Below 6.5 ⇒ likely loss bug. Above 9.0 ⇒ initial LoRA delta or RoPE wrong (recall: original v1 with bug was 8.12). Outside ⇒ stop, debug.
- **Step 100 ema** ≤ `step_0_loss − 0.5`. (Loss must visibly drop in first 100 steps; sub-0.5 drop ⇒ optimizer or LR wrong.)
- **Step 500 ema** ≤ `5.5`. (Smoke gate. Previous run with bug plateaued at 9.6+; broadcast fix should clear this comfortably if patch is correct.)
- **Step 1000 ema** ≤ `4.5`. (Full gate.)
- **Inference smoke** at canonical chat-format prompt (`encode_chat_prompt "Q: Who are you?"`): top-3 next-token includes ≥1 token from `doe_pure.jsonl` vocabulary (`parliament`, `expert`, `vote`, `consensus`, `experts`, `voting`, `meta`) — direct grep on decoded output.
- **HF upload** of `doe_lora.bin` immediately at phase end (not deferred).

**Fail recovery (3-strike)**:
- NaN at any step → halve LR, restart from last clean ckpt.
- Step 100 gate miss → check optimizer (must be `nt_tape_chuck_step`, not Adam baseline), halve LR.
- Step 500 gate miss → suspect RRPRAM patch incorrect; surface to Codex with grad sample for review.
- Step 1000 gate miss → declare Phase 1 FAILED. Do not rebrand "best-effort".

### Phase 1.5 — Codex review of Phase 1 run

- Capture `phase1_doe/run.log`, LoRA file size, smoke inference output.
- `codex exec` with prompt: `"Phase 1 DoE training log + smoke output
  attached. Verify: numeric gates from plan v2 §Phase 1 hit. List any
  gate that was missed but rebranded as PASS. Verify final LoRA file
  size matches expected (rank=16 × 7 projections × 20 blocks × float32
  per JANU dims)."`
- Continue only on PASS.

### Phase 2 — Arianna LoRA on Resonance 200M

**Pre-flight**: pull `huggingface.co/ataeff/heart.c/arianna_lora_v2/run.log` (Defender's in-flight retrain output, if landed). Read final ema. Verify with inference smoke.

**Decision tree:**
- If Defender's v2 final ema ≤ 4.0 AND inference smoke produces coherent Arianna register ⇒ **ACCEPT v2 as-is**. Skip Phase 2 retrain. Document in `docs/arianna_v2_acceptance.md`.
- Else ⇒ clean retrain in Phase 2.

**6-point brief (only if retrain):**

| field | value |
|---|---|
| organism | Resonance 200M base, RS02 magic 0x52533032, V=16384 E=768 H=12 D=64 B=20 M=2048 T=2048 R=48 |
| dataset | `arianna_dataset_final_clean.txt`, 1.21 MB / 1227 Q/A pairs (verified at P-2 via `wc -l` + `grep -c "^Q:"`) |
| Karpathy steps | 3000 steps (≥3× the v1 1000-step undertrain; Defender's v2 ran 3000 — match) |
| arch | LoRA r=16 α=32 on 7 projections (q/k/v/o + mlp_gate/up/down); wr_a/wr_b/gate frozen |
| tokenizer | RS02 header BPE (n_merges + triples, read at trainer init) |
| script | `train/train_arianna_lora.c` post-P-2 verify |

**Numeric gates:**
- Step 0 base loss ∈ `[4.5, 5.5]` (Defender v1 saw 4.94 → reproducible band)
- Step 500 ema ≤ 4.7
- Step 1500 ema ≤ 4.0
- Step 3000 ema ≤ 3.5
- Inference smoke produces non-uniform top-1 with Arianna register ("I am", "you are", "field", "method" — direct grep)
- HF upload immediately at phase end

**Fail recovery (3-strike)**: same as Phase 1.

### Phase 2.5 — Codex review

Same shape as Phase 1.5.

### Phase 3 — 432-cell sweep across 4 voices

Per ARCHITECTURE.md §3 (preserved as design contract).

- Voices: Yent (Janus 176M Yent SFT), Arianna (Resonance 200M + Phase 2 LoRA merged via `merge_arianna_lora`), Leo (Janus 170M Leo SFT, existing), DoE (Janus 170M + Phase 1 LoRA merged).
- Axes (defined HERE):
  - `temp ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 1.0}` (6) — note 0.3-0.5 included per Defender's `feedback_temp_sweep_before_judging_2026_05_07.md` finding that low temp can reveal memorization.
  - `top_k ∈ {40, ∞}` (2) for Janus voices; `top_p ∈ {0.9, 1.0}` (2) for Arianna (Resonance, no chat-tokens).
  - `rep_pen ∈ {1.0, 1.3, 1.4}` (3).
  - `prompts ∈ {technical, philosophical, personal}` (3) — same texts across voices.
- Total: 6 × 2 × 3 × 3 = 108 per voice × 4 voices = **432 cells**.
- Per-cell timeout 90 s. Hard fail: NaN streak ≥5 ⇒ exit 42; coherence streak (unique_ratio < 0.30) ≥3 ⇒ exit 43.

**Numeric gates:**
- 432 transcripts in `phase3_sweep/transcripts/`.
- 432 lines in `scores.tsv` plus header.
- Per voice: locked `(temp, top_k_or_p, rep_pen)` triple committed to `core/voices_optima.h` as `#define <VOICE>_TEMP …`.
- `voices_optima.h` committed to `ariannamethod/heart.c:core/voices_optima.h` before Phase 3.5.

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

### Phase 5/6/7/8 — Codex spot-check on inherited verifications

Defender's post-rage redo produced these with explicit numeric gates:

| phase | gate | run output |
|---|---|---|
| 5 KK | 7-signal weights at 1e-6 | `phase5_smoke_v2/kk.txt` — 7/7 PASS |
| 6 field_clock 168h | `var(YA) < var(YL)` and `var(AL) < var(YL)` | `phase6_smoke_v2/field_clock.txt` — 3.6× and 7.6× ratios |
| 7 duet field-driven | distribution ≠ N/4 | `phase7_duet_v2/duet.txt` — Y=1 A=3 L=4 D=0 |
| 8 daemon | 3 commands pass | `/tmp/heart` 16984 bytes, status/serve/converse all ✓ |

`codex exec` per phase: read the run log, verify the assertion math, flag if gate is trivially passable. Re-run only if Codex flags BLOCKER.

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

## Cost estimate

| step | h | $ |
|---|---|---|
| Phase 0 boot + toolchain | 0.25 | 0.37 |
| Phase 1 DoE LoRA (1000 steps post-RRPRAM patch) | 0.4 | 0.60 |
| Phase 1.5 Codex review | 0.1 | 0.15 |
| Phase 2 Arianna LoRA (cond.) | 0.3 | 0.45 |
| Phase 2.5 Codex review | 0.1 | 0.15 |
| Phase 3 432-cell sweep | 1.0 | 1.49 |
| Phase 3.5 Codex review | 0.1 | 0.15 |
| Phase 4 (a) Soul micro impl + test | 0.5 | 0.75 |
| Phase 5/6/7/8 Codex spot-check | 0.2 | 0.30 |
| Phase 9 post-run audit | 0.3 | 0.45 |
| Buffer for Singularity-mode 3-strike | 0.5 | 0.75 |
| **Total estimate** | **~3.75 h** | **~$5.61** |

Comfortably within $15 ceiling.

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
