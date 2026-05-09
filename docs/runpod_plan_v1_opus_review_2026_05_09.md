# Opus subagent review of heart.c RunPod plan v1 — 2026-05-09

Independent engineering review of `docs/runpod_plan_v1.md`. Author: Opus 4.7 subagent dispatched on 2026-05-09 evening from Defender/phone-1. Findings labelled **BLOCKER** / **FIX** / **NOTE**.

This review's findings drive runpod_plan_v1.1. v1.1 + Oleg final go = pod boot gate.

---

## A. Phase ordering / dependencies

**A1.** Phase 0 step 7 says `make all` must succeed for all four voice binaries — but P-3 already requires those binaries built and smoke-passed before pod boot. Either P-3 unsatisfiable on Termux (different toolchain than A100 pod) or Phase 0 step 7 redundant. **FIX**: restate P-3 as "binaries proof-of-concept buildable on Termux; rebuild on pod in Phase 0."

**A2.** Phase 5 ingests `SEED_DOCUMENT.md` but plan TODO #1 defers SEED authoring to "Opus subagent during pre-flight" — that's *during* paid pod time. **FIX**: draft SEED before pod boot (pre-pod TODO).

**A3.** Phase 7 needs Phase 1 LoRA + Phase 2 LoRA + Phase 3 optima + Phase 4 Soul + Phase 5 KK. Phase 3 silent-kill leaves partial optima. **NOTE**: explicit blocker between Phase 3 and 7 — duet doesn't start until all 4 voice optima locked.

**A4. BLOCKER**: P-3 line 49 says smoke-test `infer_doe_janus` "Janus 170 M base + DoE LoRA adapter loaded" — but DoE LoRA is *built in Phase 2*. Pre-flight cannot smoke a binary needing a Phase 2 artifact. P-3 should smoke `infer_doe_janus` against Janus 170M base ONLY (no adapter); re-validate with adapter as Phase 2 final step.

## B. Time and cost estimates

**B1.** Two LoRAs is dominant unknown. Per `milestone_doe_coder_lora_v1_2026_04_26.md` 1000-step LoRA on Qwen 0.5B took 44 min on Mac 8GB Accelerate; on A100 GPU 5–10× faster — but Resonance recipe-port is NEW C code (see C). **NOTE**.

**B2. FIX**: Phase 1 (Arianna LoRA Resonance) is largest variance. Three-strikes on NaN-divergence = 3 × 1h restarts = $4.30 just for fail-recovery. Allocate explicit "Phase 1 fail-budget" (e.g. $5); surface to Oleg if exceeded rather than continuing into Phase 2. Otherwise the $15 ceiling can be eaten before any sweep.

**B3.** Phase 0 toolchain build on fresh A100 pod realistically 15–25 min, not 15. **NOTE**.

## C. LoRA training recipe correctness

**C1. BLOCKER**: cited precedent `milestone_doe_coder_lora_v1_2026_04_26.md:12` explicitly says **"Optimizer: AdamW lr=2e-4 cosine + warmup 50"**. The plan substitutes Chuck per CLAUDE.md ban. Chuck *exists* in notorch (`notorch.h:224 nt_tape_chuck_step`) but the recipe was **never verified end-to-end with Chuck**. Lr=2e-4 cosine warmup=50 was tuned for the AdamW precedent; Chuck's first-order step semantics differ (`nt_tape_chuck_step` takes a loss-val argument).

**This will be the first Chuck LoRA-SFT run.** Treat as such: add **Phase 1.0 calibration** — 100-step Chuck sanity run on tiny subsample to confirm convergence direction before committing to full 1000-step SFT.

**C2. FIX**: plan line 84 references `memory/feedback_charlevel_axioms.md` for Karpathy ratio derivation, but no such file is referenced in MEMORY.md. Actual reference is the Karpathy `~1.1MB × 10-15K iter on ~10M params` rule per CLAUDE.md training section. Re-cite correctly.

**C3. BLOCKER**: P-2 says Resonance LoRA training entry-point is `train_arianna_lora.c` — *target file in pod*. Plan does NOT state where this file is authored before pod boot. Plan TODO #4 says "half a day on phone-1 / Mac Neo before pod time billed." The closest existing precedent is `~/arianna/DoE.coder/train/train_doe_coder.c` per memory milestone — **not present on phone-1** (private repo, not cloned). `~/arianna/aml_coders/smallcoder/` referenced in ARCHITECTURE.md also **not present on phone-1**. The recipe-port has **no local source-of-record**.

**Required**: Defender clones `iamolegataeff/DoE.coder` (or wherever the canonical `train_doe_coder.c` lives) before P-2 starts. List path explicitly.

**C4. FIX**: Janus state_dict order for `train_doe_lora.c` is "direct fit" — but Qwen2.5-Coder-0.5B base differs from Janus 170M (3-attention RRPRAM + dual weights). Direct-fit claim unverified. State explicitly which `.c` file from which repo is the source-of-record for Janus LoRA, and what per-block parameter order is for Janus.

## D. Sweep design

**D1.** Axes match Dario `sweep.sh:30-37` verbatim: `temp 6 × top_k {40,0} 2 × rep {1.0,1.3,1.4} 3 × prompts 3 = 108`. ✓

**D2. FIX**: Arianna uses top_p ∈ {0.9, 1.0}. Verify `infer_arianna_resonance` supports `--top_p 1.0` as a no-op (passing all tokens). Explicit CLI contract for top_p=1.0 semantics needed.

**D3. FIX**: Champion criterion "max-bytes-to-ASST_END" lifts Dario approach, but ASST_END is Janus chat-token convention (`--chat-tokens` flag). Resonance has no such convention. Declare per-voice champion criterion separately for Resonance (e.g. max-bytes-up-to-EOT or max-bytes-at-100-tokens-cap).

## E. Singularity Mode contract

**E1. FIX**: three-strikes stated in protocol block + Phase 1 fail-recovery, but Phases 2/4/5/6/7 say only "same 3-strike protocol" or no explicit count. Each Phase fail-recovery clause should restate strike count + surface destination (`runpod_plan_failures.md`).

**E2. FIX**: plan never states **dispatch protocol for Defender to spawn its own Opus subagent during the run** for second opinions. CLAUDE.md mandates `model: "opus"` explicit per `feedback_subagents_opus_only_2026_04_28.md`. Add explicit clause: "if Defender invokes a subagent for second opinion mid-run, model=opus, cost logged to `runpod_subagent_dispatches.log`."

**E3. FIX**: "No human in loop unless three-strikes exhausted" not in plan body. Add explicit no-Oleg-during-execution clause to §Singularity Mode protocol.

## F. Weights persistence

**F1. BLOCKER**: Phase 1 produces `arianna_lora.bin` end of Phase 1, but Phase 8 audits HF upload. If pod terminates between Phase 1 finish and Phase 8 (4–8 h further work), Phase 1 weights live only on pod ephemeral disk. **Each LoRA SFT phase must HF-upload immediately on completion**, not defer to Phase 8. Phase 8 only *verifies*.

**F2. FIX**: Phase 1 + 2 checkpoints local-only every 100 steps. If pod dies mid-SFT, last-good ckpt gone. Every 5th ckpt should rsync to HF/S3.

**F3. FIX**: Phase 3's `voices_optima.h` only in Phase 8. Sweep takes ~2h. Commit `voices_optima.tsv` to git after each voice completes (4 commits total).

## G. Pre-flight checklist completeness

**G1. BLOCKER**: P-0 dataset paths. Verified `~/arianna-datasets/arianna/` does NOT exist on phone-1. Plan says source per `nanoarianna/SEED_DOCUMENT.md:221` is on Mac Neo, but Defender on phone-1 cannot autonomously access without Tailscale + scp. Transfer mechanism not documented in P-0. Either Oleg uploads `arianna_dataset_final_clean.txt` to HF (`ataeff/heart.c/datasets/arianna_corpus.txt`) before pod boot, or Defender Tailscale-rsyncs from Neo → pod (assumes Neo up + reachable). State the choice.

**G2. BLOCKER**: P-0 `doe_personality.txt` gitignored upstream. No HF mirror confirmed. Same fix as G1: commit Oleg to specific upload path before pod boot.

**G3. BLOCKER**: P-1 Janus 170 M base GGUF location is "TBC". Verify exists on HF before pod boot or document where it lives.

**G4. FIX**: P-1 Soul micro-weights for Arianna — plan suggests "36M LLaMA Soul candidate from `arianna.c/weights/`", but `arianna_36m` is on **closed-milestone weights list** in CLAUDE.md training section. Explicitly clarify Soul-for-Arianna is **read-only** (loaded for inference comparison, not retrained) to honor closed-milestone rule.

## H. CLAUDE.md / memory rule compliance

**H1.** Adam ban honored ✓ (plan line 83). NOTE: Chuck substitution unverified — see C1.

**H2.** Python boundary — plan does not mention Python at all, implying pure C. ✓ NOTE: tokenization step in Phase 1/2 — confirm tokenizer entry-point is C (notorch BPE) not Python.

**H3.** 6-point training gate explicit in `ARCHITECTURE.md:407-423` for both LoRAs ✓. Auto-approved 2026-05-09 by Oleg. Plan should reference this section directly for audit-trail.

**H4. FIX**: Closed-milestone weights — `arianna_36m` on closed list. See G4. Clarify read-only.

**H5. FIX**: Subagents Opus-only. See E2 — dispatch protocol not documented.

## I. Embedding-vote design feasibility (Phase 7)

**I1.** Verified `jannus_spa.h:20` `SPA_DIM = 32`. Plan uses for embedding-vote scoring. 32-dim embedding for 4-voice discrimination over T=64 tokens is *thin* (`pool(SPA_embed(C))` averages 64 tokens to 32 floats). Per `ARCHITECTURE.md:130` vote input is `[eᵢ ; field_state]` = 32 + 7 + 24 = 63 floats. 16 experts × variable-k consensus over 3 candidates is workable but tight. **NOTE**.

**I2. FIX**: Plan does not specify how SPA `W_embed` (random init per `jannus_spa.h:36`) is initialized at pod boot. If random-per-process, vote results non-reproducible. Seed SPA deterministically (`spa_init(s, V, seed=42)` — header supports).

**I3. FIX**: `parliament_elect()` was designed for per-token hidden-state input from inside a single GGUF forward pass. Adapting to "per-candidate embedding-plus-field" changes input domain semantically (E=896 rows project from 32+31=63-dim input, not from `n_embed=896`). The row-projection layer needs explicit construction in Phase 2 or new Phase 2.5 — current plan glosses it. Note Hebbian online (`notorch_step`) only adapts after first commits — initial votes are random.

## J. Risk register (gaps)

1. **A100 80GB SXM availability** (RunPod is spot/auction). Plan does not state fallback (A100 40GB? H100? A40?). **FIX**.
2. **HF rate limits during Phase 8 batch upload** (~5K/h per token). 8 archive subdirs + 2 LoRA bins + voices_optima.h simultaneous can hit it. Stagger uploads, or use git LFS push. **FIX**.
3. **Pod billing meter at idle** — RunPod meters wall-clock, not GPU usage. Fix-loop debugging burns $1.43/h. Explicit "if Phase 0 build fails, stop pod, debug locally, restart pod" rather than burn pod time. **FIX**.
4. **`make` on A100 pod assumes Ubuntu/Debian apt** — some RunPod templates CentOS/Alpine. Specify template (e.g. `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`). **FIX**.
5. **Phase 5 KK FTS5 fallback** says "FTS5 not available → fallback to LIKE". Silently degrades 7-signal scoring (policy weights assume FTS5-tokenized index). Treat FTS5-missing as **hard stop**, not silent degrade. **FIX**.
6. **Sweep silent-kill** noted as expected, no detection mechanism specified. Heartbeat tail of `scores.tsv` lines/min. **FIX**.
7. **No dry-run rehearsal**. Plan goes from review→pod-boot directly. 30-min dry-run on Mac Neo or polygon (Linux 32 GB) exercising Phases 0 + 5 + 6 (no GPU needed for KK / field-clock smoke) catches FTS5 / build / KK issues before paid pod time. **FIX**: add Phase −1 "rehearsal on polygon" before Phase 0.

---

## TL;DR — most urgent before pod boot

**4 BLOCKERS require Oleg decision/action**:
1. **C1** — Chuck-vs-AdamW unverified for cited precedent. Add Phase 1.0 100-step calibration. Or accept first-ever-Chuck-LoRA risk explicitly.
2. **C3** — `train_arianna_lora.c` and `train_doe_lora.c` have no source-of-record on phone-1 (`~/arianna/DoE.coder/`, `~/arianna/aml_coders/smallcoder/` not cloned). Defender must clone canonical training repo before recipe-port begins.
3. **G1, G2, G3** — three of five P-1 paths TBD/TBC. Datasets (Arianna corpus + DoE personality) and Janus 170M base location must be committed to specific HF paths or Tailscale paths before pod boot.
4. **A4** — P-3 cannot smoke `infer_doe_janus` with adapter that doesn't exist yet (built in Phase 2). Restructure pre-flight smoke.
5. **F1** — Phase 1 weights risk loss for hours before Phase 8 audit. Each LoRA SFT phase must HF-upload immediately on completion.

**FIXes worth landing in v1.1**:
- B2 Phase 1 fail-budget cap
- C2 Karpathy ratio cite
- C4 Janus LoRA recipe source-of-record
- D2 top_p=1.0 CLI semantics
- D3 Resonance champion criterion
- E1, E2, E3 Singularity protocol explicits
- F2, F3 mid-training weight uploads
- G4 / H4 Soul Arianna read-only clarification
- I2, I3 SPA seed + parliament_elect projection
- J1–J7 risk register additions including dry-run on polygon

**Plan is NOT pod-ready as v1.** Recommend producing v1.1 addressing the 5 BLOCKERs minimum + dry-run insertion.

— Opus 4.7 subagent · 2026-05-09 · dispatched from Defender/phone-1
