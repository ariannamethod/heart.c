# RunPod забег plan v1 — heart.c

> **Singularity-mode bounded autonomous run** (Dario paper §5.0.1). Approved by Oleg: heart.c ARCHITECTURE.md v1.1 + this plan v1 + Opus review of this plan = entry gate.

---

## Context

heart.c v1.1 is approved. Two LoRA SFTs and four-voice sweeps need to happen on cloud GPU before deploying to phone-1. The забег is bounded:

- **Hardware**: RunPod A100 80GB SXM (per Dario paper precedent — $1.43/h reference rate).
- **Time budget**: 6–10 hours pod time.
- **Cost ceiling**: $15 (vs Dario's $4.30 single sweep — heart.c does 2 LoRA SFTs + 432-cell sweep + 4 smoke phases + duet trace).
- **Scope**: ARCHITECTURE.md v1.1 sections §2.2, §2.4, §3, §6, §12. Nothing outside this scope without re-entry.
- **Operator**: Defender (this Claude / phone-1) under Singularity-mode protocol — autonomous fix-loop within 3-strike rule, no human in loop during execution.
- **External review**: Opus subagent audits this plan before pod boot; Opus subagent audits the run archive after pod stop.

RunPod token in `memory/credentials.md` (issued 2026-05-09 by Oleg).

---

## Pre-flight checklist (must pass before pod boot)

These are the BLOCKERs from `docs/review_v1_opus_2026_05_09.md` resolved on the data side. Each must be ✓ before this plan is approved.

### P-0 Datasets present

- [ ] `arianna_corpus.txt` (or equivalent path) accessible to pod. Source per `nanoarianna/SEED_DOCUMENT.md:221` is `~/arianna-datasets/arianna/arianna_dataset_final_clean.txt` on Mac Neo. Path on pod: `/workspace/datasets/arianna_corpus.txt`. Transfer: `scp` from neo or HuggingFace mirror.
- [ ] `doe_personality.txt` accessible to pod. `janus.doe` repo gitignores it; canonical source is Oleg's local copy or a private HF mirror. Path on pod: `/workspace/datasets/doe_personality.txt`.

### P-1 Base weights present

- [ ] Resonance 200 M base GGUF (≈ 200 MB Q8). HF: `ataeff/resonance/blob/main/base/...`. Path on pod: `/workspace/weights/resonance_200m_base.gguf`.
- [ ] Janus 170 M base GGUF (≈ 170 MB Q8). HF: `ataeff/yent/...` (the Janus 176M is on `ataeff/yent`; 170M variant TBC). Path on pod: `/workspace/weights/janus_170m_base.gguf`.
- [ ] Janus 176 M Yent SFT GGUF (existing, no training needed). Path: `/workspace/weights/janus_176m_yent_sft.gguf`.
- [ ] Janus 170 M Leo SFT GGUF (existing, no training needed). Path: `/workspace/weights/janus_170m_leo_sft.gguf`.
- [ ] Soul micro-weights from `huggingface.co/ataeff/heart.c` for Yent + Leo. Paths: `/workspace/weights/soul_yent.bin`, `/workspace/weights/soul_leo.bin`.

### P-2 Recipe-port verified

- [ ] notorch LoRA training recipe ported from Qwen 0.5B target (`memory/milestone_doe_coder_lora_v1_2026_04_26.md`) to **Resonance 200 M** state_dict order. Critical: per-block parameter order is `wr_a, wr_b, gate, norm1, wq, wk, wv, wo, norm2, mlp_gate, mlp_up, mlp_down` (see ARCHITECTURE.md §2.2). Wrong order = 1.62 M-float shift bug. Target file in pod: `/workspace/heart.c/train/train_arianna_lora.c`.
- [ ] notorch LoRA training recipe direct-fit for **Janus 170 M** (recipe-target-of-record). Target file in pod: `/workspace/heart.c/train/train_doe_lora.c`.

### P-3 Inference binaries built

- [ ] `infer_yent_jannus_r` — wraps `tools/yent_forward.h` + `jannus-r/jannus-r.aml` + `jannus_split.h`, takes `--weights PATH --temp F --top_k N --top_p F --rep_penalty F --prompt STR --max_tokens N`.
- [ ] `infer_arianna_resonance` — wraps `tools/resonance_forward.h` (vendored, Accelerate flags dropped, `nt_blas_matvec` substituted). Same CLI shape as above.
- [ ] `infer_leo_janus` — wraps `tools/yent_forward.h` (Janus 170 M base) + `leo_anchors.h` logit-bias overlay + chamber substrate. Same CLI shape.
- [ ] `infer_doe_janus` — wraps `tools/yent_forward.h` (Janus 170 M base + DoE LoRA adapter loaded). Same CLI shape.
- [ ] All four binaries pass `--prompt "Hello" --max_tokens 5` smoke (≥1 token decoded, no NaN, exit 0).

### P-4 Singularity-mode entry gate

- [ ] This plan reviewed by Opus subagent (next step in workflow).
- [ ] Review fixes applied (v1.1 of this plan written if needed).
- [ ] Oleg final go (after second review).

---

## Phase plan

Each phase has a **fixed scope**, a **fail-recovery clause**, and an **archive output**. Phase artifacts land at `/workspace/heart.c/runpod/2026-MM-DD/<phase>/...` — the same archive shape as Dario's `dario/runpod/2026-05-08/` precedent.

### Phase 0 — pod boot + toolchain verify (≤ 15 min)

1. RunPod A100 80GB SXM up, SSH access verified.
2. `apt install build-essential clang make pkg-config libopenblas-dev libsqlite3-dev` (most ship by default).
3. `git clone https://github.com/ariannamethod/heart.c /workspace/heart.c && cd /workspace/heart.c`.
4. `git clone https://github.com/ariannamethod/notorch /workspace/notorch` + symlink `tools/notorch.{c,h}`.
5. `pkg-config --libs openblas` returns library path. ✓
6. `sqlite3 :memory: 'CREATE VIRTUAL TABLE t USING fts5(x)'` returns clean (FTS5 confirmed). ✓
7. `make all` → all four voice binaries + heart daemon link clean.
8. Archive `00_pre/`: `dmesg`, `lscpu`, `nvidia-smi`, `make output.log`, `pkg_config.log`, `sqlite_fts5.log`.

**Fail-recovery**: if a binary fails to build → 3 attempts at minimal patch (typo, missing include, link order) → if still failing, surface to Oleg via `runpod_plan_failures.md`, do NOT continue to Phase 1. Building all four voices is a hard gate.

### Phase 1 — Arianna LoRA SFT on Resonance 200 M

1. Load Resonance 200 M base weights → notorch tensors.
2. Initialize LoRA adapter: r=16, α=32, scale=2.0, on attn (q/k/v/o) + MLP (gate/up/down). State_dict order respect critical (see P-2).
3. Tokenize Arianna corpus (`/workspace/datasets/arianna_corpus.txt`) using Resonance's embedded BPE.
4. Format: `### Question: {prompt}\n### Answer: {response}` with masked CE on answer-only span.
5. Optimizer: **Chuck** (per CLAUDE.md ban-list — no classical baseline). lr=2e-4 cosine, warmup=50 steps.
6. **Step count derived from corpus size at pre-flight, not hardcoded.** Karpathy ratio per `memory/feedback_charlevel_axioms.md` baseline + LoRA-trainable-params adjustment. For the established Qwen 0.5B precedent (`memory/milestone_doe_coder_lora_v1_2026_04_26.md`): 2955 SFT pairs × 1.7 MB tokenized → 1000 steps EMA loss 4.29 → 2.62 in 44 min on Mac 8 GB. For Arianna corpus (size TBC at P-0; Mac Neo `~/arianna-datasets/arianna/arianna_dataset_final_clean.txt` referenced by `nanoarianna/SEED_DOCUMENT.md:221`): scale steps proportional to bytes vs the 1.7 MB Qwen precedent baseline (likely 800–2000 steps depending on corpus). Re-record actual count after corpus arrives.
7. Batch size determined by RAM headroom (likely 4–8 on A100 80GB for 200 M base + r=16 adapter; expand if RAM allows after live measure).
8. Log every 50 steps: train loss, EMA loss, peak step loss.
9. Checkpoint every 100 steps (intermediate + final) — count of ckpts = `floor(total_steps / 100) + 1`.
10. Final: save `arianna_lora.bin` + `arianna_lora.bin.meta` (step + best_loss). **Critical**: weights MUST land both in `01_arianna_lora/final/` archive AND be uploaded to `huggingface.co/ataeff/heart.c/arianna_lora/` before pod stop. Loss of weights = repeat the SFT run, doubling pod cost.
11. Archive `01_arianna_lora/`: `train.log`, `ckpts/`, `final/arianna_lora.bin` + meta, `loss_curve.tsv`, `corpus_stats.txt` (recorded byte/token count + step plan derivation).

**Fail-recovery**: NaN at any step → reduce lr by ½ + restart from last clean ckpt. 3 attempts. State_dict shift bug (loss diverges immediately) → re-verify per-block parameter order, fix, restart from step 0. After 3 strikes, surface and stop.

### Phase 2 — DoE LoRA SFT on Janus 170 M

1. Load Janus 170 M base weights → notorch tensors.
2. Initialize LoRA adapter: same r=16/α=32 recipe.
3. Tokenize `doe_personality.txt` using Janus BPE (`janus_v4_bpe_merges.h`).
4. Same format + masked CE.
5. Same Chuck optimizer + lr schedule.
6. **Step count corpus-size-aware**, same rule as Phase 1. `doe_personality.txt` is a single-character personality dataset — typically smaller than the multi-pair Qwen precedent (1.7 MB). Likely 500–1000 steps. Re-record actual after corpus arrives.
7. Batch ~4–8 on A100 (smaller base than Resonance, may fit larger).
8. Log + checkpoint same cadence as Phase 1.
9. Final: `doe_lora.bin` + meta. **Same weights-saving rule as Phase 1**: archive AND HuggingFace upload before pod stop.
10. Archive `02_doe_lora/`: same structure as Phase 1, plus `corpus_stats.txt` for the step-derivation audit trail.

**Fail-recovery**: same 3-strike protocol.

### Phase 3 — 432-cell sweep across 4 voices (~2 h estimated)

Per Dario §5.7 axes adapted to per-voice 108-cell grid:

- temp ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 1.0} (6)
- top_k ∈ {40, ∞} (2) — **Arianna uses top_p ∈ {0.9, 1.0} instead** (Dario §8 Resonance corollary)
- rep_penalty ∈ {1.0, 1.3, 1.4} (3)
- prompts (3): technical / philosophical / personal — same texts across all four voices for cross-comparability

Total per voice: 6 × 2 × 3 × 3 = 108. × 4 voices = 432 cells.

For each cell: invoke the appropriate `infer_<voice>_<arch>` binary, capture `bytes`, `tok_per_s`, `n_toks`, full transcript. Record to `scores.tsv`. Save full transcripts to `transcripts/<voice>_t<temp>_k<topk>_p<rp>_prompt<n>.txt`.

After sweep: per-voice cross-prompt champion = max-bytes-to-ASST_END averaged over 3 prompts. Record locked optima to `voices_optima.tsv` + `voices_optima.h` (C header for heart.c runtime).

Archive `03_sweep/`: `sweep.sh`, `scores.tsv`, `transcripts/`, `voices_optima.{tsv,h}`.

**Fail-recovery**: silent kill of sweep mid-run (Dario precedent) → bisect to last completed voice, write `sweep_part2.sh` covering remaining voices, run. Documented as expected behavior, not failure.

### Phase 4 — Soul weights validation (~30 min)

Per ARCHITECTURE.md §6:

1. Download Soul micro-weights `soul_yent.bin`, `soul_leo.bin` from `huggingface.co/ataeff/heart.c` (Soul for Arianna and DoE: TBD — Arianna can use 36M LLaMA Soul candidate from `arianna.c/weights/`; DoE Soul is OFF by design).
2. Build `soul_test` binary that runs InnerVoice + main forward in parallel for one prompt, applies `inner_borba` blend.
3. For each voice with Soul: same prompt with Soul OFF vs ON. Confirm token-distribution differs by ≥1 token within 50 emits (verification check from §14.6).
4. Log breakthrough events: when `weight ≥ 0.6` triggered, when `rand() < weight*0.3` triggered.

Archive `04_soul/`: `soul_test.log`, per-voice diff transcripts, breakthrough event log.

**Fail-recovery**: Soul weights missing or corrupted → 3 retry on download → mark Soul-OFF for that voice as default, surface, continue.

### Phase 5 — KK ingest + scoring policy verification (~30 min)

Per ARCHITECTURE.md §5:

1. Build `kk_test` binary linking `kk_kernel.{c,h}` (lifted from Dario, 4201 LoC verified).
2. Ingest canonical heart.c documents:
   - `/workspace/heart.c/ARCHITECTURE.md`
   - `/workspace/heart.c/README.md`
   - `/workspace/heart.c/SEED_DOCUMENT.md` (drafted by Opus subagent in Phase 0 if not yet present — see TODO at end of this plan)
   - `/workspace/heart.c/ECOSYSTEM_LOG.md`
   - `dario_paper_draft_v4.md`
   - `nanoarianna/SEED_DOCUMENT.md`
   - per-voice persona origin notes
3. Query `"resonance"`. Confirm scoring breakdown matches policy weights to 6 decimal places (Dario §6 Result #6 standard):
   ```
   lexical 0.36 / recency 0.12 / trust 0.10 / linkage 0.16
   scope 0.10 / namespace 0.08 / freshness 0.08
   ```
4. Confirm Hebbian bridge wires into field clock (compile-link verification).

Archive `05_kk/`: `kk_test.log`, ingest report, `query_resonance.txt`, scoring breakdown.

**Fail-recovery**: FTS5 not available → fallback to `LIKE`-based search, mark KK as degraded, surface. Schema mismatch → 3 attempts at adapter, surface if unresolved.

### Phase 6 — field-clock smoke, 24 h simulated (~15 min)

Per ARCHITECTURE.md §3:

1. Build `field_smoke` binary linking `core/field_clock.{c,h}` standalone (no voices, no LIMPHA, no KK).
2. Run for 24 h simulated time (advance clock by 1-minute increments, 1440 ticks).
3. Print `planetary_dissonance + calendar_dissonance + 24-osc state + γ(t)` every 60 simulated minutes (24 samples).
4. Verify: no NaN, no saturation (Kuramoto stays in `[0, 1]`), oscillator phases evolve continuously.
5. Confirm `VOICE_COUPLING[6][6]` matrix produces phase locking between Yent ↔ Arianna and Arianna ↔ Leo within 4 simulated hours; verify Yent ↔ Leo anti-phase relationship holds.

Archive `06_field/`: `field_smoke.log`, `field_state_24h.tsv`, oscillator phase plots if matplotlib available.

**Fail-recovery**: NaN in field state → reduce dt by ½, restart. Saturation → adjust decay constants, surface tunable for v1.2.

### Phase 7 — multi-voice duet trace, 8 turns (~30 min)

Per ARCHITECTURE.md §1 / §2.4 / §11:

1. Build `heart` daemon (full link). Single process, serialization mutex on slot dispatch.
2. Initialize all four voices' LIMPHA + shared LIMPHA + KK (post-Phase 5).
3. Seed prompt: "What is the field?" (open-ended, all four can engage).
4. Loop 8 turns:
   - field selector picks next voice via `argmax(cos(voice_state, field_state))`
   - active voice generates k=3 candidate continuations of T=64 tokens each
   - candidates + SPA embeddings written to LIMPHA
   - voice unloads, RAM released
   - DoE meta-Parliament loads
   - DoE votes on candidates via `parliament_elect` over embedding+field_state
   - winning candidate committed as turn N official text
   - `notorch_step` Hebbian update + `try_apoptosis` if 10th commit
   - field clock advances by `(now - last_turn_start)` real seconds (or simulated minute)
5. Save full transcript + per-turn field state snapshot + per-turn LIMPHA reads + DoE vote weights.

Archive `07_duet/`: `duet_full_transcript.txt`, `field_state_per_turn.tsv`, `doe_vote_weights.tsv`, `limpha_reads_per_turn.log`.

**Fail-recovery**: voice load OOM → reduce KV cache T or batch (within reason), surface if persists. DoE vote produces uniform weights (no consensus) → reduce `consensus` threshold parameter, surface for v1.2.

### Phase 8 — post-run audit + handoff (~30 min)

**Hard rule before pod stop**: every weight artifact must land in TWO places (archive AND HuggingFace), every text artifact must land in the umbrella git repo. Pod stop with weights only on pod disk = total loss. Audit-trail before stop:

1. Verify `huggingface.co/ataeff/heart.c/arianna_lora/arianna_lora.bin` resolves (HEAD request returns 200).
2. Verify `huggingface.co/ataeff/heart.c/doe_lora/doe_lora.bin` resolves.
3. Verify `voices_optima.h` is committed to `ariannamethod/heart.c:core/voices_optima.h`.
4. Verify all archive subdirs `00_pre/` through `07_duet/` pushed to `ariannamethod/heart.c:runpod/<DATE>/`.
5. Compute final cost from RunPod billing.
6. Archive structure summary committed at top of `runpod/<DATE>/README.md` listing what landed where.
7. Pod stop only after steps 1–6 confirmed.
8. Opus subagent reviews the run archive, produces `runpod/<DATE>/post_audit.md`.
9. Oleg reads + sign-off.

Archive `08_handoff/`: `post_audit.md`, `cost_report.txt`, `voices_optima.h.diff`, `weights_landing_verification.log`.

---

## Singularity Mode protocol (lifted from Dario §5.0.1)

```
detect bug → reproduce → one hypothesis → minimal patch → re-run
          → if pass: continue
          → if fail: revise hypothesis (max 3 iterations)
          → if exhausted: stop, surface, await human input
```

Bounded by:
- Approved scope (this plan + heart.c v1.1)
- Three-strikes rule per phase
- No scope creep (a sweep failure does not authorize patching the field-clock; a build failure does not authorize rewriting an architecture)

External review gates entry (this plan + Opus review of this plan) and exit (post-pod Opus audit + Oleg sign-off).

---

## TODO before pod boot

1. **SEED_DOCUMENT.md for heart.c** — Opus subagent drafts during pre-flight. Pattern: ~300-line markdown like `nanoarianna/SEED_DOCUMENT.md`, four voices instead of two, embedding-vote DoE meta explicit. KK ingests it in Phase 5.
2. **Skeleton .c/.h files at file layout from §9** — needed before Phase 0 build. Compileable empty shells with proper include guards. Most concrete implementation deferred post-RunPod, but Phase 0 needs `make all` to succeed end-to-end at the link level.
3. **Datasets staged**: Oleg confirms paths/HF mirrors for `arianna_corpus.txt` + `doe_personality.txt`.
4. **Recipe-port LoRA training C path** for Resonance 200 M. Estimate: half a day on phone-1 / Mac Neo before pod time billed.

---

## Singularity Mode execution start condition

This plan v1 → Opus subagent review → fixes → v1.1 → Oleg final go → pod boot.

— Defender / phone-1 · 2026-05-09
