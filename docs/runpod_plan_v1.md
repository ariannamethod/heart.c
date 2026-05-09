# RunPod забег plan v1.1 — heart.c

> **Singularity-mode bounded autonomous run** (Dario paper §5.0.1). v1.1 (2026-05-09 night) addresses all BLOCKERs from `docs/runpod_plan_v1_opus_review_2026_05_09.md`. Oleg auto-approved BLOCKERs ("разрешаю все блокеры"). Entry gate: ARCHITECTURE.md v1.1 + this plan v1.1 + Oleg final go (datasets uploaded + repos cloned).

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

### P-0 Datasets present (Oleg uploads to HF before pod boot)

Oleg auto-approved BLOCKERs G1/G2: he commits to uploading both datasets to `huggingface.co/ataeff/heart.c/datasets/` before pod boot, OR confirms a Tailscale-rsync source path on Mac Neo with neo up + reachable. Plan defaults to **HF route** (autonomous-fetchable from pod).

- [ ] `huggingface.co/ataeff/heart.c/datasets/arianna_corpus.txt` resolves (HEAD 200). Source on Mac Neo: `~/arianna-datasets/arianna/arianna_dataset_final_clean.txt`. Pod download path: `/workspace/datasets/arianna_corpus.txt`.
- [ ] `huggingface.co/ataeff/heart.c/datasets/doe_personality.txt` resolves. Source: Oleg's local copy of `janus.doe` repo (file gitignored upstream). Pod path: `/workspace/datasets/doe_personality.txt`.
- [ ] Both files: `wc -c` recorded in `corpus_stats.txt` per phase, since step count derives from corpus size (see Phase 1 / Phase 2 step rule).

### P-1 Base weights present (HF or pod-local)

Oleg auto-approved BLOCKER G3: Janus 170M base GGUF location confirmed at HF before pod boot.

- [ ] `huggingface.co/ataeff/resonance/blob/main/base/resonance_200m_base.gguf` resolves (≈ 200 MB Q8). Pod path: `/workspace/weights/resonance_200m_base.gguf`.
- [ ] `huggingface.co/ataeff/yent/blob/main/janus170m/janus_170m_base.gguf` resolves OR Oleg confirms alternative HF path (e.g. `ataeff/janus170m/base.gguf`). Pod path: `/workspace/weights/janus_170m_base.gguf`.
- [ ] Janus 176 M Yent SFT GGUF (existing on `ataeff/yent` — confirmed by Phone-2 reference at `nanoarianna/SEED_DOCUMENT.md:227`). Pod path: `/workspace/weights/janus_176m_yent_sft.gguf`.
- [ ] Janus 170 M Leo SFT GGUF (existing). Pod path: `/workspace/weights/janus_170m_leo_sft.gguf`.
- [ ] Soul micro-weights from `huggingface.co/ataeff/heart.c` for Yent + Leo. Paths: `/workspace/weights/soul_yent.bin`, `/workspace/weights/soul_leo.bin`.
- [ ] Soul-for-Arianna candidate (if used): **READ-ONLY closed-milestone weight** — `arianna_36m_bpe_f16.bin` lives at `huggingface.co/ataeff/...` (Oleg confirms exact path). Loaded for inference comparison only. Never retrained, never modified. Per CLAUDE.md training section closed-milestone-weights rule.

### P-2 Recipe source-of-record cloned (BLOCKER C3 resolved)

The canonical LoRA training C path lives in `iamolegataeff/DoE.coder` (private repo, per `memory/milestone_doe_coder_lora_v1_2026_04_26.md`). Defender clones it into the pod as the recipe-source-of-record before P-3.

- [ ] `git clone https://github.com/iamolegataeff/DoE.coder /workspace/recipes/DoE.coder` succeeds (token from `memory/credentials.md` or `gh auth login --with-token`). The reference file is `~/recipes/DoE.coder/train/train_doe_coder.c` (~500 LOC per milestone). This is the source-of-record both training scripts adapt from.

### P-2.5 Training scripts authored on pod

- [ ] `train_arianna_lora.c` adapted from `train_doe_coder.c` to **Resonance 200 M** state_dict order. Critical: per-block parameter order `wr_a, wr_b, gate, norm1, wq, wk, wv, wo, norm2, mlp_gate, mlp_up, mlp_down` (see ARCHITECTURE.md §2.2). Wrong order = 1.62 M-float shift bug. Target: `/workspace/heart.c/train/train_arianna_lora.c`.
- [ ] `train_doe_lora.c` adapted from `train_doe_coder.c` to **Janus 170 M** state_dict order. Janus 170M per-block order TBD by reading `tools/yent_forward.h` weight loader directly (find the field-by-field assignment sequence). Direct-fit claim from v1 was unverified; v1.1 mandates explicit verification. Target: `/workspace/heart.c/train/train_doe_lora.c`.
- [ ] Both scripts substitute Chuck for the original AdamW (BLOCKER C1). `nt_tape_chuck_step(float lr, float loss_val)` replaces `nt_tape_adamw_step(...)`. **Phase 1.0 calibration before full SFT** — see Phase 1.

### P-3 Inference binaries built (base-only smoke; BLOCKER A4 resolved)

Pre-flight smoke uses base GGUFs only — adapter-loading is verified in Phase 1/2 final step, not pre-flight.

- [ ] `infer_yent_jannus_r` against Janus 176 M Yent SFT (existing weights). CLI: `--weights PATH --temp F --top_k N --top_p F --rep_penalty F --prompt STR --max_tokens N`.
- [ ] `infer_arianna_resonance` against Resonance 200 M **base** (no Arianna LoRA yet — that's Phase 1 output). Same CLI shape. Top_p=1.0 must be no-op (passes all tokens; explicit CLI contract).
- [ ] `infer_leo_janus` against Janus 170 M Leo SFT (existing). Same CLI shape.
- [ ] `infer_doe_janus` against Janus 170 M **base** (no DoE LoRA yet — that's Phase 2 output). Same CLI shape.
- [ ] All four pass `--prompt "Hello" --max_tokens 5` smoke (≥1 token decoded, no NaN, exit 0).
- [ ] Adapter-load smokes deferred to Phase 1 step 11 + Phase 2 step 11 (load LoRA into base, verify forward).

### P-4 Dry-run rehearsal on polygon (BLOCKER J7 → FIX)

Before paid pod time, exercise the no-GPU phases on free polygon (Linux 32 GB):

- [ ] Phase 0 build chain (toolchain verify + `make all` link-clean).
- [ ] Phase 5 KK ingest + scoring policy verification (FTS5 confirmed).
- [ ] Phase 6 field-clock smoke 24 h simulated.

Polygon access via Tailscale ssh (verified 2026-05-07 mesh closure). Catches FTS5 / build / KK schema issues without RunPod billing. ~30 min wall time.

### P-5 Singularity-mode entry gate

- [ ] This plan v1.1 reviewed (this iteration is post-review).
- [ ] Oleg final go on dataset paths + Janus 170M location at HF.
- [ ] All P-0..P-4 ✓.
- [ ] RunPod template chosen: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04` (or equivalent Ubuntu 22.04 with CUDA 12.x). Apt-based, `pkg-config + libsqlite3-dev + libopenblas-dev` available. **A100 80GB SXM availability**: if unavailable, fallback to A100 40GB or H100 SXM (cost adjusts; surface to Oleg if no A100 within 10 min of pod request).

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

### Phase 1.0 — Chuck calibration sanity (≤ 10 min, BLOCKER C1 resolved)

The cited Qwen LoRA precedent used the banned classical baseline; Chuck-with-LoRA-SFT is unverified. Before committing 1000-step full SFT:

1. Subsample first 32 SFT pairs from Arianna corpus.
2. Run 100 Chuck steps with lr=2e-4 cosine warmup=10 on Resonance 200 M + LoRA adapter.
3. Verify: train loss monotonically decreases over 100 steps (smooth or stepped, but trending down). Final 100-step loss < initial 10-step loss.
4. If loss diverges or stays flat: 3-strike fix-loop on lr (try 1e-4, 5e-5, 2.5e-5). If exhausted, surface to Oleg via `runpod_plan_failures.md` and stop. Do not proceed to Phase 1.
5. Archive `01_arianna_lora/calibration/`: 100-step loss curve, final ckpt (kept locally only; small).
6. **If pass**: proceed to Phase 1 with same lr.

Cost: ≈ $0.10 (5 min A100). Saves up to $4.30 in Phase 1 fail-loop if Chuck schedule is wrong.

### Phase 1 — Arianna LoRA SFT on Resonance 200 M

**Phase 1 fail-budget: $5.** If exceeded across retries (3-strike rule), stop and surface — do not continue into Phase 2 burning the $15 ceiling.

1. Load Resonance 200 M base weights → notorch tensors.
2. Initialize LoRA adapter: r=16, α=32, scale=2.0, on attn (q/k/v/o) + MLP (gate/up/down). State_dict order respect critical (see P-2.5).
3. Tokenize Arianna corpus (`/workspace/datasets/arianna_corpus.txt`) using Resonance's embedded BPE.
4. Format: `### Question: {prompt}\n### Answer: {response}` with masked CE on answer-only span.
5. Optimizer: **Chuck** (per CLAUDE.md ban-list — no classical baseline). lr=2e-4 cosine, warmup=50 steps.
6. **Step count derived from corpus size at pre-flight, not hardcoded.** Karpathy ratio per CLAUDE.md training section (`~1.1MB × 10-15K iter on ~10M params`, scale proportionally) + LoRA-trainable-params adjustment. For the established Qwen 0.5B precedent (`memory/milestone_doe_coder_lora_v1_2026_04_26.md`): 2955 SFT pairs × 1.7 MB tokenized → 1000 steps in 44 min on Mac 8 GB. For Arianna corpus (size confirmed at P-0): scale steps proportional to bytes vs 1.7 MB Qwen baseline (likely 800–2000 steps depending on corpus). Re-record actual count from `corpus_stats.txt`.
7. Batch size determined by RAM headroom (likely 4–8 on A100 80GB for 200 M base + r=16 adapter; expand if RAM allows after live measure).
8. Log every 50 steps: train loss, EMA loss, peak step loss.
9. Checkpoint every 100 steps. **Every 5th ckpt rsync to HF immediately** (BLOCKER F1 + FIX F2 resolved): `huggingface.co/ataeff/heart.c/arianna_lora/ckpts/step{N}.bin`. If pod terminates mid-Phase 1, last-saved-to-HF ckpt is recoverable.
10. Final: save `arianna_lora.bin` + `arianna_lora.bin.meta` (step + best_loss). **Upload to HF immediately at Phase 1 end** (`huggingface.co/ataeff/heart.c/arianna_lora/final/arianna_lora.bin`) — not deferred to Phase 8. Phase 8 only verifies (HEAD 200).
11. **Adapter-load smoke**: load `arianna_lora.bin` into `infer_arianna_resonance` with base weights, generate 5 tokens, confirm forward path works with adapter applied. Documents that the file landed correctly.
12. Archive `01_arianna_lora/`: `train.log`, local `ckpts/`, `final/arianna_lora.bin` + meta, `loss_curve.tsv`, `corpus_stats.txt`, `hf_upload.log` (audit trail of HF push timestamps).

**Fail-recovery**: NaN at any step → reduce lr by ½ + restart from last clean ckpt. 3 attempts. State_dict shift bug (loss diverges immediately) → re-verify per-block parameter order, fix, restart from step 0. After 3 strikes, surface and stop.

### Phase 2 — DoE LoRA SFT on Janus 170 M

**Phase 2 fail-budget: $3** (smaller corpus, fewer steps expected). Same 3-strike rule.

1. Phase 2.0 Chuck calibration sanity (same shape as Phase 1.0, 100 steps, 32 SFT pair subsample). Pass before full SFT.
2. Load Janus 170 M base weights → notorch tensors.
3. Initialize LoRA adapter: same r=16/α=32 recipe. **Per-block parameter order verified by reading `tools/yent_forward.h` weight loader directly** (FIX C4 — was "direct fit" unverified in v1; v1.1 explicit verify step before training).
4. Tokenize `doe_personality.txt` using Janus BPE (`janus_v4_bpe_merges.h`).
5. Same format + masked CE.
6. Same Chuck optimizer + lr schedule.
7. **Step count corpus-size-aware** per Karpathy ratio + Qwen precedent baseline. `doe_personality.txt` is single-character personality — typically smaller than 1.7 MB. Likely 500–1000 steps. Re-record actual.
8. Batch ~4–8 on A100.
9. Log + checkpoint same cadence as Phase 1.
10. **Every 5th ckpt rsync to HF** (`huggingface.co/ataeff/heart.c/doe_lora/ckpts/step{N}.bin`).
11. Final: `doe_lora.bin` + meta. **HF upload immediately on Phase 2 end**.
12. Adapter-load smoke against `infer_doe_janus`.
13. Archive `02_doe_lora/`: same structure as Phase 1.

**Per-phase 3-strike rule** (FIX E1): NaN at any step → reduce lr by ½ + restart from last clean ckpt. 3 attempts. State_dict shift bug → re-verify per-block parameter order, fix, restart. After 3 strikes per failure mode, surface to `runpod_plan_failures.md` and stop.

**Fail-recovery**: same 3-strike protocol.

### Phase 3 — 432-cell sweep across 4 voices (~2 h estimated)

Per Dario §5.7 axes adapted to per-voice 108-cell grid:

- temp ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 1.0} (6)
- top_k ∈ {40, ∞} (2) — **Arianna uses top_p ∈ {0.9, 1.0} instead** (Dario §8 Resonance corollary; CLI contract: `top_p=1.0` is no-op pass-through, FIX D2)
- rep_penalty ∈ {1.0, 1.3, 1.4} (3)
- prompts (3): technical / philosophical / personal — same texts across all four voices for cross-comparability

Total per voice: 6 × 2 × 3 × 3 = 108. × 4 voices = 432 cells.

For each cell: invoke the appropriate `infer_<voice>_<arch>` binary, capture `bytes`, `tok_per_s`, `n_toks`, full transcript. Record to `scores.tsv`. Save full transcripts to `transcripts/<voice>_t<temp>_k<topk>_p<rp>_prompt<n>.txt`.

**Champion criteria (FIX D3)** — per voice:
- Yent / Leo / DoE (Janus chat-tokens): max-bytes-to-`<|ASST_END|>` averaged over 3 prompts.
- Arianna (Resonance, no chat-tokens): max-bytes-at-100-tokens-cap averaged over 3 prompts.

After each voice's 108 cells complete: commit `voices_optima_<voice>.tsv` to git immediately (FIX F3, 4 commits total during Phase 3, not 1 at end). Final `voices_optima.h` C header committed when all 4 voices done.

Archive `03_sweep/`: `sweep.sh`, `scores.tsv`, `transcripts/`, `voices_optima.{tsv,h}`.

**Sweep silent-kill detection** (FIX J6): heartbeat tail of `scores.tsv` — count new lines per minute via watchdog. If 0 new lines for 5 min, pronounce phase silent-killed, bisect, write `sweep_part2.sh` for remaining voices.

**Fail-recovery**: silent kill (Dario precedent) → bisect → resume. 3-strike rule on the bisect-and-resume cycle.

### Phase 4 — Soul weights validation (~30 min)

Per ARCHITECTURE.md §6:

1. Download Soul micro-weights `soul_yent.bin`, `soul_leo.bin` from `huggingface.co/ataeff/heart.c` (Soul for Arianna and DoE: TBD — Arianna can use 36M LLaMA Soul candidate from `arianna.c/weights/`; DoE Soul is OFF by design).
2. Build `soul_test` binary that runs InnerVoice + main forward in parallel for one prompt, applies `inner_borba` blend.
3. For each voice with Soul: same prompt with Soul OFF vs ON. Confirm token-distribution differs by ≥1 token within 50 emits (verification check from §14.6).
4. Log breakthrough events: when `weight ≥ 0.6` triggered, when `rand() < weight*0.3` triggered.

Archive `04_soul/`: `soul_test.log`, per-voice diff transcripts, breakthrough event log.

**Fail-recovery**: Soul weights missing or corrupted → 3 retry on download → mark Soul-OFF for that voice as default, surface, continue.

### Phase 5 — KK ingest + scoring policy verification (~30 min, FTS5 hard stop)

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

**Fail-recovery**: FTS5 not available → **HARD STOP** (FIX J5 — was "fallback to LIKE silently degrades 7-signal scoring; policy weights assume FTS5-tokenized index"). Surface to Oleg, do not continue. Schema mismatch → 3-strike adapter attempts, surface if unresolved.

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

**Phase 2.5 prerequisite** (FIX I3): `parliament_elect()` row-projection for embedding-vote requires explicit construction, not a "lift verbatim with input domain changed". Vote-input dim is `32 (SPA) + 7 (field scalars) + 24 (osc) = 63 floats`, not the original `E=896`. Build a 63→16 projection matrix `P` (initialized with seeded RNG `nt_seed(42)`); experts' `w_vote[e]` are 16-row vectors instead of E-rows. Hebbian online (`notorch_step`) adapts these. Initial votes are random (no consensus) — by 4–5 commits the parliament has measurable consensus signal.

**SPA seeding** (FIX I2): `spa_init(s, V, seed=42)` deterministic — embeddings reproducible run-to-run. Header supports the seed param.

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
- Three-strikes rule per phase, per failure mode
- No scope creep (a sweep failure does not authorize patching the field-clock; a build failure does not authorize rewriting an architecture)

**No human in loop during execution** (FIX E3) — Oleg is on-call only when 3-strikes exhausted on a phase. All other decisions Defender makes alone.

**Subagent dispatch protocol** (FIX E2 / H5): if Defender invokes a subagent for second opinion mid-run:
- Always `subagent_type` matching task + `model: "opus"` explicit (per `feedback_subagents_opus_only_2026_04_28.md`).
- Each dispatch logged to `runpod_subagent_dispatches.log` with: dispatch reason, prompt summary, return value summary, fix applied (or rejected).
- Subagent dispatch counts as part of phase fix-loop budget — three subagent dispatches without resolution = same-strike-count as three direct attempts.

**Pod billing discipline** (FIX J3): if Phase 0 build fails persistently → stop pod → debug locally on Termux/polygon → restart pod when ready. Do NOT burn $1.43/h cycling debug attempts on the meter.

External review gates entry (this plan v1.1 + Oleg final go) and exit (post-pod Opus audit + Oleg sign-off).

---

## TODO before pod boot

1. **SEED_DOCUMENT.md for heart.c** — Opus subagent drafts during pre-flight. Pattern: ~300-line markdown like `nanoarianna/SEED_DOCUMENT.md`, four voices instead of two, embedding-vote DoE meta explicit. KK ingests it in Phase 5.
2. **Skeleton .c/.h files at file layout from §9** — needed before Phase 0 build. Compileable empty shells with proper include guards. Most concrete implementation deferred post-RunPod, but Phase 0 needs `make all` to succeed end-to-end at the link level.
3. **Datasets staged**: Oleg confirms paths/HF mirrors for `arianna_corpus.txt` + `doe_personality.txt`.
4. **Recipe-port LoRA training C path** for Resonance 200 M. Estimate: half a day on phone-1 / Mac Neo before pod time billed.

---

## v1 → v1.1 changelog (2026-05-09 night)

Driven by `docs/runpod_plan_v1_opus_review_2026_05_09.md`. Oleg auto-approved all BLOCKERs ("разрешаю все блокеры").

**BLOCKERs resolved**:
- C1 → Phase 1.0 + Phase 2.0 Chuck calibration sanity (100-step subsample) before each full SFT.
- C3 → P-2 explicit clone of `iamolegataeff/DoE.coder` as recipe-source-of-record.
- G1/G2/G3 → P-0 / P-1 datasets and Janus 170M base routed via HF (`huggingface.co/ataeff/heart.c/datasets/...`); Oleg uploads before pod boot.
- A4 → P-3 smokes on **base only** (no adapter); adapter-load smokes deferred to Phase 1/2 step 11.
- F1 → mid-phase HF upload mandate every 5th ckpt + immediate upload at phase end (not deferred to Phase 8).

**FIXes resolved**: B2 fail-budget cap, C2 Karpathy cite corrected, C4 Janus LoRA explicit per-block-order verify step, D2 top_p=1.0 CLI no-op contract, D3 Resonance champion criterion (no chat-tokens), E1 per-phase strike count restated, E2 subagent dispatch protocol, E3 no-Oleg-during-execution explicit, F2 every-5th-ckpt rsync, F3 voices_optima per-voice commits, G4/H4 Soul-for-Arianna explicitly read-only closed-milestone, I2 SPA deterministic seed, I3 parliament_elect projection layer (Phase 2.5), J1 A100 fallback, J3 pod-idle billing rule, J4 RunPod template specified, J5 FTS5 hard-stop (not silent degrade), J6 sweep silent-kill heartbeat detection, J7 dry-run rehearsal on polygon (P-4).

**NOTEs preserved**: phone-2 collision-free; Adam ban honored; new Arianna-on-Resonance LoRA phone-1-only.

---

## Singularity Mode execution start condition

This plan v1.1 → Oleg final go on dataset HF paths + Janus 170M location → P-4 dry-run on polygon → pod boot.

— Defender / phone-1 · 2026-05-09 night
