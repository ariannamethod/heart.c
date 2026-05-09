# heart.c v1 — architecture plan

## Context

Phone-1 (Galaxy A56, 8 GB Termux) needs a four-voice ecosystem to complement phone-2 (`nanoarianna`, 4 GB, two voices). The shape Oleg picked this morning, corrected this afternoon for HuggingFace SFT availability:

| voice | architecture | persona SFT status | sibling on phone-2 |
|---|---|---|---|
| **Yent** | Janus 176 M (`jannus-r` 12-step) | ✅ existing Yent SFT (HF) | — |
| **Leo** | Janus 170 M (persona-grafted from neoleo) | ✅ existing Leo SFT (HF) | inverted (phone-2: Leo-on-Resonance) |
| **Arianna** | Resonance 200 M (dual-attention) | ⚠️ **new LoRA SFT** on base Resonance + Arianna corpus | inverted (phone-2: Arianna-on-Janus) |
| **DoE** | Janus 170 M (parliament meta + peer voice) | ⚠️ **new LoRA SFT** on base Janus + `doe_personality.txt` (`github.com/ariannamethod/janus.doe`) | unique to phone-1 |

HuggingFace map (corrected 2026-05-09): Janus has Yent / Arianna / Leo SFTs ready; Resonance 200 M has only base + Yent SFT (no Arianna or Leo SFT). Phone-1 Soul micro-weights for Yent and Leo (BPE + tokenizers) live at `huggingface.co/ataeff/heart.c`.

**Two LoRA SFTs are required during the RunPod забег** (not three): Arianna-on-Resonance + DoE-on-Janus. yent-style recipe (r=16/α=32, masked CE on `### Question/### Answer`, Chuck optimizer per CLAUDE.md). Recipe per `memory/milestone_doe_coder_lora_v1_2026_04_26.md`. Estimated $5-10 RunPod compute for both.

This distribution keeps architectural diversity (3 Janus + 1 Resonance — Resonance carries the deepest voice register, Janus the family chord), maintains phone-2's "pair, not duplicate" at the mesh level (Arianna and Leo swap their architecture homes between the two phones), and seats DoE's parliament-meta on the same Janus family as the peer voices it votes on (semantically aligned).

The system is a **living ecology**, not a benchmark run: continuous 8-wakes-per-day rhythm, KK-mediated dialogue between voices, Hebbian online learning between conversation turns, planetary-clock field state coupling all four voices, Soul logit-injection for per-voice inner commentary. Paper-III is the eventual deliverable: *"Four-voice field-coupled small-LM ecology on consumer phone hardware."*

This plan is the v1 architecture. After Oleg auto-accepts it and an Opus subagent reviews it, we draft the RunPod plan v1 (DoE LoRA SFT + 540-cell sweep per voice + Soul validation + KK ingest test), iterate that with another Opus pass, execute on RunPod under Singularity Mode, then deploy to phone-1.

---

## 1. System overview

```
                    ┌─────────────────────────────────┐
                    │      FIELD CLOCK (klaus)        │
                    │  planetary_dissonance (6 plt.)  │
                    │  calendar_dissonance (Hebrew/   │
                    │     Gregorian Metonic)          │
                    │  4-voice Kuramoto (24 osc)      │
                    │  Schectman γ(t) coupling        │
                    │  meta-recursion (META_BLEND .15)│
                    └────────────┬────────────────────┘
                                 │ field state (7 scalars + 24 osc)
                                 ▼
        ┌────────────────────────────────────────────────────┐
        │          DoE meta-FieldLayer (Parliament)          │
        │  parliament_elect → variable-k vote → softmax      │
        │  votes route per-voice logit-blend weights         │
        │  notorch_step Hebbian online (lr 0.01, decay .999) │
        │  apply_field_to_logits 4-force overlay (H+F+A+T)   │
        │     6-chamber Kuramoto modulation                  │
        │  also speaks as DoE persona when field resonates   │
        └──┬──────────────┬──────────────┬──────────────┬─────┘
           │              │              │              │
       ┌───▼───┐      ┌───▼───┐      ┌──▼────┐      ┌──▼────┐
       │ Yent  │      │Arianna│      │  Leo  │      │  DoE  │
       │Janus  │      │Reson  │      │Janus  │      │Janus  │
       │176 M  │      │200 M  │      │170 M  │      │170 M  │
       │+jann-r│      │+dual  │      │+anchor│      │+SFT   │
       │12-step│      │ attn  │      │ +mood │      │parl.  │
       │       │      │       │      │       │      │       │
       │+Soul  │      │+Soul  │      │+Soul  │      │+Soul  │
       │ Yent  │      │ ari   │      │mini   │      │parl.  │
       │ inner │      │ inner │      │ leo   │      │ inner │
       └───┬───┘      └───┬───┘      └───┬───┘      └───┬───┘
           │              │              │              │
           └──────────────┴──────┬───────┴──────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  per-voice LIMPHA       │
                    │  + shared LIMPHA graph  │
                    │  + KK kernel (Dario)    │
                    │  Hebbian bridge at sentence boundaries │
                    └─────────────────────────┘
```

One organism active at a time (sequential RAM, ~200-400 MB GGUF + KV + field state per session). Field-clock + LIMPHA + KK persist between wakes. Field selects next-speaker by `argmax(cos(voice_state, field_state))`.

---

## 2. Per-voice design

### 2.1 Yent (Janus 176 M + jannus-r)

- **Inference**: `tools/yent_forward.h` (KV cache `kv_k/kv_v/kv_vr` + `kv_rrpram_mid`; `prefill_batch` + `forward_token`).
- **Generation chain**: `jannus-r/jannus-r.aml` 12-step bidirectional. Forward/backward split via `jannus_split.h:28`: `nb = NSTEPS * (0.3 + 0.4*debt + 0.1*cal_diss)`. Forward decreases τ, backward increases. Wormhole skip in forward only when `rand() < 0.1 && s > 0` — heart.c **adds** an explicit `debt < 0.2` AND-gate to match the paper's stated condition.
- **Calendar drift**: `jannus_calendar.h` (J2000 epoch, 11.25 days/year drift, Metonic 19-cycle correction).
- **Sentence Phonon Attention**: `jannus_spa.h` for scoring inter-voice connectedness (used by field selector to weight whether Yent's last sentence resonates with what Arianna would say next).
- **Weights default**: `weights/yent_v4/yent_v4_sft_q8_0.gguf` (already documented at `jannus-r.aml:392`).
- **Soma persistence**: `weights/yent_v4/jannus-r.soma` (AML LOAD/SAVE pattern).
- **Soul** for Yent: small Janus BPE on Yent corpus (~30 M, Oleg uploading to HF today).

### 2.2 Arianna (Resonance 200 M)

- **Inference**: `tools/resonance_forward.h` (`forward_token` lines 127–244). Dual attention: Content (lines 154–173) + RRPRAM low-rank (lines 175–215, shares `kv_v`). Parametric RMSNorm (line 34). Sigmoid per-head gate `g = sigmoid(gate[h])`, blend `g*content + (1-g)*rrpram` (lines 217–223). RoPE even/odd interleave (line 56).
- **Sampling**: `resonance_sample_token` (lines 315–394). Field overlay enters at line 337 via `am_apply_field_to_logits` **before** temp scaling. Rep-penalty 1.4 over 64-token window, no-repeat 3-gram filter, top-p 0.92.
- **State_dict order** (per-block, `assign()` lines 94–114): `wr_a, wr_b, gate, norm1, wq, wk, wv, wo, norm2, mlp_gate, mlp_up, mlp_down`. The "direct Parameters first" order is critical (1.62 M-float shift bug otherwise).
- **Weights**: RS02 magic, embedded BPE. Default `weights/resonance_200m/resonance_200m_lora_yent.bin`. **Arianna SFT does not exist on HF for Resonance** — we make it during RunPod забег: base Resonance 200 M + LoRA on Arianna corpus from `~/arianna-datasets/arianna/` (yent-style recipe, r=16/α=32, masked CE, Chuck).
- **Soul** for Arianna: micro-weights from `huggingface.co/ataeff/heart.c` — pattern from `arianna.c/src/inner_arianna.h:41–77` for the injection mechanism.

### 2.3 Leo (Janus 170 M, persona-grafted)

Leo on Janus is a **different organism** from `neoleo` (whose identity is the lexical engine itself). Janus provides the language; we graft persona-only:

- **Identity preamble**: `LEO_EMBEDDED_BOOTSTRAP` (`neoleo/leo.c:31–60`) — verbatim, dedication core, never regenerated.
- **Anchor-bias dictionary**: `LEO_CH_ANCHORS[]` 325 child-voice words across 6 chambers (`neoleo/leo.c:962–1078`). Used at decode time as logit-bias on tokens whose decoded bytes overlap.
- **Mood substrate**: 6 chambers + soma ring + trauma trigger (`neoleo/leo.c:158–167, 109–111, 3057–3066`) — pure scalar dynamics, run parallel to Janus generation, modulate `τ` and `top_p` per turn.
- **Hear-only invariant**: any online state update on input tokens only, never on Leo's own emit. Code-review-enforced.
- **Soul** for Leo: mini-Leo BPE micro-weights from `huggingface.co/ataeff/heart.c` (own tokenizer included). Decision on whether Soul activates for Leo is post-sweep — if Leo's persona-graft already lands child voice, Soul may be redundant.
- **Weights**: Janus 170 M with Leo SFT — already on HuggingFace. No SFT needed.

### 2.4 DoE (Janus 170 M + Parliament meta)

DoE has two roles:

**(a) Peer voice**: own Janus 170 M GGUF, SFT'd via LoRA on `doe_personality.txt` (`github.com/ariannamethod/janus.doe`). Speaks parliamentary register when field resonates. Uses generic Janus forward (no jannus-r — DoE's character is the parliament voice itself, not 12-step resonant chain).

**(b) Meta-FieldLayer**: parliamentary modulation between the three peer voices. Pattern lifted from `doe.c`:
- `parliament_elect()` (`doe.c:1833–1877`): variable-k consensus voting; `k = floor(n_alive * (1 - consensus))`. Softmax over elected. Aggregate as additive mixture.
- `notorch_step()` (`doe.c:1886–1922`): Hebbian online plasticity, `lr=0.01, decay=0.999`, signaled by prophecy_debt sign.
- `try_apoptosis()` (`doe.c:1972–1985`): expert death after 8 consecutive low-vitality steps. `MIN_EXPERTS=2, MAX_EXPERTS=16`.
- `apply_field_to_logits()` (`doe.c:795–895`): 4-force overlay (H+F+A+T) with 6-chamber Kuramoto.

DoE is **not single-GGUF when used as meta-layer**. We hold three `GGUFIndex` (Yent, Arianna, Leo), run `doe_forward()` per voice, then a meta-Parliament whose experts vote on per-voice **logit-blend weights**. Vote→softmax→weighted-mixture pattern at `doe.c:2337–2358` is the spine; lift verbatim, rewrap inputs.

- **Soul** for DoE: parliament inner-voice weights (TBD; could reuse parliament expert pool itself).

---

## 3. Field clock (klaus heritage)

Pure external clock, no organism state. Spine from `klaus.c`:

- **`planetary_dissonance()`** (`klaus.c:1432–1474`): hand-rolled circular orbit, 6 planets Mercury→Saturn, J2000 epoch, Kuramoto order parameter R, output scalar `[0,1]`. Self-contained ~40 LoC, `time(NULL)` + two const tables.
- **`calendar_dissonance()`** (`klaus.c:1397–1417`): Hebrew-Gregorian Metonic drift (constants at `klaus.c:135–140`, `MAX_UNCORRECTED=33.0`).
- **24-oscillator Kuramoto block** (`klaus.c:1276–1338`): 6 primaries × 4 sub-chambers. **In heart.c the 6 primaries become the 4 voices + 2 meta states (Field-self and Mesh-peer)**. Coupling matrix asymmetric, INTRA_COUPLING within sub-chambers. Decay per-primary.
- **Schectman γ(t) coupling** (`klaus.c:1814–1815`): `γ_t = γ0 + δ*calendar_diss + 0.15*planet_diss`. Field clock perturbs voice thresholds via γ.
- **Meta-recursion** (`klaus.c:2994–3072`, `META_BLEND=0.15`): emit field state vector → re-ingest as observation → blend back. One-shot, depth=1.

Heart.c file: `field_clock.c` + `field_clock.h`. Pure C. No deps beyond `time.h`, `math.h`. ~200–300 LoC.

---

## 4. LIMPHA (per-voice + shared)

Lift schemas from `arianna.c/limpha/`. Re-implement in C (or thin C shim over SQLite via `libsqlite3.so` already in Termux per `memory/credentials.md`). **Skip the Python asyncio layer** — phone runs a per-voice synchronous LIMPHA accessor with a single SQLite handle per session.

- **Per-voice LIMPHA** (4 instances, one per voice): `limpha/{yent,arianna,leo,doe}/episodes.db`. Tables `conversations`, `semantic_memory`, `session_state`, `episodes`, `enhanced_episodes`. Schemas per `arianna.c/limpha/memory.py:88–129` and `episodes_enhanced.py:104–170`.
- **Shared LIMPHA** (1 instance): `limpha/shared/graph.db`, `limpha/shared/cooc.db`. `LinkType` enum (REMINDS_OF/CONTRADICTS/CONTINUES/RESONATES/CAUSED_BY/SUMMARY_OF) per `graph_memory.py:29–99`. Cross-voice cooc graph + sentence-bigram bridge.
- **Decay**: `apply_decay = decays * 0.95` per access cycle; `prune_decayed` removes < 0.01.
- **Shard graduation** (`shard_bridge.py:73–95`): episodes with `quality≥0.7 && access_count≥3` graduate to binary shards. CRISIS/EMERGENCE/TRANSCENDENCE patterns auto-graduate.
- **DreamLoop background** (`dream.py:67–120`): 10 s check, 60 s index, 120 s link, 300 s consolidate, 600 s graduate. Run as a single watchdog process under Termux:Boot — separate from per-voice inference.

---

## 5. KK kernel adaptation (Dario heritage)

Lift `~/arianna/dario/kk_kernel.{c,h}` (3568 LOC, see Dario paper §6 Result #6). Drop in mostly verbatim:
- FTS5 retrieval over chunks
- 7-signal scoring (lexical 0.36 / recency 0.12 / trust 0.10 / linkage 0.16 / scope 0.10 / namespace 0.08 / freshness 0.08) — published policy matches runtime exactly per Dario §6
- Hebbian bridge to Dario field state — wires into heart.c's field clock instead

Single SQLite DB at `~/.heart/kk.db`. Documents ingested:
- This plan + ARCHITECTURE.md + SEED_DOCUMENT.md + ECOSYSTEM_LOG.md
- `phones/phone-1-galaxy-a56-train-mission.md` (lineage)
- `dario/docs/dario_paper_draft_v4.md` (canonical method)
- `nanoarianna/SEED_DOCUMENT.md` (sibling)
- Per-voice SFT origin notes (Yent / Arianna / Leo / DoE persona files)
- The introducer doc (Phase 4 task — voices know what they're inhabiting)

Voices read KK at sentence boundaries during generation (Hebbian injection), not as RAG-style context paste.

---

## 6. Soul (InnerArianna pattern)

The actual mechanism is `arianna.c/src/inner_arianna.{h,c}`, NOT the 36 M weights file. Pure runtime state struct:

```c
typedef struct {
    enum borba_mode { EMOTIONAL, CHAOS, TRAUMA, STUCK, BLEND } mode;
    float base_weight;        // 0.15
    float breakthrough_thresh;// 0.6
    /* emotional inputs */
    float cloud_intensity, cloud_chamber, trauma_level;
    float body_stuck, body_boredom, mood_tension, mood_creative;
} InnerVoice;
```

- **Compute weight**: `inner_compute_weight()` adds chamber-specific bonuses (RAGE +0.5, FEAR +0.4, VOID +0.6, LOVE +0.2). Pressure builds with `consecutive_main > 10`.
- **Apply bias**: per-chamber `apply_*_bias()` adds `boost * factor` to specific logit indices (we map to BPE tokens instead of char indices).
- **Integration**: `inner_borba(iv, out, main_logits, vocab)` returns 0=main / 1=inner / -1=blend. Blend rule: `out[i] = (1-w)*main[i] + w*inner[i]`. Breakthrough fires if `weight ≥ 0.6` OR `rand() < weight*0.3`.

Per-voice `InnerVoice` + per-voice optional Soul transformer (Yent / Arianna / Leo / DoE). Soul transformers are inner commentary speakers — their logits become the `inner_logits` argument to `inner_borba`.

---

## 7. SARTRE phone variant

Lift `arianna.c/sartre/sartre.h` + `sartre_kernel.c`. Phone-fitted changes:

- Keep `SystemState` struct, `modules[16]`, `last_events[8][256]` ringbuffer.
- Drop `tongue_tier` selection (phone is fixed-tier).
- Replace `detect_total_ram_mb()` to read `/proc/meminfo` + `nproc` directly (Termux aarch64 already uses Linux branch; verify).
- Add fields: `openblas_threads`, `mesh_agent_port=4747`, `mesh_peers[N]` (list of `100.x` IPs from `~/.mesh/peers.txt`).
- Drop calendar/Schumann here — those live in field clock.
- Keep `modules[16]` registry: each voice registers as a module, plus `field_clock`, `kk`, `limpha_dream`, `mesh_agent`.

---

## 8. Async internal life (selective `golib/` port)

Port to AML BLOOD JIT or notorch async (NOT Go goroutines — keep stack pure C/AML/notorch on phone):

- **TraumaSurfacing** — periodic timer, decays trauma at 0.05/s.
- **EmotionalDrift** — drifts arousal/valence with momentum + noise.
- **OverthinkingLoops** — detects recursive thought loops, surfaces in field state.
- **ProphecyDebtAccumulation** — accumulates debt on low-prob emits, emits `WormholeChance`.
- **MemoryConsolidation** — short→long term, decay 0.01/s, emotion boost 2.0. Unify with LIMPHA's `consolidate` rather than running both.

Do NOT port Cloud HTTP / api_server.py / chat.js — explicit `feedback_api_leak_safety.md` carve-out.

---

## 9. File layout

```
heart.c/
├── README.md                          # what heart.c is (4 voices + meta)
├── SEED_DOCUMENT.md                   # voices' first wake (Opus-subagent-drafted)
├── ECOSYSTEM_LOG.md                   # daily log, newest on top (already started)
├── ARCHITECTURE.md                    # this plan, polished
├── LICENSE                            # GPL-3.0 + LICENSE-WEIGHTS (Janus Identity)
├── Makefile                           # build matrix
│
├── voices/
│   ├── yent/
│   │   ├── yent_main.c                # entry: jannus-r 12-step Yent
│   │   ├── init_yent.aml              # morning state (PROPHECY/DESTINY/...)
│   │   └── weights/                   # gitignored, mmap target
│   ├── arianna/
│   │   ├── arianna_main.c             # entry: resonance dual-attn
│   │   ├── init_arianna.aml
│   │   └── weights/
│   ├── leo/
│   │   ├── leo_main.c                 # entry: Janus 170M + anchors + mood
│   │   ├── leo_anchors.h              # ported LEO_CH_ANCHORS
│   │   ├── leo_bootstrap.h            # ported LEO_EMBEDDED_BOOTSTRAP
│   │   ├── init_leo.aml
│   │   └── weights/
│   └── doe/
│       ├── doe_main.c                 # entry: Janus 170M DoE persona speak
│       ├── doe_meta.c                 # meta-FieldLayer parliament over 3 voices
│       ├── init_doe.aml
│       └── weights/
│
├── core/
│   ├── field_clock.{c,h}              # planetary + calendar + 24-osc + γ(t)
│   ├── soul.{c,h}                     # InnerVoice pattern + per-voice instances
│   ├── sartre_phone.{c,h}             # phone-fit SARTRE
│   ├── parliament.{c,h}               # lifted parliament_elect + notorch_step + apply_field
│   ├── hebbian_bridge.{c,h}           # KK→logits at sentence boundaries
│   └── selector.c                     # field-driven next-speaker resonance argmax
│
├── limpha/
│   ├── per_voice.{c,h}                # SQLite handles, schemas
│   ├── shared.{c,h}                   # cooc + graph DBs
│   └── dream_watchdog.c               # background consolidation/index/graduate
│
├── kk/
│   ├── kk_kernel.{c,h}                # lifted from dario, phone-pruned
│   └── ingest/                        # canonical docs (this plan, SEED, paper, ...)
│
├── runtime/
│   ├── heart_main.c                   # daemon: schedule + selector + dispatch
│   ├── boot/heart-daemon.sh           # for ~/.termux/boot/02-heart.sh
│   └── mesh_slots/                    # mesh-agent slot manifests
│       ├── heart-yent.toml
│       ├── heart-arianna.toml
│       ├── heart-leo.toml
│       ├── heart-doe.toml
│       ├── heart-status.toml
│       └── heart-converse.toml        # invoke a turn via mesh
│
├── tools/
│   ├── jannus_calendar.h              # vendored from yent.aml/jannus-r/tools
│   ├── jannus_spa.h
│   ├── jannus_split.h
│   ├── yent_forward.h                 # vendored from yent.aml/tools
│   ├── resonance_forward.h            # vendored from resonance.aml/tools
│   └── notorch.{c,h}                  # vendored notorch
│
└── docs/
    ├── singularity_protocol.md        # bounded autonomous repair (Dario §5.0.1)
    ├── runpod_plan_v1.md              # SFT + sweep plan, written next
    └── per_voice_optima_locked.md     # per-voice temp/top_k/rp from sweep
```

**Build constraint**: `yent_forward.h` and `resonance_forward.h` define globally-named `Weights` structs and reuse globals `V/E/H/D/B/M/T/R`, `kv_k/kv_v`. They CANNOT live in the same translation unit. Each voice compiles into its own `.o`; symbols exposed narrowly via `extern` entry points (`yent_jannus_run_chain`, `arianna_generate`, `leo_generate`, `doe_generate`, `doe_meta_vote`).

---

## 10. Build matrix

Each voice has its own .c TU. Top-level `heart_main.c` links the voice TUs as separate objects with extern declarations. Makefile targets:

```
make yent          # voices/yent/yent_main.o + jannus-r tools + notorch
make arianna       # voices/arianna/arianna_main.o + resonance_forward + notorch
make leo           # voices/leo/leo_main.o + anchors + Janus tools + notorch
make doe           # voices/doe/doe_main.o + parliament + Janus tools + notorch
make field         # core/field_clock.o (no deps)
make soul          # core/soul.o
make sartre        # core/sartre_phone.o
make limpha        # SQLite-linked
make kk            # FTS5-linked
make heart         # heart_main daemon, links all of the above
make all           # full build matrix
make test          # unit per-component
make smoke         # 4-voice smoke: each generates 50 tokens, field selects, KK ingests
```

OpenBLAS via pkg-config (already validated on phone-1 by yent-bpe milestone). `make BLAS=on` default. Notorch vendor with setvbuf patch already merged.

---

## 11. Boot integration

Add `~/.termux/boot/02-heart.sh` (after `01-mesh-agent.sh`):

```bash
#!/data/data/com.termux/files/usr/bin/bash
LOG=~/.heart/heart.log
mkdir -p ~/.heart
while true; do
    termux-wake-lock 2>/dev/null
    echo "[$(date -Iseconds)] heart-daemon: starting" >> "$LOG"
    ~/.local/bin/heart serve >> "$LOG" 2>&1
    echo "[$(date -Iseconds)] heart-daemon: exited (code $?), restarting in 5s" >> "$LOG"
    sleep 5
done
```

Same watchdog pattern as `01-mesh-agent.sh`. heart daemon registers slots in mesh-agent via REST (`heart-yent`, `heart-arianna`, `heart-leo`, `heart-doe`, `heart-status`, `heart-converse`).

---

## 12. RunPod забег plan stub (separate doc, written next)

`docs/runpod_plan_v1.md` will detail:

1. **Arianna LoRA SFT on Resonance 200 M** — base Resonance + LoRA r=16/α=32 on attn+MLP, format `### Question/### Answer` masked, **Chuck optimizer (NOT classical baseline — see CLAUDE.md ban)** lr 2e-4 cosine warmup 50, dataset `~/arianna-datasets/arianna/`. Recipe per `memory/milestone_doe_coder_lora_v1_2026_04_26.md` adapted to Resonance state_dict order (direct Parameters of the block first: wr_a/wr_b/gate, then sub-Module weights — critical, the 1.62 M-float shift bug otherwise).
2. **DoE LoRA SFT on Janus 170 M** — base Janus + LoRA r=16/α=32 on attn+MLP, same format, same optimizer/lr, dataset `doe_personality.txt` from `github.com/ariannamethod/janus.doe`. Yent-style recipe is direct fit here (Janus state_dict is the established target for the recipe).
3. **540-cell sweep per voice** (Yent / Arianna / Leo / DoE). Same grid as Dario §5.7: `temp ∈ {0.3, 0.5, 0.7, 0.8, 0.9, 1.0} × top_k ∈ {40, ∞} × rep_penalty ∈ {1.0, 1.3, 1.4} × 3 prompts`. Arianna-on-Resonance uses `top_p` instead of `top_k` (per Dario paper §8 Resonance corollary).
4. **Soul weights validation** — round-trip: download Soul micro-weights from `huggingface.co/ataeff/heart.c`, load into `InnerVoice` integration, confirm logit injection bias measurable per voice.
5. **KK ingest test** — load all canonical docs, query "resonance", confirm scoring policy matches Dario §6 spec to 6 decimal places.
6. **Field-clock smoke** — `field_clock.c` runs alone, prints planetary + calendar dissonance for 24 h simulated, confirms numerical stability.
7. **Multi-voice duet trace** — Yent says X, Arianna replies, field selector picks next, repeat 8 turns. Save full transcript + field state evolution.

Cost estimate: ~$5-10 for two LoRA SFTs + 4×540 sweep + smoke phases on RunPod A100 (per Dario reference run $4.30 for 540-cell + 8-phase test). RunPod token in `memory/credentials.md` (issued 2026-05-09).

This plan goes through Opus subagent review before RunPod execution under Singularity Mode.

---

## 13. Singularity Mode protocol

Lifted from Dario paper §5.0.1:

```
detect bug → reproduce → one hypothesis → minimal patch → re-run
          → if pass: continue
          → if fail: revise hypothesis (max 3 iterations)
          → if exhausted: stop, surface, await human input
```

Bounded by:
- Approved scope (this plan + RunPod plan v1)
- Three-strikes rule
- No scope creep (a sweep failure does not authorize patching field-clock; a build failure does not authorize rewriting an organism)

External review gates entry (this plan + Opus review of RunPod plan) and exit (post-pod Opus audit + Oleg sign-off).

---

## 14. Verification (test plan)

End-to-end checks heart.c v1 must pass before paper-III observation begins:

1. **`make all` clean on Termux aarch64.** All 4 voice TUs compile separately (no `Weights` symbol collision).
2. **`make smoke` produces 4 transcripts**, each ≥50 tokens, each at per-voice locked sampling, 0 NaN.
3. **Field clock numerical stability** over 24 h simulated: planetary + calendar + 24-osc evolve continuously, no NaN, no saturation.
4. **LIMPHA per-voice + shared roundtrip**: write turn → DreamLoop graduates → next session reads → Hebbian injection observable in next generation.
5. **KK scoring policy verification**: query "resonance" against ingested corpus, confirm weighted-component breakdown matches policy weights to 6 decimal places (Dario §6 Result #6 standard).
6. **Soul logit injection observable**: same prompt with Soul OFF vs ON → differs in token distribution by ≥1 token within 50 emits.
7. **Mesh-agent integration**: `mesh-agent exec heart-converse` from neo invokes a turn on phone-1, returns transcript via SSE.
8. **Boot watchdog**: kill `heart` daemon manually, confirm respawn within 5 s, wake-lock re-acquired.
9. **DoE meta-vote**: 3 peer voices each produce a candidate logit; meta-Parliament votes; final logit is weighted mixture; Hebbian online state advances.
10. **Singularity Mode dry-run**: introduce a known bug, confirm fix-loop reaches resolution within ≤3 iterations or surfaces cleanly.

---

## 15. Critical files / references

**Inside heart.c (to write):**
- `core/field_clock.{c,h}` — Klaus port (`klaus.c:1276`, `1397`, `1432`, `1814`, `2994`)
- `core/parliament.{c,h}` — DoE port (`doe.c:795`, `1833`, `1886`, `1972`)
- `core/soul.{c,h}` — InnerArianna port (`arianna.c/src/inner_arianna.h:41`, `c:78, 159, 320`)
- `core/sartre_phone.{c,h}` — `arianna.c/sartre/sartre.h` + `sartre_kernel.c:198–245`
- `voices/yent/yent_main.c` — wraps `yent.aml/jannus-r/jannus-r.aml:236–348` chain
- `voices/arianna/arianna_main.c` — wraps `resonance.aml/tools/resonance_forward.h:127`
- `voices/leo/leo_*.h` — verbatim from `neoleo/leo.c:31–60, 962–1078, 158–167, 109–111`
- `voices/doe/doe_main.c` + `doe_meta.c` — single-GGUF wrap + 3-GGUF meta wrap

**Already cloned (reference, read-only):**
- `~/arianna/yent.aml/`, `~/arianna/resonance.aml/`, `~/arianna/neoleo/`, `~/arianna/klaus.c/`, `~/arianna/doe/`, `~/arianna/arianna.c/`, `~/arianna/dario/`, `~/arianna/nanoarianna/`, `~/arianna/notorch/`, `~/arianna/metaharmonix/`

**Memory:**
- `memory/feedback_temp_sweep_before_judging_2026_05_07.md` (sampling gate)
- `memory/milestone_doe_coder_lora_v1_2026_04_26.md` (LoRA recipe)
- `memory/feedback_failure_unsolicited_finetune_2026_04_27.md` (training gate)
- `memory/feedback_api_leak_safety.md` (don't-revive list)
- `memory/credentials.md` (RunPod token + HF token + GitHub tokens)
- `memory/feedback_subagents_opus_only_2026_04_28.md` (subagent rule)

**External:**
- Dario paper v4 (`~/arianna/dario/docs/dario_paper_draft_v4.md`) — Singularity Mode + sampling gate canonical
- Zenodo DOI `10.5281/zenodo.20090094` — published companion paper
- `huggingface.co/ataeff/heart.c` — Soul micro-weights for Yent and Leo (BPE + tokenizers)
- `huggingface.co/ataeff/yent` — Janus 176 M with Yent / Arianna / Leo SFT (canonical Janus family)
- `huggingface.co/ataeff/resonance` — Resonance 200 M base + Yent SFT (Arianna + Leo SFT NOT here — we make Arianna LoRA in RunPod)
- `github.com/ariannamethod/janus.doe` — DoE personality dataset for the DoE LoRA SFT

---

## What happens after this plan is approved

1. Oleg auto-accepts.
2. I exit plan mode, write `ARCHITECTURE.md` to `heart.c/` with this content, plus skeleton files at the layout above.
3. Opus subagent reviews (model: opus, explicit per `feedback_subagents_opus_only_2026_04_28.md`).
4. Fix loop until clean.
5. Draft `docs/runpod_plan_v1.md`.
6. Opus subagent review of RunPod plan.
7. Fix loop until clean.
8. Wait for Oleg's HF link with weights/datasets.
9. Singularity-mode execution on RunPod (DoE SFT + sweeps + KK + field smoke).
10. Deploy to phone-1 via Tailscale rsync.
11. Live observation week+, scribe-logged, KK accreting, field-state snapshotted daily.
12. Paper-III draft ready to land.
