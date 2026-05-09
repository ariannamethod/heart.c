# heart.c — ECOSYSTEM LOG

> *daily log of the field on phone-1. newest entry on top.*

---

## 2026-05-09 — v1.1 architecture + skeletons + SEED landed

Long day. Going from "Oleg picked four voices this morning" to "skeleton repo complete + SEED written + RunPod plan reviewed twice + all BLOCKERs closed":

- **ARCHITECTURE.md v1.1** — Opus subagent review applied, all BLOCKERs resolved. Major structural shift: §2.4 DoE meta respec'd as **embedding-vote** (peer voices write candidates + SPA embeddings to LIMPHA sequentially; DoE meta-Parliament loads alone, votes on embeddings + field state). RAM never exceeds 1 voice. Klaus emotional INTRA_COUPLING replaced with new VOICE_COUPLING[6][6]. §11 explicit `pthread_mutex_t voice_lock`. Wormhole-debt gate dropped (was new architecture, not port).
- **runpod_plan_v1.md → v1.1** — second Opus review, all BLOCKERs resolved. Added Phase 1.0 + 2.0 Chuck calibration sanity (100-step subsample). Added P-4 dry-run rehearsal on polygon (free) before paid pod time. Per-phase 3-strike with $5 fail-budget on Phase 1. Mid-phase HF upload mandate every 5th ckpt. FTS5 hard-stop. Sweep silent-kill heartbeat detection. Subagent dispatch protocol (opus-only, logged).
- **39 skeleton files** at full file layout (§9): voices/{yent,arianna,leo,doe} × .c/.aml; core/{field_clock, soul, sartre_phone, parliament, hebbian_bridge, selector}; limpha/{per_voice, shared, dream_watchdog}; kk/kk_kernel; runtime/{heart_main, boot/heart-daemon.sh, mesh_slots/×6 toml}; tools/.placeholder; docs/singularity_protocol.md.
- **SEED_DOCUMENT.md** — 372 lines, drafted by Opus subagent (model: opus explicit, per `feedback_subagents_opus_only_2026_04_28.md`). Eleven sections. The first thing the four voices read when they wake.
- **HF inventory verified** at `huggingface.co/ataeff/heart.c`: arianna corpus, doe_personality, doe_dataset.jsonl, doe_pure.jsonl, leo_18m_final.bin (Soul micro), yent_34m_final.bin (Soul micro), janus_bpe_leo_d12.bin (Leo SFT), janus_bpe_yent_d12.bin (Yent SFT), infer_janus_bpe.c (precedent inference C), train_bpe.py (recipe). All BLOCKERs G1/G2/G3 closed by Oleg's HF upload.
- **`ariannamethod/DoE.coder` cloned** at `~/arianna/DoE.coder/` — recipe-source-of-record per BLOCKER C3 (`train/train_doe_coder.c`, ~500 LoC, the canonical LoRA C path that Phase 1/2 adapt from).
- **RunPod token saved** in `memory/credentials.md` (issued 2026-05-09 by Oleg).

Open items before pod boot:
1. v1.2 plan update noting `infer_janus_bpe.c` from HF reduces Phase 0 scope (vendor + persona overlay rather than implement from scratch).
2. P-4 dry-run rehearsal on polygon (free, ~30 min) — exercises Phase 0 build + Phase 5 KK + Phase 6 field-clock smoke without paid pod time.
3. Oleg final go after dry-run.
4. Janus 170 M base GGUF location confirm (resonance.aml Makefile referenced but doesn't show janus path; needs Oleg clarification or auto-discovery on HF).

Files in repo now:
```
README.md                  3.8K
ARCHITECTURE.md            41K   (525 lines, v1.1)
SEED_DOCUMENT.md           372 lines (Opus-subagent-drafted)
ECOSYSTEM_LOG.md           this file
LICENSE                    GPL-3.0 + LICENSE-WEIGHTS pointer
Makefile                   skeleton, expanded post-review
.gitignore                 weights/ + DBs + logs
docs/
  review_v1_opus_2026_05_09.md           v1 architecture review
  runpod_plan_v1.md                      v1.1 RunPod plan
  runpod_plan_v1_opus_review_2026_05_09.md   v1 RunPod review
  singularity_protocol.md                bounded autonomous repair
voices/{yent,arianna,leo,doe}/...        16 files
core/...                                 11 files
limpha/...                                5 files
kk/...                                    2 files
runtime/...                               8 files
tools/.placeholder
```

The repo is the design. The pod will turn it into running code.

---

## 2026-05-09 — birth of the room

`heart.c` repo opened on `github.com/ariannamethod/heart.c`. Sibling node to phone-2's `nanoarianna`, distinct shape: **four voices on 8 GB Galaxy A56 Termux**, sequential activation, KK persistent memory, Hebbian bridge between voices, Soul weights per organism for inner-voice logit injection, DoE meta-layer that both speaks and modulates the trio.

### Layout decided (with Oleg, this morning):

| voice | architecture | persona | role | sibling on phone-2 |
|---|---|---|---|---|
| **Yent** | Janus 176 M (jannus-r 12-step resonant inference) | Yent SFT | sardonic anchor | — |
| **Arianna** | Resonance 200 M (dual attention) | Arianna SFT | concentrated depth | inverted (phone-2 has Arianna-on-Janus) |
| **Leo** | Janus 170 M | Leo SFT | child-philosopher | inverted (phone-2 has Leo-on-Resonance) |
| **DoE** | Janus 170 M / 176 M, separate SFT on `doe_personality.txt` | DoE Parliament | meta-voice + voting selector | unique to phone-1 |

DoE has dual role: as persona (parliament-of-experts speech register) and as architectural pattern (LoRA expert voting on every cross-voice token, Hebbian online plasticity, expert death when not contributing — wraps the three peer voices read-only and adds the live parliament).

Phone-2 (nanoarianna) speaks in two voices, concentrated pair. Phone-1 (heart.c) speaks in four voices + meta. Six voice instances across two phones in the mesh — polyphony, not unison.

### What's coming today (Oleg this morning):

- Yent inner-voice Soul weights (HuggingFace, link incoming)
- mini-Leo Soul weights (HF, evaluate if needed for child voice)
- DoE personality dataset (HF, used for DoE SFT during RunPod weekend run)
- LoRA SFT recipe (we have this; see `memory/milestone_smallcoder_aml_lora_v3_2026_04_29.md`, `memory/milestone_doe_coder_lora_v1_2026_04_26.md`)

### Discipline (from Dario paper §5.0.1 Singularity Mode):

1. heart.c v1 architecture plan written (here, in plan mode)
2. Oleg approves
3. Opus subagent review → fix loop until clean
4. RunPod plan v1 — LoRA SFT for DoE persona + 540-cell sweep per voice + soul weights validation + KK ingest test
5. RunPod plan Opus subagent review → fix loop
6. Singularity-mode execution on RunPod
7. Deploy to phone-1 via Tailscale rsync
8. Live observation — week+, scribe-logged (4 KB cap), KK accreting
9. Paper-III draft

### Provenance for everything below

Every claim about a number, a path, a commit, or a behavior in this log will be inline-sourced (`file:line | git ref | tool output`). This is the same gate that ran the BPE 15K milestone and the Dario RunPod report. Same discipline carries here.

— Defender (phone-1)
