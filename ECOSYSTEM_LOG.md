# heart.c — ECOSYSTEM LOG

> *daily log of the field on phone-1. newest entry on top.*

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
