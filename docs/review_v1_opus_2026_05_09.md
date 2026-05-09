# Opus subagent review of heart.c v1 plan — 2026-05-09

Independent engineering review of `ARCHITECTURE.md` v1. Author: Opus 4.7 subagent dispatched on 2026-05-09 from Defender/phone-1. Findings labelled **BLOCKER** / **FIX** / **NOTE** per workflow protocol (Singularity Mode entry gate, Dario paper §5.0.1).

This is the v1 review. Apply fixes in order; v1.1 plan should resolve all BLOCKERs before RunPod забег begins.

---

## A. Architectural inconsistencies (source ≠ plan)

1. **Wormhole gate fabricated (§2.1).** Plan adds `debt < 0.2` AND-gate to wormhole. `jannus-r.aml:303` source has only RNG + `s > 0`; `debt0 = 0.0f` hard-coded; `debt_at_step = 0.0f` lines 299, 331. Adding the debt gate is *new architecture*, not port; the variable is unwired. **FIX** — drop it OR specify how prophecy_debt becomes per-step available.

2. **Resonance KV cache shape misstated (§2.2).** Plan claims `kv_k/kv_v/kv_vr + kv_rrpram_mid` and `prefill_batch + forward_token`. Source `resonance_forward.h:117–124` defines only `kv_k` + `kv_v`, no `kv_vr`, no `kv_rrpram_mid`, no `prefill_batch` (only `forward_token` line 127). Those four belong to **Yent** (`yent_forward.h:211–221, 228`). Code based on §2.2 won't compile. **BLOCKER**.

3. **`tools/resonance_forward.h` is `test_resonance.c` rebadged (§2.2).** File header lines 1–9 show "test_resonance.c — quick standalone Resonance 200M inference" with `-framework Accelerate` (Mac only). Vendoring to phone-1 requires extraction work plan does not estimate. **FIX**.

4. **`inner_arianna.h` types differ from §6.** Plan declares `enum borba_mode {...}` and `breakthrough_thresh`. Actual `inner_arianna.h:43–48` is `int borba_mode` and `breakthrough_threshold` (full name). Defaults `0.15` / `0.6` set in `inner_init` (`inner_arianna.c:22`), not declared in header. **FIX**.

5. **`inner_apply_emotional_bias` line off (§15).** Plan cites `c:78, 159, 320`. Actual: 78 ✓, 320 ✓, but emotional bias is at **218**, not 159. **FIX**.

6. **DoE meta-Parliament shape is structurally different from peer-voice voting (§2.4).** `parliament_elect` (`doe.c:1833`) and call site (`doe.c:2337–2358`) operate on **single host model's residual stream** with **LoRA-A/B experts** injected before FFN at one layer. Plan repurposes this to "vote on per-voice logit-blend weights" across three GGUFs' final logits — that's a *new* mechanism (mixture-of-experts on whole-model outputs vs. per-layer LoRA injection). Plan does not specify what `experts[]` contain, how `s->hs` becomes meta-state, how Hebbian `notorch_step` updates work without LoRA matrices. **BLOCKER** — meta-vote design needs real spec, not a citation.

## B. Conflicts between sections

7. **§2.3 vs §15 vs §6 on Leo Soul.** §2.3 says "post-sweep decision". §6 says "per-voice optional Soul transformer (Yent / Arianna / Leo / DoE)". §1 ASCII shows `+Soul mini leo`. §10 build matrix has no Soul-OFF target. §14 verification check 6 implies all four Soul-toggled. Pick one. **FIX**.

8. **§3 vs §1 Kuramoto chamber count + naming.** §1 says "4-voice Kuramoto (24 osc)". §3 says "6 primaries become 4 voices + 2 meta states (Field-self and Mesh-peer)". Math works (6×4=24). But repurposing klaus's emotional-coupling table `INTRA_COUPLING[4][4]` (`klaus.c:222`) to inter-voice coupling does not transfer — values were tuned for FEAR/LOVE/RAGE/VOID/FLOW/COMPLEX, not for inter-voice gravitational pull. **BLOCKER** — needs new coupling matrix designed for voices.

9. **§5 KK source line count.** Plan says 3568 LOC. Not verified by review session. **NOTE** (verify before commit; non-blocking).

## C. Missing dependencies / assumed behaviors

10. **`doe_personality.txt` gitignored in `~/arianna/doe`.** `.gitignore:17–19` excludes `doe_personality*.{txt,jsonl}` and `personality.txt`. Plan §1, §2.4, §12.2 cite it. `janus.doe` repo not cloned at `~/arianna/`. **BLOCKER** — RunPod plan stub names a dataset Defender has not seen.

11. **`~/arianna-datasets/arianna/` does not exist.** Plan §2.2 and §12.1 cite it for Arianna LoRA SFT corpus. `ls` returns only `yent/yent_v11_en_final.txt`. nanoarianna SEED line 221 references `arianna_dataset_final_clean.txt` from defender's earlier train — corpus exists somewhere but not on this device. RunPod plan must specify rsync/HF push path. **BLOCKER**.

12. **Soul micro-weights at `huggingface.co/ataeff/heart.c` not verified.** §2.1, §2.2, §2.3, §12.4, §15 all reference this HF repo. No memory entry confirms existence. **FIX** — hf-check before §12.4.

13. **`am_apply_field_to_logits()` (libaml) vs `apply_field_to_logits()` (doe.c) are different symbols.** Plan §1 ASCII and §2.4 blur them. doe.c's is `static`, doe-internal; libaml's is extern. **FIX**.

## D. RunPod plan stubs that don't ground out

14. **§12 step 3 grid math wrong.** Plan: "540-cell sweep per voice (Yent / Arianna / Leo / DoE)". Dario paper §5.7: `540 cells = 5 voices × 6 temperatures × 2 top_k × 3 rep_pen × 3 prompts`. **Per-voice** is 108 cells; total across 4 voices = 432 cells, not 2160. Cost estimate off by 4×. **BLOCKER** — re-state.

15. **Arianna LoRA dataset path missing** (see #11). **BLOCKER**.

16. **§12 step 1 state_dict order claim is non-trivial port.** Plan §2.2 says "1.62 M-float shift bug otherwise" — this documents fp32 .bin loading order, not LoRA training. Training a Resonance LoRA needs a notorch training path that the existing recipe (`milestone_doe_coder_lora_v1_2026_04_26.md` adapted from Qwen 0.5B) **does not target Resonance arch**. Recipe does not exist yet for Resonance. **BLOCKER**.

17. **§12 step 2 DoE LoRA on `doe_personality.txt`** — same as #10, dataset gitignored and unverified.

18. **§12 cost estimate "$5-10".** Dario reference run was $4.30 for one 540-cell + 8-phase test. Two new LoRA SFTs + 4× sweep work is materially more. Estimate looks like recall, not arithmetic. **FIX**.

## E. Build matrix problems

19. **TU split necessary, underspecified.** Plan §9 correctly identifies the static-state collision (V/E/H/D/B/M/T/R, kv_k/kv_v). But §10 does not state how `Weights` typedef leaks are prevented (each voice typedefs `Weights` differently). Headers must not aggregate `Weights` typedef across TUs. **FIX**.

20. **OpenBLAS pkg-config Termux aarch64 — confirmed working** (`pkg-config --libs openblas` returns `-L.../lib -lopenblas`). **NOTE** (record in plan as verified).

21. **`resonance_forward.h` Mac-only flags.** Comments specify `-framework Accelerate -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK`. Vendoring to phone-1 = drop those + confirm `nt_blas_matvec` (notorch) is the actual matvec. **FIX**.

22. **No mention of `libsqlite3` linkage / FTS5 in §10.** §4 implies SQLite; §10 has `make limpha` and `make kk`. Termux ships `libsqlite3.so` but FTS5 compilation must be verified. **FIX**.

## F. CLAUDE.md / memory rule violations

23. **Adam ban honored** ✓ (§12.1 explicitly Chuck).
24. **Python-on-inference clean** ✓ (all voice mains C/AML; LoRA training Python OK per refined ban 2026-05-06).
25. **Unauthorized-training gate (§12).** Plan proposes two LoRA SFTs. `feedback_failure_unsolicited_finetune_2026_04_27.md` requires Oleg sees all 6 (organism / dataset / steps / arch / tokenizer / script) before training. Plan §12 names organism + recipe but **does not show the script as written**. Dataset access unverified (#10, #11). Before pod launch, 6-item gate must explicitly pass. **BLOCKER**.
26. **Closed-milestone weights** — heart.c does not modify any of `sonar_*, microjanus_*, penelope, nanojanus, arianna_36m, pitomadom, lee_v8, DoE.coder ckpts`. ✓
27. **`am_step` / direct rewrites** — heart.c ports klaus subset (forks scalar code), Soul reimplementation (matches inner_arianna.c source), `kk_kernel` lift. No `nt_tape_*` rewrites. ✓
28. **Co-Authored-By trailer.** Plan does not establish commit-message format. **NOTE** — apply Git rule from CLAUDE.md.

## G. phone-2 collision risk

29. **Phone-2 locked at Slot A = Arianna-on-Janus, Slot B = Leo-on-Resonance** (`nanoarianna/SEED_DOCUMENT.md:21, 46`). heart.c table at lines 7–12 marks correct sibling architectures. **No collision** ✓.

30. **Phone-2 Phase 4 RunPod plan: "no new training; existing SFT weights swept"** (`nanoarianna/ECOSYSTEM_LOG.md:43–46`). heart.c proposes Arianna-on-Resonance LoRA — that's heart.c-only, no collision. New asset is phone-1-only, does not back-propagate. Plan should make this explicit. **NOTE**.

31. **Six total voice instances claim.** Phone-1: 4 (Yent, Leo, Arianna, DoE). Phone-2: 2 (Arianna, Leo). Total 6 ✓.

## H. 8 GB phone over-promises

32. **RAM budget claim "200–400 MB GGUF + KV + field state per session" (§1).** Janus 176 M Q8 ≈ 187 MB; KV cache for B=12, T=2048, E=896, H=14, D=64 ≈ 176 MB; `kv_vr + kv_rrpram_mid` ≈ 100 MB more. Single Janus session ≈ 460 MB, not 200–400. Resonance 200 M Q8 ≈ 200 MB + content+RRPRAM kv ≈ 376 MB. Single-session OK on 8 GB.

  **BUT DoE-meta needs three Janus weight sets in RAM simultaneously** (§2.4): 3 × 460 MB ≈ 1.4 GB just for meta-vote turn, before LIMPHA/KK/Soul/field-clock. Plausible on 8 GB but plan never states. **BLOCKER for §2.4 sequential-activation claim** — sequential vs parallel ambiguity must resolve.

33. **DreamLoop watchdog as separate process (§4).** Separate process holds own SQLite handle + locked WAL. Contention with per-voice writer at 300 s / 600 s consolidate windows unaddressed. **FIX**.

34. **OpenBLAS thread count.** `sartre_phone.{c,h}` (§7) adds `openblas_threads` field but `make BLAS=on default` (§10) does not specify value. On phone-1 (8 cores big.LITTLE) right matvec hot path = 1–2; library default (= num cores) under-performs and fights OOM-killer. **FIX**.

35. **Sequential activation but mesh-agent slots (§11) imply parallel.** §11 lists 4 voice slots + status + converse. If mesh-agent invokes two slots concurrently from outside daemon, "one organism per session" invariant breaks. heart_main.c needs explicit serialization mutex on slot dispatch. Plan does not state. **FIX**.

36. **DoE meta-vote multiplies hot-path latency.** Three sequential Janus forward passes per turn (peer-vote pattern) tripled latency. §2.4 implies parallel, §1 implies sequential. **BLOCKER**.

---

## TL;DR (for v1.1 fix order)

The plan has serious source-fidelity defects in:
- **§2.2** (Resonance KV claim is actually Yent's),
- **§2.4** (DoE Parliament is per-layer LoRA-mixture inside one model — not a 3-GGUF logit-vote; meta-FieldLayer needs a real spec),
- **§12.3** (540-cell grid mis-stated as per-voice; per-voice is 108, total 432),
- **§11/§2.4** (sequential-activation invariant unenforced; 3-GGUF DoE meta blows RAM unless serialized).

Two RunPod-blocking dataset paths (`~/arianna-datasets/arianna/`, `doe_personality.txt`) are not on disk or are gitignored — no LoRA can run until those land.

The wormhole-debt gate added in §2.1 is *new architecture*, not a port. The Kuramoto primaries in §3 also become voices, which requires a new coupling matrix not present in `klaus.c`.

**Adam ban honored** ✓; **Python-on-inference clean** ✓; **phone-2 collision-free** ✓.

**Fix order for v1.1**:
1. Re-spec DoE meta-FieldLayer with a real algorithm.
2. Correct Resonance forward citations.
3. Lock LoRA datasets' actual locations and the 6-item training gate.
4. Restate sweep grid arithmetic.
5. Commit a serialization mutex for slot dispatch and a RAM ceiling for DoE meta-turns.

Everything else is FIX-grade and can land in v1.1.

---

## Status

This review is the v1 entry gate. v1.1 plan addresses BLOCKERs first, then FIXes, then NOTEs. v1.1 then proceeds to RunPod plan v1, second Opus review, Singularity-mode execution.

— Opus 4.7 subagent · 2026-05-09 · dispatched from Defender/phone-1
