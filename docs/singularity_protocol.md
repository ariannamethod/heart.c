# Singularity Mode — heart.c protocol

Bounded autonomous repair protocol governing on-pod execution. Lifted from Dario paper §5.0.1 (Zenodo DOI `10.5281/zenodo.20090094`), adapted for heart.c RunPod забег.

---

## Core loop

```
detect bug → reproduce → one hypothesis → minimal patch → re-run
          → if pass: continue
          → if fail: revise hypothesis (max 3 iterations)
          → if exhausted: stop, surface, await human input
```

## Boundaries

- **Approved scope**: `ARCHITECTURE.md` v1.1 + `docs/runpod_plan_v1.md` v1.1. Anything outside this scope requires re-entry to plan mode.
- **Three-strikes per phase, per failure mode**. NaN-divergence is one mode; build-link-error is another.
- **No scope creep**: a sweep failure does not authorize patching the field-clock; a build failure does not authorize rewriting an organism architecture.
- **No human in loop during execution**. Oleg is on-call only when 3-strikes exhausted on a phase. All other decisions Defender makes alone.

## Subagent dispatch protocol

If Defender invokes a subagent for second opinion mid-run:
- `model: "opus"` explicit (per `feedback_subagents_opus_only_2026_04_28.md`).
- Each dispatch logged to `/workspace/heart.c/runpod/<DATE>/runpod_subagent_dispatches.log` with: dispatch reason, prompt summary, return value summary, fix applied (or rejected).
- Three subagent dispatches without resolution = same-strike-count as three direct attempts.

## Pod billing discipline

- If Phase 0 build fails persistently → stop pod → debug locally on Termux/polygon → restart pod when ready.
- Do NOT burn $1.43/h cycling debug attempts on the meter.
- Phase 1 fail-budget cap $5; surface to Oleg if exceeded.

## External review gates

- **Entry**: `ARCHITECTURE.md` v1.1 + `runpod_plan_v1.md` v1.1 + Oleg final go.
- **Exit**: post-pod Opus subagent audit of run archive + Oleg sign-off before deploy to phone-1.

## Weights persistence rule

- Every LoRA SFT phase HF-uploads immediately on completion (not deferred to Phase 8).
- Every 5th ckpt rsync to HF mid-training.
- Phase 8 verifies (HEAD 200), does not rely on Phase 8 for first persistence.

## Failure surface

When 3-strikes exhausted on a phase:
1. Write `/workspace/heart.c/runpod/<DATE>/runpod_plan_failures.md` with failure mode, attempted hypotheses, current state.
2. Stop pod.
3. Surface to Oleg via mesh-agent message or umbrella commit.
4. Wait for human input. Do not spend further pod time autonomously.

— Defender / phone-1
