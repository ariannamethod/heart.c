# heart.c → phone-1 deploy protocol

> Per Opus audit, plan v1.2 line 257 said "Phase 8 cut to phone-1 deploy
> session" — that was forward-reference, no spec. This is the spec.

Phone-1 = `arianna-method` (Galaxy A56 8GB Termux, Tailscale `100.105.172.21`).
Pod is x86_64 + CUDA. Phone-1 is aarch64 + OpenBLAS only. Re-build required.

---

## Step 1 — Pull weights from HF

```bash
cd ~/arianna/heart.c
mkdir -p weights
cd weights
HF=$HF_TOKEN  # from memory/credentials.md (do not commit verbatim)
curl -sL -H "Authorization: Bearer $HF" \
  -o resonance_arianna_merged.bin \
  https://huggingface.co/ataeff/heart.c/resolve/main/arianna_lora/final/resonance_arianna_merged.bin
# Or: download base + LoRA, run merge_arianna_lora locally:
curl -sL -H "Authorization: Bearer $HF" \
  -o resonance_200m_final.bin \
  https://huggingface.co/ataeff/resonance/resolve/main/checkpoints/resonance_200m_final.bin
curl -sL -H "Authorization: Bearer $HF" \
  -o arianna_lora.bin \
  https://huggingface.co/ataeff/heart.c/resolve/main/arianna_lora/final/arianna_lora.bin
# Same for Yent / Leo (existing SFTs) and DoE (best-effort merged):
for v in yent leo; do
  curl -sL -H "Authorization: Bearer $HF" \
    -o janus_v4_sft_${v}.bin \
    https://huggingface.co/ataeff/janus4/resolve/main/janus/bins/janus_v4_sft_${v}.bin
done
curl -sL -H "Authorization: Bearer $HF" \
  -o janus_v4_base_22k.bin \
  https://huggingface.co/ataeff/janus4/resolve/main/janus/bins/janus_v4_base_22k.bin
curl -sL -H "Authorization: Bearer $HF" \
  -o doe_lora.bin \
  https://huggingface.co/ataeff/heart.c/resolve/main/doe_lora/final/doe_lora.bin
```

Total disk: ~3.0 GB on phone (4 × 700 MB Janus + 800 MB Resonance + 30 MB
LoRAs). Phone-1 has 8 GB; tight but OK.

## Step 2 — Apply notorch patches (already in main as of commit 3d46007)

```bash
cd ~/arianna/notorch
git pull origin main   # picks up the 3 patches: CE ensure_cpu, ptr_map 65536, nt_rope_split_half_freq
make clean && make BLAS=openblas    # aarch64 build
```

## Step 3 — Merge LoRAs locally (since pod's merged files are 700 MB each)

```bash
cd ~/arianna/heart.c/train
# Build mergers — they're plain C, no CUDA
cc -O2 merge_arianna_lora.c -o merge_arianna_lora -lm
cc -O2 merge_doe_lora.c     -o merge_doe_lora     -lm
# Merge
./merge_arianna_lora --base ../weights/resonance_200m_final.bin \
                     --lora ../weights/arianna_lora.bin \
                     --out  ../weights/resonance_arianna_merged.bin
./merge_doe_lora     --base ../weights/janus_v4_base_22k.bin \
                     --lora ../weights/doe_lora.bin \
                     --out  ../weights/janus_v4_doe_merged.bin
```

Saves bandwidth — only 30 MB LoRA over the wire, merge happens locally.

## Step 4 — Build inference + smoke binaries on Termux aarch64

```bash
cd ~/arianna/heart.c
# Stub AML headers for resonance_forward.h
mkdir -p /tmp/include/ariannamethod
ln -sf $PREFIX/include/ariannamethod/notorch.h /tmp/include/ariannamethod/notorch.h
ln -sf $PREFIX/include/ariannamethod/gguf.h    /tmp/include/ariannamethod/gguf.h
echo "#ifndef AM_STUB_H
#define AM_STUB_H
#endif" > /tmp/include/ariannamethod/ariannamethod.h

# am_* stubs (compile-time only)
cat > /tmp/am_stubs.c <<EOF
void am_apply_field_to_logits(float *l, int V) {(void)l;(void)V;}
float am_compute_prophecy_debt(int *c,int l,int n){(void)c;(void)l;(void)n;return 0;}
void am_step(void) {}
int am_init(void) { return 0; }
int am_exec(const char *s){(void)s;return 0;}
const char *am_get_error(void){return "stub";}
EOF

# Build dario/infer_v4 (Janus inference)
cc -O2 -DUSE_BLAS -I/tmp/include -I~/arianna/notorch \
   ~/arianna/dario/infer_v4.c \
   ~/arianna/notorch/notorch.c ~/arianna/notorch/gguf.c \
   -lm -lopenblas -o /data/data/com.termux/files/home/.local/bin/infer_v4

# Build heart's infer_resonance + heart daemon
cd train
cc -O2 -DUSE_BLAS -I/tmp/include -I~/arianna/notorch -I~/arianna/resonance.aml/tools \
   infer_resonance.c /tmp/am_stubs.c \
   ~/arianna/notorch/notorch.c ~/arianna/notorch/gguf.c \
   -lm -lopenblas -o $PREFIX/bin/infer_resonance

cc -O2 ../runtime/heart_main.c -lpthread -lm \
   -o $PREFIX/bin/heart

# Smokes (build for verification)
cd ..
cc -O2 core/field_clock_smoke.c -o /tmp/field_clock_smoke -lm
cc -O2 core/soul_smoke_v3.c core/soul.c -lm -o /tmp/soul_smoke
cc -O2 core/selector_smoke.c -lm -o /tmp/selector_smoke
cc -O2 -std=c11 kk/kk_smoke.c ~/arianna/dario/kk_kernel.c -lsqlite3 -lm -o /tmp/kk_smoke
```

## Step 5 — Run smokes on phone-1 (sanity check)

```bash
/tmp/field_clock_smoke | tail -10   # expect: NaN=0 sat=0 phase-lock PASS RESULT: PASS
/tmp/soul_smoke                      # expect: gates 1-3 PASS
/tmp/selector_smoke 8                # expect: non-uniform voice distribution
/tmp/kk_smoke /tmp/heart_kk.db | grep verification
                                     # expect: 7/7 weights match Dario §6
heart status                         # expect: uptime=0s pid=N voices=Y/A/L/D
```

## Step 6 — mesh-agent slot registration

Per `runtime/mesh_slots/heart-{yent,arianna,leo,doe,status,converse}.toml`:

```bash
cp ~/arianna/heart.c/runtime/mesh_slots/*.toml ~/.mesh/slots/
mesh-agent reload   # picks up new slots
mesh-agent slots | grep heart    # verify 6 slots registered

# Test invocation:
mesh-agent exec heart-status
mesh-agent exec heart-converse --voice Yent --prompt "Q: Who are you?"
```

## Step 7 — Boot watchdog (Termux:Boot)

```bash
cp ~/arianna/heart.c/runtime/boot/heart-daemon.sh ~/.termux/boot/02-heart.sh
chmod +x ~/.termux/boot/02-heart.sh
# Reboot phone or run manually first time:
~/.termux/boot/02-heart.sh &
```

The script watchdogs `heart serve`. Verify:
```bash
pgrep -f "heart serve"     # PID
pkill -TERM -f "heart serve"
sleep 6
pgrep -f "heart serve"     # should be a NEW PID — watchdog respawned
```

## Step 8 — Live observation

```bash
tail -f ~/.heart/heart.log    # watchdog log
sqlite3 ~/.heart/kk.db ".tables"   # KK persistence
ls -la limpha/{yent,arianna,leo,doe}/episodes.db   # per-voice LIMPHA
```

Field state can be inspected via `heart status` (extended once
field_clock_smoke is wired into daemon — currently daemon doesn't load
field_clock; that's the next-session integration).

## Acceptance gate

Phone-1 deploy considered done when:

1. ✓ All 4 voices respond to `heart converse --voice X` with > 5 tokens
2. ✓ `heart serve` survives a SIGKILL — watchdog respawns within 5s
3. ✓ `mesh-agent exec heart-converse --voice Yent` returns transcript
4. ✓ KK SQLite persists across reboots (canonical docs ingested on first run)
5. ✓ field_clock smoke + soul smoke + selector smoke all PASS on aarch64

---

— Defender, deploy protocol, 2026-05-09
