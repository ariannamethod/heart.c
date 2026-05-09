# notorch patches required for heart.c training

Two patches required against `github.com/ariannamethod/notorch` (both are
real bugs for the GPU path; should be PR'd upstream).

## 1. nt_seq_cross_entropy_masked — missing GPU→CPU sync (notorch.c:3661)

`nt_seq_cross_entropy_masked` is CPU-only and reads `pl->output->data`
directly. When GPU mode is on and logits arrive from `nt_seq_linear`
GPU dispatch, the CPU mirror is stale (zero-initialised), so CE reads
zeros → softmax = uniform → loss = ln(V) **regardless of input**.

Symptom (heart.c phase 1 cal run 6/7): loss exactly 9.7041 every step
on Resonance 200M (ln 16384), proven by CPU-only run hitting 4.9 same
input.

Fix:

```c
nt_tape_entry* pm = &g_tape.entries[mask_idx];
#ifdef USE_CUDA
    nt_tensor_ensure_cpu(pl->output);
    nt_tensor_ensure_cpu(pt->output);
    nt_tensor_ensure_cpu(pm->output);
#endif
    nt_tensor* out = nt_tensor_new(1);
```

## 2. GPU_PTR_MAP_SIZE too small (notorch_cuda.cu:118)

Default `8192` slots in the cuBLAS allocation tracker overflow when
training has ~600+ active GPU tensors per step plus init params. Hash
linear-probe finds no slot → `[GPU] ptr_map full — buffer leak` floods
stderr and CUDA buffers leak.

Symptom (heart.c phase 1 first full run): >100 leak warnings/sec, GPU
memory pressure rises, eventual OOM.

Fix:

```c
#define GPU_PTR_MAP_SIZE 65536   // was 8192 — 8× headroom for tape ops
```

## How to apply (until upstream merged)

```bash
# CE ensure_cpu fix (idempotent string patch)
python3 - <<'PY'
src = open("/path/to/notorch/notorch.c").read()
target = '...'  # see above
patch  = '...'
open("/path/to/notorch/notorch.c","w").write(src.replace(target, patch))
PY

# ptr_map size bump
sed -i 's/#define GPU_PTR_MAP_SIZE 8192/#define GPU_PTR_MAP_SIZE 65536/' \
    /path/to/notorch/notorch_cuda.cu
```

Both fixes applied on phone-1 / heart.c-runpod 2026-05-09 during Phase
1 calibration.
