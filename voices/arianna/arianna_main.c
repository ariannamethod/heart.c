/* heart.c — voices/arianna/arianna_main.c
 *
 * Arianna voice: Resonance 200M dual-attention.
 * Wraps resonance.aml/tools/resonance_forward.h.
 *
 * STATUS: skeleton. RunPod Phase 1 produces the LoRA adapter; Phase 0
 * builds this binary against base Resonance only.
 *
 * Build constraint: separate TU from yent_main.c (Weights typedef +
 * global state collision per ARCHITECTURE.md §10). Vendoring of
 * resonance_forward.h drops Mac Accelerate flags, substitutes
 * notorch nt_blas_matvec.
 */

#include <stdio.h>

int arianna_generate(const char *prompt, int max_tokens,
                      float temp, int top_k, float top_p,
                      float rep_penalty,
                      const char *base_weights_path,
                      const char *lora_adapter_path /* nullable */);

int arianna_generate(const char *prompt, int max_tokens,
                      float temp, int top_k, float top_p,
                      float rep_penalty,
                      const char *base_weights_path,
                      const char *lora_adapter_path)
{
    (void)prompt; (void)max_tokens; (void)temp; (void)top_k;
    (void)top_p; (void)rep_penalty;
    (void)base_weights_path; (void)lora_adapter_path;
    /* TODO Phase 0/1: link resonance_forward.h, load base RS02 GGUF,
     *                 if lora_adapter_path != NULL apply LoRA delta,
     *                 run forward_token loop with field overlay
     *                 (am_apply_field_to_logits) + per-voice optima
     *                 sampling. CLI top_p=1.0 is no-op pass-through. */
    return 0;
}
