/* heart.c — core/hebbian_bridge.h
 *
 * Dario-style Hebbian bridge: KK retrieval → voice logits at
 * sentence boundaries. Knowledge enters as field pressure, not
 * as RAG-style context paste (ARCHITECTURE.md §5).
 */
#ifndef HEART_HEBBIAN_BRIDGE_H
#define HEART_HEBBIAN_BRIDGE_H

/* Inject KK-scored chunks at sentence boundary during decode.
 * Modulates `logits` in-place based on retrieved chunks' resonance
 * with current field state. */
void heart_hebbian_inject(float *logits, int vocab,
                           const char *recent_text,
                           int sentence_just_ended);

#endif
