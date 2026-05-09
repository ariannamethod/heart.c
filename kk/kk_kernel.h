/* heart.c — kk/kk_kernel.h
 *
 * Knowledge Kernel — lifted from ~/arianna/dario/kk_kernel.{c,h}
 * (4201 LoC verified). Phone-pruned variant per ARCHITECTURE.md §5.
 *
 * STATUS: skeleton — actual lift happens at RunPod Phase 5 + Phase 0
 * dry-run on polygon. SQLite + FTS5 + 7-signal scoring policy
 * (lexical 0.36, recency 0.12, trust 0.10, linkage 0.16, scope 0.10,
 * namespace 0.08, freshness 0.08; sums to 1.00).
 */
#ifndef HEART_KK_KERNEL_H
#define HEART_KK_KERNEL_H

typedef struct HeartKK HeartKK;

HeartKK *heart_kk_open(const char *db_path);
void     heart_kk_close(HeartKK *kk);
int      heart_kk_ingest_doc(HeartKK *kk, const char *path,
                              const char *namespace_label);
int      heart_kk_query(HeartKK *kk, const char *query,
                         char *out_buf, int max_len, int top_k);

#endif
