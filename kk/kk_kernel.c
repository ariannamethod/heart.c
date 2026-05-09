/* heart.c — kk/kk_kernel.c — STATUS: skeleton, lift dario kk at Phase 5 */
#include "kk_kernel.h"
struct HeartKK { int dummy; };
HeartKK *heart_kk_open(const char *p) { (void)p; return 0; }
void     heart_kk_close(HeartKK *kk)  { (void)kk; }
int      heart_kk_ingest_doc(HeartKK *kk, const char *p, const char *n)
{ (void)kk; (void)p; (void)n; return 0; }
int      heart_kk_query(HeartKK *kk, const char *q, char *b, int m, int k)
{ (void)kk; (void)q; (void)b; (void)m; (void)k; return 0; }
