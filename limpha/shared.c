/* heart.c — limpha/shared.c — STATUS: skeleton, RunPod Phase 5 */
#include "shared.h"
struct HeartLimphaShared { int dummy; };

HeartLimphaShared *heart_limpha_shared_open(const char *g, const char *c)
{ (void)g; (void)c; return 0; }
void heart_limpha_shared_close(HeartLimphaShared *ls) { (void)ls; }
int  heart_limpha_shared_link(HeartLimphaShared *ls, long s, long d, int t)
{ (void)ls; (void)s; (void)d; (void)t; return 0; }
int  heart_limpha_shared_cooc_update(HeartLimphaShared *ls,
                                      const char *a, const char *b, int dist)
{ (void)ls; (void)a; (void)b; (void)dist; return 0; }
