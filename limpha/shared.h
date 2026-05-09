/* heart.c — limpha/shared.h
 *
 * Shared LIMPHA: cross-voice cooc graph + sentence-bigram bridge.
 * One SQLite each: limpha/shared/graph.db, limpha/shared/cooc.db.
 *
 * LinkType enum lifted from arianna.c/limpha/graph_memory.py:29:
 *   REMINDS_OF / CONTRADICTS / CONTINUES / RESONATES / CAUSED_BY / SUMMARY_OF
 */
#ifndef HEART_LIMPHA_SHARED_H
#define HEART_LIMPHA_SHARED_H

enum heart_limpha_link_type {
    LINK_REMINDS_OF = 0,
    LINK_CONTRADICTS,
    LINK_CONTINUES,
    LINK_RESONATES,
    LINK_CAUSED_BY,
    LINK_SUMMARY_OF,
};

typedef struct HeartLimphaShared HeartLimphaShared;

HeartLimphaShared *heart_limpha_shared_open(const char *graph_path,
                                             const char *cooc_path);
void  heart_limpha_shared_close(HeartLimphaShared *ls);
int   heart_limpha_shared_link(HeartLimphaShared *ls,
                                long src_episode, long dst_episode,
                                int link_type);
int   heart_limpha_shared_cooc_update(HeartLimphaShared *ls,
                                       const char *word_a,
                                       const char *word_b,
                                       int distance);

#endif
