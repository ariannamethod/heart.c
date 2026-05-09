/* heart.c — limpha/per_voice.h
 *
 * Per-voice LIMPHA: 4 SQLite handles (one per voice), schemas
 * lifted from arianna.c/limpha/{memory.py:88, episodes.py:104,
 * episodes_enhanced.py:104, graph_memory.py:29, search.py}.
 *
 * Tables (per ARCHITECTURE.md §4):
 *   conversations, semantic_memory, session_state, episodes,
 *   enhanced_episodes
 *
 * Decay: apply_decay = decays * 0.95 per cycle; prune < 0.01.
 * Mode: WAL + busy_timeout=5000ms, BEGIN IMMEDIATE on writes
 * (FIX from review for SQLite contention with DreamLoop).
 */
#ifndef HEART_LIMPHA_PER_VOICE_H
#define HEART_LIMPHA_PER_VOICE_H

#include <stdint.h>

typedef struct HeartLimphaVoice HeartLimphaVoice;

HeartLimphaVoice *heart_limpha_voice_open(const char *db_path);
void              heart_limpha_voice_close(HeartLimphaVoice *lv);
int               heart_limpha_voice_write_turn(HeartLimphaVoice *lv,
                                                 const char *text,
                                                 const char *role);
int               heart_limpha_voice_read_recent(HeartLimphaVoice *lv,
                                                  char *buf, int n,
                                                  int max_turns);
int               heart_limpha_voice_apply_decay(HeartLimphaVoice *lv);

#endif
