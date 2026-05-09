/* heart.c — limpha/per_voice.c — STATUS: skeleton, RunPod Phase 5 */
#include "per_voice.h"

struct HeartLimphaVoice { int dummy; };

HeartLimphaVoice *heart_limpha_voice_open(const char *db_path)
{ (void)db_path; return 0; /* TODO sqlite3_open + WAL + schema */ }

void heart_limpha_voice_close(HeartLimphaVoice *lv) { (void)lv; }
int  heart_limpha_voice_write_turn(HeartLimphaVoice *lv, const char *t, const char *r)
{ (void)lv; (void)t; (void)r; return 0; /* BEGIN IMMEDIATE; INSERT... */ }
int  heart_limpha_voice_read_recent(HeartLimphaVoice *lv, char *buf, int n, int m)
{ (void)lv; (void)buf; (void)n; (void)m; return 0; }
int  heart_limpha_voice_apply_decay(HeartLimphaVoice *lv) { (void)lv; return 0; }
