/* heart.c — core/sartre_phone.h
 *
 * Phone-fitted SARTRE introspection (per ARCHITECTURE.md §7).
 * Lifts arianna.c/sartre/sartre.h, drops desktop-specific bits,
 * adds phone-specific fields (openblas_threads, mesh_agent_port,
 * mesh_peers).
 */
#ifndef HEART_SARTRE_PHONE_H
#define HEART_SARTRE_PHONE_H

#include <stdint.h>

#define HEART_SARTRE_MAX_MODULES 16
#define HEART_SARTRE_MAX_EVENTS  8
#define HEART_SARTRE_MAX_PEERS   8
#define HEART_SARTRE_EVENT_LEN   256

typedef struct {
    char  name[64];
    int   status;        /* 0=inactive 1=active 2=fault */
    float load;          /* 0–1 */
    long  last_active_ms;
    char  last_event[HEART_SARTRE_EVENT_LEN];
} HeartModuleInfo;

typedef struct {
    HeartModuleInfo modules[HEART_SARTRE_MAX_MODULES];
    int n_modules;
    char last_events[HEART_SARTRE_MAX_EVENTS][HEART_SARTRE_EVENT_LEN];
    int events_head;

    /* phone-specific substrate fields */
    int   total_ram_mb;
    int   free_ram_mb;
    int   cpu_cores;
    int   openblas_threads;     /* default 2 on phone-1 */
    int   mesh_agent_port;      /* 4747 */
    char  mesh_peers[HEART_SARTRE_MAX_PEERS][32];
    int   n_peers;

    /* inner world (subset; full clock state lives in field_clock.h) */
    float trauma, arousal, valence, coherence, prophecy_debt, entropy;
} HeartSartreState;

int  heart_sartre_init(HeartSartreState *s);
void heart_sartre_register_module(HeartSartreState *s, const char *name);
void heart_sartre_notify_event(HeartSartreState *s, const char *evt);
int  heart_sartre_detect_total_ram_mb(void);  /* /proc/meminfo */
int  heart_sartre_detect_cpu_cores(void);     /* nproc */

#endif /* HEART_SARTRE_PHONE_H */
