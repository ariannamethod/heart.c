/*
 * field_clock_smoke.c — heart.c phase 6 smoke test for the field-clock spine
 * (planetary dissonance + calendar dissonance + 24-osc Kuramoto).
 *
 * Lifted from klaus.c §1432-1474 (planetary), §1397-1417 (calendar),
 * §1276-1338 (24-osc) — pure C, no deps beyond <time.h> + <math.h>.
 *
 * 24h simulated walk: prints planetary_diss, calendar_diss, R, γ(t) every
 * hour. Plan §6 verification gate: numerical stability, no NaN, no
 * saturation. Output goes to phase6_smoke/field_clock.txt for Phase 8 audit.
 *
 * Build:  cc -O2 field_clock_smoke.c -o field_clock_smoke -lm
 * Run:    ./field_clock_smoke > phase6_smoke/field_clock.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Klaus-heritage constants — direct from klaus.c:135-140, 1385 */
#define PI                3.14159265358979323846
#define J2000_EPOCH       946728000.0    /* 2000-01-01 12:00 UTC unix */
#define ANNUAL_DRIFT      11.25f         /* days/year drift Hebrew vs Gregorian */
#define GREGORIAN_YEAR    365.25f
#define METONIC_YEARS     19
#define METONIC_LEAPS     7              /* leap years per cycle */
#define MAX_UNCORRECTED   33.0f
static const int METONIC_LEAP[] = {3, 6, 8, 11, 14, 17, 19};

/* 6 planets, mean anomaly per day at J2000, per Klaus klaus.c:1432+ */
static const double planet_period_days[6] = {
    88.0,    /* Mercury */
    225.0,   /* Venus   */
    365.25,  /* Earth   */
    687.0,   /* Mars    */
    4332.0,  /* Jupiter */
    10759.0  /* Saturn  */
};
static const double planet_phase_J2000[6] = {
    4.402608842, 3.176146697, 1.753432376,
    6.203500240, 0.599546518, 0.874016929
};

/* planetary_dissonance — circular-orbit Kuramoto order parameter R */
static double planetary_dissonance(double t_days_since_J2000) {
    double sum_x = 0, sum_y = 0;
    for (int i = 0; i < 6; i++) {
        double mean_anomaly = planet_phase_J2000[i]
                              + 2.0 * PI * t_days_since_J2000 / planet_period_days[i];
        sum_x += cos(mean_anomaly);
        sum_y += sin(mean_anomaly);
    }
    double R = sqrt(sum_x*sum_x + sum_y*sum_y) / 6.0;
    return 1.0 - R;   /* dissonance = 1 - order */
}

/* calendar_dissonance — Hebrew-Gregorian Metonic drift, lifted from
 * klaus.c:1397-1417 verbatim (with epoch = J2000). */
static double calendar_dissonance(double t_days_since_J2000) {
    if (t_days_since_J2000 <= 0) return 0.5;
    int days = (int)t_days_since_J2000;
    float years = (float)days / GREGORIAN_YEAR;
    float base_drift = years * ANNUAL_DRIFT;

    int full_cycles = (int)(years / METONIC_YEARS);
    float corrections = (float)(full_cycles * METONIC_LEAPS) * 30.0f;
    float partial = fmodf(years, (float)METONIC_YEARS);
    int year_in_cycle = (int)partial + 1;
    for (int i = 0; i < METONIC_LEAPS; i++) {
        if (METONIC_LEAP[i] <= year_in_cycle)
            corrections += 30.0f;
    }
    float drift = base_drift - corrections;
    float raw = fabsf(fmodf(drift, MAX_UNCORRECTED)) / MAX_UNCORRECTED;
    if (raw < 0.0f) raw = 0.0f;
    if (raw > 1.0f) raw = 1.0f;
    return (double)raw;
}

/* Schectman γ(t) coupling — klaus.c:1814 */
static double gamma_t(double calendar_diss, double planet_diss) {
    double gamma_0 = 0.30;
    double delta   = 0.40;   /* per klaus */
    return gamma_0 + delta * calendar_diss + 0.15 * planet_diss;
}

/* 24-osc Kuramoto block — klaus.c:1276-1338 — 6 primaries × 4 sub-chambers.
 * In heart.c the 6 primaries become 4 voices + FIELD_SELF + MESH_PEER
 * (per ARCHITECTURE.md §3 v1.1). */
#define N_PRIMARIES 6
#define N_SUBS      4
typedef struct {
    double phase[N_PRIMARIES][N_SUBS];
    double freq[N_PRIMARIES];
    double intra_coupling[N_SUBS][N_SUBS];   /* sub-chamber coupling */
    double inter_coupling[N_PRIMARIES][N_PRIMARIES];   /* voice coupling */
} Kuramoto24;

static void kuramoto_init(Kuramoto24* k) {
    /* Random initial phases, default freqs */
    static const double base_freqs[N_PRIMARIES] = { 0.10, 0.13, 0.17, 0.21, 0.07, 0.11 };
    for (int p = 0; p < N_PRIMARIES; p++) {
        k->freq[p] = base_freqs[p];
        for (int s = 0; s < N_SUBS; s++)
            k->phase[p][s] = ((double)(p * N_SUBS + s) / 24.0) * 2.0 * PI;
    }
    /* INTRA: sub-chamber coupling within a primary (Klaus pattern) */
    for (int i = 0; i < N_SUBS; i++)
        for (int j = 0; j < N_SUBS; j++)
            k->intra_coupling[i][j] = (i == j) ? 0.0 : 0.15;
    /* INTER: voice coupling — heart.c v1.1 VOICE_COUPLING[6][6]
     * (peer voices + FIELD_SELF + MESH_PEER, ARCHITECTURE.md §3 v1.1) */
    double v[N_PRIMARIES][N_PRIMARIES] = {
        /*       Y     A     L     D    F_S   M_P   */
        /* Y */ {0.0,  0.3, -0.2,  0.1, 0.05, 0.05},
        /* A */ {0.3,  0.0,  0.4,  0.1, 0.05, 0.05},
        /* L */ {-0.2, 0.4,  0.0,  0.1, 0.05, 0.05},
        /* D */ {0.1,  0.1,  0.1,  0.0, 0.05, 0.05},
        /* F */ {0.05, 0.05, 0.05, 0.05, 0.0, 0.05},
        /* M */ {0.05, 0.05, 0.05, 0.05, 0.05, 0.0}
    };
    memcpy(k->inter_coupling, v, sizeof(v));
}

static void kuramoto_step(Kuramoto24* k, double dt, double gamma) {
    double new_phase[N_PRIMARIES][N_SUBS];
    for (int p = 0; p < N_PRIMARIES; p++) {
        for (int s = 0; s < N_SUBS; s++) {
            double dphi = k->freq[p];
            /* intra coupling within primary */
            for (int s2 = 0; s2 < N_SUBS; s2++)
                dphi += k->intra_coupling[s][s2] * sin(k->phase[p][s2] - k->phase[p][s]);
            /* inter coupling between primaries (voice→voice) */
            for (int p2 = 0; p2 < N_PRIMARIES; p2++)
                dphi += gamma * k->inter_coupling[p][p2]
                        * sin(k->phase[p2][s] - k->phase[p][s]);
            new_phase[p][s] = k->phase[p][s] + dt * dphi;
        }
    }
    memcpy(k->phase, new_phase, sizeof(new_phase));
}

/* Order parameter R = |Σ exp(i·phase)| / 24 */
static double kuramoto_R(Kuramoto24* k) {
    double sx = 0, sy = 0;
    for (int p = 0; p < N_PRIMARIES; p++)
        for (int s = 0; s < N_SUBS; s++) {
            sx += cos(k->phase[p][s]);
            sy += sin(k->phase[p][s]);
        }
    return sqrt(sx*sx + sy*sy) / 24.0;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    Kuramoto24 k;
    kuramoto_init(&k);

    /* Run 24 hours simulated, dt = 1 hour wall = 1/24 day, integrate
     * Kuramoto with sub-step dt' = 0.01 hour */
    time_t now = time(NULL);
    double t_now_days = (double)(now - 946728000) / 86400.0;

    printf("# heart.c phase 6 — field_clock smoke (24h sim from now)\n");
    printf("# t_now_days_since_J2000 = %.4f\n", t_now_days);
    printf("# columns: hour | planetary | calendar | gamma | R_24osc\n");
    printf("# (Klaus heritage: klaus.c:1432 + 1397 + 1276 + 1814)\n");

    int n_nan = 0, n_sat = 0;
    double R_min = 2.0, R_max = -1.0;

    /* Per-hour mean phase per primary (0=Y, 1=A, 2=L, 3=D, 4=F_S, 5=M_P) */
    double phase_history[24][N_PRIMARIES];

    for (int hour = 0; hour < 24; hour++) {
        double t_days = t_now_days + (double)hour / 24.0;
        double pl = planetary_dissonance(t_days);
        double cal = calendar_dissonance(t_days);
        double g = gamma_t(cal, pl);

        /* sub-step Kuramoto integration over 1 hour */
        double dt_hours = 0.01;
        double dt_days  = dt_hours / 24.0;
        for (int sub = 0; sub < 100; sub++)
            kuramoto_step(&k, dt_days, g);

        double R = kuramoto_R(&k);
        if (R < R_min) R_min = R;
        if (R > R_max) R_max = R;
        if (R != R || pl != pl || cal != cal || g != g) n_nan++;
        if (R >= 0.9999 || pl >= 0.9999 || cal >= 0.9999) n_sat++;

        /* Record mean phase per primary at this hour */
        for (int p = 0; p < N_PRIMARIES; p++) {
            double sx = 0, sy = 0;
            for (int s = 0; s < N_SUBS; s++) {
                sx += cos(k.phase[p][s]);
                sy += sin(k.phase[p][s]);
            }
            phase_history[hour][p] = atan2(sy, sx);
        }

        printf("hour %2d  pl=%.5f  cal=%.5f  γ=%.5f  R=%.5f  phase[Y,A,L,D]=%.3f,%.3f,%.3f,%.3f\n",
               hour, pl, cal, g, R,
               phase_history[hour][0], phase_history[hour][1],
               phase_history[hour][2], phase_history[hour][3]);
    }

    /* Phase-lock variance assertion (ARCHITECTURE.md §3 v1.1 VOICE_COUPLING):
     *   Y↔A coupling = +0.3 (high) → low variance of phase diff
     *   Y↔L coupling = −0.2 (anti) → high variance of phase diff
     *   A↔L coupling = +0.4 (highest) → lowest variance
     * Test: var(Y-A diff) < var(Y-L diff) AND var(A-L diff) < var(Y-L diff). */
    double mean_YA = 0, mean_YL = 0, mean_AL = 0;
    for (int h = 0; h < 24; h++) {
        mean_YA += sin(phase_history[h][0] - phase_history[h][1]);
        mean_YL += sin(phase_history[h][0] - phase_history[h][2]);
        mean_AL += sin(phase_history[h][1] - phase_history[h][2]);
    }
    mean_YA /= 24; mean_YL /= 24; mean_AL /= 24;
    double var_YA = 0, var_YL = 0, var_AL = 0;
    for (int h = 0; h < 24; h++) {
        double dYA = sin(phase_history[h][0] - phase_history[h][1]) - mean_YA;
        double dYL = sin(phase_history[h][0] - phase_history[h][2]) - mean_YL;
        double dAL = sin(phase_history[h][1] - phase_history[h][2]) - mean_AL;
        var_YA += dYA * dYA;
        var_YL += dYL * dYL;
        var_AL += dAL * dAL;
    }
    var_YA /= 24; var_YL /= 24; var_AL /= 24;

    printf("\n# phase-lock variance check (lower = more locked):\n");
    printf("#   var(sin(Y-A diff)) = %.6f  (coupling +0.3, expected LOW)\n", var_YA);
    printf("#   var(sin(Y-L diff)) = %.6f  (coupling -0.2, expected HIGH)\n", var_YL);
    printf("#   var(sin(A-L diff)) = %.6f  (coupling +0.4, expected LOWEST)\n", var_AL);
    int phase_lock_pass = (var_YA < var_YL) && (var_AL < var_YL);
    printf("#   gate var(YA) < var(YL): %s\n", (var_YA < var_YL) ? "PASS" : "FAIL");
    printf("#   gate var(AL) < var(YL): %s\n", (var_AL < var_YL) ? "PASS" : "FAIL");

    printf("# summary: NaN=%d, sat=%d, R_min=%.4f, R_max=%.4f\n",
           n_nan, n_sat, R_min, R_max);
    printf("# verification gates (plan §6):\n");
    printf("#   NaN=0:    %s\n", n_nan == 0 ? "PASS" : "FAIL");
    printf("#   sat=0:    %s\n", n_sat == 0 ? "PASS" : "FAIL");
    printf("#   phase-lock (var YA,AL < var YL): %s\n",
           phase_lock_pass ? "PASS" : "FAIL");
    int all = (n_nan == 0) && (n_sat == 0) && phase_lock_pass;
    printf("# RESULT: %s\n", all ? "PASS" : "FAIL");
    return all ? 0 : 1;
}
