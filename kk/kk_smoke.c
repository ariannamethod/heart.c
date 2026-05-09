/*
 * kk_smoke.c — heart.c phase 5 KK ingest smoke test.
 *
 * Reuses dario/kk_kernel.{c,h} verbatim (no port — direct link).
 * Ingests heart.c canonical docs + queries for "resonance" and verifies
 * 7-signal scoring per Dario paper §6 Result #6:
 *   lexical 0.36, recency 0.12, trust 0.10, linkage 0.16,
 *   scope 0.10, namespace 0.08, freshness 0.08
 *
 * Build:
 *   cc -O2 -I/workspace/heart.c-runpod/dario \
 *      kk_smoke.c \
 *      /workspace/heart.c-runpod/dario/kk_kernel.c \
 *      -lsqlite3 -lm -o kk_smoke
 *
 * Output: per-query top-3 with resonance + component breakdowns.
 */

#include "/workspace/heart.c-runpod/dario/kk_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Phase 5 verification — assert published Dario §6 weights at 1e-6 */
static int verify_dario_section_6_policy(void) {
    const char* names[7] = {
        "lexical", "recency", "trust", "linkage",
        "scope", "namespace", "freshness"
    };
    const double expected[7] = { 0.36, 0.12, 0.10, 0.16, 0.10, 0.08, 0.08 };
    double computed[7];
    kk_get_default_weights(computed);
    int pass = 0;
    printf("\n=== Dario §6 7-signal policy verification (1e-6 tolerance) ===\n");
    for (int i = 0; i < 7; i++) {
        double diff = fabs(computed[i] - expected[i]);
        int ok = diff < 1e-6;
        printf("  signal=%-9s computed=%.6f  expected=%.6f  diff=%.2e  %s\n",
               names[i], computed[i], expected[i], diff, ok ? "PASS" : "FAIL");
        if (ok) pass++;
    }
    double total = 0;
    for (int i = 0; i < 7; i++) total += computed[i];
    printf("  total=%.6f (must = 1.000000): %s\n",
           total, fabs(total - 1.0) < 1e-6 ? "PASS" : "FAIL");
    printf("  policy verification: %d/7 weights match\n", pass);
    return (pass == 7) && (fabs(total - 1.0) < 1e-6);
}

int main(int argc, char** argv) {
    const char* db = (argc > 1) ? argv[1] : "/tmp/heart_kk.db";
    const char* doc_dir = (argc > 2) ? argv[2] : "/workspace/heart.c-runpod/heart.c";

    int policy_pass = verify_dario_section_6_policy();

    /* delete any existing DB so the smoke is reproducible */
    unlink(db);

    kk_ctx* k = kk_open(db);
    if (!k) { fprintf(stderr, "[kk] open failed\n"); return 1; }
    fprintf(stderr, "[kk] db open: %s\n", db);

    /* Ingest heart.c canonical docs */
    const char* docs[] = {
        "/workspace/heart.c-runpod/heart.c/README.md",
        "/workspace/heart.c-runpod/heart.c/ARCHITECTURE.md",
        "/workspace/heart.c-runpod/heart.c/SEED_DOCUMENT.md",
        "/workspace/heart.c-runpod/heart.c/ECOSYSTEM_LOG.md",
        "/workspace/heart.c-runpod/heart.c/docs/singularity_protocol.md",
        NULL
    };
    int ingested = 0;
    for (int i = 0; docs[i]; i++) {
        int rc = kk_ingest_file(k, docs[i], "public", "ground");
        if (rc > 0) {
            ingested++;
            fprintf(stderr, "[kk] ingested %s (%d chunks)\n", docs[i], rc);
        } else {
            fprintf(stderr, "[kk] ingest failed for %s (rc=%d)\n", docs[i], rc);
        }
    }
    fprintf(stderr, "[kk] %d docs ingested\n", ingested);
    (void)doc_dir;

    /* Test queries — heart.c domain */
    const char* queries[] = {
        "resonance",
        "DoE Parliament voices",
        "field clock planetary calendar",
        "Singularity Mode three strikes",
        NULL
    };
    for (int q = 0; queries[q]; q++) {
        kk_result* res = NULL;
        int n = kk_retrieve(k, queries[q], "public", NULL, 3,
                             KK_PROFILE_BALANCED, &res);
        printf("\n=== query: \"%s\" → %d results ===\n", queries[q], n);
        for (int i = 0; i < n && i < 3; i++) {
            printf("[%d] resonance=%.4f  lex=%.4f rec=%.4f trust=%.4f linkage=%.4f fresh=%.4f\n",
                   i, res[i].resonance, res[i].lexical, res[i].recency,
                   res[i].trust, res[i].linkage, res[i].freshness);
            const char* sha = res[i].sha256 ? res[i].sha256 : "?";
            printf("    sha=%.16s...\n", sha);
        }
        kk_free_results(res, n);
    }

    kk_close(k);
    fprintf(stderr, "\n[kk] smoke complete\n");
    printf("\n=== verification gate: Dario §6 policy = %s ===\n",
           policy_pass ? "PASS" : "FAIL");
    return policy_pass ? 0 : 1;
}
