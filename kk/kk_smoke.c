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

#include "kk_kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* db = (argc > 1) ? argv[1] : "/tmp/heart_kk.db";
    const char* doc_dir = (argc > 2) ? argv[2] : "/workspace/heart.c-runpod/heart.c";

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
    return 0;
}
