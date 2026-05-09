# heart.c — Makefile (skeleton, expanded after Opus review of plan)
#
# Build matrix per ARCHITECTURE.md §10. Each voice compiles into its own .o
# because yent_forward.h and resonance_forward.h define globally-named
# Weights structs and reuse globals (V/E/H/D/B/M/T/R, kv_k/kv_v) that
# CANNOT live in the same translation unit. Heart binary links the .o's
# behind narrow extern entry points.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -I. -Itools -Icore
LDFLAGS ?=
LIBS    ?= -lm -lopenblas -lsqlite3

# Termux openblas via pkg-config
ifeq ($(BLAS),on)
    CFLAGS  += -DUSE_BLAS $(shell pkg-config --cflags openblas)
    LDFLAGS += $(shell pkg-config --libs-only-L openblas)
endif

.PHONY: all yent arianna leo doe field soul sartre limpha kk heart smoke test clean

all: heart

# ── voices (separate TUs) ──────────────────────────────────────────────
yent:
	@echo "TODO: voices/yent/yent_main.c (jannus-r 12-step + yent_forward.h)"

arianna:
	@echo "TODO: voices/arianna/arianna_main.c (resonance_forward.h)"

leo:
	@echo "TODO: voices/leo/leo_main.c (Janus 170M + leo_anchors.h + leo_bootstrap.h)"

doe:
	@echo "TODO: voices/doe/doe_main.c (peer) + voices/doe/doe_meta.c (parliament over 3)"

# ── core (shared layer) ────────────────────────────────────────────────
field:
	@echo "TODO: core/field_clock.{c,h} — Klaus heritage"

soul:
	@echo "TODO: core/soul.{c,h} — InnerArianna pattern"

sartre:
	@echo "TODO: core/sartre_phone.{c,h}"

# ── memory layer ───────────────────────────────────────────────────────
limpha:
	@echo "TODO: limpha/per_voice + limpha/shared (SQLite-linked)"

kk:
	@echo "TODO: kk/kk_kernel.{c,h} — Dario heritage, FTS5"

# ── daemon ─────────────────────────────────────────────────────────────
heart: yent arianna leo doe field soul sartre limpha kk
	@echo "TODO: runtime/heart_main.c — daemon, schedule + selector + dispatch"

# ── tests ──────────────────────────────────────────────────────────────
smoke:
	@echo "TODO: 4-voice smoke run, 50 tokens each, 0 NaN, KK ingests, field selects"

test:
	@echo "TODO: per-component unit tests"

clean:
	rm -f *.o *.a *.so heart heart_main \
	      voices/*/yent voices/*/arianna voices/*/leo voices/*/doe
	rm -f core/*.o limpha/*.o kk/*.o runtime/*.o voices/*/*.o
