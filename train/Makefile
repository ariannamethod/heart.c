# heart.c train/ — LoRA SFT trainers (C, notorch, Chuck)
#
# Compiles against vendored notorch sources (../../notorch/ on pod, or
# ~/arianna/notorch/ via env override). No PyTorch / no Python in train path.

NOTORCH_DIR ?= /workspace/heart.c-runpod/notorch
NT_SRC      = $(NOTORCH_DIR)/notorch.c $(NOTORCH_DIR)/gguf.c
NT_INC      = -I$(NOTORCH_DIR)

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
endif
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
endif

CC      = cc
CFLAGS  = -O2 -Wall -Wextra -std=c11 $(NT_INC) $(BLAS_FLAGS)
LDFLAGS = -lm

.PHONY: all clean

all: train_arianna_lora train_doe_lora

train_arianna_lora: train_arianna_lora.c $(NOTORCH_DIR)/notorch.h $(NOTORCH_DIR)/gguf.h
	$(CC) $(CFLAGS) -o $@ train_arianna_lora.c $(NT_SRC) $(LDFLAGS)
	@echo "Built: $@"

train_doe_lora: train_doe_lora.c $(NOTORCH_DIR)/notorch.h $(NOTORCH_DIR)/gguf.h
	$(CC) $(CFLAGS) -o $@ train_doe_lora.c $(NT_SRC) $(LDFLAGS)
	@echo "Built: $@"

clean:
	rm -f train_arianna_lora train_doe_lora
