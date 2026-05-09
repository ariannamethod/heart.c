# heart.c train/ — LoRA SFT trainers (C, notorch, Chuck)
#
# Compiles against vendored notorch sources (../../notorch/ on pod, or
# ~/arianna/notorch/ via env override). No PyTorch / no Python in train path.

NOTORCH_DIR ?= /workspace/notorch
DARIO_DIR   ?= /workspace/dario
HEART_TOOLS ?= /workspace/heart.c/tools
AML_DIR     ?= /workspace/ariannamethod.ai/core
NT_SRC      = $(NOTORCH_DIR)/notorch.c $(NOTORCH_DIR)/gguf.c
NT_INC      = -I$(NOTORCH_DIR) -I$(DARIO_DIR) -I$(HEART_TOOLS) -I$(AML_DIR)

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  BLAS_DEFS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK
  BLAS_LIBS = -framework Accelerate
endif
ifeq ($(UNAME), Linux)
  BLAS_DEFS = -DUSE_BLAS
  BLAS_LIBS = -lopenblas
endif

# CUDA: enabled when nvcc is available and CUDA=on (default on Linux+nvcc).
CUDA ?= auto
NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(CUDA), on)
  USE_CUDA := 1
endif
ifeq ($(CUDA), auto)
  ifneq ($(NVCC),)
    USE_CUDA := 1
  endif
endif

ifeq ($(USE_CUDA), 1)
  CUDA_DEFS = -DUSE_CUDA
  CUDA_LIBS = -lcublas -lcudart -L/usr/local/cuda/lib64
  CUDA_OBJ  = notorch_cuda.o
else
  CUDA_DEFS =
  CUDA_LIBS =
  CUDA_OBJ  =
endif

CC      = cc
CFLAGS  = -O2 -Wall -Wextra -std=c11 $(NT_INC) $(BLAS_DEFS) $(CUDA_DEFS)
LDFLAGS = -lm $(BLAS_LIBS) $(CUDA_LIBS)

.PHONY: all clean

all: train_arianna_lora train_doe_lora merge_arianna_lora merge_doe_lora infer_resonance

notorch_cuda.o: $(NOTORCH_DIR)/notorch_cuda.cu $(NOTORCH_DIR)/notorch_cuda.h
	nvcc -O2 -DUSE_CUDA -c $< -o $@

train_arianna_lora: train_arianna_lora.c $(NOTORCH_DIR)/notorch.h $(NOTORCH_DIR)/gguf.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ train_arianna_lora.c $(NT_SRC) $(CUDA_OBJ) $(LDFLAGS)
	@echo "Built: $@ (USE_CUDA=$(USE_CUDA))"

train_doe_lora: train_doe_lora.c $(NOTORCH_DIR)/notorch.h $(NOTORCH_DIR)/gguf.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ train_doe_lora.c $(NT_SRC) $(CUDA_OBJ) $(LDFLAGS)
	@echo "Built: $@ (USE_CUDA=$(USE_CUDA))"

# Mergers — pure C, no notorch tape, only matmul. Build CPU-only with libm.
merge_arianna_lora: merge_arianna_lora.c
	$(CC) -O2 -Wall -Wextra -std=c11 -o $@ merge_arianna_lora.c -lm
	@echo "Built: $@"

merge_doe_lora: merge_doe_lora.c
	$(CC) -O2 -Wall -Wextra -std=c11 -o $@ merge_doe_lora.c -lm
	@echo "Built: $@"

# Inference driver for Resonance — links resonance_forward.h (vendored) +
# notorch.c + gguf.c + ariannamethod.c (AML field overlay).
# _POSIX_C_SOURCE=200809L exposes CLOCK_MONOTONIC on Linux glibc.
infer_resonance: infer_resonance.c
	$(CC) -O2 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L $(BLAS_DEFS) \
	  -I$(NOTORCH_DIR) -I$(DARIO_DIR) -I$(HEART_TOOLS) -I$(AML_DIR) \
	  -o $@ infer_resonance.c $(NT_SRC) $(AML_DIR)/ariannamethod.c \
	  -lm $(BLAS_LIBS)
	@echo "Built: $@"

clean:
	rm -f train_arianna_lora train_doe_lora merge_arianna_lora merge_doe_lora infer_resonance notorch_cuda.o
