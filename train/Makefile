# heart.c train/ — LoRA SFT trainers (C, notorch, Chuck)
#
# Compiles against vendored notorch sources (../../notorch/ on pod, or
# ~/arianna/notorch/ via env override). No PyTorch / no Python in train path.

NOTORCH_DIR ?= /workspace/notorch
DARIO_DIR   ?= /workspace/dario
NT_SRC      = $(NOTORCH_DIR)/notorch.c $(NOTORCH_DIR)/gguf.c
NT_INC      = -I$(NOTORCH_DIR) -I$(DARIO_DIR)

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

all: train_arianna_lora train_doe_lora

notorch_cuda.o: $(NOTORCH_DIR)/notorch_cuda.cu $(NOTORCH_DIR)/notorch_cuda.h
	nvcc -O2 -DUSE_CUDA -c $< -o $@

train_arianna_lora: train_arianna_lora.c $(NOTORCH_DIR)/notorch.h $(NOTORCH_DIR)/gguf.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ train_arianna_lora.c $(NT_SRC) $(CUDA_OBJ) $(LDFLAGS)
	@echo "Built: $@ (USE_CUDA=$(USE_CUDA))"

train_doe_lora: train_doe_lora.c $(NOTORCH_DIR)/notorch.h $(NOTORCH_DIR)/gguf.h $(CUDA_OBJ)
	$(CC) $(CFLAGS) -o $@ train_doe_lora.c $(NT_SRC) $(CUDA_OBJ) $(LDFLAGS)
	@echo "Built: $@ (USE_CUDA=$(USE_CUDA))"

clean:
	rm -f train_arianna_lora train_doe_lora notorch_cuda.o
