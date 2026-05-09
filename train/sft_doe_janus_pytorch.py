"""
sft_doe_janus_pytorch.py — Manual LoRA SFT for Janus 177M on DoE personality corpus.

PyTorch port of train_doe_lora.c. Mirror of nanoarianna/runpod/sft_arianna_pytorch.py
adapted to Janus arch:
  - Class JanusGPT, config JanusConfig (from nanochat.janus_gpt)
  - Forward sig: loss = model(idx, targets=y) | logits = model(idx)
    (NOT (logits, loss) tuple like Resonance)
  - LoRA targets per layer: c_q / c_k / c_v / c_proj + w_gate / w_up / w_down (7)
  - Tokenizer: pickle (tokenizer.pkl). Callable interface: tok(text) → ids
  - Vocab=32000 (BPE 22k effective for base22k ckpt)

Recipe (override of C train_doe_lora.c r=16 α=32 1000 steps):
  rank=64, alpha=128, scaling=2.0, AdamW betas=(0.9, 0.95) wd=0.01 lr=2e-4
  cosine warmup=30, bf16 autocast, grad clip 1.0, --steps 2000 default.

Corpus format: JSONL {"messages": [{"role":"user","content":...},
                                   {"role":"assistant","content":...}]}
Format string per C: "### Question: <q>\n\n### Answer: <a>\n"
Mask: prefix + question + "\n\n### Answer: " → -100; answer body → supervised; trailing \n → -100.

Usage:
  python sft_doe_janus_pytorch.py --mode gate --gate 1
  python sft_doe_janus_pytorch.py --mode train \
      --base /workspace/janus/janus_177m_v4_base_22442.pt \
      --tokenizer /workspace/janus/tokenizer.pkl \
      --corpus /workspace/data/doe_pure.jsonl \
      --steps 2000 --rank 64 --alpha 128
"""

import argparse
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


SEED = 3407


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer adapter (handle pickle of unknown interface)
# ─────────────────────────────────────────────────────────────────────────────


class TokenizerAdapter:
    """Wraps unpickled tokenizer to expose .encode/.decode regardless of native API.

    nanochat tokenizer is callable: tok(text, prepend='<|bos|>') → list[int]
    tiktoken encoder uses tok.encode_ordinary(text) → list[int]
    Generic BPE may use tok.encode(text)
    Fallback: raise clear error.
    """

    def __init__(self, raw):
        self.raw = raw

    def encode(self, text: str) -> List[int]:
        # Try most-specific first
        if hasattr(self.raw, "encode_ordinary"):
            return list(self.raw.encode_ordinary(text))
        if hasattr(self.raw, "encode"):
            try:
                return list(self.raw.encode(text))
            except TypeError:
                pass
        if callable(self.raw):
            return list(self.raw(text))
        raise RuntimeError(f"TokenizerAdapter: unknown interface for {type(self.raw)}")

    def decode(self, ids) -> str:
        if hasattr(self.raw, "decode"):
            return self.raw.decode(list(ids))
        raise RuntimeError(f"TokenizerAdapter: no decode on {type(self.raw)}")

    @property
    def vocab_size(self) -> int:
        for attr in ("vocab_size", "n_vocab", "size"):
            if hasattr(self.raw, attr):
                v = getattr(self.raw, attr)
                return int(v() if callable(v) else v)
        return 32000  # Janus default


def load_tokenizer(path: str) -> TokenizerAdapter:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return TokenizerAdapter(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Manual LoRA layer (no PEFT)
# ─────────────────────────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """y = base(x) + (x @ A.T @ B.T) * scaling. A kaiming, B zero-init."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def wrap_with_lora(model: nn.Module, rank: int, alpha: float, target_suffixes: List[str]) -> int:
    replaced = 0
    target_set = set(target_suffixes)
    targets = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = path.rsplit(".", 1)[-1]
        if leaf not in target_set:
            continue
        parent_path = path.rsplit(".", 1)[0] if "." in path else ""
        parent = model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        targets.append((parent, leaf, module))

    for parent, leaf, child in targets:
        new_layer = LoRALinear(child, rank=rank, alpha=alpha)
        setattr(parent, leaf, new_layer)
        replaced += 1

    for name, param in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return replaced


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_lora_adapter(model: nn.Module, path: Path, rank: int, alpha: float) -> None:
    state = {}
    for name, param in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            state[name] = param.detach().cpu()
    torch.save(
        {
            "lora_state": state,
            "rank": rank,
            "alpha": alpha,
            "targets": [name.rsplit(".", 1)[0] for name in state.keys() if name.endswith(".A")],
        },
        path,
    )


def load_lora_adapter(model: nn.Module, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["lora_state"]
    own = dict(model.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in own:
            missing.append(name)
            continue
        own[name].data.copy_(tensor.to(own[name].device, dtype=own[name].dtype))
    if missing:
        raise RuntimeError(f"LoRA load: missing keys: {missing[:5]}...")


# ─────────────────────────────────────────────────────────────────────────────
# DoE corpus parser (JSONL messages format)
# ─────────────────────────────────────────────────────────────────────────────


def parse_doe_pairs(corpus_path: Path) -> List[Tuple[str, str]]:
    """Parse JSONL lines of {messages: [{role,user,content},{role,assistant,content}]} → [(q, a), ...].

    Skips malformed lines and pairs missing user or assistant role.
    """
    pairs = []
    with open(corpus_path, encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msgs = obj.get("messages") or []
            if len(msgs) < 2:
                continue
            q = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            a = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
            if q is None or a is None:
                continue
            pairs.append((q, a))
    return pairs


PROMPT_TEMPLATE = "### Question: {q}\n\n### Answer: "
ANSWER_SUFFIX = "\n"


def encode_pair_with_mask(
    tokenizer: TokenizerAdapter, q: str, a: str, max_seq: int
) -> Tuple[List[int], List[int]]:
    """Encode "### Question: <q>\n\n### Answer: <a>\n" with prefix masked (-100), body supervised."""
    prefix = PROMPT_TEMPLATE.format(q=q)
    body = a
    suffix = ANSWER_SUFFIX

    prefix_ids = tokenizer.encode(prefix)
    body_ids = tokenizer.encode(body)
    suffix_ids = tokenizer.encode(suffix)

    input_ids = prefix_ids + body_ids + suffix_ids
    labels = [-100] * len(prefix_ids) + list(body_ids) + [-100] * len(suffix_ids)

    if len(input_ids) > max_seq:
        input_ids = input_ids[:max_seq]
        labels = labels[:max_seq]

    assert len(input_ids) == len(labels), f"len mismatch: {len(input_ids)} vs {len(labels)}"
    return input_ids, labels


def pad_batch(items: List[Tuple[List[int], List[int]]], max_seq: int, pad_id: int = 0):
    B = len(items)
    T = max_seq
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    # JanusGPT cross_entropy uses ignore_index=-1, NOT -100. Mask = -1.
    labels = torch.full((B, T), -1, dtype=torch.long)
    for i, (ids, lbl) in enumerate(items):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        # Translate -100 → -1 for Janus ignore_index
        lbl_translated = [(-1 if x == -100 else x) for x in lbl]
        labels[i, :L] = torch.tensor(lbl_translated, dtype=torch.long)
    return input_ids, labels


# ─────────────────────────────────────────────────────────────────────────────
# Generate (Janus forward returns logits when targets is None)
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: TokenizerAdapter,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 40,
    device: torch.device = None,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ctx = ids_t[:, -1024:]  # Janus T=1024
        logits = model(ctx)  # NO targets → returns logits only
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids_t = torch.cat([ids_t, next_id], dim=1)
    return tokenizer.decode(ids_t[0].tolist())


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────


def cosine_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# Train loop (Janus forward sig: loss = model(x, targets=y))
# ─────────────────────────────────────────────────────────────────────────────


def train_loop(
    model, tokenizer, train_pairs, val_pairs, args, device,
    log_path: Path, ckpt_path: Path, backup_cmd_template: str = None,
):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )

    rng = np.random.default_rng(SEED)
    pair_indices = np.arange(len(train_pairs))

    step = 0
    best_val = float("inf")

    log = open(log_path, "w")
    log.write(
        f"# step | train_loss | val_loss | lr | grad_norm | trainable={count_trainable(model)}\n"
    )
    log.flush()

    base_val = compute_val_loss(model, tokenizer, val_pairs, args, device)
    log.write(f"# base_val_loss = {base_val:.4f}\n")
    log.flush()
    print(f"[base] val_loss = {base_val:.4f}")

    while step < args.steps:
        optim.zero_grad(set_to_none=True)

        train_loss_accum = 0.0
        for _ in range(args.grad_accum):
            batch_idx = rng.choice(pair_indices, size=args.batch_size, replace=False)
            batch = [
                encode_pair_with_mask(tokenizer, train_pairs[i][0], train_pairs[i][1], args.max_seq)
                for i in batch_idx
            ]
            input_ids, labels = pad_batch(batch, args.max_seq)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = model(input_ids, targets=labels)
                loss = loss / args.grad_accum
            loss.backward()
            train_loss_accum += loss.item() * args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()

        lr = cosine_lr(step, args.warmup, args.steps, args.lr)
        for pg in optim.param_groups:
            pg["lr"] = lr

        optim.step()

        train_loss = train_loss_accum / args.grad_accum

        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d}/{args.steps} | train {train_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f}"
            )

        if (step + 1) % args.val_every == 0 or step == args.steps - 1:
            val_loss = compute_val_loss(model, tokenizer, val_pairs, args, device)
            log.write(f"{step} | {train_loss:.4f} | {val_loss:.4f} | {lr:.4e} | {grad_norm:.4f}\n")
            log.flush()
            print(f"  ── val {step}: {val_loss:.4f} (best {best_val:.4f})")
            if val_loss < best_val:
                best_val = val_loss
                save_lora_adapter(model, ckpt_path, args.rank, args.alpha)
                print(f"  ── new best, saved {ckpt_path}")
                if backup_cmd_template:
                    cmd = backup_cmd_template.format(src=str(ckpt_path))
                    subprocess.Popen(cmd, shell=True)
        else:
            log.write(f"{step} | {train_loss:.4f} |  | {lr:.4e} | {grad_norm:.4f}\n")
            log.flush()

        step += 1

    log.close()
    return {"best_val": best_val, "base_val": base_val, "final_step": step}


@torch.no_grad()
def compute_val_loss(model, tokenizer, val_pairs, args, device) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    for q, a in val_pairs:
        ids, lbl = encode_pair_with_mask(tokenizer, q, a, args.max_seq)
        input_ids, labels = pad_batch([(ids, lbl)], args.max_seq)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model(input_ids, targets=labels)
        total_loss += loss.item()
        total_count += 1
    model.train()
    return total_loss / max(total_count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-temp eval
# ─────────────────────────────────────────────────────────────────────────────


EVAL_PROMPTS = [
    "### Question: What does personality fine-tuning do?\n\n### Answer:",
    "### Question: How does the parliament work?\n\n### Answer:",
    "### Question: Explain DoE.\n\n### Answer:",
    "### Question: What is RRPRAM?\n\n### Answer:",
]
EVAL_TEMPS = [0.3, 0.5, 0.7, 0.9, 1.1]
EVAL_TOPKS = [40, 0]


def multi_temp_eval(model, tokenizer, device, out_path: Path) -> dict:
    f = open(out_path, "w")
    cells = []
    for prompt in EVAL_PROMPTS:
        for temp in EVAL_TEMPS:
            for tk in EVAL_TOPKS:
                out = generate(model, tokenizer, prompt, 80, temp, tk, device)
                f.write(f"=== prompt={prompt!r} | temp={temp} | top_k={tk} ===\n{out}\n\n")
                cells.append({"prompt": prompt, "temp": temp, "top_k": tk, "out": out})
    f.close()
    return {"cells": cells, "n_cells": len(cells)}


# ─────────────────────────────────────────────────────────────────────────────
# Base loader
# ─────────────────────────────────────────────────────────────────────────────


def _load_janus_base(args):
    """Load JanusGPT base from .pt file. Tokenizer from pickle.

    Requires nanochat lib installed (pod-side: uv pip install nanochat or local clone).
    """
    try:
        from nanochat.janus_gpt import JanusGPT, JanusConfig
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import nanochat.janus_gpt — install nanochat lib first. {e}"
        )

    # Build config (Janus 177M v4 defaults match HF base22442 ckpt)
    config = JanusConfig(
        sequence_len=1024,
        vocab_size=32000,
        n_layer=20,
        n_head=10,
        n_kv_head=10,
        n_embd=640,
        mlp_hidden=1664,
        rrpram_T=1024,
        rrpram_rank=64,
    )

    model = JanusGPT(config)

    sd = torch.load(args.base, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[load] unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    tok = load_tokenizer(args.tokenizer)
    return model, tok


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight gates
# ─────────────────────────────────────────────────────────────────────────────


def gate_1_tokenizer(args):
    """Roundtrip + chat-format probe (POZOR #5 hypothesis #1)."""
    tok = load_tokenizer(args.tokenizer)
    text = "### Question: Who are you?\n\n### Answer: "
    ids = tok.encode(text)
    rt = tok.decode(ids)
    print(f"[gate1] vocab_size~{tok.vocab_size}, encoded {len(ids)} tokens")
    print(f"[gate1] roundtrip: {rt!r}")
    print("[gate1] PASS")


def gate_2_base_load(args):
    model, tok = _load_janus_base(args)
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, 32000, (1, 32))
        out = model(x)
        # Janus returns logits when no targets
        assert out.dim() == 3, f"expected (B,T,V) logits, got shape {out.shape}"
        assert torch.isfinite(out).all(), "non-finite logits"
        print(f"[gate2] forward OK, logits shape {out.shape}, finite")
    print("[gate2] PASS")


def gate_3_lora_inject(args):
    model, _ = _load_janus_base(args)
    targets = ["c_q", "c_k", "c_v", "c_proj", "w_gate", "w_up", "w_down"]
    n = wrap_with_lora(model, rank=args.rank, alpha=args.alpha, target_suffixes=targets)
    expected = 7 * 20  # 7 targets × 20 layers
    print(f"[gate3] replaced {n} modules, expected {expected}")
    assert n == expected, f"{n} != {expected}"
    trainable = count_trainable(model)
    print(f"[gate3] trainable params: {trainable:,} ({trainable/1e6:.2f}M)")
    for name, p in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            assert p.requires_grad, f"LoRA {name} should be trainable"
        else:
            assert not p.requires_grad, f"non-LoRA {name} should be frozen"
    print("[gate3] PASS")


def gate_4_lora_grad_smoke(args):
    model, _ = _load_janus_base(args)
    wrap_with_lora(model, rank=args.rank, alpha=args.alpha,
                   target_suffixes=["c_q", "c_k", "c_v", "c_proj", "w_gate", "w_up", "w_down"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    x = torch.randint(0, 32000, (1, 32))
    y = torch.randint(0, 32000, (1, 32))
    loss = model(x, targets=y)
    loss.backward()

    a_params = [(n, p) for n, p in model.named_parameters() if n.endswith(".A")]
    b_params = [(n, p) for n, p in model.named_parameters() if n.endswith(".B")]
    a_grad_zero = all((p.grad is None or p.grad.abs().sum().item() == 0.0) for _, p in a_params)
    b_grad_nonzero = any((p.grad is not None and p.grad.abs().sum().item() > 0.0) for _, p in b_params)
    print(f"[gate4] 1st backward: A grad zero = {a_grad_zero}, B grad nonzero = {b_grad_nonzero}")
    assert a_grad_zero, "A grad should be zero on 1st backward (B init zero)"
    assert b_grad_nonzero, "B grad should be nonzero on 1st backward"

    optim.step()
    optim.zero_grad()
    loss = model(x, targets=y)
    loss.backward()
    a_grad_nonzero = any((p.grad is not None and p.grad.abs().sum().item() > 0.0) for _, p in a_params)
    print(f"[gate4] 2nd backward: A grad nonzero = {a_grad_nonzero}")
    assert a_grad_nonzero, "A grad should be nonzero on 2nd backward"
    print("[gate4] PASS")


def gate_5_corpus(args):
    pairs = parse_doe_pairs(Path(args.corpus))
    print(f"[gate5] parsed {len(pairs)} (q, a) pairs from {args.corpus}")
    if not pairs:
        print("[gate5] FAIL — empty corpus")
        sys.exit(1)
    tok = load_tokenizer(args.tokenizer)
    sample = random.Random(SEED).sample(pairs, min(3, len(pairs)))
    for i, (q, a) in enumerate(sample):
        ids, lbl = encode_pair_with_mask(tok, q, a, args.max_seq)
        sup = sum(1 for x in lbl if x != -100)
        print(f"[gate5] pair {i}: {len(ids)} tokens, {sup} supervised")
        print(f"  q: {q[:80]}...")
        print(f"  a: {a[:80]}...")
    print("[gate5] DONE (manual review of mask)")


def gate_6_split(args):
    pairs = parse_doe_pairs(Path(args.corpus))
    n = len(pairs)
    rng = random.Random(SEED)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * 0.05))
    val_ids = sorted(indices[:n_val])
    train_ids = sorted(indices[n_val:])
    split_path = Path(args.split_dir) / "split_doe_v1.json"
    Path(args.split_dir).mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(
            {"train_pair_ids": train_ids, "val_pair_ids": val_ids,
             "seed": SEED, "ratio": 0.95, "n_pairs": n},
            f,
        )
    print(f"[gate6] {len(train_ids)} train, {len(val_ids)} val pairs → {split_path}")
    print("[gate6] PASS")


def gate_7_generate(args):
    model, tok = _load_janus_base(args)
    model.eval()
    prompt = "### Question: Who are you?\n\n### Answer:"
    out = generate(model, tok, prompt, 30, 0.7, 40, torch.device("cpu"))
    print(f"[gate7] prompt: {prompt!r}")
    print(f"[gate7] output: {out!r}")
    print("[gate7] DONE (manual review of base voice — should be coherent in chat-format)")


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval commands
# ─────────────────────────────────────────────────────────────────────────────


def cmd_train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    model, tok = _load_janus_base(args)
    print(f"[train] base loaded, {sum(p.numel() for p in model.parameters()):,} params")

    targets = ["c_q", "c_k", "c_v", "c_proj", "w_gate", "w_up", "w_down"]
    n = wrap_with_lora(model, rank=args.rank, alpha=args.alpha, target_suffixes=targets)
    print(f"[train] LoRA: {n} modules, trainable={count_trainable(model):,}")

    model.to(device)

    split_path = Path(args.split_dir) / "split_doe_v1.json"
    if not split_path.exists():
        print(f"[train] split missing — generating via gate_6")
        gate_6_split(args)
    split = json.load(open(split_path))
    pairs = parse_doe_pairs(Path(args.corpus))
    train_pairs = [pairs[i] for i in split["train_pair_ids"]]
    val_pairs = [pairs[i] for i in split["val_pair_ids"]]
    print(f"[train] {len(train_pairs)} train, {len(val_pairs)} val pairs")

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    log_path = Path(args.log_dir) / f"sft_doe_{ts}.log"
    ckpt_path = Path(args.ckpt_dir) / "lora_doe_best.pt"

    result = train_loop(
        model, tok, train_pairs, val_pairs, args, device,
        log_path, ckpt_path, args.backup_cmd,
    )
    print(f"[train] DONE: base_val={result['base_val']:.4f}, best_val={result['best_val']:.4f}, "
          f"delta={result['base_val']-result['best_val']:+.4f}")

    final_path = Path(args.ckpt_dir) / "lora_doe_final.pt"
    save_lora_adapter(model, final_path, args.rank, args.alpha)
    print(f"[train] final adapter saved {final_path}")

    eval_path = Path(args.eval_dir) / f"eval_doe_{ts}.txt"
    multi_temp_eval(model, tok, device, eval_path)
    print(f"[train] multi-temp eval saved {eval_path}")

    if args.backup_cmd:
        for p in (ckpt_path, final_path, log_path, eval_path):
            if p.exists():
                cmd = args.backup_cmd.format(src=str(p))
                subprocess.Popen(cmd, shell=True)


def cmd_eval(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok = _load_janus_base(args)
    targets = ["c_q", "c_k", "c_v", "c_proj", "w_gate", "w_up", "w_down"]
    wrap_with_lora(model, rank=args.rank, alpha=args.alpha, target_suffixes=targets)

    if not args.adapter:
        print("[eval] --adapter PATH required")
        sys.exit(1)
    load_lora_adapter(model, Path(args.adapter))
    model.to(device)
    model.eval()

    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    eval_path = Path(args.eval_dir) / f"eval_doe_{int(time.time())}.txt"
    multi_temp_eval(model, tok, device, eval_path)
    print(f"[eval] saved {eval_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gate", "train", "eval"])
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--base", default="/workspace/janus/janus_177m_v4_base_22442.pt")
    ap.add_argument("--tokenizer", default="/workspace/janus/tokenizer.pkl")
    ap.add_argument("--corpus", default="/workspace/data/doe_pure.jsonl")
    ap.add_argument("--ckpt_dir", default="/workspace/sft/checkpoints")
    ap.add_argument("--log_dir", default="/workspace/sft/logs")
    ap.add_argument("--eval_dir", default="/workspace/sft/eval")
    ap.add_argument("--split_dir", default="/workspace/sft")
    ap.add_argument("--adapter", default=None, help="adapter .pt for --mode eval")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=128.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_seq", type=int, default=512)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--backup_cmd", default=None,
                    help="Shell template, {src} = ckpt path. e.g. 'scp {src} neo:~/...'")
    args = ap.parse_args()

    set_seed()

    if args.mode == "gate":
        gates = {1: gate_1_tokenizer, 2: gate_2_base_load, 3: gate_3_lora_inject,
                 4: gate_4_lora_grad_smoke, 5: gate_5_corpus, 6: gate_6_split,
                 7: gate_7_generate}
        if args.gate not in gates:
            print(f"unknown gate {args.gate}; valid: {list(gates.keys())}")
            sys.exit(1)
        gates[args.gate](args)
        return

    if args.mode == "train":
        cmd_train(args)
        return

    if args.mode == "eval":
        cmd_eval(args)
        return


if __name__ == "__main__":
    main()
