from __future__ import annotations

import glob
import json
import math
import os
import pickle
import random
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024/")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 200))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 32768))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 8192))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    max_val_batches: int = int(os.environ.get("MAX_VAL_BATCHES", 0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: float = float(os.environ.get("MLP_MULT", 3.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims: int = int(os.environ.get("ROPE_DIMS", 16))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    xsa_last_k_layers: int = int(os.environ.get("XSA_LAST_K_LAYERS", 0))

    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.04))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))

    use_ema: bool = bool(int(os.environ.get("USE_EMA", "0")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.999))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path.rstrip('/')}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path.rstrip('/')}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            if warmdown_start <= step < self.iterations:
                return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return remaining_ms / max(warmdown_ms, 1e-9)
        return 1.0


CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_scale", "q_gain")
INT8_KEEP_FLOAT_MAX_NUMEL = 65536
INT8_CLIP_Q = 99.99984 / 100.0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_master() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Runpod backbone script")
    torch.cuda.set_device(0)
    return 0, 1, 0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def zeropower_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    x = x / (torch.sqrt(torch.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.t()
    for _ in range(steps):
        a_mat = x @ x.t()
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.t()
    return x.to(dtype=g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern: str, rank: int, world_size: int, log_fn=None):
        all_files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.files = all_files[rank::world_size] if len(all_files) >= world_size else all_files
        self.epoch = 1
        self.file_idx = 0
        self.pos = 0
        self.log_fn = log_fn
        self.tokens = load_data_shard(self.files[0])

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class TokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, log_fn=None):
        self.stream = TokenStream(pattern, rank=rank, world_size=world_size, log_fn=log_fn)

    def next_batch(self, batch_tokens: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = torch.from_numpy(chunk[:-1].reshape(-1, seq_len)).to(device=device, dtype=torch.long, non_blocking=True)
        y = torch.from_numpy(chunk[1:].reshape(-1, seq_len)).to(device=device, dtype=torch.long, non_blocking=True)
        return x, y


class RMSNormNoWeight(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), bias=None)


def apply_partial_rope(x: torch.Tensor, rope_dims: int, base: float) -> torch.Tensor:
    if rope_dims <= 0:
        return x
    b, h, t, d = x.shape
    rope_dims = min(rope_dims, d)
    rope_dims = rope_dims - (rope_dims % 2)
    if rope_dims == 0:
        return x
    x_rope = x[..., :rope_dims]
    x_pass = x[..., rope_dims:]
    half = rope_dims // 2
    freqs = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (freqs / max(half, 1)))
    pos = torch.arange(t, device=x.device, dtype=torch.float32)
    theta = torch.outer(pos, inv_freq)
    cos = torch.cos(theta)[None, None, :, :].to(dtype=x.dtype)
    sin = torch.sin(theta)[None, None, :, :].to(dtype=x.dtype)
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:rope_dims]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return torch.cat([rot, x_pass], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, rope_dims: int, qk_gain_init: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, self.kv_dim)
        self.c_v = CastedLinear(dim, self.kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * qk_gain_init)
        self.scale = self.head_dim ** -0.5
        self.rope_base = rope_base
        self.rope_dims = rope_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = rms_norm(q)
        k = rms_norm(k)
        q = apply_partial_rope(q, self.rope_dims, self.rope_base)
        k = apply_partial_rope(k, self.rope_dims, self.rope_base)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True, scale=self.scale)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden_dim = int(round((dim * mlp_mult) / 64.0) * 64)
        self.fc = CastedLinear(dim, hidden_dim)
        self.proj = CastedLinear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float, rope_base: float, rope_dims: int, qk_gain_init: float, layer_idx: int, use_ln_scale: bool = True):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, rope_dims, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1) if use_ln_scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resid_scale.to(dtype=x.dtype)[None, None, :] * x
        attn_in = self.attn_norm(x) * self.ln_scale
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(attn_in)
        mlp_in = self.mlp_norm(x) * self.ln_scale
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_in)
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.blocks = nn.ModuleList([
            Block(
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                rope_base=args.rope_base,
                rope_dims=args.rope_dims,
                qk_gain_init=args.qk_gain_init,
                layer_idx=i,
            )
            for i in range(args.num_layers)
        ])
        self.final_norm = RMSNormNoWeight()
        self.out_proj = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size)
        with torch.no_grad():
            self.tok_emb.weight.normal_(mean=0.0, std=args.tied_embed_init_std)
            for block in self.blocks:
                block.attn.proj.weight.zero_()
                block.mlp.proj.weight.zero_()

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        c = self.args.logit_softcap
        return c * torch.tanh(logits / c)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = rms_norm(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        x = self(input_ids).reshape(-1, self.args.model_dim)
        y = target_ids.reshape(-1)
        if self.args.tie_embeddings:
            logits = self.softcap(x @ self.tok_emb.weight.to(dtype=x.dtype).t())
        else:
            logits = self.softcap(self.out_proj(x))
        return F.cross_entropy(logits.float(), y, reduction="mean")


class Muon:
    def __init__(self, named_params: dict[str, nn.Parameter], keys: list[str], args: Hyperparameters):
        self.args = args
        self.keys = keys
        self.buffers = {k: torch.zeros_like(named_params[k], dtype=torch.float32) for k in keys}

    @torch.no_grad()
    def step(self, named_params: dict[str, nn.Parameter], named_grads: dict[str, torch.Tensor], step: int, lr_mul: float):
        if self.args.muon_momentum_warmup_steps > 0:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        for k in self.keys:
            p = named_params[k]
            g = named_grads[k].float().add_(p.float(), alpha=self.args.weight_decay)
            buf = self.buffers[k].mul_(momentum).add_(g)
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            p.add_(g_ortho.to(dtype=p.dtype), alpha=-lr * scale)


class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        named_params = dict(model.named_parameters())
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in named_params.items()
            if p.ndim == 2 and (k.startswith("blocks.") or k == self.embed_key or k == "out_proj.weight") and not any(tag in k for tag in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in named_params.items()
            if p.ndim < 2 or any(tag in k for tag in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if self.embed_key in self.matrix_keys:
            self.matrix_keys.remove(self.embed_key)
        self.muon = Muon(named_params, self.matrix_keys, args)
        embed_params = [named_params[self.embed_key]] if self.embed_key in named_params else []
        scalar_params = [named_params[k] for k in self.scalar_keys if k in named_params]
        self.adam_embed = torch.optim.AdamW(embed_params, lr=args.tied_embed_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay)
        self.adam_scalar = torch.optim.AdamW(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.weight_decay)

    @torch.no_grad()
    def step(self, model: GPT, step: int, lr_mul: float):
        named_params = dict(model.named_parameters())
        named_grads = {k: p.grad for k, p in named_params.items() if p.grad is not None}
        self.muon.step(named_params, named_grads, step=step, lr_mul=lr_mul)
        for group in self.adam_embed.param_groups:
            group["lr"] = self.args.tied_embed_lr * lr_mul
        for group in self.adam_scalar.param_groups:
            group["lr"] = self.args.scalar_lr * lr_mul
        self.adam_embed.step()
        self.adam_scalar.step()
        self.adam_embed.zero_grad(set_to_none=True)
        self.adam_scalar.zero_grad(set_to_none=True)
        for k in self.matrix_keys:
            named_params[k].grad = None


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].lerp_(v.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


@torch.no_grad()
def quantize_float_array(arr: np.ndarray):
    f32 = np.asarray(arr, dtype=np.float32)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
        return q, scale.astype(np.float16), True
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8)
    return q, scale, False


@torch.no_grad()
def quantize_state_dict_int8(model: nn.Module):
    quantized = {}
    scales = {}
    dtypes = {}
    passthrough = {}
    qmeta = {}
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        if not np.issubdtype(arr.dtype, np.floating):
            passthrough[name] = np.ascontiguousarray(arr)
            continue
        if arr.size <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = np.ascontiguousarray(arr.astype(np.float16 if arr.dtype in (np.float32,) else arr.dtype, copy=False))
            continue
        q, s, per_row = quantize_float_array(arr)
        quantized[name] = np.ascontiguousarray(q)
        scales[name] = np.ascontiguousarray(s)
        dtypes[name] = str(tensor.dtype).replace("torch.", "")
        if per_row:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
    return {
        "__quant_format__": "int8_only_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }


@torch.no_grad()
def dequantize_state_dict_int8(obj):
    out = {}
    for name, q in obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(obj["scales"][name], dtype=np.float32)
        if obj.get("qmeta", {}).get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            arr = q_np.astype(np.float32) * float(scale)
        out[name] = torch.from_numpy(arr)
    for name, arr in obj["passthrough"].items():
        out[name] = torch.from_numpy(np.array(arr, copy=True))
    return out


def build_sentencepiece_luts(sp, vocab_size):
    sp_vs = int(sp.vocab_size())
    ts = max(sp_vs, vocab_size)
    base_bytes = np.zeros(ts, dtype=np.int16)
    has_space = np.zeros(ts, dtype=np.bool_)
    is_boundary = np.ones(ts, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_space, is_boundary


def validate_dataset_tokenizer_pair(data_path, tokenizer_path):
    dataset_dir = Path(data_path).resolve()
    actual = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if entry is None:
        return dataset_dir.name, actual, None
    tn = entry.get("tokenizer_name")
    te = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tn), None) if tn else None
    expected_name = Path((te or {}).get("model_path") or (te or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError("Tokenizer mismatch")
    exp = (entry.get("stats") or {}).get("files_train")
    if exp is not None:
        exp = int(exp)
        if actual > exp:
            raise ValueError("Too many train shards")
    return dataset_dir.name, actual, exp


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files]))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


@torch.no_grad()
def eval_val(args: Hyperparameters, model: GPT, val_tokens: np.ndarray, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, device: torch.device, log_fn=None) -> tuple[float, float]:
    model.eval()
    val_batch_tokens = args.val_batch_size
    val_batch_seqs = max(val_batch_tokens // args.train_seq_len, 1)
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    full_total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_batches = full_total_batches if args.max_val_batches <= 0 else min(full_total_batches, args.max_val_batches)
    total_loss_sum, total_tokens, total_bytes = 0.0, 0.0, 0.0
    for batch_idx, bss in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        if args.max_val_batches > 0 and batch_idx > args.max_val_batches:
            break
        bse = min(bss + val_batch_seqs, total_seqs)
        rs, re = bss * args.train_seq_len, bse * args.train_seq_len + 1
        chunk = val_tokens[rs:re]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.long, non_blocking=True)
        y = torch.from_numpy(y_np).to(device=device, dtype=torch.long, non_blocking=True)
        bl = model.loss(x, y)
        ct = float(y.numel())
        total_loss_sum += float(bl.item()) * ct
        prev, tgt = x_np.reshape(-1), y_np.reshape(-1)
        bn = base_bytes_lut[tgt].astype(np.int16, copy=True)
        bn += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.int16)
        total_tokens += ct
        total_bytes += float(bn.astype(np.float64).sum())
        if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    vl = total_loss_sum / total_tokens
    vb = (vl / math.log(2.0)) * (total_tokens / total_bytes)
    model.train()
    return vl, vb


def main():
    args = Hyperparameters()
    rank, world_size, local_rank = setup_distributed()
    seed_all(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    out_dir = Path(args.out_dir)
    if is_master():
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()
    logfile = out_dir / f"{args.run_id}.txt"

    def log(msg, console=True):
        if is_master() and console:
            print(msg)
        if is_master():
            with logfile.open("a", encoding="utf-8") as f:
                print(msg, file=f)

    log(str(logfile))
    log(f"Python {sys.version}", console=False)
    log(f"Torch {torch.__version__}", console=False)
    log(f"CUDA {torch.version.cuda}", console=False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError("VOCAB_SIZE mismatch")
    dataset_name, actual_train_files, _ = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    model = GPT(args).to(device)
    model.train()
    opt = SplitOptimizers(model, args)
    ema = EMA(model, args.ema_decay) if args.use_ema else None

    n_params = sum(p.numel() for p in model.parameters())
    log(f"model_params:{n_params}")
    log(f"dataset:{dataset_name} shards:{actual_train_files} val_tokens:{val_tokens.size - 1}")
    log(f"iterations:{args.iterations} batch_tokens:{args.train_batch_tokens} accum:{args.grad_accum_steps}")
    log(f"val_batch_size:{args.val_batch_size} max_val_batches:{args.max_val_batches}")
    log(f"arch:num_layers:{args.num_layers} model_dim:{args.model_dim} mlp_mult:{args.mlp_mult} heads:{args.num_heads} kv_heads:{args.num_kv_heads}")
    log(f"wd:{args.weight_decay} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} embed_lr:{args.tied_embed_lr} use_ema:{int(args.use_ema)}")

    train_loader = TokenLoader(args.train_files, rank=rank, world_size=world_size, log_fn=log if is_master() else None)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if is_master() and (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            eval_model = model
            restore_state = None
            if ema is not None:
                restore_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            vl, vb = eval_val(args, eval_model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, device, log_fn=log)
            log(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{train_time_ms:.0f}ms")
            if restore_state is not None:
                model.load_state_dict(restore_state, strict=True)
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        st0 = time.perf_counter()
        total_loss = 0.0
        for micro_idx in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(args.microbatch_tokens, args.train_seq_len, device)
            loss = model.loss(x, y) / args.grad_accum_steps
            loss.backward()
            total_loss += float(loss.item())
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        opt.step(model, step=step, lr_mul=lr_mul)
        if ema is not None:
            ema.update(model)
        torch.cuda.synchronize(device)

        sms = 1000.0 * (time.perf_counter() - st0)
        atm = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if is_master() and args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{total_loss:.4f} train_time:{atm:.0f}ms step_avg:{atm / step:.2f}ms tok_s:{args.train_batch_tokens / (sms / 1000.0):.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and atm >= max_wallclock_ms:
            stop_after_step = step

    if is_master():
        export_model = GPT(args).to(device)
        if ema is not None:
            ema.copy_to(export_model)
        else:
            export_model.load_state_dict(model.state_dict(), strict=True)
        export_model.eval()
        state_path = out_dir / f"{args.run_id}_torch_model.pt"
        torch.save(export_model.state_dict(), state_path)
        log(f"saved:{state_path} bytes:{state_path.stat().st_size}")

        quant_obj = quantize_state_dict_int8(export_model)
        blob = zlib.compress(pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL), level=9)
        qp = out_dir / f"{args.run_id}_torch_model.int8.ptz"
        with qp.open("wb") as f:
            f.write(blob)
        code_bytes = Path(__file__).read_bytes()
        log(f"int8_zlib:{qp.stat().st_size} bytes | code:{len(code_bytes)} | total:{qp.stat().st_size + len(code_bytes)}")

        roundtrip = GPT(args).cpu()
        rt_state = dequantize_state_dict_int8(pickle.loads(zlib.decompress(qp.read_bytes())))
        missing, unexpected = roundtrip.load_state_dict(rt_state, strict=False)
        if missing or unexpected:
            log(f"roundtrip_state_warnings missing:{missing} unexpected:{unexpected}")
        roundtrip = roundtrip.to(device)
        qvl, qvb = eval_val(args, roundtrip, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, device, log_fn=log)
        log(f"final_int8_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f}")
        log(f"final_int8_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")

    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
