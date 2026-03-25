from __future__ import annotations

import glob
import io
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16


class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    max_val_batches: int = int(os.environ.get("MAX_VAL_BATCHES", 8))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    cluster_codes: int = int(os.environ.get("CLUSTER_CODES", 32))
    cluster_iters: int = int(os.environ.get("CLUSTER_ITERS", 12))
    cluster_min_numel: int = int(os.environ.get("CLUSTER_MIN_NUMEL", 131072))
    cluster_scope: str = os.environ.get("CLUSTER_SCOPE", "matrices")

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,skip_weights",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
MX_DTYPE_FROM_NAME = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_Q = 99.99984 / 100.0


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunks.append(min(remaining, usable_chunk))
        remaining -= chunks[-1]
    return chunks


def accumulate_flat_grads(accum, grads_tree, scale):
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


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
    def __init__(self, pattern, log_fn=None, dataset_name=""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch, self.file_idx, self.pos = 1, 0, 0
        self.log_fn, self.dataset_name = log_fn, dataset_name
        self.tokens = load_data_shard(self.files[0])

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
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
    def __init__(self, pattern, log_fn=None, dataset_name=""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens, seq_len):
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


class CastedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x):
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.proj(y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, dim * mlp_mult)
        self.proj = CastedLinear(dim * mlp_mult, dim)

    def __call__(self, x):
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))))

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads, mlp_mult,
                 tie_embeddings, tied_embed_init_std, logit_softcap, rope_base, qk_gain_init):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, model_dim), dtype=mx.float32)
        self.blocks = [
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

    def softcap(self, logits):
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids):
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def loss(self, input_ids, target_ids):
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


class Muon:
    def __init__(self, keys, params, args):
        self.keys, self.args = keys, args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params, grads, step, lr_mul):
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model, args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights"
            or (k.startswith("blocks.") and (p.ndim < 2 or any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)))
        ]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(learning_rate=args.tied_embed_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps, bias_correction=True)
        self.adam_scalar = optim.Adam(learning_rate=args.scalar_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps, bias_correction=True)

    def step(self, model, grads_tree, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]}, {self.embed_key: params[self.embed_key]}
        ))
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        sg = {k: grads[k] for k in self.scalar_keys if k in grads}
        sp = {k: params[k] for k in self.scalar_keys if k in params}
        if sg:
            updated.update(self.adam_scalar.apply_gradients(sg, sp))
        model.update(tree_unflatten(list(updated.items())))


def _np_float32(arr):
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name, arr, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr):
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q), scale


def kmeans_1d(values: np.ndarray, k: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    values = values.astype(np.float32, copy=False).reshape(-1)
    if values.size == 0:
        return np.zeros((0,), dtype=np.int16), np.zeros((0,), dtype=np.float32)
    unique_vals = np.unique(values)
    if unique_vals.size <= k:
        codebook = unique_vals.astype(np.float32, copy=False)
        centers = codebook
        idx = np.searchsorted(codebook, values).astype(np.int16)
        return idx, centers
    qs = np.linspace(0.0, 1.0, num=k, dtype=np.float32)
    centers = np.quantile(values, qs).astype(np.float32)
    for _ in range(iters):
        dists = np.abs(values[:, None] - centers[None, :])
        idx = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = idx == j
            if np.any(mask):
                new_centers[j] = values[mask].mean(dtype=np.float32)
        if np.allclose(new_centers, centers, rtol=0.0, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    dists = np.abs(values[:, None] - centers[None, :])
    idx = np.argmin(dists, axis=1).astype(np.int16)
    return idx, centers.astype(np.float32)


def should_cluster_tensor(name: str, arr, args: Hyperparameters) -> bool:
    if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
        return False
    if not mx.issubdtype(arr.dtype, mx.floating):
        return False
    if int(arr.size) < args.cluster_min_numel:
        return False
    if args.cluster_scope == "matrices" and arr.ndim != 2:
        return False
    if name == "tok_emb.weight":
        return False
    return True


def quantize_state_dict_clustered(flat_state, args: Hyperparameters, log_fn=None):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    clustered_codes, clustered_codebooks, clustered_shapes = {}, {}, {}
    stats = dict.fromkeys((
        "param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
        "baseline_tensor_bytes", "int8_payload_bytes", "clustered_tensors"
    ), 0)
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if should_cluster_tensor(name, arr, args):
            f32 = _np_float32(arr)
            codes, codebook = kmeans_1d(f32, args.cluster_codes, args.cluster_iters)
            clustered_codes[name] = np.ascontiguousarray(codes)
            clustered_codebooks[name] = np.ascontiguousarray(codebook.astype(np.float16))
            clustered_shapes[name] = tuple(int(x) for x in f32.shape)
            dtypes[name] = str(arr.dtype).split(".")[-1]
            stats["clustered_tensors"] += 1
            stats["int8_payload_bytes"] += int(clustered_codes[name].nbytes + clustered_codebooks[name].nbytes)
            if log_fn:
                log_fn(f"clustered:{name} shape:{clustered_shapes[name]} codes:{args.cluster_codes}", console=False)
            continue
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name], scales[name] = q, s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj = {
        "__quant_format__": "clustered_or_int8_v1",
        "clustered_codes": clustered_codes,
        "clustered_codebooks": clustered_codebooks,
        "clustered_shapes": clustered_shapes,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_clustered(quant_obj):
    out, qmeta = {}, quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, codes in quant_obj.get("clustered_codes", {}).items():
        codebook = np.asarray(quant_obj["clustered_codebooks"][name], dtype=np.float32)
        shape = tuple(int(x) for x in quant_obj["clustered_shapes"][name])
        idx = np.asarray(codes, dtype=np.int16).reshape(-1)
        arr = codebook[idx].reshape(shape)
        out[name] = mx.array(arr, dtype=MX_DTYPE_FROM_NAME[quant_obj["dtypes"][name]])
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[quant_obj["dtypes"][name]])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig = passthrough_orig_dtypes.get(name)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig]) if isinstance(orig, str) else mx.array(out_arr)
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


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files]))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum = None
    for ct in chunk_sizes:
        x, y = train_loader.next_batch(ct, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=None):
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = val_batch_tokens // args.train_seq_len
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
        x_np, y_np = chunk[:-1].reshape(-1, args.train_seq_len), chunk[1:].reshape(-1, args.train_seq_len)
        bl = compiled_loss(mx.array(x_np, dtype=mx.int32), mx.array(y_np, dtype=mx.int32)).astype(mx.float32)
        mx.eval(bl)
        ct = float(y_np.size)
        total_loss_sum += float(bl.item()) * ct
        prev, tgt = x_np.reshape(-1), y_np.reshape(-1)
        bn = base_bytes_lut[tgt].astype(np.int16, copy=True)
        bn += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.int16)
        total_tokens += ct
        total_bytes += float(bn.astype(np.float64).sum())
        if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    vl = total_loss_sum / total_tokens
    return vl, (vl / math.log(2.0)) * (total_tokens / total_bytes)


def clip_grad_tree(grads_tree, max_norm):
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    tsq = sum(float(np.sum(np.square(_np_float32(g)), dtype=np.float64)) for g in flat.values())
    if tsq <= 0 or math.sqrt(tsq) <= max_norm:
        return grads_tree
    s = max_norm / (math.sqrt(tsq) + 1e-12)
    return tree_unflatten([(k, g * s) for k, g in flat.items()])


def main():
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg, console=True):
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Python {sys.version}", console=False)
    log(f"MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError("VOCAB_SIZE mismatch")
    dataset_name, actual_train_files, _ = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    )
    opt = SplitOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"model_params:{n_params}")
    log(f"dataset:{dataset_name} shards:{actual_train_files} val_tokens:{val_tokens.size-1}")
    log(f"iterations:{args.iterations} batch_tokens:{args.train_batch_tokens} accum:{args.grad_accum_steps}")
    log(f"val_batch_size:{args.val_batch_size} max_val_batches:{args.max_val_batches}")
    log(f"cluster_export:codes:{args.cluster_codes} iters:{args.cluster_iters} min_numel:{args.cluster_min_numel} scope:{args.cluster_scope}")

    if args.warmup_steps > 0:
        for ws in range(args.warmup_steps):
            accum, wl = None, mx.array(0.0, dtype=mx.float32)
            gs = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                wl, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, gs)
            mx.eval(wl, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log(f"warmup:{ws+1}/{args.warmup_steps}")
        vbt = args.val_batch_size // args.grad_accum_steps
        wvs = min(vbt // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        wc = val_tokens[:wvs * args.train_seq_len + 1]
        mx.eval(compiled_loss(mx.array(wc[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32),
                              mx.array(wc[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)))
        mx.synchronize()
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            log(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{train_time_ms:.0f}ms")
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lm = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        st0 = time.perf_counter()
        accum, tl = None, mx.array(0.0, dtype=mx.float32)
        gs = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, gs)
            tl = tl + loss.astype(mx.float32) * gs
            if args.mlx_eager_eval:
                mx.eval(tl, accum)
        grads = clip_grad_tree(tree_unflatten(list(accum.items())), args.grad_clip_norm)
        tlv = float(tl.item())
        opt.step(model, grads, step=step, lr_mul=lm)
        mx.synchronize()

        sms = 1000.0 * (time.perf_counter() - st0)
        atm = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(f"step:{step}/{args.iterations} train_loss:{tlv:.4f} train_time:{atm:.0f}ms step_avg:{atm/step:.2f}ms tok_s:{args.train_batch_tokens/(sms/1000):.0f}")
        if max_wallclock_ms is not None and stop_after_step is None and atm >= max_wallclock_ms:
            stop_after_step = step

    flat_state = {k: v for k, v in tree_flatten(model.state)}
    op = out_dir / f"{args.run_id}_mlx_model.npz"
    mx.savez(str(op), **flat_state)
    log(f"saved:{op} bytes:{op.stat().st_size}")

    qo, qs = quantize_state_dict_clustered(flat_state, args, log_fn=log)
    qb = zlib.compress(pickle.dumps(qo, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    qp = out_dir / f"{args.run_id}_mlx_model.clustered.ptz"
    with qp.open("wb") as f:
        f.write(qb)
    cb = len(code.encode("utf-8"))
    log(f"clustered_zlib:{qp.stat().st_size} bytes | code:{cb} | total:{qp.stat().st_size+cb} | clustered_tensors:{qs['clustered_tensors']}")

    model.update(tree_unflatten(list(dequantize_state_dict_clustered(pickle.loads(zlib.decompress(qp.read_bytes()))).items())))
    qt0 = time.perf_counter()
    qvl, qvb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
    log(f"final_clustered_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000*(time.perf_counter()-qt0):.0f}ms")
    log(f"final_clustered_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")


if __name__ == "__main__":
    main()
