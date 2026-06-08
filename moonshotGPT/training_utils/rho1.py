"""Shared rho-1 helpers for token/sequence-level loss selection during training.

This module is intentionally standalone so trainers can adopt it later with
minimal churn. It keeps rho-specific concerns together:

- configuration validation
- ref-loss shard discovery and preflight alignment checks
- lazy memmap loading of ref-loss batches
- rho-1 masking and per-step metric aggregation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os

import numpy as np
import torch

RHO1_MODES = ("delta", "ref_only", "student_only")
RHO1_GRANULARITIES = ("token", "sequence")


@dataclass(frozen=True, slots=True)
class Rho1Config:
    ref_loss_dir: str = ""
    keep_frac: float = 1.0
    warmup_steps: int = 0
    mode: str = "delta"
    ref_loss_cap: float = 0.0
    granularity: str = "token"

    @property
    def enabled(self) -> bool:
        return bool(self.ref_loss_dir)

    def validate(self) -> None:
        if not self.enabled:
            return
        if not os.path.isdir(self.ref_loss_dir):
            raise FileNotFoundError(f"--rho_ref_loss_dir not found: {self.ref_loss_dir}")
        if not (0.0 < float(self.keep_frac) <= 1.0):
            raise ValueError(f"--rho_keep_frac must be in (0,1], got {self.keep_frac}")
        if self.mode not in RHO1_MODES:
            raise ValueError(
                f"--rho_mode must be one of {list(RHO1_MODES)}, got {self.mode}"
            )
        if self.granularity not in RHO1_GRANULARITIES:
            raise ValueError(
                f"--rho_granularity must be one of {list(RHO1_GRANULARITIES)}, got {self.granularity}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"--rho_warmup_steps must be >=0, got {self.warmup_steps}")
        if self.ref_loss_cap < 0:
            raise ValueError(f"--rho_ref_loss_cap must be >=0, got {self.ref_loss_cap}")

    def run_name_suffix(self) -> str:
        if not self.enabled:
            return ""
        mode_tag = "studentfocus" if self.mode == "student_only" else self.mode
        if self.granularity == "sequence":
            mode_tag = f"{mode_tag}seq"
        keep_frac_tag = int(round(float(self.keep_frac) * 1000.0))
        return f"_rho{mode_tag}_k{keep_frac_tag:04d}_wu{int(self.warmup_steps)}"


@dataclass(slots=True)
class Rho1BatchResult:
    loss: torch.Tensor
    kept_tokens: int | None
    keep_frac: float | None
    kept_sequences: int | None
    keep_seq_frac: float | None
    ref_loss_mean: float | None


@dataclass(frozen=True, slots=True)
class Rho1OptStepSummary:
    keep_frac_mean: float | None
    kept_tokens_mean: int | None
    keep_seq_frac_mean: float | None
    kept_sequences_mean: int | None
    ref_loss_mean: float | None


@dataclass(frozen=True, slots=True)
class Rho1GapOptStepSummary:
    delta_mean_all: float | None
    delta_median_all: float | None
    delta_p90_all: float | None
    delta_pos_frac_all: float | None
    delta_pos_mean_all: float | None
    delta_mean_kept: float | None
    delta_pos_frac_kept: float | None
    delta_mean_seq_all: float | None
    delta_median_seq_all: float | None
    delta_p90_seq_all: float | None
    delta_pos_frac_seq_all: float | None
    delta_pos_mean_seq_all: float | None
    delta_mean_seq_kept: float | None
    delta_pos_frac_seq_kept: float | None


@dataclass(slots=True)
class Rho1OptStepAccumulator:
    keep_frac_sum: float = 0.0
    keep_frac_count: int = 0
    kept_tokens_sum: int = 0
    keep_seq_frac_sum: float = 0.0
    keep_seq_frac_count: int = 0
    kept_sequences_sum: int = 0
    ref_loss_sum: float = 0.0
    ref_loss_count: int = 0

    def update(self, batch_result: Rho1BatchResult) -> None:
        if batch_result.keep_frac is not None:
            self.keep_frac_sum += float(batch_result.keep_frac)
            self.keep_frac_count += 1
        if batch_result.kept_tokens is not None:
            self.kept_tokens_sum += int(batch_result.kept_tokens)
        if batch_result.keep_seq_frac is not None:
            self.keep_seq_frac_sum += float(batch_result.keep_seq_frac)
            self.keep_seq_frac_count += 1
        if batch_result.kept_sequences is not None:
            self.kept_sequences_sum += int(batch_result.kept_sequences)
        if batch_result.ref_loss_mean is not None:
            self.ref_loss_sum += float(batch_result.ref_loss_mean)
            self.ref_loss_count += 1

    def finalize(self) -> Rho1OptStepSummary:
        keep_frac_mean = None
        kept_tokens_mean = None
        keep_seq_frac_mean = None
        kept_sequences_mean = None
        ref_loss_mean = None

        if self.keep_frac_count > 0:
            keep_frac_mean = self.keep_frac_sum / self.keep_frac_count
            kept_tokens_mean = int(round(self.kept_tokens_sum / self.keep_frac_count))
        if self.keep_seq_frac_count > 0:
            keep_seq_frac_mean = self.keep_seq_frac_sum / self.keep_seq_frac_count
            kept_sequences_mean = int(round(self.kept_sequences_sum / self.keep_seq_frac_count))
        if self.ref_loss_count > 0:
            ref_loss_mean = self.ref_loss_sum / self.ref_loss_count

        return Rho1OptStepSummary(
            keep_frac_mean=keep_frac_mean,
            kept_tokens_mean=kept_tokens_mean,
            keep_seq_frac_mean=keep_seq_frac_mean,
            kept_sequences_mean=kept_sequences_mean,
            ref_loss_mean=ref_loss_mean,
        )

    def reset(self) -> None:
        self.keep_frac_sum = 0.0
        self.keep_frac_count = 0
        self.kept_tokens_sum = 0
        self.keep_seq_frac_sum = 0.0
        self.keep_seq_frac_count = 0
        self.kept_sequences_sum = 0
        self.ref_loss_sum = 0.0
        self.ref_loss_count = 0


@dataclass(slots=True)
class Rho1GapOptStepAccumulator:
    delta_all_chunks: list[np.ndarray] = field(default_factory=list)
    delta_all_sum: float = 0.0
    delta_all_count: int = 0
    delta_all_pos_sum: float = 0.0
    delta_all_pos_count: int = 0
    delta_kept_sum: float = 0.0
    delta_kept_count: int = 0
    delta_kept_pos_count: int = 0
    delta_seq_all_chunks: list[np.ndarray] = field(default_factory=list)
    delta_seq_all_sum: float = 0.0
    delta_seq_all_count: int = 0
    delta_seq_all_pos_sum: float = 0.0
    delta_seq_all_pos_count: int = 0
    delta_seq_kept_sum: float = 0.0
    delta_seq_kept_count: int = 0
    delta_seq_kept_pos_count: int = 0

    def update(
        self,
        *,
        token_loss: torch.Tensor,
        ref_loss: torch.Tensor,
        ref_valid_mask: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> None:
        if (
            token_loss.shape != ref_loss.shape
            or token_loss.shape != ref_valid_mask.shape
            or token_loss.shape != keep_mask.shape
        ):
            raise ValueError("rho gap diagnostics expect all tensors to share the same shape.")

        delta = token_loss.detach().float() - ref_loss.detach().float()
        valid_delta = delta[ref_valid_mask]
        if int(valid_delta.numel()) > 0:
            valid_np = valid_delta.detach().cpu().numpy().astype(np.float32, copy=False)
            self.delta_all_chunks.append(valid_np)
            self.delta_all_sum += float(valid_np.sum(dtype=np.float64))
            self.delta_all_count += int(valid_np.size)
            valid_pos = valid_np[valid_np > 0.0]
            self.delta_all_pos_count += int(valid_pos.size)
            if int(valid_pos.size) > 0:
                self.delta_all_pos_sum += float(valid_pos.sum(dtype=np.float64))

        kept_valid_mask = keep_mask & ref_valid_mask
        kept_delta = delta[kept_valid_mask]
        if int(kept_delta.numel()) > 0:
            kept_np = kept_delta.detach().cpu().numpy().astype(np.float32, copy=False)
            self.delta_kept_sum += float(kept_np.sum(dtype=np.float64))
            self.delta_kept_count += int(kept_np.size)
            self.delta_kept_pos_count += int((kept_np > 0.0).sum())

        sequence_count = int(token_loss.shape[0]) if token_loss.ndim > 0 else 1
        flat_shape = (sequence_count, -1)
        flat_delta = delta.reshape(flat_shape)
        flat_valid_mask = ref_valid_mask.reshape(flat_shape)
        flat_kept_valid_mask = kept_valid_mask.reshape(flat_shape)

        seq_valid_counts = flat_valid_mask.sum(dim=1)
        seq_valid_mask = seq_valid_counts > 0
        if bool(seq_valid_mask.any()):
            seq_delta_all = (
                (flat_delta * flat_valid_mask.to(flat_delta.dtype)).sum(dim=1)[seq_valid_mask]
                / seq_valid_counts[seq_valid_mask].to(flat_delta.dtype)
            )
            seq_all_np = seq_delta_all.detach().cpu().numpy().astype(np.float32, copy=False)
            self.delta_seq_all_chunks.append(seq_all_np)
            self.delta_seq_all_sum += float(seq_all_np.sum(dtype=np.float64))
            self.delta_seq_all_count += int(seq_all_np.size)
            seq_all_pos = seq_all_np[seq_all_np > 0.0]
            self.delta_seq_all_pos_count += int(seq_all_pos.size)
            if int(seq_all_pos.size) > 0:
                self.delta_seq_all_pos_sum += float(seq_all_pos.sum(dtype=np.float64))

        seq_kept_counts = flat_kept_valid_mask.sum(dim=1)
        seq_kept_mask = seq_kept_counts > 0
        if bool(seq_kept_mask.any()):
            seq_delta_kept = (
                (flat_delta * flat_kept_valid_mask.to(flat_delta.dtype)).sum(dim=1)[seq_kept_mask]
                / seq_kept_counts[seq_kept_mask].to(flat_delta.dtype)
            )
            seq_kept_np = seq_delta_kept.detach().cpu().numpy().astype(np.float32, copy=False)
            self.delta_seq_kept_sum += float(seq_kept_np.sum(dtype=np.float64))
            self.delta_seq_kept_count += int(seq_kept_np.size)
            self.delta_seq_kept_pos_count += int((seq_kept_np > 0.0).sum())

    def finalize(self) -> Rho1GapOptStepSummary:
        delta_mean_all = None
        delta_median_all = None
        delta_p90_all = None
        delta_pos_frac_all = None
        delta_pos_mean_all = None
        delta_mean_kept = None
        delta_pos_frac_kept = None
        delta_mean_seq_all = None
        delta_median_seq_all = None
        delta_p90_seq_all = None
        delta_pos_frac_seq_all = None
        delta_pos_mean_seq_all = None
        delta_mean_seq_kept = None
        delta_pos_frac_seq_kept = None

        if self.delta_all_count > 0:
            delta_mean_all = self.delta_all_sum / self.delta_all_count
            delta_pos_frac_all = self.delta_all_pos_count / self.delta_all_count
            if self.delta_all_pos_count > 0:
                delta_pos_mean_all = self.delta_all_pos_sum / self.delta_all_pos_count
            flat = np.concatenate(self.delta_all_chunks, axis=0)
            delta_median_all = float(np.quantile(flat, 0.5))
            delta_p90_all = float(np.quantile(flat, 0.9))

        if self.delta_kept_count > 0:
            delta_mean_kept = self.delta_kept_sum / self.delta_kept_count
            delta_pos_frac_kept = self.delta_kept_pos_count / self.delta_kept_count

        if self.delta_seq_all_count > 0:
            delta_mean_seq_all = self.delta_seq_all_sum / self.delta_seq_all_count
            delta_pos_frac_seq_all = self.delta_seq_all_pos_count / self.delta_seq_all_count
            if self.delta_seq_all_pos_count > 0:
                delta_pos_mean_seq_all = self.delta_seq_all_pos_sum / self.delta_seq_all_pos_count
            flat_seq = np.concatenate(self.delta_seq_all_chunks, axis=0)
            delta_median_seq_all = float(np.quantile(flat_seq, 0.5))
            delta_p90_seq_all = float(np.quantile(flat_seq, 0.9))

        if self.delta_seq_kept_count > 0:
            delta_mean_seq_kept = self.delta_seq_kept_sum / self.delta_seq_kept_count
            delta_pos_frac_seq_kept = self.delta_seq_kept_pos_count / self.delta_seq_kept_count

        return Rho1GapOptStepSummary(
            delta_mean_all=delta_mean_all,
            delta_median_all=delta_median_all,
            delta_p90_all=delta_p90_all,
            delta_pos_frac_all=delta_pos_frac_all,
            delta_pos_mean_all=delta_pos_mean_all,
            delta_mean_kept=delta_mean_kept,
            delta_pos_frac_kept=delta_pos_frac_kept,
            delta_mean_seq_all=delta_mean_seq_all,
            delta_median_seq_all=delta_median_seq_all,
            delta_p90_seq_all=delta_p90_seq_all,
            delta_pos_frac_seq_all=delta_pos_frac_seq_all,
            delta_pos_mean_seq_all=delta_pos_mean_seq_all,
            delta_mean_seq_kept=delta_mean_seq_kept,
            delta_pos_frac_seq_kept=delta_pos_frac_seq_kept,
        )

    def reset(self) -> None:
        self.delta_all_chunks = []
        self.delta_all_sum = 0.0
        self.delta_all_count = 0
        self.delta_all_pos_sum = 0.0
        self.delta_all_pos_count = 0
        self.delta_kept_sum = 0.0
        self.delta_kept_count = 0
        self.delta_kept_pos_count = 0
        self.delta_seq_all_chunks = []
        self.delta_seq_all_sum = 0.0
        self.delta_seq_all_count = 0
        self.delta_seq_all_pos_sum = 0.0
        self.delta_seq_all_pos_count = 0
        self.delta_seq_kept_sum = 0.0
        self.delta_seq_kept_count = 0
        self.delta_seq_kept_pos_count = 0


def _list_split_shards(data_dir: str, split: str, max_shards: int | None = None) -> list[str]:
    prefix = f"{split}_"
    shards = [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.startswith(prefix) and name.endswith(".bin")
    ]
    if not shards:
        raise FileNotFoundError(
            f"No shards found for split='{split}' in '{data_dir}' "
            f"(expected files like {split}_*.bin)."
        )
    if max_shards is not None:
        if max_shards <= 0:
            raise ValueError(f"max_shards must be > 0 when provided, got {max_shards}")
        shards = shards[:max_shards]
    return shards


def _count_uint16_tokens(path: str) -> int:
    nbytes = os.path.getsize(path)
    if nbytes % 2 != 0:
        raise ValueError(f"uint16 shard has odd byte size (not divisible by 2): {path}")
    return nbytes // 2


def _count_ref_loss_tokens(path: str, ref_dtype) -> int:
    itemsize = int(np.dtype(ref_dtype).itemsize)
    nbytes = os.path.getsize(path)
    if nbytes % itemsize != 0:
        raise ValueError(
            f"Ref-loss file byte size not divisible by dtype itemsize={itemsize}: {path}"
        )
    return nbytes // itemsize


def _to_scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got shape={tuple(value.shape)}")
        return int(value.item())
    return int(value)


def _score_tensor_for_mode(
    *,
    token_loss: torch.Tensor,
    ref_loss: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "delta":
        return token_loss.detach().float() - ref_loss
    if mode == "ref_only":
        return -ref_loss
    return token_loss.detach().float()


def resolve_ref_loss_file(ref_loss_dir: str, shard_path: str):
    base = os.path.basename(str(shard_path))
    if base.endswith(".bin"):
        base = base[:-4]
    for suffix, dtype in (("f16", np.float16), ("f32", np.float32)):
        candidate = os.path.join(ref_loss_dir, f"{base}.ref_loss.{suffix}.bin")
        if os.path.exists(candidate):
            return candidate, dtype
    raise FileNotFoundError(
        f"No ref-loss file found for shard '{shard_path}' in '{ref_loss_dir}'. "
        "Expected one of: *.ref_loss.f16.bin or *.ref_loss.f32.bin"
    )


def ref_loss_meta_path(ref_loss_path: str) -> str:
    if not ref_loss_path.endswith(".bin"):
        raise ValueError(f"Unexpected ref-loss filename (expected .bin): {ref_loss_path}")
    return ref_loss_path[:-4] + ".meta.json"


def validate_rho_ref_loss_alignment(
    data_dir: str,
    ref_loss_dir: str,
    seq_len: int,
    micro_batch_size: int,
    split: str = "train",
    max_shards: int | None = None,
    max_reported_errors: int = 12,
) -> int:
    """Fail fast if rho ref-loss files do not match the intended training setup."""

    shards = _list_split_shards(data_dir, split, max_shards=max_shards)
    expected_seq_len = int(seq_len)
    expected_bs = int(micro_batch_size)
    expected_stride = expected_seq_len * expected_bs
    expected_block = expected_stride + 1

    errors: list[str] = []

    def _expect_meta_int(meta: dict, key: str, expected: int, shard_name: str) -> None:
        if key not in meta:
            errors.append(f"{shard_name}: meta missing key '{key}' (expected {expected}).")
            return
        try:
            got = int(meta.get(key))
        except Exception:
            errors.append(
                f"{shard_name}: meta key '{key}' is non-integer ({meta.get(key)!r}); expected {expected}."
            )
            return
        if got != int(expected):
            errors.append(f"{shard_name}: meta key '{key}'={got} but expected {expected}.")

    for shard_path in shards:
        shard_name = os.path.basename(shard_path)

        try:
            shard_tokens = _count_uint16_tokens(shard_path)
        except Exception as exc:
            errors.append(f"{shard_name}: cannot read shard token count: {exc}")
            continue

        try:
            ref_path, ref_dtype = resolve_ref_loss_file(ref_loss_dir, shard_path)
        except Exception as exc:
            errors.append(f"{shard_name}: {exc}")
            continue

        try:
            ref_tokens = _count_ref_loss_tokens(ref_path, ref_dtype)
        except Exception as exc:
            errors.append(f"{shard_name}: cannot read ref-loss token count: {exc}")
            continue
        if ref_tokens != shard_tokens:
            errors.append(
                f"{shard_name}: ref token count mismatch: shard={shard_tokens}, ref={ref_tokens} ({ref_path})."
            )

        meta_path = ref_loss_meta_path(ref_path)
        if not os.path.exists(meta_path):
            errors.append(f"{shard_name}: missing ref-loss metadata file: {meta_path}")
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception as exc:
            errors.append(f"{shard_name}: failed to parse meta JSON {meta_path}: {exc}")
            continue

        _expect_meta_int(meta, "seq_len", expected_seq_len, shard_name)
        _expect_meta_int(meta, "batch_size", expected_bs, shard_name)
        _expect_meta_int(meta, "stride_tokens", expected_stride, shard_name)
        _expect_meta_int(meta, "block_tokens", expected_block, shard_name)

        if "source_num_tokens" in meta:
            try:
                source_tokens = int(meta.get("source_num_tokens"))
                if source_tokens != shard_tokens:
                    errors.append(
                        f"{shard_name}: meta source_num_tokens={source_tokens} but shard has {shard_tokens}."
                    )
            except Exception:
                errors.append(
                    f"{shard_name}: meta source_num_tokens is non-integer ({meta.get('source_num_tokens')!r})."
                )

        source_base = meta.get("source_shard_basename")
        if source_base is not None and str(source_base) != shard_name:
            errors.append(
                f"{shard_name}: meta source_shard_basename={source_base!r} does not match shard name."
            )

    if errors:
        shown = errors[:max_reported_errors]
        lines = "\n".join(f"  - {msg}" for msg in shown)
        hidden = max(0, len(errors) - len(shown))
        if hidden > 0:
            lines += f"\n  - ... and {hidden} more validation errors."
        raise ValueError(
            "Rho ref-loss preflight validation failed before training.\n"
            f"Checked split='{split}' in data_dir='{data_dir}' against ref_loss_dir='{ref_loss_dir}'.\n"
            f"Expected seq_len={expected_seq_len}, micro_batch_size={expected_bs}.\n"
            "Common fix: recompute ref-loss shards with matching --seq_len and --batch_size.\n"
            f"Details:\n{lines}"
        )

    return len(shards)


class Rho1RefLossLoader:
    """Lazy memmap-backed loader for rho ref-loss shards."""

    def __init__(self, ref_loss_dir: str) -> None:
        self.ref_loss_dir = str(ref_loss_dir)
        self._cache: dict[str, dict[str, object]] = {}

    def load_batch(self, shard_meta: dict, expected_bt: int) -> tuple[np.ndarray, np.ndarray]:
        shard_path = shard_meta.get("shard_path", None)
        if shard_path is None:
            raise KeyError("Batch meta missing shard_path; cannot map to ref-loss file.")

        start = _to_scalar_int(shard_meta.get("start"))
        end = _to_scalar_int(shard_meta.get("end"))
        if (end - start) != (expected_bt + 1):
            raise ValueError(
                f"Unexpected block span from meta: end-start={end-start}, expected {expected_bt+1}. "
                "Reference-loss alignment requires matching seq_len and micro_batch_size."
            )

        cache_key = str(shard_path)
        if cache_key not in self._cache:
            ref_path, ref_dtype = resolve_ref_loss_file(self.ref_loss_dir, cache_key)
            self._cache[cache_key] = {
                "path": ref_path,
                "dtype": ref_dtype,
                "mm": np.memmap(ref_path, dtype=ref_dtype, mode="r"),
            }

        mm = self._cache[cache_key]["mm"]
        lo = start + 1
        hi = end
        arr = np.asarray(mm[lo:hi], dtype=np.float32)
        if arr.size != expected_bt:
            raise ValueError(
                f"Ref-loss slice size mismatch for shard '{shard_path}': got {arr.size}, expected {expected_bt}."
            )

        valid = np.isfinite(arr)
        if not valid.any():
            raise ValueError(
                f"Ref-loss slice has no finite values for shard '{shard_path}' [{lo}:{hi}]. "
                "Check precompute alignment (same seq_len and micro_batch_size)."
            )

        arr = np.where(valid, arr, np.float32(1e9)).astype(np.float32, copy=False)
        return arr, valid


def compute_rho1_loss_from_reference(
    token_loss: torch.Tensor,
    ref_loss: torch.Tensor,
    ref_valid_mask: torch.Tensor,
    config: Rho1Config,
    gap_accumulator: Rho1GapOptStepAccumulator | None = None,
) -> Rho1BatchResult:
    """Apply rho-1 selection to a token-loss tensor using aligned ref losses."""

    if token_loss.shape != ref_loss.shape or token_loss.shape != ref_valid_mask.shape:
        raise ValueError(
            "rho-1 expects token_loss, ref_loss, and ref_valid_mask to have identical shapes."
        )

    score = _score_tensor_for_mode(
        token_loss=token_loss,
        ref_loss=ref_loss,
        mode=config.mode,
    )

    if config.granularity == "sequence":
        sequence_count = int(token_loss.shape[0]) if token_loss.ndim > 0 else 1
        flat_shape = (sequence_count, -1)
        flat_token_loss = token_loss.reshape(flat_shape)
        flat_ref_loss = ref_loss.reshape(flat_shape)
        flat_ref_valid_mask = ref_valid_mask.reshape(flat_shape)
        flat_score = score.reshape(flat_shape)

        candidate_token_mask = flat_ref_valid_mask
        if config.ref_loss_cap > 0:
            candidate_token_mask = candidate_token_mask & (flat_ref_loss <= float(config.ref_loss_cap))

        candidate_seq_mask = candidate_token_mask.any(dim=1)
        candidate_seq_count = int(candidate_seq_mask.sum().item())
        if candidate_seq_count <= 0:
            candidate_token_mask = flat_ref_valid_mask
            candidate_seq_mask = candidate_token_mask.any(dim=1)
            candidate_seq_count = int(candidate_seq_mask.sum().item())

        if candidate_seq_count <= 0:
            keep_seq_mask = torch.ones(sequence_count, dtype=torch.bool, device=token_loss.device)
        elif config.keep_frac >= 1.0:
            keep_seq_mask = candidate_seq_mask
        else:
            candidate_token_counts = candidate_token_mask.sum(dim=1).clamp_min(1)
            seq_scores = torch.full(
                (sequence_count,),
                float("-inf"),
                dtype=flat_score.dtype,
                device=flat_score.device,
            )
            seq_scores[candidate_seq_mask] = (
                (flat_score * candidate_token_mask.to(flat_score.dtype)).sum(dim=1)[candidate_seq_mask]
                / candidate_token_counts[candidate_seq_mask].to(flat_score.dtype)
            )
            k = max(1, int(math.ceil(float(config.keep_frac) * candidate_seq_count)))
            candidate_scores = seq_scores[candidate_seq_mask]
            if k >= int(candidate_scores.numel()):
                keep_seq_mask = candidate_seq_mask
            else:
                threshold = torch.topk(candidate_scores, k, sorted=False).values.min()
                keep_seq_mask = candidate_seq_mask & (seq_scores >= threshold)

        expand_shape = (sequence_count,) + (1,) * (token_loss.ndim - 1)
        keep_mask = keep_seq_mask.reshape(expand_shape).expand_as(token_loss)
        keep_mask_f = keep_mask.to(token_loss.dtype)
        loss = (token_loss * keep_mask_f).sum() / keep_mask_f.sum().clamp_min(1.0)
        kept_tokens = int(keep_mask_f.sum().detach().item())
        keep_frac = float(kept_tokens / max(1, token_loss.numel()))
        ref_loss_mean = float(ref_loss[ref_valid_mask].mean().detach().item())

        if gap_accumulator is not None:
            gap_accumulator.update(
                token_loss=token_loss,
                ref_loss=ref_loss,
                ref_valid_mask=ref_valid_mask,
                keep_mask=keep_mask,
            )

        return Rho1BatchResult(
            loss=loss,
            kept_tokens=kept_tokens,
            keep_frac=keep_frac,
            kept_sequences=int(keep_seq_mask.sum().detach().item()),
            keep_seq_frac=float(int(keep_seq_mask.sum().detach().item()) / max(1, sequence_count)),
            ref_loss_mean=ref_loss_mean,
        )

    candidate_mask = ref_valid_mask
    if config.ref_loss_cap > 0:
        candidate_mask = candidate_mask & (ref_loss <= float(config.ref_loss_cap))

    candidate_count = int(candidate_mask.sum().item())
    if candidate_count <= 0:
        candidate_mask = ref_valid_mask
        candidate_count = int(candidate_mask.sum().item())

    if candidate_count <= 0:
        keep_mask = torch.ones_like(token_loss, dtype=torch.bool)
    elif config.keep_frac >= 1.0:
        keep_mask = candidate_mask
    else:
        k = max(1, int(math.ceil(float(config.keep_frac) * candidate_count)))
        candidate_scores = score[candidate_mask]
        if k >= int(candidate_scores.numel()):
            keep_mask = candidate_mask
        else:
            threshold = torch.topk(candidate_scores, k, sorted=False).values.min()
            keep_mask = candidate_mask & (score >= threshold)

    keep_mask_f = keep_mask.to(token_loss.dtype)
    loss = (token_loss * keep_mask_f).sum() / keep_mask_f.sum().clamp_min(1.0)
    kept_tokens = int(keep_mask_f.sum().detach().item())
    keep_frac = float(kept_tokens / max(1, token_loss.numel()))
    ref_loss_mean = float(ref_loss[ref_valid_mask].mean().detach().item())

    if gap_accumulator is not None:
        gap_accumulator.update(
            token_loss=token_loss,
            ref_loss=ref_loss,
            ref_valid_mask=ref_valid_mask,
            keep_mask=keep_mask,
        )

    return Rho1BatchResult(
        loss=loss,
        kept_tokens=kept_tokens,
        keep_frac=keep_frac,
        kept_sequences=None,
        keep_seq_frac=None,
        ref_loss_mean=ref_loss_mean,
    )


def compute_rho1_batch_result(
    token_loss: torch.Tensor,
    shard_meta: dict | None,
    opt_step: int,
    config: Rho1Config,
    ref_loss_loader: Rho1RefLossLoader | None,
    gap_accumulator: Rho1GapOptStepAccumulator | None = None,
) -> Rho1BatchResult:
    """Return the training loss and rho-1 metrics for one micro-batch."""

    baseline_loss = token_loss.mean()

    if not config.enabled:
        return Rho1BatchResult(
            loss=baseline_loss,
            kept_tokens=None,
            keep_frac=None,
            kept_sequences=None,
            keep_seq_frac=None,
            ref_loss_mean=None,
        )

    should_select_tokens = opt_step >= config.warmup_steps
    needs_ref_loss = should_select_tokens or (gap_accumulator is not None)

    if not needs_ref_loss:
        return Rho1BatchResult(
            loss=baseline_loss,
            kept_tokens=int(token_loss.numel()),
            keep_frac=1.0,
            kept_sequences=(
                int(token_loss.shape[0]) if config.granularity == "sequence" and token_loss.ndim > 0 else None
            ),
            keep_seq_frac=(1.0 if config.granularity == "sequence" else None),
            ref_loss_mean=None,
        )

    if not isinstance(shard_meta, dict):
        raise RuntimeError(
            "Rho masking requires dict metadata per batch. "
            "Run with a shard_loader that returns meta."
        )
    if ref_loss_loader is None:
        raise ValueError("rho-1 is enabled, but no ref-loss loader was provided.")

    ref_arr, ref_valid_np = ref_loss_loader.load_batch(
        shard_meta=shard_meta,
        expected_bt=int(token_loss.numel()),
    )
    ref_loss = torch.from_numpy(ref_arr).to(device=token_loss.device, non_blocking=True)
    ref_loss = ref_loss.reshape_as(token_loss)
    ref_valid_mask = torch.from_numpy(ref_valid_np).to(
        device=token_loss.device,
        non_blocking=True,
    )
    ref_valid_mask = ref_valid_mask.reshape_as(token_loss)

    if not should_select_tokens:
        keep_mask = torch.ones_like(token_loss, dtype=torch.bool)
        if gap_accumulator is not None:
            gap_accumulator.update(
                token_loss=token_loss,
                ref_loss=ref_loss,
                ref_valid_mask=ref_valid_mask,
                keep_mask=keep_mask,
            )
        return Rho1BatchResult(
            loss=baseline_loss,
            kept_tokens=int(token_loss.numel()),
            keep_frac=1.0,
            kept_sequences=(
                int(token_loss.shape[0]) if config.granularity == "sequence" and token_loss.ndim > 0 else None
            ),
            keep_seq_frac=(1.0 if config.granularity == "sequence" else None),
            ref_loss_mean=float(ref_loss[ref_valid_mask].mean().detach().item()),
        )

    return compute_rho1_loss_from_reference(
        token_loss=token_loss,
        ref_loss=ref_loss,
        ref_valid_mask=ref_valid_mask,
        config=config,
        gap_accumulator=gap_accumulator,
    )
