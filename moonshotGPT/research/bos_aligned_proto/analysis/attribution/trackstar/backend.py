"""TrackStar/Bergson integration for faithful training-example EWoK attribution.

This module isolates the optional Bergson dependency behind the same
checkpoint-local `CheckpointScores` contract that the existing TRAK backend
uses. The surrounding attribution pipeline remains responsible for checkpoint
selection, candidate selection, EWoK target creation, export, and comparison.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from transformers.pytorch_utils import Conv1D as HFConv1D

from .bergson_datasets import (
    BergsonCandidateDataset,
    CandidateIndexMetadata,
    build_candidate_index_metadata,
)
from .bergson_queries import collect_query_module_grads, score_bundle_diagnostics, subset_target_bundle
from ..common.candidates import CandidateSelection
from ..common.checkpoints import CheckpointRef, load_checkpoint_state_dict
from ..common.ewok_targets import CheckpointScores, EWOKTargetBundle, TargetDiagnostics
from ..common.training_examples import ExampleManifest
from .config import TrackstarConfig

if TYPE_CHECKING:
    from ..run_trak import RunExecutionContext


@dataclass(frozen=True)
class _BergsonRuntime:
    """Resolved Bergson entry points behind one stable local interface.

    Bergson's import surface has moved a bit across versions, so we resolve the
    pieces we need once up front and store them here instead of scattering
    optional imports throughout the backend.
    """

    collect_gradients: Any
    load_gradients: Any
    Scorer: Any | None
    GradientProcessor: Any | None
    IndexConfig: Any | None
    PreprocessConfig: Any | None
    AdamNormalizer: Any | None


@dataclass(frozen=True)
class BergsonShardResult:
    """One rank-local slice of the final target-by-candidate score matrix."""

    rank: int
    target_indices: tuple[int, ...]
    score_matrix: np.ndarray
    diagnostics: tuple[TargetDiagnostics, ...]


def _missing_bergson_message(detail: str | None = None) -> str:
    """Construct a consistent optional-dependency error message."""

    suffix = "" if not detail else f" Original import error: {detail}"
    return (
        "The `bergson` package is required to run attribution with the TrackStar backend. "
        "Install EleutherAI Bergson from https://github.com/EleutherAI/bergson/ and rerun."
        + suffix
    )


def _resolve_optional_attr(*candidates: str) -> Any | None:
    """Return the first importable dotted attribute from a list of candidates."""

    for dotted_name in candidates:
        module_name, _, attr_name = dotted_name.rpartition(".")
        if not module_name:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        value = getattr(module, attr_name, None)
        if value is not None:
            return value
    return None


def _load_bergson_runtime() -> _BergsonRuntime:
    """Load the Bergson functions/classes TrackStar needs at runtime.

    We use Bergson programmatically rather than via its CLI. This helper keeps
    that dependency isolated and tolerant to small API moves such as
    `bergson.Scorer` vs `bergson.scoring.Scorer`.
    """

    try:
        importlib.import_module("bergson")
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(_missing_bergson_message(str(exc))) from exc

    collect_gradients = _resolve_optional_attr(
        "bergson.collect_gradients",
        "bergson.trackstar.collect_gradients",
    )
    load_gradients = _resolve_optional_attr(
        "bergson.load_gradients",
        "bergson.trackstar.load_gradients",
    )
    scorer_ctor = _resolve_optional_attr(
        "bergson.Scorer",
        "bergson.scoring.Scorer",
    )
    processor_ctor = _resolve_optional_attr(
        "bergson.GradientProcessor",
        "bergson.processor.GradientProcessor",
    )
    index_cfg_ctor = _resolve_optional_attr(
        "bergson.IndexConfig",
        "bergson.config.IndexConfig",
    )
    preprocess_cfg_ctor = _resolve_optional_attr(
        "bergson.PreprocessConfig",
        "bergson.config.PreprocessConfig",
    )
    adam_normalizer_ctor = _resolve_optional_attr(
        "bergson.AdamNormalizer",
        "bergson.gradients.AdamNormalizer",
    )
    if collect_gradients is None or load_gradients is None:
        raise ImportError(
            _missing_bergson_message(
                "Could not resolve bergson.collect_gradients/load_gradients from the installed package."
            )
        )
    return _BergsonRuntime(
        collect_gradients=collect_gradients,
        load_gradients=load_gradients,
        Scorer=scorer_ctor,
        GradientProcessor=processor_ctor,
        IndexConfig=index_cfg_ctor,
        PreprocessConfig=preprocess_cfg_ctor,
        AdamNormalizer=adam_normalizer_ctor,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload and ensure its parent directory exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one small JSON sidecar file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _target_bundle_fingerprint(bundle: EWOKTargetBundle) -> str:
    """Hash the query bundle properties that affect shard assembly reuse."""

    payload = {
        "target_ids": list(bundle.target_ids),
        "score_view": bundle.score_view,
        "score_reduction": bundle.score_reduction,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _target_indices_for_rank(num_targets: int, rank: int, world_size: int) -> tuple[int, ...]:
    """Split target rows into contiguous rank-local ranges.

    The score matrix rows correspond to target order, so this helper shards by
    row index while preserving global ordering for later reassembly.
    """

    if world_size <= 1:
        return tuple(range(num_targets))
    start = (num_targets * rank) // world_size
    end = (num_targets * (rank + 1)) // world_size
    return tuple(range(start, end))


def _resolve_max_positions(model: torch.nn.Module) -> int | None:
    """Best-effort lookup for the model's supported sequence length."""

    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(config, attr, None)
        if value is None:
            continue
        resolved = int(value)
        if resolved > 0:
            return resolved
    return None


@contextmanager
def _periodic_heartbeat(
    *,
    enabled: bool,
    interval_seconds: float,
    emit: Callable[[float], None],
):
    """Emit periodic elapsed-time messages while an opaque step is running."""

    if not enabled:
        yield
        return

    stop_event = threading.Event()
    start_time = time.monotonic()

    def _worker() -> None:
        while not stop_event.wait(interval_seconds):
            emit(time.monotonic() - start_time)

    thread = threading.Thread(target=_worker, name="trackstar-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=max(0.1, interval_seconds))


@contextmanager
def _patched_candidate_forward_for_external_shift(model: torch.nn.Module):
    """Temporarily let Bergson score external-shift examples at max context.

    The main trainer feeds GPT-2-style checkpoints externally shifted training
    pairs of length `seq_len`, which means candidate windows represent
    `seq_len + 1` raw tokens. Bergson's CE collector instead expects unshifted
    labels and applies the causal shift internally, so the candidate adapter
    intentionally hands Bergson the full `seq_len + 1` chunk.

    A plain Hugging Face GPT-2 forward pass cannot accept that 1-token-longer
    chunk when `n_positions == seq_len`; it would try to index the positional
    embedding table one step past the end and trigger a CUDA device-side
    assert. During candidate indexing only, this shim trims the final token
    before the actual model forward, then appends one dummy logits row so
    Bergson's own `logits[:, :-1]` shift still exposes the intended `seq_len`
    next-token predictions.
    """

    max_positions = _resolve_max_positions(model)
    if max_positions is None:
        yield
        return

    original_forward = model.forward

    def _truncate_kwargs(kwargs: dict[str, Any], expected_len: int) -> dict[str, Any]:
        truncated = dict(kwargs)
        for name in ("attention_mask", "position_ids", "token_type_ids"):
            value = truncated.get(name)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and int(value.shape[1]) == expected_len:
                truncated[name] = value[:, :-1]
        return truncated

    def patched_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        arg_style = "kwargs"
        if input_ids is None and args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                input_ids = first
                arg_style = "args"

        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
            return original_forward(*args, **kwargs)

        seq_len = int(input_ids.shape[1])
        if seq_len <= max_positions:
            return original_forward(*args, **kwargs)
        if seq_len != max_positions + 1:
            raise ValueError(
                "TrackStar candidate indexing only supports sequences up to one token longer than the "
                f"checkpoint context window. Got sequence length {seq_len} for max_positions={max_positions}."
            )

        truncated_ids = input_ids[:, :-1]
        if arg_style == "kwargs":
            call_args = args
            call_kwargs = dict(kwargs)
            call_kwargs["input_ids"] = truncated_ids
            call_kwargs = _truncate_kwargs(call_kwargs, seq_len)
        else:
            call_args = (truncated_ids, *args[1:])
            call_kwargs = _truncate_kwargs(kwargs, seq_len)

        outputs = original_forward(*call_args, **call_kwargs)
        logits = getattr(outputs, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError(
                "Patched TrackStar candidate forward expected model(...).logits to be a rank-3 tensor, "
                f"got {type(logits)!r} with shape {getattr(logits, 'shape', None)}"
            )

        dummy_row = torch.zeros_like(logits[:, :1, :])
        padded_logits = torch.cat([logits, dummy_row], dim=1)
        if hasattr(outputs, "logits"):
            outputs.logits = padded_logits
            return outputs
        return SimpleNamespace(logits=padded_logits)

    model.forward = patched_forward
    try:
        yield
    finally:
        model.forward = original_forward


def _damped_psd_power(
    H: torch.Tensor,
    power: float,
    *,
    damping_factor: float = 0.1,
    dtype: torch.dtype = torch.float64,
    regularizer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a damped power of a p.s.d. matrix.

    This mirrors Bergson's TrackStar preprocessing step closely enough for the
    local wrapper: add adaptive damping, eigendecompose, then apply the
    requested power such as `-0.5` for the split preconditioner `H^(-1/2)`.
    """

    original_dtype = H.dtype
    H = H.to(dtype=dtype)
    if regularizer is not None:
        regularizer = regularizer.to(dtype=dtype, device=H.device)
        H = H + damping_factor * regularizer
    else:
        damping_val = damping_factor * H.abs().mean()
        H = H + damping_val * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    eigval, eigvec = torch.linalg.eigh(H)
    return (eigvec * eigval.pow(power) @ eigvec.mH).to(original_dtype)


def _serialize_shard(path: Path, shard: BergsonShardResult) -> None:
    """Persist one rank-local shard so the root rank can assemble results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rank": int(shard.rank),
            "target_indices": list(shard.target_indices),
            "score_matrix": torch.from_numpy(np.asarray(shard.score_matrix, dtype=np.float64)),
            "diagnostics": [diag.to_json() for diag in shard.diagnostics],
        },
        path,
    )


def _deserialize_shard(path: Path) -> BergsonShardResult:
    """Load one previously-written shard file from disk."""

    payload = torch.load(path, map_location="cpu")
    diagnostics = tuple(TargetDiagnostics(**row) for row in payload["diagnostics"])
    return BergsonShardResult(
        rank=int(payload["rank"]),
        target_indices=tuple(int(idx) for idx in payload["target_indices"]),
        score_matrix=np.asarray(payload["score_matrix"], dtype=np.float64),
        diagnostics=diagnostics,
    )


def assemble_sharded_scores(
    *,
    num_targets: int,
    num_candidates: int,
    shard_results: Sequence[BergsonShardResult],
) -> tuple[np.ndarray, tuple[TargetDiagnostics, ...]]:
    """Reconstruct the full score matrix from rank-local shard files.

    The distributed path keeps the main export contract unchanged: root still
    returns one dense `[num_targets, num_candidates]` matrix plus target-level
    diagnostics. This helper validates each shard before placing its rows into
    the global matrix so partial or misordered writes fail loudly.
    """

    score_matrix = np.zeros((num_targets, num_candidates), dtype=np.float64)
    diagnostics_by_index: list[TargetDiagnostics | None] = [None] * num_targets

    for shard in sorted(shard_results, key=lambda item: (item.target_indices[:1], item.rank)):
        if shard.score_matrix.shape[0] != len(shard.target_indices):
            raise ValueError(
                "Shard score matrix row count does not match target_indices: "
                f"{shard.score_matrix.shape[0]} vs {len(shard.target_indices)}"
            )
        if shard.score_matrix.shape[1] != num_candidates:
            raise ValueError(
                "Shard score matrix candidate count mismatch: "
                f"{shard.score_matrix.shape[1]} vs {num_candidates}"
            )
        if len(shard.diagnostics) != len(shard.target_indices):
            raise ValueError(
                "Shard diagnostics length does not match target_indices: "
                f"{len(shard.diagnostics)} vs {len(shard.target_indices)}"
            )
        for row_offset, target_idx in enumerate(shard.target_indices):
            score_matrix[int(target_idx), :] = shard.score_matrix[row_offset, :]
            diagnostics_by_index[int(target_idx)] = shard.diagnostics[row_offset]

    missing = [idx for idx, diag in enumerate(diagnostics_by_index) if diag is None]
    if missing:
        raise ValueError(f"Missing sharded score rows for target indices: {missing}")
    return score_matrix, tuple(diag for diag in diagnostics_by_index if diag is not None)


class BergsonAttributionBackend:
    """Checkpoint-local Bergson scorer for training examples against EWoK targets."""

    _SUPPORTED_BERGSON_MODULES = (
        torch.nn.Linear,
        HFConv1D,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )

    def __init__(
        self,
        *,
        config: TrackstarConfig,
        model: torch.nn.Module,
        tokenizer,
        execution_context: "RunExecutionContext",
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.execution_context = execution_context
        self.runtime = _load_bergson_runtime()
        self._status("loaded Bergson runtime")

    def _status(self, message: str, *, root_only: bool = False) -> None:
        """Emit a timestamped progress line for long-running attribution work."""

        if root_only and not self.execution_context.is_root:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[trackstar][rank {self.execution_context.rank}/{self.execution_context.world_size}]"
        print(f"{timestamp} {prefix} {message}", flush=True)

    def _load_checkpoint_into_model(self, checkpoint: CheckpointRef) -> dict[str, torch.Tensor]:
        """Load one checkpoint into the shared model instance.

        We reload weights into an existing model rather than constructing a new
        model per checkpoint. The checkpoint loader already patches in missing
        tied-weight aliases, which is why strict loading now succeeds for the
        GPT-2 style checkpoints used in these experiments.
        """

        self._status(f"loading checkpoint step={checkpoint.step} from {checkpoint.path}")
        state_dict = load_checkpoint_state_dict(checkpoint.path, model=self.model)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.config.device)
        self.model.eval()
        return state_dict

    def _step_cache_dir(self, checkpoint: CheckpointRef) -> Path:
        """Return the cache directory for one checkpoint's TrackStar artifacts."""

        path = Path(self.config.cache_dir) / f"step{checkpoint.step:08d}" / "trackstar"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _optimizer_state_path(self, checkpoint: CheckpointRef) -> Path:
        """Return the optional optimizer-state path next to one checkpoint."""

        return Path(checkpoint.path) / "optimizer.pt"

    def _candidate_uses_adam_second_moment_correction(
        self,
        checkpoint: CheckpointRef,
    ) -> bool:
        """Return `True` when we can Adam-correct candidate gradients for this checkpoint."""

        return (
            self.runtime.GradientProcessor is not None
            and self.runtime.AdamNormalizer is not None
            and self._optimizer_state_path(checkpoint).exists()
        )

    def _candidate_correction_status(
        self,
        checkpoint: CheckpointRef,
    ) -> tuple[bool, str]:
        """Describe whether candidate-side Adam correction will be used."""

        if self.runtime.GradientProcessor is None:
            return (
                False,
                "Bergson GradientProcessor is unavailable; using raw candidate CE gradients",
            )
        if self.runtime.AdamNormalizer is None:
            return (
                False,
                "Bergson AdamNormalizer is unavailable; using raw candidate CE gradients",
            )
        optimizer_state_path = self._optimizer_state_path(checkpoint)
        if not optimizer_state_path.exists():
            return (
                False,
                f"optimizer state {optimizer_state_path} is missing; using raw candidate CE gradients",
            )
        return (
            True,
            f"using Adam second-moment correction from {optimizer_state_path}",
        )

    def _base_model(self) -> torch.nn.Module:
        """Return the model subtree Bergson indexes against."""

        return getattr(self.model, "base_model", self.model)

    def _candidate_gradient_modules(self) -> dict[str, torch.nn.Module]:
        """Return the base-model modules Bergson will collect candidate gradients for."""

        modules: dict[str, torch.nn.Module] = {}
        for name, module in self._base_model().named_modules():
            if not name:
                continue
            if not isinstance(module, self._SUPPORTED_BERGSON_MODULES):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                continue
            modules[str(name)] = module
        if not modules:
            raise ValueError("Model does not expose any Bergson-supported 2D weight modules")
        return modules

    def _ordered_optimizer_param_names(self) -> tuple[str, ...]:
        """Rebuild the training-time AdamW parameter ordering from the live model."""

        named_params = [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]
        decay_names = [name for name, param in named_params if param.dim() >= 2]
        nodecay_names = [name for name, param in named_params if param.dim() < 2]
        return tuple(decay_names + nodecay_names)

    def _load_candidate_adam_normalizers(
        self,
        checkpoint: CheckpointRef,
    ) -> dict[str, Any]:
        """Load Bergson Adam normalizers for candidate-side CE gradients.

        The saved optimizer state does not preserve parameter names, only the
        per-group parameter ids and their moving statistics. We reconstruct the
        original AdamW ordering from the current model using the same
        `decay + nodecay` grouping as the training code, then map those states
        back onto Bergson's base-model module names.
        """

        optimizer_state_path = self._optimizer_state_path(checkpoint)
        optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
        state = optimizer_state.get("state")
        param_groups = optimizer_state.get("param_groups")
        if not isinstance(state, dict) or not isinstance(param_groups, list):
            raise ValueError(
                f"Unexpected optimizer state structure in {optimizer_state_path}; expected dict[state, param_groups]"
            )

        ordered_names = self._ordered_optimizer_param_names()
        ordered_state_ids: list[int] = []
        state_lr_by_id: dict[int, float] = {}
        for group in param_groups:
            params = group.get("params")
            if not isinstance(params, list):
                raise ValueError(f"Optimizer param group in {optimizer_state_path} is missing a list-valued `params`")
            lr = float(group.get("lr", 1.0))
            for param_id in params:
                normalized_id = int(param_id)
                ordered_state_ids.append(normalized_id)
                state_lr_by_id[normalized_id] = lr

        if len(ordered_state_ids) != len(ordered_names):
            raise ValueError(
                "Optimizer parameter count does not match reconstructed model ordering: "
                f"{len(ordered_state_ids)} vs {len(ordered_names)}"
            )

        state_id_by_name = {
            name: state_id for state_id, name in zip(ordered_state_ids, ordered_names, strict=True)
        }
        param_name_by_id = {
            id(param): name for name, param in self.model.named_parameters() if param.requires_grad
        }

        normalizers: dict[str, Any] = {}
        for module_name, module in self._candidate_gradient_modules().items():
            weight = getattr(module, "weight")
            full_param_name = param_name_by_id.get(id(weight))
            if full_param_name is None:
                raise ValueError(f"Could not resolve full parameter name for candidate module {module_name!r}")
            state_id = state_id_by_name.get(full_param_name)
            if state_id is None:
                raise ValueError(
                    f"Could not locate optimizer state for candidate module {module_name!r} ({full_param_name})"
                )
            param_state = state.get(state_id)
            if not isinstance(param_state, dict):
                raise ValueError(f"Optimizer state entry {state_id} for {full_param_name} is malformed")
            exp_avg_sq = param_state.get("exp_avg_sq")
            if not isinstance(exp_avg_sq, torch.Tensor):
                raise ValueError(
                    f"Optimizer state for {full_param_name} does not contain `exp_avg_sq`; cannot Adam-correct"
                )

            lr = state_lr_by_id[state_id]
            if lr <= 0.0:
                raise ValueError(f"Optimizer group lr for {full_param_name} is non-positive: {lr}")

            weight_avg_sq = exp_avg_sq.detach().to(device=weight.device) / (lr * lr)
            if isinstance(module, HFConv1D):
                weight_avg_sq = weight_avg_sq.mT

            normalizers[module_name] = self.runtime.AdamNormalizer(weight_avg_sq=weight_avg_sq)

        return normalizers

    @staticmethod
    def _compute_trackstar_lambda(
        *,
        query_covariances: Mapping[str, torch.Tensor],
        index_covariances: Mapping[str, torch.Tensor],
        target_components: int,
    ) -> float:
        """Mirror Bergson's pooled-spectrum compute_lambda logic locally."""

        query_eigvals_list: list[torch.Tensor] = []
        index_eigvals_list: list[torch.Tensor] = []

        for name, query_cov in query_covariances.items():
            index_cov = index_covariances.get(name)
            if index_cov is None:
                continue

            query_eigvals = torch.linalg.eigvalsh(query_cov.to(dtype=torch.float64)).clamp(min=0)
            index_eigvals = torch.linalg.eigvalsh(index_cov.to(dtype=torch.float64)).clamp(min=0)
            query_eigvals_list.append(query_eigvals)
            index_eigvals_list.append(index_eigvals)

        if not query_eigvals_list:
            return 0.99

        all_query = torch.cat(query_eigvals_list)
        all_index = torch.cat(index_eigvals_list)
        total = int(all_query.numel())

        if target_components <= 0:
            return 1.0
        if target_components > total:
            target_components = total

        sorted_query = torch.sort(all_query, descending=True).values
        sorted_index = torch.sort(all_index, descending=True).values
        k = target_components - 1
        sigma_query = float(sorted_query[k].item())
        sigma_index = float(sorted_index[k].item())

        denom = sigma_query + sigma_index
        if denom == 0.0:
            return 0.99

        lam = sigma_index / denom
        return max(0.0, min(1.0, lam))

    def _resolve_hessian_lambda(
        self,
        *,
        query_covariances: Mapping[str, torch.Tensor],
        index_covariances: Mapping[str, torch.Tensor],
    ) -> tuple[float, float, float, str]:
        """Return TrackStar lambda, derived weights, and where lambda came from."""

        if self.config.hessian_lambda is not None:
            lam = float(self.config.hessian_lambda)
            if not 0.0 <= lam <= 1.0:
                raise ValueError(f"Hessian lambda must be in [0, 1], got {lam}")
            return lam, 1.0 - lam, lam, "fixed override"

        target_components = int(self.config.hessian_target_components)
        lam = self._compute_trackstar_lambda(
            query_covariances=query_covariances,
            index_covariances=index_covariances,
            target_components=target_components,
        )
        return lam, 1.0 - lam, lam, f"compute_lambda(k={target_components})"

    @staticmethod
    def _feature_gram_matrix(features: torch.Tensor) -> torch.Tensor:
        """Compute the projected-gradient autocorrelation for one module."""

        if features.ndim != 2:
            raise ValueError(f"Expected 2D projected gradients, got {tuple(features.shape)}")
        dim = int(features.shape[1])
        if features.shape[0] == 0:
            return torch.zeros((dim, dim), dtype=torch.float32)
        projected = features.to(dtype=torch.float32)
        return projected.mT @ projected

    def _global_query_gram_matrices(
        self,
        query_grads: dict[str, torch.Tensor],
        modules: Sequence[str],
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Aggregate query-side projected-gradient autocorrelations across ranks."""

        if not modules:
            return {}, 0

        first_module = str(modules[0])
        local_count = int(query_grads[first_module].shape[0])
        if self.execution_context.is_distributed and dist.is_initialized():
            reduce_device = next(self.model.parameters()).device
            total_count = torch.tensor([local_count], device=reduce_device, dtype=torch.float32)
            dist.all_reduce(total_count)
            gram_matrices: dict[str, torch.Tensor] = {}
            for name in modules:
                local_features = query_grads[name].to(device=reduce_device, dtype=torch.float32)
                gram = self._feature_gram_matrix(local_features)
                gram = gram.to(device=reduce_device, dtype=torch.float32)
                dist.all_reduce(gram)
                gram_matrices[str(name)] = gram.cpu()
            return gram_matrices, int(total_count.item())

        gram_matrices = {
            str(name): self._feature_gram_matrix(query_grads[name]).cpu()
            for name in modules
        }
        return gram_matrices, local_count

    def _build_mixed_hessian_preconditioners(
        self,
        *,
        index_grads: dict[str, np.ndarray],
        query_grads: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build split `H^(-1/2)` preconditioners from candidate and query features."""

        if not self.config.use_hessian_correction:
            self._status("Hessian correction disabled; scoring with optimizer-scaled cosine similarity")
            return {}

        modules = [name for name in query_grads if name in index_grads]
        if not modules:
            raise ValueError("No overlapping modules were found for mixed Hessian correction")

        query_grams, total_query_count = self._global_query_gram_matrices(query_grads, modules)
        if total_query_count <= 0:
            raise ValueError("Cannot build query-side Hessian correction with zero query gradients")

        index_covariances: dict[str, torch.Tensor] = {}
        query_covariances: dict[str, torch.Tensor] = {}
        for name in modules:
            index_features = torch.from_numpy(np.asarray(index_grads[name])).to(dtype=torch.float32)
            index_count = int(index_features.shape[0])
            if index_count <= 0:
                raise ValueError(f"Cannot build index-side Hessian correction with zero rows for module {name!r}")

            index_covariances[str(name)] = self._feature_gram_matrix(index_features) / float(index_count)
            query_covariances[str(name)] = query_grams[name].to(dtype=torch.float32) / float(total_query_count)

        lam, index_weight, query_weight, lambda_source = self._resolve_hessian_lambda(
            query_covariances=query_covariances,
            index_covariances=index_covariances,
        )
        self._status(
            "building mixed Hessian-style preconditioners with "
            f"lambda={lam:.3f} from {lambda_source} "
            f"(index/query mix {index_weight:.3f}/{query_weight:.3f})",
            root_only=not self.execution_context.is_distributed,
        )

        split_preconditioners: dict[str, torch.Tensor] = {}
        for name in modules:
            H_index = index_covariances[name]
            H_query = query_covariances[name]
            H_mixed = index_weight * H_index + query_weight * H_query
            split_preconditioners[str(name)] = _damped_psd_power(H_mixed, power=-0.5).to(
                dtype=torch.float32,
                device="cpu",
            )

        self._status(
            f"finished mixed Hessian-style preconditioners for {len(split_preconditioners)} module(s)",
            root_only=not self.execution_context.is_distributed,
        )
        return split_preconditioners

    def _expected_index_metadata(
        self,
        *,
        checkpoint: CheckpointRef,
        candidate_ids: Sequence[int],
    ) -> CandidateIndexMetadata:
        """Describe the exact candidate index we expect for this checkpoint."""

        projection_dim = int(self.config.proj_dim) if self.config.use_fast_jl else 0
        return build_candidate_index_metadata(
            checkpoint=checkpoint,
            candidate_ids=candidate_ids,
            projection_dim=projection_dim,
            use_fast_jl=self.config.use_fast_jl,
            adam_second_moment_correction=self._candidate_uses_adam_second_moment_correction(checkpoint),
        )

    def _index_is_reusable(self, metadata_path: Path, expected: CandidateIndexMetadata) -> bool:
        """Return `True` when a cached index exactly matches the requested one."""

        if not metadata_path.exists():
            return False
        cached = _read_json(metadata_path)
        gradient_dir = self._resolve_gradient_dir(metadata_path.parent)
        return (
            cached == expected.to_json()
            and gradient_dir is not None
            and self._gradient_dir_has_nonzero_sizes(gradient_dir)
        )

    @staticmethod
    def _partial_index_dir(index_dir: Path) -> Path:
        """Return Bergson's temporary partial-run directory for an index path."""

        return Path(str(index_dir) + ".part")

    @classmethod
    def _is_gradient_dir(cls, path: Path) -> bool:
        """Return `True` when a directory contains the files `load_gradients(...)` expects."""

        return (
            path.is_dir()
            and (path / "info.json").exists()
            and (path / "gradients.bin").exists()
        )

    @classmethod
    def _resolve_gradient_dir(cls, index_dir: Path) -> Path | None:
        """Locate the actual Bergson gradient artifact directory.

        When using Bergson's lower-level collector directly, artifacts may be
        left in `run_path.part/` rather than being promoted to `run_path/`.
        Supporting both locations lets us recover from earlier interrupted runs
        and from wrapper versions that predated explicit finalization.
        """

        if cls._is_gradient_dir(index_dir):
            return index_dir
        partial_dir = cls._partial_index_dir(index_dir)
        if cls._is_gradient_dir(partial_dir):
            return partial_dir
        return None

    @staticmethod
    def _gradient_dir_has_nonzero_sizes(gradient_dir: Path) -> bool:
        """Return `True` when Bergson recorded at least one non-empty module slot.

        Earlier wrapper versions constructed the processor with
        `projection_dim=0` when projection was disabled. Bergson interprets that
        as a literal `(0, 0)` projected shape, which yields a formally valid but
        useless index where every module has gradient width 0. Treat those
        artifacts as invalid so we rebuild them automatically.
        """

        info_path = gradient_dir / "info.json"
        if not info_path.exists():
            return False
        try:
            info = _read_json(info_path)
        except json.JSONDecodeError:
            return False
        grad_sizes = info.get("grad_sizes", {})
        if not isinstance(grad_sizes, dict):
            return False
        return any(int(size) > 0 for size in grad_sizes.values())

    def _promote_partial_index(self, index_dir: Path) -> None:
        """Move Bergson's partial-run artifacts into the stable cache directory."""

        partial_dir = self._partial_index_dir(index_dir)
        if not partial_dir.exists():
            return
        index_dir.mkdir(parents=True, exist_ok=True)
        for child in partial_dir.iterdir():
            destination = index_dir / child.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(child), str(destination))
        partial_dir.rmdir()

    def _build_or_reuse_candidate_index(
        self,
        *,
        checkpoint: CheckpointRef,
        candidate_dataset: BergsonCandidateDataset,
        candidate_ids: Sequence[int],
    ) -> tuple[Path, CandidateIndexMetadata]:
        """Build or reuse Bergson's candidate gradient index for one checkpoint.

        Bergson owns the gradient collection/indexing step, but the outer
        pipeline still owns candidate selection. That means we hand Bergson a
        checkpoint-local candidate dataset in the exact order chosen by our
        exposure/candidate logic, then cache the resulting index under a hash of
        that ordered candidate list.
        """

        expected = self._expected_index_metadata(
            checkpoint=checkpoint,
            candidate_ids=candidate_ids,
        )
        step_cache = self._step_cache_dir(checkpoint)
        index_dir = step_cache / f"index_{expected.fingerprint}"
        metadata_path = index_dir / "candidate_index_meta.json"
        if self._index_is_reusable(metadata_path, expected):
            self._status(
                f"reusing candidate index for step={checkpoint.step} at {index_dir}",
                root_only=not self.execution_context.is_distributed,
            )
            if self.execution_context.is_distributed and dist.is_initialized():
                dist.barrier()
            return index_dir, expected

        self._status(
            f"building candidate index for step={checkpoint.step} with {len(candidate_ids)} candidate example(s)",
            root_only=not self.execution_context.is_distributed,
        )
        use_adam_correction, correction_status = self._candidate_correction_status(checkpoint)
        self._status(
            correction_status,
            root_only=not self.execution_context.is_distributed,
        )
        candidate_normalizers: dict[str, Any] = {}
        if use_adam_correction:
            candidate_normalizers = self._load_candidate_adam_normalizers(checkpoint)
            self._status(
                f"loaded Adam second-moment statistics for {len(candidate_normalizers)} candidate module(s)",
                root_only=not self.execution_context.is_distributed,
            )
        partial_dir = self._partial_index_dir(index_dir)
        if self.execution_context.is_root:
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
            # Remove stale cache contents for this fingerprint before rebuilding.
            # The index directory is generated output under the experiment cache.
            if index_dir.exists():
                shutil.rmtree(index_dir)

        projection_dim = int(self.config.proj_dim) if self.config.use_fast_jl else 0
        processor = None
        if self.runtime.GradientProcessor is not None:  # pragma: no branch - optional API
            processor_projection_dim = int(self.config.proj_dim) if self.config.use_fast_jl else None
            try:
                processor_kwargs: dict[str, Any] = {"projection_dim": processor_projection_dim}
                if candidate_normalizers:
                    processor_kwargs["normalizers"] = candidate_normalizers
                processor = self.runtime.GradientProcessor(**processor_kwargs)
            except TypeError as exc:
                self._status(
                    "fallback: Bergson GradientProcessor constructor rejected keyword args "
                    f"({type(exc).__name__}: {exc}); retrying with attribute assignment",
                    root_only=not self.execution_context.is_distributed,
                )
                processor = self.runtime.GradientProcessor()
                if processor_projection_dim is not None:
                    setattr(processor, "projection_dim", processor_projection_dim)
                if candidate_normalizers:
                    setattr(processor, "normalizers", candidate_normalizers)

        preprocess_cfg = None
        if self.runtime.PreprocessConfig is not None:  # pragma: no branch - optional API
            try:
                preprocess_cfg = self.runtime.PreprocessConfig(aggregation="none")
            except TypeError as exc:
                self._status(
                    "fallback: Bergson PreprocessConfig constructor rejected aggregation='none' "
                    f"({type(exc).__name__}: {exc}); retrying with default constructor",
                    root_only=not self.execution_context.is_distributed,
                )
                preprocess_cfg = self.runtime.PreprocessConfig()

        if self.runtime.IndexConfig is None:
            raise ImportError(
                _missing_bergson_message("The installed Bergson package does not expose IndexConfig.")
            )
        # Bergson's indexing API expects a named loss family even though the
        # query-time attribution objective is our custom EWoK softplus loss.
        # The candidate index stores per-example training gradients; the custom
        # query loss only enters later when we build query gradients.
        index_kwargs = {
            "run_path": str(index_dir),
            "model": str(checkpoint.path),
            "loss_fn": "ce",
            # Full raw TrackStar preconditioners are prohibitively large for
            # GPT-2-medium-scale modules. This integration uses Bergson for the
            # projected candidate index and query-time scoring, but does not
            # rely on saved preconditioners in the main path.
            "skip_preconditioners": True,
        }
        if projection_dim > 0:
            index_kwargs["projection_dim"] = projection_dim
        if self.execution_context.distributed_mode == "fsdp":
            index_kwargs["fsdp"] = True
        try:
            index_cfg = self.runtime.IndexConfig(**index_kwargs)
        except TypeError as exc:
            self._status(
                "fallback: Bergson IndexConfig constructor rejected the full TrackStar kwargs "
                f"({type(exc).__name__}: {exc}); retrying with model-only initialization",
                root_only=not self.execution_context.is_distributed,
            )
            index_cfg = self.runtime.IndexConfig(model=str(checkpoint.path))
        if hasattr(index_cfg, "skip_preconditioners"):
            index_cfg.skip_preconditioners = True

        collect_kwargs: dict[str, Any] = {}
        if preprocess_cfg is not None:
            collect_kwargs["preprocess_cfg"] = preprocess_cfg

        with _patched_candidate_forward_for_external_shift(self.model):
            with _periodic_heartbeat(
                enabled=self.config.show_progress and self.execution_context.is_root,
                interval_seconds=30.0,
                emit=lambda elapsed: self._status(
                    "candidate index build still running "
                    f"for step={checkpoint.step} ({elapsed / 60.0:.1f} min elapsed)",
                    root_only=True,
                ),
            ):
                self.runtime.collect_gradients(
                    self.model,
                    candidate_dataset,
                    processor,
                    index_cfg,
                    **collect_kwargs,
                )
        self._status(
            f"finished candidate index build for step={checkpoint.step} at {index_dir}",
            root_only=not self.execution_context.is_distributed,
        )

        if self.execution_context.is_distributed and dist.is_initialized():
            dist.barrier()
        if self.execution_context.is_root:
            self._promote_partial_index(index_dir)
            _write_json(metadata_path, expected.to_json())
        if self.execution_context.is_distributed and dist.is_initialized():
            dist.barrier()
        return index_dir, expected

    @staticmethod
    def _normalize_loaded_gradients(loaded: Any) -> dict[str, np.ndarray]:
        """Normalize Bergson's structured gradient outputs into 2D numpy arrays.

        Different Bergson versions expose loaded gradients as plain dicts,
        dataset-like objects, or dicts containing nested `grads`/`values`
        entries. The scoring path only needs a `module_name -> [n, d]` mapping,
        so we coerce the supported shapes into that minimal common form.
        """

        if isinstance(loaded, np.ndarray) and loaded.dtype.names is not None:
            items = ((name, loaded[name]) for name in loaded.dtype.names)
        elif isinstance(loaded, dict):
            items = loaded.items()
        elif hasattr(loaded, "column_names"):
            items = ((name, loaded[name]) for name in loaded.column_names)
        elif hasattr(loaded, "items"):
            items = loaded.items()
        else:
            raise TypeError(f"Unsupported structured gradient container type: {type(loaded)!r}")

        normalized: dict[str, np.ndarray] = {}
        for name, value in items:
            grads = value
            if isinstance(value, dict):
                grads = value.get("grads", value.get("values", value))
            array = np.asarray(grads, dtype=np.float64)
            if array.ndim != 2:
                continue
            normalized[str(name)] = array
        if not normalized:
            raise ValueError("No structured index gradients were loaded from Bergson")
        return normalized

    def _load_index_gradients(self, index_dir: Path) -> dict[str, np.ndarray]:
        """Load the structured candidate gradients that back query-time scoring."""

        gradient_dir = self._resolve_gradient_dir(index_dir)
        if gradient_dir is None:
            raise FileNotFoundError(
                "Could not find Bergson gradient artifacts under either "
                f"{index_dir} or {self._partial_index_dir(index_dir)}"
            )
        loaded = self.runtime.load_gradients(str(gradient_dir), structured=True)
        return self._normalize_loaded_gradients(loaded)

    @staticmethod
    def _normalize_score_matrix(
        scores: Any,
        *,
        num_targets: int,
        num_candidates: int,
    ) -> np.ndarray:
        """Accept either Bergson score orientation and return `[targets, candidates]`.

        Some scorer paths naturally produce `query x index`, while others expose
        `index x query`. Export and comparison code in this repo expects rows to
        be targets and columns to be candidate examples, so we normalize here.
        """

        matrix = np.asarray(scores, dtype=np.float64)
        if matrix.shape == (num_targets, num_candidates):
            return matrix
        if matrix.shape == (num_candidates, num_targets):
            return matrix.T
        raise ValueError(
            "Unexpected Bergson score matrix shape "
            f"{matrix.shape}; expected ({num_targets}, {num_candidates}) or "
            f"({num_candidates}, {num_targets})"
        )

    @staticmethod
    def _unit_normalize_query_grads(
        query_grads: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Unit-normalize query rows across the full concatenated module space."""

        if not query_grads:
            return {}

        squared_norms: torch.Tensor | None = None
        for grads in query_grads.values():
            if grads.ndim != 2:
                raise ValueError(f"Expected 2D query grads, got {grads.shape}")
            contrib = grads.to(dtype=torch.float32).pow(2).sum(dim=1)
            squared_norms = contrib if squared_norms is None else squared_norms + contrib

        assert squared_norms is not None
        inv_norms = squared_norms.clamp_min_(1e-12).rsqrt_().unsqueeze(1)
        return {
            name: grads * inv_norms.to(device=grads.device, dtype=grads.dtype)
            for name, grads in query_grads.items()
        }

    def _score_queries_with_runtime(
        self,
        *,
        index_grads: dict[str, np.ndarray],
        query_grads: dict[str, torch.Tensor],
        split_preconditioners: dict[str, torch.Tensor] | None,
        num_targets: int,
        num_candidates: int,
    ) -> np.ndarray:
        """Score query gradients with Bergson's scorer when available.

        The optional scorer path is preferred because it lets Bergson apply its
        native scoring logic. TrackStar now defaults to cosine-normalized
        similarity, so we unit-normalize the query rows here and ask Bergson to
        normalize the index rows at score time. If that constructor or call path
        is absent in the installed version, we fall back to an equivalent local
        implementation in `_score_queries_direct(...)`.
        """

        if split_preconditioners:
            self._status(
                "applying mixed Hessian-style correction before cosine scoring",
                root_only=not self.execution_context.is_distributed,
            )
            query_grads = {
                name: (
                    grads.to(
                        device=split_preconditioners[name].device,
                        dtype=split_preconditioners[name].dtype,
                    )
                    @ split_preconditioners[name]
                ).to(device="cpu", dtype=grads.dtype)
                for name, grads in query_grads.items()
            }
            query_grads = self._unit_normalize_query_grads(query_grads)
            return self._score_queries_direct(
                index_grads=index_grads,
                query_grads=query_grads,
                split_preconditioners=split_preconditioners,
            )

        query_grads = self._unit_normalize_query_grads(query_grads)
        if self.runtime.Scorer is None:
            self._status(
                "fallback: Bergson Scorer is unavailable; using the direct local cosine scorer",
                root_only=not self.execution_context.is_distributed,
            )
            return self._score_queries_direct(
                index_grads=index_grads,
                query_grads=query_grads,
                split_preconditioners=None,
            )
        try:
            scorer = self.runtime.Scorer(
                query_grads=query_grads,
                modules=list(query_grads),
                writer=None,
                device=self.config.device,
                dtype=torch.float32,
                unit_normalize=True,
            )
            scores = scorer.score(index_grads)
            return self._normalize_score_matrix(
                scores,
                num_targets=num_targets,
                num_candidates=num_candidates,
            )
        except Exception as exc:
            self._status(
                "fallback: Bergson runtime scorer failed "
                f"({type(exc).__name__}: {exc}); using the direct local cosine scorer",
                root_only=not self.execution_context.is_distributed,
            )
            return self._score_queries_direct(
                index_grads=index_grads,
                query_grads=query_grads,
                split_preconditioners=None,
            )

    @staticmethod
    def _score_queries_direct(
        *,
        index_grads: dict[str, np.ndarray],
        query_grads: dict[str, torch.Tensor],
        split_preconditioners: dict[str, torch.Tensor] | None,
    ) -> np.ndarray:
        """Fallback scorer that computes cosine similarity directly.

        This path keeps the backend robust when Bergson's higher-level scoring
        API changes shape. Both index and query gradients are assumed to be
        organized as `[examples, feature_dim]` for each module name.
        """

        modules = [name for name in query_grads if name in index_grads]
        if not modules:
            raise ValueError("No overlapping gradient modules were found between Bergson index and query grads")
        first_module = modules[0]
        num_targets = int(query_grads[first_module].shape[0])
        num_candidates = int(index_grads[first_module].shape[0])
        score_matrix = np.zeros((num_targets, num_candidates), dtype=np.float64)
        index_sq_norms = np.zeros((num_candidates,), dtype=np.float64)
        split_preconditioners_np = (
            {
                name: split_preconditioners[name].detach().cpu().numpy().astype(np.float64, copy=False)
                for name in modules
                if name in split_preconditioners
            }
            if split_preconditioners
            else {}
        )
        for name in modules:
            query = np.asarray(query_grads[name].detach().cpu(), dtype=np.float64)
            index = np.asarray(index_grads[name], dtype=np.float64)
            if query.ndim != 2 or index.ndim != 2:
                raise ValueError(f"Expected 2D grads for module {name!r}, got {query.shape} and {index.shape}")
            if query.shape[1] != index.shape[1]:
                raise ValueError(
                    f"Gradient dimension mismatch for module {name!r}: query {query.shape[1]} vs index {index.shape[1]}"
                )
            if name in split_preconditioners_np:
                index = index @ split_preconditioners_np[name]
            score_matrix += query @ index.T
            index_sq_norms += np.square(index).sum(axis=1)

        index_inv_norms = np.reciprocal(np.sqrt(np.clip(index_sq_norms, 1e-12, None)))
        return score_matrix * index_inv_norms[None, :]

    def _assembly_dir(self, checkpoint: CheckpointRef, bundle: EWOKTargetBundle) -> Path:
        """Return the directory used for temporary distributed shard files."""

        return self._step_cache_dir(checkpoint) / f"assemble_{_target_bundle_fingerprint(bundle)}"

    def _write_local_shard(
        self,
        *,
        checkpoint: CheckpointRef,
        bundle: EWOKTargetBundle,
        shard_result: BergsonShardResult,
    ) -> Path:
        """Write this rank's local results to disk for later root assembly."""

        assembly_dir = self._assembly_dir(checkpoint, bundle)
        shard_path = assembly_dir / f"rank{self.execution_context.rank:05d}.pt"
        _serialize_shard(shard_path, shard_result)
        return shard_path

    def _assemble_shards(
        self,
        *,
        checkpoint: CheckpointRef,
        bundle: EWOKTargetBundle,
        num_candidates: int,
    ) -> tuple[np.ndarray, tuple[TargetDiagnostics, ...]]:
        """Load all rank-local shard files and reconstruct the final result."""

        assembly_dir = self._assembly_dir(checkpoint, bundle)
        shard_paths = [assembly_dir / f"rank{rank:05d}.pt" for rank in range(self.execution_context.world_size)]
        missing = [str(path) for path in shard_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Bergson shard files: {missing}")
        shard_results = [_deserialize_shard(path) for path in shard_paths]
        return assemble_sharded_scores(
            num_targets=len(bundle.items),
            num_candidates=num_candidates,
            shard_results=shard_results,
        )

    def score_checkpoint(
        self,
        *,
        checkpoint: CheckpointRef,
        manifest: ExampleManifest,
        candidate_selection: CandidateSelection,
        target_bundle: EWOKTargetBundle,
    ) -> CheckpointScores | None:
        """Run the full TrackStar attribution workflow for one checkpoint.

        The outer attribution pipeline has already decided:

        - which checkpoint is being scored
        - which training examples are candidate influences
        - which EWoK targets define the query set

        This method's job is to turn those decisions into a dense attribution
        matrix:

        1. reload the checkpoint into the shared model
        2. build or reuse the candidate gradient index
        3. compute item-level query gradients under the custom EWoK loss
        4. score every target against every candidate example
        5. return the unchanged `CheckpointScores` shape used by export/compare

        Query gradients stay item-level in the main path even though the query
        utilities also support reduced views. That preserves the existing export
        contract: one score row per EWoK item.
        """

        self._load_checkpoint_into_model(checkpoint)

        candidate_ids = tuple(int(candidate_id) for candidate_id in candidate_selection.candidate_ids)
        self._status(
            "scoring checkpoint "
            f"step={checkpoint.step} against {len(candidate_ids)} candidate example(s) "
            f"and {len(target_bundle.items)} total target(s)",
            root_only=True,
        )
        candidate_dataset = BergsonCandidateDataset(manifest, candidate_ids)
        index_dir, _ = self._build_or_reuse_candidate_index(
            checkpoint=checkpoint,
            candidate_dataset=candidate_dataset,
            candidate_ids=candidate_ids,
        )

        local_target_indices = _target_indices_for_rank(
            len(target_bundle.items),
            self.execution_context.rank,
            self.execution_context.world_size,
        )
        local_bundle = subset_target_bundle(target_bundle, local_target_indices)
        self._status(
            f"assigned {len(local_bundle.items)} local target(s) to this rank",
            root_only=not self.execution_context.is_distributed,
        )
        local_diagnostics = score_bundle_diagnostics(
            self.model,
            self.tokenizer,
            local_bundle,
            batch_size=self.config.batch_size,
            temperature=self.config.temperature,
            show_progress=self.config.show_progress and self.execution_context.is_root,
            progress_desc=f"step {checkpoint.step} EWoK diagnostics",
        )

        if local_bundle.items:
            self._status(f"loading index gradients from {index_dir}", root_only=not self.execution_context.is_distributed)
            index_grads = self._load_index_gradients(index_dir)
            query_weight_normalizers: dict[str, Any] = {}
            if self._candidate_uses_adam_second_moment_correction(checkpoint):
                query_weight_normalizers = self._load_candidate_adam_normalizers(checkpoint)
                corrected_modules = sum(1 for name in index_grads if name in query_weight_normalizers)
                self._status(
                    "applying Adam second-moment correction to "
                    f"query gradients for {corrected_modules} module(s)",
                    root_only=not self.execution_context.is_distributed,
                )
            self._status(
                f"collecting query gradients for {len(local_bundle.items)} local target(s) "
                f"across {len(index_grads)} module(s)",
                root_only=not self.execution_context.is_distributed,
            )
            query_target_ids, query_grads = collect_query_module_grads(
                self.model,
                self.tokenizer,
                local_bundle,
                batch_size=self.config.batch_size,
                temperature=self.config.temperature,
                module_names=tuple(index_grads),
                reduction="item",
                projection_dim=int(self.config.proj_dim) if self.config.use_fast_jl else None,
                projection_type="rademacher",
                weight_normalizers=query_weight_normalizers,
                show_progress=self.config.show_progress and self.execution_context.is_root,
                progress_desc=f"step {checkpoint.step} query gradients",
            )
            if tuple(query_target_ids) != local_bundle.target_ids:
                raise ValueError("Item-level query gradient ordering drifted away from target bundle ordering")
            self._status(
                f"scoring {len(local_bundle.items)} target gradient(s) against {len(candidate_ids)} candidate example(s)",
                root_only=not self.execution_context.is_distributed,
            )
            split_preconditioners = self._build_mixed_hessian_preconditioners(
                index_grads=index_grads,
                query_grads=query_grads,
            )
            local_score_matrix = self._score_queries_with_runtime(
                index_grads=index_grads,
                query_grads=query_grads,
                split_preconditioners=split_preconditioners,
                num_targets=len(local_bundle.items),
                num_candidates=len(candidate_ids),
            )
        else:
            local_score_matrix = np.zeros((0, len(candidate_ids)), dtype=np.float64)
            self._status("no local targets assigned; creating an empty local score shard")

        if self.execution_context.world_size == 1:
            self._status(
                f"finished checkpoint step={checkpoint.step}; score matrix shape={local_score_matrix.shape}",
                root_only=True,
            )
            return CheckpointScores(
                checkpoint_step=checkpoint.step,
                checkpoint_path=str(checkpoint.path),
                candidate_ids=candidate_ids,
                target_ids=target_bundle.target_ids,
                score_matrix=local_score_matrix,
                target_diagnostics=local_diagnostics,
            )

        self._write_local_shard(
            checkpoint=checkpoint,
            bundle=target_bundle,
            shard_result=BergsonShardResult(
                rank=self.execution_context.rank,
                target_indices=local_target_indices,
                score_matrix=local_score_matrix,
                diagnostics=local_diagnostics,
            ),
        )
        self._status(
            f"wrote local shard for checkpoint step={checkpoint.step} with {len(local_target_indices)} target item(s)"
        )
        if dist.is_initialized():
            dist.barrier()
        if not self.execution_context.is_root:
            return None

        self._status(f"assembling shard results for checkpoint step={checkpoint.step}", root_only=True)
        score_matrix, diagnostics = self._assemble_shards(
            checkpoint=checkpoint,
            bundle=target_bundle,
            num_candidates=len(candidate_ids),
        )
        self._status(
            f"finished checkpoint step={checkpoint.step}; assembled score matrix shape={score_matrix.shape}",
            root_only=True,
        )
        return CheckpointScores(
            checkpoint_step=checkpoint.step,
            checkpoint_path=str(checkpoint.path),
            candidate_ids=candidate_ids,
            target_ids=target_bundle.target_ids,
            score_matrix=score_matrix,
            target_diagnostics=diagnostics,
        )


def build_backend(
    *,
    config: TrackstarConfig,
    model: torch.nn.Module,
    tokenizer,
    execution_context: "RunExecutionContext",
) -> BergsonAttributionBackend:
    """Construct the TrackStar backend behind the shared backend factory API."""

    return BergsonAttributionBackend(
        config=config,
        model=model,
        tokenizer=tokenizer,
        execution_context=execution_context,
    )


__all__ = [
    "BergsonAttributionBackend",
    "BergsonShardResult",
    "assemble_sharded_scores",
    "build_backend",
]
