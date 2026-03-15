"""TRAK integration and scalar scoring for BOS rows and EWoK targets.

This module is the only place in the package that should know about the
external ``trak`` library. It defines the scalar outputs used on the train and
target sides, handles optional ``trak`` availability, computes EWoK margin
diagnostics, and exposes the backend that featurizes candidate rows and scores
targets for one checkpoint at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.ewok import BABYLM_COMPLETION_CHOICE, EWOK_PAPER_CONTEXT_SENSITIVITY

from .candidates import CandidateSelection
from .checkpoints import CheckpointRef, load_checkpoint_state_dict
from .config import TRAKConfig
from .ewok_targets import EWOKTargetBundle, iter_target_batches
from .row_dataset import FiniteBOSRowDataset, RowManifest, iter_row_batches

try:
    from trak import TRAKer as _TRAKer
except ImportError as exc:  # pragma: no cover - exercised via build_backend error path
    _TRAKer = None
    _TRAK_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - depends on optional dependency
    _TRAK_IMPORT_ERROR = None

try:  # pragma: no cover - depends on optional dependency
    from trak.modelout_functions import AbstractModelOutput as _AbstractModelOutput
except ImportError as exc:  # pragma: no cover - exercised via build_backend error path
    _AbstractModelOutput = object
    if _TRAK_IMPORT_ERROR is None:
        _TRAK_IMPORT_ERROR = exc


def _missing_trak_message() -> str:
    detail = "" if _TRAK_IMPORT_ERROR is None else f" Original import error: {_TRAK_IMPORT_ERROR}"
    return (
        "The `traker` package is required to run BOS-row TRAK attribution. "
        "Install it with `pip install traker` or `pip install 'traker[fast]'` before "
        "running `analysis.trak.run_trak`." + detail
    )


def _trak_is_available() -> bool:
    return _TRAKer is not None and _AbstractModelOutput is not object


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _resolve_bos_token_id(tokenizer) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must define bos_token_id, eos_token_id, or pad_token_id")


def _is_train_batch(batch: Any) -> bool:
    return isinstance(batch, tuple) and len(batch) == 2


def _is_target_batch(batch: Any) -> bool:
    return isinstance(batch, tuple) and len(batch) == 12


def row_mean_correct_token_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits
    token_logprobs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs.mean(dim=1)


def reduce_masked_token_logprobs(
    token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction!r}")
    token_mask = token_mask.to(dtype=token_logprobs.dtype)
    summed = (token_logprobs * token_mask).sum(dim=1)
    if reduction == "sum":
        return summed
    counts = token_mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def compute_view_margins(
    *,
    score_view: str,
    s11: torch.Tensor,
    s12: torch.Tensor,
    s22: torch.Tensor,
    s21: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score_view == BABYLM_COMPLETION_CHOICE:
        return s11 - s12, s22 - s21
    if score_view == EWOK_PAPER_CONTEXT_SENSITIVITY:
        return s11 - s21, s22 - s12
    raise ValueError(f"Unsupported score_view: {score_view!r}")


def compute_softplus_score(
    *,
    score_view: str,
    s11: torch.Tensor,
    s12: torch.Tensor,
    s22: torch.Tensor,
    s21: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    margin_1, margin_2 = compute_view_margins(
        score_view=score_view,
        s11=s11,
        s12=s12,
        s22=s22,
        s21=s21,
    )
    combined_margin = 0.5 * (margin_1 + margin_2)
    softplus_loss = 0.5 * (
        F.softplus((-margin_1) / temperature) + F.softplus((-margin_2) / temperature)
    )
    score = -softplus_loss
    return score, margin_1, margin_2, combined_margin


def _conditional_target_token_logprobs(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_start: torch.Tensor,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _resolve_model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    target_start = target_start.to(device)

    batch_size = input_ids.shape[0]
    bos_column = torch.full((batch_size, 1), bos_token_id, dtype=input_ids.dtype, device=device)
    bos_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)

    model_input_ids = torch.cat([bos_column, input_ids], dim=1)
    model_attention_mask = torch.cat([bos_mask, attention_mask], dim=1)

    logits = model(input_ids=model_input_ids, attention_mask=model_attention_mask).logits[:, :-1, :]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    valid_lengths = attention_mask.sum(dim=1, keepdim=True)
    target_mask = (positions >= target_start.unsqueeze(1)) & (positions < valid_lengths)
    return token_logprobs, target_mask


def _reduce_conditional_scores(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_start: torch.Tensor,
    bos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_logprobs, target_mask = _conditional_target_token_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_start=target_start,
        bos_token_id=bos_token_id,
    )
    mean_scores = reduce_masked_token_logprobs(token_logprobs, target_mask, reduction="mean")
    sum_scores = reduce_masked_token_logprobs(token_logprobs, target_mask, reduction="sum")
    return mean_scores, sum_scores


@dataclass(frozen=True)
class TargetDiagnostics:
    target_id: str
    domain: str
    score_view: str
    score_reduction: str
    s11_mean: float
    s12_mean: float
    s22_mean: float
    s21_mean: float
    s11_sum: float
    s12_sum: float
    s22_sum: float
    s21_sum: float
    margin_1: float
    margin_2: float
    combined_margin: float
    softplus_loss: float
    score: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointScores:
    checkpoint_step: int
    checkpoint_path: str
    candidate_row_ids: tuple[int, ...]
    target_ids: tuple[str, ...]
    score_matrix: np.ndarray
    target_diagnostics: tuple[TargetDiagnostics, ...]


def score_target_batch(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, ...],
    *,
    score_view: str,
    score_reduction: str,
    temperature: float,
    bos_token_id: int,
) -> dict[str, torch.Tensor]:
    if not _is_target_batch(batch):
        raise TypeError("Expected an EWoK target batch with 12 tensor fields")

    c1t1_ids, c1t1_mask, c1t1_start = batch[0], batch[1], batch[2]
    c1t2_ids, c1t2_mask, c1t2_start = batch[3], batch[4], batch[5]
    c2t2_ids, c2t2_mask, c2t2_start = batch[6], batch[7], batch[8]
    c2t1_ids, c2t1_mask, c2t1_start = batch[9], batch[10], batch[11]

    s11_mean, s11_sum = _reduce_conditional_scores(
        model,
        input_ids=c1t1_ids,
        attention_mask=c1t1_mask,
        target_start=c1t1_start,
        bos_token_id=bos_token_id,
    )
    s12_mean, s12_sum = _reduce_conditional_scores(
        model,
        input_ids=c1t2_ids,
        attention_mask=c1t2_mask,
        target_start=c1t2_start,
        bos_token_id=bos_token_id,
    )
    s22_mean, s22_sum = _reduce_conditional_scores(
        model,
        input_ids=c2t2_ids,
        attention_mask=c2t2_mask,
        target_start=c2t2_start,
        bos_token_id=bos_token_id,
    )
    s21_mean, s21_sum = _reduce_conditional_scores(
        model,
        input_ids=c2t1_ids,
        attention_mask=c2t1_mask,
        target_start=c2t1_start,
        bos_token_id=bos_token_id,
    )

    if score_reduction == "mean":
        active_s11, active_s12, active_s22, active_s21 = s11_mean, s12_mean, s22_mean, s21_mean
    elif score_reduction == "sum":
        active_s11, active_s12, active_s22, active_s21 = s11_sum, s12_sum, s22_sum, s21_sum
    else:
        raise ValueError(f"Unsupported score_reduction: {score_reduction!r}")

    score, margin_1, margin_2, combined_margin = compute_softplus_score(
        score_view=score_view,
        s11=active_s11,
        s12=active_s12,
        s22=active_s22,
        s21=active_s21,
        temperature=temperature,
    )

    return {
        "score": score,
        "softplus_loss": -score,
        "margin_1": margin_1,
        "margin_2": margin_2,
        "combined_margin": combined_margin,
        "s11_mean": s11_mean,
        "s12_mean": s12_mean,
        "s22_mean": s22_mean,
        "s21_mean": s21_mean,
        "s11_sum": s11_sum,
        "s12_sum": s12_sum,
        "s22_sum": s22_sum,
        "s21_sum": s21_sum,
    }


def score_target_bundle(
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    *,
    batch_size: int,
    temperature: float,
) -> tuple[TargetDiagnostics, ...]:
    bos_token_id = _resolve_bos_token_id(tokenizer)
    diagnostics: list[TargetDiagnostics] = []
    model.eval()
    with torch.no_grad():
        for prepared in iter_target_batches(bundle, tokenizer, batch_size):
            scores = score_target_batch(
                model,
                prepared.batch,
                score_view=bundle.score_view,
                score_reduction=bundle.score_reduction,
                temperature=temperature,
                bos_token_id=bos_token_id,
            )
            for idx, item in enumerate(prepared.items):
                diagnostics.append(
                    TargetDiagnostics(
                        target_id=item.target_id,
                        domain=item.domain,
                        score_view=bundle.score_view,
                        score_reduction=bundle.score_reduction,
                        s11_mean=float(scores["s11_mean"][idx].item()),
                        s12_mean=float(scores["s12_mean"][idx].item()),
                        s22_mean=float(scores["s22_mean"][idx].item()),
                        s21_mean=float(scores["s21_mean"][idx].item()),
                        s11_sum=float(scores["s11_sum"][idx].item()),
                        s12_sum=float(scores["s12_sum"][idx].item()),
                        s22_sum=float(scores["s22_sum"][idx].item()),
                        s21_sum=float(scores["s21_sum"][idx].item()),
                        margin_1=float(scores["margin_1"][idx].item()),
                        margin_2=float(scores["margin_2"][idx].item()),
                        combined_margin=float(scores["combined_margin"][idx].item()),
                        softplus_loss=float(scores["softplus_loss"][idx].item()),
                        score=float(scores["score"][idx].item()),
                    )
                )
    return tuple(diagnostics)


class _BOSRowEWOKModelOutput(_AbstractModelOutput):  # pragma: no cover - exercised only when traker is installed
    def __init__(
        self,
        *,
        tokenizer,
        score_view: str,
        score_reduction: str,
        temperature: float,
    ) -> None:
        super().__init__()
        self._bos_token_id = _resolve_bos_token_id(tokenizer)
        self._score_view = score_view
        self._score_reduction = score_reduction
        self._temperature = float(temperature)

    def get_output(self, *args):
        model = args[0]
        batch = args[-1]
        if _is_train_batch(batch):
            return row_mean_correct_token_logp(model, batch[0], batch[1])
        if _is_target_batch(batch):
            return score_target_batch(
                model,
                batch,
                score_view=self._score_view,
                score_reduction=self._score_reduction,
                temperature=self._temperature,
                bos_token_id=self._bos_token_id,
            )["score"]
        raise TypeError(f"Unsupported batch schema with {len(batch)} fields")

    def get_out_to_loss_grad(self, *args):
        output = self.get_output(*args)
        batch = args[-1]
        if _is_train_batch(batch):
            return torch.full_like(output, fill_value=-1.0)
        if _is_target_batch(batch):
            return torch.ones_like(output)
        raise TypeError(f"Unsupported batch schema with {len(batch)} fields")


class TrakAttributionBackend:
    """Checkpoint-local TRAK scorer for BOS rows against EWoK targets."""

    def __init__(self, *, config: TRAKConfig, model: torch.nn.Module, tokenizer) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self._model_output = _BOSRowEWOKModelOutput(
            tokenizer=tokenizer,
            score_view=config.ewok_score_view,
            score_reduction=config.score_reduction,
            temperature=config.temperature,
        )

    def _load_checkpoint_into_model(self, checkpoint: CheckpointRef) -> dict[str, torch.Tensor]:
        state_dict = load_checkpoint_state_dict(checkpoint.path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.config.device)
        self.model.eval()
        return state_dict

    def _new_traker(self, *, train_set_size: int, save_dir: Path):
        if not _trak_is_available():
            raise ImportError(_missing_trak_message())
        return _TRAKer(
            model=self.model,
            task=self._model_output,
            train_set_size=int(train_set_size),
            save_dir=str(save_dir),
            device=self.config.device,
            proj_dim=int(self.config.proj_dim),
        )

    @staticmethod
    def _normalize_score_matrix(
        scores: Any,
        *,
        num_targets: int,
        num_candidates: int,
    ) -> np.ndarray:
        matrix = np.asarray(scores, dtype=np.float64)
        if matrix.shape == (num_targets, num_candidates):
            return matrix
        if matrix.shape == (num_candidates, num_targets):
            return matrix.T
        raise ValueError(
            "Unexpected TRAK score matrix shape "
            f"{matrix.shape}; expected ({num_targets}, {num_candidates}) or "
            f"({num_candidates}, {num_targets})"
        )

    def score_checkpoint(
        self,
        *,
        checkpoint: CheckpointRef,
        manifest: RowManifest,
        candidate_selection: CandidateSelection,
        target_bundle: EWOKTargetBundle,
    ) -> CheckpointScores:
        state_dict = self._load_checkpoint_into_model(checkpoint)
        target_diagnostics = score_target_bundle(
            self.model,
            self.tokenizer,
            target_bundle,
            batch_size=self.config.batch_size,
            temperature=self.config.temperature,
        )

        candidate_row_ids = tuple(int(row_id) for row_id in candidate_selection.row_ids)
        dataset = FiniteBOSRowDataset(manifest, candidate_row_ids)
        save_dir = Path(self.config.cache_dir) / f"step{checkpoint.step:08d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        traker = self._new_traker(train_set_size=len(dataset), save_dir=save_dir)

        model_id = 0
        traker.load_checkpoint(state_dict, model_id=model_id)
        for prepared in iter_row_batches(dataset, batch_size=self.config.batch_size):
            traker.featurize(
                batch=prepared.batch,
                num_samples=len(prepared.row_ids),
                inds=prepared.local_inds,
            )
        traker.finalize_features()

        exp_name = f"{self.config.exp_name}_step{checkpoint.step:08d}"
        traker.start_scoring_checkpoint(
            exp_name=exp_name,
            checkpoint=state_dict,
            model_id=model_id,
            num_targets=len(target_bundle.items),
        )
        for prepared in iter_target_batches(target_bundle, self.tokenizer, self.config.batch_size):
            traker.score(
                batch=prepared.batch,
                num_samples=len(prepared.items),
            )
        raw_scores = traker.finalize_scores(exp_name=exp_name)
        score_matrix = self._normalize_score_matrix(
            raw_scores,
            num_targets=len(target_bundle.items),
            num_candidates=len(candidate_row_ids),
        )

        return CheckpointScores(
            checkpoint_step=checkpoint.step,
            checkpoint_path=str(checkpoint.path),
            candidate_row_ids=candidate_row_ids,
            target_ids=target_bundle.target_ids,
            score_matrix=score_matrix,
            target_diagnostics=target_diagnostics,
        )


def build_backend(*, config: TRAKConfig, model: torch.nn.Module, tokenizer) -> TrakAttributionBackend:
    if not _trak_is_available():
        raise ImportError(_missing_trak_message())
    return TrakAttributionBackend(config=config, model=model, tokenizer=tokenizer)


__all__ = [
    "CheckpointScores",
    "TargetDiagnostics",
    "TrakAttributionBackend",
    "build_backend",
    "compute_softplus_score",
    "compute_view_margins",
    "reduce_masked_token_logprobs",
    "row_mean_correct_token_logp",
    "score_target_batch",
    "score_target_bundle",
]
