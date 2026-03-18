"""TRAK integration for BOS rows and shared EWoK scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..common.candidates import CandidateSelection
from ..common.checkpoints import CheckpointRef, load_checkpoint_state_dict
from ..common.ewok_targets import (
    CheckpointScores,
    EWOKTargetBundle,
    compute_softplus_score,
    compute_view_margins,
    iter_target_batches,
    reduce_masked_token_logprobs,
    score_target_batch,
    score_target_bundle,
)
from ..common.row_dataset import FiniteBOSRowDataset, RowManifest, iter_row_batches
from .config import TRAKConfig

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
        "running `analysis.attribution.run_trak`." + detail
    )


def _trak_is_available() -> bool:
    return _TRAKer is not None and _AbstractModelOutput is not object


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
        state_dict = load_checkpoint_state_dict(checkpoint.path, model=self.model)
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
    "TrakAttributionBackend",
    "build_backend",
    "compute_softplus_score",
    "compute_view_margins",
    "reduce_masked_token_logprobs",
    "row_mean_correct_token_logp",
    "score_target_batch",
    "score_target_bundle",
]
