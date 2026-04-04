"""One-step target-loss sanity checks for TrackStar attribution outputs.

This module answers a simple debugging question:

Given a fixed checkpoint ``theta`` and a targeted EWoK query loss ``L_Q(theta)``,
do tiny SGD steps on top-ranked candidates reduce that query loss more often
than matched-random or bottom-ranked candidates?

The implementation intentionally stays close to the existing attribution and
training surfaces:

- candidate pools come from the saved TrackStar/attribution row summaries
- candidate minibatches are reconstructed as exact training windows
- the target loss is the same EWoK softplus loss used by TrackStar queries
- the update is a tiny plain SGD step on the candidate CE loss
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..build_matched_cpt_pools import SUPPORTED_SCORE_MODES, load_candidate_score_frame
from ..common.checkpoints import build_model_from_checkpoint, load_tokenizer_from_checkpoint
from ..common.export import write_json, write_jsonl
from ..common.ewok_targets import (
    EWOKTargetBundle,
    build_ewok_targets,
    iter_target_batches,
    score_target_batch,
)
from ..common.training_examples import FiniteTrainingExampleDataset, build_example_manifest


DEFAULT_SCORE_MODE = "net_pooled"
DEFAULT_GROUP_SIZE = 1_000
DEFAULT_CANDIDATE_BATCH_SIZE = 4
DEFAULT_NUM_TRIALS = 64
DEFAULT_UPDATE_LR = 1e-5
DEFAULT_TARGET_BATCH_SIZE = 8
DEFAULT_SEED = 42
DEFAULT_DEVICE = "auto"
DEFAULT_EWOK_VARIANT = "fast"
DEFAULT_SCORE_VIEW = "babylm_completion_choice"
DEFAULT_SCORE_REDUCTION = "mean"
DEFAULT_TEMPERATURE = 1.0
GROUP_NAMES = ("top", "matched_random", "bottom")


@dataclass(frozen=True)
class CandidateGroup:
    name: str
    frame: pd.DataFrame

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.frame["candidate_id"].tolist())


def _build_tqdm(*, enabled: bool, total: int, desc: str, unit: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit=unit)


def _resolve_bos_token_id(tokenizer) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must define bos_token_id, eos_token_id, or pad_token_id")


def _resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device='cuda' but CUDA is not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device={raw!r}; expected one of ('auto', 'cuda', 'cpu')")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_attribution_defaults(attribution_dir: str | Path) -> dict[str, Any]:
    config_path = Path(attribution_dir).expanduser().resolve() / "config.json"
    if not config_path.exists():
        return {}
    return _read_json(config_path)


def _infer_seq_len_from_scored_frame(frame: pd.DataFrame) -> int:
    token_counts = (
        frame["token_offset_end"].astype(int) - frame["token_offset_start"].astype(int)
    ).dropna()
    if token_counts.empty:
        raise ValueError("Could not infer seq_len from row summary: missing token offsets")
    unique = sorted(int(value) for value in token_counts.unique().tolist())
    if len(unique) != 1:
        raise ValueError(f"Expected one token_count in row summary, got {unique}")
    seq_len = int(unique[0]) - 1
    if seq_len <= 0:
        raise ValueError(f"Derived non-positive seq_len={seq_len} from token_count={unique[0]}")
    return seq_len


def _resolve_candidate_kind(frame: pd.DataFrame) -> str:
    kinds = sorted(str(value) for value in frame["candidate_kind"].dropna().unique().tolist())
    if len(kinds) != 1:
        raise ValueError(f"Expected one candidate_kind in row summary, got {kinds}")
    return kinds[0]


def _sample_random_matched_controls(
    *,
    treated: pd.DataFrame,
    candidates: pd.DataFrame,
    rng: np.random.Generator,
    excluded_candidate_ids: set[int] | None = None,
    allow_relaxed_shard_match: bool = True,
) -> pd.DataFrame:
    excluded_ids = {int(value) for value in (excluded_candidate_ids or set())}
    excluded_ids.update(int(value) for value in treated["candidate_id"].tolist())

    available = candidates.loc[~candidates["candidate_id"].isin(excluded_ids)].copy()
    used_candidate_ids: set[int] = set()
    rows: list[dict[str, Any]] = []

    for treated_row in treated.itertuples(index=False):
        treated_series = pd.Series(treated_row._asdict())
        pool = available.loc[~available["candidate_id"].isin(used_candidate_ids)].copy()
        if pool.empty:
            raise ValueError("Matched-random sampling ran out of available candidates")

        base_mask = (
            (pool["candidate_kind"] == treated_series["candidate_kind"])
            & (pool["token_count"] == treated_series["token_count"])
        )
        same_shard_mask = base_mask & (pool["shard_path"] == treated_series["shard_path"])

        level_frames: list[tuple[str, pd.DataFrame]] = [
            ("exact_shard_random", pool.loc[same_shard_mask].copy()),
        ]
        if allow_relaxed_shard_match:
            level_frames.append(("matched_kind_tokens_random", pool.loc[base_mask].copy()))

        chosen_row: pd.Series | None = None
        chosen_level: str | None = None
        for level_name, level_frame in level_frames:
            if level_frame.empty:
                continue
            choice_index = int(rng.integers(len(level_frame)))
            chosen_row = level_frame.iloc[choice_index]
            chosen_level = level_name
            break

        if chosen_row is None or chosen_level is None:
            raise ValueError(
                "Could not sample a matched-random control for treated candidate "
                f"{int(treated_series['candidate_id'])}"
            )

        chosen_candidate_id = int(chosen_row["candidate_id"])
        used_candidate_ids.add(chosen_candidate_id)
        record = dict(chosen_row)
        record["match_level"] = chosen_level
        record["matched_to_candidate_id"] = int(treated_series["candidate_id"])
        rows.append(record)

    return pd.DataFrame.from_records(rows).reset_index(drop=True)


def build_candidate_groups(
    scored: pd.DataFrame,
    *,
    group_size: int,
    rng: np.random.Generator,
    allow_relaxed_shard_match: bool = True,
) -> dict[str, CandidateGroup]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    working = scored.copy().dropna(subset=["selection_score"]).reset_index(drop=True)
    if len(working) < int(group_size) * 3:
        raise ValueError(
            f"Need at least {int(group_size) * 3} scored candidates to build top/random/bottom groups; "
            f"found {len(working)}"
        )

    top = (
        working.sort_values(["selection_score", "candidate_id"], ascending=[False, True])
        .head(int(group_size))
        .reset_index(drop=True)
    )

    remaining_for_bottom = working.loc[~working["candidate_id"].isin(top["candidate_id"])].copy()
    bottom = (
        remaining_for_bottom.sort_values(["selection_score", "candidate_id"], ascending=[True, True])
        .head(int(group_size))
        .reset_index(drop=True)
    )
    if len(bottom) < int(group_size):
        raise ValueError("Could not construct a bottom-ranked pool of the requested size")

    matched_random = _sample_random_matched_controls(
        treated=top,
        candidates=working,
        rng=rng,
        excluded_candidate_ids={int(value) for value in bottom["candidate_id"].tolist()},
        allow_relaxed_shard_match=allow_relaxed_shard_match,
    )

    groups = {
        "top": CandidateGroup(name="top", frame=top),
        "matched_random": CandidateGroup(name="matched_random", frame=matched_random),
        "bottom": CandidateGroup(name="bottom", frame=bottom),
    }
    return groups


def compute_query_loss_mean(
    model: torch.nn.Module,
    tokenizer,
    bundle: EWOKTargetBundle,
    *,
    batch_size: int,
    temperature: float,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    bos_token_id = _resolve_bos_token_id(tokenizer)
    total_loss = 0.0
    total_items = 0
    num_batches = max(1, math.ceil(len(bundle.items) / int(batch_size)))
    progress = _build_tqdm(
        enabled=show_progress,
        total=num_batches,
        desc=progress_desc or "Target query loss",
        unit="batch",
    )
    model.eval()
    try:
        with torch.no_grad():
            for prepared in iter_target_batches(bundle, tokenizer, int(batch_size)):
                scores = score_target_batch(
                    model,
                    prepared.batch,
                    score_view=bundle.score_view,
                    score_reduction=bundle.score_reduction,
                    temperature=float(temperature),
                    bos_token_id=bos_token_id,
                )
                total_loss += float(scores["softplus_loss"].sum().item())
                total_items += len(prepared.items)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    if total_items <= 0:
        raise ValueError("Target bundle is empty; cannot compute query loss")
    return total_loss / total_items


def compute_candidate_batch_loss(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    logits = model(input_ids=input_ids).logits
    vocab_size = int(logits.shape[-1])
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="mean",
    )


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def sample_candidate_batch(
    dataset: FiniteTrainingExampleDataset,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if len(dataset) < int(batch_size):
        raise ValueError(
            f"Requested candidate_batch_size={batch_size}, but the pool only has {len(dataset)} rows"
        )
    sample_indices = rng.choice(len(dataset), size=int(batch_size), replace=False)
    samples = [dataset[int(index)] for index in sample_indices.tolist()]
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples], dim=0),
        "labels": torch.stack([sample["labels"] for sample in samples], dim=0),
        "candidate_ids": tuple(int(sample["candidate_id"]) for sample in samples),
        "local_indices": tuple(int(index) for index in sample_indices.tolist()),
    }


def measure_one_step_delta(
    model: torch.nn.Module,
    *,
    base_state: dict[str, torch.Tensor],
    tokenizer,
    bundle: EWOKTargetBundle,
    baseline_query_loss: float,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    update_lr: float,
    target_batch_size: int,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    model.load_state_dict(base_state, strict=True)
    model.to(device)
    model.zero_grad(set_to_none=True)
    model.train()

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    train_loss = compute_candidate_batch_loss(model, input_ids=input_ids, labels=labels)
    train_loss.backward()

    grad_sq_sum = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            grad_sq_sum += float(torch.sum(grad.detach() * grad.detach()).item())
            parameter.add_(grad, alpha=-float(update_lr))

    after_query_loss = compute_query_loss_mean(
        model,
        tokenizer,
        bundle,
        batch_size=int(target_batch_size),
        temperature=float(temperature),
    )
    model.zero_grad(set_to_none=True)
    return {
        "candidate_train_loss": float(train_loss.detach().item()),
        "candidate_grad_norm_l2": float(math.sqrt(max(grad_sq_sum, 0.0))),
        "query_loss_before": float(baseline_query_loss),
        "query_loss_after": float(after_query_loss),
        "delta_q": float(after_query_loss - baseline_query_loss),
    }


def _summarize_trials(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(list(rows))
    summary: dict[str, Any] = {
        "num_trials": int(len(frame)),
    }
    if frame.empty:
        return summary

    for column in (
        "delta_q",
        "candidate_train_loss",
        "candidate_grad_norm_l2",
        "batch_selection_score_mean",
    ):
        if column not in frame.columns:
            continue
        values = frame[column].astype(float)
        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    if "delta_q" in frame.columns:
        delta = frame["delta_q"].astype(float)
        summary["fraction_negative_delta"] = float((delta < 0.0).mean())
        summary["fraction_positive_delta"] = float((delta > 0.0).mean())
    return summary


def run_one_step_sanity_check(
    *,
    base_ckpt: str | Path,
    attribution_dir: str | Path,
    data_dir: str | Path,
    step: int,
    output_dir: str | Path,
    score_mode: str = DEFAULT_SCORE_MODE,
    target_id: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    candidate_batch_size: int = DEFAULT_CANDIDATE_BATCH_SIZE,
    num_trials_per_group: int = DEFAULT_NUM_TRIALS,
    update_lr: float = DEFAULT_UPDATE_LR,
    ewok_filter_spec: str | Path | None = None,
    ewok_variant: str = DEFAULT_EWOK_VARIANT,
    ewok_score_view: str = DEFAULT_SCORE_VIEW,
    score_reduction: str = DEFAULT_SCORE_REDUCTION,
    temperature: float = DEFAULT_TEMPERATURE,
    target_batch_size: int = DEFAULT_TARGET_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    show_progress: bool = True,
) -> dict[str, Path]:
    if score_mode not in SUPPORTED_SCORE_MODES:
        raise ValueError(f"Unsupported score_mode={score_mode!r}; expected one of {SUPPORTED_SCORE_MODES!r}")
    if num_trials_per_group <= 0:
        raise ValueError("num_trials_per_group must be > 0")
    if update_lr <= 0:
        raise ValueError("update_lr must be > 0")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    scored = load_candidate_score_frame(
        attribution_dir=attribution_dir,
        step=int(step),
        score_mode=score_mode,
        target_id=target_id,
    )
    candidate_kind = _resolve_candidate_kind(scored)
    seq_len = _infer_seq_len_from_scored_frame(scored)
    manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=candidate_kind,
        seq_len=seq_len,
    )
    phase_progress = _build_tqdm(
        enabled=show_progress,
        total=4,
        desc="One-step sanity setup",
        unit="phase",
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("candidate manifest")
        phase_progress.update(1)

    groups = build_candidate_groups(
        scored,
        group_size=int(group_size),
        rng=rng,
        allow_relaxed_shard_match=True,
    )
    group_datasets = {
        name: FiniteTrainingExampleDataset(manifest, group.candidate_ids)
        for name, group in groups.items()
    }
    if phase_progress is not None:
        phase_progress.set_postfix_str("candidate groups")
        phase_progress.update(1)

    model_device = _resolve_device(device)
    model = build_model_from_checkpoint(base_ckpt, device=str(model_device))
    tokenizer = load_tokenizer_from_checkpoint(base_ckpt)
    base_state = _clone_state_dict_to_cpu(model)

    bundle = build_ewok_targets(
        score_view=str(ewok_score_view),
        target_scope="overall",
        score_reduction=str(score_reduction),
        variant=str(ewok_variant),
        filter_spec_path=ewok_filter_spec,
        max_targets=0,
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("model + targets")
        phase_progress.update(1)
    baseline_query_loss = compute_query_loss_mean(
        model,
        tokenizer,
        bundle,
        batch_size=int(target_batch_size),
        temperature=float(temperature),
        show_progress=show_progress,
        progress_desc="Baseline target loss",
    )
    if phase_progress is not None:
        phase_progress.set_postfix_str("baseline ready")
        phase_progress.update(1)
        phase_progress.close()

    config_path = output_root / "config.json"
    write_json(
        config_path,
        {
            "base_ckpt": str(Path(base_ckpt).expanduser().resolve()),
            "attribution_dir": str(Path(attribution_dir).expanduser().resolve()),
            "data_dir": str(Path(data_dir).expanduser().resolve()),
            "step": int(step),
            "score_mode": score_mode,
            "target_id": target_id,
            "group_size": int(group_size),
            "candidate_batch_size": int(candidate_batch_size),
            "num_trials_per_group": int(num_trials_per_group),
            "update_lr": float(update_lr),
            "ewok_filter_spec": None if ewok_filter_spec is None else str(Path(ewok_filter_spec).expanduser().resolve()),
            "ewok_variant": str(ewok_variant),
            "ewok_score_view": str(ewok_score_view),
            "score_reduction": str(score_reduction),
            "temperature": float(temperature),
            "target_batch_size": int(target_batch_size),
            "seed": int(seed),
            "device": str(model_device),
            "candidate_kind": str(candidate_kind),
            "seq_len": int(seq_len),
            "baseline_query_loss": float(baseline_query_loss),
            "num_targets": int(len(bundle.items)),
        },
    )

    for name, group in groups.items():
        group.frame.to_csv(output_root / f"group_{name}_candidates.csv", index=False)

    all_rows: list[dict[str, Any]] = []
    progress = _build_tqdm(
        enabled=show_progress,
        total=len(GROUP_NAMES) * int(num_trials_per_group),
        desc=f"One-step sanity trials ({score_mode})",
        unit="batch",
    )
    try:
        for group_name in GROUP_NAMES:
            group = groups[group_name]
            dataset = group_datasets[group_name]
            score_lookup = {
                int(row.candidate_id): float(row.selection_score)
                for row in group.frame.itertuples(index=False)
            }
            for trial_index in range(int(num_trials_per_group)):
                batch = sample_candidate_batch(
                    dataset,
                    batch_size=int(candidate_batch_size),
                    rng=rng,
                )
                metrics = measure_one_step_delta(
                    model,
                    base_state=base_state,
                    tokenizer=tokenizer,
                    bundle=bundle,
                    baseline_query_loss=baseline_query_loss,
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    update_lr=float(update_lr),
                    target_batch_size=int(target_batch_size),
                    temperature=float(temperature),
                    device=model_device,
                )
                candidate_scores = [score_lookup[int(candidate_id)] for candidate_id in batch["candidate_ids"]]
                row = {
                    "group": group_name,
                    "trial_index": int(trial_index),
                    "candidate_ids": list(batch["candidate_ids"]),
                    "local_pool_indices": list(batch["local_indices"]),
                    "batch_selection_score_mean": float(np.mean(candidate_scores)),
                    "batch_selection_score_min": float(np.min(candidate_scores)),
                    "batch_selection_score_max": float(np.max(candidate_scores)),
                    **metrics,
                }
                all_rows.append(row)
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(group_name)
    finally:
        if progress is not None:
            progress.close()

    results_path = output_root / "trial_results.jsonl"
    write_jsonl(results_path, all_rows)

    summary_payload = {
        "baseline_query_loss": float(baseline_query_loss),
        "score_mode": score_mode,
        "target_id": target_id,
        "candidate_kind": str(candidate_kind),
        "seq_len": int(seq_len),
        "num_targets": int(len(bundle.items)),
        "groups": {
            name: {
                "pool": {
                    "num_candidates": int(len(groups[name].frame)),
                    "selection_score_mean": float(groups[name].frame["selection_score"].mean()),
                    "selection_score_min": float(groups[name].frame["selection_score"].min()),
                    "selection_score_max": float(groups[name].frame["selection_score"].max()),
                },
                "trials": _summarize_trials([row for row in all_rows if row["group"] == name]),
            }
            for name in GROUP_NAMES
        },
    }
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary_payload)

    return {
        "root": output_root,
        "config": config_path,
        "results": results_path,
        "summary": summary_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-step target-loss sanity check over top/matched-random/bottom candidate minibatches "
            "from a TrackStar attribution run."
        )
    )
    parser.add_argument("--base_ckpt", required=True, help="Checkpoint directory to reset to before each trial")
    parser.add_argument("--attribution_dir", required=True, help="Attribution output directory with row_summary_*.csv")
    parser.add_argument("--data_dir", required=True, help="Training-data directory matching the attribution run")
    parser.add_argument("--step", type=int, required=True, help="Attribution checkpoint step to read from")
    parser.add_argument("--output_dir", default=None, help="Where to write sanity-check artifacts")
    parser.add_argument(
        "--score_mode",
        choices=SUPPORTED_SCORE_MODES,
        default=DEFAULT_SCORE_MODE,
        help="How to rank attribution candidates before building top/random/bottom pools",
    )
    parser.add_argument("--target_id", default=None, help="Required when score_mode=per_query")
    parser.add_argument("--group_size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--candidate_batch_size", type=int, default=DEFAULT_CANDIDATE_BATCH_SIZE)
    parser.add_argument("--num_trials_per_group", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--update_lr", type=float, default=DEFAULT_UPDATE_LR)
    parser.add_argument("--ewok_filter_spec", default=None, help="Defaults to attribution config when available")
    parser.add_argument("--ewok_variant", default=None, help="Defaults to attribution config when available")
    parser.add_argument("--ewok_score_view", default=None, help="Defaults to attribution config when available")
    parser.add_argument("--score_reduction", default=None, help="Defaults to attribution config when available")
    parser.add_argument("--temperature", type=float, default=None, help="Defaults to attribution config when available")
    parser.add_argument("--target_batch_size", type=int, default=DEFAULT_TARGET_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=DEFAULT_DEVICE)
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    attribution_defaults = _load_attribution_defaults(args.attribution_dir)
    resolved_filter_spec = args.ewok_filter_spec
    if resolved_filter_spec is None:
        resolved_filter_spec = attribution_defaults.get("ewok_filter_spec")
    resolved_variant = args.ewok_variant or attribution_defaults.get("ewok_variant", DEFAULT_EWOK_VARIANT)
    resolved_score_view = args.ewok_score_view or attribution_defaults.get("ewok_score_view", DEFAULT_SCORE_VIEW)
    resolved_reduction = args.score_reduction or attribution_defaults.get("score_reduction", DEFAULT_SCORE_REDUCTION)
    resolved_temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(attribution_defaults.get("temperature", DEFAULT_TEMPERATURE))
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path(args.attribution_dir).expanduser().resolve()
            / f"one_step_sanity_step{int(args.step):08d}_{args.score_mode}"
        )

    artifacts = run_one_step_sanity_check(
        base_ckpt=args.base_ckpt,
        attribution_dir=args.attribution_dir,
        data_dir=args.data_dir,
        step=int(args.step),
        output_dir=output_dir,
        score_mode=args.score_mode,
        target_id=args.target_id,
        group_size=int(args.group_size),
        candidate_batch_size=int(args.candidate_batch_size),
        num_trials_per_group=int(args.num_trials_per_group),
        update_lr=float(args.update_lr),
        ewok_filter_spec=resolved_filter_spec,
        ewok_variant=str(resolved_variant),
        ewok_score_view=str(resolved_score_view),
        score_reduction=str(resolved_reduction),
        temperature=float(resolved_temperature),
        target_batch_size=int(args.target_batch_size),
        seed=int(args.seed),
        device=args.device,
        show_progress=not bool(args.no_progress),
    )
    print(f"wrote one-step sanity-check artifacts under {artifacts['root']}")
    print(f"results: {artifacts['results']}")
    print(f"summary: {artifacts['summary']}")
    return 0


__all__ = [
    "CandidateGroup",
    "build_arg_parser",
    "build_candidate_groups",
    "compute_candidate_batch_loss",
    "compute_query_loss_mean",
    "main",
    "measure_one_step_delta",
    "run_one_step_sanity_check",
    "sample_candidate_batch",
]
