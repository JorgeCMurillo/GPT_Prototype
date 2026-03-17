"""Helpers for comparing GPT and Word2Vec on the BabyLM completion metric.

This module keeps the notebook light by centralizing:
- checkpoint resolution and model loading
- per-item BabyLM-completion scoring for GPT
- per-item BabyLM-completion scoring for Word2Vec
- joined dataframe construction
- domain-level correlation summaries
- focused diagnostic plots for failures and token-level GPT behavior
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from evaluation.ewok import per_token_conditional_log_likelihood, per_token_log_likelihood
from evaluation.ewok_data import load_ewok_df
from research.bos_aligned_proto.analysis.trak.checkpoints import build_model_from_checkpoint
from research.w2v_lexical_probe.eval_ewok_word2vec import (
    build_prediction_dataframe as build_word2vec_prediction_dataframe,
)
from research.w2v_lexical_probe.eval_ewok_word2vec import (
    ewok_per_item_records as word2vec_ewok_per_item_records,
)
from research.w2v_lexical_probe.eval_ewok_word2vec import write_prediction_dataframe

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - fallback for minimal envs
    pearsonr = None
    spearmanr = None

try:  # pragma: no cover - import lazily in notebook-light envs
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_BOS_RUNS_DIR = _PROJECT_DIR / "runs" / "research" / "bos_aligned_proto"
_W2V_RUNS_DIR = _PROJECT_DIR / "runs" / "research" / "w2v_lexical_probe"


def _discover_default_run_dir(env_key: str, search_dir: Path) -> Path:
    env_path = os.environ.get(env_key)
    if env_path:
        return Path(env_path).expanduser()

    candidates = sorted(path for path in search_dir.iterdir() if path.is_dir()) if search_dir.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    return search_dir / "<run_name>"


DEFAULT_GPT_RUN_DIR = _discover_default_run_dir("MOONSHOT_GPT_RUN_DIR", _BOS_RUNS_DIR)
DEFAULT_W2V_RUN_DIR = _discover_default_run_dir("MOONSHOT_W2V_RUN_DIR", _W2V_RUNS_DIR)

_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class GPTCheckpointBundle:
    run_dir: Path
    checkpoint_dir: Path
    tokenizer_source: str
    model: torch.nn.Module
    tokenizer: object


def _require_matplotlib():
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting in this notebook/module. "
            "Use an environment with matplotlib installed."
        )
    return plt


def _normalize_score_reduction(score_reduction: str) -> str:
    value = str(score_reduction).strip().lower()
    if value not in {"sum", "mean"}:
        raise ValueError(f"score_reduction must be 'sum' or 'mean', got: {score_reduction!r}")
    return value


def _ensure_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _score_sign(margin: float, eps: float = 0.0) -> str:
    if margin > eps:
        return "win"
    if margin < -eps:
        return "loss"
    return "tie"


def _reduce_token_logps(token_logps: torch.Tensor, score_reduction: str) -> float:
    if token_logps.numel() == 0:
        return 0.0
    if score_reduction == "sum":
        return float(token_logps.sum().item())
    return float(token_logps.mean().item())


def _pair_to_mean(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        left, right = value[0], value[1]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if math.isfinite(float(left)) and math.isfinite(float(right)):
                return 0.5 * (float(left) + float(right))
    return None


def _resolve_final_checkpoint_dir(run_or_checkpoint_dir: str | Path) -> Path:
    path = _ensure_path(run_or_checkpoint_dir)
    if path.name.startswith("ckpt_"):
        return path

    candidates = sorted(path.glob("ckpt_final_step*"))
    if candidates:
        return candidates[-1]

    periodic = sorted(path.glob("ckpt_periodic_step*"))
    if periodic:
        return periodic[-1]

    raise FileNotFoundError(f"Could not find a checkpoint under {path}")


def _extract_step_from_checkpoint_name(checkpoint_dir: Path) -> int | None:
    name = checkpoint_dir.name
    if "step" not in name:
        return None
    try:
        return int(name.split("step")[-1])
    except Exception:
        return None


def _resolve_tokenizer_source(checkpoint_dir: Path) -> str:
    if all((checkpoint_dir / filename).exists() for filename in _TOKENIZER_FILENAMES):
        return str(checkpoint_dir)

    step = _extract_step_from_checkpoint_name(checkpoint_dir)
    parent = checkpoint_dir.parent
    if step is not None:
        sibling = parent / f"ckpt_periodic_step{step:07d}"
        if sibling.is_dir() and all((sibling / filename).exists() for filename in _TOKENIZER_FILENAMES):
            return str(sibling)

    periodic = sorted(parent.glob("ckpt_periodic_step*"))
    if periodic:
        latest = periodic[-1]
        if all((latest / filename).exists() for filename in _TOKENIZER_FILENAMES):
            return str(latest)

    return "gpt2"


def _candidate_checkpoint_fallbacks(checkpoint_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    step = _extract_step_from_checkpoint_name(checkpoint_dir)
    parent = checkpoint_dir.parent

    if checkpoint_dir.name.startswith("ckpt_final_") and step is not None:
        sibling = parent / f"ckpt_periodic_step{step:07d}"
        if sibling.is_dir():
            candidates.append(sibling)

    for periodic in reversed(sorted(parent.glob("ckpt_periodic_step*"))):
        if periodic == checkpoint_dir or periodic in candidates:
            continue
        candidates.append(periodic)
    return candidates


def _load_gpt_model_with_fallbacks(checkpoint_dir: Path, device: str) -> tuple[torch.nn.Module, Path]:
    errors: list[tuple[Path, Exception]] = []
    for candidate in [checkpoint_dir, *_candidate_checkpoint_fallbacks(checkpoint_dir)]:
        try:
            model = build_model_from_checkpoint(candidate, device=device)
            return model, candidate
        except Exception as exc:  # pragma: no cover - exercised in notebook/runtime
            errors.append((candidate, exc))

    attempted = ", ".join(str(path) for path, _exc in errors)
    last_error = errors[-1][1] if errors else RuntimeError("No checkpoint candidates attempted")
    raise RuntimeError(
        f"Failed to load GPT checkpoint. Attempted: {attempted}"
    ) from last_error


def load_gpt_checkpoint_bundle(
    run_or_checkpoint_dir: str | Path = DEFAULT_GPT_RUN_DIR,
    *,
    device: str | None = None,
) -> GPTCheckpointBundle:
    requested_checkpoint_dir = _resolve_final_checkpoint_dir(run_or_checkpoint_dir)
    run_dir = (
        requested_checkpoint_dir.parent
        if requested_checkpoint_dir.name.startswith("ckpt_")
        else requested_checkpoint_dir
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint_dir = _load_gpt_model_with_fallbacks(requested_checkpoint_dir, device=device)
    if checkpoint_dir != requested_checkpoint_dir:
        warnings.warn(
            f"Falling back from {requested_checkpoint_dir.name} to {checkpoint_dir.name} "
            "because the requested GPT checkpoint could not be loaded.",
            stacklevel=2,
        )
    tokenizer_source = _resolve_tokenizer_source(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return GPTCheckpointBundle(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        tokenizer_source=str(tokenizer_source),
        model=model,
        tokenizer=tokenizer,
    )


def load_ewok_dataframe(ewok_variant: str = "fast") -> pd.DataFrame:
    ewok_df, _source = load_ewok_df(ewok_variant)
    ewok_df = ewok_df.convert_dtypes()
    ewok_df = ewok_df.reset_index().rename(columns={"index": "row_index"})
    ewok_df["domain"] = ewok_df["Domain"]
    return ewok_df


def _gpt_item_record(
    *,
    domain: str,
    row_index: int,
    score_reduction: str,
    s11: float,
    s12: float,
    s22: float,
    s21: float,
) -> dict:
    m1 = s11 - s12
    m2 = s22 - s21
    m = 0.5 * (m1 + m2)
    return {
        "domain": domain,
        "row_index": int(row_index),
        "score_reduction": score_reduction,
        "S11_logp_T1_given_C1": float(s11),
        "S12_logp_T2_given_C1": float(s12),
        "S22_logp_T2_given_C2": float(s22),
        "S21_logp_T1_given_C2": float(s21),
        "margin_official_m1": float(m1),
        "margin_symmetric_m2": float(m2),
        "margin_combined": float(m),
        "gpt_margin_official_m1": float(m1),
        "gpt_margin_symmetric_m2": float(m2),
        "gpt_margin_combined": float(m),
        "gpt_correct_official": float(1.0 if m1 > 0.0 else 0.0),
        "gpt_correct_symmetric": float(1.0 if m2 > 0.0 else 0.0),
        "gpt_correct_combined": float(1.0 if m > 0.0 else 0.0),
        "gpt_completion_sign_official": _score_sign(m1),
        "gpt_completion_sign_symmetric": _score_sign(m2),
        "gpt_completion_sign_combined": _score_sign(m),
    }


def score_gpt_babylm_items(
    model,
    tokenizer,
    *,
    ewok_df: pd.DataFrame | None = None,
    ewok_variant: str = "fast",
    batch_size: int = 8,
    score_reduction: str = "mean",
) -> pd.DataFrame:
    score_reduction = _normalize_score_reduction(score_reduction)
    if ewok_df is None:
        ewok_df = load_ewok_dataframe(ewok_variant)

    records: list[dict] = []
    for domain in ewok_df["domain"].unique():
        df = ewok_df[ewok_df["domain"] == domain].reset_index(drop=True)
        ctx1 = df["Context1"].astype(str).tolist()
        tgt1 = df["Target1"].astype(str).tolist()
        ctx2 = df["Context2"].astype(str).tolist()
        tgt2 = df["Target2"].astype(str).tolist()

        r11 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt1, batch_size=batch_size)
        r12 = per_token_conditional_log_likelihood(model, tokenizer, ctx1, tgt2, batch_size=batch_size)
        r22 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt2, batch_size=batch_size)
        r21 = per_token_conditional_log_likelihood(model, tokenizer, ctx2, tgt1, batch_size=batch_size)

        for i in range(len(df)):
            records.append(
                _gpt_item_record(
                    domain=str(domain),
                    row_index=int(df.loc[i, "row_index"]),
                    score_reduction=score_reduction,
                    s11=_reduce_token_logps(r11[i], score_reduction),
                    s12=_reduce_token_logps(r12[i], score_reduction),
                    s22=_reduce_token_logps(r22[i], score_reduction),
                    s21=_reduce_token_logps(r21[i], score_reduction),
                )
            )

    item_df = pd.DataFrame.from_records(records)
    return ewok_df.merge(item_df, on=["row_index", "domain"], how="left", validate="one_to_one")


def _load_cached_gpt_records(
    ewok_items_path: Path,
    *,
    target_step: int | None = None,
    score_reduction: str = "mean",
) -> list[dict]:
    records_by_key: dict[tuple[int, str], dict] = {}
    with ewok_items_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            if payload.get("score_reduction") != score_reduction:
                continue
            if target_step is not None and payload.get("step") != target_step:
                continue
            row_index = payload.get("row_index")
            domain = payload.get("domain")
            if not isinstance(row_index, int) or not isinstance(domain, str):
                continue
            records_by_key[(row_index, domain)] = payload
    return list(records_by_key.values())


def _final_gpt_step_from_step_metrics(run_dir: Path) -> int | None:
    metrics_path = run_dir / "step_metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return None
    steps = [
        int(rec["step"])
        for rec in payload
        if isinstance(rec, dict) and isinstance(rec.get("step"), int) and isinstance(rec.get("eval_full_mean"), dict)
    ]
    return max(steps) if steps else None


def load_cached_gpt_babylm_items(
    run_dir: str | Path = DEFAULT_GPT_RUN_DIR,
    *,
    ewok_df: pd.DataFrame | None = None,
    ewok_variant: str = "fast",
    score_reduction: str = "mean",
) -> pd.DataFrame:
    run_dir = _ensure_path(run_dir)
    if ewok_df is None:
        ewok_df = load_ewok_dataframe(ewok_variant)

    items_path = run_dir / "ewok_items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(f"Cached GPT ewok_items.jsonl not found at {items_path}")

    final_step = _final_gpt_step_from_step_metrics(run_dir)
    records = _load_cached_gpt_records(items_path, target_step=final_step, score_reduction=score_reduction)
    if not records:
        raise ValueError(
            f"No cached GPT EWoK records found for score_reduction={score_reduction!r} "
            f"and step={final_step!r} in {items_path}"
        )

    item_df = pd.DataFrame.from_records(records)
    item_df["gpt_margin_official_m1"] = item_df["margin_official_m1"].astype(float)
    item_df["gpt_margin_symmetric_m2"] = item_df["margin_symmetric_m2"].astype(float)
    item_df["gpt_margin_combined"] = item_df["margin_combined"].astype(float)
    item_df["gpt_correct_official"] = item_df["correct_official"].astype(float)
    item_df["gpt_correct_symmetric"] = item_df["correct_symmetric"].astype(float)
    item_df["gpt_correct_combined"] = item_df["correct_combined"].astype(float)
    item_df["gpt_completion_sign_official"] = item_df["gpt_margin_official_m1"].map(_score_sign)
    item_df["gpt_completion_sign_symmetric"] = item_df["gpt_margin_symmetric_m2"].map(_score_sign)
    item_df["gpt_completion_sign_combined"] = item_df["gpt_margin_combined"].map(_score_sign)
    return ewok_df.merge(item_df, on=["row_index", "domain"], how="left", validate="one_to_one")


def load_or_compute_gpt_babylm_items(
    run_dir: str | Path = DEFAULT_GPT_RUN_DIR,
    *,
    ewok_variant: str = "fast",
    score_reduction: str = "mean",
    batch_size: int = 8,
    device: str | None = None,
    prefer_cached: bool = True,
) -> pd.DataFrame:
    ewok_df = load_ewok_dataframe(ewok_variant)
    if prefer_cached and ewok_variant == "fast":
        try:
            return load_cached_gpt_babylm_items(
                run_dir,
                ewok_df=ewok_df,
                ewok_variant=ewok_variant,
                score_reduction=score_reduction,
            )
        except Exception:
            pass

    bundle = load_gpt_checkpoint_bundle(run_dir, device=device)
    return score_gpt_babylm_items(
        bundle.model,
        bundle.tokenizer,
        ewok_df=ewok_df,
        batch_size=batch_size,
        score_reduction=score_reduction,
    )


def load_or_compute_word2vec_babylm_items(
    run_dir: str | Path = DEFAULT_W2V_RUN_DIR,
    *,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
    filter_agent_names: bool = True,
    cache_if_missing: bool = True,
) -> pd.DataFrame:
    run_dir = _ensure_path(run_dir)
    if ewok_variant == "fast" and ewok_text_preprocessing == "probe":
        cached_path = run_dir / "ewok_word2vec_predictions.csv"
        if cached_path.exists():
            df = pd.read_csv(cached_path)
            df["domain"] = df["domain"].astype(str)
            return df

    df = build_word2vec_prediction_dataframe(
        run_dir,
        filter_agent_names=filter_agent_names,
        ewok_variant=ewok_variant,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    df["word2vec_completion_sign_official"] = df["word2vec_m1"].map(_score_sign)
    df["word2vec_completion_sign_symmetric"] = df["word2vec_m2"].map(_score_sign)
    df["word2vec_completion_sign_combined"] = df["word2vec_margin_combined"].map(_score_sign)

    if cache_if_missing and ewok_variant == "fast" and ewok_text_preprocessing == "probe":
        write_prediction_dataframe(run_dir, df)
    return df


def _standardize_gpt_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "row_index",
        "domain",
        "Context1",
        "Context2",
        "Target1",
        "Target2",
        "ContextType",
        "ContextDiff",
        "TargetDiff",
        "S11_logp_T1_given_C1",
        "S12_logp_T2_given_C1",
        "S22_logp_T2_given_C2",
        "S21_logp_T1_given_C2",
        "gpt_margin_official_m1",
        "gpt_margin_symmetric_m2",
        "gpt_margin_combined",
        "gpt_correct_official",
        "gpt_correct_symmetric",
        "gpt_correct_combined",
        "gpt_completion_sign_official",
        "gpt_completion_sign_symmetric",
        "gpt_completion_sign_combined",
    ]
    return df[keep].copy()


def _standardize_word2vec_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "row_index",
        "domain",
        "S11_cos_C1_T1",
        "S12_cos_C1_T2",
        "S22_cos_C2_T2",
        "S21_cos_C2_T1",
        "word2vec_m1",
        "word2vec_m2",
        "word2vec_margin_combined",
        "babylm_completion_choice_correct_official",
        "babylm_completion_choice_correct_symmetric",
        "babylm_completion_choice_correct_combined",
        "word2vec_completion_sign_official",
        "word2vec_completion_sign_symmetric",
        "word2vec_completion_sign_combined",
        "empty_context1_vector",
        "empty_target1_vector",
        "empty_context2_vector",
        "empty_target2_vector",
        "context1_in_vocab_tokens",
        "target1_in_vocab_tokens",
        "context2_in_vocab_tokens",
        "target2_in_vocab_tokens",
    ]
    out = df[keep].copy()
    out = out.rename(
        columns={
            "babylm_completion_choice_correct_official": "word2vec_correct_official",
            "babylm_completion_choice_correct_symmetric": "word2vec_correct_symmetric",
            "babylm_completion_choice_correct_combined": "word2vec_correct_combined",
        }
    )
    return out


def build_joint_babylm_completion_dataframe(
    *,
    gpt_df: pd.DataFrame,
    word2vec_df: pd.DataFrame,
) -> pd.DataFrame:
    base_gpt = _standardize_gpt_columns(gpt_df)
    base_w2v = _standardize_word2vec_columns(word2vec_df)
    merged = base_gpt.merge(base_w2v, on=["row_index", "domain"], how="inner", validate="one_to_one")
    merged["gpt_minus_word2vec_margin_combined"] = (
        merged["gpt_margin_combined"] - merged["word2vec_margin_combined"]
    )
    return merged


def prepare_joint_babylm_completion_dataframe(
    *,
    gpt_run_dir: str | Path = DEFAULT_GPT_RUN_DIR,
    word2vec_run_dir: str | Path = DEFAULT_W2V_RUN_DIR,
    ewok_variant: str = "fast",
    score_reduction: str = "mean",
    ewok_text_preprocessing: str = "probe",
    filter_agent_names: bool = True,
    gpt_batch_size: int = 8,
    gpt_device: str | None = None,
    prefer_cached_gpt: bool = True,
    cache_word2vec_if_missing: bool = True,
) -> pd.DataFrame:
    gpt_df = load_or_compute_gpt_babylm_items(
        gpt_run_dir,
        ewok_variant=ewok_variant,
        score_reduction=score_reduction,
        batch_size=gpt_batch_size,
        device=gpt_device,
        prefer_cached=prefer_cached_gpt,
    )
    word2vec_df = load_or_compute_word2vec_babylm_items(
        word2vec_run_dir,
        ewok_variant=ewok_variant,
        ewok_text_preprocessing=ewok_text_preprocessing,
        filter_agent_names=filter_agent_names,
        cache_if_missing=cache_word2vec_if_missing,
    )
    return build_joint_babylm_completion_dataframe(gpt_df=gpt_df, word2vec_df=word2vec_df)


def _safe_corr(function, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if function is None:
        return float("nan"), float("nan")
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")
    try:
        stat, p_value = function(x, y)
        return float(stat), float(p_value)
    except Exception:
        return float("nan"), float("nan")


def _axis_limits(values: np.ndarray, *, pad_frac: float = 0.08) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    vmin = float(arr.min())
    vmax = float(arr.max())
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * pad_frac, 1e-3)
        return vmin - pad, vmax + pad
    pad = max((vmax - vmin) * pad_frac, 1e-3)
    return vmin - pad, vmax + pad


def summarize_domain_correlations(
    joint_df: pd.DataFrame,
    *,
    score_x: str = "word2vec_margin_combined",
    score_y: str = "gpt_margin_combined",
) -> pd.DataFrame:
    rows: list[dict] = []
    for domain, domain_df in joint_df.groupby("domain", sort=True):
        x = domain_df[score_x].to_numpy(dtype=np.float64)
        y = domain_df[score_y].to_numpy(dtype=np.float64)
        pearson_r, pearson_p = _safe_corr(pearsonr, x, y)
        spearman_rho, spearman_p = _safe_corr(spearmanr, x, y)
        rows.append(
            {
                "domain": domain,
                "n_items": int(len(domain_df)),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "gpt_mean_margin": float(domain_df["gpt_margin_combined"].mean()),
                "word2vec_mean_margin": float(domain_df["word2vec_margin_combined"].mean()),
                "gpt_acc_combined": float((domain_df["gpt_margin_combined"] > 0.0).mean()),
                "word2vec_acc_combined": float((domain_df["word2vec_margin_combined"] > 0.0).mean()),
                "strict_agreement_rate": float(
                    (
                        domain_df["gpt_completion_sign_combined"]
                        == domain_df["word2vec_completion_sign_combined"]
                    ).mean()
                ),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)
    return summary_df


def _with_significance_columns(
    df: pd.DataFrame,
    *,
    pearson_col: str = "pearson_p",
    spearman_col: str = "spearman_p",
    thresholds: tuple[float, ...] = (0.05, 0.01, 0.005),
) -> pd.DataFrame:
    out = df.copy()
    for prefix, p_col in (("pearson", pearson_col), ("spearman", spearman_col)):
        for threshold in thresholds:
            label = f"{prefix}_p_le_{threshold:g}"
            out[label] = out[p_col] <= float(threshold)
    return out


def build_domain_correlation_significance_table(
    summary_df: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.05, 0.01, 0.005),
) -> pd.DataFrame:
    out = _with_significance_columns(summary_df, thresholds=thresholds)
    ordered_columns = [
        "domain",
        "pearson_r",
        "pearson_p",
        *[f"pearson_p_le_{threshold:g}" for threshold in thresholds],
        "spearman_rho",
        "spearman_p",
        *[f"spearman_p_le_{threshold:g}" for threshold in thresholds],
    ]
    available_columns = [column for column in ordered_columns if column in out.columns]
    return out[available_columns].sort_values("domain").reset_index(drop=True)


def build_metadata_group_correlation_table(
    joint_df: pd.DataFrame,
    *,
    group_specs: tuple[str | tuple[str, ...], ...] = (
        "ContextType",
        "ContextDiff",
        "TargetDiff",
        ("ContextType", "ContextDiff"),
        ("ContextType", "TargetDiff"),
        ("ContextDiff", "TargetDiff"),
    ),
    score_x: str = "word2vec_margin_combined",
    score_y: str = "gpt_margin_combined",
    thresholds: tuple[float, ...] = (0.05, 0.01, 0.005),
) -> pd.DataFrame:
    rows: list[dict] = []

    for spec in group_specs:
        group_cols = (spec,) if isinstance(spec, str) else tuple(spec)
        missing = [col for col in group_cols if col not in joint_df.columns]
        if missing:
            continue
        grouped = (
            joint_df.groupby(group_cols[0], dropna=False, sort=True)
            if len(group_cols) == 1
            else joint_df.groupby(list(group_cols), dropna=False, sort=True)
        )
        for key, group_df in grouped:
            key_tuple = (key,) if len(group_cols) == 1 else tuple(key)
            x = group_df[score_x].to_numpy(dtype=np.float64)
            y = group_df[score_y].to_numpy(dtype=np.float64)
            pearson_r, pearson_p = _safe_corr(pearsonr, x, y)
            spearman_rho, spearman_p = _safe_corr(spearmanr, x, y)
            value_map = {
                column: ("<NA>" if pd.isna(value) else str(value))
                for column, value in zip(group_cols, key_tuple)
            }
            rows.append(
                {
                    "group_by": " x ".join(group_cols),
                    "group_value": " | ".join(f"{column}={value_map[column]}" for column in group_cols),
                    "n_items": int(len(group_df)),
                    **{column: value_map[column] for column in group_cols},
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p": spearman_p,
                    "gpt_mean_margin": float(group_df["gpt_margin_combined"].mean()),
                    "word2vec_mean_margin": float(group_df["word2vec_margin_combined"].mean()),
                    "gpt_acc_combined": float((group_df["gpt_margin_combined"] > 0.0).mean()),
                    "word2vec_acc_combined": float((group_df["word2vec_margin_combined"] > 0.0).mean()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _with_significance_columns(out, thresholds=thresholds)
    ordered_columns = [
        "group_by",
        "group_value",
        "n_items",
        "pearson_r",
        "pearson_p",
        *[f"pearson_p_le_{threshold:g}" for threshold in thresholds],
        "spearman_rho",
        "spearman_p",
        *[f"spearman_p_le_{threshold:g}" for threshold in thresholds],
        "gpt_mean_margin",
        "word2vec_mean_margin",
        "gpt_acc_combined",
        "word2vec_acc_combined",
    ]
    return out[ordered_columns].sort_values(
        by=["group_by", "spearman_rho", "pearson_r"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def plot_domain_correlation_grid(
    joint_df: pd.DataFrame,
    *,
    score_x: str = "word2vec_margin_combined",
    score_y: str = "gpt_margin_combined",
    shared_limits: bool = False,
    figsize: tuple[int, int] = (18, 14),
) -> tuple[plt.Figure, np.ndarray]:
    _require_matplotlib()
    summary_df = summarize_domain_correlations(joint_df, score_x=score_x, score_y=score_y)
    domains = summary_df["domain"].tolist()
    ncols = 3
    nrows = 4 if len(domains) <= 12 else int(math.ceil(len(domains) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    flat_axes = axes.flatten()
    global_x_limits = _axis_limits(joint_df[score_x].to_numpy(dtype=np.float64))
    global_y_limits = _axis_limits(joint_df[score_y].to_numpy(dtype=np.float64))

    for idx, domain in enumerate(domains):
        ax = flat_axes[idx]
        domain_df = joint_df[joint_df["domain"] == domain]
        row = summary_df[summary_df["domain"] == domain].iloc[0]
        x_vals = domain_df[score_x].to_numpy(dtype=np.float64)
        y_vals = domain_df[score_y].to_numpy(dtype=np.float64)
        x_limits = global_x_limits if shared_limits else _axis_limits(x_vals)
        y_limits = global_y_limits if shared_limits else _axis_limits(y_vals)
        ax.scatter(
            domain_df[score_x],
            domain_df[score_y],
            s=32,
            alpha=0.72,
            color="#2a6f97",
            edgecolors="white",
            linewidths=0.4,
        )
        if x_limits[0] <= 0.0 <= x_limits[1]:
            ax.axvline(0.0, color="#9a9a9a", linestyle=(0, (4, 2)), linewidth=0.9)
        if y_limits[0] <= 0.0 <= y_limits[1]:
            ax.axhline(0.0, color="#9a9a9a", linestyle=(0, (4, 2)), linewidth=0.9)
        diag_min = max(x_limits[0], y_limits[0])
        diag_max = min(x_limits[1], y_limits[1])
        if diag_min < diag_max:
            ax.plot([diag_min, diag_max], [diag_min, diag_max], color="#d62728", linewidth=0.9, alpha=0.6)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_title(
            f"{domain}\n"
            f"r={row['pearson_r']:.2f}, p={row['pearson_p']:.2g} | "
            f"rho={row['spearman_rho']:.2f}, p={row['spearman_p']:.2g}",
            fontsize=9,
        )
        ax.set_xlabel("Word2Vec combined margin", fontsize=9)
        ax.set_ylabel("GPT combined margin", fontsize=9)
        ax.grid(True, alpha=0.2)

    for idx in range(len(domains), len(flat_axes)):
        flat_axes[idx].axis("off")

    title_suffix = " (shared axes)" if shared_limits else " (per-domain axes)"
    fig.suptitle(
        f"BabyLM completion: per-item GPT vs Word2Vec correlation by domain{title_suffix}",
        fontsize=14,
    )
    return fig, axes


def plot_domain_failure_quadrants(
    joint_df: pd.DataFrame,
    domain: str,
    *,
    score_x: str = "word2vec_margin_combined",
    score_y: str = "gpt_margin_combined",
) -> tuple[plt.Figure, plt.Axes]:
    _require_matplotlib()
    domain_df = joint_df[joint_df["domain"] == domain].copy()
    if domain_df.empty:
        raise ValueError(f"No rows found for domain={domain!r}")

    def bucket(row: pd.Series) -> str:
        wx = row["word2vec_completion_sign_combined"]
        gy = row["gpt_completion_sign_combined"]
        if wx == "win" and gy == "win":
            return "both win"
        if wx != "win" and gy != "win":
            return "both non-win"
        if wx == "win" and gy != "win":
            return "w2v win / GPT non-win"
        return "GPT win / w2v non-win"

    domain_df["agreement_bucket"] = domain_df.apply(bucket, axis=1)
    palette = {
        "both win": "#1b9e77",
        "both non-win": "#d95f02",
        "w2v win / GPT non-win": "#7570b3",
        "GPT win / w2v non-win": "#e7298a",
    }

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    for bucket_name, bucket_df in domain_df.groupby("agreement_bucket"):
        ax.scatter(
            bucket_df[score_x],
            bucket_df[score_y],
            s=42,
            alpha=0.75,
            color=palette[bucket_name],
            label=f"{bucket_name} (n={len(bucket_df)})",
            edgecolors="white",
            linewidths=0.4,
        )

    x_limits = _axis_limits(domain_df[score_x].to_numpy(dtype=np.float64))
    y_limits = _axis_limits(domain_df[score_y].to_numpy(dtype=np.float64))
    if y_limits[0] <= 0.0 <= y_limits[1]:
        ax.axhline(0.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.0)
    if x_limits[0] <= 0.0 <= x_limits[1]:
        ax.axvline(0.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("Word2Vec combined margin", fontsize=10)
    ax.set_ylabel("GPT combined margin", fontsize=10)
    ax.set_title(f"Failure overlap quadrants: {domain}", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    return fig, ax


def plot_ranked_domain_margins(
    joint_df: pd.DataFrame,
    domain: str,
    *,
    sort_by: str = "word2vec_margin_combined",
) -> tuple[plt.Figure, plt.Axes]:
    _require_matplotlib()
    domain_df = joint_df[joint_df["domain"] == domain].copy()
    if domain_df.empty:
        raise ValueError(f"No rows found for domain={domain!r}")
    domain_df = domain_df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    x = np.arange(len(domain_df))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(x, domain_df["word2vec_margin_combined"], marker="o", linewidth=1.8, markersize=3.8, label="Word2Vec")
    ax.plot(x, domain_df["gpt_margin_combined"], marker="s", linewidth=1.8, markersize=3.6, label="GPT-2 Medium")
    ax.axhline(0.0, color="#d62728", linestyle=(0, (8, 2, 2, 2)), linewidth=1.0, label="decision boundary")
    ax.set_title(f"Ranked combined margins in {domain} (sorted by {sort_by})", fontsize=12)
    ax.set_xlabel("Item rank", fontsize=10)
    ax.set_ylabel("Combined BabyLM margin", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig, ax


def find_domain_failure_cases(
    joint_df: pd.DataFrame,
    domain: str,
    *,
    require_word2vec_nonwin: bool = True,
) -> pd.DataFrame:
    domain_df = joint_df[joint_df["domain"] == domain].copy()
    if require_word2vec_nonwin:
        domain_df = domain_df[domain_df["word2vec_completion_sign_combined"] != "win"]
    domain_df = domain_df.sort_values(
        by=["word2vec_margin_combined", "gpt_margin_combined"],
        ascending=[True, True],
    ).reset_index(drop=True)
    return domain_df[
        [
            "row_index",
            "domain",
            "Context1",
            "Target1",
            "Context2",
            "Target2",
            "word2vec_margin_combined",
            "gpt_margin_combined",
            "word2vec_completion_sign_combined",
            "gpt_completion_sign_combined",
        ]
    ]


def select_interesting_item(
    joint_df: pd.DataFrame,
    *,
    domain: str = "material-dynamics",
    strategy: str = "word2vec_nonwin_then_gpt_nonwin",
) -> pd.Series:
    return select_interesting_items(joint_df, domain=domain, strategy=strategy, top_k=1).iloc[0]


def select_interesting_items(
    joint_df: pd.DataFrame,
    *,
    domain: str = "material-dynamics",
    strategy: str = "word2vec_nonwin_then_gpt_nonwin",
    top_k: int = 5,
) -> pd.DataFrame:
    domain_df = joint_df[joint_df["domain"] == domain].copy()
    if domain_df.empty:
        raise ValueError(f"No rows found for domain={domain!r}")

    if strategy == "word2vec_nonwin_then_gpt_nonwin":
        filtered = domain_df[
            (domain_df["word2vec_completion_sign_combined"] != "win")
            & (domain_df["gpt_completion_sign_combined"] != "win")
        ].copy()
        if filtered.empty:
            filtered = domain_df[domain_df["word2vec_completion_sign_combined"] != "win"].copy()
        if filtered.empty:
            filtered = domain_df.copy()
        filtered = filtered.sort_values("word2vec_margin_combined", ascending=True)
        return filtered.head(top_k).reset_index(drop=True)

    if strategy == "largest_disagreement":
        domain_df["abs_gap"] = np.abs(
            domain_df["gpt_margin_combined"] - domain_df["word2vec_margin_combined"]
        )
        return domain_df.sort_values("abs_gap", ascending=False).head(top_k).reset_index(drop=True)

    raise ValueError(f"Unknown selection strategy: {strategy}")


def _token_labels_for_target(tokenizer, context: str, target: str) -> list[str]:
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_text = context + " " + target
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    target_ids = full_ids[len(context_ids) :]
    return tokenizer.convert_ids_to_tokens(target_ids)


def _shorten_context(text: str, *, width: int = 88, max_lines: int = 2) -> str:
    wrapped = textwrap.wrap(str(text).strip(), width=width)
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    kept = wrapped[:max_lines]
    kept[-1] = kept[-1].rstrip() + " ..."
    return "\n".join(kept)


def _paired_token_ticklabels(
    left_tokens: list[str],
    right_tokens: list[str],
    *,
    left_label: str,
    right_label: str,
) -> tuple[np.ndarray, list[str]]:
    max_len = max(len(left_tokens), len(right_tokens))
    positions = np.arange(1, max_len + 1)
    labels: list[str] = []
    for idx in range(max_len):
        left = left_tokens[idx] if idx < len(left_tokens) else "-"
        right = right_tokens[idx] if idx < len(right_tokens) else "-"
        labels.append(f"{left_label}: {left}\n{right_label}: {right}")
    return positions, labels


def gpt_target_token_logprob_details(
    model,
    tokenizer,
    *,
    context: str,
    target: str,
    label: str,
) -> pd.DataFrame:
    full_text = context + " " + target
    token_logps, attention_mask = per_token_log_likelihood(model, tokenizer, [full_text])
    valid_length = int(attention_mask[0].sum().item())
    context_len = len(tokenizer.encode(context, add_special_tokens=False))
    target_logps = token_logps[0, context_len:valid_length].detach().cpu().numpy()
    token_labels = _token_labels_for_target(tokenizer, context, target)
    n = min(len(token_labels), len(target_logps))
    return pd.DataFrame(
        {
            "label": label,
            "token_position": np.arange(1, n + 1),
            "token": token_labels[:n],
            "token_logprob": target_logps[:n],
            "cumulative_logprob": np.cumsum(target_logps[:n]),
        }
    )


def plot_gpt_token_logprob_panels(
    model,
    tokenizer,
    item_row: pd.Series | dict,
    *,
    annotate_tokens: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    _require_matplotlib()
    row = dict(item_row)
    pairs = (
        ("C_1", "T_1", row["Context1"], row["Target1"], "tab:blue"),
        ("C_1", "T_2", row["Context1"], row["Target2"], "tab:orange"),
        ("C_2", "T_2", row["Context2"], row["Target2"], "tab:green"),
        ("C_2", "T_1", row["Context2"], row["Target1"], "tab:red"),
    )
    details = [
        gpt_target_token_logprob_details(
            model,
            tokenizer,
            context=context,
            target=target,
            label=f"{target_label} under {context_label}",
        )
        for context_label, target_label, context, target, _color in pairs
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    panel_specs = (
        (
            axes[0],
            ((details[0], pairs[0][4]), (details[1], pairs[1][4])),
            f"Per-token log probs under C_1\n{_shorten_context(row['Context1'])}",
            ("T_1", "T_2"),
        ),
        (
            axes[1],
            ((details[2], pairs[2][4]), (details[3], pairs[3][4])),
            f"Per-token log probs under C_2\n{_shorten_context(row['Context2'])}",
            ("T_2", "T_1"),
        ),
    )

    for ax, series_bundle, title, tick_pair_labels in panel_specs:
        first_df, first_color = series_bundle[0]
        second_df, second_color = series_bundle[1]
        tick_positions, tick_labels = _paired_token_ticklabels(
            first_df["token"].tolist(),
            second_df["token"].tolist(),
            left_label=tick_pair_labels[0],
            right_label=tick_pair_labels[1],
        )
        for detail_df, color in series_bundle:
            ax.plot(
                detail_df["token_position"],
                detail_df["token_logprob"],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=color,
                label=detail_df["label"].iloc[0],
            )
            if annotate_tokens:
                for _, point in detail_df.iterrows():
                    ax.annotate(
                        point["token"],
                        (point["token_position"], point["token_logprob"]),
                        textcoords="offset points",
                        xytext=(0, 6 if color in {"tab:blue", "tab:green"} else -12),
                        ha="center",
                        fontsize=8,
                        color=color,
                    )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_xlabel("Target tokens", fontsize=10)
        ax.set_ylabel("Per-token log prob", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"GPT token-level completion evidence for row {row['row_index']} ({row['domain']})",
        fontsize=14,
    )
    return fig, axes


__all__ = [
    "DEFAULT_GPT_RUN_DIR",
    "DEFAULT_W2V_RUN_DIR",
    "GPTCheckpointBundle",
    "build_domain_correlation_significance_table",
    "build_metadata_group_correlation_table",
    "build_joint_babylm_completion_dataframe",
    "find_domain_failure_cases",
    "gpt_target_token_logprob_details",
    "load_cached_gpt_babylm_items",
    "load_ewok_dataframe",
    "load_gpt_checkpoint_bundle",
    "load_or_compute_gpt_babylm_items",
    "load_or_compute_word2vec_babylm_items",
    "plot_domain_correlation_grid",
    "plot_domain_failure_quadrants",
    "plot_gpt_token_logprob_panels",
    "plot_ranked_domain_margins",
    "prepare_joint_babylm_completion_dataframe",
    "score_gpt_babylm_items",
    "select_interesting_item",
    "select_interesting_items",
    "summarize_domain_correlations",
]
