"""Evaluate a trained Word2Vec lexical probe on EWoK.

How it works:
- Load a saved Word2Vec run from `--run_dir`.
- Load the EWoK benchmark rows from the shared EWoK loader.
- Tokenize each context and target with the same normalization used during
  Word2Vec training.
- Mean-pool token vectors for `Context1`, `Context2`, `Target1`, and `Target2`.
- Compute cosine similarities:
  - `S11 = cos(C1, T1)`
  - `S12 = cos(C1, T2)`
  - `S22 = cos(C2, T2)`
  - `S21 = cos(C2, T1)`
- Evaluate two scoring conventions from those same four scores:
  - BabyLM completion choice:
    - `m1 = S11 - S12`
    - `m2 = S22 - S21`
    - per-target accuracies use `0 / 0.5 / 1` for `< / == / >`
  - EWoK paper context sensitivity:
    - per-target accuracies use `0 / 0.5 / 1` for `< / == / >`
    - `k1 = compare(S11, S21)`
    - `k2 = compare(S22, S12)`
    - combined paper context sensitivity is `1` iff `k1 == k2`
- Aggregate domain metrics, and optionally write per-item artifacts for later
  correlation and error analysis.

Main helpers:
- `ewok_per_item_records(...)`
  Computes per-item cosine scores and margins.
- `build_prediction_dataframe(...)`
  Returns the original EWoK rows augmented with Word2Vec prediction columns
  such as `word2vec_m1` and `word2vec_m2`.
- `evaluate(...)`
  Returns the summary metrics in a shape similar to `evaluation/ewok.py`.

Example runs:
```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name>
```

```bash
python -m research.w2v_lexical_probe.eval_ewok_word2vec \
  --run_dir runs/research/w2v_lexical_probe/<run_name> \
  --write_per_item
```

The second form also writes:
- `ewok_items.jsonl`
- `ewok_word2vec_predictions.csv`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.ewok_data import load_ewok_df

from .model import LoadedWord2VecRun, load_word2vec_run
from .text import PaperEvalWordTokenizer, TextNormalizationConfig, WordTokenizer

BABYLM_COMPLETION_CHOICE = "babylm_completion_choice"
EWOK_CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
_VALID_EWOK_VARIANTS = ("fast", "full")
_VALID_EWOK_TEXT_PREPROCESSING = ("probe", "paper")
EWOK_AGENT_NAME_TOKENS = frozenset(
    {
        "john",
        "david",
        "michael",
        "robert",
        "daniel",
        "mohammed",
        "ahmed",
        "ali",
        "abdul",
        "jose",
        "ana",
        "jean",
        "nushi",
        "ying",
        "hong",
        "andrea",
        "francis",
        "jesse",
        "yun",
        "jin",
        "li",
        "chao",
        "carmen",
        "wei",
        "yan",
        "alex",
        "maria",
        "fatima",
        "mary",
        "elena",
    }
)


def _build_ewok_agent_name_drop_tokens() -> set[str]:
    """Match the paper's filler-agent filtering for Word2Vec evaluation.

    The paper notebook lowercases, strips punctuation, splits on whitespace, and
    then drops tokens that are either an exact filler-agent name or end in
    ``s`` with a filler-agent stem (e.g. ``alexs`` after punctuation removal
    from ``Alex's``). Our tokenizer preserves apostrophes, so we expand the drop
    set to include both ``name + s`` and ``name + 's'`` variants.
    """

    drop_tokens: set[str] = set(EWOK_AGENT_NAME_TOKENS)
    for name in EWOK_AGENT_NAME_TOKENS:
        drop_tokens.add(f"{name}s")
        drop_tokens.add(f"{name}'s")
    return drop_tokens


def _cosine_similarity(vec_a, vec_b) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _pooled_vector(run: LoadedWord2VecRun, tokenizer: WordTokenizer, text: str):
    tokens = tokenizer.tokenize(text)
    vector, kept_tokens = run.mean_pool(tokens)
    if vector is None:
        return None, tokens, kept_tokens
    return np.asarray(vector, dtype=np.float32), tokens, kept_tokens


def _half_credit_accuracy_from_margin(margin: float) -> float:
    if margin > 0.0:
        return 1.0
    if margin < 0.0:
        return 0.0
    return 0.5


def _paper_context_sensitivity_combined(acc1: float, acc2: float) -> float:
    return 1.0 if acc1 == acc2 else 0.0


def _normalize_ewok_variant(ewok_variant: str) -> str:
    normalized = str(ewok_variant).strip().lower()
    if normalized not in _VALID_EWOK_VARIANTS:
        valid = ", ".join(_VALID_EWOK_VARIANTS)
        raise ValueError(f"Unknown ewok_variant {ewok_variant!r}; expected one of: {valid}")
    return normalized


def _normalize_ewok_text_preprocessing(ewok_text_preprocessing: str) -> str:
    normalized = str(ewok_text_preprocessing).strip().lower()
    if normalized not in _VALID_EWOK_TEXT_PREPROCESSING:
        valid = ", ".join(_VALID_EWOK_TEXT_PREPROCESSING)
        raise ValueError(
            f"Unknown ewok_text_preprocessing {ewok_text_preprocessing!r}; expected one of: {valid}"
        )
    return normalized


def get_ewok_output_paths(
    run_dir: str | Path,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    variant = _normalize_ewok_variant(ewok_variant)
    preprocessing = _normalize_ewok_text_preprocessing(ewok_text_preprocessing)
    prefix = "ewok" if variant == "fast" else f"ewok_{variant}"
    suffix = "" if preprocessing == "probe" else f"_{preprocessing}prep"
    if variant == "fast" and preprocessing == "probe":
        return {
            "metrics": run_dir / "ewok_metrics.json",
            "items": run_dir / "ewok_items.jsonl",
            "prediction_df": run_dir / "ewok_word2vec_predictions.csv",
        }
    return {
        "metrics": run_dir / f"{prefix}{suffix}_metrics.json",
        "items": run_dir / f"{prefix}{suffix}_items.jsonl",
        "prediction_df": run_dir / f"{prefix}{suffix}_word2vec_predictions.csv",
    }


def load_ewok_eval_data(ewok_variant: str = "fast"):
    ewok_df, source = load_ewok_df(ewok_variant)
    ewok_df = ewok_df.convert_dtypes()
    return ewok_df, source


def build_tokenizer_for_run(
    run: LoadedWord2VecRun,
    *,
    filter_agent_names: bool = True,
    ewok_text_preprocessing: str = "probe",
) -> tuple[WordTokenizer, TextNormalizationConfig]:
    preprocessing = _normalize_ewok_text_preprocessing(ewok_text_preprocessing)
    drop_tokens = _build_ewok_agent_name_drop_tokens() if filter_agent_names else None
    if preprocessing == "paper":
        text_config = TextNormalizationConfig(
            lowercase=True,
            strip_html=False,
            replace_urls=False,
            replace_emails=False,
            collapse_whitespace=False,
        )
        tokenizer = PaperEvalWordTokenizer(drop_tokens=drop_tokens)
    else:
        text_config = TextNormalizationConfig(**run.summary.get("normalization", {}))
        tokenizer = WordTokenizer(text_config, drop_tokens=drop_tokens)
    return tokenizer, text_config


def _load_eval_context(
    run_dir: str | Path,
    *,
    filter_agent_names: bool = True,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
):
    ewok_df, source = load_ewok_eval_data(ewok_variant)
    run = load_word2vec_run(run_dir)
    tokenizer, text_config = build_tokenizer_for_run(
        run,
        filter_agent_names=filter_agent_names,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    return ewok_df, source, run, tokenizer, text_config


def _build_item_record(
    domain: str,
    row_index: int,
    margin_eps: float,
    s11: float,
    s12: float,
    s22: float,
    s21: float,
    ctx1_vec,
    tgt1_vec,
    ctx2_vec,
    tgt2_vec,
    ctx1_tokens,
    ctx1_kept,
    tgt1_tokens,
    tgt1_kept,
    ctx2_tokens,
    ctx2_kept,
    tgt2_tokens,
    tgt2_kept,
) -> dict:
    m1 = s11 - s12
    m2 = s22 - s21
    m = 0.5 * (m1 + m2)
    k1 = s11 - s21
    k2 = s22 - s12
    babylm_official = _half_credit_accuracy_from_margin(m1)
    babylm_symmetric = _half_credit_accuracy_from_margin(m2)
    babylm_combined = _half_credit_accuracy_from_margin(m)
    ewok_official = _half_credit_accuracy_from_margin(k1)
    ewok_symmetric = _half_credit_accuracy_from_margin(k2)
    ewok_combined = _paper_context_sensitivity_combined(ewok_official, ewok_symmetric)
    return {
        "domain": domain,
        "row_index": row_index,
        "S11_cos_C1_T1": float(s11),
        "S12_cos_C1_T2": float(s12),
        "S22_cos_C2_T2": float(s22),
        "S21_cos_C2_T1": float(s21),
        "margin_official_m1": float(m1),
        "margin_symmetric_m2": float(m2),
        "margin_combined": float(m),
        # Short aliases make downstream correlation analysis less awkward.
        "word2vec_m1": float(m1),
        "word2vec_m2": float(m2),
        "word2vec_margin_combined": float(m),
        "babylm_completion_choice_correct_official": float(babylm_official),
        "babylm_completion_choice_correct_symmetric": float(babylm_symmetric),
        "babylm_completion_choice_correct_combined": float(babylm_combined),
        "correct_official": float(babylm_official),
        "correct_symmetric": float(babylm_symmetric),
        "correct_combined": float(babylm_combined),
        "near_tie_official": bool(abs(m1) < margin_eps),
        "near_tie_symmetric": bool(abs(m2) < margin_eps),
        "near_tie_combined": bool(abs(m) < margin_eps),
        "ewok_context_sensitivity_margin_official_k1": float(k1),
        "ewok_context_sensitivity_margin_symmetric_k2": float(k2),
        "ewok_context_sensitivity_correct_official": float(ewok_official),
        "ewok_context_sensitivity_correct_symmetric": float(ewok_symmetric),
        "ewok_context_sensitivity_correct_combined": float(ewok_combined),
        "empty_context1_vector": bool(ctx1_vec is None),
        "empty_target1_vector": bool(tgt1_vec is None),
        "empty_context2_vector": bool(ctx2_vec is None),
        "empty_target2_vector": bool(tgt2_vec is None),
        "context1_tokens": len(ctx1_tokens),
        "context1_in_vocab_tokens": len(ctx1_kept),
        "target1_tokens": len(tgt1_tokens),
        "target1_in_vocab_tokens": len(tgt1_kept),
        "context2_tokens": len(ctx2_tokens),
        "context2_in_vocab_tokens": len(ctx2_kept),
        "target2_tokens": len(tgt2_tokens),
        "target2_in_vocab_tokens": len(tgt2_kept),
    }


def ewok_per_item_records(
    run_dir: str | Path,
    margin_eps: float = 1e-6,
    filter_agent_names: bool = True,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
):
    """Return per-item Word2Vec EWoK records for both BabyLM and paper conventions."""
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    ewok_df, _source, run, tokenizer, _text_config = _load_eval_context(
        run_dir,
        filter_agent_names=filter_agent_names,
        ewok_variant=ewok_variant,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    return ewok_per_item_records_for_run(
        run,
        ewok_df=ewok_df,
        tokenizer=tokenizer,
        margin_eps=margin_eps,
    )


def ewok_per_item_records_for_run(
    run: LoadedWord2VecRun,
    *,
    ewok_df: pd.DataFrame | None = None,
    tokenizer: WordTokenizer | None = None,
    margin_eps: float = 1e-6,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
):
    """Return per-item Word2Vec EWoK records for an already-loaded run."""
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")
    if ewok_df is None:
        ewok_df, _source = load_ewok_eval_data(ewok_variant)
    if tokenizer is None:
        tokenizer, _text_config = build_tokenizer_for_run(
            run,
            ewok_text_preprocessing=ewok_text_preprocessing,
        )

    records = []

    for domain in ewok_df["Domain"].unique():
        df = ewok_df[ewok_df["Domain"] == domain].reset_index()

        for i in range(len(df)):
            context1 = str(df.loc[i, "Context1"])
            target1 = str(df.loc[i, "Target1"])
            context2 = str(df.loc[i, "Context2"])
            target2 = str(df.loc[i, "Target2"])

            ctx1_vec, ctx1_tokens, ctx1_kept = _pooled_vector(run, tokenizer, context1)
            tgt1_vec, tgt1_tokens, tgt1_kept = _pooled_vector(run, tokenizer, target1)
            ctx2_vec, ctx2_tokens, ctx2_kept = _pooled_vector(run, tokenizer, context2)
            tgt2_vec, tgt2_tokens, tgt2_kept = _pooled_vector(run, tokenizer, target2)

            s11 = _cosine_similarity(ctx1_vec, tgt1_vec) if ctx1_vec is not None and tgt1_vec is not None else 0.0
            s12 = _cosine_similarity(ctx1_vec, tgt2_vec) if ctx1_vec is not None and tgt2_vec is not None else 0.0
            s22 = _cosine_similarity(ctx2_vec, tgt2_vec) if ctx2_vec is not None and tgt2_vec is not None else 0.0
            s21 = _cosine_similarity(ctx2_vec, tgt1_vec) if ctx2_vec is not None and tgt1_vec is not None else 0.0

            records.append(
                _build_item_record(
                    domain=domain,
                    row_index=int(df.loc[i, "index"]),
                    margin_eps=margin_eps,
                    s11=s11,
                    s12=s12,
                    s22=s22,
                    s21=s21,
                    ctx1_vec=ctx1_vec,
                    tgt1_vec=tgt1_vec,
                    ctx2_vec=ctx2_vec,
                    tgt2_vec=tgt2_vec,
                    ctx1_tokens=ctx1_tokens,
                    ctx1_kept=ctx1_kept,
                    tgt1_tokens=tgt1_tokens,
                    tgt1_kept=tgt1_kept,
                    ctx2_tokens=ctx2_tokens,
                    ctx2_kept=ctx2_kept,
                    tgt2_tokens=tgt2_tokens,
                    tgt2_kept=tgt2_kept,
                )
            )

    return records


def _prediction_dataframe_from_records(
    ewok_df: pd.DataFrame,
    records: list[dict],
) -> pd.DataFrame:
    """Attach Word2Vec predictions to the original EWoK rows."""
    base_df = ewok_df.reset_index().rename(columns={"index": "row_index"})
    base_df["domain"] = base_df["Domain"]
    item_df = pd.DataFrame.from_records(records)
    if item_df.empty:
        return base_df
    merged = base_df.merge(item_df, on=["row_index", "domain"], how="left", validate="one_to_one")
    return merged


def build_prediction_dataframe(
    run_dir: str | Path,
    margin_eps: float = 1e-6,
    filter_agent_names: bool = True,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
) -> pd.DataFrame:
    """Return the original EWoK rows augmented with Word2Vec item predictions."""
    ewok_df, _source, _run, _tokenizer, _text_config = _load_eval_context(
        run_dir,
        filter_agent_names=filter_agent_names,
        ewok_variant=ewok_variant,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    records = ewok_per_item_records(
        run_dir,
        margin_eps=margin_eps,
        filter_agent_names=filter_agent_names,
        ewok_variant=ewok_variant,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    return _prediction_dataframe_from_records(ewok_df, records)


def _summarize_babylm_completion_choice(
    ewok_df: pd.DataFrame,
    item_df: pd.DataFrame,
    margin_eps: float,
):
    domain_scores_official = {}
    domain_scores_full = {}
    domain_margin_stats = {}
    macro_metric_keys = ["acc_combined", "mean_signed_m", "mean_abs_m", "tie_rate_m"]
    macro_metric_values = {key: [] for key in macro_metric_keys}
    total_items = 0

    for domain in ewok_df["Domain"].unique():
        df = item_df[item_df["domain"] == domain].reset_index(drop=True)
        m1_arr = df["margin_official_m1"].to_numpy(dtype=np.float64)
        m2_arr = df["margin_symmetric_m2"].to_numpy(dtype=np.float64)
        m_arr = 0.5 * (m1_arr + m2_arr)
        acc1_arr = df["babylm_completion_choice_correct_official"].to_numpy(dtype=np.float64)
        acc2_arr = df["babylm_completion_choice_correct_symmetric"].to_numpy(dtype=np.float64)
        acc_combined_arr = df["babylm_completion_choice_correct_combined"].to_numpy(dtype=np.float64)

        acc1 = float(acc1_arr.mean())
        acc2 = float(acc2_arr.mean())
        acc_combined = float(acc_combined_arr.mean())

        domain_scores_official[domain] = acc1
        domain_scores_full[domain] = (acc1, acc2)
        stats = {
            "n": int(len(df)),
            "acc_combined": acc_combined,
            "mean_signed_m": float(m_arr.mean()),
            "mean_abs_m": float(np.abs(m_arr).mean()),
            "tie_rate_m": float((np.abs(m_arr) < margin_eps).mean()),
        }
        domain_margin_stats[domain] = stats
        total_items += int(len(df))
        for key in macro_metric_keys:
            macro_metric_values[key].append(stats[key])

    avg_official = float(np.mean(list(domain_scores_official.values())))
    avg_symmetric = float(np.mean([values[1] for values in domain_scores_full.values()]))
    domain_scores_full["average"] = (avg_official, avg_symmetric)
    domain_margin_stats["average"] = {
        "n": int(total_items),
        "acc_combined": float(np.mean(macro_metric_values["acc_combined"])),
        "mean_signed_m": float(np.mean(macro_metric_values["mean_signed_m"])),
        "mean_abs_m": float(np.mean(macro_metric_values["mean_abs_m"])),
        "tie_rate_m": float(np.mean(macro_metric_values["tie_rate_m"])),
    }

    return {
        "label": "BabyLM Completion Choice",
        "tie_policy": "count_equal_as_half",
        "domain_scores_official": domain_scores_official,
        "domain_scores_full": domain_scores_full,
        "domain_margin_stats": domain_margin_stats,
    }


def _summarize_ewok_context_sensitivity(
    ewok_df: pd.DataFrame,
    item_df: pd.DataFrame,
):
    domain_scores_official = {}
    domain_scores_full = {}
    domain_stats = {}
    total_items = 0

    for domain in ewok_df["Domain"].unique():
        df = item_df[item_df["domain"] == domain].reset_index(drop=True)
        acc1 = float(df["ewok_context_sensitivity_correct_official"].to_numpy(dtype=np.float64).mean())
        acc2 = float(df["ewok_context_sensitivity_correct_symmetric"].to_numpy(dtype=np.float64).mean())
        acc_combined = float(df["ewok_context_sensitivity_correct_combined"].to_numpy(dtype=np.float64).mean())
        domain_scores_official[domain] = acc1
        domain_scores_full[domain] = (acc1, acc2)
        domain_stats[domain] = {
            "n": int(len(df)),
            "acc_combined": acc_combined,
        }
        total_items += int(len(df))

    avg_official = float(np.mean(list(domain_scores_official.values())))
    avg_symmetric = float(np.mean([values[1] for values in domain_scores_full.values()]))
    domain_scores_full["average"] = (avg_official, avg_symmetric)
    domain_stats["average"] = {
        "n": int(total_items),
        "acc_combined": float(np.mean([stats["acc_combined"] for stats in domain_stats.values() if "acc_combined" in stats])),
    }

    return {
        "label": "EWoK Paper Context Sensitivity",
        "paper_tie_policy": "count_equal_as_half",
        "paper_combined_definition": "context_sensitivity = as.integer(Accuracy_T1 == Accuracy_T2)",
        "domain_scores_official": domain_scores_official,
        "domain_scores_full": domain_scores_full,
        "domain_context_sensitivity_stats": domain_stats,
    }


def _summarize_records_all_methods(
    ewok_df: pd.DataFrame,
    records: list[dict],
    margin_eps: float,
):
    item_df = pd.DataFrame.from_records(records)
    return {
        BABYLM_COMPLETION_CHOICE: _summarize_babylm_completion_choice(
            ewok_df,
            item_df,
            margin_eps,
        ),
        EWOK_CONTEXT_SENSITIVITY: _summarize_ewok_context_sensitivity(
            ewok_df,
            item_df,
        ),
    }


def _legacy_results_from_metrics(
    metrics_by_method: dict[str, dict],
    *,
    per_item: list[dict] | None,
    return_per_item: bool,
):
    babylm_metrics = metrics_by_method[BABYLM_COMPLETION_CHOICE]
    if return_per_item:
        return (
            babylm_metrics["domain_scores_official"],
            babylm_metrics["domain_scores_full"],
            per_item,
            babylm_metrics["domain_margin_stats"],
        )
    return (
        babylm_metrics["domain_scores_official"],
        babylm_metrics["domain_scores_full"],
        babylm_metrics["domain_margin_stats"],
    )


def evaluate(
    run_dir: str | Path,
    return_per_item: bool = False,
    margin_eps: float = 1e-6,
    return_all_methods: bool = False,
    filter_agent_names: bool = True,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
):
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")

    ewok_df, _source, _run, _tokenizer, _text_config = _load_eval_context(
        run_dir,
        filter_agent_names=filter_agent_names,
        ewok_variant=ewok_variant,
        ewok_text_preprocessing=ewok_text_preprocessing,
    )
    return evaluate_loaded_run(
        _run,
        ewok_df=ewok_df,
        tokenizer=_tokenizer,
        return_per_item=return_per_item,
        margin_eps=margin_eps,
        return_all_methods=return_all_methods,
        filter_agent_names=filter_agent_names,
    )


def evaluate_loaded_run(
    run: LoadedWord2VecRun,
    *,
    ewok_df: pd.DataFrame | None = None,
    tokenizer: WordTokenizer | None = None,
    return_per_item: bool = False,
    margin_eps: float = 1e-6,
    return_all_methods: bool = False,
    filter_agent_names: bool = True,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
):
    if margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {margin_eps}")
    if ewok_df is None:
        ewok_df, _source = load_ewok_eval_data(ewok_variant)
    if tokenizer is None:
        tokenizer, _text_config = build_tokenizer_for_run(
            run,
            filter_agent_names=filter_agent_names,
            ewok_text_preprocessing=ewok_text_preprocessing,
        )

    records = ewok_per_item_records_for_run(
        run,
        ewok_df=ewok_df,
        tokenizer=tokenizer,
        margin_eps=margin_eps,
    )
    metrics_by_method = _summarize_records_all_methods(
        ewok_df,
        records,
        margin_eps=margin_eps,
    )
    per_item = records if return_per_item else None
    if return_all_methods:
        if return_per_item:
            return metrics_by_method, per_item
        return metrics_by_method
    return _legacy_results_from_metrics(
        metrics_by_method,
        per_item=per_item,
        return_per_item=return_per_item,
    )


def write_prediction_dataframe(
    run_dir: Path,
    prediction_df: pd.DataFrame,
    filename: str = "ewok_word2vec_predictions.csv",
) -> Path:
    output_path = run_dir / filename
    prediction_df.to_csv(output_path, index=False)
    return output_path


def _write_outputs(
    run_dir: Path,
    metrics_by_method: dict[str, dict],
    per_item: list[dict] | None,
    write_per_item: bool,
    metadata: dict,
    write_prediction_df: bool,
    prediction_df: pd.DataFrame | None = None,
    ewok_variant: str = "fast",
    ewok_text_preprocessing: str = "probe",
) -> None:
    output_paths = get_ewok_output_paths(run_dir, ewok_variant, ewok_text_preprocessing)
    babylm_metrics = metrics_by_method[BABYLM_COMPLETION_CHOICE]
    ewok_metrics = metrics_by_method[EWOK_CONTEXT_SENSITIVITY]

    metrics_payload = {
        # Legacy top-level fields remain BabyLM completion choice for compatibility.
        "domain_scores_official": babylm_metrics["domain_scores_official"],
        "domain_scores_full": babylm_metrics["domain_scores_full"],
        "domain_margin_stats": babylm_metrics["domain_margin_stats"],
        "ewok_context_sensitivity_domain_scores_official": ewok_metrics["domain_scores_official"],
        "ewok_context_sensitivity_domain_scores_full": ewok_metrics["domain_scores_full"],
        "metrics_by_method": metrics_by_method,
        "metadata": metadata,
    }
    with output_paths["metrics"].open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    if per_item is not None:
        with output_paths["items"].open("w", encoding="utf-8") as f:
            for record in per_item:
                f.write(json.dumps(record) + "\n")

    if write_prediction_df:
        if prediction_df is None:
            prediction_df = build_prediction_dataframe(
                run_dir,
                margin_eps=metadata.get("margin_eps", 1e-6),
                filter_agent_names=metadata.get("filter_ewok_agent_names", True),
                ewok_variant=ewok_variant,
                ewok_text_preprocessing=ewok_text_preprocessing,
            )
        write_prediction_dataframe(run_dir, prediction_df, filename=output_paths["prediction_df"].name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Word2Vec run on EWoK.")
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Run directory containing vectors.kv or legacy model.pt plus vocab.json",
    )
    parser.add_argument("--margin_eps", type=float, default=1e-6, help="Threshold used to flag near-ties")
    parser.add_argument(
        "--ewok_variant",
        choices=_VALID_EWOK_VARIANTS,
        default="fast",
        help="Which EWoK dataset to evaluate: fast subset or full filtered set (default: fast)",
    )
    parser.add_argument(
        "--ewok_text_preprocessing",
        choices=_VALID_EWOK_TEXT_PREPROCESSING,
        default="probe",
        help="How to tokenize EWoK text before vector lookup: probe-style tokenizer or paper-style lowercase/punctuation-strip split",
    )
    parser.add_argument("--write_per_item", action="store_true", help="Also write ewok_items.jsonl")
    parser.add_argument(
        "--filter_ewok_agent_names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop recurring EWoK filler agent names from tokenized contexts/targets during evaluation (default: on)",
    )
    parser.add_argument(
        "--write_prediction_df",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a merged EWoK dataframe with Word2Vec scores and margins to CSV",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_paths = get_ewok_output_paths(
        run_dir,
        args.ewok_variant,
        args.ewok_text_preprocessing,
    )
    ewok_df, source = load_ewok_eval_data(args.ewok_variant)
    run = load_word2vec_run(run_dir)
    records = ewok_per_item_records(
        run_dir,
        margin_eps=args.margin_eps,
        filter_agent_names=args.filter_ewok_agent_names,
        ewok_variant=args.ewok_variant,
        ewok_text_preprocessing=args.ewok_text_preprocessing,
    )
    metrics_by_method = _summarize_records_all_methods(
        ewok_df,
        records,
        margin_eps=args.margin_eps,
    )
    metadata = {
        "run_dir": str(run_dir),
        "ewok_source": str(source),
        "ewok_variant": args.ewok_variant,
        "ewok_text_preprocessing": args.ewok_text_preprocessing,
        "normalization": run.summary.get("normalization", {}),
        "margin_eps": float(args.margin_eps),
        "filter_ewok_agent_names": bool(args.filter_ewok_agent_names),
    }
    prediction_df = _prediction_dataframe_from_records(ewok_df, records) if args.write_prediction_df else None
    _write_outputs(
        run_dir,
        metrics_by_method=metrics_by_method,
        per_item=(records if args.write_per_item else None),
        write_per_item=args.write_per_item,
        metadata=metadata,
        write_prediction_df=args.write_prediction_df,
        prediction_df=prediction_df,
        ewok_variant=args.ewok_variant,
        ewok_text_preprocessing=args.ewok_text_preprocessing,
    )
    output_payload = {
        "run_dir": str(run_dir),
        "ewok_metrics": str(output_paths["metrics"]),
    }
    if args.write_prediction_df:
        output_payload["ewok_word2vec_predictions"] = str(output_paths["prediction_df"])
    print(json.dumps(output_payload, indent=2))
