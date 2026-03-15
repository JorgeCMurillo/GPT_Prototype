"""Download Google's pretrained Word2Vec model if needed and evaluate it on EWoK.

The script materializes the pretrained vectors into a normal lexical-probe run
directory so the shared loader and evaluator can treat it like any other
Word2Vec run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import gensim.downloader as gensim_api
    from gensim import __version__ as GENSIM_VERSION
    _GENSIM_IMPORT_ERROR = None
except Exception as exc:
    gensim_api = None
    GENSIM_VERSION = None
    _GENSIM_IMPORT_ERROR = exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from research.w2v_lexical_probe.eval_ewok_word2vec import (
        BABYLM_COMPLETION_CHOICE,
        EWOK_CONTEXT_SENSITIVITY,
        build_prediction_dataframe,
        evaluate_loaded_run,
        get_ewok_output_paths,
        load_ewok_eval_data,
        write_prediction_dataframe,
    )
    from research.w2v_lexical_probe.model import (
        Vocabulary,
        Word2VecTrainingConfig,
        build_vocabulary_from_keyed_vectors,
        load_word2vec_run,
        save_word2vec_run,
    )
    from research.w2v_lexical_probe.text import TextNormalizationConfig
else:
    from .eval_ewok_word2vec import (
        BABYLM_COMPLETION_CHOICE,
        EWOK_CONTEXT_SENSITIVITY,
        build_prediction_dataframe,
        evaluate_loaded_run,
        get_ewok_output_paths,
        load_ewok_eval_data,
        write_prediction_dataframe,
    )
    from .model import (
        Vocabulary,
        Word2VecTrainingConfig,
        build_vocabulary_from_keyed_vectors,
        load_word2vec_run,
        save_word2vec_run,
    )
    from .text import TextNormalizationConfig

DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "runs" / "research" / "w2v_lexical_probe"
DEFAULT_PRETRAINED_MODEL = "word2vec-google-news-300"


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _default_run_name(model_name: str) -> str:
    return f"google_pretrained_{_safe_slug(model_name)}"


def _require_gensim_downloader() -> None:
    if _GENSIM_IMPORT_ERROR is not None or gensim_api is None:
        raise RuntimeError(
            "gensim with downloader support is required. Install it with `pip install gensim`."
        ) from _GENSIM_IMPORT_ERROR


def _pretrained_run_complete(run_dir: Path) -> bool:
    required = ["vectors.kv", "vocab.json", "train_summary.json"]
    return all((run_dir / name).exists() for name in required)


def _build_pretrained_vocabulary(keyed_vectors) -> Vocabulary:
    try:
        return build_vocabulary_from_keyed_vectors(keyed_vectors)
    except Exception:
        # Some pretrained KeyedVectors omit token counts. For evaluation we only
        # need stable token-id mapping and vectors.
        id_to_token = list(keyed_vectors.index_to_key)
        counts = [0] * len(id_to_token)
        return Vocabulary(
            id_to_token=id_to_token,
            counts=counts,
            total_tokens_seen=0,
            total_tokens_retained=0,
        )


def _build_pretrained_summary(
    *,
    run_name: str,
    model_name: str,
    vocabulary: Vocabulary,
    training_config: Word2VecTrainingConfig,
    normalization: TextNormalizationConfig,
) -> dict:
    return {
        "backend": "gensim_pretrained_google",
        "gensim_version": GENSIM_VERSION,
        "run_name": run_name,
        "source_model": model_name,
        "source_provider": "Google pretrained Word2Vec via gensim.downloader",
        "source_type": "pretrained_import",
        "normalization": normalization.to_dict(),
        "training_config": training_config.to_dict(),
        "vocabulary": {
            "documents_seen": 0,
            "vocab_size": vocabulary.size,
            "total_tokens_seen": vocabulary.total_tokens_seen,
            "total_tokens_retained": vocabulary.total_tokens_retained,
            "min_count": 0,
        },
        "training": {
            "kind": "pretrained_import",
            "epochs_completed": 0,
            "corpus_count": 0,
            "corpus_total_words": vocabulary.total_tokens_seen,
            "effective_words": 0,
            "raw_words": 0,
        },
    }


def _ensure_pretrained_run(
    *,
    run_dir: Path,
    run_name: str,
    model_name: str,
    force_reimport: bool,
) -> tuple[Path, bool]:
    if _pretrained_run_complete(run_dir) and not force_reimport:
        print(f"[info] reusing existing pretrained run at {run_dir}")
        return run_dir, False

    _require_gensim_downloader()
    print(f"[info] downloading/loading pretrained model `{model_name}` via gensim.downloader")
    keyed_vectors = gensim_api.load(model_name)

    normalization = TextNormalizationConfig(lowercase=False)
    vocabulary = _build_pretrained_vocabulary(keyed_vectors)
    training_config = Word2VecTrainingConfig(
        embedding_dim=int(keyed_vectors.vector_size),
        window_size=0,
        negative_samples=0,
        min_count=0,
        epochs=0,
        learning_rate=0.0,
        batch_words=0,
        seed=0,
        checkpoint_every_epochs=0,
        workers=1,
        sample=0.0,
        sg=1,
        hs=0,
    )
    summary = _build_pretrained_summary(
        run_name=run_name,
        model_name=model_name,
        vocabulary=vocabulary,
        training_config=training_config,
        normalization=normalization,
    )
    save_word2vec_run(
        run_dir,
        keyed_vectors,
        vocabulary,
        training_config,
        summary,
    )
    print(f"[info] saved pretrained run to {run_dir}")
    return run_dir, True


def _write_eval_outputs(
    *,
    run_dir: Path,
    metrics_by_method: dict[str, dict],
    per_item: list[dict] | None,
    metadata: dict,
    write_prediction_df: bool,
    prediction_df,
    ewok_variant: str,
    ewok_text_preprocessing: str,
) -> None:
    output_paths = get_ewok_output_paths(run_dir, ewok_variant, ewok_text_preprocessing)
    babylm_metrics = metrics_by_method[BABYLM_COMPLETION_CHOICE]
    ewok_metrics = metrics_by_method[EWOK_CONTEXT_SENSITIVITY]
    payload = {
        "domain_scores_official": babylm_metrics["domain_scores_official"],
        "domain_scores_full": babylm_metrics["domain_scores_full"],
        "domain_margin_stats": babylm_metrics["domain_margin_stats"],
        "ewok_context_sensitivity_domain_scores_official": ewok_metrics["domain_scores_official"],
        "ewok_context_sensitivity_domain_scores_full": ewok_metrics["domain_scores_full"],
        "metrics_by_method": metrics_by_method,
        "metadata": metadata,
    }
    with output_paths["metrics"].open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if per_item is not None:
        with output_paths["items"].open("w", encoding="utf-8") as f:
            for record in per_item:
                f.write(json.dumps(record) + "\n")

    if write_prediction_df and prediction_df is not None:
        write_prediction_dataframe(run_dir, prediction_df, filename=output_paths["prediction_df"].name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensure the pretrained Google Word2Vec model exists as a lexical-probe run and evaluate it on EWoK."
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_PRETRAINED_MODEL,
        help="gensim.downloader model id to import; defaults to the Google News Word2Vec model",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Parent directory where the lexical-probe run directory should live",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional run directory name override; defaults to google_pretrained_<model_name>",
    )
    parser.add_argument(
        "--margin_eps",
        type=float,
        default=1e-6,
        help="Near-tie threshold forwarded to the shared evaluator",
    )
    parser.add_argument(
        "--ewok_variant",
        choices=("fast", "full"),
        default="fast",
        help="Which EWoK dataset to evaluate: fast subset or full filtered set (default: fast)",
    )
    parser.add_argument(
        "--ewok_text_preprocessing",
        choices=("probe", "paper"),
        default="probe",
        help="How to tokenize EWoK text before vector lookup: probe-style tokenizer or paper-style lowercase/punctuation-strip split",
    )
    parser.add_argument(
        "--force_reimport",
        action="store_true",
        help="Re-import the pretrained vectors into the run directory even if artifacts already exist",
    )
    parser.add_argument(
        "--filter_ewok_agent_names",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop recurring EWoK filler agent names from tokenized contexts/targets during evaluation (default: on)",
    )
    parser.add_argument("--write_per_item", action="store_true", help="Also write ewok_items.jsonl")
    parser.add_argument(
        "--write_prediction_df",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write ewok_word2vec_predictions.csv",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.margin_eps < 0:
        raise ValueError(f"margin_eps must be >= 0, got: {args.margin_eps}")

    run_name = args.run_name or _default_run_name(args.model_name)
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    run_dir, imported = _ensure_pretrained_run(
        run_dir=run_dir,
        run_name=run_name,
        model_name=args.model_name,
        force_reimport=args.force_reimport,
    )

    run = load_word2vec_run(run_dir)
    output_paths = get_ewok_output_paths(
        run_dir,
        args.ewok_variant,
        args.ewok_text_preprocessing,
    )
    ewok_df, ewok_source = load_ewok_eval_data(args.ewok_variant)
    results = evaluate_loaded_run(
        run,
        ewok_df=ewok_df,
        return_per_item=args.write_per_item,
        margin_eps=args.margin_eps,
        return_all_methods=True,
        filter_agent_names=args.filter_ewok_agent_names,
        ewok_variant=args.ewok_variant,
        ewok_text_preprocessing=args.ewok_text_preprocessing,
    )
    if args.write_per_item:
        metrics_by_method, per_item = results
    else:
        metrics_by_method = results
        per_item = None

    prediction_df = None
    if args.write_prediction_df:
        prediction_df = build_prediction_dataframe(
            run_dir,
            margin_eps=args.margin_eps,
            filter_agent_names=args.filter_ewok_agent_names,
            ewok_variant=args.ewok_variant,
            ewok_text_preprocessing=args.ewok_text_preprocessing,
        )

    metadata = {
        "run_dir": str(run_dir),
        "source_model": args.model_name,
        "ewok_source": str(ewok_source),
        "ewok_variant": args.ewok_variant,
        "ewok_text_preprocessing": args.ewok_text_preprocessing,
        "normalization": run.summary.get("normalization", {}),
        "margin_eps": float(args.margin_eps),
        "imported_pretrained_model": bool(imported),
        "filter_ewok_agent_names": bool(args.filter_ewok_agent_names),
    }
    _write_eval_outputs(
        run_dir=run_dir,
        metrics_by_method=metrics_by_method,
        per_item=per_item,
        metadata=metadata,
        write_prediction_df=args.write_prediction_df,
        prediction_df=prediction_df,
        ewok_variant=args.ewok_variant,
        ewok_text_preprocessing=args.ewok_text_preprocessing,
    )
    output_payload = {
        "run_dir": str(run_dir),
        "imported_pretrained_model": bool(imported),
        "ewok_metrics": str(output_paths["metrics"]),
    }
    if args.write_prediction_df:
        output_payload["ewok_word2vec_predictions"] = str(output_paths["prediction_df"])
    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
