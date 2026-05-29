"""Compare EWoK metrics from multiple Word2Vec lexical-probe runs.

This script assumes each run has already been evaluated with
``eval_ewok_word2vec.py`` or with ``train_word2vec.py --eval_after_train``.
It does not train models and it does not rescore EWoK; it only compares saved
EWoK metric artifacts across runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .eval_ewok_word2vec import (
    BABYLM_COMPLETION_CHOICE,
    EWOK_CONTEXT_SENSITIVITY,
    get_ewok_output_paths,
)


DOMAIN_ORDER = (
    "agent-properties",
    "material-dynamics",
    "material-properties",
    "physical-dynamics",
    "physical-interactions",
    "physical-relations",
    "quantitative-properties",
    "social-interactions",
    "social-properties",
    "social-relations",
    "spatial-relations",
    "average",
)
VALID_METHODS = (BABYLM_COMPLETION_CHOICE, EWOK_CONTEXT_SENSITIVITY)
VALID_SCORE_KINDS = ("pair_average", "combined")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "runs" / "research" / "w2v_lexical_probe" / "comparisons"


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_dir: Path


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "comparison"


def _parse_run_spec(raw: str) -> RunSpec:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
    else:
        path = raw
        label = Path(path).expanduser().name
    if not label:
        raise ValueError(f"Run label cannot be empty in spec: {raw!r}")
    return RunSpec(label=_safe_slug(label), run_dir=Path(path).expanduser().resolve())


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_last_jsonl_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def _metric_paths(run_dir: Path, ewok_variant: str, ewok_text_preprocessing: str) -> dict[str, Path]:
    paths = get_ewok_output_paths(run_dir, ewok_variant, ewok_text_preprocessing)
    interval_prefix = "ewok" if ewok_variant == "fast" else f"ewok_{ewok_variant}"
    interval_suffix = "" if ewok_text_preprocessing == "probe" else f"_{ewok_text_preprocessing}prep"
    paths["interval_metrics"] = run_dir / f"{interval_prefix}{interval_suffix}_interval_metrics.jsonl"
    return paths


def _load_metric_payload(run_dir: Path, ewok_variant: str, ewok_text_preprocessing: str) -> tuple[dict, Path]:
    paths = _metric_paths(run_dir, ewok_variant, ewok_text_preprocessing)
    if paths["metrics"].exists():
        return _load_json(paths["metrics"]), paths["metrics"]
    interval_payload = _load_last_jsonl_record(paths["interval_metrics"])
    if interval_payload is not None:
        return interval_payload, paths["interval_metrics"]
    raise FileNotFoundError(
        f"No EWoK metrics found for {run_dir}. Expected {paths['metrics']} "
        f"or {paths['interval_metrics']}."
    )


def _pair_to_mean(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        left, right = value[0], value[1]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return 0.5 * (float(left) + float(right))
    return None


def _method_payload(payload: dict, method: str) -> dict:
    return payload.get("metrics_by_method", {}).get(method, payload)


def extract_scores(payload: dict, *, method: str, score_kind: str) -> dict[str, float]:
    """Extract one scalar score per EWoK domain from a saved metric payload."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of: {', '.join(VALID_METHODS)}")
    if score_kind not in VALID_SCORE_KINDS:
        raise ValueError(
            f"Unknown score_kind {score_kind!r}; expected one of: {', '.join(VALID_SCORE_KINDS)}"
        )

    method_payload = _method_payload(payload, method)
    if method == BABYLM_COMPLETION_CHOICE:
        full_scores = method_payload.get("domain_scores_full") or payload.get("domain_scores_full") or {}
        stats = method_payload.get("domain_margin_stats") or payload.get("domain_margin_stats") or {}
    else:
        full_scores = (
            method_payload.get("domain_scores_full")
            or payload.get("ewok_context_sensitivity_domain_scores_full")
            or {}
        )
        stats = method_payload.get("domain_context_sensitivity_stats") or {}

    scores: dict[str, float] = {}
    for domain in DOMAIN_ORDER:
        if score_kind == "pair_average":
            scalar = _pair_to_mean(full_scores.get(domain))
        else:
            domain_stats = stats.get(domain, {})
            scalar = domain_stats.get("acc_combined") if isinstance(domain_stats, dict) else None
        if isinstance(scalar, (int, float)):
            scores[domain] = float(scalar)
    return scores


def write_comparison_csv(
    output_path: Path,
    *,
    scores_by_label: dict[str, dict[str, float]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(scores_by_label)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", *labels])
        writer.writeheader()
        for domain in DOMAIN_ORDER:
            if not any(domain in scores for scores in scores_by_label.values()):
                continue
            row = {"domain": domain}
            for label, scores in scores_by_label.items():
                row[label] = scores.get(domain)
            writer.writerow(row)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare saved Word2Vec EWoK metrics across runs.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as label=/path/to/run, or just /path/to/run. Repeat for each run.",
    )
    parser.add_argument("--ewok_variant", choices=("fast", "full"), default="fast")
    parser.add_argument("--ewok_text_preprocessing", choices=("probe", "paper"), default="probe")
    parser.add_argument(
        "--method",
        action="append",
        choices=VALID_METHODS,
        help="Metric method to compare. Repeat to select multiple. Defaults to both.",
    )
    parser.add_argument(
        "--score_kind",
        action="append",
        choices=VALID_SCORE_KINDS,
        help="Score scalar to compare. Repeat to select multiple. Defaults to pair_average.",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for comparison CSV/JSON files")
    parser.add_argument("--output_prefix", default=None, help="Optional filename prefix")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_specs = [_parse_run_spec(raw) for raw in args.run]
    methods = args.method or list(VALID_METHODS)
    score_kinds = args.score_kind or ["pair_average"]
    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = args.output_prefix or "ewok_word2vec_runs"

    payloads: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for spec in run_specs:
        payload, source_path = _load_metric_payload(
            spec.run_dir,
            args.ewok_variant,
            args.ewok_text_preprocessing,
        )
        payloads[spec.label] = payload
        sources[spec.label] = str(source_path)

    outputs: dict[str, str] = {}
    averages: dict[str, dict[str, float | None]] = {}
    for method in methods:
        for score_kind in score_kinds:
            scores_by_label = {
                label: extract_scores(payload, method=method, score_kind=score_kind)
                for label, payload in payloads.items()
            }
            filename = f"{_safe_slug(prefix)}_{method}_{score_kind}.csv"
            csv_path = write_comparison_csv(output_dir / filename, scores_by_label=scores_by_label)
            key = f"{method}:{score_kind}"
            outputs[key] = str(csv_path)
            averages[key] = {
                label: scores.get("average")
                for label, scores in scores_by_label.items()
            }

    summary = {
        "ewok_variant": args.ewok_variant,
        "ewok_text_preprocessing": args.ewok_text_preprocessing,
        "runs": {spec.label: str(spec.run_dir) for spec in run_specs},
        "metric_sources": sources,
        "outputs": outputs,
        "averages": averages,
    }
    summary_path = output_dir / f"{_safe_slug(prefix)}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
