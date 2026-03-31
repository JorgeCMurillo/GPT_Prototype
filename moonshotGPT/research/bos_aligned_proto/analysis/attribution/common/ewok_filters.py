"""File-based EWoK target filtering helpers.

The attribution pipeline already records rich per-item EWoK metadata such as
domain, context type, context difference, and target difference. This module
turns those fields into a small reusable filter-spec surface so users can
define query subsets in JSON files rather than editing Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd


EWOK_VARIANTS = ("fast", "full")


def normalize_ewok_variant(value: str | None) -> str:
    resolved = str(value or "fast").strip().lower()
    if resolved not in EWOK_VARIANTS:
        raise ValueError(f"Unknown EWoK variant {resolved!r}; expected one of {EWOK_VARIANTS!r}")
    return resolved


def normalize_ewok_string(value: Any) -> str:
    return str(value).strip()


def normalize_ewok_context_diff(value: Any) -> str:
    raw = normalize_ewok_string(value)
    if raw == "variable_swap":
        return "variable swap"
    return raw


def _normalize_string_list(
    raw_values: Any,
    *,
    normalizer: Callable[[Any], str],
) -> tuple[str, ...]:
    if raw_values is None:
        return ()
    if isinstance(raw_values, str):
        values = [raw_values]
    elif isinstance(raw_values, Sequence):
        values = list(raw_values)
    else:
        raise TypeError(f"Expected a string or sequence of strings, got {type(raw_values)!r}")
    normalized = {normalizer(value) for value in values if str(value).strip()}
    return tuple(sorted(normalized))


def _normalize_int_list(raw_values: Any) -> tuple[int, ...]:
    if raw_values is None:
        return ()
    if isinstance(raw_values, (int, float)):
        values = [raw_values]
    elif isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
        values = list(raw_values)
    else:
        raise TypeError(f"Expected an int or sequence of ints, got {type(raw_values)!r}")
    normalized = {int(value) for value in values}
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class EWOKTargetFilterSpec:
    """Declarative filter for selecting a subset of EWoK query targets."""

    name: str | None = None
    description: str | None = None
    variant: str | None = None
    domains: tuple[str, ...] = ()
    context_types: tuple[str, ...] = ()
    context_diffs: tuple[str, ...] = ()
    target_diffs: tuple[str, ...] = ()
    row_indices: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.domains,
                self.context_types,
                self.context_diffs,
                self.target_diffs,
                self.row_indices,
            )
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def load_ewok_target_filter_spec(path: str | Path) -> EWOKTargetFilterSpec:
    """Load one JSON filter spec from disk.

    Supported keys:
    - `variant`
    - `domains`
    - `context_types`
    - `context_diffs`
    - `target_diffs`
    - `row_indices`

    Singular aliases such as `domain` and `context_type` are also accepted for
    convenience when writing tiny ad hoc specs.
    """

    spec_path = Path(path).expanduser().resolve()
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"EWoK filter spec must be a JSON object, got {type(payload)!r}")

    variant = payload.get("variant")
    resolved_variant = None if variant is None else normalize_ewok_variant(str(variant))
    return EWOKTargetFilterSpec(
        name=payload.get("name"),
        description=payload.get("description"),
        variant=resolved_variant,
        domains=_normalize_string_list(
            payload.get("domains", payload.get("domain")),
            normalizer=normalize_ewok_string,
        ),
        context_types=_normalize_string_list(
            payload.get("context_types", payload.get("context_type")),
            normalizer=normalize_ewok_string,
        ),
        context_diffs=_normalize_string_list(
            payload.get("context_diffs", payload.get("context_diff")),
            normalizer=normalize_ewok_context_diff,
        ),
        target_diffs=_normalize_string_list(
            payload.get("target_diffs", payload.get("target_diff")),
            normalizer=normalize_ewok_string,
        ),
        row_indices=_normalize_int_list(payload.get("row_indices", payload.get("row_index"))),
    )


def apply_ewok_target_filter(
    df: pd.DataFrame,
    spec: EWOKTargetFilterSpec,
) -> pd.DataFrame:
    """Return the subset of EWoK rows matching the filter spec.

    Filtering is conjunctive across fields: if a spec provides both `domains`
    and `context_types`, the returned rows must satisfy both constraints.
    """

    if spec.is_empty:
        return df.copy()

    frame = df.copy()
    frame["_domain_norm"] = frame["Domain"].map(normalize_ewok_string)
    frame["_context_type_norm"] = frame["ContextType"].map(normalize_ewok_string)
    frame["_context_diff_norm"] = frame["ContextDiff"].map(normalize_ewok_context_diff)
    frame["_target_diff_norm"] = frame["TargetDiff"].map(normalize_ewok_string)

    mask = pd.Series(True, index=frame.index)
    if spec.domains:
        mask &= frame["_domain_norm"].isin(spec.domains)
    if spec.context_types:
        mask &= frame["_context_type_norm"].isin(spec.context_types)
    if spec.context_diffs:
        mask &= frame["_context_diff_norm"].isin(spec.context_diffs)
    if spec.target_diffs:
        mask &= frame["_target_diff_norm"].isin(spec.target_diffs)
    if spec.row_indices:
        if "index" not in frame.columns:
            raise KeyError("EWoK filter spec uses row_indices, but the dataframe has no reset index column")
        mask &= frame["index"].isin(spec.row_indices)

    filtered = frame.loc[mask].copy()
    return filtered.drop(
        columns=[
            "_domain_norm",
            "_context_type_norm",
            "_context_diff_norm",
            "_target_diff_norm",
        ],
        errors="ignore",
    )


def build_ewok_filter_catalog(df: pd.DataFrame, *, variant: str) -> dict[str, Any]:
    """Summarize the categorical support of an EWoK dataframe for documentation."""

    frame = df.copy()
    frame["DomainNorm"] = frame["Domain"].map(normalize_ewok_string)
    frame["ContextTypeNorm"] = frame["ContextType"].map(normalize_ewok_string)
    frame["ContextDiffNorm"] = frame["ContextDiff"].map(normalize_ewok_context_diff)
    frame["TargetDiffNorm"] = frame["TargetDiff"].map(normalize_ewok_string)
    return {
        "variant": normalize_ewok_variant(variant),
        "row_count": int(len(frame)),
        "domains": {str(key): int(value) for key, value in frame["DomainNorm"].value_counts().sort_index().items()},
        "context_types": {
            str(key): int(value) for key, value in frame["ContextTypeNorm"].value_counts().sort_index().items()
        },
        "context_diffs": {
            str(key): int(value) for key, value in frame["ContextDiffNorm"].value_counts().sort_index().items()
        },
        "target_diffs": {
            str(key): int(value) for key, value in frame["TargetDiffNorm"].value_counts().sort_index().items()
        },
        "notes": [
            "ContextDiff is normalized so raw 'variable_swap' is treated as 'variable swap'.",
            "Filters intersect across fields; for example, domains + context_types means both must match.",
        ],
    }


__all__ = [
    "EWOKTargetFilterSpec",
    "EWOK_VARIANTS",
    "apply_ewok_target_filter",
    "build_ewok_filter_catalog",
    "load_ewok_target_filter_spec",
    "normalize_ewok_context_diff",
    "normalize_ewok_string",
    "normalize_ewok_variant",
]
