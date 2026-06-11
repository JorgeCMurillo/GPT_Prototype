"""EWoK pair construction for linear probing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from evaluation.ewok_data import load_ewok_df
from research.bos_aligned_proto.analysis.attribution.common.ewok_filters import (
    apply_ewok_target_filter,
    load_ewok_target_filter_spec,
    normalize_ewok_context_diff,
    normalize_ewok_string,
    normalize_ewok_variant,
)


PAIR_ROLES = ("c1t1", "c1t2", "c2t2", "c2t1")
PAIR_LABELS = {
    "c1t1": 1,
    "c1t2": 0,
    "c2t2": 1,
    "c2t1": 0,
}


@dataclass(frozen=True)
class EWOKProbePair:
    pair_id: str
    row_index: int
    domain: str
    role: str
    label: int
    context: str
    target: str
    text: str
    concept_a: str
    concept_b: str
    context_type_raw: str
    context_type: str
    context_diff_raw: str
    context_diff: str
    target_diff_raw: str
    target_diff: str

    def to_json(self) -> dict:
        return asdict(self)


def _pair_specs(row) -> tuple[tuple[str, str, str], ...]:
    return (
        ("c1t1", str(row.Context1), str(row.Target1)),
        ("c1t2", str(row.Context1), str(row.Target2)),
        ("c2t2", str(row.Context2), str(row.Target2)),
        ("c2t1", str(row.Context2), str(row.Target1)),
    )


def build_probe_pairs_from_dataframe(
    df: pd.DataFrame,
    *,
    variant: str,
    max_targets: int = 0,
) -> tuple[EWOKProbePair, ...]:
    """Create one binary probe example for each EWoK context-target pairing."""

    resolved_variant = normalize_ewok_variant(variant)
    source = df.convert_dtypes()
    if source.index.is_unique:
        frame = source.reset_index()
    else:
        frame = source.reset_index(drop=True).reset_index()
    if max_targets > 0:
        frame = frame.head(int(max_targets))

    pairs: list[EWOKProbePair] = []
    for row in frame.itertuples(index=False):
        row_index = int(row.index)
        domain = normalize_ewok_string(row.Domain)
        for role, context, target in _pair_specs(row):
            text = f"{context} {target}"
            pairs.append(
                EWOKProbePair(
                    pair_id=f"ewok-{resolved_variant}:{row_index}:{role}",
                    row_index=row_index,
                    domain=domain,
                    role=role,
                    label=PAIR_LABELS[role],
                    context=context,
                    target=target,
                    text=text,
                    concept_a=str(row.ConceptA),
                    concept_b=str(row.ConceptB),
                    context_type_raw=str(row.ContextType),
                    context_type=normalize_ewok_string(row.ContextType),
                    context_diff_raw=str(row.ContextDiff),
                    context_diff=normalize_ewok_context_diff(row.ContextDiff),
                    target_diff_raw=str(row.TargetDiff),
                    target_diff=normalize_ewok_string(row.TargetDiff),
                )
            )
    return tuple(pairs)


def build_ewok_probe_pairs(
    *,
    variant: str = "fast",
    filter_spec_path: str | Path | None = None,
    max_targets: int = 0,
) -> tuple[EWOKProbePair, ...]:
    """Load EWoK and create binary probe examples for every selected row."""

    resolved_variant = normalize_ewok_variant(variant)
    filter_spec = None
    if filter_spec_path is not None:
        filter_spec = load_ewok_target_filter_spec(filter_spec_path)
        if filter_spec.variant is not None:
            resolved_variant = filter_spec.variant

    df, _ = load_ewok_df(resolved_variant)
    if filter_spec is not None:
        df = apply_ewok_target_filter(df.reset_index(), filter_spec).set_index("index")
        if df.empty:
            spec_label = filter_spec.name or str(Path(filter_spec_path).expanduser())
            raise ValueError(f"EWoK filter spec {spec_label!r} matched zero rows.")

    pairs = build_probe_pairs_from_dataframe(
        df,
        variant=resolved_variant,
        max_targets=max_targets,
    )
    if not pairs:
        raise ValueError("No EWoK probe pairs were constructed.")
    return pairs


def pair_records_to_json(pairs: Iterable[EWOKProbePair]) -> list[dict]:
    return [pair.to_json() for pair in pairs]


def validate_pair_alignment(pair_ids: Sequence[str], pairs: Sequence[EWOKProbePair]) -> None:
    expected = [pair.pair_id for pair in pairs]
    got = [str(pair_id) for pair_id in pair_ids]
    if got != expected:
        raise ValueError("Activation cache pair_ids do not match the current EWoK pair construction.")


__all__ = [
    "EWOKProbePair",
    "PAIR_LABELS",
    "PAIR_ROLES",
    "build_ewok_probe_pairs",
    "build_probe_pairs_from_dataframe",
    "pair_records_to_json",
    "validate_pair_alignment",
]
