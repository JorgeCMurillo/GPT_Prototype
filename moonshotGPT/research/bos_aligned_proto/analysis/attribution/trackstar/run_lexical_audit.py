"""Lexical baseline audit for TrackStar top-window retrievals."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from research.bos_aligned_proto.analysis.attribution.common.checkpoints import (
    load_tokenizer_from_checkpoint,
)
from research.bos_aligned_proto.analysis.attribution.trackstar.per_query_top_examples import (
    DEFAULT_KEYWORDS,
    QUERY_TERM_STOPWORDS,
)


WORD_RE = re.compile(r"[a-z][a-z-]+")
COMPLETION_SIDE_TARGET_MARKER = ":completion_side:"


def _step_tag(step: int) -> str:
    return f"step{int(step):08d}"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _extract_terms(value: str, *, min_len: int = 2) -> tuple[str, ...]:
    terms: list[str] = []
    for term in WORD_RE.findall(_normalize(value)):
        if len(term) < min_len:
            continue
        if term in QUERY_TERM_STOPWORDS:
            continue
        terms.append(term)
    return tuple(dict.fromkeys(terms))


def _simple_stem(value: str) -> str:
    term = _normalize(value)
    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 5 and term.endswith("ing"):
        base = term[:-3]
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(term) > 4 and term.endswith("ed"):
        base = term[:-2]
        if len(base) > 3 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(term) > 4 and term.endswith(("ses", "xes", "zes", "ches", "shes")):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _word_sets(text: str) -> tuple[set[str], set[str]]:
    words = set(WORD_RE.findall(_normalize(text)))
    stems = {_simple_stem(word) for word in words}
    return words, stems


def _exact_hits(words: set[str], terms: Sequence[str]) -> list[str]:
    return [term for term in terms if term in words]


def _stem_hits(stems: set[str], terms: Sequence[str]) -> list[str]:
    return [term for term in terms if _simple_stem(term) in stems]


def _term_metadata(item: dict[str, Any]) -> dict[str, Any]:
    context1_terms = _extract_terms(str(item.get("context1", "")))
    target1_terms = _extract_terms(str(item.get("target1", "")))
    context2_terms = _extract_terms(str(item.get("context2", "")))
    target2_terms = _extract_terms(str(item.get("target2", "")))
    side1_terms = tuple(dict.fromkeys(context1_terms + target1_terms))
    side2_terms = tuple(dict.fromkeys(context2_terms + target2_terms))
    query_terms = tuple(dict.fromkeys(side1_terms + side2_terms))
    concept_terms = tuple(
        dict.fromkeys(
            _extract_terms(str(item.get("concept_a", "")))
            + _extract_terms(str(item.get("concept_b", "")))
        )
    )
    return {
        "context1_terms": context1_terms,
        "target1_terms": target1_terms,
        "context2_terms": context2_terms,
        "target2_terms": target2_terms,
        "side1_terms": side1_terms,
        "side2_terms": side2_terms,
        "query_terms": query_terms,
        "concept_terms": concept_terms,
    }


def _material_terms() -> tuple[str, ...]:
    by_stem: dict[str, str] = {}
    for term in DEFAULT_KEYWORDS:
        normalized = _normalize(term)
        if not normalized or " " in normalized:
            continue
        by_stem.setdefault(_simple_stem(normalized), normalized)
    return tuple(by_stem.values())


def _completion_side_from_target_id(target_id: str) -> str:
    if COMPLETION_SIDE_TARGET_MARKER not in target_id:
        return ""
    _base, side = str(target_id).rsplit(COMPLETION_SIDE_TARGET_MARKER, 1)
    return side if side in {"c1_t1", "c2_t2"} else ""


def _base_target_id(target_id: str) -> str:
    if COMPLETION_SIDE_TARGET_MARKER not in target_id:
        return str(target_id)
    base, _side = str(target_id).rsplit(COMPLETION_SIDE_TARGET_MARKER, 1)
    return base


def _apply_completion_side_terms(item: dict[str, Any]) -> dict[str, Any]:
    side = _completion_side_from_target_id(str(item.get("target_id", "")))
    item["base_target_id"] = _base_target_id(str(item.get("target_id", "")))
    item["completion_side"] = side
    if side == "c1_t1":
        item["query_terms"] = item["side1_terms"]
        item["active_side_terms"] = item["side1_terms"]
        item["active_side_text"] = f"{item.get('context1')} {item.get('target1')}"
    elif side == "c2_t2":
        item["query_terms"] = item["side2_terms"]
        item["active_side_terms"] = item["side2_terms"]
        item["active_side_text"] = f"{item.get('context2')} {item.get('target2')}"
    else:
        item["active_side_terms"] = item["query_terms"]
        item["active_side_text"] = (
            f"{item.get('context1')} {item.get('target1')} | "
            f"{item.get('context2')} {item.get('target2')}"
        )
    return item


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


class CandidateDecoder:
    def __init__(self, checkpoint_dir: str | Path, *, max_tokens: int) -> None:
        self.tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)
        self.max_tokens = int(max_tokens)
        self._memmaps: dict[str, np.memmap] = {}

    def _memmap_for_path(self, path: str | Path) -> np.memmap:
        key = str(path)
        if key not in self._memmaps:
            self._memmaps[key] = np.memmap(key, dtype=np.uint16, mode="r")
        return self._memmaps[key]

    def decode_row(self, row: pd.Series | dict[str, Any]) -> str:
        shard_path = str(row["shard_path"])
        start = int(row["token_offset_start"])
        end = int(row["token_offset_end"])
        if self.max_tokens > 0:
            end = min(end, start + self.max_tokens)
        memmap = self._memmap_for_path(shard_path)
        tokens = np.asarray(memmap[start:end], dtype=np.int64).tolist()
        text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return " ".join(str(text).split())


def _resolve_step_file(root: Path, name: str, step: int) -> Path:
    path = root / f"{name}_{_step_tag(step)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return path


def _summarize(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": int(len(frame)),
            "targets": int(frame["target_id"].nunique()) if len(frame) else 0,
            "mean_score": float(frame["score"].mean()) if "score" in frame and frame["score"].notna().any() else np.nan,
            "any_query_exact_rate": float(frame["has_query_exact_hit"].mean()) if len(frame) else np.nan,
            "any_query_stem_rate": float(frame["has_query_stem_hit"].mean()) if len(frame) else np.nan,
            "any_concept_stem_rate": float(frame["has_concept_stem_hit"].mean()) if len(frame) else np.nan,
            "any_material_keyword_rate": float(frame["has_material_keyword_hit"].mean()) if len(frame) else np.nan,
            "mean_query_exact_fraction": float(frame["query_exact_overlap_fraction"].mean()) if len(frame) else np.nan,
            "mean_query_stem_fraction": float(frame["query_stem_overlap_fraction"].mean()) if len(frame) else np.nan,
            "mean_material_keyword_hit_count": float(frame["material_keyword_hit_count"].mean()) if len(frame) else np.nan,
            "median_material_keyword_hit_count": float(frame["material_keyword_hit_count"].median()) if len(frame) else np.nan,
        }
    )


def _jsonify_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, list)).any():
            out[column] = out[column].map(lambda value: json.dumps(value) if isinstance(value, list) else value)
    return out


def _format_table(frame: pd.DataFrame, columns: Sequence[str], *, max_rows: int = 20) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.head(max_rows).to_dict(orient="records"):
        vals: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_audit(
    *,
    attribution_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    step: int,
    domain: str,
    topk: int,
    randomk: int,
    seed: int,
    max_tokens: int,
    preview_chars: int,
) -> dict[str, Path]:
    root = Path(attribution_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_items_path = root / "target_items.jsonl"
    if not target_items_path.exists():
        raise FileNotFoundError(f"Missing required artifact: {target_items_path}")
    top_rows_path = _resolve_step_file(root, "top_rows", step)
    row_summary_path = _resolve_step_file(root, "row_summary", step)

    items: dict[str, dict[str, Any]] = {}
    for item in _load_jsonl(target_items_path):
        if str(item.get("domain")) != domain:
            continue
        target_id = str(item["target_id"])
        item = dict(item)
        item.update(_term_metadata(item))
        item = _apply_completion_side_terms(item)
        items[target_id] = item
    if not items:
        raise ValueError(f"No target_items found for domain={domain!r}")

    top_rows = pd.read_csv(top_rows_path)
    top_rows["target_id"] = top_rows["target_id"].astype(str)
    top_rows = top_rows[top_rows["target_id"].isin(items)].copy()
    top_rows["source"] = "top"
    top_rows = (
        top_rows.sort_values(["target_id", "rank", "score"], ascending=[True, True, False])
        .groupby("target_id", group_keys=False)
        .head(int(topk))
    )

    pool = pd.read_csv(
        row_summary_path,
        usecols=[
            "candidate_id",
            "candidate_kind",
            "shard_path",
            "local_example_idx",
            "token_offset_start",
            "token_offset_end",
            "row_id",
            "local_row_idx",
        ],
    )

    random_rows: list[pd.DataFrame] = []
    for target_id in sorted(items):
        random_state = _stable_seed(target_id, seed)
        sample = pool.sample(
            n=min(int(randomk), len(pool)),
            replace=int(randomk) > len(pool),
            random_state=random_state,
        ).copy()
        sample["target_id"] = target_id
        sample["domain"] = domain
        sample["rank"] = np.arange(1, len(sample) + 1)
        sample["score"] = np.nan
        sample["source"] = "random"
        random_rows.append(sample)
    random_frame = pd.concat(random_rows, ignore_index=True) if random_rows else pd.DataFrame()

    all_rows = pd.concat([top_rows, random_frame], ignore_index=True, sort=False)
    unique_candidates = all_rows.drop_duplicates("candidate_id")[
        ["candidate_id", "shard_path", "token_offset_start", "token_offset_end"]
    ]
    print(
        f"domain={domain} targets={len(items)} rows={len(all_rows)} "
        f"unique_candidates={len(unique_candidates)}",
        flush=True,
    )

    decoder = CandidateDecoder(checkpoint_dir, max_tokens=max_tokens)
    decoded: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(unique_candidates.to_dict(orient="records"), start=1):
        candidate_id = int(row["candidate_id"])
        text = decoder.decode_row(row)
        words, stems = _word_sets(text)
        decoded[candidate_id] = {
            "text_preview": text[: int(preview_chars)],
            "words": words,
            "stems": stems,
        }
        if index % 2500 == 0 or index == len(unique_candidates):
            print(f"decoded {index} / {len(unique_candidates)}", flush=True)

    material_terms = _material_terms()
    records: list[dict[str, Any]] = []
    for row in all_rows.to_dict(orient="records"):
        target_id = str(row["target_id"])
        item = items[target_id]
        candidate_id = int(row["candidate_id"])
        text_info = decoded[candidate_id]
        words = text_info["words"]
        stems = text_info["stems"]

        query_exact_hits = _exact_hits(words, item["query_terms"])
        query_stem_hits = _stem_hits(stems, item["query_terms"])
        concept_exact_hits = _exact_hits(words, item["concept_terms"])
        concept_stem_hits = _stem_hits(stems, item["concept_terms"])
        side1_stem_hits = _stem_hits(stems, item["side1_terms"])
        side2_stem_hits = _stem_hits(stems, item["side2_terms"])
        material_keyword_hits = _stem_hits(stems, material_terms)

        side1_fraction = len(side1_stem_hits) / len(item["side1_terms"]) if item["side1_terms"] else 0.0
        side2_fraction = len(side2_stem_hits) / len(item["side2_terms"]) if item["side2_terms"] else 0.0
        if side1_fraction > side2_fraction:
            side_with_more_overlap = "side1"
        elif side2_fraction > side1_fraction:
            side_with_more_overlap = "side2"
        else:
            side_with_more_overlap = "tie"

        record = {
            "source": str(row["source"]),
            "target_id": target_id,
            "base_target_id": item.get("base_target_id", target_id),
            "completion_side": item.get("completion_side", ""),
            "domain": domain,
            "rank": int(row["rank"]),
            "score": float(row["score"]) if pd.notna(row.get("score")) else np.nan,
            "candidate_id": candidate_id,
            "candidate_kind": str(row.get("candidate_kind", "")),
            "shard_path": str(row.get("shard_path", "")),
            "local_example_idx": row.get("local_example_idx"),
            "token_offset_start": int(row["token_offset_start"]),
            "token_offset_end": int(row["token_offset_end"]),
            "row_id": row.get("row_id"),
            "local_row_idx": row.get("local_row_idx"),
            "concept_a": item.get("concept_a"),
            "concept_b": item.get("concept_b"),
            "context1": item.get("context1"),
            "target1": item.get("target1"),
            "context2": item.get("context2"),
            "target2": item.get("target2"),
            "side1_text": f"{item.get('context1')} {item.get('target1')}",
            "side2_text": f"{item.get('context2')} {item.get('target2')}",
            "active_side_text": item.get("active_side_text", ""),
            "context_type": item.get("context_type"),
            "context_diff": item.get("context_diff"),
            "target_diff": item.get("target_diff"),
            "query_terms": list(item["query_terms"]),
            "side1_terms": list(item["side1_terms"]),
            "side2_terms": list(item["side2_terms"]),
            "active_side_terms": list(item["active_side_terms"]),
            "concept_terms": list(item["concept_terms"]),
            "query_exact_hits": query_exact_hits,
            "query_exact_hit_count": len(query_exact_hits),
            "query_exact_overlap_fraction": len(query_exact_hits) / len(item["query_terms"])
            if item["query_terms"]
            else 0.0,
            "query_stem_hits": query_stem_hits,
            "query_stem_hit_count": len(query_stem_hits),
            "query_stem_overlap_fraction": len(query_stem_hits) / len(item["query_terms"])
            if item["query_terms"]
            else 0.0,
            "concept_exact_hits": concept_exact_hits,
            "concept_stem_hits": concept_stem_hits,
            "side1_stem_hits": side1_stem_hits,
            "side2_stem_hits": side2_stem_hits,
            "side1_stem_overlap_fraction": side1_fraction,
            "side2_stem_overlap_fraction": side2_fraction,
            "side_stem_overlap_bias": side1_fraction - side2_fraction,
            "side_with_more_overlap": side_with_more_overlap,
            "material_keyword_hits": material_keyword_hits,
            "material_keyword_hit_count": len(material_keyword_hits),
            "has_query_exact_hit": bool(query_exact_hits),
            "has_query_stem_hit": bool(query_stem_hits),
            "has_concept_stem_hit": bool(concept_stem_hits),
            "has_material_keyword_hit": bool(material_keyword_hits),
            "text_preview": text_info["text_preview"],
        }
        records.append(record)

    rows = pd.DataFrame.from_records(records)
    rows_path = out_dir / f"{domain.replace('-', '_')}_lexical_rows_{_step_tag(step)}.csv"
    _jsonify_list_columns(rows).to_csv(rows_path, index=False)

    summary_records: list[dict[str, Any]] = []
    for source, source_frame in rows.groupby("source"):
        summary_records.append({"source": source, "rank_bin": f"all{topk if source == 'top' else randomk}", **_summarize(source_frame).to_dict()})
        if source == "top":
            for limit in (1, 5, 10, 20, 50):
                if limit > int(topk):
                    continue
                sub = source_frame[source_frame["rank"] <= limit]
                summary_records.append({"source": source, "rank_bin": f"top{limit}", **_summarize(sub).to_dict()})
    aggregate = pd.DataFrame.from_records(summary_records)
    aggregate_path = out_dir / f"{domain.replace('-', '_')}_lexical_aggregate_{_step_tag(step)}.csv"
    aggregate.to_csv(aggregate_path, index=False)

    query_records: list[dict[str, Any]] = []
    for target_id, group in rows.groupby("target_id"):
        item = items[target_id]
        query_record: dict[str, Any] = {
            "target_id": target_id,
            "base_target_id": item.get("base_target_id", target_id),
            "completion_side": item.get("completion_side", ""),
            "concept_a": item.get("concept_a"),
            "concept_b": item.get("concept_b"),
            "side1_text": f"{item.get('context1')} {item.get('target1')}",
            "side2_text": f"{item.get('context2')} {item.get('target2')}",
            "active_side_text": item.get("active_side_text", ""),
        }
        for source in ("top", "random"):
            sub = group[group["source"] == source]
            if len(sub):
                summary = _summarize(sub)
                for key, value in summary.items():
                    query_record[f"{source}_{key}"] = value
        top5 = group[(group["source"] == "top") & (group["rank"] <= 5)]
        query_record["top5_any_query_stem_rate"] = (
            float(top5["has_query_stem_hit"].mean()) if len(top5) else np.nan
        )
        query_record["top5_mean_material_keyword_hit_count"] = (
            float(top5["material_keyword_hit_count"].mean()) if len(top5) else np.nan
        )
        query_records.append(query_record)
    query_summary = pd.DataFrame.from_records(query_records)
    query_summary["query_stem_enrichment"] = (
        query_summary["top_any_query_stem_rate"] - query_summary["random_any_query_stem_rate"]
    )
    query_summary["material_keyword_count_enrichment"] = (
        query_summary["top_mean_material_keyword_hit_count"]
        - query_summary["random_mean_material_keyword_hit_count"]
    )
    query_summary_path = out_dir / f"{domain.replace('-', '_')}_lexical_query_summary_{_step_tag(step)}.csv"
    query_summary.sort_values(
        ["query_stem_enrichment", "top_any_query_stem_rate"],
        ascending=[False, False],
    ).to_csv(query_summary_path, index=False)

    supported = rows[
        (rows["source"] == "top")
        & (rows["rank"] <= 10)
        & (rows["has_query_stem_hit"])
    ].sort_values(["query_stem_hit_count", "score"], ascending=[False, False])
    supported_path = out_dir / f"{domain.replace('-', '_')}_lexically_supported_top_examples_{_step_tag(step)}.csv"
    _jsonify_list_columns(supported.head(100)).to_csv(supported_path, index=False)

    low_lexical = rows[
        (rows["source"] == "top")
        & (rows["rank"] <= 5)
        & (~rows["has_query_stem_hit"])
        & (~rows["has_material_keyword_hit"])
    ].sort_values("score", ascending=False)
    low_lexical_path = out_dir / f"{domain.replace('-', '_')}_high_score_low_lexical_examples_{_step_tag(step)}.csv"
    _jsonify_list_columns(low_lexical.head(100)).to_csv(low_lexical_path, index=False)

    report_path = out_dir / f"{domain.replace('-', '_')}_lexical_audit_report_{_step_tag(step)}.md"
    report_lines = [
        f"# {domain} lexical audit ({_step_tag(step)})",
        "",
        f"- Queries: {len(items)}",
        f"- Rows: {len(rows)} ({topk} top + {randomk} random per query)",
        f"- Unique decoded candidates: {len(unique_candidates)}",
        "",
        "## Aggregate",
        "",
        _format_table(
            aggregate,
            [
                "source",
                "rank_bin",
                "rows",
                "any_query_stem_rate",
                "any_material_keyword_rate",
                "mean_query_stem_fraction",
                "mean_material_keyword_hit_count",
            ],
            max_rows=20,
        ),
        "",
        "## Strongest Query-Stem Enrichment",
        "",
        _format_table(
            query_summary.sort_values(
                ["query_stem_enrichment", "top_any_query_stem_rate"],
                ascending=[False, False],
            ),
            [
                "target_id",
                "concept_a",
                "concept_b",
                "top_any_query_stem_rate",
                "random_any_query_stem_rate",
                "query_stem_enrichment",
                "top5_any_query_stem_rate",
            ],
            max_rows=12,
        ),
        "",
        "## Highest-Score Top-5 With Low Lexical Support",
        "",
        _format_table(
            low_lexical,
            ["target_id", "rank", "score", "candidate_id", "concept_a", "concept_b", "text_preview"],
            max_rows=12,
        ),
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "rows": rows_path,
        "aggregate": aggregate_path,
        "query_summary": query_summary_path,
        "supported_examples": supported_path,
        "low_lexical_examples": low_lexical_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--domain", default="material-dynamics")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--randomk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max_tokens", type=int, default=1025)
    parser.add_argument("--preview_chars", type=int, default=800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_audit(
        attribution_dir=args.attribution_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        step=args.step,
        domain=args.domain,
        topk=args.topk,
        randomk=args.randomk,
        seed=args.seed,
        max_tokens=args.max_tokens,
        preview_chars=args.preview_chars,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
