#!/usr/bin/env python3
"""Mine discourse-tracking candidate pools from decoded training examples."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha1
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..attribution.common.checkpoints import load_tokenizer_from_checkpoint
from ..attribution.common.training_examples import (
    FiniteTrainingExampleDataset,
    PACKED_INDEX_FORMAT,
    PackedIndexView,
    build_example_manifest,
)
from .features import (
    build_rule_based_pools,
    cluster_promising_pool,
    compute_text_features,
    compute_spacy_doc_features,
    feature_columns,
    preview_text,
    resolve_backend,
)


REQUIRED_CANDIDATE_COLUMNS = (
    "candidate_id",
    "candidate_kind",
    "shard_path",
    "local_example_idx",
    "token_offset_start",
    "token_offset_end",
)


def _clean_decoded_text(text: str, *, bos_token: str | None, eos_token: str | None) -> str:
    cleaned = str(text)
    if bos_token:
        cleaned = cleaned.replace(bos_token, "\n[BOS]\n")
    if eos_token and eos_token != bos_token:
        cleaned = cleaned.replace(eos_token, "\n[EOS]\n")
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = cleaned.strip()
    if cleaned.startswith("[BOS]"):
        cleaned = cleaned.removeprefix("[BOS]").strip()
    return cleaned


def _load_candidate_frame(candidate_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(candidate_csv)
    if "group" in frame.columns:
        overall = frame.loc[frame["group"] == "overall"].copy()
        if not overall.empty:
            frame = overall
    missing = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Candidate CSV is missing required columns: {missing}")
    frame = frame.drop_duplicates(subset=["candidate_id"]).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(int)
    frame["token_count"] = frame["token_offset_end"].astype(int) - frame["token_offset_start"].astype(int)
    if frame["candidate_kind"].nunique() != 1:
        raise ValueError(
            "This miner expects one candidate_kind per run. "
            f"Found {sorted(str(value) for value in frame['candidate_kind'].unique().tolist())!r}"
        )
    return frame.reset_index(drop=True)


@lru_cache(maxsize=8)
def _load_data_meta(data_dir: str) -> dict[str, Any]:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found under {data_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=256)
def _memmap_shard(shard_path: str) -> np.memmap:
    return np.memmap(Path(shard_path).expanduser().resolve(), dtype=np.uint16, mode="r")


def _full_tokens_from_sample(sample: dict[str, Any]):
    import torch

    input_ids = sample["input_ids"]
    labels = sample["labels"]
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Expected tensor-valued input_ids and labels from FiniteTrainingExampleDataset")
    return torch.cat([input_ids, labels[-1:].clone()], dim=0)


def _infer_seq_len(frame: pd.DataFrame, seq_len: int | None) -> int:
    if seq_len is not None:
        return int(seq_len)
    token_counts = sorted(int(value) for value in frame["token_count"].dropna().unique().tolist())
    if len(token_counts) != 1:
        raise ValueError(
            "Could not infer one seq_len from candidate token counts. "
            f"Found token_count values {token_counts!r}; pass --seq_len explicitly."
        )
    token_count = int(token_counts[0])
    if token_count <= 1:
        raise ValueError(f"token_count must be > 1, got {token_count}")
    return token_count - 1


def _build_progress(*, enabled: bool, total: int, desc: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(total=total, desc=desc, unit="candidate", dynamic_ncols=True)


def _decode_candidate_texts_manifest(
    frame: pd.DataFrame,
    *,
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    seq_len: int | None,
    show_progress: bool,
) -> tuple[dict[int, str], dict[str, Any]]:
    resolved_seq_len = _infer_seq_len(frame, seq_len)
    candidate_kind = str(frame["candidate_kind"].iloc[0])
    manifest = build_example_manifest(
        data_dir,
        split="train",
        candidate_kind=candidate_kind,
        seq_len=resolved_seq_len,
    )
    candidate_ids = tuple(int(value) for value in frame["candidate_id"].tolist())
    dataset = FiniteTrainingExampleDataset(manifest, candidate_ids)
    tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)
    texts: dict[int, str] = {}
    progress = _build_progress(enabled=show_progress, total=len(candidate_ids), desc="Decoding candidates")
    try:
        for row_index, candidate_id in enumerate(candidate_ids):
            sample = dataset[row_index]
            full_tokens = _full_tokens_from_sample(sample)
            decoded = tokenizer.decode(full_tokens.tolist(), clean_up_tokenization_spaces=False)
            texts[candidate_id] = _clean_decoded_text(
                decoded,
                bos_token=tokenizer.bos_token,
                eos_token=tokenizer.eos_token,
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return texts, {
        "candidate_kind": str(manifest.candidate_kind),
        "seq_len": int(manifest.seq_len),
        "row_tokens": int(manifest.example_tokens),
        "data_dir": str(manifest.data_dir),
        "decode_strategy": "manifest",
    }


def _decode_candidate_tokens_sparse(
    *,
    record: dict[str, Any],
    data_dir: Path,
    meta: dict[str, Any],
    packed_index_view: PackedIndexView | None,
) -> np.ndarray:
    data_format = str(meta.get("format", ""))
    if data_format == PACKED_INDEX_FORMAT:
        if packed_index_view is None:
            raise RuntimeError("Packed-index sparse decoding requires a PackedIndexView")
        split_name = Path(str(record["shard_path"])).name.split("_", 1)[0]
        tokens = packed_index_view.reconstruct_row(split_name, int(record["candidate_id"]))
        return np.asarray(tokens, dtype=np.int64)

    shard_path = str(Path(str(record["shard_path"])).expanduser().resolve())
    mm = _memmap_shard(shard_path)
    start = int(record["token_offset_start"])
    end = int(record["token_offset_end"])
    return np.asarray(mm[start:end], dtype=np.int64)


def _decode_candidate_texts_sparse(
    frame: pd.DataFrame,
    *,
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    seq_len: int | None,
    show_progress: bool,
) -> tuple[dict[int, str], dict[str, Any]]:
    data_path = Path(data_dir).expanduser().resolve()
    meta = _load_data_meta(str(data_path))
    resolved_seq_len = _infer_seq_len(frame, seq_len)
    candidate_kind = str(frame["candidate_kind"].iloc[0])
    tokenizer = load_tokenizer_from_checkpoint(checkpoint_dir)
    packed_index_view = PackedIndexView(data_path) if str(meta.get("format", "")) == PACKED_INDEX_FORMAT else None

    texts: dict[int, str] = {}
    progress = _build_progress(enabled=show_progress, total=len(frame), desc="Decoding candidates")
    try:
        for record in frame.to_dict(orient="records"):
            candidate_id = int(record["candidate_id"])
            tokens = _decode_candidate_tokens_sparse(
                record=record,
                data_dir=data_path,
                meta=meta,
                packed_index_view=packed_index_view,
            )
            decoded = tokenizer.decode(tokens.tolist(), clean_up_tokenization_spaces=False)
            texts[candidate_id] = _clean_decoded_text(
                decoded,
                bos_token=tokenizer.bos_token,
                eos_token=tokenizer.eos_token,
            )
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    example_tokens = (
        int(meta["row_tokens"])
        if "row_tokens" in meta
        else int(resolved_seq_len) + 1
    )
    return texts, {
        "candidate_kind": candidate_kind,
        "seq_len": int(resolved_seq_len),
        "row_tokens": int(example_tokens),
        "data_dir": str(data_path),
        "decode_strategy": "sparse",
    }


def _decode_candidate_texts(
    frame: pd.DataFrame,
    *,
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    seq_len: int | None,
    show_progress: bool,
    decode_strategy: str,
) -> tuple[dict[int, str], dict[str, Any]]:
    resolved = str(decode_strategy).strip().lower()
    if resolved not in {"auto", "sparse", "manifest"}:
        raise ValueError(f"Unsupported decode_strategy={decode_strategy!r}; expected auto, sparse, or manifest")
    if resolved == "manifest":
        return _decode_candidate_texts_manifest(
            frame,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            seq_len=seq_len,
            show_progress=show_progress,
        )
    if resolved in {"auto", "sparse"}:
        return _decode_candidate_texts_sparse(
            frame,
            data_dir=data_dir,
            checkpoint_dir=checkpoint_dir,
            seq_len=seq_len,
            show_progress=show_progress,
        )
    raise AssertionError("unreachable")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decode candidate training spans, score discourse-tracking features, "
            "and export interpretable positive/random/negative pools."
        )
    )
    parser.add_argument("--candidate_csv", required=True, help="CSV with candidate metadata, usually row_summary_stepXXXXXXXX.csv")
    parser.add_argument("--data_dir", required=True, help="Training data directory backing those candidates")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint/tokenizer directory used to decode tokens")
    parser.add_argument("--output_dir", required=True, help="Where to write features and pool exports")
    parser.add_argument("--seq_len", type=int, default=None, help="Optional explicit seq_len override")
    parser.add_argument(
        "--decode_strategy",
        choices=("auto", "sparse", "manifest"),
        default="auto",
        help="How to decode candidate text. 'sparse' decodes directly from candidate shard/offset metadata; 'manifest' builds the full example manifest first.",
    )
    parser.add_argument("--max_candidates", type=int, default=0, help="Deterministically subsample candidates before decoding")
    parser.add_argument("--sample_seed", type=int, default=42, help="Seed for deterministic candidate subsampling and random pool selection")
    parser.add_argument("--parser_backend", choices=("auto", "spacy", "regex"), default="auto")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    parser.add_argument("--embedding_backend", choices=("none", "tfidf_svd", "sentence_transformers"), default="none")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--n_process",
        type=int,
        default=1,
        help="Number of CPU worker processes for feature scoring. spaCy uses nlp.pipe(..., n_process=N); regex mode uses a process pool.",
    )
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument("--min_cluster_size", type=int, default=8)
    parser.add_argument("--min_sentences", type=int, default=3)
    parser.add_argument("--min_entities", type=int, default=2)
    parser.add_argument("--min_text_tokens", type=int, default=96)
    parser.add_argument("--no_progress", action="store_false", dest="show_progress")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _print_backend_banner(backend_info) -> None:
    print(
        "parser backend: "
        f"{backend_info.used} "
        f"(requested={backend_info.requested}; detail={backend_info.detail})"
    )


def _build_feature_row(
    record: dict[str, Any],
    *,
    text: str,
    features: dict[str, Any],
) -> dict[str, Any]:
    return {
        **record,
        **features,
        "text_sha1": sha1(text.encode("utf-8")).hexdigest(),
        "text_preview": preview_text(text),
    }


def _compute_feature_row_regex_worker(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    record, text = payload
    features = compute_text_features(
        text,
        parser_backend="regex",
    )
    return _build_feature_row(record, text=text, features=features)


def _score_feature_rows(
    *,
    frame: pd.DataFrame,
    texts: dict[int, str],
    parser_backend: str,
    spacy_model: str,
    nlp,
    n_process: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    progress = _build_progress(enabled=bool(show_progress), total=len(records), desc="Scoring discourse features")
    try:
        if parser_backend == "spacy":
            if nlp is None:
                raise RuntimeError("spaCy backend selected but no spaCy pipeline is available")
            ordered_texts = [texts[int(record["candidate_id"])] for record in records]
            feature_rows: list[dict[str, Any]] = []
            docs = nlp.pipe(
                ordered_texts,
                n_process=max(1, int(n_process)),
            )
            for record, text, doc in zip(records, ordered_texts, docs):
                feature_rows.append(
                    _build_feature_row(
                        record,
                        text=text,
                        features=compute_spacy_doc_features(doc),
                    )
                )
                if progress is not None:
                    progress.update(1)
            return feature_rows

        if int(n_process) <= 1:
            feature_rows = []
            for record in records:
                text = texts[int(record["candidate_id"])]
                feature_rows.append(
                    _build_feature_row(
                        record,
                        text=text,
                        features=compute_text_features(
                            text,
                            parser_backend=parser_backend,
                            spacy_model=spacy_model,
                            nlp=nlp,
                        ),
                    )
                )
                if progress is not None:
                    progress.update(1)
            return feature_rows

        tasks = [(record, texts[int(record["candidate_id"])]) for record in records]
        feature_rows = []
        chunksize = max(1, min(64, len(tasks) // max(1, int(n_process) * 4)))
        with ProcessPoolExecutor(max_workers=int(n_process)) as executor:
            for row in executor.map(_compute_feature_row_regex_worker, tasks, chunksize=chunksize):
                feature_rows.append(row)
                if progress is not None:
                    progress.update(1)
        return feature_rows
    finally:
        if progress is not None:
            progress.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if int(args.n_process) <= 0:
        raise ValueError("--n_process must be > 0")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _load_candidate_frame(args.candidate_csv)
    if int(args.max_candidates) > 0 and len(frame) > int(args.max_candidates):
        frame = (
            frame.sample(n=int(args.max_candidates), random_state=int(args.sample_seed), replace=False)
            .sort_values("candidate_id")
            .reset_index(drop=True)
        )

    texts, decode_info = _decode_candidate_texts(
        frame,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len,
        show_progress=bool(args.show_progress),
        decode_strategy=str(args.decode_strategy),
    )
    parser_backend, nlp, backend_info = resolve_backend(
        parser_backend=args.parser_backend,
        spacy_model=args.spacy_model,
    )
    _print_backend_banner(backend_info)
    if int(args.n_process) > 1:
        print(f"feature scoring workers: {int(args.n_process)}")

    feature_rows = _score_feature_rows(
        frame=frame,
        texts=texts,
        parser_backend=parser_backend,
        spacy_model=args.spacy_model,
        nlp=nlp,
        n_process=int(args.n_process),
        show_progress=bool(args.show_progress),
    )

    feature_frame = pd.DataFrame.from_records(feature_rows)
    feature_frame, pool_summary = build_rule_based_pools(
        feature_frame,
        seed=int(args.sample_seed),
        min_sentences=int(args.min_sentences),
        min_entities=int(args.min_entities),
        min_text_tokens=int(args.min_text_tokens),
    )
    feature_frame = feature_frame.sort_values(["priority_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    cluster_summary_payload: dict[str, Any] = {"ran": False, "reason": "embedding_backend=none"}
    if args.embedding_backend != "none":
        clustered_positive, cluster_summary_frame, cluster_summary_payload = cluster_promising_pool(
            feature_frame,
            text_lookup=texts,
            embedding_backend=args.embedding_backend,
            embedding_model=args.embedding_model,
            num_clusters=int(args.num_clusters),
            min_cluster_size=int(args.min_cluster_size),
            seed=int(args.sample_seed),
        )
        if cluster_summary_payload.get("ran"):
            cluster_assignments = clustered_positive[
                ["candidate_id", "cluster_id", "is_selected_cluster", "priority_score", "text_preview"]
            ].copy()
            cluster_assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)
            cluster_summary_frame.to_csv(output_dir / "cluster_summary.csv", index=False)
            selected_cluster_ids = set(
                int(value) for value in cluster_assignments.loc[cluster_assignments["is_selected_cluster"], "candidate_id"].tolist()
            )
            feature_frame["is_selected_cluster_pool"] = feature_frame["candidate_id"].astype(int).isin(selected_cluster_ids)
        else:
            feature_frame["is_selected_cluster_pool"] = False
    else:
        feature_frame["is_selected_cluster_pool"] = False

    feature_frame.to_csv(output_dir / "candidate_features.csv", index=False)

    text_rows = [
        {
            "candidate_id": int(candidate_id),
            "text_sha1": sha1(text.encode("utf-8")).hexdigest(),
            "text_preview": preview_text(text),
            "text": text,
        }
        for candidate_id, text in texts.items()
    ]
    _write_jsonl(output_dir / "candidate_text.jsonl", text_rows)

    pool_specs = {
        "positive_pool": feature_frame["is_positive_pool"],
        "random_control_pool": feature_frame["is_random_control_pool"],
        "negative_low_binding_pool": feature_frame["is_negative_low_binding_pool"],
        "negative_repetition_pool": feature_frame["is_negative_repetition_pool"],
        "selected_cluster_pool": feature_frame["is_selected_cluster_pool"],
    }
    text_by_id = {int(row["candidate_id"]): row["text"] for row in text_rows}
    for label, mask in pool_specs.items():
        subset = feature_frame.loc[mask].copy()
        subset.to_csv(output_dir / f"{label}.csv", index=False)
        _write_jsonl(
            output_dir / f"{label}.jsonl",
            [
                {
                    **record,
                    "text": text_by_id[int(record["candidate_id"])],
                }
                for record in subset.to_dict(orient="records")
            ],
        )

    summary = {
        "candidate_csv": str(Path(args.candidate_csv).expanduser().resolve()),
        "output_dir": str(output_dir),
        "decode": decode_info,
        "parser_backend": backend_info.to_json(),
        "feature_columns": list(feature_columns()),
        "pool_summary": pool_summary,
        "cluster_summary": cluster_summary_payload,
        "artifacts": {
            "candidate_features": str(output_dir / "candidate_features.csv"),
            "candidate_text": str(output_dir / "candidate_text.jsonl"),
            "positive_pool": str(output_dir / "positive_pool.csv"),
            "random_control_pool": str(output_dir / "random_control_pool.csv"),
            "negative_low_binding_pool": str(output_dir / "negative_low_binding_pool.csv"),
            "negative_repetition_pool": str(output_dir / "negative_repetition_pool.csv"),
            "selected_cluster_pool": str(output_dir / "selected_cluster_pool.csv"),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
