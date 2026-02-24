import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_bin(path: str, arr: np.ndarray) -> None:
    """Write raw token bytes atomically."""
    assert arr.dtype == np.uint16
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        arr.tofile(f)
    os.replace(tmp, path)


def _iter_tokenized_docs(
    ds,
    tokenizer,
    text_field: str,
    batch_docs: int,
    bos_token_id: int,
    max_docs: int,
    stats: Dict[str, int],
) -> Iterator[np.ndarray]:
    """Yield per-document uint16 arrays: [BOS] + token_ids."""
    batch_texts: List[str] = []

    def flush_batch() -> Iterator[np.ndarray]:
        nonlocal batch_texts
        if not batch_texts:
            return iter(())
        enc = tokenizer(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        to_yield = []
        for ids in enc["input_ids"]:
            arr = np.asarray([bos_token_id] + ids, dtype=np.int64)
            if arr.max(initial=0) >= 2**16 or arr.min(initial=0) < 0:
                raise ValueError("Token id out of uint16 range; expected GPT-2 token ids.")
            to_yield.append(arr.astype(np.uint16, copy=False))
        batch_texts = []
        return iter(to_yield)

    for ex in ds:
        text = ex.get(text_field, None)
        if text is None:
            raise KeyError(f"Example missing text field '{text_field}'. Keys: {list(ex.keys())}")
        batch_texts.append(text)
        stats["docs_processed"] += 1

        if len(batch_texts) >= batch_docs:
            for doc in flush_batch():
                yield doc

        if max_docs > 0 and stats["docs_processed"] >= max_docs:
            break

    if batch_texts:
        for doc in flush_batch():
            yield doc


def _find_largest_that_fits(doc_buffer: List[np.ndarray], remaining: int) -> int:
    best_idx = -1
    best_len = 0
    for i, doc in enumerate(doc_buffer):
        n = int(doc.size)
        if n <= remaining and n > best_len:
            best_idx = i
            best_len = n
    return best_idx


def _find_shortest(doc_buffer: List[np.ndarray]) -> int:
    shortest_idx = 0
    shortest_len = int(doc_buffer[0].size)
    for i in range(1, len(doc_buffer)):
        n = int(doc_buffer[i].size)
        if n < shortest_len:
            shortest_idx = i
            shortest_len = n
    return shortest_idx


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Pretokenize FineWeb/FineWeb-Edu into BOS-aligned row-packed uint16 .bin shards "
            "with nanochat-style best-fit + shortest-crop packing"
        )
    )
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu or HuggingFaceFW/fineweb")
    p.add_argument("--config", type=str, default="sample-10BT",
                   help="Dataset config/name, e.g. sample-10BT or sample-100BT")
    p.add_argument("--data_files", type=str, default=None,
                   help="Optional data files for local datasets (e.g. '/tmp/docs.jsonl'). If set, --config is ignored.")
    p.add_argument("--split", type=str, default="train", help="Dataset split (usually 'train')")
    p.add_argument("--text_field", type=str, default="text", help="Field containing raw text")
    p.add_argument("--out_dir", type=str, default="fineweb_edu_10B_bosrow", help="Output directory")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path")
    p.add_argument("--batch_docs", type=int, default=256, help="Docs per tokenizer batch")
    p.add_argument("--seq_len", type=int, default=1024, help="Training sequence length (rows are seq_len+1)")
    p.add_argument("--buffer_docs", type=int, default=1000, help="Best-fit candidate buffer size")
    p.add_argument("--shard_rows", type=int, default=0,
                   help="Rows per shard. 0 means auto-size to ~100M tokens/shard")
    p.add_argument("--val_shards", type=int, default=1, help="Number of initial shards to label as val")
    p.add_argument("--max_docs", type=int, default=0, help="If >0, stop after this many docs (debug)")
    args = p.parse_args()

    if args.seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if args.batch_docs <= 0:
        raise ValueError("batch_docs must be > 0")
    if args.buffer_docs <= 0:
        raise ValueError("buffer_docs must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos = int(tokenizer.eos_token_id)
    if bos is None:
        raise ValueError("Tokenizer has no eos_token_id; GPT-2 should have one (50256).")

    if args.data_files:
        # Local debug path: avoid streaming mode because some environments disallow shared-memory setup.
        ds = load_dataset(args.dataset, data_files=args.data_files, split=args.split, streaming=False)
    else:
        ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)

    row_tokens = args.seq_len + 1
    auto_rows = max(1, 100_000_000 // row_tokens)
    shard_rows = auto_rows if args.shard_rows <= 0 else int(args.shard_rows)
    shard_tokens = shard_rows * row_tokens

    # Flat buffer for writing full/partial shards.
    token_buf = np.empty((shard_tokens,), dtype=np.uint16)
    buf_rows = 0
    shard_idx = 0

    stats: Dict[str, int] = {
        "docs_processed": 0,
        "rows_written_total": 0,
        "tokens_written_total": 0,
        "tokens_cropped_total": 0,
    }

    def shard_split(i: int) -> str:
        return "val" if i < args.val_shards else "train"

    def shard_path(i: int) -> str:
        return os.path.join(args.out_dir, f"{shard_split(i)}_{i:06d}.bin")

    def flush_rows(num_rows: int) -> None:
        nonlocal shard_idx
        if num_rows <= 0:
            return
        end = num_rows * row_tokens
        arr = token_buf if num_rows == shard_rows else token_buf[:end].copy()
        atomic_write_bin(shard_path(shard_idx), arr)
        shard_idx += 1
        stats["rows_written_total"] += num_rows
        stats["tokens_written_total"] += end

    token_docs = _iter_tokenized_docs(
        ds=ds,
        tokenizer=tokenizer,
        text_field=args.text_field,
        batch_docs=args.batch_docs,
        bos_token_id=bos,
        max_docs=args.max_docs,
        stats=stats,
    )
    token_docs_iter = iter(token_docs)

    doc_buffer: List[np.ndarray] = []
    docs_exhausted = False

    def refill_buffer() -> None:
        nonlocal docs_exhausted
        while len(doc_buffer) < args.buffer_docs and not docs_exhausted:
            try:
                doc_buffer.append(next(token_docs_iter))
            except StopIteration:
                docs_exhausted = True
                break

    pbar = tqdm(unit="tok", desc="packing", dynamic_ncols=True)
    start_time = time.time()

    refill_buffer()
    while True:
        if not doc_buffer:
            break

        row = np.empty((row_tokens,), dtype=np.uint16)
        pos = 0

        while pos < row_tokens:
            if not doc_buffer:
                refill_buffer()
                if not doc_buffer:
                    break  # drop incomplete trailing row

            remaining = row_tokens - pos
            best_idx = _find_largest_that_fits(doc_buffer, remaining)

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                n = int(doc.size)
                row[pos:pos + n] = doc
                pos += n
            else:
                shortest_idx = _find_shortest(doc_buffer)
                doc = doc_buffer.pop(shortest_idx)
                n = int(doc.size)
                row[pos:pos + remaining] = doc[:remaining]
                stats["tokens_cropped_total"] += (n - remaining)
                pos += remaining

            if len(doc_buffer) < args.buffer_docs:
                refill_buffer()

        if pos < row_tokens:
            break

        row_offset = buf_rows * row_tokens
        token_buf[row_offset:row_offset + row_tokens] = row
        buf_rows += 1
        pbar.update(row_tokens)

        if buf_rows == shard_rows:
            flush_rows(buf_rows)
            buf_rows = 0

    pbar.close()

    if buf_rows > 0:
        flush_rows(buf_rows)

    elapsed = time.time() - start_time
    crop_fraction = (
        float(stats["tokens_cropped_total"]) /
        float(stats["tokens_cropped_total"] + stats["tokens_written_total"])
        if (stats["tokens_cropped_total"] + stats["tokens_written_total"]) > 0
        else 0.0
    )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "data_files": args.data_files,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "use_fast": True,
        "vocab_size": int(tokenizer.vocab_size),
        "dtype": "uint16",
        "format": "bos_row_packed_bestfit",
        "seq_len": int(args.seq_len),
        "row_tokens": int(row_tokens),
        "packing_algo": "largest_fit_then_shortest_crop",
        "batch_docs": int(args.batch_docs),
        "buffer_docs": int(args.buffer_docs),
        "bos_token_id": int(bos),
        "eos_token_id": int(tokenizer.eos_token_id),
        "bos_is_eos": True,
        "doc_format": "[BOS] + gpt2_bpe(text)",
        "shard_rows": int(shard_rows),
        "shard_tokens": int(shard_tokens),
        "val_shards": int(args.val_shards),
        "num_shards_total": int(shard_idx),
        "rows_written_total": int(stats["rows_written_total"]),
        "tokens_written_total": int(stats["tokens_written_total"]),
        "tokens_cropped_total": int(stats["tokens_cropped_total"]),
        "crop_fraction": float(crop_fraction),
        "docs_processed": int(stats["docs_processed"]),
        "elapsed_seconds": float(elapsed),
        "throughput_tokens_per_sec": float(stats["tokens_written_total"] / max(elapsed, 1e-9)),
        "notes": (
            "Rows are fixed length (seq_len+1), each row starts with BOS, and are packed with "
            "largest-fit then shortest-doc crop fallback."
        ),
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))

    print("\nDone.")
    print(f"  out_dir: {args.out_dir}")
    print(f"  shards:  {meta['num_shards_total']} (val_shards={args.val_shards})")
    print(f"  rows:    {meta['rows_written_total']:,}")
    print(f"  tokens:  {meta['tokens_written_total']:,}")
    print(f"  cropped: {meta['tokens_cropped_total']:,} ({100.0 * meta['crop_fraction']:.2f}%)")
    print(f"  docs:    {meta['docs_processed']:,}")
    print(f"  speed:   {meta['throughput_tokens_per_sec']:.0f} tok/s")


if __name__ == "__main__":
    main()
