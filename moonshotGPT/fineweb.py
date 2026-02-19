import os
import json
import time
import argparse
from datetime import datetime, timezone

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


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


def main():
    p = argparse.ArgumentParser(description="Pretokenize FineWeb/FineWeb-Edu to raw uint16 .bin shards (GPT-2 tokenizer)")
    p.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu or HuggingFaceFW/fineweb")
    p.add_argument("--config", type=str, default="sample-10BT",
                   help="Dataset config/name, e.g. sample-10BT or sample-100BT")
    p.add_argument("--split", type=str, default="train", help="Dataset split (usually 'train')")
    p.add_argument("--text_field", type=str, default="text", help="Field containing raw text")
    p.add_argument("--out_dir", type=str, default="fineweb_edu_10B", help="Output directory")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name/path")
    p.add_argument("--batch_docs", type=int, default=256, help="Docs per tokenizer batch (speed knob)")
    p.add_argument("--shard_tokens", type=int, default=100_000_000, help="Tokens per shard (uint16). 100M ~= 200MB")
    p.add_argument("--val_shards", type=int, default=1, help="Number of initial shards to label as 'val'")
    p.add_argument("--max_docs", type=int, default=0, help="If >0, stop after this many docs (debug)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Fast GPT-2 tokenizer (HF)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # GPT-2 uses the same id for BOS/EOS conceptually
    BOS = int(tok.eos_token_id)
    vocab_size = int(tok.vocab_size)

    if BOS is None:
        raise ValueError("Tokenizer has no eos_token_id; GPT-2 should have one (50256).")

    # Streaming dataset iterator (no giant download into RAM)
    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)

    # Shard buffer
    shard_tokens = int(args.shard_tokens)
    buf = np.empty((shard_tokens,), dtype=np.uint16)
    buf_len = 0
    shard_idx = 0

    total_tokens_written = 0
    total_docs = 0

    def shard_split(i: int) -> str:
        return "val" if i < args.val_shards else "train"

    def shard_path(i: int) -> str:
        split = shard_split(i)
        # name shards with split and an absolute index (like Karpathy style, but raw tokens)
        return os.path.join(args.out_dir, f"{split}_{i:06d}.bin")

    def flush_full_shard():
        nonlocal shard_idx, buf_len, total_tokens_written
        assert buf_len == shard_tokens
        path = shard_path(shard_idx)
        atomic_write_bin(path, buf)
        shard_idx += 1
        total_tokens_written += shard_tokens
        # reset
        return 0

    # Progress bar: tokens written (no known total, but still useful)
    pbar = tqdm(unit="tok", desc="writing", dynamic_ncols=True)

    batch_texts = []

    def process_token_ids(ids_list):
        """Take a list[int] token ids for one doc, prepend BOS, and stream into shards."""
        nonlocal buf_len, shard_idx, total_tokens_written

        # Prepend BOS
        # Convert to uint16 array once, then stream slices into buf
        arr = np.asarray([BOS] + ids_list, dtype=np.int64)
        # Safety: GPT-2 ids must fit uint16
        if arr.max(initial=0) >= 2**16 or arr.min(initial=0) < 0:
            raise ValueError("Token id out of uint16 range; are you sure this is GPT-2 tokenization?")
        arr = arr.astype(np.uint16, copy=False)

        offset = 0
        n = int(arr.size)
        while offset < n:
            space = shard_tokens - buf_len
            take = min(space, n - offset)
            buf[buf_len:buf_len + take] = arr[offset:offset + take]
            buf_len += take
            offset += take

            pbar.update(take)

            if buf_len == shard_tokens:
                buf_len = flush_full_shard()

    # Main loop
    start_time = time.time()

    for ex in ds:
        text = ex.get(args.text_field, None)
        if text is None:
            raise KeyError(f"Example missing text field '{args.text_field}'. Keys: {list(ex.keys())}")

        batch_texts.append(text)
        total_docs += 1

        if args.max_docs and total_docs >= args.max_docs:
            # process what we have and stop
            pass

        if len(batch_texts) >= args.batch_docs or (args.max_docs and total_docs >= args.max_docs):
            enc = tok(
                batch_texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            for ids in enc["input_ids"]:
                process_token_ids(ids)

            batch_texts.clear()

        if args.max_docs and total_docs >= args.max_docs:
            break

    # Process remaining partial batch
    if batch_texts:
        enc = tok(
            batch_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for ids in enc["input_ids"]:
            process_token_ids(ids)
        batch_texts.clear()

    pbar.close()

    # Flush final partial shard (if any)
    if buf_len > 0:
        path = shard_path(shard_idx)
        atomic_write_bin(path, buf[:buf_len].copy())
        total_tokens_written += buf_len
        shard_idx += 1

    elapsed = time.time() - start_time

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer": args.tokenizer,
        "use_fast": True,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "bos_token_id": BOS,
        "eos_token_id": int(tok.eos_token_id),
        "bos_is_eos": True,
        "doc_format": "[BOS] + gpt2_bpe(text)",  # BOS per document, no trailing delimiter needed
        "shard_tokens": shard_tokens,
        "val_shards": int(args.val_shards),
        "num_shards_total": int(shard_idx),
        "tokens_written_total": int(total_tokens_written),
        "docs_processed": int(total_docs),
        "elapsed_seconds": float(elapsed),
        "throughput_tokens_per_sec": float(total_tokens_written / max(elapsed, 1e-9)),
        "notes": "Raw .bin shards contain a single 1D stream of uint16 token IDs; each doc is prepended with BOS (=50256).",
    }

    meta_path = os.path.join(args.out_dir, "meta.json")
    atomic_write_bytes(meta_path, json.dumps(meta, indent=2).encode("utf-8"))

    print("\nDone.")
    print(f"  out_dir: {args.out_dir}")
    print(f"  shards:  {meta['num_shards_total']} (val_shards={args.val_shards})")
    print(f"  tokens:  {meta['tokens_written_total']:,}")
    print(f"  docs:    {meta['docs_processed']:,}")
    print(f"  speed:   {meta['throughput_tokens_per_sec']:.0f} tok/s")


if __name__ == "__main__":
    main()
