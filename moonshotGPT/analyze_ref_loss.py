import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from transformers import AutoTokenizer


def atomic_write_json(path: str, payload: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _infer_ref_dtype(path: str):
    if path.endswith(".f16.bin"):
        return np.float16
    if path.endswith(".f32.bin"):
        return np.float32
    # fallback for unknown naming
    return np.float16


def _list_ref_files(ref_loss_dir: str, split: str) -> List[str]:
    out = []
    for name in sorted(os.listdir(ref_loss_dir)):
        if not name.startswith(f"{split}_"):
            continue
        if ".ref_loss." not in name or not name.endswith(".bin"):
            continue
        out.append(os.path.join(ref_loss_dir, name))
    return out


def _source_shard_for_ref(data_dir: str, ref_loss_path: str) -> str:
    name = os.path.basename(ref_loss_path)
    marker = ".ref_loss."
    i = name.find(marker)
    if i < 0:
        raise ValueError(f"Unexpected ref-loss filename: {name}")
    stem = name[:i]
    return os.path.join(data_dir, stem + ".bin")


def _meta_path_for_ref(ref_loss_path: str) -> str:
    if not ref_loss_path.endswith(".bin"):
        raise ValueError(f"Expected .bin file: {ref_loss_path}")
    return ref_loss_path[:-4] + ".meta.json"


def _load_meta_or_none(ref_loss_path: str) -> Dict:
    mp = _meta_path_for_ref(ref_loss_path)
    if not os.path.exists(mp):
        return {}
    with open(mp, "r") as f:
        return json.load(f)


def _sample_finite_values(mm: np.memmap, sample_n: int, rng: np.random.Generator) -> np.ndarray:
    if sample_n <= 0:
        return np.empty((0,), dtype=np.float32)
    n = int(mm.shape[0])
    if n <= 0:
        return np.empty((0,), dtype=np.float32)

    draw = min(n, max(sample_n * 2, sample_n + 1024))
    idx = rng.integers(0, n, size=draw, endpoint=False)
    vals = np.asarray(mm[idx], dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size > sample_n:
        vals = vals[:sample_n]
    return vals


def _compute_position_stats(
    ref_mm: np.memmap,
    covered_tokens: int,
    seq_len: int,
    pos_sum: np.ndarray,
    pos_count: np.ndarray,
    chunk_tokens: int,
) -> None:
    if covered_tokens <= 0:
        return
    start = 0
    while start < covered_tokens:
        end = min(covered_tokens, start + chunk_tokens)
        vals = np.asarray(ref_mm[1 + start : 1 + end], dtype=np.float32)
        finite = np.isfinite(vals)
        if finite.any():
            idx = np.arange(start, end, dtype=np.int64)
            pos = (idx % seq_len).astype(np.int64)
            pos_f = pos[finite]
            vals_f = vals[finite]
            np.add.at(pos_sum, pos_f, vals_f)
            np.add.at(pos_count, pos_f, 1)
        start = end


def _compute_bos_nonbos_stats(
    src_mm: np.memmap,
    ref_mm: np.memmap,
    covered_tokens: int,
    bos_id: int,
    chunk_tokens: int,
) -> Tuple[float, int, float, int]:
    bos_sum = 0.0
    bos_count = 0
    non_sum = 0.0
    non_count = 0

    if covered_tokens <= 0:
        return bos_sum, bos_count, non_sum, non_count

    start = 0
    while start < covered_tokens:
        end = min(covered_tokens, start + chunk_tokens)
        loss_vals = np.asarray(ref_mm[1 + start : 1 + end], dtype=np.float32)
        tok_vals = np.asarray(src_mm[1 + start : 1 + end], dtype=np.int64)

        finite = np.isfinite(loss_vals)
        if finite.any():
            lv = loss_vals[finite]
            tv = tok_vals[finite]
            bos_mask = (tv == bos_id)
            if bos_mask.any():
                bos_sum += float(lv[bos_mask].sum())
                bos_count += int(bos_mask.sum())
            non_mask = ~bos_mask
            if non_mask.any():
                non_sum += float(lv[non_mask].sum())
                non_count += int(non_mask.sum())

        start = end

    return bos_sum, bos_count, non_sum, non_count


def _write_tail_csv(
    out_csv: str,
    rows: List[Dict],
) -> None:
    if not rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["split", "ref_file", "source_shard", "token_index", "token_id", "is_bos", "ref_loss", "left_context", "center_token", "right_context"],
            )
            w.writeheader()
        return

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["split", "ref_file", "source_shard", "token_index", "token_id", "is_bos", "ref_loss", "left_context", "center_token", "right_context"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _render_plots(
    analysis_dir: str,
    sample_vals: np.ndarray,
    percentile_map: Dict[str, float],
    pos_mean: np.ndarray,
    bos_mean: float,
    nonbos_mean: float,
) -> None:
    if plt is None:
        return

    os.makedirs(analysis_dir, exist_ok=True)

    if sample_vals.size > 0:
        plt.figure(figsize=(10, 5))
        plt.hist(sample_vals, bins=200, color="#2a6f97", alpha=0.9)
        plt.xlabel("Reference Loss")
        plt.ylabel("Count")
        plt.title("Reference Loss Distribution (sample)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "ref_loss_hist.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.hist(sample_vals, bins=200, color="#2a6f97", alpha=0.9, log=True)
        plt.xlabel("Reference Loss")
        plt.ylabel("Count (log)")
        plt.title("Reference Loss Distribution (sample, log-y)")
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "ref_loss_hist_logy.png"))
        plt.close()

        xs = np.sort(sample_vals)
        ys = (np.arange(xs.size, dtype=np.float64) + 1.0) / float(xs.size)

        plt.figure(figsize=(10, 5))
        plt.plot(xs, ys, color="#1b4332", linewidth=2)
        plt.xlabel("Reference Loss")
        plt.ylabel("CDF")
        plt.title("Reference Loss CDF (sample)")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "ref_loss_cdf_full.png"))
        plt.close()

        p10 = percentile_map.get("p10", None)
        if p10 is not None:
            m = xs <= p10
            if m.any():
                plt.figure(figsize=(10, 5))
                plt.plot(xs[m], ys[m], color="#386641", linewidth=2)
                plt.xlabel("Reference Loss")
                plt.ylabel("CDF")
                plt.title("Reference Loss CDF (0-10th percentile region)")
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, "ref_loss_cdf_low_tail_0_10.png"))
                plt.close()

        p90 = percentile_map.get("p90", None)
        if p90 is not None:
            m = xs >= p90
            if m.any():
                plt.figure(figsize=(10, 5))
                plt.plot(xs[m], ys[m], color="#9d0208", linewidth=2)
                plt.xlabel("Reference Loss")
                plt.ylabel("CDF")
                plt.title("Reference Loss CDF (90-100th percentile region)")
                plt.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig(os.path.join(analysis_dir, "ref_loss_cdf_high_tail_90_100.png"))
                plt.close()

    if pos_mean.size > 0:
        plt.figure(figsize=(11, 4.5))
        plt.plot(np.arange(pos_mean.size), pos_mean, color="#277da1", linewidth=1.5)
        plt.xlabel("Position in sequence (0..T-1)")
        plt.ylabel("Mean reference loss")
        plt.title("Reference Loss by Sequence Position")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "ref_loss_by_position.png"))
        plt.close()

    if np.isfinite(bos_mean) and np.isfinite(nonbos_mean):
        plt.figure(figsize=(6, 4))
        labels = ["BOS/EOS token", "Non-BOS token"]
        vals = [bos_mean, nonbos_mean]
        plt.bar(labels, vals, color=["#577590", "#43aa8b"])
        plt.ylabel("Mean reference loss")
        plt.title("Reference Loss: BOS vs Non-BOS")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, "ref_loss_bos_vs_nonbos.png"))
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze precomputed reference-token losses")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with source train_*.bin / val_*.bin shards")
    p.add_argument("--ref_loss_dir", type=str, required=True, help="Directory with *.ref_loss.*.bin files")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to analyze")
    p.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer for decoded examples")
    p.add_argument("--seq_len", type=int, default=0, help="Override seq_len; 0 means read from ref meta")
    p.add_argument("--sample_tokens", type=int, default=5_000_000, help="Total sampled loss values for distribution estimates")
    p.add_argument("--tail_examples", type=int, default=128, help="How many low/high-loss examples to export")
    p.add_argument("--context_window", type=int, default=24, help="Token window on each side for tail examples")
    p.add_argument("--chunk_tokens", type=int, default=2_000_000, help="Chunk size for streaming stats")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_files", type=int, default=0, help="If >0, cap files analyzed (debug)")
    p.add_argument("--analysis_dir", type=str, default=None, help="Output analysis directory")
    args = p.parse_args()

    ref_files = _list_ref_files(args.ref_loss_dir, args.split)
    if args.max_files > 0:
        ref_files = ref_files[: args.max_files]

    if not ref_files:
        raise FileNotFoundError(f"No ref-loss files found in {args.ref_loss_dir} for split={args.split}")

    analysis_dir = args.analysis_dir or os.path.join(args.ref_loss_dir, f"analysis_{args.split}")
    os.makedirs(analysis_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos_id = int(tokenizer.eos_token_id)

    rng = np.random.default_rng(args.seed)

    total_tokens = 0
    total_finite_est = 0
    total_covered_tokens = 0

    sample_vals_all = []

    seq_len = args.seq_len
    batch_size = None

    pos_sum = None
    pos_count = None

    bos_sum_total = 0.0
    bos_count_total = 0
    non_sum_total = 0.0
    non_count_total = 0

    tail_candidates_low = []
    tail_candidates_high = []

    per_file_sample_target = max(1, args.sample_tokens // max(1, len(ref_files)))

    for ref_path in ref_files:
        ref_dtype = _infer_ref_dtype(ref_path)
        ref_mm = np.memmap(ref_path, dtype=ref_dtype, mode="r")
        n_tokens = int(ref_mm.shape[0])
        total_tokens += n_tokens

        meta = _load_meta_or_none(ref_path)
        file_seq_len = int(meta.get("seq_len", 0))
        file_batch_size = int(meta.get("batch_size", 0))
        covered_tokens = int(meta.get("covered_target_tokens", max(0, n_tokens - 1)))
        total_covered_tokens += covered_tokens

        if seq_len <= 0:
            seq_len = file_seq_len if file_seq_len > 0 else 1024
        if batch_size is None and file_batch_size > 0:
            batch_size = file_batch_size

        if pos_sum is None:
            pos_sum = np.zeros((seq_len,), dtype=np.float64)
            pos_count = np.zeros((seq_len,), dtype=np.int64)

        sampled = _sample_finite_values(ref_mm, per_file_sample_target, rng)
        if sampled.size > 0:
            total_finite_est += int(sampled.size)
            sample_vals_all.append(sampled)

        src_path = _source_shard_for_ref(args.data_dir, ref_path)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing source shard for {ref_path}: {src_path}")
        src_mm = np.memmap(src_path, dtype=np.uint16, mode="r")

        _compute_position_stats(
            ref_mm=ref_mm,
            covered_tokens=covered_tokens,
            seq_len=seq_len,
            pos_sum=pos_sum,
            pos_count=pos_count,
            chunk_tokens=args.chunk_tokens,
        )

        bs, bc, ns, nc = _compute_bos_nonbos_stats(
            src_mm=src_mm,
            ref_mm=ref_mm,
            covered_tokens=covered_tokens,
            bos_id=bos_id,
            chunk_tokens=args.chunk_tokens,
        )
        bos_sum_total += bs
        bos_count_total += bc
        non_sum_total += ns
        non_count_total += nc

        # Build tail candidates from random draws in covered region.
        # We draw more than needed then keep best/worst globally.
        if covered_tokens > 0 and args.tail_examples > 0:
            draw = min(covered_tokens, max(args.tail_examples * 8, 2048))
            idx = rng.integers(1, covered_tokens + 1, size=draw, endpoint=False)
            vals = np.asarray(ref_mm[idx], dtype=np.float32)
            finite = np.isfinite(vals)
            idx = idx[finite]
            vals = vals[finite]
            if idx.size > 0:
                for ii, vv in zip(idx.tolist(), vals.tolist()):
                    tail_candidates_low.append((float(vv), int(ii), ref_path, src_path))
                    tail_candidates_high.append((float(vv), int(ii), ref_path, src_path))

    if sample_vals_all:
        sample_vals = np.concatenate(sample_vals_all).astype(np.float32, copy=False)
    else:
        sample_vals = np.empty((0,), dtype=np.float32)

    if sample_vals.size > args.sample_tokens:
        sample_vals = sample_vals[: args.sample_tokens]

    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    percentile_map = {}
    if sample_vals.size > 0:
        for pctl in percentiles:
            percentile_map[f"p{pctl}"] = float(np.percentile(sample_vals, pctl))

    pos_mean = np.full((seq_len,), np.nan, dtype=np.float64)
    if pos_sum is not None and pos_count is not None:
        valid = pos_count > 0
        pos_mean[valid] = pos_sum[valid] / pos_count[valid]

    bos_mean = float(bos_sum_total / bos_count_total) if bos_count_total > 0 else float("nan")
    nonbos_mean = float(non_sum_total / non_count_total) if non_count_total > 0 else float("nan")

    # Final low/high examples.
    tail_candidates_low.sort(key=lambda x: x[0])
    tail_candidates_high.sort(key=lambda x: x[0], reverse=True)
    low_pick = tail_candidates_low[: args.tail_examples]
    high_pick = tail_candidates_high[: args.tail_examples]

    def materialize_tail_rows(cands: List[Tuple[float, int, str, str]]) -> List[Dict]:
        rows = []
        for loss_val, tok_idx, ref_path, src_path in cands:
            src_mm = np.memmap(src_path, dtype=np.uint16, mode="r")
            token_id = int(src_mm[tok_idx])
            lo = max(0, tok_idx - args.context_window)
            hi = min(int(src_mm.shape[0]), tok_idx + args.context_window + 1)
            window = np.asarray(src_mm[lo:hi], dtype=np.int64)
            center = tok_idx - lo
            left_ids = window[:center].tolist()
            center_id = [int(window[center])] if 0 <= center < window.size else []
            right_ids = window[center + 1 :].tolist() if 0 <= center < window.size else []

            rows.append(
                {
                    "split": args.split,
                    "ref_file": os.path.basename(ref_path),
                    "source_shard": os.path.basename(src_path),
                    "token_index": int(tok_idx),
                    "token_id": token_id,
                    "is_bos": bool(token_id == bos_id),
                    "ref_loss": float(loss_val),
                    "left_context": tokenizer.decode(left_ids, skip_special_tokens=False),
                    "center_token": tokenizer.decode(center_id, skip_special_tokens=False),
                    "right_context": tokenizer.decode(right_ids, skip_special_tokens=False),
                }
            )
        return rows

    low_rows = materialize_tail_rows(low_pick)
    high_rows = materialize_tail_rows(high_pick)

    _write_tail_csv(os.path.join(analysis_dir, "tail_examples_low.csv"), low_rows)
    _write_tail_csv(os.path.join(analysis_dir, "tail_examples_high.csv"), high_rows)

    _render_plots(
        analysis_dir=analysis_dir,
        sample_vals=sample_vals,
        percentile_map=percentile_map,
        pos_mean=pos_mean,
        bos_mean=bos_mean,
        nonbos_mean=nonbos_mean,
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "ref_loss_dir": args.ref_loss_dir,
        "analysis_dir": analysis_dir,
        "split": args.split,
        "num_ref_files": len(ref_files),
        "total_tokens": int(total_tokens),
        "total_covered_tokens": int(total_covered_tokens),
        "sample_tokens_effective": int(sample_vals.size),
        "sample_finite_values": int(total_finite_est),
        "seq_len": int(seq_len),
        "batch_size_from_meta": int(batch_size) if batch_size is not None else None,
        "bos_token_id": int(bos_id),
        "percentiles": percentile_map,
        "mean_loss_sample": float(sample_vals.mean()) if sample_vals.size > 0 else None,
        "std_loss_sample": float(sample_vals.std()) if sample_vals.size > 0 else None,
        "bos_mean_loss": None if not np.isfinite(bos_mean) else float(bos_mean),
        "nonbos_mean_loss": None if not np.isfinite(nonbos_mean) else float(nonbos_mean),
        "plots_generated": bool(plt is not None),
        "tail_examples_low_csv": os.path.join(analysis_dir, "tail_examples_low.csv"),
        "tail_examples_high_csv": os.path.join(analysis_dir, "tail_examples_high.csv"),
    }

    atomic_write_json(os.path.join(analysis_dir, "ref_loss_analysis_summary.json"), summary)
    print(f"Wrote analysis summary to {os.path.join(analysis_dir, 'ref_loss_analysis_summary.json')}")


if __name__ == "__main__":
    main()
