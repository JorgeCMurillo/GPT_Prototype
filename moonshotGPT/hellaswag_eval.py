#!/usr/bin/env python3
"""Evaluate a Hugging Face causal language model on HellaSwag.

Usage examples:
  python hellaswag_eval.py --model gpt2
  python hellaswag_eval.py --model /path/to/local/checkpoint --batch-size 32 --max-examples 1000
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on HellaSwag")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="HF model ID or local model path")
    group.add_argument(
        "--model-dir",
        help="Local directory containing checkpoint folders (e.g., ckpt_periodic_step*)",
    )
    parser.add_argument(
        "--model-pattern",
        default="ckpt_*",
        help="Glob pattern used with --model-dir to find checkpoint subfolders",
    )
    parser.add_argument(
        "--dataset",
        default="hellaswag",
        help="Dataset name or local dataset path for datasets.load_dataset (default: hellaswag)",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config name passed to datasets.load_dataset",
    )
    parser.add_argument("--split", default="validation", help="HellaSwag split (default: validation)")
    parser.add_argument("--batch-size", type=int, default=16, help="Candidate batch size")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "hellaswag_eval_results"),
        help="Directory for saved evaluation results",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from Hub repos",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision (branch/tag/commit)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load models/tokenizers/datasets from local cache/files only",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torch_dtype(dtype_arg: str, device: torch.device):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16

    # auto
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def preprocess_hellaswag_text(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_context(example: Dict) -> str:
    ctx_a = preprocess_hellaswag_text(example["ctx_a"])
    ctx_b = preprocess_hellaswag_text(example["ctx_b"])
    if ctx_b:
        ctx_b = ctx_b[0].upper() + ctx_b[1:]
    return f"{ctx_a} {ctx_b}".strip()


@dataclass
class Candidate:
    example_idx: int
    choice_idx: int
    prompt: str
    continuation: str


def infer_max_seq_len(model, tokenizer) -> int:
    candidates: List[int] = []

    model_max = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_max, int) and model_max > 0:
        candidates.append(model_max)

    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        candidates.append(tok_max)

    if not candidates:
        return 2048

    return min(candidates)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def parse_step_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"step(\d+)", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (-1, path.name)


def resolve_model_list(args: argparse.Namespace) -> List[str]:
    if args.model:
        return [args.model]

    base = Path(args.model_dir).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"--model-dir does not exist: {base}")

    discovered: List[Path] = []

    # Allow passing a direct checkpoint directory to --model-dir.
    if (base / "config.json").exists():
        discovered.append(base)

    for p in sorted(base.glob(args.model_pattern)):
        if p.is_dir() and (p / "config.json").exists():
            discovered.append(p)

    # De-duplicate while preserving order.
    unique: List[Path] = []
    seen = set()
    for p in discovered:
        if str(p) not in seen:
            seen.add(str(p))
            unique.append(p)

    if not unique:
        raise SystemExit(
            f"No checkpoint folders found in {base} matching '{args.model_pattern}' with config.json"
        )

    unique.sort(key=parse_step_key)
    return [str(p) for p in unique]


def load_model_and_tokenizer(
    model_ref: str,
    revision: str | None,
    trust_remote_code: bool,
    dtype,
    device: torch.device,
    local_files_only: bool,
):
    print(f"Loading tokenizer: {model_ref}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print(f"Loading model: {model_ref}")
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        revision=revision,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    if getattr(model, "get_input_embeddings", None) is not None:
        embed = model.get_input_embeddings()
        if embed is not None and embed.num_embeddings < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    max_seq_len = infer_max_seq_len(model, tokenizer)
    return model, tokenizer, max_seq_len


def load_dataset_compat(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    local_files_only: bool,
):
    ds_args: List[str] = [dataset_name]
    if dataset_config is not None:
        ds_args.append(dataset_config)

    ds_kwargs = {"split": split}

    if local_files_only:
        # Older datasets versions may not support load_dataset(..., local_files_only=...).
        # download_config is the most version-stable way to request offline behavior.
        try:
            from datasets import DownloadConfig

            ds_kwargs["download_config"] = DownloadConfig(local_files_only=True)
        except Exception:
            pass

    try:
        return load_dataset(*ds_args, **ds_kwargs)
    except TypeError:
        # Fallback for very old datasets versions that don't accept download_config.
        if "download_config" in ds_kwargs:
            ds_kwargs.pop("download_config")
            return load_dataset(*ds_args, **ds_kwargs)
        raise


def score_candidates(
    model,
    tokenizer,
    candidates: Sequence[Candidate],
    batch_size: int,
    device: torch.device,
    max_seq_len: int,
) -> Tuple[List[float], List[int]]:
    all_sums: List[float] = []
    all_lens: List[int] = []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id; set pad_token before scoring")

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]

        input_id_rows: List[List[int]] = []
        cont_start_rows: List[int] = []

        for c in batch:
            prompt_ids = tokenizer.encode(c.prompt, add_special_tokens=False)
            cont_ids = tokenizer.encode(c.continuation, add_special_tokens=False)

            if not cont_ids:
                cont_ids = [tokenizer.eos_token_id or pad_id]

            total_len = len(prompt_ids) + len(cont_ids)
            if total_len > max_seq_len:
                overflow = total_len - max_seq_len
                if overflow < len(prompt_ids):
                    prompt_ids = prompt_ids[overflow:]
                else:
                    prompt_ids = []
                    cont_ids = cont_ids[:max_seq_len]

            input_ids = prompt_ids + cont_ids
            cont_start = len(prompt_ids)

            if len(input_ids) < 2:
                input_ids = [pad_id] + input_ids
                cont_start += 1

            input_id_rows.append(input_ids)
            cont_start_rows.append(cont_start)

        max_len = max(len(x) for x in input_id_rows)

        input_tensor = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        attn_tensor = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        cont_mask = torch.zeros((len(batch), max_len - 1), dtype=torch.bool, device=device)

        for i, (ids, cont_start) in enumerate(zip(input_id_rows, cont_start_rows)):
            seq_len = len(ids)
            input_tensor[i, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
            attn_tensor[i, :seq_len] = 1

            start_idx = max(cont_start - 1, 0)
            end_idx = max(seq_len - 1, 0)
            if start_idx < end_idx:
                cont_mask[i, start_idx:end_idx] = True

        with torch.no_grad():
            logits = model(input_ids=input_tensor, attention_mask=attn_tensor).logits

        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        next_tokens = input_tensor[:, 1:]
        token_logp = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

        token_logp = token_logp.masked_fill(~cont_mask, 0.0)
        seq_sum = token_logp.sum(dim=1)
        seq_len = cont_mask.sum(dim=1)

        all_sums.extend(seq_sum.detach().cpu().tolist())
        all_lens.extend(seq_len.detach().cpu().tolist())

    return all_sums, all_lens


def evaluate_hellaswag(
    model,
    tokenizer,
    dataset,
    batch_size: int,
    device: torch.device,
    max_seq_len: int,
) -> Dict:
    candidates: List[Candidate] = []
    labels: List[int] = []
    choice_counts: List[int] = []

    for ex_idx, ex in enumerate(dataset):
        context = build_context(ex)
        label = int(ex["label"])
        labels.append(label)
        n_choices = len(ex["endings"])
        choice_counts.append(n_choices)

        for choice_idx, ending in enumerate(ex["endings"]):
            continuation = " " + preprocess_hellaswag_text(ending)
            candidates.append(
                Candidate(
                    example_idx=ex_idx,
                    choice_idx=choice_idx,
                    prompt=context,
                    continuation=continuation,
                )
            )

    sums, lens = score_candidates(
        model=model,
        tokenizer=tokenizer,
        candidates=candidates,
        batch_size=batch_size,
        device=device,
        max_seq_len=max_seq_len,
    ) 

    n_examples = len(labels)

    pred_raw: List[int] = []
    pred_norm: List[int] = []

    correct_raw = 0
    correct_norm = 0

    cursor = 0
    for ex_idx in range(n_examples):
        n_choices = choice_counts[ex_idx]
        start = cursor
        end = cursor + n_choices

        ex_sums = sums[start:end]
        ex_lens = lens[start:end]
        ex_norm = [s / max(l, 1) for s, l in zip(ex_sums, ex_lens)]

        raw_choice = max(range(n_choices), key=lambda i: ex_sums[i])
        norm_choice = max(range(n_choices), key=lambda i: ex_norm[i])

        pred_raw.append(raw_choice)
        pred_norm.append(norm_choice)

        gold = labels[ex_idx]
        correct_raw += int(raw_choice == gold)
        correct_norm += int(norm_choice == gold)
        cursor = end

    acc = correct_raw / n_examples if n_examples else 0.0
    acc_norm = correct_norm / n_examples if n_examples else 0.0

    return {
        "num_examples": n_examples,
        "accuracy": acc,
        "accuracy_norm": acc_norm,
        "labels": labels,
        "pred_raw": pred_raw,
        "pred_norm": pred_norm,
    }


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    dtype = get_torch_dtype(args.dtype, device)
    model_list = resolve_model_list(args)

    # Local paths should avoid accidental Hub/network lookups.
    local_files_only_default = args.local_files_only
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ({args.split})")
    try:
        dataset = load_dataset_compat(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load dataset '{args.dataset}' split '{args.split}': {exc}\n"
            "Tip: ensure internet access, or pre-cache the dataset and rerun."
        ) from exc
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    all_results: List[Dict] = []
    total_models = len(model_list)

    for idx, model_ref in enumerate(model_list, start=1):
        print(f"\n[{idx}/{total_models}] Evaluating model: {model_ref}")

        path_exists = Path(model_ref).expanduser().exists()
        local_only = local_files_only_default or path_exists

        model, tokenizer, max_seq_len = load_model_and_tokenizer(
            model_ref=model_ref,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            device=device,
            local_files_only=local_only,
        )

        start_time = time.time()
        metrics = evaluate_hellaswag(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=args.batch_size,
            device=device,
            max_seq_len=max_seq_len,
        )
        elapsed = time.time() - start_time

        result = {
            "model": model_ref,
            "revision": args.revision,
            "split": args.split,
            "batch_size": args.batch_size,
            "max_seq_len": max_seq_len,
            "dtype": str(dtype),
            "device": str(device),
            "elapsed_seconds": elapsed,
            "timestamp_unix": int(time.time()),
            "num_examples": metrics["num_examples"],
            "accuracy": metrics["accuracy"],
            "accuracy_norm": metrics["accuracy_norm"],
        }
        all_results.append(result)

        print("HellaSwag results")
        print(f"  Model: {result['model']}")
        print(f"  Examples: {result['num_examples']}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Accuracy (length-normalized): {result['accuracy_norm']:.4f}")
        print(f"  Elapsed: {result['elapsed_seconds']:.2f}s")

        ts = int(time.time())
        model_tag = sanitize_name(Path(model_ref).name if path_exists else model_ref)
        run_name = f"hellaswag_{model_tag}_{ts}"
        metrics_path = output_dir / f"{run_name}_metrics.json"
        preds_path = output_dir / f"{run_name}_predictions.json"

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        with preds_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "labels": metrics["labels"],
                    "pred_raw": metrics["pred_raw"],
                    "pred_norm": metrics["pred_norm"],
                },
                f,
                indent=2,
            )

        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved predictions to: {preds_path}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(all_results) > 1:
        summary_ts = int(time.time())
        summary_path = output_dir / f"hellaswag_summary_{summary_ts}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_models": len(all_results),
                    "results": all_results,
                },
                f,
                indent=2,
            )
        print(f"\nSaved summary to: {summary_path}")

    print("\nScore summary")
    for r in all_results:
        print(
            f"  {r['model']}: accuracy={r['accuracy']:.4f}, "
            f"accuracy_norm={r['accuracy_norm']:.4f}"
        )


if __name__ == "__main__":
    main()
