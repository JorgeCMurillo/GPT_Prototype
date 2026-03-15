#!/usr/bin/env python3
"""Evaluate causal language models on the DCLM CORE benchmark.

This closely follows nanochat's CORE evaluation semantics while adapting them
to the Hugging Face model/tokenizer APIs used in moonshotGPT.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import fcntl
import torch
import torch.distributed as dist
import yaml
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer


CORE_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
DEFAULT_EVAL_BUNDLE_DIR = _PROJECT_DIR / "eval_bundle"


@dataclass(frozen=True)
class CoreTaskMeta:
    label: str
    task_type: str
    dataset_uri: str
    num_fewshot: int
    continuation_delimiter: str
    random_baseline_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on the DCLM CORE benchmark")
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
        "--bundle-dir",
        default=str(DEFAULT_EVAL_BUNDLE_DIR),
        help="Local CORE eval bundle directory (default: <repo>/eval_bundle)",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        help="Optional max examples per CORE task (-1 = all)",
    )
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
        default=str(_PROJECT_DIR / "core_eval_results"),
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
        help="Load models/tokenizers/core bundle from local files only",
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

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


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
    if (base / "config.json").exists():
        discovered.append(base)

    for p in sorted(base.glob(args.model_pattern)):
        if p.is_dir() and (p / "config.json").exists():
            discovered.append(p)

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

    return model, tokenizer


def resolve_bos_token_id(tokenizer) -> int:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        return int(bos_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return int(eos_token_id)
    raise RuntimeError("Tokenizer must define bos_token_id or eos_token_id for CORE evaluation.")


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


def _encode_batch(tokenizer, texts: Sequence[str]) -> List[List[int]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return [list(map(int, row)) for row in encoded]


def _prepend_bos(token_lists: Sequence[Sequence[int]], bos_token_id: int) -> List[List[int]]:
    return [[int(bos_token_id), *list(tokens)] for tokens in token_lists]


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(choice=choice, **context) for choice in item["choices"]]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    return [template.render(context=context_option, **context) for context_option in item["context_options"]]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        "left": range(min_len),
        "right": range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    batch_size = len(tokens)
    seq_len = max(len(x) for x in tokens)
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i, seq in enumerate(tokens):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def batch_sequences_mc(tokenizer, prompts, bos_token_id):
    tokens = _prepend_bos(_encode_batch(tokenizer, prompts), bos_token_id)
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts, bos_token_id):
    tokens = _prepend_bos(_encode_batch(tokenizer, prompts), bos_token_id)
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts, bos_token_id):
    tokens = _prepend_bos(_encode_batch(tokenizer, prompts), bos_token_id)
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx
    assert tokens_without == tokens_with[:start_idx]
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids, attention_mask=None):
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        logits.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)
    losses[:, -1] = float("nan")
    predictions = logits.argmax(dim=-1)
    return losses, predictions


def _crop_to_model_limit(tokens, start_idxs, end_idxs, max_tokens: int):
    new_tokens, new_start_idxs, new_end_idxs = [], [], []
    for token_row, start_idx, end_idx in zip(tokens, start_idxs, end_idxs):
        if len(token_row) > max_tokens:
            num_to_crop = len(token_row) - max_tokens
            cropped = token_row[-max_tokens:]
            start_idx = start_idx - num_to_crop
            end_idx = end_idx - num_to_crop
            assert start_idx >= 0
            assert end_idx >= 0
            new_tokens.append(cropped)
            new_start_idxs.append(start_idx)
            new_end_idxs.append(end_idx)
        else:
            new_tokens.append(token_row)
            new_start_idxs.append(start_idx)
            new_end_idxs.append(end_idx)
    return new_tokens, new_start_idxs, new_end_idxs


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta: CoreTaskMeta):
    item = data[idx]
    fewshot_examples = []
    if task_meta.num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, task_meta.num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    bos_token_id = resolve_bos_token_id(tokenizer)
    if task_meta.task_type == "multiple_choice":
        prompts = render_prompts_mc(item, task_meta.continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts, bos_token_id)
    elif task_meta.task_type == "schema":
        prompts = render_prompts_schema(item, task_meta.continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts, bos_token_id)
    elif task_meta.task_type == "language_modeling":
        prompts = render_prompts_lm(item, task_meta.continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts, bos_token_id)
    else:
        raise ValueError(f"Unsupported task type: {task_meta.task_type}")

    max_tokens = infer_max_seq_len(model, tokenizer)
    tokens, start_idxs, end_idxs = _crop_to_model_limit(tokens, start_idxs, end_idxs, max_tokens)

    pad_token_id = bos_token_id
    input_ids, attention_mask = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    losses, predictions = forward_model(model, input_ids, attention_mask=attention_mask)

    if task_meta.task_type == "language_modeling":
        start_idx = start_idxs[0]
        end_idx = end_idxs[0]
        predicted_tokens = predictions[0, start_idx - 1 : end_idx - 1]
        actual_tokens = input_ids[0, start_idx:end_idx]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    else:
        mean_losses = [
            losses[i, start_idx - 1 : end_idx - 1].mean().item()
            for i, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item["gold"]
    return bool(is_correct)


def evaluate_task(
    model,
    tokenizer,
    data,
    device,
    task_meta: CoreTaskMeta,
    *,
    distributed: bool = True,
):
    use_distributed = bool(distributed and dist.is_available() and dist.is_initialized())
    rank = dist.get_rank() if use_distributed else 0
    world_size = dist.get_world_size() if use_distributed else 1

    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        correct[idx] = float(evaluate_example(idx, model, tokenizer, data, device, task_meta))

    if use_distributed and world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    return float(correct.mean().item())


def _lock_file(lock_path: Path):
    os.makedirs(lock_path.parent, exist_ok=True)
    handle = open(lock_path, "a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _unlock_file(handle) -> None:
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _validate_eval_bundle_dir(bundle_dir: Path) -> None:
    required = [
        bundle_dir / "core.yaml",
        bundle_dir / "eval_meta_data.csv",
        bundle_dir / "eval_data",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"CORE eval bundle is incomplete at {bundle_dir}. Missing: {', '.join(missing)}"
        )


def _download_to_path(url: str, out_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "moonshotGPT-core-eval"})
    with urllib.request.urlopen(req) as response, out_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ensure_eval_bundle(
    bundle_dir: str | os.PathLike[str] | None = None,
    *,
    local_files_only: bool = False,
    url: str = CORE_BUNDLE_URL,
) -> Path:
    target_dir = Path(bundle_dir or DEFAULT_EVAL_BUNDLE_DIR).expanduser().resolve()
    if target_dir.exists():
        _validate_eval_bundle_dir(target_dir)
        return target_dir

    if local_files_only:
        raise FileNotFoundError(f"CORE eval bundle not found at {target_dir} and local_files_only=True")

    lock_handle = _lock_file(target_dir.parent / ".eval_bundle.lock")
    try:
        if target_dir.exists():
            _validate_eval_bundle_dir(target_dir)
            return target_dir

        tmp_root = Path(tempfile.mkdtemp(prefix="core_eval_bundle_", dir=str(target_dir.parent)))
        zip_path = tmp_root / "eval_bundle.zip"
        try:
            _download_to_path(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_root)
            extracted_dir = tmp_root / "eval_bundle"
            if not extracted_dir.exists():
                raise FileNotFoundError(f"Downloaded CORE bundle did not contain eval_bundle/: {url}")
            _validate_eval_bundle_dir(extracted_dir)
            shutil.move(str(extracted_dir), str(target_dir))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        _validate_eval_bundle_dir(target_dir)
        return target_dir
    finally:
        _unlock_file(lock_handle)


def load_core_bundle(
    bundle_dir: str | os.PathLike[str] | None = None,
    *,
    local_files_only: bool = False,
    url: str = CORE_BUNDLE_URL,
) -> Tuple[Path, List[CoreTaskMeta]]:
    resolved_bundle_dir = ensure_eval_bundle(bundle_dir, local_files_only=local_files_only, url=url)
    config_path = resolved_bundle_dir / "core.yaml"
    eval_meta_data_path = resolved_bundle_dir / "eval_meta_data.csv"

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    raw_tasks = config["icl_tasks"]

    random_baselines: Dict[str, float] = {}
    with eval_meta_data_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            random_baselines[str(row["Eval Task"])] = float(row["Random baseline"])

    tasks = [
        CoreTaskMeta(
            label=str(task["label"]),
            task_type=str(task["icl_task_type"]),
            dataset_uri=str(task["dataset_uri"]),
            num_fewshot=int(task["num_fewshot"][0]),
            continuation_delimiter=str(task.get("continuation_delimiter", " ")),
            random_baseline_pct=float(random_baselines[str(task["label"])]),
        )
        for task in raw_tasks
    ]
    return resolved_bundle_dir, tasks


def _load_jsonl_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line.strip()) for line in handle if line.strip()]


def evaluate_core(
    model,
    tokenizer,
    device=None,
    *,
    max_per_task: int = -1,
    bundle_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
    url: str = CORE_BUNDLE_URL,
    distributed: bool = True,
) -> Dict[str, object]:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_bundle_dir, tasks = load_core_bundle(bundle_dir, local_files_only=local_files_only, url=url)
    data_base_path = resolved_bundle_dir / "eval_data"
    results: Dict[str, float] = {}
    centered_results: Dict[str, float] = {}
    examples_per_task: Dict[str, int] = {}

    for task in tasks:
        data_path = data_base_path / task.dataset_uri
        data = _load_jsonl_rows(data_path)
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if int(max_per_task) > 0:
            data = data[: int(max_per_task)]
        accuracy = evaluate_task(
            model,
            tokenizer,
            data,
            device,
            task,
            distributed=distributed,
        )
        results[task.label] = float(accuracy)
        baseline = 0.01 * float(task.random_baseline_pct)
        centered_results[task.label] = float((accuracy - baseline) / (1.0 - baseline))
        examples_per_task[task.label] = int(len(data))

    core_metric = float(sum(centered_results.values()) / max(1, len(centered_results)))
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "num_tasks": int(len(tasks)),
        "max_per_task": int(max_per_task),
        "examples_per_task": examples_per_task,
        "bundle_source": str(resolved_bundle_dir),
        "bundle_dir": str(resolved_bundle_dir),
    }


def _write_core_csv(path: Path, core_results: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        results = core_results["results"]
        centered_results = core_results["centered_results"]
        for label in results:
            acc = results[label]
            centered = centered_results[label]
            handle.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
        handle.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = get_torch_dtype(args.dtype, device)
    model_refs = resolve_model_list(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_ref in model_refs:
        print(f"Loading model/tokenizer from: {model_ref}")
        model, tokenizer = load_model_and_tokenizer(
            model_ref=model_ref,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            device=device,
            local_files_only=args.local_files_only,
        )
        core_results = evaluate_core(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_per_task=args.max_per_task,
            bundle_dir=args.bundle_dir,
            local_files_only=args.local_files_only,
            distributed=False,
        )
        slug = sanitize_name(Path(model_ref).name if os.path.isdir(model_ref) else model_ref)
        json_path = output_dir / f"{slug}.json"
        csv_path = output_dir / f"{slug}.csv"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(core_results, handle, indent=2)
        _write_core_csv(csv_path, core_results)
        print(f"CORE metric for {model_ref}: {core_results['core_metric']:.4f}")
        print(f"Saved JSON: {json_path}")
        print(f"Saved CSV:  {csv_path}")


__all__ = [
    "CORE_BUNDLE_URL",
    "CoreTaskMeta",
    "DEFAULT_EVAL_BUNDLE_DIR",
    "ensure_eval_bundle",
    "evaluate_core",
    "evaluate_example",
    "evaluate_task",
    "infer_max_seq_len",
    "load_core_bundle",
    "main",
    "resolve_bos_token_id",
]


if __name__ == "__main__":
    main()
