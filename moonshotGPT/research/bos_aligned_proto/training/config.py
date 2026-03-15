"""Central definition of the training CLI and runtime configuration.

This module keeps the argument parser and the default values in one place.
The training entrypoint can then accept a single TrainConfig object instead
of a long list of keyword arguments, which makes the code easier to read and
safer to change over time.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROTO_ROOT = os.path.dirname(_THIS_DIR)
_RESEARCH_ROOT = os.path.dirname(_PROTO_ROOT)
_REPO_ROOT = os.path.dirname(_RESEARCH_ROOT)
DEFAULT_EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "runs", "research", "bos_aligned_proto")

CLI_DESCRIPTION = (
    "Train GPT-2 with BOS-row-packed memmapped uint16 .bin shards + "
    "token-budget accumulation (step-based) + exposure + ewok per-item"
)


@dataclass
class TrainConfig:
    seed: int = 42
    micro_batch_size: int = 10
    total_batch_tokens: int = 524288
    max_train_steps: int = 20000
    data_dir: str = ""
    experiments_dir: str = DEFAULT_EXPERIMENTS_DIR
    seq_len: int = 1024
    vocab_size: int = 50257
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    num_workers: int = 0
    shuffle_blocks: bool = True
    grad_clip: float = 1.0
    learning_rate: float = 6e-4
    warmup_iters: int = 700
    learning_rate_decay_frac: float = 0.0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eval_every: int = 200
    hellaswag_every: int = 500
    hellaswag_batch_size: int = 8
    hellaswag_max_examples: int = 4096
    hellaswag_dataset: str = "hellaswag"
    hellaswag_dataset_config: Optional[str] = None
    hellaswag_split: str = "validation"
    hellaswag_local_files_only: bool = False
    core_every: int = 2000
    core_max_per_task: int = 500
    core_bundle_dir: str = ""
    core_local_files_only: bool = False
    ewok_every: int = 250
    ewok_batch_size: int = 4
    save_every: int = 2000
    exposure_every: int = 100
    push_to_hub: bool = False
    skip_final_ewok: bool = False
    include_ewok_sum_plots: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=10,
        help="Per-process/GPU micro-batch size (sequences)",
    )
    parser.add_argument(
        "--total_batch_tokens",
        type=int,
        default=524288,
        help="Global tokens per optimizer step target",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20000,
        help="Total optimizer steps to run",
    )

    parser.add_argument(
        "--experiments_dir",
        type=str,
        default=DEFAULT_EXPERIMENTS_DIR,
        help="Parent directory where run folders are created; defaults to repo-root runs/research/bos_aligned_proto",
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="Model vocab size (GPT-2 tokenizer is 50257; >50257 allowed with CE on first 50257 logits)",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=768,
        help="Transformer hidden size (GPT-2 small uses 768; medium uses 1024)",
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=12,
        help="Attention heads (GPT-2 small uses 12; medium uses 16)",
    )
    parser.add_argument(
        "--n_layer",
        type=int,
        default=12,
        help="Transformer depth (GPT-2 small uses 12 layers; medium uses 24)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing BOS-row-packed train_*.bin, val_*.bin, meta.json",
    )
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--shuffle_blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable block-level shuffle (recommended).",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6e-4,
        help="Peak learning rate (llm.c default for d12 run)",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=700,
        help="Warmup iterations for llm.c-style LR schedule",
    )
    parser.add_argument(
        "--learning_rate_decay_frac",
        type=float,
        default=0.0,
        help="Final LR fraction for llm.c-style cosine schedule",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="AdamW decay for matrix/embedding params (llm.c-style)",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Clip norm (<=0 disables)")

    parser.add_argument(
        "--eval_every",
        type=int,
        default=200,
        help="Quick val-loss eval every N optimizer steps (0 disables)",
    )
    parser.add_argument(
        "--hellaswag_every",
        type=int,
        default=500,
        help="Run HellaSwag eval every N optimizer steps (0 disables)",
    )
    parser.add_argument(
        "--hellaswag_batch_size",
        type=int,
        default=8,
        help="Batch size for HellaSwag evaluate_hellaswag()",
    )
    parser.add_argument(
        "--hellaswag_max_examples",
        type=int,
        default=4096,
        help="Optional max examples for HellaSwag split",
    )
    parser.add_argument(
        "--hellaswag_dataset",
        type=str,
        default="hellaswag",
        help="Dataset name/path passed to datasets.load_dataset for HellaSwag",
    )
    parser.add_argument(
        "--hellaswag_dataset_config",
        type=str,
        default=None,
        help="Optional datasets config name for HellaSwag",
    )
    parser.add_argument(
        "--hellaswag_split",
        type=str,
        default="validation",
        help="Dataset split used for HellaSwag eval",
    )
    parser.add_argument(
        "--hellaswag_local_files_only",
        action="store_true",
        help="Load HellaSwag dataset from local cache/files only",
    )
    parser.add_argument(
        "--core_every",
        type=int,
        default=2000,
        help="Run CORE eval every N optimizer steps (0 disables periodic and final CORE)",
    )
    parser.add_argument(
        "--core_max_per_task",
        type=int,
        default=500,
        help="Optional max examples per CORE task (-1 = all)",
    )
    parser.add_argument(
        "--core_bundle_dir",
        type=str,
        default="",
        help="Optional local CORE eval bundle directory override",
    )
    parser.add_argument(
        "--core_local_files_only",
        action="store_true",
        help="Load CORE eval bundle from local files only (never auto-download)",
    )
    parser.add_argument(
        "--ewok_every",
        type=int,
        default=250,
        help="Run EWoK eval every N optimizer steps (0 disables)",
    )
    parser.add_argument(
        "--ewok_batch_size",
        type=int,
        default=4,
        help="Batch size inside EWoK evaluate()",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=2000,
        help="Save checkpoint every N optimizer steps (0 disables)",
    )
    parser.add_argument(
        "--exposure_every",
        type=int,
        default=100,
        help="Log exposure meta every N optimizer steps (0 disables)",
    )

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--skip_final_ewok",
        action="store_true",
        help="Skip final EWoK eval at the end (useful for smoke tests)",
    )
    parser.add_argument(
        "--include_ewok_sum_plots",
        action="store_true",
        help="Include EWOK sum-reduction plots in the auto-generated plot_step_metrics analysis",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    return TrainConfig(**vars(args))
