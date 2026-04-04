import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluation.ewok import BABYLM_COMPLETION_CHOICE
from research.bos_aligned_proto.analysis.attribution.trackstar.one_step_sanity import (
    build_candidate_groups,
    compute_query_loss_mean,
    measure_one_step_delta,
)
from research.bos_aligned_proto.analysis.attribution.build_matched_cpt_pools import (
    load_candidate_score_frame,
)
from research.bos_aligned_proto.analysis.attribution.common.ewok_targets import (
    EWOKTargetBundle,
    EWOKTargetItem,
)


def _write_u16(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype=np.uint16).tofile(path)


def _make_row_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "row_data"
    data_dir.mkdir()
    (data_dir / "meta.json").write_text(
        json.dumps({"format": "bos_row_packed_bestfit", "row_tokens": 4, "seq_len": 3}),
        encoding="utf-8",
    )
    _write_u16(data_dir / "train_000000.bin", [10, 11, 12, 13, 20, 21, 22, 23])
    _write_u16(data_dir / "train_000001.bin", [30, 31, 32, 33, 40, 41, 42, 43])
    return data_dir


def _make_attribution_output(tmp_path: Path, data_dir: Path) -> Path:
    out_dir = tmp_path / "attrib"
    out_dir.mkdir()
    row_summary = pd.DataFrame(
        [
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 0,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000000.bin"),
                "local_example_idx": 0,
                "token_offset_start": 0,
                "token_offset_end": 4,
                "row_id": 0,
                "local_row_idx": 0,
                "mean_score": -0.10,
                "mean_abs_score": 0.10,
                "positive_score_sum": 0.00,
                "negative_score_sum": -0.20,
                "max_abs_score": 0.20,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 1,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000000.bin"),
                "local_example_idx": 1,
                "token_offset_start": 4,
                "token_offset_end": 8,
                "row_id": 1,
                "local_row_idx": 1,
                "mean_score": 0.35,
                "mean_abs_score": 0.35,
                "positive_score_sum": 0.70,
                "negative_score_sum": 0.00,
                "max_abs_score": 0.40,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 2,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000001.bin"),
                "local_example_idx": 0,
                "token_offset_start": 0,
                "token_offset_end": 4,
                "row_id": 2,
                "local_row_idx": 0,
                "mean_score": 0.05,
                "mean_abs_score": 0.05,
                "positive_score_sum": 0.10,
                "negative_score_sum": 0.00,
                "max_abs_score": 0.10,
                "target_count": 2,
            },
            {
                "checkpoint_step": 1000,
                "group": "overall",
                "candidate_id": 3,
                "candidate_kind": "bos_packed_row",
                "shard_path": str(data_dir / "train_000001.bin"),
                "local_example_idx": 1,
                "token_offset_start": 4,
                "token_offset_end": 8,
                "row_id": 3,
                "local_row_idx": 1,
                "mean_score": -0.20,
                "mean_abs_score": 0.20,
                "positive_score_sum": 0.00,
                "negative_score_sum": -0.40,
                "max_abs_score": 0.25,
                "target_count": 2,
            },
        ]
    )
    row_summary.to_csv(out_dir / "row_summary_step00001000.csv", index=False)
    np.save(
        out_dir / "dense_scores_step00001000.npy",
        np.asarray(
            [
                [-0.10, 0.80, 0.20, -0.30],
                [0.00, -0.10, 0.05, 0.40],
            ],
            dtype=np.float64,
        ),
    )
    with (out_dir / "target_items.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"target_id": "q-alpha"}) + "\n")
        handle.write(json.dumps({"target_id": "q-beta"}) + "\n")
    return out_dir


class _ToyTokenizer:
    def __init__(self) -> None:
        self.bos_token_id = 0
        self.vocab = {
            "A": 1,
            "B": 2,
            "X": 3,
            "Y": 4,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        pieces = [part for part in str(text).split() if part]
        return [int(self.vocab[piece]) for piece in pieces]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del add_special_tokens, return_attention_mask
        if return_tensors != "pt":
            raise ValueError("Toy tokenizer only supports return_tensors='pt'")
        ids = self.encode(text)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


class _ToyBigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.transition_logits = nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        return SimpleNamespace(logits=self.transition_logits[input_ids])


def _make_toy_bundle() -> EWOKTargetBundle:
    item = EWOKTargetItem(
        target_id="toy-target",
        domain="toy-domain",
        row_index=0,
        score_view=BABYLM_COMPLETION_CHOICE,
        concept_a="A",
        concept_b="B",
        context1="A",
        context2="B",
        target1="X",
        target2="Y",
        context_type_raw="direct",
        context_type="direct",
        context_diff_raw="variable_swap",
        context_diff="variable swap",
        target_diff_raw="variable_swap",
        target_diff="variable swap",
    )
    return EWOKTargetBundle(
        items=(item,),
        groups={"overall": ("toy-target",)},
        source_path=Path("toy-ewok.jsonl"),
        score_view=BABYLM_COMPLETION_CHOICE,
        score_reduction="mean",
    )


def test_build_candidate_groups_produces_non_overlapping_top_random_bottom(tmp_path: Path) -> None:
    data_dir = _make_row_data(tmp_path)
    attrib_dir = _make_attribution_output(tmp_path, data_dir)
    scored = load_candidate_score_frame(
        attribution_dir=attrib_dir,
        step=1000,
        score_mode="net_pooled",
    )

    groups = build_candidate_groups(
        scored,
        group_size=1,
        rng=np.random.default_rng(0),
    )

    assert set(groups.keys()) == {"top", "matched_random", "bottom"}
    assert groups["top"].candidate_ids == (1,)
    assert groups["bottom"].candidate_ids == (0,)
    assert groups["matched_random"].candidate_ids[0] in {2, 3}

    all_ids = {
        int(groups["top"].candidate_ids[0]),
        int(groups["matched_random"].candidate_ids[0]),
        int(groups["bottom"].candidate_ids[0]),
    }
    assert len(all_ids) == 3


def test_measure_one_step_delta_makes_helpful_batch_more_negative_than_harmful_batch() -> None:
    tokenizer = _ToyTokenizer()
    bundle = _make_toy_bundle()
    model = _ToyBigramLM(vocab_size=5)
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    baseline_loss = compute_query_loss_mean(
        model,
        tokenizer,
        bundle,
        batch_size=1,
        temperature=1.0,
    )

    helpful = measure_one_step_delta(
        model,
        base_state=base_state,
        tokenizer=tokenizer,
        bundle=bundle,
        baseline_query_loss=baseline_loss,
        input_ids=torch.tensor([[1], [2]], dtype=torch.long),
        labels=torch.tensor([[3], [4]], dtype=torch.long),
        update_lr=0.5,
        target_batch_size=1,
        temperature=1.0,
        device=torch.device("cpu"),
    )
    harmful = measure_one_step_delta(
        model,
        base_state=base_state,
        tokenizer=tokenizer,
        bundle=bundle,
        baseline_query_loss=baseline_loss,
        input_ids=torch.tensor([[1], [2]], dtype=torch.long),
        labels=torch.tensor([[4], [3]], dtype=torch.long),
        update_lr=0.5,
        target_batch_size=1,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    assert helpful["delta_q"] < 0.0
    assert harmful["delta_q"] > 0.0
    assert helpful["delta_q"] < harmful["delta_q"]
