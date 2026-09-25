from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

pe = import_module("moonshotGPT.hf_ewok_eval.prompted_ewok_eval")
rq = import_module("moonshotGPT.hf_ewok_eval.run_queue")


class _FakeModel:
    hf_device_map = {"model.embed_tokens": "cuda:0"}

    def to(self, device):
        del device
        return self

    def eval(self):
        return self

    def get_input_embeddings(self):
        return None

    def parameters(self):
        yield rq.torch.zeros(1)


class _FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    padding_side = "right"


class _BoundaryMergingTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "prefix": [10],
            " True": [11],
            " False": [12],
            "prefix ": [10, 99],
            "prefix True": [10, 11],
        }
        return list(mapping[str(text)])


class _UniformLogitModel:
    def parameters(self):
        yield rq.torch.zeros(1)

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        return {"logits": rq.torch.zeros((*input_ids.shape, 128), device=input_ids.device)}


class _FakeSharedEwokModule:
    BABYLM_COMPLETION_CHOICE = "babylm_completion_choice"
    EWOK_CONTEXT_SENSITIVITY = "ewok_context_sensitivity"
    SRC = Path("/tmp/ewok_fast_jsonl.zip")
    ewok_df = pd.DataFrame(
        [
            {
                "Domain": "spatial-relations",
                "Context1": "The shop is in front of Chao.",
                "Target1": "The shop is to the right of Chao.",
                "Context2": "The shop is behind Chao.",
                "Target2": "The shop is to the left of Chao.",
            }
        ]
    )

    def _summarize_records_all_methods(self, records, margin_eps):
        del margin_eps
        babylm_correct = [1.0 if row["babylm_completion_choice_correct_official"] else 0.0 for row in records]
        context_correct = [1.0 if row["ewok_context_sensitivity_correct_official"] else 0.0 for row in records]
        return {
            self.BABYLM_COMPLETION_CHOICE: {
                "domain_scores_full": {"average": [float(sum(babylm_correct) / len(babylm_correct)), 0.0]},
                "domain_scores_official": {"average": float(sum(babylm_correct) / len(babylm_correct))},
            },
            self.EWOK_CONTEXT_SENSITIVITY: {
                "domain_scores_full": {"average": [float(sum(context_correct) / len(context_correct)), 0.0]},
                "domain_scores_official": {"average": float(sum(context_correct) / len(context_correct))},
            },
        }


def test_render_choice_prompt_replaces_placeholders() -> None:
    rendered = pe._render_choice_prompt(
        template_text="Context={{Ci}} | 1={{T1}} | 2={{T2}}",
        context_text="Ali sees a candle.",
        target_1_text="It glows.",
        target_2_text="It swims.",
    )

    assert rendered == "Context=Ali sees a candle. | 1=It glows. | 2=It swims."


def test_render_context_choice_prompt_replaces_placeholders() -> None:
    rendered = pe._render_context_choice_prompt(
        template_text="Statement={{Tj}} | C1={{C1}} | C2={{C2}}",
        statement_text="The candle glows.",
        context_1_text="Ali sees a lit candle.",
        context_2_text="Ali sees a stone.",
    )

    assert rendered == "Statement=The candle glows. | C1=Ali sees a lit candle. | C2=Ali sees a stone."


def test_parse_choice_response_text_handles_whitespace_and_invalid() -> None:
    assert pe._parse_choice_response_text(" 2\n") == 2
    assert pe._parse_choice_response_text("Answer: 1") == 1
    assert pe._parse_choice_response_text("I think it through.\nAnswer: 2") == 2
    assert (
        pe._parse_choice_response_text(
            '1. Wrong\n2. Right\nThus answer: 2.<|end|><|start|>assistant<|channel|>final<|message|>Answer: 2'
        )
        == 2
    )
    assert pe._parse_choice_response_text("neither") is None


def test_conditional_target_logps_scores_label_tokens_across_tokenizer_boundary() -> None:
    results = pe._conditional_target_token_logps(
        _UniformLogitModel(),
        _BoundaryMergingTokenizer(),
        prefixes=["prefix"],
        targets=["True"],
        answer_separator=" ",
        batch_size=1,
    )

    assert len(results) == 1
    assert results[0].numel() == 1
    assert results[0].item() == pytest.approx(-rq.torch.log(rq.torch.tensor(128.0)).item())


def test_build_choice_record_computes_babylm_and_context_margins() -> None:
    inference_c1 = pe.PromptChoiceInference(
        raw_response_text=None,
        predicted_choice=1,
        response_valid=True,
        choice_1_score=-0.2,
        choice_2_score=-1.4,
    )
    inference_c2 = pe.PromptChoiceInference(
        raw_response_text=None,
        predicted_choice=2,
        response_valid=True,
        choice_1_score=-1.0,
        choice_2_score=-0.1,
    )

    record = pe._build_choice_record(
        domain="spatial-relations",
        row_index=7,
        score_reduction="mean",
        inference_mode="choice_answer_logprob",
        prompt_template_name="direct_choice",
        prompt_template_source="builtin:direct_choice",
        prompt_c1="prompt one",
        prompt_c2="prompt two",
        inference_c1=inference_c1,
        inference_c2=inference_c2,
        swap_targets_c1=False,
        swap_targets_c2=False,
        margin_eps=1e-6,
        store_prompts=False,
    )

    assert record["babylm_completion_choice_correct_official"] is True
    assert record["babylm_completion_choice_correct_symmetric"] is True
    assert record["ewok_context_sensitivity_correct_official"] is True
    assert record["predicted_choice_c1"] == 1
    assert record["predicted_choice_c2"] == 2
    assert record["margin_official_m1"] == pytest.approx(1.2)
    assert record["ewok_context_sensitivity_margin_official_k1"] == pytest.approx(0.8)


def test_build_context_choice_record_maps_generated_context_choices_to_k_margins() -> None:
    inference_t1 = pe.PromptChoiceInference(
        raw_response_text="Reasoning: context 1 supports the statement.\nAnswer: 1",
        predicted_choice=1,
        response_valid=True,
        choice_1_score=1.0,
        choice_2_score=0.0,
    )
    inference_t2 = pe.PromptChoiceInference(
        raw_response_text="Reasoning: context 2 supports the statement.\nAnswer: 2",
        predicted_choice=2,
        response_valid=True,
        choice_1_score=0.0,
        choice_2_score=1.0,
    )

    record = pe._build_context_choice_record(
        domain="spatial-relations",
        row_index=5,
        score_reduction="mean",
        inference_mode="context_choice_generate",
        prompt_template_name="context_choice",
        prompt_template_source="/tmp/context_choice.txt",
        prompt_t1="statement one prompt",
        prompt_t2="statement two prompt",
        inference_t1=inference_t1,
        inference_t2=inference_t2,
        swap_contexts_t1=False,
        swap_contexts_t2=False,
        margin_eps=1e-6,
        store_prompts=True,
    )

    assert record["predicted_context_t1"] == 1
    assert record["predicted_context_t2"] == 2
    assert record["context_1_score_given_t1_prompt"] == pytest.approx(1.0)
    assert record["context_2_score_given_t1_prompt"] == pytest.approx(0.0)
    assert record["context_1_score_given_t2_prompt"] == pytest.approx(0.0)
    assert record["context_2_score_given_t2_prompt"] == pytest.approx(1.0)
    assert record["ewok_context_sensitivity_margin_official_k1"] == pytest.approx(1.0)
    assert record["ewok_context_sensitivity_margin_symmetric_k2"] == pytest.approx(1.0)
    assert record["ewok_context_sensitivity_correct_combined"] is True
    assert record["prompt_t1_context_choice"] == "statement one prompt"
    assert "context 2 supports" in record["response_text_t2_context_choice"]


def test_build_context_choice_record_remaps_swapped_display_contexts() -> None:
    inference_t1 = pe.PromptChoiceInference(
        raw_response_text="Answer: 2",
        predicted_choice=2,
        response_valid=True,
        choice_1_score=0.0,
        choice_2_score=1.0,
    )
    inference_t2 = pe.PromptChoiceInference(
        raw_response_text="Answer: 1",
        predicted_choice=1,
        response_valid=True,
        choice_1_score=1.0,
        choice_2_score=0.0,
    )

    record = pe._build_context_choice_record(
        domain="spatial-relations",
        row_index=6,
        score_reduction="mean",
        inference_mode="context_choice_generate",
        prompt_template_name="context_choice",
        prompt_template_source="/tmp/context_choice.txt",
        prompt_t1="statement one prompt",
        prompt_t2="statement two prompt",
        inference_t1=inference_t1,
        inference_t2=inference_t2,
        swap_contexts_t1=True,
        swap_contexts_t2=True,
        margin_eps=1e-6,
        store_prompts=False,
    )

    assert record["display_context_1_id_t1"] == 2
    assert record["display_gold_context_t1"] == 2
    assert record["displayed_predicted_context_t1"] == 2
    assert record["predicted_context_t1"] == 1
    assert record["displayed_predicted_context_t2"] == 1
    assert record["predicted_context_t2"] == 2
    assert record["ewok_context_sensitivity_correct_official"] is True
    assert record["ewok_context_sensitivity_correct_symmetric"] is True


def test_build_choice_record_remaps_swapped_display_choices_to_canonical_targets() -> None:
    inference_c1 = pe.PromptChoiceInference(
        raw_response_text="Answer: 1",
        predicted_choice=1,
        response_valid=True,
        choice_1_score=1.0,
        choice_2_score=0.0,
    )
    inference_c2 = pe.PromptChoiceInference(
        raw_response_text="Answer: 2",
        predicted_choice=2,
        response_valid=True,
        choice_1_score=0.0,
        choice_2_score=1.0,
    )

    record = pe._build_choice_record(
        domain="spatial-relations",
        row_index=3,
        score_reduction="mean",
        inference_mode="choice_generate",
        prompt_template_name="direct_choice",
        prompt_template_source="builtin:direct_choice",
        prompt_c1="prompt one",
        prompt_c2="prompt two",
        inference_c1=inference_c1,
        inference_c2=inference_c2,
        swap_targets_c1=True,
        swap_targets_c2=True,
        margin_eps=1e-6,
        store_prompts=False,
    )

    assert record["display_target_1_id_c1"] == 2
    assert record["display_gold_choice_c1"] == 2
    assert record["displayed_predicted_choice_c1"] == 1
    assert record["predicted_choice_c1"] == 2
    assert record["choice_1_score_given_c1_prompt"] == pytest.approx(0.0)
    assert record["choice_2_score_given_c1_prompt"] == pytest.approx(1.0)


def test_run_single_model_writes_prompted_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = rq.ModelSpec(model_id="test/model", param_count_b=1.0)
    output_root = tmp_path / "results"
    downloads_root = tmp_path / "downloads"
    shared = _FakeSharedEwokModule()

    def _fake_download_assets(spec, *, download_root, hf_token, disable_xet, max_retries):
        del hf_token, disable_xet, max_retries
        download_dir = Path(download_root) / spec.model_slug
        model_dir = download_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return {
            "download_dir": download_dir,
            "model_dir": model_dir,
            "tokenizer_dir": model_dir,
            "tokenizer_source": spec.model_id,
            "tokenizer_mode": "model",
            "tokenizer_override_path": None,
            "weight_family": "safetensors",
            "ignore_patterns": [],
        }

    def _fake_prompted_records(**kwargs):
        del kwargs
        return [
            {
                "domain": "spatial-relations",
                "row_index": 0,
                "score_reduction": "mean",
                "prompt_inference_mode": "choice_answer_logprob",
                "prompt_template_name": "direct_choice",
                "prompt_template_source": "builtin:direct_choice",
                "gold_choice_c1": 1,
                "gold_choice_c2": 2,
                "predicted_choice_c1": 1,
                "predicted_choice_c2": 2,
                "response_text_c1": None,
                "response_text_c2": None,
                "response_valid_c1": True,
                "response_valid_c2": True,
                "choice_1_score_given_c1_prompt": -0.1,
                "choice_2_score_given_c1_prompt": -1.2,
                "choice_2_score_given_c2_prompt": -0.2,
                "choice_1_score_given_c2_prompt": -1.0,
                "margin_official_m1": 1.1,
                "margin_symmetric_m2": 0.8,
                "margin_combined": 0.95,
                "correct_official": True,
                "correct_symmetric": True,
                "correct_combined": True,
                "near_tie_official": False,
                "near_tie_symmetric": False,
                "near_tie_combined": False,
                "babylm_completion_choice_margin_official_m1": 1.1,
                "babylm_completion_choice_margin_symmetric_m2": 0.8,
                "babylm_completion_choice_margin_combined": 0.95,
                "babylm_completion_choice_correct_official": True,
                "babylm_completion_choice_correct_symmetric": True,
                "babylm_completion_choice_correct_combined": True,
                "babylm_completion_choice_near_tie_official": False,
                "babylm_completion_choice_near_tie_symmetric": False,
                "babylm_completion_choice_near_tie_combined": False,
                "ewok_context_sensitivity_margin_official_k1": 0.9,
                "ewok_context_sensitivity_margin_symmetric_k2": 1.0,
                "ewok_context_sensitivity_margin_combined": 0.95,
                "ewok_context_sensitivity_correct_official": True,
                "ewok_context_sensitivity_correct_symmetric": True,
                "ewok_context_sensitivity_correct_combined": True,
                "ewok_context_sensitivity_near_tie_official": False,
                "ewok_context_sensitivity_near_tie_symmetric": False,
                "ewok_context_sensitivity_near_tie_combined": False,
                "ewok_paper_context_sensitivity_margin_official_k1": 0.9,
                "ewok_paper_context_sensitivity_margin_symmetric_k2": 1.0,
                "ewok_paper_context_sensitivity_margin_combined": 0.95,
                "ewok_paper_context_sensitivity_correct_official": True,
                "ewok_paper_context_sensitivity_correct_symmetric": True,
                "ewok_paper_context_sensitivity_correct_combined": True,
                "ewok_paper_context_sensitivity_near_tie_official": False,
                "ewok_paper_context_sensitivity_near_tie_symmetric": False,
                "ewok_paper_context_sensitivity_near_tie_combined": False,
            }
        ]

    monkeypatch.setattr(pe.rq, "_download_assets", _fake_download_assets)
    monkeypatch.setattr(pe.rq, "_bitsandbytes_available", lambda: False)
    monkeypatch.setattr(pe.rq, "_load_model_and_tokenizer_for_attempt", lambda *args, **kwargs: (_FakeModel(), _FakeTokenizer()))
    monkeypatch.setattr(pe.rq, "detect_hardware", lambda: {"cuda_available": True, "num_gpus": 1, "gpus": [{"index": 0, "free_memory_bytes": 80 * 1024**3, "total_memory_bytes": 80 * 1024**3}]})
    monkeypatch.setattr(pe, "prompted_ewok_score_records_all_methods", _fake_prompted_records)

    summary = pe._run_single_model(
        spec=spec,
        output_root=output_root,
        downloads_root=downloads_root,
        dtype_name="auto",
        disable_xet=True,
        hf_token=None,
        download_retries=1,
        shared_ewok_module=shared,
        ewok_df=shared.ewok_df,
        variant="fast",
        domains=[],
        limit=None,
        prompt_template_name="direct_choice",
        prompt_template_source="builtin:direct_choice",
        template_text=pe.BUILTIN_PROMPT_TEMPLATES["direct_choice"],
        inference_mode="choice_answer_logprob",
        score_reduction="mean",
        margin_eps=1e-6,
        answer_separator="",
        max_new_tokens=4,
        target_permutation_mode="alternate",
        target_permutation_seed=11,
        overwrite=False,
        show_progress=False,
        store_prompts=False,
        emit_summary=False,
    )

    run_dir = output_root / "test_model__direct_choice__choice_answer_logprob__perm_alternate"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "ewok_metrics.json").read_text(encoding="utf-8"))
    items = [
        json.loads(line)
        for line in (run_dir / "ewok_items.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["status"] == "completed"
    assert summary["eval_method"] == "prompted_choice"
    assert summary["prompt_template_name"] == "direct_choice"
    assert summary["inference_mode"] == "choice_answer_logprob"
    assert summary["target_permutation_mode"] == "alternate"
    assert manifest["cleanup"]["removed_download_dir"] is True
    assert manifest["settings"]["target_permutation_mode"] == "alternate"
    assert manifest["settings"]["target_permutation_seed"] == 11
    assert metrics["prompting"]["inference_mode"] == "choice_answer_logprob"
    assert metrics["prompting"]["target_permutation_mode"] == "alternate"
    assert items[0]["babylm_completion_choice_correct_official"] is True
