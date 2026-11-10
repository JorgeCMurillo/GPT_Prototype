"""Document boundaries, model isolation, eval fallback, and resume invariants."""

import copy
from contextlib import nullcontext

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from research.bos_aligned_proto.training.config import parse_args
from training_utils import document_attention as attention


def reference_varlen(q, k, v, cu, maximum, scale):
    del maximum
    outputs = []
    indices = cu.tolist()
    repeats = q.shape[1] // k.shape[1]
    for start, end in zip(indices, indices[1:]):
        query = q[start:end].transpose(0, 1)
        key = k[start:end].transpose(0, 1).repeat_interleave(repeats, 0)
        value = v[start:end].transpose(0, 1).repeat_interleave(repeats, 0)
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scale,
        )
        outputs.append(output.transpose(0, 1))
    return torch.cat(outputs)


def model():
    config = Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32, bos_token_id=0,
        eos_token_id=0, pad_token_id=None, attention_dropout=0.0,
        tie_word_embeddings=True, use_cache=False,
    )
    return AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")


def kwargs(ids, mode="flash_varlen_document"):
    return attention.attention_forward_kwargs(ids, attention_mode=mode, bos_token_id=0, device="cpu")


def test_offsets_include_fragments_rows_consecutive_bos_and_last_token():
    ids = torch.tensor([[3, 4, 0, 0, 5, 0], [0, 6, 7, 8, 9, 0]])
    original = ids.clone()
    isolated = kwargs(ids)
    assert isolated["document_cu_seqlens"].tolist() == [0, 2, 3, 5, 6, 11, 12]
    assert isolated["document_cu_seqlens"].dtype == torch.int32
    assert isolated["document_max_seqlen"] == 5
    baseline = kwargs(ids, "flash_varlen")
    assert baseline["document_cu_seqlens"].tolist() == [0, 6, 12]
    assert baseline["document_max_seqlen"] == 6
    torch.testing.assert_close(ids, original)
    assert kwargs(ids, "sdpa") == {}


def test_attention_adapter_matches_full_model_and_gradients(monkeypatch):
    monkeypatch.setattr(attention, "_run_varlen", reference_varlen)
    torch.manual_seed(42)
    standard = model()
    packed = copy.deepcopy(standard)
    attention.configure_attention(packed, "flash_varlen")
    ids = torch.tensor([[0, 2, 3, 0, 4, 5], [6, 7, 0, 8, 9, 10]])
    targets = (ids + 1) % 32
    a = standard(ids).logits
    b = packed(ids, **kwargs(ids, "flash_varlen")).logits
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    for result, lm in ((a, standard), (b, packed)):
        torch.nn.functional.cross_entropy(result.reshape(-1, 32), targets.reshape(-1)).backward()
    for (name_a, pa), (name_b, pb) in zip(standard.named_parameters(), packed.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-4, atol=1e-6)


def test_full_model_isolation_causality_and_fragment_equivalence(monkeypatch):
    monkeypatch.setattr(attention, "_run_varlen", reference_varlen)
    torch.manual_seed(7)
    lm = model()
    attention.configure_attention(lm, "flash_varlen_document")
    ids = torch.tensor([[1, 2, 0, 4, 5, 6, 0]])
    output = lm(ids, **kwargs(ids)).logits
    changed = ids.clone()
    changed[0, :2] = torch.tensor([11, 12])
    modified = lm(changed, **kwargs(changed)).logits
    torch.testing.assert_close(output[:, 2:], modified[:, 2:], rtol=0, atol=0)
    # The second document matches an independent forward at the original positions.
    fragment = lm(ids[:, 2:6], position_ids=torch.arange(2, 6)[None]).logits
    torch.testing.assert_close(output[:, 2:6], fragment, rtol=1e-5, atol=1e-6)
    future_changed = ids.clone()
    future_changed[0, 5] = 13
    future_output = lm(future_changed, **kwargs(future_changed)).logits
    torch.testing.assert_close(output[:, :5], future_output[:, :5], rtol=0, atol=0)
    # Backpropagating from the later document cannot reach earlier token embeddings.
    output[:, 2:6, 3].sum().backward()
    # Tied output embeddings also receive gradients; inspect input hidden states instead.
    embeddings = lm.get_input_embeddings()(ids).detach().requires_grad_()
    lm.zero_grad(set_to_none=True)
    isolated = lm(inputs_embeds=embeddings, **kwargs(ids)).logits
    isolated[:, 2:6, 3].sum().backward()
    assert embeddings.grad[:, :2].abs().max().item() == 0
    assert embeddings.grad[:, 2:6].abs().max().item() > 0
    ordinary = lm(ids).logits
    ordinary_changed = lm(changed).logits
    assert (ordinary[:, 2:6] - ordinary_changed[:, 2:6]).abs().max().item() > 1e-5


def test_padding_and_cached_inference_keep_standard_sdpa_behavior():
    torch.manual_seed(42)
    standard = model()
    adapted = copy.deepcopy(standard)
    attention.configure_attention(adapted, "flash_varlen_document")
    ids = torch.tensor([[0, 2, 3, 4], [0, 5, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    torch.testing.assert_close(standard(ids, attention_mask=mask).logits,
                               adapted(ids, attention_mask=mask).logits)
    for lm in (standard, adapted):
        lm.config.use_cache = True
    a = standard(ids[:1, :3], use_cache=True)
    b = adapted(ids[:1, :3], use_cache=True)
    next_a = standard(ids[:1, 3:], past_key_values=a.past_key_values, use_cache=True)
    next_b = adapted(ids[:1, 3:], past_key_values=b.past_key_values, use_cache=True)
    torch.testing.assert_close(next_a.logits, next_b.logits)


def test_config_defaults_and_resume_mode_are_explicit():
    assert parse_args(["--data_dir", "/tmp/unused"]).attention_mode == "sdpa"
    assert parse_args(["--data_dir", "/tmp/unused", "--attention_mode",
                       "flash_varlen_document"]).attention_mode == "flash_varlen_document"
    attention.validate_attention_resume({}, "sdpa")
    attention.validate_attention_resume({"attention_mode": "flash_varlen_document"}, "flash_varlen_document")
    with pytest.raises(ValueError, match="Resume attention_mode mismatch"):
        attention.validate_attention_resume({}, "flash_varlen_document")
    with pytest.raises(ValueError, match="Resume attention_mode mismatch"):
        attention.validate_attention_resume({"attention_mode": "flash_varlen_document"}, "flash_varlen")


def test_validation_compares_both_views_without_changing_primary_loss(monkeypatch):
    from evaluation.runner import run_parallel_validation

    monkeypatch.setattr(attention, "_run_varlen", reference_varlen)
    torch.manual_seed(42)
    lm = model()
    attention.configure_attention(lm, "flash_varlen_document")
    ids = torch.tensor([[0, 2, 3, 0, 4, 5]])
    labels = torch.tensor([[2, 3, 0, 4, 5, 6]])
    with torch.no_grad():
        ordinary = torch.nn.functional.cross_entropy(
            lm(ids).logits.reshape(-1, 32), labels.reshape(-1)).item()
        isolated = torch.nn.functional.cross_entropy(
            lm(ids, **kwargs(ids)).logits.reshape(-1, 32), labels.reshape(-1)).item()

    class Accelerator:
        is_main_process = True
        is_local_main_process = True

        def wait_for_everyone(self):
            pass

        def reduce(self, value, reduction):
            assert reduction == "sum"
            return value

    rows = []
    result = run_parallel_validation(
        accelerator=Accelerator(), model=lm, val_loader=[(ids, labels)],
        device="cpu", autocast_ctx=nullcontext, sdpa_backends=[torch.nn.attention.SDPBackend.MATH],
        loss_vocab_size=32, step=1, val_batch_limit=1,
        append_jsonl_fn=lambda path, row: rows.append(row), scalars_path="unused",
        attention_mode="flash_varlen_document", document_bos_token_id=0,
    )
    assert result == pytest.approx(ordinary, abs=1e-6)
    assert rows[0]["val_loss"] == pytest.approx(ordinary, abs=1e-6)
    assert rows[0]["document_isolated_val_loss"] == pytest.approx(isolated, abs=1e-6)
    assert lm.training


def test_checkpoint_loads_without_registering_training_backend(tmp_path):
    lm = model()
    attention.configure_attention(lm, "flash_varlen_document")
    lm.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, attn_implementation="sdpa", local_files_only=True)
    assert loaded.config.pretraining_attention_mode == "flash_varlen_document"
    assert loaded.config._attn_implementation == "sdpa"
    torch.testing.assert_close(lm(torch.tensor([[0, 1, 2]])).logits,
                               loaded(torch.tensor([[0, 1, 2]])).logits)
