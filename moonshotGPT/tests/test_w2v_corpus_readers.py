from pathlib import Path

import pytest

from research.w2v_lexical_probe.corpus import (
    TextLineCorpusConfig,
    TextLineCorpusReader,
    build_corpus_reader,
)
from research.w2v_lexical_probe.compare_ewok_word2vec_runs import extract_scores
from research.w2v_lexical_probe.model import Word2VecTrainingConfig
from research.w2v_lexical_probe.run_fineweb_word_budget_ewok_ci import summarize_domain_scores
from research.w2v_lexical_probe.train_word2vec import _build_run_name


def test_text_line_corpus_reader_yields_nonempty_lines(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "babylm_text"
    corpus_dir.mkdir()
    (corpus_dir / "a.train").write_text("first sentence\n\nsecond sentence\n", encoding="utf-8")
    (corpus_dir / "b.train").write_text("third sentence\n", encoding="utf-8")

    reader = TextLineCorpusReader(
        TextLineCorpusConfig(
            data_dir=str(corpus_dir),
            glob_pattern="*.train",
            max_docs=2,
        )
    )

    assert list(reader.iter_documents()) == ["first sentence", "second sentence"]
    assert reader.source_name == "babylm_text"

    description = reader.describe()
    assert description["corpus_format"] == "raw_text_lines"
    assert description["baseline_role"] == "text_corpus_lexical_baseline"
    assert description["max_docs"] == 2
    assert len(description["files"]) == 2


def test_build_corpus_reader_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown corpus_format"):
        build_corpus_reader(corpus_format="spreadsheet", data_dir=str(tmp_path))


def test_text_line_run_name_includes_corpus_format(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "train_10M"
    corpus_dir.mkdir()
    (corpus_dir / "sample.train").write_text("small text\n", encoding="utf-8")
    config, reader = build_corpus_reader(
        corpus_format="text_dir",
        data_dir=str(corpus_dir),
        max_docs=10,
    )

    run_name = _build_run_name(
        reader,
        Word2VecTrainingConfig(
            embedding_dim=64,
            window_size=3,
            negative_samples=2,
            min_count=1,
            epochs=1,
            seed=7,
        ),
        config,
    )

    assert run_name == "sgns_train_10M_raw_text_lines_d64_w3_neg2_mc1_ep1_seed7_docs10"


def test_extract_scores_reads_pair_average_and_combined_metrics() -> None:
    payload = {
        "metrics_by_method": {
            "babylm_completion_choice": {
                "domain_scores_full": {
                    "physical-dynamics": [0.25, 0.75],
                    "average": [0.4, 0.6],
                },
                "domain_margin_stats": {
                    "physical-dynamics": {"acc_combined": 0.55},
                    "average": {"acc_combined": 0.5},
                },
            }
        }
    }

    pair_scores = extract_scores(
        payload,
        method="babylm_completion_choice",
        score_kind="pair_average",
    )
    combined_scores = extract_scores(
        payload,
        method="babylm_completion_choice",
        score_kind="combined",
    )

    assert pair_scores["physical-dynamics"] == 0.5
    assert pair_scores["average"] == 0.5
    assert combined_scores["physical-dynamics"] == 0.55


def test_summarize_domain_scores_uses_replicate_variation_for_ci() -> None:
    rows = [
        {"word_budget": 10, "domain": "physical-dynamics", "score": 0.4},
        {"word_budget": 10, "domain": "physical-dynamics", "score": 0.6},
        {"word_budget": 10, "domain": "physical-dynamics", "score": 0.5},
        {"word_budget": 10, "domain": "physical-dynamics", "score": 0.7},
        {"word_budget": 10, "domain": "physical-dynamics", "score": 0.3},
    ]

    summary = summarize_domain_scores(rows)

    assert len(summary) == 1
    assert summary[0]["n"] == 5
    assert summary[0]["mean"] == pytest.approx(0.5)
    assert summary[0]["ci95_low"] < 0.5
    assert summary[0]["ci95_high"] > 0.5
