from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

ewok = import_module("moonshotGPT.evaluation.ewok")


def _record(domain: str, pmi_m1: float, pmi_m2: float) -> dict[str, object]:
    pmi_m = 0.5 * (float(pmi_m1) + float(pmi_m2))
    return {
        "domain": domain,
        "margin_official_m1": 1.0,
        "margin_symmetric_m2": 1.0,
        "margin_combined": 1.0,
        "ewok_context_sensitivity_margin_official_k1": 1.0,
        "ewok_context_sensitivity_margin_symmetric_k2": 1.0,
        "ewok_context_sensitivity_margin_combined": 1.0,
        "pmi_completion_choice_margin_official_m1": float(pmi_m1),
        "pmi_completion_choice_margin_symmetric_m2": float(pmi_m2),
        "pmi_completion_choice_margin_combined": float(pmi_m),
    }


def test_shared_summary_adds_pmi_completion_choice_when_records_have_pmi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ewok, "ewok_df", pd.DataFrame({"Domain": ["domain-a", "domain-b"]}))
    records = [
        _record("domain-a", 1.0, 1.0),
        _record("domain-a", -1.0, 1.0),
        _record("domain-b", 1.0, -1.0),
        _record("domain-b", 1.0, -1.0),
    ]

    metrics = ewok._summarize_records_all_methods(records, margin_eps=1e-6)

    assert ewok.PMI_COMPLETION_CHOICE in metrics
    pmi = metrics[ewok.PMI_COMPLETION_CHOICE]
    assert pmi["domain_scores_full"]["domain-a"] == pytest.approx((0.5, 1.0))
    assert pmi["domain_scores_full"]["domain-b"] == pytest.approx((1.0, 0.0))
    assert pmi["domain_scores_full"]["average"] == pytest.approx((0.75, 0.5))
    assert pmi["domain_margin_stats"]["average"]["acc_combined"] == pytest.approx(0.625)


def test_shared_summary_skips_pmi_completion_choice_without_pmi_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ewok, "ewok_df", pd.DataFrame({"Domain": ["domain-a"]}))
    record = _record("domain-a", 1.0, 1.0)
    for key in list(record):
        if str(key).startswith("pmi_completion_choice_"):
            del record[key]

    metrics = ewok._summarize_records_all_methods([record], margin_eps=1e-6)

    assert ewok.PMI_COMPLETION_CHOICE not in metrics


@pytest.mark.parametrize("method", [
    ewok.BABYLM_COMPLETION_CHOICE,
    ewok.EWOK_CONTEXT_SENSITIVITY,
    ewok.PMI_COMPLETION_CHOICE,
])
def test_combined_accuracy_counts_decisions_not_average_margin(monkeypatch, method) -> None:
    monkeypatch.setattr(ewok, "ewok_df", pd.DataFrame({"Domain": ["domain-a"]}))
    record = _record("domain-a", 2.0, -1.0)
    record.update({
        "margin_official_m1": 2.0,
        "margin_symmetric_m2": -1.0,
        "margin_combined": 0.5,
        "ewok_context_sensitivity_margin_official_k1": 2.0,
        "ewok_context_sensitivity_margin_symmetric_k2": -1.0,
        "ewok_context_sensitivity_margin_combined": 0.5,
    })

    metrics = ewok._summarize_records_all_methods([record], margin_eps=1e-6)

    assert metrics[method]["domain_margin_stats"]["average"]["acc_combined"] == pytest.approx(0.5)
