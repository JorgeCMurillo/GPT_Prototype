import pandas as pd

from research.bos_aligned_proto.spatial_synth.generate_ewok_answer_exposure_csv import iter_answer_rows


def test_iter_answer_rows_emits_correct_c1_t1_and_c2_t2_pairs() -> None:
    df = pd.DataFrame(
        [
            {
                "Domain": "spatial-relations",
                "ConceptA": "above",
                "ConceptB": "below",
                "ContextType": "direct",
                "ContextDiff": "antonym",
                "TargetDiff": "concept swap",
                "Context1": "The baseball is below the candle.",
                "Context2": "The baseball is above the candle.",
                "Target1": "The candle is above the baseball.",
                "Target2": "The candle is below the baseball.",
            }
        ]
    )

    rows = list(iter_answer_rows(df, ewok_variant="fast", sides=("official", "symmetric")))

    assert [row["answer_side"] for row in rows] == ["C1_T1", "C2_T2"]
    assert rows[0]["context"] == "The baseball is below the candle."
    assert rows[0]["completion"] == "The candle is above the baseball."
    assert rows[0]["text"] == "The baseball is below the candle. The candle is above the baseball."
    assert rows[1]["context"] == "The baseball is above the candle."
    assert rows[1]["completion"] == "The candle is below the baseball."
    assert rows[1]["text"] == "The baseball is above the candle. The candle is below the baseball."


def test_iter_answer_rows_can_filter_to_spatial_domain() -> None:
    df = pd.DataFrame(
        [
            {"Domain": "spatial-relations", "Context1": "C1.", "Context2": "C2.", "Target1": "T1.", "Target2": "T2."},
            {"Domain": "social-relations", "Context1": "S1.", "Context2": "S2.", "Target1": "U1.", "Target2": "U2."},
        ]
    )

    rows = list(iter_answer_rows(df, ewok_variant="fast", sides=("official",), domains=("spatial-relations",)))

    assert len(rows) == 1
    assert rows[0]["domain"] == "spatial-relations"
    assert rows[0]["answer_side"] == "C1_T1"
