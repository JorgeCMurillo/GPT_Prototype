from research.bos_aligned_proto.analysis.attribution.trackstar.per_query_top_examples import (
    extract_keyword_snippets,
    keywords_for_item,
    keyword_hits_in_text,
    query_action_terms_for_item,
)


def test_keywords_include_query_terms_and_default_material_terms() -> None:
    item = {
        "concept_a": "break",
        "concept_b": "drip",
        "context1": "Ali sees something that is rigid.",
        "context2": "Ali sees something that is liquid.",
        "target1": "It breaks.",
        "target2": "It drips.",
    }

    terms = keywords_for_item(item, extra_keywords=("nanotube, mesh",))

    assert "rigid" in terms
    assert "liquid" in terms
    assert "break" in terms
    assert "drip" in terms
    assert "material" in terms
    assert "nanotube" in terms
    assert "mesh" in terms
    assert "ali" not in terms


def test_extract_keyword_snippets_returns_non_overlapping_ranked_windows() -> None:
    text = (
        "This opening is generic. "
        "The new carbon nanotube material bends under load and detects ion signals. "
        + ("ordinary filler sentence with no relevant terms. " * 8)
        +
        "Another distant passage discusses water, oil, and fuel flow in soil."
    )

    snippets = extract_keyword_snippets(
        text,
        ("material", "carbon", "water", "oil", "fuel", "soil"),
        window_chars=70,
        max_snippets=2,
    )

    assert len(snippets) == 2
    assert {"carbon", "material"}.issubset(set(snippets[0]["matched_terms"]))
    assert {"water", "oil", "fuel", "soil"}.issubset(set(snippets[1]["matched_terms"]))
    assert "**carbon**" in snippets[0]["highlighted_text"]
    assert snippets[0]["char_end"] <= snippets[1]["char_start"]


def test_extract_keyword_snippets_handles_no_hits() -> None:
    assert extract_keyword_snippets("No useful words here.", ("material",), window_chars=80, max_snippets=2) == []


def test_query_action_terms_exclude_concepts_and_names() -> None:
    item = {
        "concept_a": "rug",
        "concept_b": "play-doh",
        "context1": "Ali sees something that is rigid.",
        "context2": "Ali sees something that is liquid.",
        "target1": "It breaks.",
        "target2": "It drips.",
    }

    terms = query_action_terms_for_item(item)

    assert terms == ("rigid", "liquid", "breaks", "drips")
    assert "rug" not in terms
    assert "play-doh" not in terms
    assert "ali" not in terms


def test_keyword_hits_in_text_returns_query_action_overlap() -> None:
    hits = keyword_hits_in_text(
        "The brittle surface breaks under load, while a liquid sample drips slowly.",
        ("rigid", "liquid", "breaks", "drips"),
    )

    assert hits == ("liquid", "breaks", "drips")
