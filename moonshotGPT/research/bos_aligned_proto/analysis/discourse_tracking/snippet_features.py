"""Sentence-snippet feature extraction for discourse selectors."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .features import (
    ENTITY_PATTERN,
    TITLECASE_STOPWORDS,
    WORD_PATTERN,
    canonicalize_entity_text,
    compute_spacy_doc_features,
    compute_text_features,
)
from .sentence_windows import split_sentences_regex


ATTRIBUTE_WORDS = {
    "big",
    "small",
    "large",
    "little",
    "fragile",
    "sturdy",
    "soft",
    "hard",
    "bouncy",
    "transparent",
    "opaque",
    "cold",
    "warm",
    "hot",
    "smooth",
    "rough",
    "heavy",
    "light",
    "wet",
    "dry",
    "open",
    "closed",
    "empty",
    "full",
    "shy",
    "confident",
    "boastful",
    "humble",
    "dominant",
    "submissive",
    "trustworthy",
    "untrustworthy",
}

CHANGE_VERBS = {
    "accelerate",
    "accelerated",
    "accelerates",
    "accelerating",
    "became",
    "began",
    "begin",
    "beginning",
    "begins",
    "begun",
    "become",
    "becomes",
    "break",
    "breaking",
    "broke",
    "broken",
    "burn",
    "burned",
    "burning",
    "burns",
    "burnt",
    "change",
    "changed",
    "changes",
    "changing",
    "closes",
    "closing",
    "contract",
    "contracted",
    "contracting",
    "contracts",
    "cool",
    "cooled",
    "cooling",
    "cools",
    "crack",
    "cracked",
    "cracking",
    "cracks",
    "decrease",
    "decreased",
    "decreases",
    "decreasing",
    "dried",
    "drying",
    "dries",
    "drop",
    "dropped",
    "dropping",
    "drops",
    "emptied",
    "empties",
    "emptying",
    "expand",
    "expanded",
    "expanding",
    "expands",
    "fall",
    "falling",
    "fell",
    "falls",
    "fill",
    "filled",
    "filling",
    "fills",
    "fix",
    "fixed",
    "fixes",
    "fixing",
    "float",
    "floated",
    "floating",
    "floats",
    "freeze",
    "freezes",
    "freezing",
    "froze",
    "frozen",
    "grow",
    "grew",
    "growing",
    "grows",
    "increase",
    "increased",
    "increases",
    "increasing",
    "melt",
    "melted",
    "melting",
    "melts",
    "move",
    "moved",
    "moves",
    "moving",
    "opened",
    "opening",
    "opens",
    "rise",
    "rising",
    "rises",
    "rose",
    "roll",
    "rolled",
    "rolling",
    "rolls",
    "shrink",
    "shrank",
    "shrinking",
    "shrinks",
    "shrunk",
    "sink",
    "sank",
    "sinking",
    "sinks",
    "sunk",
    "slide",
    "slid",
    "slides",
    "sliding",
    "slow",
    "slowed",
    "slowing",
    "slows",
    "split",
    "splits",
    "splitting",
    "start",
    "started",
    "starting",
    "starts",
    "stop",
    "stopped",
    "stopping",
    "stops",
    "tear",
    "tearing",
    "tears",
    "thaw",
    "thawed",
    "thawing",
    "thaws",
    "tore",
    "torn",
    "turn",
    "turned",
    "turning",
    "turns",
    "warm",
    "warmed",
    "warming",
    "warms",
}
CHANGE_VERB_LEMMAS = {
    "accelerate",
    "become",
    "begin",
    "break",
    "burn",
    "change",
    "close",
    "contract",
    "cool",
    "crack",
    "decrease",
    "drop",
    "dry",
    "empty",
    "expand",
    "fall",
    "fill",
    "fix",
    "float",
    "freeze",
    "grow",
    "increase",
    "melt",
    "move",
    "open",
    "rise",
    "roll",
    "shrink",
    "sink",
    "slide",
    "slow",
    "split",
    "start",
    "stop",
    "tear",
    "thaw",
    "turn",
    "warm",
}

TEMPORAL_MARKERS = {
    "after",
    "afterward",
    "afterwards",
    "before",
    "eventually",
    "finally",
    "later",
    "next",
    "once",
    "then",
    "until",
    "when",
    "while",
}

MENTAL_STATE_WORDS = {
    "believe",
    "believed",
    "believes",
    "doubt",
    "doubted",
    "doubts",
    "feel",
    "felt",
    "feels",
    "hate",
    "hated",
    "hates",
    "hear",
    "heard",
    "hope",
    "hoped",
    "hopes",
    "imagine",
    "imagined",
    "imagines",
    "intend",
    "intended",
    "intends",
    "know",
    "knew",
    "knows",
    "like",
    "liked",
    "likes",
    "prefer",
    "preferred",
    "prefers",
    "remember",
    "remembered",
    "remembers",
    "see",
    "saw",
    "sees",
    "think",
    "thought",
    "thinks",
    "want",
    "wanted",
    "wants",
}
MENTAL_STATE_LEMMAS = {
    "believe",
    "doubt",
    "feel",
    "hate",
    "hear",
    "hope",
    "imagine",
    "intend",
    "know",
    "like",
    "prefer",
    "remember",
    "see",
    "think",
    "want",
}

PREFERENCE_GOAL_INTENT_WORDS = {
    "choose",
    "chooses",
    "chose",
    "decide",
    "decided",
    "decides",
    "hope",
    "hoped",
    "hopes",
    "intend",
    "intended",
    "intends",
    "prefer",
    "preferred",
    "prefers",
    "want",
    "wanted",
    "wants",
}
PREFERENCE_GOAL_INTENT_LEMMAS = {
    "choose",
    "decide",
    "hope",
    "intend",
    "prefer",
    "want",
}
WEAK_INTERNAL_STATE_WORDS = {
    "feel",
    "felt",
    "feels",
    "hate",
    "hated",
    "hates",
    "hear",
    "heard",
    "like",
    "liked",
    "likes",
    "see",
    "saw",
    "sees",
}
WEAK_INTERNAL_STATE_LEMMAS = {
    "feel",
    "hate",
    "hear",
    "like",
    "see",
}
STRONG_INTERNAL_STATE_WORDS = (MENTAL_STATE_WORDS - WEAK_INTERNAL_STATE_WORDS) | PREFERENCE_GOAL_INTENT_WORDS
STRONG_INTERNAL_STATE_LEMMAS = (
    MENTAL_STATE_LEMMAS - WEAK_INTERNAL_STATE_LEMMAS
) | PREFERENCE_GOAL_INTENT_LEMMAS

RELATION_CUE_WORDS = {
    "accused",
    "accuses",
    "affected",
    "affects",
    "appointed",
    "appoints",
    "arrested",
    "arrests",
    "asked",
    "asks",
    "attacked",
    "attacks",
    "blamed",
    "blames",
    "bought",
    "builds",
    "built",
    "buys",
    "called",
    "calls",
    "captured",
    "captures",
    "caused",
    "causes",
    "chased",
    "chases",
    "commanded",
    "commands",
    "comforted",
    "comforts",
    "created",
    "creates",
    "defeated",
    "defeats",
    "fired",
    "fires",
    "followed",
    "follows",
    "founded",
    "founds",
    "gave",
    "gives",
    "governed",
    "governs",
    "helped",
    "helps",
    "hired",
    "hires",
    "hindered",
    "hinders",
    "insulted",
    "insults",
    "invaded",
    "invades",
    "joined",
    "joins",
    "killed",
    "kills",
    "led",
    "leads",
    "left",
    "leaves",
    "married",
    "marries",
    "met",
    "meets",
    "obeyed",
    "obeys",
    "opposed",
    "opposes",
    "owned",
    "owns",
    "paid",
    "pays",
    "pushed",
    "pushes",
    "questioned",
    "questions",
    "replied",
    "replies",
    "returned",
    "returns",
    "recruited",
    "recruits",
    "replaced",
    "replaces",
    "ruled",
    "rules",
    "saw",
    "sees",
    "sold",
    "sells",
    "showed",
    "shows",
    "succeeded",
    "succeeds",
    "supported",
    "supports",
    "supervised",
    "supervises",
    "taught",
    "teaches",
    "thanked",
    "thanks",
    "told",
    "tells",
    "trained",
    "trains",
    "warned",
    "warns",
}

ATTRIBUTE_COPULA_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|became|becomes|seemed|seems|looked|looks|felt|feels|remained|remains)\b",
    re.IGNORECASE,
)
STATE_COMPLEMENT_RE = re.compile(
    r"\b(?:believ(?:e|ed|es)|doubt(?:ed|s)?|think(?:s|ing)?|thought|know(?:s)?|knew|"
    r"want(?:ed|s)?|intend(?:ed|s)?|hope(?:d|s)?|decide(?:d|s)?)\s+(?:that|to)\b",
    re.IGNORECASE,
)
SECOND_PERSON_INSTRUCTION_RE = re.compile(
    r"\b(?:if|when|once|after)?\s*you\s+(?:just\s+|really\s+)?"
    r"(?:can|could|should|must|want|need|choose|decide|prefer|intend)\b",
    re.IGNORECASE,
)
TUTORIAL_MARKER_RE = re.compile(r"\b(?:step\s*\d+|guide|tutorial|instructions?)\b", re.IGNORECASE)
PATENT_INTENT_RE = re.compile(
    r"\b(?:preferred embodiments?|embodiments? of the invention|some embodiments?|"
    r"intended to (?:encompass|include|cover|be used|be limiting|limit)|"
    r"not intended to(?: be)? (?:limiting|limit)|"
    r"within the scope of (?:some embodiments?|the invention))\b",
    re.IGNORECASE,
)
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d{1,3}[.)]|[A-Za-z][.)])\s+")
DENSE_LIST_SEPARATOR_RE = re.compile(r"[\u2012\u2013\u2014;]\s*")
INLINE_LIST_GLYPH_RE = re.compile(r"[\u2022\u2023\u25E6\u2043\u2219\u25AA\u25AB\u25CF\u25CB\u25A0\u25A1]")
DOI_RE = re.compile(r"\b(?:doi:?\s*)?10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ET_AL_RE = re.compile(r"\bet\s+al\.?", re.IGNORECASE)
AUTHOR_INITIAL_RE = re.compile(r"\b[A-Z][a-zA-Z'\u00C0-\u024F-]+(?:\s+[A-Z]){1,3}\b")
CITATION_YEAR_VOLUME_RE = re.compile(r"\b(?:19|20)\d{2},\s*\d{1,4}\s*:")
CATALOG_MARKER_RE = re.compile(
    r"\b(?:manufacturer|exporter|supplier|price|products?|services?|specifications?|"
    r"overview|contact supplier|get price|see all results for this question|"
    r"schematic diagram|model|itemtrade|crusher|mill|pdf)\b",
    re.IGNORECASE,
)
FIELD_LABEL_RE = re.compile(
    r"\b(?:advantages?|disadvantages?|features?|supplies|instructions?|overview|"
    r"keywords?|results?|materials?|tools?|chapter|figure|table|step\s*\d+)\s*:",
    re.IGNORECASE,
)
SYMPTOM_MARKER_RE = re.compile(
    r"\b(?:symptoms?|painful?|pains?|swelling|redness|dryness|sensation|burning|"
    r"stinging|thirst|appetite|nausea|vomiting|fever|headache|cough|mucus|"
    r"ulcers?|inflammation|tonsils?|throat|stomach|salivation|deglutition|"
    r"spasmodic|suffocation)\b",
    re.IGNORECASE,
)
LIST_NOISE_GATE_MAX = 0.85
HEAVY_LIST_NOISE_GATE_MAX = 0.95
BIBLIOGRAPHY_NOISE_GATE_MAX = 0.85
REPEATED_3GRAM_GATE_MAX = 0.20
DUPLICATE_SENTENCE_GATE_MAX = 0.25

def _word_counts(text: str) -> tuple[list[str], dict[str, int]]:
    words = [token.lower() for token in WORD_PATTERN.findall(str(text))]
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return words, counts


def _count_lexicon(words: Sequence[str], lexicon: set[str]) -> int:
    return int(sum(1 for word in words if word in lexicon))


def _count_internal_state_lexicon(text: str, lexicon: set[str]) -> int:
    count = 0
    for token in WORD_PATTERN.findall(str(text)):
        lower = token.lower()
        if lower not in lexicon:
            continue
        if lower in {"hope", "hopes"} and token[:1].isupper():
            continue
        count += 1
    return int(count)


def _contains_internal_state_word(text: str, lexicon: set[str]) -> bool:
    return _count_internal_state_lexicon(text, lexicon) > 0


def _entity_spans(sentence: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in ENTITY_PATTERN.finditer(str(sentence)):
        raw = match.group(0).strip()
        if raw in TITLECASE_STOPWORDS:
            continue
        canonical = canonicalize_entity_text(raw)
        if canonical and canonical not in TITLECASE_STOPWORDS:
            spans.append((canonical, int(match.start()), int(match.end())))
    return spans


def _sentences_for_features(text: str) -> list[str]:
    return [span.text for span in split_sentences_regex(text)]


def _contains_any_word(text: str, words: set[str]) -> bool:
    tokens = {token.lower() for token in WORD_PATTERN.findall(str(text))}
    return bool(tokens & words)


def _directed_relation_edges(sentences: Sequence[str]) -> list[tuple[str, str, str, int]]:
    edges: list[tuple[str, str, str, int]] = []
    for sentence_idx, sentence in enumerate(sentences):
        entities = _entity_spans(sentence)
        if len(entities) < 2:
            continue
        words = [word.lower() for word in WORD_PATTERN.findall(sentence)]
        cues = [word for word in words if word in RELATION_CUE_WORDS]
        if not cues:
            continue
        predicate = cues[0]
        for left_idx, (left, _left_start, left_end) in enumerate(entities):
            for right, right_start, _right_end in entities[left_idx + 1 :]:
                between = sentence[left_end:right_start].lower()
                if predicate in between or cues:
                    edges.append((left, right, predicate, int(sentence_idx)))
                    break
    return edges


def _attribute_features(text: str, sentences: Sequence[str], sentence_count: int) -> dict[str, float | int]:
    words, _ = _word_counts(text)
    property_word_count = _count_lexicon(words, ATTRIBUTE_WORDS)
    unique_attribute_count = len({word for word in words if word in ATTRIBUTE_WORDS})
    entity_attribute_edge_count = 0
    attribute_sentence_count = 0
    entities_with_attribute: set[str] = set()
    entities_seen: set[str] = set()
    for sentence in sentences:
        sentence_entities = {entity for entity, _start, _end in _entity_spans(sentence)}
        entities_seen.update(sentence_entities)
        sentence_has_attribute = _contains_any_word(sentence, ATTRIBUTE_WORDS)
        if sentence_has_attribute:
            attribute_sentence_count += 1
        if sentence_entities and sentence_has_attribute:
            edge_count = max(1, len(sentence_entities)) if ATTRIBUTE_COPULA_RE.search(sentence) else 1
            entity_attribute_edge_count += edge_count
            entities_with_attribute.update(sentence_entities)
    return {
        "property_word_count": int(property_word_count),
        "unique_attribute_count": int(unique_attribute_count),
        "entity_attribute_edge_count": int(entity_attribute_edge_count),
        "attribute_sentence_count": int(attribute_sentence_count),
        "attribute_density": float(max(entity_attribute_edge_count, property_word_count) / max(1, sentence_count)),
        "property_word_density": float(property_word_count / max(1, len(words))),
        "entity_with_attribute_fraction": float(len(entities_with_attribute) / max(1, len(entities_seen))),
    }


def _relation_role_features(sentences: Sequence[str], sentence_count: int) -> dict[str, float | int]:
    edges = _directed_relation_edges(sentences)
    two_entity_relation_sentences = 0
    possessive_relation_count = 0
    for sentence in sentences:
        if len({entity for entity, _start, _end in _entity_spans(sentence)}) >= 2 and _contains_any_word(sentence, RELATION_CUE_WORDS):
            two_entity_relation_sentences += 1
        possessive_relation_count += len(re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*'s\b", sentence))

    unordered_pairs: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for subject, obj, _predicate, _sentence_idx in edges:
        key = tuple(sorted((subject, obj)))
        unordered_pairs.setdefault(key, set()).add((subject, obj))
    reversed_pair_count = sum(1 for directions in unordered_pairs.values() if len(directions) >= 2)
    same_pair_multi_relation_count = sum(1 for directions in unordered_pairs.values() if directions)
    return {
        "directed_relation_count": int(len(edges)),
        "directed_relation_density": float(len(edges) / max(1, sentence_count)),
        "two_entity_relation_sentence_fraction": float(two_entity_relation_sentences / max(1, sentence_count)),
        "possessive_relation_count": int(possessive_relation_count),
        "possessive_relation_density": float(possessive_relation_count / max(1, sentence_count)),
        "reversed_pair_count": int(reversed_pair_count),
        "role_alternating_pair_count": int(reversed_pair_count),
        "same_pair_multi_relation_count": int(same_pair_multi_relation_count),
        "reversal_fraction": float(reversed_pair_count / max(1, len(unordered_pairs))),
    }


def _layout_noise_features(text: str, sentence_count: int) -> dict[str, float | int]:
    raw = str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    line_count = len(lines)
    list_marker_count = sum(1 for line in lines if LIST_MARKER_RE.match(line))
    inline_list_glyph_count = len(INLINE_LIST_GLYPH_RE.findall(raw))
    doi_count = len(DOI_RE.findall(raw))
    et_al_count = len(ET_AL_RE.findall(raw))
    author_initial_count = len(AUTHOR_INITIAL_RE.findall(raw))
    citation_year_volume_count = len(CITATION_YEAR_VOLUME_RE.findall(raw))
    pipe_char_count = raw.count("|")
    pipe_table_line_count = sum(1 for line in lines if line.count("|") >= 2)
    catalog_marker_count = len(CATALOG_MARKER_RE.findall(raw))
    field_label_count = len(FIELD_LABEL_RE.findall(raw))
    symptom_marker_count = len(SYMPTOM_MARKER_RE.findall(raw))
    symptom_separator_count = raw.count(".-") + raw.count(";")
    short_structured_line_count = sum(
        1
        for line in lines
        if len(line) <= 90 and not line.endswith((".", "?", "!")) and len(WORD_PATTERN.findall(line)) <= 8
    )
    dense_separator_count = len(DENSE_LIST_SEPARATOR_RE.findall(raw))
    newline_count = raw.count("\n")
    bullet_line_fraction = float(list_marker_count / max(1, line_count))
    short_line_fraction = float(short_structured_line_count / max(1, line_count)) if line_count >= 6 else 0.0
    newline_density = float(newline_count / max(1, sentence_count))
    dense_separator_density = float(dense_separator_count / max(1, sentence_count))
    inline_list_glyph_density = float(inline_list_glyph_count / max(1, sentence_count))
    pipe_table_noise_score = min(
        1.0,
        max(
            pipe_table_line_count / 4.0,
            pipe_char_count / max(20.0, float(sentence_count * 12)),
        ),
    )
    catalog_noise_score = min(1.0, (catalog_marker_count + field_label_count) / 6.0)
    symptom_list_context_score = min(
        1.0,
        max(
            symptom_separator_count / 8.0,
            list_marker_count / 4.0,
            inline_list_glyph_count / 8.0,
            short_line_fraction,
        ),
    )
    symptom_list_noise_score = min(1.0, symptom_marker_count / 10.0) * symptom_list_context_score
    table_catalog_symptom_noise_score = min(
        1.0,
        max(pipe_table_noise_score, catalog_noise_score, symptom_list_noise_score),
    )
    bibliography_noise_score = min(
        1.0,
        max(
            float(doi_count > 0),
            float(et_al_count > 0),
            min(1.0, citation_year_volume_count / 2.0),
            min(1.0, author_initial_count / 12.0),
        ),
    )
    layout_noise_score = min(
        1.0,
        max(
            bullet_line_fraction,
            min(1.0, list_marker_count / 4.0),
            min(1.0, inline_list_glyph_count / 8.0),
            short_line_fraction,
            min(1.0, newline_density / 4.0),
            min(1.0, dense_separator_density / 10.0),
            min(1.0, inline_list_glyph_density / 4.0),
        ),
    )
    return {
        "line_count": int(line_count),
        "list_marker_count": int(list_marker_count),
        "inline_list_glyph_count": int(inline_list_glyph_count),
        "short_structured_line_count": int(short_structured_line_count),
        "doi_count": int(doi_count),
        "et_al_count": int(et_al_count),
        "author_initial_count": int(author_initial_count),
        "citation_year_volume_count": int(citation_year_volume_count),
        "pipe_char_count": int(pipe_char_count),
        "pipe_table_line_count": int(pipe_table_line_count),
        "catalog_marker_count": int(catalog_marker_count),
        "field_label_count": int(field_label_count),
        "symptom_marker_count": int(symptom_marker_count),
        "symptom_separator_count": int(symptom_separator_count),
        "bullet_line_fraction": float(bullet_line_fraction),
        "newline_density": float(newline_density),
        "dense_separator_density": float(dense_separator_density),
        "inline_list_glyph_density": float(inline_list_glyph_density),
        "pipe_table_noise_score": float(pipe_table_noise_score),
        "catalog_noise_score": float(catalog_noise_score),
        "symptom_list_noise_score": float(symptom_list_noise_score),
        "table_catalog_symptom_noise_score": float(table_catalog_symptom_noise_score),
        "bibliography_noise_score": float(bibliography_noise_score),
        "layout_noise_score": float(layout_noise_score),
    }


def _state_update_features(text: str, sentences: Sequence[str], sentence_count: int) -> dict[str, float | int]:
    words, _ = _word_counts(text)
    change_verb_count = _count_lexicon(words, CHANGE_VERBS)
    temporal_marker_count = _count_lexicon(words, TEMPORAL_MARKERS)
    before_after_marker_count = sum(1 for word in words if word in {"before", "after"})
    result_state_pattern_count = len(
        re.findall(r"\b(?:became|becomes|turned|turns|remained|remains|left|leaves)\b", text, flags=re.IGNORECASE)
    )
    same_entity_event_chain_count = 0
    entity_change_sentences: dict[str, set[int]] = {}
    for sentence_idx, sentence in enumerate(sentences):
        if not _contains_any_word(sentence, CHANGE_VERBS):
            continue
        for entity, _start, _end in _entity_spans(sentence):
            entity_change_sentences.setdefault(entity, set()).add(int(sentence_idx))
    same_entity_event_chain_count = sum(1 for seen in entity_change_sentences.values() if len(seen) >= 2)
    return {
        "change_verb_count": int(change_verb_count),
        "change_verb_density": float(change_verb_count / max(1, sentence_count)),
        "temporal_marker_count": int(temporal_marker_count),
        "temporal_marker_density": float(temporal_marker_count / max(1, sentence_count)),
        "before_after_marker_count": int(before_after_marker_count),
        "result_state_pattern_count": int(result_state_pattern_count),
        "same_entity_event_chain_count": int(same_entity_event_chain_count),
    }


def _state_update_features_spacy(doc: Any, text: str, sentence_count: int) -> dict[str, float | int]:
    words = [token.text.lower() for token in doc if getattr(token, "is_alpha", False)]
    change_verb_count = sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and (token.lemma_.lower() in CHANGE_VERB_LEMMAS or token.text.lower() in CHANGE_VERBS)
    )
    temporal_marker_count = _count_lexicon(words, TEMPORAL_MARKERS)
    before_after_marker_count = sum(1 for word in words if word in {"before", "after"})
    result_state_pattern_count = sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and token.lemma_.lower() in {"become", "turn", "remain", "leave"}
    )

    entity_change_sentences: dict[str, set[int]] = {}
    try:
        doc_sentences = list(doc.sents)
    except ValueError:
        doc_sentences = []
    for sentence_idx, sentence in enumerate(doc_sentences):
        has_change = any(
            token.pos_ in {"VERB", "AUX"}
            and (token.lemma_.lower() in CHANGE_VERB_LEMMAS or token.text.lower() in CHANGE_VERBS)
            for token in sentence
        )
        if not has_change:
            continue
        for entity, _start, _end in _entity_spans(sentence.text):
            entity_change_sentences.setdefault(entity, set()).add(int(sentence_idx))
    same_entity_event_chain_count = sum(1 for seen in entity_change_sentences.values() if len(seen) >= 2)
    return {
        "change_verb_count": int(change_verb_count),
        "change_verb_density": float(change_verb_count / max(1, sentence_count)),
        "temporal_marker_count": int(temporal_marker_count),
        "temporal_marker_density": float(temporal_marker_count / max(1, sentence_count)),
        "before_after_marker_count": int(before_after_marker_count),
        "result_state_pattern_count": int(result_state_pattern_count),
        "same_entity_event_chain_count": int(same_entity_event_chain_count),
    }


def _internal_state_noise_features(text: str) -> dict[str, float | int]:
    instructional_second_person_count = len(SECOND_PERSON_INSTRUCTION_RE.findall(text))
    tutorial_marker_count = len(TUTORIAL_MARKER_RE.findall(text))
    patent_intent_marker_count = len(PATENT_INTENT_RE.findall(text))
    internal_state_instructional_noise_score = min(
        1.0,
        max(
            instructional_second_person_count / 3.0,
            min(1.0, tutorial_marker_count / 4.0) if instructional_second_person_count > 0 else 0.0,
        ),
    )
    patent_intent_noise_score = min(1.0, patent_intent_marker_count / 3.0)
    return {
        "instructional_second_person_count": int(instructional_second_person_count),
        "tutorial_marker_count": int(tutorial_marker_count),
        "patent_intent_marker_count": int(patent_intent_marker_count),
        "internal_state_instructional_noise_score": float(internal_state_instructional_noise_score),
        "patent_intent_noise_score": float(patent_intent_noise_score),
        "internal_state_false_positive_noise_score": float(
            max(internal_state_instructional_noise_score, patent_intent_noise_score)
        ),
    }


def _internal_state_features(text: str, sentences: Sequence[str], sentence_count: int) -> dict[str, float | int]:
    strong_internal_state_count = _count_internal_state_lexicon(text, STRONG_INTERNAL_STATE_WORDS)
    weak_internal_state_count = _count_internal_state_lexicon(text, WEAK_INTERNAL_STATE_WORDS)
    mental_verb_count = strong_internal_state_count + weak_internal_state_count
    preference_goal_intent_count = _count_internal_state_lexicon(text, PREFERENCE_GOAL_INTENT_WORDS)
    agent_state_edge_count = 0
    strong_agent_state_edge_count = 0
    weak_agent_state_edge_count = 0
    for sentence in sentences:
        if not _entity_spans(sentence):
            continue
        has_strong_state = _contains_internal_state_word(sentence, STRONG_INTERNAL_STATE_WORDS)
        has_weak_state = _contains_internal_state_word(sentence, WEAK_INTERNAL_STATE_WORDS)
        if has_strong_state or has_weak_state:
            agent_state_edge_count += 1
        if has_strong_state:
            strong_agent_state_edge_count += 1
        if has_weak_state and not has_strong_state:
            weak_agent_state_edge_count += 1
    state_complement_count = len(STATE_COMPLEMENT_RE.findall(text))
    return {
        "mental_verb_count": int(mental_verb_count),
        "mental_state_density": float(mental_verb_count / max(1, sentence_count)),
        "strong_internal_state_count": int(strong_internal_state_count),
        "strong_internal_state_density": float(strong_internal_state_count / max(1, sentence_count)),
        "weak_internal_state_count": int(weak_internal_state_count),
        "weak_internal_state_density": float(weak_internal_state_count / max(1, sentence_count)),
        "preference_goal_intent_count": int(preference_goal_intent_count),
        "preference_goal_intent_density": float(preference_goal_intent_count / max(1, sentence_count)),
        "agent_state_edge_count": int(agent_state_edge_count),
        "agent_state_edge_density": float(agent_state_edge_count / max(1, sentence_count)),
        "strong_agent_state_edge_count": int(strong_agent_state_edge_count),
        "strong_agent_state_edge_density": float(strong_agent_state_edge_count / max(1, sentence_count)),
        "weak_agent_state_edge_count": int(weak_agent_state_edge_count),
        "weak_agent_state_edge_density": float(weak_agent_state_edge_count / max(1, sentence_count)),
        "state_complement_count": int(state_complement_count),
        **_internal_state_noise_features(text),
    }


def _internal_state_features_spacy(doc: Any, text: str, sentence_count: int) -> dict[str, float | int]:
    strong_internal_state_count = sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and (
            token.lemma_.lower() in STRONG_INTERNAL_STATE_LEMMAS
            or token.text.lower() in STRONG_INTERNAL_STATE_WORDS
        )
    )
    weak_internal_state_count = sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and (
            token.lemma_.lower() in WEAK_INTERNAL_STATE_LEMMAS
            or token.text.lower() in WEAK_INTERNAL_STATE_WORDS
        )
    )
    mental_verb_count = strong_internal_state_count + weak_internal_state_count
    preference_goal_intent_count = sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and (
            token.lemma_.lower() in PREFERENCE_GOAL_INTENT_LEMMAS
            or token.text.lower() in PREFERENCE_GOAL_INTENT_WORDS
        )
    )
    agent_state_edge_count = 0
    strong_agent_state_edge_count = 0
    weak_agent_state_edge_count = 0
    try:
        doc_sentences = list(doc.sents)
    except ValueError:
        doc_sentences = []
    for sentence in doc_sentences:
        has_strong_state = any(
            token.pos_ in {"VERB", "AUX"}
            and (
                token.lemma_.lower() in STRONG_INTERNAL_STATE_LEMMAS
                or token.text.lower() in STRONG_INTERNAL_STATE_WORDS
            )
            for token in sentence
        )
        has_weak_state = any(
            token.pos_ in {"VERB", "AUX"}
            and (
                token.lemma_.lower() in WEAK_INTERNAL_STATE_LEMMAS
                or token.text.lower() in WEAK_INTERNAL_STATE_WORDS
            )
            for token in sentence
        )
        if not _entity_spans(sentence.text):
            continue
        if has_strong_state or has_weak_state:
            agent_state_edge_count += 1
        if has_strong_state:
            strong_agent_state_edge_count += 1
        if has_weak_state and not has_strong_state:
            weak_agent_state_edge_count += 1
    state_complement_count = len(STATE_COMPLEMENT_RE.findall(text))
    return {
        "mental_verb_count": int(mental_verb_count),
        "mental_state_density": float(mental_verb_count / max(1, sentence_count)),
        "strong_internal_state_count": int(strong_internal_state_count),
        "strong_internal_state_density": float(strong_internal_state_count / max(1, sentence_count)),
        "weak_internal_state_count": int(weak_internal_state_count),
        "weak_internal_state_density": float(weak_internal_state_count / max(1, sentence_count)),
        "preference_goal_intent_count": int(preference_goal_intent_count),
        "preference_goal_intent_density": float(preference_goal_intent_count / max(1, sentence_count)),
        "agent_state_edge_count": int(agent_state_edge_count),
        "agent_state_edge_density": float(agent_state_edge_count / max(1, sentence_count)),
        "strong_agent_state_edge_count": int(strong_agent_state_edge_count),
        "strong_agent_state_edge_density": float(strong_agent_state_edge_count / max(1, sentence_count)),
        "weak_agent_state_edge_count": int(weak_agent_state_edge_count),
        "weak_agent_state_edge_density": float(weak_agent_state_edge_count / max(1, sentence_count)),
        "state_complement_count": int(state_complement_count),
        **_internal_state_noise_features(text),
    }


def compute_snippet_features(
    text: str,
    *,
    parser_backend: str = "regex",
    spacy_model: str = "en_core_web_sm",
    nlp: Any | None = None,
) -> dict[str, float | int]:
    doc = None
    if parser_backend == "spacy" and nlp is not None:
        doc = nlp(text)
        base = compute_spacy_doc_features(doc)
    else:
        base = compute_text_features(
            text,
            parser_backend=parser_backend,
            spacy_model=spacy_model,
            nlp=nlp,
        )
    sentences = _sentences_for_features(text)
    sentence_count = max(1, int(base.get("sentence_count", len(sentences) or 1)))
    state_update = (
        _state_update_features_spacy(doc, text, sentence_count)
        if doc is not None
        else _state_update_features(text, sentences, sentence_count)
    )
    internal_state = (
        _internal_state_features_spacy(doc, text, sentence_count)
        if doc is not None
        else _internal_state_features(text, sentences, sentence_count)
    )
    return {
        **base,
        **_attribute_features(text, sentences, sentence_count),
        **_relation_role_features(sentences, sentence_count),
        **_layout_noise_features(text, sentence_count),
        **state_update,
        **internal_state,
    }
