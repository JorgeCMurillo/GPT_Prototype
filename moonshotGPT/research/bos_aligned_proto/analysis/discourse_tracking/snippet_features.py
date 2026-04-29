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

RELATION_CUE_WORDS = {
    "asked",
    "asks",
    "called",
    "calls",
    "chased",
    "chases",
    "comforted",
    "comforts",
    "defeated",
    "defeats",
    "followed",
    "follows",
    "gave",
    "gives",
    "helped",
    "helps",
    "hindered",
    "hinders",
    "insulted",
    "insults",
    "led",
    "leads",
    "met",
    "meets",
    "obeyed",
    "obeys",
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
    "saw",
    "sees",
    "showed",
    "shows",
    "supervised",
    "supervises",
    "taught",
    "teaches",
    "thanked",
    "thanks",
    "told",
    "tells",
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
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*]|\d{1,3}[.)]|[A-Za-z][.)])\s+")
DENSE_LIST_SEPARATOR_RE = re.compile(r"[\u2012\u2013\u2014;]\s*")
LIST_NOISE_GATE_MAX = 0.85
HEAVY_LIST_NOISE_GATE_MAX = 0.95
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
    layout_noise_score = min(
        1.0,
        max(
            bullet_line_fraction,
            min(1.0, list_marker_count / 4.0),
            short_line_fraction,
            min(1.0, newline_density / 4.0),
            min(1.0, dense_separator_density / 10.0),
        ),
    )
    return {
        "line_count": int(line_count),
        "list_marker_count": int(list_marker_count),
        "short_structured_line_count": int(short_structured_line_count),
        "bullet_line_fraction": float(bullet_line_fraction),
        "newline_density": float(newline_density),
        "dense_separator_density": float(dense_separator_density),
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


def _internal_state_features(text: str, sentences: Sequence[str], sentence_count: int) -> dict[str, float | int]:
    words, _ = _word_counts(text)
    mental_verb_count = _count_lexicon(words, MENTAL_STATE_WORDS)
    preference_goal_intent_count = _count_lexicon(words, PREFERENCE_GOAL_INTENT_WORDS)
    agent_state_edge_count = 0
    for sentence in sentences:
        if _entity_spans(sentence) and _contains_any_word(sentence, MENTAL_STATE_WORDS):
            agent_state_edge_count += 1
    state_complement_count = len(STATE_COMPLEMENT_RE.findall(text))
    return {
        "mental_verb_count": int(mental_verb_count),
        "mental_state_density": float(mental_verb_count / max(1, sentence_count)),
        "preference_goal_intent_count": int(preference_goal_intent_count),
        "preference_goal_intent_density": float(preference_goal_intent_count / max(1, sentence_count)),
        "agent_state_edge_count": int(agent_state_edge_count),
        "agent_state_edge_density": float(agent_state_edge_count / max(1, sentence_count)),
        "state_complement_count": int(state_complement_count),
    }


def _internal_state_features_spacy(doc: Any, text: str, sentence_count: int) -> dict[str, float | int]:
    mental_verb_count = sum(
        1
        for token in doc
        if token.pos_ in {"VERB", "AUX"}
        and (token.lemma_.lower() in MENTAL_STATE_LEMMAS or token.text.lower() in MENTAL_STATE_WORDS)
    )
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
    try:
        doc_sentences = list(doc.sents)
    except ValueError:
        doc_sentences = []
    for sentence in doc_sentences:
        has_mental_state = any(
            token.pos_ in {"VERB", "AUX"}
            and (token.lemma_.lower() in MENTAL_STATE_LEMMAS or token.text.lower() in MENTAL_STATE_WORDS)
            for token in sentence
        )
        if has_mental_state and _entity_spans(sentence.text):
            agent_state_edge_count += 1
    state_complement_count = len(STATE_COMPLEMENT_RE.findall(text))
    return {
        "mental_verb_count": int(mental_verb_count),
        "mental_state_density": float(mental_verb_count / max(1, sentence_count)),
        "preference_goal_intent_count": int(preference_goal_intent_count),
        "preference_goal_intent_density": float(preference_goal_intent_count / max(1, sentence_count)),
        "agent_state_edge_count": int(agent_state_edge_count),
        "agent_state_edge_density": float(agent_state_edge_count / max(1, sentence_count)),
        "state_complement_count": int(state_complement_count),
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
