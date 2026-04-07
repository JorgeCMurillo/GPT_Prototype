"""Feature extraction and fixed-rule pool building for discourse-tracking text.

The goal here is not perfect coreference resolution. It is to build an
interpretable first-pass screen for candidate training spans that plausibly
pressure a language model to maintain identity and role assignments over
multiple sentences.

The implementation prefers spaCy when available because sentence segmentation,
NER, and dependency edges make the features more faithful. When spaCy is not
installed, a lightweight regex fallback still lets the workflow run in this
repository's default environment.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


TRACKED_ENTITY_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "FAC",
    "EVENT",
    "WORK_OF_ART",
    "PRODUCT",
}
TITLECASE_STOPWORDS = {
    "A",
    "An",
    "And",
    "As",
    "At",
    "But",
    "By",
    "For",
    "From",
    "He",
    "Her",
    "His",
    "I",
    "If",
    "In",
    "It",
    "Its",
    "My",
    "No",
    "Not",
    "Of",
    "On",
    "Or",
    "Our",
    "She",
    "That",
    "The",
    "Their",
    "Then",
    "There",
    "They",
    "This",
    "Those",
    "To",
    "We",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "With",
    "You",
}
PRONOUNS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
    "itself",
    "himself",
    "herself",
    "themselves",
    "we",
    "us",
    "our",
    "ours",
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}
OBJECT_DEPS = {"dobj", "obj", "iobj", "pobj", "dative", "oprd", "obl"}
ATTRIBUTE_DEPS = {"attr", "acomp", "xcomp"}
VERB_CUE_PATTERN = re.compile(
    r"\b(?:"
    r"is|are|was|were|be|been|being|has|have|had|said|says|tell|tells|told|"
    r"ask|asks|asked|give|gives|gave|given|met|meet|meets|help|helps|helped|"
    r"call|calls|called|saw|see|sees|seen|found|find|finds|joined|joins|join|"
    r"supports|supported|support|led|lead|leads|follow|follows|followed|"
    r"defeated|defeats|defeat|married|marries|marry|visited|visits|visit|"
    r"left|leaves|leave|sent|sends|send|took|takes|take"
    r")\b",
    re.IGNORECASE,
)
ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")
WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]*")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    used: str
    detail: str

    def to_json(self) -> dict[str, str]:
        return {
            "requested": self.requested,
            "used": self.used,
            "detail": self.detail,
        }


def load_spacy_pipeline(model_name: str = "en_core_web_sm") -> tuple[Any | None, BackendInfo]:
    try:
        import spacy
    except ImportError:
        return None, BackendInfo(requested="spacy", used="regex", detail="spaCy is not installed")

    try:
        nlp = spacy.load(model_name, disable=["textcat"])
    except OSError as exc:
        return None, BackendInfo(
            requested="spacy",
            used="regex",
            detail=f"spaCy model {model_name!r} is unavailable: {exc}",
        )
    return nlp, BackendInfo(requested="spacy", used="spacy", detail=f"loaded {model_name}")


def _normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text)).strip()


def canonicalize_entity_text(text: str) -> str:
    cleaned = _normalize_space(text)
    cleaned = cleaned.strip(" \t\r\n'\"“”‘’()[]{}.,;:!?")
    return cleaned.lower()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _sentence_texts_from_regex(text: str) -> list[str]:
    normalized = str(text).replace("\r\n", "\n")
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", normalized)
    sentences = [piece.strip() for piece in pieces if piece and piece.strip()]
    return sentences or [_normalize_space(text)]


def _repetition_features(text: str, *, sentences: Sequence[str]) -> dict[str, float]:
    word_tokens = [token.lower() for token in WORD_PATTERN.findall(text)]
    unique_token_ratio = _safe_ratio(len(set(word_tokens)), len(word_tokens))

    nonempty_sentences = [_normalize_space(sentence) for sentence in sentences if sentence.strip()]
    sentence_counts = Counter(nonempty_sentences)
    duplicate_sentence_fraction = 0.0
    if nonempty_sentences:
        duplicate_sentence_fraction = _safe_ratio(
            sum(count - 1 for count in sentence_counts.values() if count > 1),
            len(nonempty_sentences),
        )

    ngrams = list(zip(word_tokens, word_tokens[1:], word_tokens[2:]))
    ngram_counts = Counter(ngrams)
    repeated_3gram_ratio = 0.0
    if ngrams:
        repeated_3gram_ratio = _safe_ratio(
            sum(count - 1 for count in ngram_counts.values() if count > 1),
            len(ngrams),
        )

    return {
        "unique_token_ratio": float(unique_token_ratio),
        "duplicate_sentence_fraction": float(duplicate_sentence_fraction),
        "repeated_3gram_ratio": float(repeated_3gram_ratio),
    }


def _feature_summary_from_mentions(
    *,
    text: str,
    sentence_entity_lists: Sequence[Sequence[str]],
    relation_count: int,
    pronoun_count: int,
) -> dict[str, float | int]:
    sentences = [list(entities) for entities in sentence_entity_lists]
    sentence_count = max(1, len(sentences))
    word_tokens = WORD_PATTERN.findall(text)
    entity_counts = Counter(entity for entities in sentences for entity in entities)
    entity_sentence_sets: dict[str, set[int]] = defaultdict(set)
    first_seen: dict[str, int] = {}
    for sentence_idx, entities in enumerate(sentences):
        for entity in entities:
            entity_sentence_sets[entity].add(sentence_idx)
            first_seen.setdefault(entity, sentence_idx)

    unique_entity_count = len(entity_sentence_sets)
    entity_mention_count = int(sum(entity_counts.values()))
    persistence_values = [
        _safe_ratio(len(sentence_ids), sentence_count) for sentence_ids in entity_sentence_sets.values()
    ]
    entity_persistence = float(max(persistence_values) if persistence_values else 0.0)
    mean_entity_persistence = float(np.mean(persistence_values)) if persistence_values else 0.0
    entity_recurrence = _safe_ratio(
        sum(1 for sentence_ids in entity_sentence_sets.values() if len(sentence_ids) >= 2),
        unique_entity_count,
    )
    entity_churn = _safe_ratio(sum(1 for first_idx in first_seen.values() if first_idx > 0), max(1, sentence_count - 1))
    sentences_with_entities = sum(1 for entities in sentences if entities)
    multi_entity_sentences = sum(1 for entities in sentences if len(set(entities)) >= 2)
    avg_entities_per_sentence = _safe_ratio(entity_mention_count, sentence_count)

    repetition = _repetition_features(text, sentences=[_normalize_space(sentence) for sentence in _sentence_texts_from_regex(text)])
    return {
        "sentence_count": int(sentence_count),
        "token_count_text": int(len(word_tokens)),
        "char_count_text": int(len(text)),
        "unique_entity_count": int(unique_entity_count),
        "entity_mention_count": int(entity_mention_count),
        "entity_persistence": float(entity_persistence),
        "mean_entity_persistence": float(mean_entity_persistence),
        "entity_recurrence": float(entity_recurrence),
        "entity_churn": float(entity_churn),
        "relation_count": int(relation_count),
        "relation_density": float(_safe_ratio(relation_count, sentence_count)),
        "pronoun_count": int(pronoun_count),
        "pronoun_density": float(_safe_ratio(pronoun_count, len(word_tokens))),
        "entity_sentence_coverage": float(_safe_ratio(sentences_with_entities, sentence_count)),
        "multi_entity_sentence_fraction": float(_safe_ratio(multi_entity_sentences, sentence_count)),
        "avg_entities_per_sentence": float(avg_entities_per_sentence),
        **repetition,
    }


def _regex_entities_for_sentence(sentence: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.finditer(sentence):
        raw = match.group(0).strip()
        if raw in TITLECASE_STOPWORDS:
            continue
        canonical = canonicalize_entity_text(raw)
        if canonical and canonical not in TITLECASE_STOPWORDS:
            entities.append(canonical)
    return entities


def _compute_features_regex(text: str) -> dict[str, float | int]:
    sentences = _sentence_texts_from_regex(text)
    sentence_entity_lists = [_regex_entities_for_sentence(sentence) for sentence in sentences]
    pronoun_count = sum(1 for token in WORD_PATTERN.findall(text) if token.lower() in PRONOUNS)
    relation_count = 0
    for sentence, entities in zip(sentences, sentence_entity_lists):
        if len(set(entities)) >= 2 and VERB_CUE_PATTERN.search(sentence):
            relation_count += max(1, len(set(entities)) - 1)
    return _feature_summary_from_mentions(
        text=text,
        sentence_entity_lists=sentence_entity_lists,
        relation_count=relation_count,
        pronoun_count=pronoun_count,
    )


def _canonical_entity_for_token(token, token_to_entity: dict[int, str]) -> str | None:
    if token.i in token_to_entity:
        return token_to_entity[token.i]
    for child in token.subtree:
        entity = token_to_entity.get(child.i)
        if entity:
            return entity
    return None


def _relation_count_from_doc(doc, token_to_entity: dict[int, str]) -> int:
    relation_count = 0
    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        subjects: set[str] = set()
        objects: set[str] = set()
        for child in token.children:
            entity = _canonical_entity_for_token(child, token_to_entity)
            if child.dep_ in SUBJECT_DEPS and entity:
                subjects.add(entity)
            elif child.dep_ in OBJECT_DEPS | ATTRIBUTE_DEPS and entity:
                objects.add(entity)
            elif child.dep_ in {"prep", "agent"}:
                for grand in child.children:
                    grand_entity = _canonical_entity_for_token(grand, token_to_entity)
                    if grand.dep_ in OBJECT_DEPS | ATTRIBUTE_DEPS and grand_entity:
                        objects.add(grand_entity)
        if subjects and objects:
            relation_count += max(1, len(subjects) * len(objects))
        elif len(subjects | objects) >= 2:
            relation_count += 1
    return relation_count


def compute_spacy_doc_features(doc) -> dict[str, float | int]:
    text = str(doc.text)
    sentence_entity_lists: list[list[str]] = []
    token_to_entity: dict[int, str] = {}
    span_keys: set[tuple[int, int]] = set()

    mention_spans = []
    for ent in doc.ents:
        if ent.label_ not in TRACKED_ENTITY_LABELS:
            continue
        span_keys.add((ent.start, ent.end))
        mention_spans.append(ent)

    if hasattr(doc, "noun_chunks"):
        for chunk in doc.noun_chunks:
            if (chunk.start, chunk.end) in span_keys:
                continue
            if not any(token.pos_ == "PROPN" for token in chunk):
                continue
            mention_spans.append(chunk)

    for span in mention_spans:
        canonical = canonicalize_entity_text(span.text)
        if not canonical:
            continue
        for token in span:
            token_to_entity[token.i] = canonical

    pronoun_count = sum(1 for token in doc if token.lower_ in PRONOUNS)
    for sentence in doc.sents:
        entities: list[str] = []
        for token in sentence:
            entity = token_to_entity.get(token.i)
            if entity:
                entities.append(entity)
        sentence_entity_lists.append(entities)

    relation_count = _relation_count_from_doc(doc, token_to_entity)
    return _feature_summary_from_mentions(
        text=text,
        sentence_entity_lists=sentence_entity_lists,
        relation_count=relation_count,
        pronoun_count=pronoun_count,
    )


def _compute_features_spacy(text: str, *, nlp) -> dict[str, float | int]:
    doc = nlp(text)
    return compute_spacy_doc_features(doc)


def resolve_backend(
    *,
    parser_backend: str,
    spacy_model: str,
) -> tuple[str, Any | None, BackendInfo]:
    requested = str(parser_backend).strip().lower()
    if requested not in {"auto", "spacy", "regex"}:
        raise ValueError(f"Unsupported parser_backend={parser_backend!r}; expected auto, spacy, or regex")
    if requested == "regex":
        return "regex", None, BackendInfo(requested=requested, used="regex", detail="regex backend requested")

    nlp, info = load_spacy_pipeline(spacy_model)
    if nlp is not None:
        return "spacy", nlp, BackendInfo(requested=requested, used="spacy", detail=info.detail)
    if requested == "spacy":
        raise RuntimeError(info.detail)
    return "regex", None, BackendInfo(requested=requested, used="regex", detail=info.detail)


def compute_text_features(
    text: str,
    *,
    parser_backend: str = "auto",
    spacy_model: str = "en_core_web_sm",
    nlp=None,
) -> dict[str, float | int]:
    normalized = _normalize_space(str(text))
    if not normalized:
        return {
            "sentence_count": 0,
            "token_count_text": 0,
            "char_count_text": 0,
            "unique_entity_count": 0,
            "entity_mention_count": 0,
            "entity_persistence": 0.0,
            "mean_entity_persistence": 0.0,
            "entity_recurrence": 0.0,
            "entity_churn": 0.0,
            "relation_count": 0,
            "relation_density": 0.0,
            "pronoun_count": 0,
            "pronoun_density": 0.0,
            "entity_sentence_coverage": 0.0,
            "multi_entity_sentence_fraction": 0.0,
            "avg_entities_per_sentence": 0.0,
            "unique_token_ratio": 0.0,
            "duplicate_sentence_fraction": 0.0,
            "repeated_3gram_ratio": 0.0,
        }

    if parser_backend == "spacy":
        if nlp is None:
            nlp, _ = load_spacy_pipeline(spacy_model)
        if nlp is None:
            raise RuntimeError("spaCy backend requested but no spaCy pipeline is available")
        return _compute_features_spacy(normalized, nlp=nlp)
    return _compute_features_regex(normalized)


def _quantile(values: Sequence[float], q: float, default: float = 0.0) -> float:
    arr = np.asarray([float(value) for value in values if pd.notna(value)], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def _zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std <= 0.0:
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=series.index)
    return (values - float(values.mean())) / std


def build_rule_based_pools(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    min_sentences: int = 3,
    min_entities: int = 2,
    min_text_tokens: int = 96,
    random_pool_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = frame.copy()
    if working.empty:
        summary = {
            "counts": {
                "total": 0,
                "eligible": 0,
                "positive": 0,
                "negative_low_binding": 0,
                "negative_repetition": 0,
                "random_control": 0,
            },
            "thresholds": {},
        }
        return working, summary

    working["eligibility_reason"] = ""
    eligible_mask = (
        (working["sentence_count"].astype(int) >= int(min_sentences))
        & (working["unique_entity_count"].astype(int) >= int(min_entities))
        & (working["token_count_text"].astype(int) >= int(min_text_tokens))
    )
    working["is_eligible"] = eligible_mask

    eligible = working.loc[eligible_mask].copy()
    thresholds = {
        "min_sentences": int(min_sentences),
        "min_entities": int(min_entities),
        "min_text_tokens": int(min_text_tokens),
        "entity_persistence_q25": _quantile(eligible.get("entity_persistence", []), 0.25),
        "entity_persistence_q75": _quantile(eligible.get("entity_persistence", []), 0.75),
        "entity_recurrence_q60": _quantile(eligible.get("entity_recurrence", []), 0.60),
        "relation_density_q40": _quantile(eligible.get("relation_density", []), 0.40),
        "relation_density_q75": _quantile(eligible.get("relation_density", []), 0.75),
        "repeated_3gram_q50": _quantile(eligible.get("repeated_3gram_ratio", []), 0.50),
        "repeated_3gram_q90": _quantile(eligible.get("repeated_3gram_ratio", []), 0.90),
        "duplicate_sentence_q75": _quantile(eligible.get("duplicate_sentence_fraction", []), 0.75),
        "unique_entities_q50": _quantile(eligible.get("unique_entity_count", []), 0.50, default=float(min_entities)),
    }
    positive_unique_entities_min = max(int(min_entities), min(4, int(math.ceil(thresholds["unique_entities_q50"]))))

    working["priority_score"] = (
        _zscore(working["entity_persistence"])
        + _zscore(working["entity_recurrence"])
        + _zscore(working["relation_density"])
        + 0.25 * _zscore(working["sentence_count"])
        + 0.25 * _zscore(working["unique_entity_count"])
        - 0.50 * _zscore(working["entity_churn"])
        - 0.75 * _zscore(working["repeated_3gram_ratio"])
        - 0.50 * _zscore(working["duplicate_sentence_fraction"])
    ).astype(float)

    positive_mask = (
        eligible_mask
        & (working["entity_persistence"] >= thresholds["entity_persistence_q75"])
        & (working["entity_recurrence"] >= thresholds["entity_recurrence_q60"])
        & (working["relation_density"] >= thresholds["relation_density_q75"])
        & (working["repeated_3gram_ratio"] <= thresholds["repeated_3gram_q50"])
        & (working["duplicate_sentence_fraction"] <= thresholds["duplicate_sentence_q75"])
        & (working["unique_entity_count"] >= positive_unique_entities_min)
    )

    fallback_positive_count = 0
    if int(positive_mask.sum()) == 0 and not eligible.empty:
        fallback_pool = working.loc[
            eligible_mask
            & (working["relation_density"] >= thresholds["relation_density_q40"])
            & (working["repeated_3gram_ratio"] <= thresholds["repeated_3gram_q90"])
            & (working["unique_entity_count"] >= int(min_entities))
        ].copy()
        if not fallback_pool.empty:
            fallback_positive_count = max(1, min(64, int(math.ceil(len(fallback_pool) * 0.10))))
            fallback_ids = set(
                int(value)
                for value in fallback_pool.sort_values(
                    ["priority_score", "candidate_id"],
                    ascending=[False, True],
                )
                .head(fallback_positive_count)["candidate_id"]
                .tolist()
            )
            positive_mask = working["candidate_id"].astype(int).isin(fallback_ids)

    negative_low_binding_mask = (
        eligible_mask
        & ~positive_mask
        & (working["entity_persistence"] <= thresholds["entity_persistence_q25"])
        & (working["relation_density"] <= thresholds["relation_density_q40"])
        & (working["repeated_3gram_ratio"] <= thresholds["repeated_3gram_q90"])
    )

    negative_repetition_mask = (
        eligible_mask
        & ~positive_mask
        & (working["repeated_3gram_ratio"] >= thresholds["repeated_3gram_q90"])
        & (working["relation_density"] <= thresholds["relation_density_q75"])
    )

    working["is_positive_pool"] = positive_mask
    working["is_negative_low_binding_pool"] = negative_low_binding_mask
    working["is_negative_repetition_pool"] = negative_repetition_mask
    working["is_random_control_pool"] = False
    working["pool_label"] = "other"
    working.loc[positive_mask, "pool_label"] = "positive"
    working.loc[negative_low_binding_mask, "pool_label"] = "negative_low_binding"
    working.loc[negative_repetition_mask, "pool_label"] = "negative_repetition"

    candidate_random_mask = eligible_mask & ~(positive_mask | negative_low_binding_mask | negative_repetition_mask)
    random_candidates = working.loc[candidate_random_mask].copy()
    if random_pool_size is None:
        random_pool_size = int(max(0, positive_mask.sum()))
    random_pool_size = min(int(random_pool_size), len(random_candidates))
    if random_pool_size > 0:
        sampled = random_candidates.sample(n=random_pool_size, random_state=int(seed), replace=False)
        sampled_ids = set(int(value) for value in sampled["candidate_id"].tolist())
        working.loc[working["candidate_id"].isin(sampled_ids), "is_random_control_pool"] = True
        working.loc[working["candidate_id"].isin(sampled_ids), "pool_label"] = "random_control"

    summary = {
        "counts": {
            "total": int(len(working)),
            "eligible": int(eligible_mask.sum()),
            "positive": int(positive_mask.sum()),
            "negative_low_binding": int(negative_low_binding_mask.sum()),
            "negative_repetition": int(negative_repetition_mask.sum()),
            "random_control": int(working["is_random_control_pool"].sum()),
        },
        "thresholds": {
            **{key: (float(value) if isinstance(value, (int, float, np.floating)) else value) for key, value in thresholds.items()},
            "positive_unique_entities_min": int(positive_unique_entities_min),
        },
        "priority_score_formula": (
            "z(entity_persistence) + z(entity_recurrence) + z(relation_density) + "
            "0.25*z(sentence_count) + 0.25*z(unique_entity_count) - "
            "0.50*z(entity_churn) - 0.75*z(repeated_3gram_ratio) - "
            "0.50*z(duplicate_sentence_fraction)"
        ),
        "positive_fallback_rule": {
            "used": bool(fallback_positive_count > 0),
            "top_fraction_of_filtered_eligible": 0.10,
            "max_count": 64,
            "count": int(fallback_positive_count),
            "filters": {
                "relation_density_gte": float(thresholds["relation_density_q40"]),
                "repeated_3gram_ratio_lte": float(thresholds["repeated_3gram_q90"]),
                "unique_entity_count_gte": int(min_entities),
            },
        },
    }
    return working, summary


def build_text_embeddings(
    texts: Sequence[str],
    *,
    backend: str = "tfidf_svd",
    embedding_model: str = "all-MiniLM-L6-v2",
    svd_dim: int = 128,
    max_features: int = 20_000,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = str(backend).strip().lower()
    if resolved not in {"auto", "tfidf_svd", "sentence_transformers"}:
        raise ValueError(
            f"Unsupported embedding backend {backend!r}; expected auto, tfidf_svd, or sentence_transformers"
        )

    if resolved in {"auto", "tfidf_svd"}:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import Normalizer

        min_df = 2 if len(texts) >= 20 else 1
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.95,
            max_features=int(max_features),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        if matrix.shape[1] <= 1:
            dense = matrix.toarray().astype(np.float32, copy=False)
            return dense, {
                "backend": "tfidf_svd",
                "detail": "tf-idf only; vocabulary too small for SVD",
                "vocab_size": int(matrix.shape[1]),
            }
        n_components = max(2, min(int(svd_dim), int(matrix.shape[1]) - 1, int(matrix.shape[0]) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=int(seed))
        dense = svd.fit_transform(matrix)
        dense = Normalizer(copy=False).fit_transform(dense).astype(np.float32, copy=False)
        return dense, {
            "backend": "tfidf_svd",
            "detail": "TF-IDF bigrams + TruncatedSVD + L2 normalization",
            "vocab_size": int(matrix.shape[1]),
            "embedding_dim": int(dense.shape[1]),
            "explained_variance_sum": float(np.asarray(svd.explained_variance_ratio_, dtype=np.float64).sum()),
        }

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it or use --embedding_backend tfidf_svd."
        ) from exc

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        list(texts),
        show_progress_bar=False,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32), {
        "backend": "sentence_transformers",
        "detail": f"SentenceTransformer({embedding_model})",
        "embedding_dim": int(embeddings.shape[1]),
    }


def cluster_promising_pool(
    frame: pd.DataFrame,
    *,
    text_lookup: dict[int, str],
    embedding_backend: str = "tfidf_svd",
    embedding_model: str = "all-MiniLM-L6-v2",
    num_clusters: int = 8,
    min_cluster_size: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    positive = frame.loc[frame["is_positive_pool"]].copy()
    if positive.empty:
        return (
            positive.assign(cluster_id=-1, is_selected_cluster=False),
            pd.DataFrame(),
            {
                "ran": False,
                "reason": "positive pool is empty",
            },
        )
    if len(positive) < max(4, int(min_cluster_size)):
        return (
            positive.assign(cluster_id=-1, is_selected_cluster=False),
            pd.DataFrame(),
            {
                "ran": False,
                "reason": "positive pool is too small for clustering",
                "positive_size": int(len(positive)),
            },
        )

    texts = [text_lookup[int(candidate_id)] for candidate_id in positive["candidate_id"].tolist()]
    embeddings, embedding_info = build_text_embeddings(
        texts,
        backend=embedding_backend,
        embedding_model=embedding_model,
        seed=seed,
    )

    from sklearn.cluster import KMeans

    resolved_clusters = max(2, min(int(num_clusters), len(positive) // 2))
    kmeans = KMeans(n_clusters=resolved_clusters, random_state=int(seed), n_init=10)
    positive["cluster_id"] = kmeans.fit_predict(embeddings)

    summary = (
        positive.groupby("cluster_id", dropna=False)
        .agg(
            cluster_size=("candidate_id", "size"),
            mean_priority_score=("priority_score", "mean"),
            mean_entity_persistence=("entity_persistence", "mean"),
            mean_entity_recurrence=("entity_recurrence", "mean"),
            mean_relation_density=("relation_density", "mean"),
            mean_repeated_3gram_ratio=("repeated_3gram_ratio", "mean"),
            mean_duplicate_sentence_fraction=("duplicate_sentence_fraction", "mean"),
            mean_sentence_count=("sentence_count", "mean"),
            mean_unique_entity_count=("unique_entity_count", "mean"),
        )
        .reset_index()
        .sort_values(["mean_priority_score", "cluster_size"], ascending=[False, False])
        .reset_index(drop=True)
    )

    persistence_bar = _quantile(positive["entity_persistence"], 0.50)
    recurrence_bar = _quantile(positive["entity_recurrence"], 0.50)
    relation_bar = _quantile(positive["relation_density"], 0.50)
    repetition_bar = _quantile(positive["repeated_3gram_ratio"], 0.50)
    duplicate_bar = _quantile(positive["duplicate_sentence_fraction"], 0.50)

    summary["is_selected_cluster"] = (
        (summary["cluster_size"].astype(int) >= int(min_cluster_size))
        & (summary["mean_entity_persistence"] >= persistence_bar)
        & (summary["mean_entity_recurrence"] >= recurrence_bar)
        & (summary["mean_relation_density"] >= relation_bar)
        & (summary["mean_repeated_3gram_ratio"] <= repetition_bar)
        & (summary["mean_duplicate_sentence_fraction"] <= duplicate_bar)
    )
    selected_clusters = set(int(value) for value in summary.loc[summary["is_selected_cluster"], "cluster_id"].tolist())
    positive["is_selected_cluster"] = positive["cluster_id"].astype(int).isin(selected_clusters)

    return positive, summary, {
        "ran": True,
        "embedding": embedding_info,
        "num_clusters": int(resolved_clusters),
        "selection_rules": {
            "min_cluster_size": int(min_cluster_size),
            "mean_entity_persistence_gte_positive_median": float(persistence_bar),
            "mean_entity_recurrence_gte_positive_median": float(recurrence_bar),
            "mean_relation_density_gte_positive_median": float(relation_bar),
            "mean_repeated_3gram_ratio_lte_positive_median": float(repetition_bar),
            "mean_duplicate_sentence_fraction_lte_positive_median": float(duplicate_bar),
        },
        "selected_cluster_count": int(len(selected_clusters)),
    }


def preview_text(text: str, *, max_chars: int = 220) -> str:
    normalized = _normalize_space(text)
    if len(normalized) <= int(max_chars):
        return normalized
    return normalized[: max(0, int(max_chars) - 3)].rstrip() + "..."


def feature_columns() -> tuple[str, ...]:
    return (
        "sentence_count",
        "token_count_text",
        "char_count_text",
        "unique_entity_count",
        "entity_mention_count",
        "entity_persistence",
        "mean_entity_persistence",
        "entity_recurrence",
        "entity_churn",
        "relation_count",
        "relation_density",
        "pronoun_count",
        "pronoun_density",
        "entity_sentence_coverage",
        "multi_entity_sentence_fraction",
        "avg_entities_per_sentence",
        "unique_token_ratio",
        "duplicate_sentence_fraction",
        "repeated_3gram_ratio",
    )
