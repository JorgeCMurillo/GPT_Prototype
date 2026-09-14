#!/usr/bin/env python3
"""Generate controlled above/below definition and synonym minimal pairs."""

import csv
import json
import re
from collections import Counter
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEXT_FIELDS = ("Context1", "Context2", "Target1", "Target2")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_row(config, direction, phrase_pair, structure, stem=None, noun=None, order=None):
    extension = noun is not None
    word_first = direction in ("word_to_definition", "word_to_synonym")
    if extension:
        assert not word_first and stem is None and order in ("first_subject", "second_subject")
        subject_number, reference_number = (("first", "second") if order == "first_subject"
                                            else ("second", "first"))
        subject = f"The {subject_number} {noun['text']}"
        reference = f"the {reference_number} {noun['text']}"
        first_phrase = phrase_pair["above" if order == "first_subject" else "below"]
        second_phrase = phrase_pair["below" if order == "first_subject" else "above"]
        texts = (
            structure["template"].format(subject=subject, phrase=first_phrase, reference=reference),
            structure["template"].format(subject=subject, phrase=second_phrase, reference=reference),
            f"The first {noun['text']} is above the second {noun['text']}.",
            f"The first {noun['text']} is below the second {noun['text']}.",
        )
        suffix = f"entity_{noun['id']}_{order}_{structure['id']}"
    else:
        glosses = {relation: structure["template"].format(phrase=phrase_pair[relation])
                   for relation in ("above", "below")}
        if word_first:
            texts = (stem["above"], stem["below"], glosses["above"], glosses["below"])
        else:
            texts = (glosses["above"].capitalize(), glosses["below"].capitalize(),
                     structure["above_target"], structure["below_target"])
        suffix = stem["id"] if stem else "reverse"
    row = {
        "probe_id": "__".join((direction, suffix, structure["id"], phrase_pair["id"]))
            if not extension else "__".join((direction, suffix, phrase_pair["id"])),
        "probe_version": config["version"],
        "probe_family": "lexical_synonym" if "synonym" in direction else "literal_definition",
        "design_block": "entity_order_extension" if extension else "base",
        "direction": direction,
        "context_stem_id": stem["id"] if stem else "",
        "definition_structure_id": ("entity_" if extension else "") + structure["id"],
        "entity_noun_id": noun["id"] if extension else "",
        "entity_noun": noun["text"] if extension else "",
        "context_entity_order": order if extension else (
            "not_applicable" if word_first else "first_subject" if structure["id"] == "ordered"
            else "generic"),
        "phrase_pair_id": phrase_pair["id"],
        "phrase_register": phrase_pair["register"],
        "above_phrase": phrase_pair["above"],
        "below_phrase": phrase_pair["below"],
        "Domain": "spatial-relations",
        "ConceptA": "above",
        "ConceptB": "below",
        "ContextType": "direct",
        "ContextDiff": "vertical relation polarity",
        "TargetDiff": "concept swap",
        **dict(zip(TEXT_FIELDS, texts)),
        "correct_target_for_context1": "Target1",
        "correct_target_for_context2": "Target2",
    }
    for field in TEXT_FIELDS:
        row[f"{field}_word_count"] = len(re.findall(r"\b\w+\b", row[field]))
        row[f"{field}_char_count"] = len(row[field])
    return row


def main():
    config = json.loads((ROOT / "components.json").read_text(encoding="utf-8"))
    stems = config["context_stems"]
    phrases = config["phrase_pairs"]
    synonyms = config["lexical_synonym_pairs"]
    structures = config["definition_structures"]
    nouns = config["reverse_entity_nouns"]
    reverse_structures = config["reverse_context_structures"]
    rows = [make_row(config, "word_to_definition", phrase, structure, stem)
            for stem, phrase, structure in product(stems, phrases, structures)]
    rows += [make_row(config, "definition_to_word", phrase, structure)
             for phrase, structure in product(phrases, structures)]
    rows += [make_row(config, "word_to_synonym", phrase, structure, stem)
             for stem, phrase, structure in product(stems, synonyms, structures)]
    rows += [make_row(config, "synonym_to_word", phrase, structure)
             for phrase, structure in product(synonyms, structures)]
    rows += [make_row(config, "definition_to_word" if phrase in phrases else "synonym_to_word",
                      phrase, structure, noun=noun, order=order)
             for phrase, noun, order, structure in product(
                 phrases + synonyms, nouns, ("first_subject", "second_subject"), reverse_structures)]
    for index, row in enumerate(rows):
        row["probe_row_index"] = index

    expected_base = (len(phrases) + len(synonyms)) * len(structures) * (len(stems) + 1)
    expected_extension = (len(phrases) + len(synonyms)) * len(nouns) * 2 * len(reverse_structures)
    expected = expected_base + expected_extension
    assert len(rows) == expected
    assert len({row["probe_id"] for row in rows}) == expected
    assert len({tuple(row[field] for field in TEXT_FIELDS) for row in rows}) == expected
    assert Counter(row["phrase_pair_id"] for row in rows) == {
        phrase["id"]: len(structures) * (len(stems) + 1) + len(nouns) * 2 * len(reverse_structures)
        for phrase in phrases + synonyms}
    for row in rows:
        assert all(row[field] and "{" not in row[field] for field in TEXT_FIELDS)
        assert row["Context1"] != row["Context2"]
        assert row["Target1"] != row["Target2"]
        gloss_fields = ("Target1", "Target2") if row["direction"] in (
            "word_to_definition", "word_to_synonym") else ("Context1", "Context2")
        first, second = (row[field].lower() for field in gloss_fields)
        first_phrase, second_phrase = (
            (row["below_phrase"], row["above_phrase"])
            if row["context_entity_order"] == "second_subject"
            else (row["above_phrase"], row["below_phrase"]))
        assert first.replace(first_phrase, "<direction>") == second.replace(
            second_phrase, "<direction>")
        assert row[f"{gloss_fields[0]}_word_count"] == row[f"{gloss_fields[1]}_word_count"]
        assert "above" not in first and "below" not in second

    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    (out / "probes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                               for row in rows), encoding="utf-8")
    write_csv(out / "probes.csv", rows)
    for name in ("context_stems", "phrase_pairs", "lexical_synonym_pairs", "definition_structures",
                 "reverse_entity_nouns", "reverse_context_structures"):
        write_csv(out / f"{name}.csv", config[name])
    matches = []
    for phrase, structure in product(phrases, structures):
        group = {"phrase_pair_id": phrase["id"], "definition_structure_id": structure["id"],
                 "definition_to_word_probe_id": "__".join(("definition_to_word", "reverse",
                     structure["id"], phrase["id"]))}
        for stem in stems:
            group[f"word_to_definition_{stem['id']}_probe_id"] = "__".join(
                ("word_to_definition", stem["id"], structure["id"], phrase["id"]))
        matches.append(group)
    write_csv(out / "direction_matches.csv", matches)
    synonym_matches = []
    for phrase, structure in product(synonyms, structures):
        group = {"phrase_pair_id": phrase["id"], "definition_structure_id": structure["id"],
                 "synonym_to_word_probe_id": "__".join(("synonym_to_word", "reverse",
                     structure["id"], phrase["id"]))}
        for stem in stems:
            group[f"word_to_synonym_{stem['id']}_probe_id"] = "__".join(
                ("word_to_synonym", stem["id"], structure["id"], phrase["id"]))
        synonym_matches.append(group)
    write_csv(out / "synonym_direction_matches.csv", synonym_matches)
    entity_order_matches = []
    for phrase, noun, structure in product(phrases + synonyms, nouns, reverse_structures):
        direction = "definition_to_word" if phrase in phrases else "synonym_to_word"
        entity_order_matches.append({
            "probe_family": "literal_definition" if phrase in phrases else "lexical_synonym",
            "phrase_pair_id": phrase["id"], "entity_noun_id": noun["id"],
            "definition_structure_id": "entity_" + structure["id"],
            **{f"{order}_probe_id": "__".join((direction,
                f"entity_{noun['id']}_{order}_{structure['id']}", phrase["id"]))
               for order in ("first_subject", "second_subject")},
        })
    write_csv(out / "entity_order_matches.csv", entity_order_matches)
    manifest = {"probe_name": config["probe_name"], "version": config["version"],
                "paired_probes": expected, "individual_choice_judgments": 2 * expected,
                "conditional_likelihoods_to_compute": 4 * expected,
                "by_direction": dict(Counter(row["direction"] for row in rows)),
                "phrase_pairs": len(phrases), "lexical_synonym_pairs": len(synonyms),
                "definition_structures": len(structures),
                "reverse_entity_nouns": len(nouns),
                "reverse_context_structures": len(reverse_structures),
                "entity_order_extension_pairs": expected_extension,
                "entity_order_matched_groups": len(entity_order_matches),
                "matched_direction_groups": len(matches),
                "synonym_matched_direction_groups": len(synonym_matches),
                "validation": "passed",
                "model_evaluation_performed": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
