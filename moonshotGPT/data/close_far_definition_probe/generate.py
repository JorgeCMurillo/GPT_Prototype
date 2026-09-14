#!/usr/bin/env python3
"""Generate the reviewed factorial design, using only the Python standard library."""

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


def main():
    config = json.loads((ROOT / "components.json").read_text(encoding="utf-8"))
    rows = []
    definitions = list(product(config["definition_structures"], config["modifiers"], config["adjective_pairs"]))

    def make_row(direction, definition, modifier, adjectives, stem=None):
        variant_id = f"{definition['id']}_{modifier['id']}_{adjectives['id']}"
        descriptions = [definition["template"].format(modifier=modifier["text"], adjective=adjectives[c]) for c in ("close", "far")]
        if direction == "word_to_definition":
            texts = [stem["close"], stem["far"], *descriptions]
            suffix = stem["id"]
        else:
            targets = config["label_targets"][0]
            texts = [s[0].upper() + s[1:] for s in descriptions] + [targets["close"], targets["far"]]
            suffix = targets["id"]
        row = {
            "probe_id": f"{direction}__{suffix}__{variant_id}",
            "probe_version": config["version"],
            "direction": direction,
            "definition_variant_id": variant_id,
            "context_stem_id": stem["id"] if stem else None,
            "context_structure_label": stem["structure_label"] if stem else None,
            "definition_structure_id": definition["id"],
            "modifier_id": modifier["id"],
            "modifier": modifier["text"],
            "adjective_pair_id": adjectives["id"],
            "close_adjective": adjectives["close"],
            "far_adjective": adjectives["far"],
            "label_target_id": "label_0" if stem is None else None,
            "Domain": "spatial-relations",
            "ConceptA": "close",
            "ConceptB": "far",
            "ContextType": "direct",
            "ContextDiff": "antonym",
            "TargetDiff": "concept swap",
            **dict(zip(TEXT_FIELDS, texts)),
            "correct_target_for_context1": "Target1",
            "correct_target_for_context2": "Target2",
        }
        for key, value in zip(TEXT_FIELDS, texts):
            row[f"{key}_word_count"] = len(re.findall(r"\b\w+\b", value))
            row[f"{key}_char_count"] = len(value)
        return row

    for stem in config["context_stems"]:
        for definition, modifier, adjectives in definitions:
            rows.append(make_row("word_to_definition", definition, modifier, adjectives, stem))
    for definition, modifier, adjectives in definitions:
        rows.append(make_row("definition_to_word", definition, modifier, adjectives))
    for index, row in enumerate(rows):
        row["probe_row_index"] = index

    # Validate the intended design, its counterfactual contrasts, and its join keys.
    assert len(rows) == 72
    assert Counter(r["direction"] for r in rows) == {"word_to_definition": 54, "definition_to_word": 18}
    assert len({r["probe_id"] for r in rows}) == 72
    assert len({tuple(r[k] for k in TEXT_FIELDS) for r in rows}) == 72
    assert Counter(r["definition_variant_id"] for r in rows) == {r["definition_variant_id"]: 4 for r in rows}
    for row in rows:
        assert all(row[k] and "{" not in row[k] for k in TEXT_FIELDS)
        definition_fields = ("Target1", "Target2") if row["direction"] == "word_to_definition" else ("Context1", "Context2")
        a, b = (row[k] for k in definition_fields)
        assert re.sub(rf"\b{row['close_adjective']}\b", "<distance>", a) == re.sub(rf"\b{row['far_adjective']}\b", "<distance>", b)
        assert row[f"{definition_fields[0]}_word_count"] == row[f"{definition_fields[1]}_word_count"]
        if row["direction"] == "definition_to_word":
            assert row["Target1"] == "The objects are close."
            assert row["Target2"] == "The objects are far."

    out = ROOT / "generated"
    out.mkdir(exist_ok=True)
    with (out / "probes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(out / "probes.csv", rows)
    for component in ("directions", "context_stems", "definition_structures", "modifiers", "adjective_pairs", "label_targets"):
        write_csv(out / f"{component}.csv", config[component])
    groups = []
    for definition, modifier, adjectives in definitions:
        variant_id = f"{definition['id']}_{modifier['id']}_{adjectives['id']}"
        group = {"definition_variant_id": variant_id}
        group["definition_to_word_probe_id"] = f"definition_to_word__label_0__{variant_id}"
        for stem in config["context_stems"]:
            group[f"word_to_definition_{stem['id']}_probe_id"] = f"word_to_definition__{stem['id']}__{variant_id}"
        groups.append(group)
    write_csv(out / "direction_matches.csv", groups)
    manifest = {
        "probe_name": config["probe_name"], "version": config["version"],
        "paired_probes": len(rows), "individual_completion_choice_judgments": 2 * len(rows),
        "conditional_likelihoods_to_compute": 4 * len(rows),
        "by_direction": dict(Counter(r["direction"] for r in rows)),
        "definition_variants": len(definitions),
        "matched_direction_groups": len(groups),
        "validation": "passed", "model_evaluation_performed": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
