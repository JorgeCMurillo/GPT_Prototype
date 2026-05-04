from collections import Counter

from research.bos_aligned_proto.spatial_synth.generate_spatial_relations_csv import (
    DIFFICULTY_LABELS,
    generate_cardinal_guard_item,
    generate_distance_contrast_item,
    generate_distance_reciprocal_item,
    generate_left_right_contrast_item,
    generate_left_right_paired_contrast_item,
    generate_pass_by_implicit_item,
    generate_pass_through_implicit_item,
    generate_relation_type_contrast_item,
    generate_turn_around_lr_item,
    generate_turn_lr_contrast_item,
    generate_turn_lr_order_variant_item,
    generate_turn_implicit_item,
    generate_items,
    generate_vertical_implicit_item,
    item_to_row,
)
from research.bos_aligned_proto.spatial_synth.train_spatial_relations_causal_lm import (
    TextRow,
    pack_texts,
    select_rows,
)


def test_generate_items_mixed_balances_difficulty_labels() -> None:
    items = generate_items(n=12, seed=123, difficulty="mixed")

    counts = Counter(item.difficulty_label for item in items)
    assert counts == {label: 4 for label in DIFFICULTY_LABELS}
    assert all(item.text for item in items)
    assert all(item.difficulty in {1, 2, 3} for item in items)


def test_generate_items_v3_preset_excludes_pass_through_family() -> None:
    items = generate_items(n=30, seed=123, difficulty="mixed", template_preset="v3")

    assert all(item.template_family != "implicit_pass_through" for item in items)


def test_left_right_contrast_item_flips_side_when_facing_around() -> None:
    import random

    item = generate_left_right_contrast_item(
        random.Random(0),
        agent="Ava",
        obj="key",
        side="right",
        before_facing="north",
        template_variant=0,
    )

    assert item.template_family == "implicit_left_right_contrast"
    assert item.before_world == "east"
    assert item.before_relative == "right"
    assert item.after_relative == "left"
    assert "faced north" in item.text
    assert "faced south" in item.text


def test_turn_lr_contrast_maps_front_left_turn_to_right_side() -> None:
    import random

    item = generate_turn_lr_contrast_item(
        random.Random(0),
        agent="Ava",
        obj="key",
        start_relation="front",
        turn="left",
        before_facing="north",
        template_variant=0,
    )

    assert item.template_family == "implicit_turn_lr_contrast"
    assert item.before_world == "north"
    assert item.before_relative == "front"
    assert item.after_relative == "right"
    assert item.operation == "implicit_turn_lr_contrast_front_turn_left"
    assert "pivoted left" in item.text
    assert "right-hand side" in item.text


def test_turn_lr_order_variant_maps_front_right_turn_to_left_side() -> None:
    import random

    item = generate_turn_lr_order_variant_item(
        random.Random(0),
        agent="Ava",
        obj="key",
        start_relation="front",
        turn="right",
        before_facing="north",
        template_variant=0,
    )

    assert item.template_family == "implicit_turn_lr_order_variants"
    assert item.before_world == "north"
    assert item.before_relative == "front"
    assert item.after_relative == "left"
    assert item.operation == "implicit_turn_lr_order_variants_front_turn_right"
    assert "turned right" in item.text
    assert "to Ava's left" in item.text


def test_left_right_paired_contrast_role_inversion_flips_reference() -> None:
    import random

    item = generate_left_right_paired_contrast_item(
        random.Random(0),
        agent="Ava",
        obj="key",
        contrast_case="role_inversion",
        side="left",
        before_facing="north",
        template_variant=0,
    )

    assert item.template_family == "implicit_left_right_paired_contrast"
    assert item.before_relative == "left"
    assert item.operation == "implicit_left_right_paired_contrast_role_inversion_left"
    assert "to Ava's left" in item.text
    assert "to the key's right" in item.text


def test_left_right_paired_contrast_turn_direction_gives_opposite_sides() -> None:
    import random

    item = generate_left_right_paired_contrast_item(
        random.Random(0),
        agent="Ava",
        obj="key",
        contrast_case="turn_direction_flip",
        start_relation="front",
        before_facing="north",
        template_variant=0,
    )

    assert item.template_family == "implicit_left_right_paired_contrast"
    assert item.before_relative == "front"
    assert item.after_relative == "right"
    assert "turned left" in item.text
    assert "to Ava's right" in item.text
    assert "turned right" in item.text
    assert "to Ava's left" in item.text


def test_distance_contrast_supports_inverse_far_relation() -> None:
    import random

    item = generate_distance_contrast_item(
        random.Random(0),
        agent="Ava",
        obj="key",
        distance_case="reach_far",
        relation_view="inverse",
        direction="north",
        before_facing="north",
    )

    assert item.template_family == "implicit_distance_contrast"
    assert item.operation == "implicit_distance_contrast_reach_far_inverse"
    assert item.before_world == "north"
    assert item.before_relative == "front"
    assert "Ava was far from the key" in item.text


def test_v8_to_v12_hypothesis_families_generate_valid_items() -> None:
    import random

    rng = random.Random(0)
    reciprocal = generate_distance_reciprocal_item(rng, agent="Ava", obj="key", relation="close", direction="north")
    around = generate_turn_around_lr_item(rng, agent="Ava", obj="key", side="right", before_facing="north")
    cardinal = generate_cardinal_guard_item(rng, agent="Ava", obj="key", case="pass", relation="east")
    relation_type = generate_relation_type_contrast_item(rng, agent="Ava", obj="key", case="inverse_cardinal")

    assert reciprocal.template_family == "implicit_distance_reciprocal"
    assert "close" in reciprocal.text
    assert "Ava" in reciprocal.text and "key" in reciprocal.text
    assert around.template_family == "implicit_turn_around_lr"
    assert around.before_relative == "right"
    assert around.after_relative == "left"
    assert cardinal.template_family == "implicit_cardinal_guard"
    assert cardinal.before_world == "east"
    assert cardinal.after_world == "west"
    assert relation_type.template_family == "implicit_relation_type_contrast"


def test_v13_to_v15_presets_generate_valid_items() -> None:
    for preset in ("v13", "v14", "v15"):
        items = generate_items(n=9, seed=123, difficulty="mixed", template_preset=preset)
        assert len(items) == 9
        assert all(item.text for item in items)


def test_v19_to_v21_presets_generate_valid_items() -> None:
    for preset in ("v19", "v20", "v21"):
        items = generate_items(n=12, seed=123, difficulty="mixed", template_preset=preset)
        assert len(items) == 12
        assert all(item.text for item in items)


def test_generated_csv_row_contains_training_text_and_metadata() -> None:
    item = generate_items(n=1, seed=7, difficulty="mixed")[0]
    row = item_to_row(item, seed=7)

    assert row["domain"] == "spatial-relations"
    assert row["text"] == item.text
    assert row["context"]
    assert row["completion"]
    assert item.text.endswith(str(row["completion"]))
    assert row["difficulty_label"] in DIFFICULTY_LABELS
    assert "before_state" in row
    assert "after_state" in row
    assert "template_family" in row


def test_vertical_implicit_inverse_role_item_is_supported() -> None:
    import random

    item = generate_vertical_implicit_item(
        random.Random(0),
        agent="Ava",
        obj="lantern",
        mover="a",
        direction="down",
        relation_view="inverse",
    )

    assert item.template_family == "implicit_vertical"
    assert item.operation == "implicit_vertical_agent_moves_down_inverse"
    assert item.observer == "a"
    assert item.after_world == "above"
    assert "the lantern" in item.text
    assert "above Ava" in item.text


def test_implicit_turn_and_pass_by_items_are_generated() -> None:
    import random

    turn_item = generate_turn_implicit_item(random.Random(1), agent="Mira", obj="box", turn="right")
    pass_item = generate_pass_by_implicit_item(random.Random(2), agent="Kai", obj="statue")
    pass_through_item = generate_pass_through_implicit_item(
        random.Random(4),
        agent="Noah",
        obj="mug",
        relation_view="object_egocentric",
        template_variant=0,
    )
    egocentric_item = generate_turn_implicit_item(
        random.Random(3),
        agent="Lena",
        obj="key",
        turn="around",
        template_variant=3,
    )

    assert turn_item.template_family == "implicit_turn"
    assert turn_item.operation == "implicit_turn_agent_right"
    assert "stayed" in turn_item.text or "remained" in turn_item.text
    assert egocentric_item.template_family == "implicit_turn"
    assert "would need to look" in egocentric_item.text
    assert pass_item.template_family == "implicit_pass_by"
    assert pass_item.after_relative == "behind"
    assert "past" in pass_item.text or "beyond" in pass_item.text
    assert pass_through_item.template_family == "implicit_pass_through"
    assert pass_through_item.observer == "a"
    assert pass_through_item.after_relative == "behind"
    assert "without stopping" in pass_through_item.text


def test_select_rows_balances_mixed_training_rows() -> None:
    rows = [
        TextRow(text=f"{label} {i}", difficulty=label, source={})
        for label in DIFFICULTY_LABELS
        for i in range(5)
    ]

    selected = select_rows(rows, difficulty="mixed", balance_mixed=True, max_examples=9, seed=0)

    counts = Counter(row.difficulty for row in selected)
    assert counts == {label: 3 for label in DIFFICULTY_LABELS}


class _TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(1, len(text.split()) + 1))


def test_pack_texts_pads_last_block_and_masks_padding_labels() -> None:
    dataset, stats = pack_texts(["one two"], _TinyTokenizer(), block_size=4)

    assert stats.examples == 1
    assert stats.tokens_with_eos == 3
    assert stats.blocks == 1
    assert stats.loss_tokens == 2
    assert stats.loss_token_fraction == 2 / 3
    assert stats.loss_weight_sum == 2.0
    assert len(dataset) == 1
    assert dataset[0]["input_ids"].tolist() == [1, 2, 0, 0]
    assert dataset[0]["attention_mask"].tolist() == [1, 1, 1, 0]
    assert dataset[0]["labels"].tolist() == [1, 2, 0, -100]
    assert dataset[0]["loss_weights"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_pack_texts_completion_loss_masks_context_labels() -> None:
    row = TextRow(
        text="The mug moved. It finished above Ava.",
        context="The mug moved.",
        completion="It finished above Ava.",
        difficulty="hard",
        source={},
    )

    dataset, stats = pack_texts([row], _TinyTokenizer(), block_size=8, loss_mode="completion")

    assert stats.examples == 1
    assert stats.tokens_with_eos == 8
    assert stats.loss_tokens == 5
    assert stats.loss_weight_sum == 5.0
    assert dataset[0]["input_ids"].tolist() == [1, 2, 3, 1, 2, 3, 4, 0]
    assert dataset[0]["attention_mask"].tolist() == [1, 1, 1, 1, 1, 1, 1, 1]
    assert dataset[0]["labels"].tolist() == [-100, -100, -100, 1, 2, 3, 4, 0]
    assert dataset[0]["loss_weights"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_pack_texts_mixed_loss_uses_full_and_completion_only_examples() -> None:
    rows = [
        TextRow(
            text="The mug moved. It finished above Ava.",
            context="The mug moved.",
            completion="It finished above Ava.",
            difficulty="hard",
            source={},
        )
        for _ in range(10)
    ]

    dataset, stats = pack_texts(
        rows,
        _TinyTokenizer(),
        block_size=8,
        loss_mode="mixed",
        mixed_full_loss_ratio=0.7,
        seed=0,
    )

    assert stats.examples == 10
    assert stats.tokens_with_eos == 80
    assert stats.full_loss_examples == 7
    assert stats.completion_loss_examples == 3
    assert stats.weighted_loss_examples == 0
    assert stats.loss_tokens == 64
    completion_blocks = sum(
        1 for idx in range(len(dataset)) if dataset[idx]["labels"].tolist()[:3] == [-100, -100, -100]
    )
    assert completion_blocks == 3


def test_pack_texts_weighted_loss_weights_context_and_completion() -> None:
    row = TextRow(
        text="The mug moved. It finished above Ava.",
        context="The mug moved.",
        completion="It finished above Ava.",
        difficulty="hard",
        source={},
    )

    dataset, stats = pack_texts([row], _TinyTokenizer(), block_size=8, loss_mode="weighted")

    assert stats.examples == 1
    assert stats.tokens_with_eos == 8
    assert stats.loss_tokens == 7
    assert stats.weighted_loss_examples == 1
    assert round(stats.loss_weight_sum, 6) == 1.0
    assert dataset[0]["labels"].tolist() == [1, 2, 3, 1, 2, 3, 4, 0]
    assert [round(x, 2) for x in dataset[0]["loss_weights"].tolist()] == [
        0.0,
        0.15,
        0.15,
        0.14,
        0.14,
        0.14,
        0.14,
        0.14,
    ]
