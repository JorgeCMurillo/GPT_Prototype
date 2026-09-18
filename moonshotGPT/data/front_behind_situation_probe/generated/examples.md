# Compact front/behind pairs

Compact-only, with no definition prefix or answer bridge. Not evaluated.

## static_adjacent / nonnumeric

Context 1: B faces the door. A stands between B and the door.

Context 2: B faces the door. B stands between A and the door.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## target_cross_forward / nonnumeric

Context 1: A starts behind B, then walks straight past B. B stays still without turning.

Context 2: A starts in front of B, then walks straight past B. B stays still without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## reference_cross_forward / nonnumeric

Context 1: B faces A, then walks straight past A without turning. A stays still.

Context 2: B faces away from A, then steps backward past A without turning. A stays still.

Answers: A is in front of B. / A is behind B.

Gold: Target2, Target1.

## approach_without_crossing / nonnumeric

Context 1: B faces away from A. A approaches B but stops before reaching B. B stays still without turning.

Context 2: B faces A. A approaches B but stops before reaching B. B stays still without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target2, Target1.

## recede_without_crossing / nonnumeric

Context 1: B faces A. A walks straight away from B. B stays still without turning.

Context 2: B faces away from A. A walks straight away from B. B stays still without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## both_same_direction / nonnumeric

Context 1: A starts in front of B. Both move equal distances in B's facing direction, without turning.

Context 2: A starts behind B. Both move equal distances opposite B's facing direction, without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## both_opposite_preserved / nonnumeric

Context 1: B faces A. They move directly away from each other without turning.

Context 2: B faces away from A. They move directly away from each other without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## both_reverse / nonnumeric

Context 1: B faces away from A. They move straight past each other without turning.

Context 2: B faces A. They move straight past each other without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## static_adjacent / numeric

Context 1: B faces increasing position numbers. A stays at 3. B stays at 2.

Context 2: B faces increasing position numbers. A stays at 3. B stays at 4.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## static_separated / numeric

Context 1: B faces increasing position numbers. A stays at 5. B stays at 1.

Context 2: B faces increasing position numbers. A stays at 1. B stays at 5.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## static_interior / numeric

Context 1: B faces increasing position numbers. A stays at 2. B stays at 4.

Context 2: B faces increasing position numbers. A stays at 4. B stays at 2.

Answers: A is in front of B. / A is behind B.

Gold: Target2, Target1.

## target_cross_forward / numeric

Context 1: B faces increasing position numbers. A moves from 1 to 5. B stays at 3.

Context 2: B faces increasing position numbers. A moves from 5 to 1. B stays at 3.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## reference_cross_forward / numeric

Context 1: B faces increasing position numbers. A stays at 3. B moves from 1 to 5 without turning.

Context 2: B faces increasing position numbers. A stays at 3. B moves from 5 to 1 without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target2, Target1.

## approach_without_crossing / numeric

Context 1: B faces increasing position numbers. A moves from 1 to 3. B stays at 4.

Context 2: B faces increasing position numbers. A moves from 5 to 3. B stays at 2.

Answers: A is in front of B. / A is behind B.

Gold: Target2, Target1.

## recede_without_crossing / numeric

Context 1: B faces increasing position numbers. A moves from 3 to 5. B stays at 2.

Context 2: B faces increasing position numbers. A moves from 3 to 1. B stays at 4.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## both_same_direction / numeric

Context 1: B faces increasing position numbers. A moves from 3 to 5. B moves from 1 to 3 without turning.

Context 2: B faces increasing position numbers. A moves from 3 to 1. B moves from 5 to 3 without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## both_opposite_preserved / numeric

Context 1: B faces increasing position numbers. A moves from 4 to 5. B moves from 2 to 1 without turning.

Context 2: B faces increasing position numbers. A moves from 2 to 1. B moves from 4 to 5 without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## both_reverse / numeric

Context 1: B faces increasing position numbers. A moves from 1 to 4. B moves from 4 to 2 without turning.

Context 2: B faces increasing position numbers. A moves from 5 to 2. B moves from 2 to 4 without turning.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## direct_labels / direct_label

Context 1: A is in front of B.

Context 2: A is behind B.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## observer_turn_front / initial_relation_then_action

Context 1: A is in front of B. B does not turn. Neither person changes location.

Context 2: A is in front of B. B turns halfway around in place. Neither person changes location.

Answers: A is in front of B. / A is behind B.

Gold: Target1, Target2.

## observer_turn_behind / initial_relation_then_action

Context 1: A is behind B. B does not turn. Neither person changes location.

Context 2: A is behind B. B turns halfway around in place. Neither person changes location.

Answers: A is in front of B. / A is behind B.

Gold: Target2, Target1.
