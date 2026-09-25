# Neutral-context review examples

Every generated rule/family/evidence combination is represented. These examples are pending human review.

## east_west / target_moves_without_crossing / direct_initial_relation

Rule: `initial_relation_unspecified`; source: `east_west__away__heads__not_applicable__implicit`

C1: A is west of B. A heads farther west. Final positions:

C2: A is east of B. A heads farther east. Final positions:

C0: A and B are at different positions on an east-west line. A heads directly farther away from B. Final positions:

T1: A is east of B.

T2: A is west of B.

## east_west / both_move / direct_initial_relation

Rule: `initial_relation_unspecified`; source: `east_west__both_equal__heads__not_applicable__not_applicable`

C1: A is west of B. Both head east for the same distance. Final positions:

C2: A is east of B. Both head west for the same distance. Final positions:

C0: A and B are at different positions on an east-west line. Both head the same distance in the same direction. Final positions:

T1: A is east of B.

T2: A is west of B.

## east_west / reference_crosses / direct_initial_relation

Rule: `initial_relation_unspecified`; source: `east_west__reference_crosses__heads__to__implicit`

C1: A is west of B. B heads straight to A and continues past A. Final positions:

C2: A is east of B. B heads straight to A and continues past A. Final positions:

C0: A and B are at different positions on an east-west line. B heads straight to A and continues past A. Final positions:

T1: A is east of B.

T2: A is west of B.

## east_west / target_crosses / direct_initial_relation

Rule: `initial_relation_unspecified`; source: `east_west__target_crosses__heads__to__implicit`

C1: A is west of B. A heads straight to B and continues past B. Final positions:

C2: A is east of B. A heads straight to B and continues past B. Final positions:

C0: A and B are at different positions on an east-west line. A heads straight to B and continues past B. Final positions:

T1: A is east of B.

T2: A is west of B.

## above_below / both_move / measured_height

Rule: `unordered_equal_comotion`; source: `both_preserve_down__ball_cone__measured_height__compact__reference_first__reference_first`

C1: Heights are measured from the floor. The cone moves from a height of 3 feet to a height of 2 feet, measured from the floor. The ball moves from a height of 5 feet to a height of 4 feet, measured from the floor.

C2: Heights are measured from the floor. The cone moves from a height of 5 feet to a height of 4 feet, measured from the floor. The ball moves from a height of 3 feet to a height of 2 feet, measured from the floor.

C0: Heights are measured from the floor. The cone and the ball each move down by the same amount and finish at different heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / named_shelves

Rule: `unordered_equal_comotion`; source: `both_preserve_down__ball_cone__named_shelves__compact__reference_first__reference_first`

C1: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves from the middle shelf to the lower shelf. The ball moves from the top shelf to the upper shelf.

C2: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves from the top shelf to the upper shelf. The ball moves from the middle shelf to the lower shelf.

C0: The rack's shelves run from bottom through lower, middle, and upper to top. The cone and the ball each move down by the same amount and finish at different heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / numbered_floors

Rule: `unordered_equal_comotion`; source: `both_preserve_down__ball_cone__numbered_floors__compact__reference_first__reference_first`

C1: Floors are numbered upward, 1 to 5. The cone moves from floor 3 to floor 2. The ball moves from floor 5 to floor 4.

C2: Floors are numbered upward, 1 to 5. The cone moves from floor 5 to floor 4. The ball moves from floor 3 to floor 2.

C0: Floors are numbered upward, 1 to 5. The cone and the ball each move down by the same amount and finish at different heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / numbered_steps

Rule: `unordered_equal_comotion`; source: `both_preserve_down__ball_cone__numbered_steps__compact__reference_first__reference_first`

C1: Steps 1 to 5 go up. The cone moves from step 3 to step 2. The ball moves from step 5 to step 4.

C2: Steps 1 to 5 go up. The cone moves from step 5 to step 4. The ball moves from step 3 to step 2.

C0: Steps 1 to 5 go up. The cone and the ball each move down by the same amount and finish at different heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / measured_height

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__measured_height__compact__reference_first__reference_first`

C1: Heights are measured from the floor. The cone moves from a height of 4 feet to a height of 2 feet, measured from the floor. The ball moves from a height of 2 feet to a height of 4 feet, measured from the floor.

C2: Heights are measured from the floor. The cone moves from a height of 2 feet to a height of 4 feet, measured from the floor. The ball moves from a height of 4 feet to a height of 2 feet, measured from the floor.

C0: Heights are measured from the floor. The cone and the ball begin at two different heights and exchange heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / named_shelves

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__named_shelves__compact__reference_first__reference_first`

C1: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves from the upper shelf to the lower shelf. The ball moves from the lower shelf to the upper shelf.

C2: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves from the lower shelf to the upper shelf. The ball moves from the upper shelf to the lower shelf.

C0: The rack's shelves run from bottom through lower, middle, and upper to top. The cone and the ball begin at two different heights and exchange heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / numbered_floors

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__numbered_floors__compact__reference_first__reference_first`

C1: Floors are numbered upward, 1 to 5. The cone moves from floor 4 to floor 2. The ball moves from floor 2 to floor 4.

C2: Floors are numbered upward, 1 to 5. The cone moves from floor 2 to floor 4. The ball moves from floor 4 to floor 2.

C0: Floors are numbered upward, 1 to 5. The cone and the ball begin at two different heights and exchange heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / both_move / numbered_steps

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__numbered_steps__compact__reference_first__reference_first`

C1: Steps 1 to 5 go up. The cone moves from step 4 to step 2. The ball moves from step 2 to step 4.

C2: Steps 1 to 5 go up. The cone moves from step 2 to step 4. The ball moves from step 4 to step 2.

C0: Steps 1 to 5 go up. The cone and the ball begin at two different heights and exchange heights.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_moves_without_crossing / measured_height

Rule: `side_unspecified_no_crossing`; source: `no_cross_down__ball_cone__measured_height__compact__reference_first__reference_first`

C1: Heights are measured from the floor. The cone remains at a height of 1 foot from the floor. The ball moves from a height of 5 feet to a height of 4 feet, measured from the floor.

C2: Heights are measured from the floor. The cone remains at a height of 5 feet from the floor. The ball moves from a height of 2 feet to a height of 1 foot, measured from the floor.

C0: Heights are measured from the floor. The cone remains in place. The ball moves down without reaching or passing the cone's level.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_moves_without_crossing / named_shelves

Rule: `side_unspecified_no_crossing`; source: `no_cross_down__ball_cone__named_shelves__compact__reference_first__reference_first`

C1: The rack's shelves run from bottom through lower, middle, and upper to top. The cone remains on the bottom shelf. The ball moves from the top shelf to the upper shelf.

C2: The rack's shelves run from bottom through lower, middle, and upper to top. The cone remains on the top shelf. The ball moves from the lower shelf to the bottom shelf.

C0: The rack's shelves run from bottom through lower, middle, and upper to top. The cone remains in place. The ball moves down without reaching or passing the cone's level.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_moves_without_crossing / numbered_floors

Rule: `side_unspecified_no_crossing`; source: `no_cross_down__ball_cone__numbered_floors__compact__reference_first__reference_first`

C1: Floors are numbered upward, 1 to 5. The cone stays on floor 1. The ball moves from floor 5 to floor 4.

C2: Floors are numbered upward, 1 to 5. The cone stays on floor 5. The ball moves from floor 2 to floor 1.

C0: Floors are numbered upward, 1 to 5. The cone remains in place. The ball moves down without reaching or passing the cone's level.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_moves_without_crossing / numbered_steps

Rule: `side_unspecified_no_crossing`; source: `no_cross_down__ball_cone__numbered_steps__compact__reference_first__reference_first`

C1: Steps 1 to 5 go up. The cone stays on step 1. The ball moves from step 5 to step 4.

C2: Steps 1 to 5 go up. The cone stays on step 5. The ball moves from step 2 to step 1.

C0: Steps 1 to 5 go up. The cone remains in place. The ball moves down without reaching or passing the cone's level.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / reference_crosses / measured_height

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__measured_height__compact__reference_first__reference_first`

C1: Heights are measured from the floor. The cone moves from a height of 5 feet to a height of 1 foot, measured from the floor. The ball remains at a height of 3 feet from the floor.

C2: Heights are measured from the floor. The cone moves from a height of 1 foot to a height of 5 feet, measured from the floor. The ball remains at a height of 3 feet from the floor.

C0: Heights are measured from the floor. The cone moves between a height of 1 foot from the floor and a height of 5 feet from the floor, from one to the other. The ball remains at a height of 3 feet from the floor.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / reference_crosses / named_shelves

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__named_shelves__compact__reference_first__reference_first`

C1: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves from the top shelf to the bottom shelf. The ball remains on the middle shelf.

C2: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves from the bottom shelf to the top shelf. The ball remains on the middle shelf.

C0: The rack's shelves run from bottom through lower, middle, and upper to top. The cone moves between the bottom shelf and the top shelf, from one to the other. The ball remains on the middle shelf.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / reference_crosses / numbered_floors

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__numbered_floors__compact__reference_first__reference_first`

C1: Floors are numbered upward, 1 to 5. The cone moves from floor 5 to floor 1. The ball stays on floor 3.

C2: Floors are numbered upward, 1 to 5. The cone moves from floor 1 to floor 5. The ball stays on floor 3.

C0: Floors are numbered upward, 1 to 5. The cone moves between floor 1 and floor 5, from one to the other. The ball remains on floor 3.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / reference_crosses / numbered_steps

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__numbered_steps__compact__reference_first__reference_first`

C1: Steps 1 to 5 go up. The cone moves from step 5 to step 1. The ball stays on step 3.

C2: Steps 1 to 5 go up. The cone moves from step 1 to step 5. The ball stays on step 3.

C0: Steps 1 to 5 go up. The cone moves between step 1 and step 5, from one to the other. The ball remains on step 3.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / static_placement / measured_height

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__measured_height__compact__reference_first__reference_first`

C1: Heights are measured from the floor. The cone is at a height of 2 feet from the floor. The ball is at a height of 4 feet from the floor.

C2: Heights are measured from the floor. The cone is at a height of 4 feet from the floor. The ball is at a height of 2 feet from the floor.

C0: Heights are measured from the floor. The cone and the ball occupy a height of 2 feet from the floor and a height of 4 feet from the floor, one object at each position.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / static_placement / named_shelves

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__named_shelves__compact__reference_first__reference_first`

C1: The rack's shelves run from bottom through lower, middle, and upper to top. The cone is on the lower shelf. The ball is on the upper shelf.

C2: The rack's shelves run from bottom through lower, middle, and upper to top. The cone is on the upper shelf. The ball is on the lower shelf.

C0: The rack's shelves run from bottom through lower, middle, and upper to top. The cone and the ball occupy the lower shelf and the upper shelf, one object at each position.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / static_placement / numbered_floors

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__numbered_floors__compact__reference_first__reference_first`

C1: Floors are numbered upward, 1 to 5. The cone is on floor 2. The ball is on floor 4.

C2: Floors are numbered upward, 1 to 5. The cone is on floor 4. The ball is on floor 2.

C0: Floors are numbered upward, 1 to 5. The cone and the ball occupy floor 2 and floor 4, one object at each position.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / static_placement / numbered_steps

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__numbered_steps__compact__reference_first__reference_first`

C1: Steps 1 to 5 go up. The cone is on step 2. The ball is on step 4.

C2: Steps 1 to 5 go up. The cone is on step 4. The ball is on step 2.

C0: Steps 1 to 5 go up. The cone and the ball occupy step 2 and step 4, one object at each position.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_crosses / measured_height

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__measured_height__compact__reference_first__reference_first`

C1: Heights are measured from the floor. The cone remains at a height of 3 feet from the floor. The ball moves from a height of 1 foot to a height of 5 feet, measured from the floor.

C2: Heights are measured from the floor. The cone remains at a height of 3 feet from the floor. The ball moves from a height of 5 feet to a height of 1 foot, measured from the floor.

C0: Heights are measured from the floor. The cone remains at a height of 3 feet from the floor. The ball moves between a height of 1 foot from the floor and a height of 5 feet from the floor, from one to the other.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_crosses / named_shelves

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__named_shelves__compact__reference_first__reference_first`

C1: The rack's shelves run from bottom through lower, middle, and upper to top. The cone remains on the middle shelf. The ball moves from the bottom shelf to the top shelf.

C2: The rack's shelves run from bottom through lower, middle, and upper to top. The cone remains on the middle shelf. The ball moves from the top shelf to the bottom shelf.

C0: The rack's shelves run from bottom through lower, middle, and upper to top. The cone remains on the middle shelf. The ball moves between the bottom shelf and the top shelf, from one to the other.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_crosses / numbered_floors

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__numbered_floors__compact__reference_first__reference_first`

C1: Floors are numbered upward, 1 to 5. The cone stays on floor 3. The ball moves from floor 1 to floor 5.

C2: Floors are numbered upward, 1 to 5. The cone stays on floor 3. The ball moves from floor 5 to floor 1.

C0: Floors are numbered upward, 1 to 5. The cone remains on floor 3. The ball moves between floor 1 and floor 5, from one to the other.

T1: The cone is below the ball.

T2: The cone is above the ball.

## above_below / target_crosses / numbered_steps

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__numbered_steps__compact__reference_first__reference_first`

C1: Steps 1 to 5 go up. The cone stays on step 3. The ball moves from step 1 to step 5.

C2: Steps 1 to 5 go up. The cone stays on step 3. The ball moves from step 5 to step 1.

C0: Steps 1 to 5 go up. The cone remains on step 3. The ball moves between step 1 and step 5, from one to the other.

T1: The cone is below the ball.

T2: The cone is above the ball.

## close_far / static_distance / distance_phrase

Rule: `unspecified_distance_magnitude`; source: `auditorium__name_jesse__obj_bag__distance_phrase__object_first`

C1: In the auditorium, a short distance separates Jesse and the bag.

C2: In the auditorium, a long distance separates Jesse and the bag.

C0: In the auditorium, a distance separates Jesse and the bag.

T1: The bag is close to Jesse.

T2: The bag is far from Jesse.

## close_far / static_distance / endpoint_placement

Rule: `unspecified_setting_position`; source: `auditorium__name_jesse__obj_bag__endpoint_placement__object_first`

C1: In the auditorium, Jesse stands at one side. The bag rests at Jesse's feet.

C2: In the auditorium, Jesse stands at one side. The bag rests at the opposite side.

C0: In the auditorium, Jesse stands at one side. The bag rests somewhere in the auditorium.

T1: The bag is close to Jesse.

T2: The bag is far from Jesse.

## closer_farther / object_moves_vs_both_move_cross_family / specific

Rule: `unspecified_before_after_distance`; source: `event_extension_v1_2__name_0__obj_bicycle__num_00__unit_ft__reference_object_moves__closer__compact__person_first__vs__name_0__obj_bicycle__num_00__unit_ft__both_move_same_separation__unchanged__compact__person_first`

C1: Maya stayed still; the bicycle, 18 ft from Maya, moved 5 ft directly toward Maya without crossing Maya's position.

C2: Starting 18 ft apart, Maya and the bicycle each moved 5 ft right.

C0: Initially, Maya and the bicycle were 18 ft apart. Their positions were recorded again after the interval.

T1: Maya is now closer to the bicycle than before.

T2: Maya is now at the same distance from the bicycle as before.

## closer_farther / reference_object_moves / specific

Rule: `unspecified_linear_motion_direction`; source: `event_extension_v1_2__name_0__obj_bicycle__num_00__unit_ft__reference_object_moves__closer__compact__person_first__vs__name_0__obj_bicycle__num_00__unit_ft__reference_object_moves__farther__compact__person_first__person_first`

C1: Maya stayed still; the bicycle, 18 ft from Maya, moved 5 ft directly toward Maya without crossing Maya's position.

C2: Maya stayed still; the bicycle, 18 ft from Maya, moved 5 ft directly away from Maya without crossing Maya's position.

C0: Maya stayed still; the bicycle, 18 ft from Maya, moved 5 ft along the straight line shared with Maya without crossing Maya's position.

T1: Maya is now closer to the bicycle than before.

T2: Maya is now farther from the bicycle than before.

## closer_farther / object_moves_vs_both_move_cross_family / absent

Rule: `unspecified_before_after_distance`; source: `event_extension_v1_2__name_0__obj_bicycle__reference_object_moves__closer__non_numeric__compact__person_first__vs__name_0__obj_bicycle__both_move_same_separation__unchanged__non_numeric__compact__person_first`

C1: Maya stayed still; the bicycle began some distance from Maya and moved directly toward Maya without crossing Maya's position.

C2: Starting some distance apart, Maya and the bicycle each moved equally far right.

C0: Initially, Maya and the bicycle were some distance apart. Their positions were recorded again after the interval.

T1: Maya is now closer to the bicycle than before.

T2: Maya is now at the same distance from the bicycle as before.

## closer_farther / reference_object_moves / absent

Rule: `unspecified_linear_motion_direction`; source: `event_extension_v1_2__name_0__obj_bicycle__reference_object_moves__closer__non_numeric__compact__person_first__vs__name_0__obj_bicycle__reference_object_moves__farther__non_numeric__compact__person_first__person_first`

C1: Maya stayed still; the bicycle began some distance from Maya and moved directly toward Maya without crossing Maya's position.

C2: Maya stayed still; the bicycle began some distance from Maya and moved directly away from Maya without crossing Maya's position.

C0: Maya stayed still; the bicycle began some distance from Maya and moved along the straight line shared with Maya without crossing Maya's position.

T1: Maya is now closer to the bicycle than before.

T2: Maya is now farther from the bicycle than before.

## closer_farther / person_moves_reference_stationary / absent

Rule: `unspecified_linear_motion_direction`; source: `person_movement_v1_2__non_numeric_v1_1__name_0__obj_bicycle__compact__legacy_wording`

C1: Starting some distance from a stationary bicycle, Maya walked some distance directly toward it without reaching or passing it.

C2: Starting some distance from a stationary bicycle, Maya walked some distance directly away from it without reaching or passing it.

C0: Starting some distance from a stationary bicycle, Maya walked some distance along the straight line containing both of them without reaching or passing it.

T1: Maya is now closer to the bicycle than before.

T2: Maya is now farther from the bicycle than before.

## closer_farther / person_moves_reference_stationary / specific

Rule: `unspecified_linear_motion_direction`; source: `person_movement_v1_2__numeric_aligned_v1_1__name_0__obj_bicycle__compact__num_00__unit_ft__legacy_wording`

C1: Starting 18 ft from a stationary bicycle, Maya walked 5 ft directly toward it without reaching or passing it.

C2: Starting 18 ft from a stationary bicycle, Maya walked 5 ft directly away from it without reaching or passing it.

C0: Starting 18 ft from a stationary bicycle, Maya walked 5 ft along the straight line containing both of them without reaching or passing it.

T1: Maya is now closer to the bicycle than before.

T2: Maya is now farther from the bicycle than before.

## east_west / both_move / named_locations

Rule: `unordered_equal_comotion`; source: `east_west__both_preserve_negative__named_locations__compact`

C1: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the village, the forest, the lake, the fountain. A moves from the fountain to the lake. B moves from the forest to the village. Final positions:

C2: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the village, the forest, the lake, the fountain. A moves from the forest to the village. B moves from the fountain to the lake. Final positions:

C0: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the village, the forest, the lake, the fountain. A and B each move one position west and finish at different positions. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / both_move / numeric

Rule: `unordered_equal_comotion`; source: `east_west__both_preserve_negative__numeric__compact`

C1: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3, 4. Both markers share a row. A moves from column 4 to column 3. B moves from column 2 to column 1. Final positions:

C2: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3, 4. Both markers share a row. A moves from column 2 to column 1. B moves from column 4 to column 3. Final positions:

C0: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3, 4. Both markers share a row. A and B each move one position west and finish at different positions. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / both_move / named_locations

Rule: `unordered_exchange`; source: `east_west__both_swap__named_locations__compact`

C1: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the tower, the bridge. A moves from the tower to the bridge. B moves from the bridge to the tower. Final positions:

C2: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the tower, the bridge. A moves from the bridge to the tower. B moves from the tower to the bridge. Final positions:

C0: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the tower, the bridge. A and B begin at two different positions and exchange positions. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / both_move / numeric

Rule: `unordered_exchange`; source: `east_west__both_swap__numeric__compact`

C1: On this fixed map, north is up and east is right. Columns run west to east: 1, 2. Both markers share a row. A moves from column 1 to column 2. B moves from column 2 to column 1. Final positions:

C2: On this fixed map, north is up and east is right. Columns run west to east: 1, 2. Both markers share a row. A moves from column 2 to column 1. B moves from column 1 to column 2. Final positions:

C0: On this fixed map, north is up and east is right. Columns run west to east: 1, 2. Both markers share a row. A and B begin at two different positions and exchange positions. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / target_moves_without_crossing / named_locations

Rule: `side_unspecified_no_crossing`; source: `east_west__no_cross_negative__named_locations__compact`

C1: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the station, the garden, the tower, the bridge. A moves from the bridge to the tower. B stays at the station. Final positions:

C2: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the station, the garden, the tower, the bridge. A moves from the garden to the station. B stays at the bridge. Final positions:

C0: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the station, the garden, the tower, the bridge. A moves one position west without reaching or passing B. B stays in place. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / target_moves_without_crossing / numeric

Rule: `side_unspecified_no_crossing`; source: `east_west__no_cross_negative__numeric__compact`

C1: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3, 4. Both markers share a row. A moves from column 4 to column 3. B stays at column 1. Final positions:

C2: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3, 4. Both markers share a row. A moves from column 2 to column 1. B stays at column 4. Final positions:

C0: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3, 4. Both markers share a row. A moves one position west without reaching or passing B. B stays in place. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / reference_crosses / named_locations

Rule: `direction_unspecified_crossing`; source: `east_west__reference_cross__named_locations__compact`

C1: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the cabin, the field, the mill. A stays at the field. B moves from the mill to the cabin. Final positions:

C2: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the cabin, the field, the mill. A stays at the field. B moves from the cabin to the mill. Final positions:

C0: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the cabin, the field, the mill. A stays at the field. B moves between the cabin and the mill, from one to the other. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / reference_crosses / numeric

Rule: `direction_unspecified_crossing`; source: `east_west__reference_cross__numeric__compact`

C1: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3. Both markers share a row. A stays at column 2. B moves from column 3 to column 1. Final positions:

C2: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3. Both markers share a row. A stays at column 2. B moves from column 1 to column 3. Final positions:

C0: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3. Both markers share a row. A stays at column 2. B moves between column 1 and column 3, from one to the other. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / static_placement / named_locations

Rule: `unordered_position_assignment`; source: `east_west__static_adjacent__named_locations__compact`

C1: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the forest, the lake. A is at the lake. B is at the forest. Final positions:

C2: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the forest, the lake. A is at the forest. B is at the lake. Final positions:

C0: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the forest, the lake. A and B occupy the forest and the lake, one marker at each. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / static_placement / numeric

Rule: `unordered_position_assignment`; source: `east_west__static_adjacent__numeric__compact`

C1: On this fixed map, north is up and east is right. Columns run west to east: 1, 2. Both markers share a row. A is at column 2. B is at column 1. Final positions:

C2: On this fixed map, north is up and east is right. Columns run west to east: 1, 2. Both markers share a row. A is at column 1. B is at column 2. Final positions:

C0: On this fixed map, north is up and east is right. Columns run west to east: 1, 2. Both markers share a row. A and B occupy column 1 and column 2, one marker at each. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / target_crosses / named_locations

Rule: `direction_unspecified_crossing`; source: `east_west__target_cross__named_locations__compact`

C1: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the school, the station, the garden. A moves from the school to the garden. B stays at the station. Final positions:

C2: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the school, the station, the garden. A moves from the garden to the school. B stays at the station. Final positions:

C0: On this fixed map, north is up and east is right. Along one line, the locations run west to east: the school, the station, the garden. A moves between the garden and the school, from one to the other. B stays at the station. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## east_west / target_crosses / numeric

Rule: `direction_unspecified_crossing`; source: `east_west__target_cross__numeric__compact`

C1: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3. Both markers share a row. A moves from column 1 to column 3. B stays at column 2. Final positions:

C2: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3. Both markers share a row. A moves from column 3 to column 1. B stays at column 2. Final positions:

C0: On this fixed map, north is up and east is right. Columns run west to east: 1, 2, 3. Both markers share a row. A moves between column 1 and column 3, from one to the other. B stays at column 2. Final positions:

T1: Marker A is east of marker B.

T2: Marker A is west of marker B.

## left_right / both_move / lettered_positions

Rule: `unordered_equal_comotion`; source: `both_preserve_left__ball_cone__lettered_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves from E to D. The ball icon moves from C to B.

C2: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves from C to B. The ball icon moves from E to D.

C0: Fixed screen slots, left to right: A, B, C, D, E. The cone icon and the ball icon each move one slot left and finish in different slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / named_positions

Rule: `unordered_equal_comotion`; source: `both_preserve_left__ball_cone__named_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves from far-right to inner-right. The ball icon moves from center to inner-left.

C2: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves from center to inner-left. The ball icon moves from far-right to inner-right.

C0: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon and the ball icon each move one slot left and finish in different slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / numbered_left_to_right

Rule: `unordered_equal_comotion`; source: `both_preserve_left__ball_cone__numbered_left_to_right__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves from 5 to 4. The ball icon moves from 3 to 2.

C2: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves from 3 to 2. The ball icon moves from 5 to 4.

C0: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon and the ball icon each move one slot left and finish in different slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / numbered_right_to_left

Rule: `unordered_equal_comotion`; source: `both_preserve_left__ball_cone__numbered_right_to_left__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves from 1 to 2. The ball icon moves from 3 to 4.

C2: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves from 3 to 4. The ball icon moves from 1 to 2.

C0: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon and the ball icon each move one slot left and finish in different slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / lettered_positions

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__lettered_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves from B to D. The ball icon moves from D to B.

C2: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves from D to B. The ball icon moves from B to D.

C0: Fixed screen slots, left to right: A, B, C, D, E. The cone icon and the ball icon begin in two different slots and exchange slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / named_positions

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__named_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves from inner-left to inner-right. The ball icon moves from inner-right to inner-left.

C2: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves from inner-right to inner-left. The ball icon moves from inner-left to inner-right.

C0: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon and the ball icon begin in two different slots and exchange slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / numbered_left_to_right

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__numbered_left_to_right__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves from 2 to 4. The ball icon moves from 4 to 2.

C2: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves from 4 to 2. The ball icon moves from 2 to 4.

C0: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon and the ball icon begin in two different slots and exchange slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / both_move / numbered_right_to_left

Rule: `unordered_exchange`; source: `both_reverse__ball_cone__numbered_right_to_left__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves from 4 to 2. The ball icon moves from 2 to 4.

C2: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves from 2 to 4. The ball icon moves from 4 to 2.

C0: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon and the ball icon begin in two different slots and exchange slots.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_moves_without_crossing / lettered_positions

Rule: `side_unspecified_no_crossing`; source: `no_cross_left__ball_cone__lettered_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: A, B, C, D, E. The cone icon stays at E. The ball icon moves from B to A.

C2: Fixed screen slots, left to right: A, B, C, D, E. The cone icon stays at A. The ball icon moves from E to D.

C0: Fixed screen slots, left to right: A, B, C, D, E. The cone icon stays in place. The ball icon moves one slot left without reaching or passing the cone icon.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_moves_without_crossing / named_positions

Rule: `side_unspecified_no_crossing`; source: `no_cross_left__ball_cone__named_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon stays at far-right. The ball icon moves from inner-left to far-left.

C2: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon stays at far-left. The ball icon moves from far-right to inner-right.

C0: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon stays in place. The ball icon moves one slot left without reaching or passing the cone icon.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_moves_without_crossing / numbered_left_to_right

Rule: `side_unspecified_no_crossing`; source: `no_cross_left__ball_cone__numbered_left_to_right__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon stays at 5. The ball icon moves from 2 to 1.

C2: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon stays at 1. The ball icon moves from 5 to 4.

C0: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon stays in place. The ball icon moves one slot left without reaching or passing the cone icon.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_moves_without_crossing / numbered_right_to_left

Rule: `side_unspecified_no_crossing`; source: `no_cross_left__ball_cone__numbered_right_to_left__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon stays at 1. The ball icon moves from 4 to 5.

C2: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon stays at 5. The ball icon moves from 1 to 2.

C0: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon stays in place. The ball icon moves one slot left without reaching or passing the cone icon.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / reference_crosses / lettered_positions

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__lettered_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves from A to E. The ball icon stays at C.

C2: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves from E to A. The ball icon stays at C.

C0: Fixed screen slots, left to right: A, B, C, D, E. The cone icon moves between A and E, from one to the other. The ball icon stays at C.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / reference_crosses / named_positions

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__named_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves from far-left to far-right. The ball icon stays at center.

C2: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves from far-right to far-left. The ball icon stays at center.

C0: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon moves between far-left and far-right, from one to the other. The ball icon stays at center.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / reference_crosses / numbered_left_to_right

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__numbered_left_to_right__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves from 1 to 5. The ball icon stays at 3.

C2: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves from 5 to 1. The ball icon stays at 3.

C0: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon moves between 1 and 5, from one to the other. The ball icon stays at 3.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / reference_crosses / numbered_right_to_left

Rule: `direction_unspecified_crossing`; source: `reference_cross_full__ball_cone__numbered_right_to_left__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves from 5 to 1. The ball icon stays at 3.

C2: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves from 1 to 5. The ball icon stays at 3.

C0: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon moves between 5 and 1, from one to the other. The ball icon stays at 3.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / static_placement / lettered_positions

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__lettered_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: A, B, C, D, E. The cone icon is at D. The ball icon is at B.

C2: Fixed screen slots, left to right: A, B, C, D, E. The cone icon is at B. The ball icon is at D.

C0: Fixed screen slots, left to right: A, B, C, D, E. The cone icon and the ball icon occupy B and D, one icon in each slot.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / static_placement / named_positions

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__named_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon is at inner-right. The ball icon is at inner-left.

C2: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon is at inner-left. The ball icon is at inner-right.

C0: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon and the ball icon occupy inner-left and inner-right, one icon in each slot.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / static_placement / numbered_left_to_right

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__numbered_left_to_right__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon is at 4. The ball icon is at 2.

C2: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon is at 2. The ball icon is at 4.

C0: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon and the ball icon occupy 2 and 4, one icon in each slot.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / static_placement / numbered_right_to_left

Rule: `unordered_position_assignment`; source: `static_inner__ball_cone__numbered_right_to_left__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon is at 2. The ball icon is at 4.

C2: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon is at 4. The ball icon is at 2.

C0: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon and the ball icon occupy 4 and 2, one icon in each slot.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_crosses / lettered_positions

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__lettered_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: A, B, C, D, E. The cone icon stays at C. The ball icon moves from E to A.

C2: Fixed screen slots, left to right: A, B, C, D, E. The cone icon stays at C. The ball icon moves from A to E.

C0: Fixed screen slots, left to right: A, B, C, D, E. The cone icon stays at C. The ball icon moves between A and E, from one to the other.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_crosses / named_positions

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__named_positions__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon stays at center. The ball icon moves from far-right to far-left.

C2: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon stays at center. The ball icon moves from far-left to far-right.

C0: Fixed screen slots, left to right: far-left, inner-left, center, inner-right, far-right. The cone icon stays at center. The ball icon moves between far-left and far-right, from one to the other.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_crosses / numbered_left_to_right

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__numbered_left_to_right__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon stays at 3. The ball icon moves from 5 to 1.

C2: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon stays at 3. The ball icon moves from 1 to 5.

C0: Fixed screen slots, left to right: 1, 2, 3, 4, 5. The cone icon stays at 3. The ball icon moves between 1 and 5, from one to the other.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## left_right / target_crosses / numbered_right_to_left

Rule: `direction_unspecified_crossing`; source: `target_cross_full__ball_cone__numbered_right_to_left__compact__reference_first__reference_first`

C1: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon stays at 3. The ball icon moves from 1 to 5.

C2: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon stays at 3. The ball icon moves from 5 to 1.

C0: Fixed screen slots, left to right: 5, 4, 3, 2, 1. The cone icon stays at 3. The ball icon moves between 5 and 1, from one to the other.

T1: The cone icon is to the right of the ball icon.

T2: The cone icon is to the left of the ball icon.

## front_behind / target_moves_without_crossing / nonnumeric

Rule: `reference_facing_unspecified`; source: `event__approach_without_crossing__nonnumeric__a_b__reference_first__template_specific__implicit_initial_facing__none`

C1: B faces away from A. A approaches B but stops before reaching B. B stays still without turning.

C2: B faces A. A approaches B but stops before reaching B. B stays still without turning.

C0: A approaches B but stops before reaching B. B stays still without turning.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / target_moves_without_crossing / numeric

Rule: `side_unspecified_no_crossing`; source: `event__approach_without_crossing__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B stays at 4. A moves from 1 to 3.

C2: B faces decreasing position numbers. B stays at 2. A moves from 5 to 3.

C0: B faces decreasing position numbers. B stays at a different position without turning. A moves without crossing B.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / both_move / nonnumeric

Rule: `reference_facing_unspecified`; source: `event__both_opposite_preserved__nonnumeric__a_b__reference_first__template_specific__implicit_initial_facing__none`

C1: B faces A. They move directly away from each other without turning.

C2: B faces away from A. They move directly away from each other without turning.

C0: A and B start at different positions and move directly away from each other without turning.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / both_move / numeric

Rule: `unordered_comotion`; source: `event__both_opposite_preserved__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B moves from 2 to 1 without turning. A moves from 4 to 5.

C2: B faces decreasing position numbers. B moves from 4 to 5 without turning. A moves from 2 to 1.

C0: B faces decreasing position numbers. B and A move without turning and finish at different positions.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / both_move / numeric

Rule: `unordered_exchange`; source: `event__both_reverse__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B moves from 4 to 2 without turning. A moves from 1 to 4.

C2: B faces decreasing position numbers. B moves from 2 to 4 without turning. A moves from 5 to 2.

C0: B faces decreasing position numbers. B and A start at different positions and pass each other without turning.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / reference_crosses / nonnumeric

Rule: `reference_facing_unspecified`; source: `event__reference_cross_forward__nonnumeric__a_b__reference_first__template_specific__implicit_initial_facing__none`

C1: B faces A, then walks straight past A without turning. A stays still.

C2: B faces away from A, then steps backward past A without turning. A stays still.

C0: B moves straight past A without turning. A stays still.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / reference_crosses / numeric

Rule: `direction_unspecified_crossing`; source: `event__reference_cross_forward__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B moves from 1 to 5 without turning. A stays at 3.

C2: B faces decreasing position numbers. B moves from 5 to 1 without turning. A stays at 3.

C0: B faces decreasing position numbers. B moves between 1 and 5, from one to the other without turning. A stays at 3.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / static_placement / nonnumeric

Rule: `facing_preserved_position_unspecified`; source: `event__static_adjacent__nonnumeric__a_b__reference_first__template_specific__away_from_anchor__door`

C1: B faces away from the door. A stands between B and the door.

C2: B faces away from the door. B stands between A and the door.

C0: B faces the door. A and B stand at different positions along the line to the door.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / static_placement / numeric

Rule: `position_unspecified`; source: `event__static_adjacent__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B stays at 2. A stays at 3.

C2: B faces decreasing position numbers. B stays at 4. A stays at 3.

C0: B faces decreasing position numbers. B and A stay at different positions.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / static_placement / numeric

Rule: `unordered_position_assignment`; source: `event__static_interior__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B stays at 4. A stays at 2.

C2: B faces decreasing position numbers. B stays at 2. A stays at 4.

C0: B faces decreasing position numbers. B and A stay at 2 and 4, one at each position.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / target_crosses / nonnumeric

Rule: `reference_facing_unspecified`; source: `event__target_cross_forward__nonnumeric__a_b__reference_first__template_specific__implicit_initial_facing__none`

C1: A starts behind B, then walks straight past B. B stays still without turning.

C2: A starts in front of B, then walks straight past B. B stays still without turning.

C0: A starts on one side of B, then walks straight past B. B stays still without turning.

T1: In front of B is A.

T2: Behind B is A.

## front_behind / target_crosses / numeric

Rule: `direction_unspecified_crossing`; source: `event__target_cross_forward__numeric__a_b__reference_first__reference_first__decreasing__none`

C1: B faces decreasing position numbers. B stays at 3. A moves from 1 to 5.

C2: B faces decreasing position numbers. B stays at 3. A moves from 5 to 1.

C0: B faces decreasing position numbers. B stays at 3. A moves between 1 and 5, from one to the other.

T1: In front of B is A.

T2: Behind B is A.
