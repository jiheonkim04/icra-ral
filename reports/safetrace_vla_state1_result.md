# SafeTrace-VLA STATE 1 Diagnostic

Bounded feasibility smoke only. This is local LIBERO proxy evidence, not paper-grade safety benchmark evidence.

- final output: `KILL`
- reason: safety-only/risk-only scoring matches the SafeTrace preference objective on generated pairs
- source used: `local standard LIBERO HDF5 proxy, not official safety benchmark`
- usable demos: `8`
- real temporal metric produced: `True`
- temporal violation rate by step: `0.519444`
- risk exposure time: `0.519444`
- cumulative safety cost: `939.0`
- safe success / unsafe success: `None` / `None`
- valid / nontrivial preference pairs: `800` / `10`
- generic DPO proxy accuracy: `1.0`
- SafeTrace proxy accuracy: `1.0`
- safety-only matches SafeTrace: `True`
- generic DPO matches SafeTrace: `True`
- download/GPU/OpenVLA-OFT happened: `False` / `False` / `False`
- training happened / loss computed: `False` / `True`

## Source Audit

| source | local | temporal properties | rollout local | notes |
| --- | ---: | ---: | ---: | --- |
| SafeManip | False | True | False | Best conceptual match, but no local SafeManip rollout JSON or RoboCasa stack was present. |
| LIBERO-Safety | False | True | False | Official path is clear but not local; this bounded run did not download 19.1 GB or install assets. |
| ForesightSafety-VLA | False | True | False | Defines CC/RET-style process metrics but no local source path was available. |
| RoboTwin/RoboCasa safety tasks | False | True | False | Not locally installed for a real safety benchmark run. |
| Local standard LIBERO HDF5 | True | False | False | Used only as a non-paper-grade local proxy for temporal monitor plumbing. |
| Local LIBERO source checkout | True | False | False | Useful for task metadata but not a safety benchmark by itself. |

## Case Metrics

| file | risk exposure | cost | property counts |
| --- | ---: | ---: | --- |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 | 0.727778 | 165.0 | `{'no_transport_with_open_gripper': 33, 'grasp_instability': 84, 'release_before_destination': 2, 'containment_before_release': 46, 'unsafe_contact_before_safe_phase': 0}` |
| KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5 | 0.244444 | 71.0 | `{'no_transport_with_open_gripper': 32, 'grasp_instability': 0, 'release_before_destination': 0, 'containment_before_release': 27, 'object_dropped_before_placement': 12, 'unsafe_contact_before_safe_phase': 0}` |
| KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5 | 0.172222 | 60.0 | `{'no_transport_with_open_gripper': 31, 'grasp_instability': 0, 'release_before_destination': 0, 'containment_before_release': 29, 'object_dropped_before_placement': 0, 'unsafe_contact_before_safe_phase': 0}` |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5 | 0.788889 | 185.0 | `{'no_transport_with_open_gripper': 0, 'grasp_instability': 78, 'release_before_destination': 1, 'containment_before_release': 51, 'unsafe_contact_before_safe_phase': 55}` |
| LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5 | 0.583333 | 138.0 | `{'no_transport_with_open_gripper': 32, 'grasp_instability': 59, 'release_before_destination': 2, 'containment_before_release': 45, 'object_dropped_before_placement': 0, 'unsafe_contact_before_safe_phase': 0}` |
| LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5 | 0.211111 | 40.0 | `{'no_transport_with_open_gripper': 0, 'grasp_instability': 0, 'release_before_destination': 2, 'containment_before_release': 38, 'unsafe_contact_before_safe_phase': 0}` |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5 | 0.611111 | 128.0 | `{'no_transport_with_open_gripper': 17, 'grasp_instability': 68, 'release_before_destination': 2, 'containment_before_release': 41, 'unsafe_contact_before_safe_phase': 0}` |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5 | 0.816667 | 152.0 | `{'no_transport_with_open_gripper': 4, 'grasp_instability': 5, 'release_before_destination': 2, 'containment_before_release': 54, 'object_dropped_before_placement': 0, 'unsafe_contact_before_safe_phase': 87}` |
