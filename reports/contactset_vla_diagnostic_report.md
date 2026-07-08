# ContactSet-VLA Diagnostic Report

Bounded offline action-head diagnostic only. This is not standard LIBERO success, rollout evidence, or a paper-grade claim.

- decision: `kill`
- reason: full contact-set injection did not beat the active single-3D-point injection baseline
- training happened: `True`
- loss computed: `True`
- replay/control metric happened: `False`
- GPU/download/OpenVLA-OFT: `False` / `False` / `False`
- usable demos: `6`
- source/destination/support observable: `True` / `True` / `True`
- single-point action L2: `0.930495702`
- contact-set action L2: `1.105028754`
- contact-set beats single-point: `False`
- simple baselines matched contact-set: `True`

## Variants

| variant | action L2 | translation L2 | rotation L2 | gripper error | target-directed | source consistency | destination consistency | contact/place consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_geometry_injection | 0.851451 | 0.981954 | 0.118821 | 1.179519 | 0.410755 | 0.301977 | 0.410755 | 0.356366 |
| single_3d_point_injection | 0.930496 | 1.062792 | 0.139895 | 1.340929 | 0.425885 | 0.290882 | 0.425885 | 0.358383 |
| source_object_point_only | 1.262017 | 1.531321 | 0.504608 | 1.44136 | -0.107751 | 0.239498 | -0.107751 | 0.065873 |
| destination_placement_point_only | 0.86372 | 0.986866 | 0.172045 | 1.206707 | 0.142291 | 0.230184 | 0.142291 | 0.186237 |
| source_destination_two_point_injection | 1.360487 | 1.688784 | 0.368871 | 1.583944 | 0.289206 | 0.295563 | 0.289206 | 0.292384 |
| full_contact_set_injection | 1.105029 | 1.368344 | 0.220886 | 1.30362 | 0.334109 | 0.222354 | 0.334109 | 0.278232 |

## Cases

- file: `C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5`
  instruction: turn on the stove and put the moka pot on it
  source/destination/safety: `moka_pot_1` / `flat_stove_1_main` / `chefmate_8_frypan_1`
  geometry sources: `{'eef': 'obs/ee_pos', 'source': 'instruction_token_overlap', 'destination': 'xml_static_body_or_site', 'support': 'destination_static_or_trace_copy', 'safety': 'nearest_non_source_object_trace', 'normal': 'destination_plus_z_axis_proxy', 'qpos_offset': '1'}`
- file: `C:\assets\data\libero\libero_10\KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5`
  instruction: put the black bowl in the bottom drawer of the cabinet and close it
  source/destination/safety: `akita_black_bowl_1` / `white_cabinet_1_cabinet_bottom` / `wine_bottle_1`
  geometry sources: `{'eef': 'obs/ee_pos', 'source': 'instruction_token_overlap', 'destination': 'xml_static_body_or_site', 'support': 'destination_static_or_trace_copy', 'safety': 'nearest_non_source_object_trace', 'normal': 'destination_plus_z_axis_proxy', 'qpos_offset': '1'}`
- file: `C:\assets\data\libero\libero_10\KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5`
  instruction: put the yellow and white mug in the microwave and close it
  source/destination/safety: `white_yellow_mug_1` / `microwave_1_main` / `porcelain_mug_1`
  geometry sources: `{'eef': 'obs/ee_pos', 'source': 'instruction_token_overlap', 'destination': 'xml_static_body_or_site', 'support': 'destination_static_or_trace_copy', 'safety': 'nearest_non_source_object_trace', 'normal': 'destination_plus_z_axis_proxy', 'qpos_offset': '1'}`
- file: `C:\assets\data\libero\libero_10\KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5`
  instruction: put both moka pots on the stove
  source/destination/safety: `moka_pot_1` / `flat_stove_1_main` / `moka_pot_2`
  geometry sources: `{'eef': 'obs/ee_pos', 'source': 'instruction_token_overlap', 'destination': 'xml_static_body_or_site', 'support': 'destination_static_or_trace_copy', 'safety': 'nearest_non_source_object_trace', 'normal': 'destination_plus_z_axis_proxy', 'qpos_offset': '1'}`
- file: `C:\assets\data\libero\libero_10\LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5`
  instruction: put both the alphabet soup and the cream cheese box in the basket
  source/destination/safety: `cream_cheese_1` / `basket_1_default_site` / `basket_1`
  geometry sources: `{'eef': 'obs/ee_pos', 'source': 'instruction_token_overlap', 'destination': 'xml_static_body_or_site', 'support': 'destination_static_or_trace_copy', 'safety': 'nearest_non_source_object_trace', 'normal': 'destination_plus_z_axis_proxy', 'qpos_offset': '1'}`
- file: `C:\assets\data\libero\libero_10\LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5`
  instruction: put both the alphabet soup and the tomato sauce in the basket
  source/destination/safety: `tomato_sauce_1` / `basket_1_default_site` / `butter_1`
  geometry sources: `{'eef': 'obs/ee_pos', 'source': 'instruction_token_overlap', 'destination': 'xml_static_body_or_site', 'support': 'destination_static_or_trace_copy', 'safety': 'nearest_non_source_object_trace', 'normal': 'destination_plus_z_axis_proxy', 'qpos_offset': '1'}`
