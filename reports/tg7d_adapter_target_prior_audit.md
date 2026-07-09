# TG-7D Adapter Target Prior Audit

Target priors are derived from instruction text plus visible object-candidate names parsed from HDF5 model XML.

- target prior from instruction and visible names only: `True`
- instruction/action link without inference leakage: `True`
- counterfactual generated without oracle eval labels: `True`

## open the middle drawer of the cabinet

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'open the middle drawer of the cabinet', 'candidate_count': 46, 'selected_candidates': ['wooden cabinet cabinet middle', 'wooden cabinet middle', 'wooden cabinet wooden cabinet middle', 'wooden cabinet middle level'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## open the top drawer and put the bowl inside

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'open the top drawer and put the bowl inside', 'candidate_count': 46, 'selected_candidates': ['akita black bowl', 'akita black bowl g'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## push the plate to the front of the stove

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'push the plate to the front of the stove', 'candidate_count': 46, 'selected_candidates': ['stove front', 'flat stove burner plate', 'plate', 'stove'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## put the bowl on the plate

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'put the bowl on the plate', 'candidate_count': 46, 'selected_candidates': ['plate', 'plate g', 'plate model', 'akita black bowl'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## put the bowl on the stove

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'put the bowl on the stove', 'candidate_count': 46, 'selected_candidates': ['stove', 'flat stove', 'stove front', 'akita black bowl'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## put the bowl on top of the cabinet

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'put the bowl on top of the cabinet', 'candidate_count': 46, 'selected_candidates': ['cabinet', 'wooden cabinet', 'wooden cabinet cabinet top', 'wooden cabinet top'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## put the cream cheese in the bowl

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'put the cream cheese in the bowl', 'candidate_count': 46, 'selected_candidates': ['cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'akita black bowl'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## put the wine bottle on the rack

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'put the wine bottle on the rack', 'candidate_count': 46, 'selected_candidates': ['wine bottle', 'wine rack', 'wine rack top', 'wine bottle g'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## put the wine bottle on top of the cabinet

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'put the wine bottle on top of the cabinet', 'candidate_count': 46, 'selected_candidates': ['wine bottle', 'wine bottle g', 'wine bottle wine bottle red', 'cabinet'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`

## turn on the stove

- visible object candidate count: `46`
- visible object candidate sample: `['akita black bowl', 'akita black bowl g', 'cabinet', 'cream cheese', 'cream cheese cream cheese', 'cream cheese g', 'flat stove', 'flat stove burner', 'flat stove burner plate', 'flat stove burnerplate', 'flat stove cook', 'flat stove g']`
- selected target prior: `{'instruction': 'turn on the stove', 'candidate_count': 46, 'selected_candidates': ['stove', 'flat stove', 'stove front', 'flat stove burner'], 'source': 'instruction_text_plus_hdf5_model_xml_visible_candidate_names', 'uses_bddl_target_labels': False, 'uses_eval_labels': False, 'uses_task_ids': False, 'uses_filenames_as_inference_labels': False, 'uses_reward_or_success_labels': False, 'uses_future_actions': False}`
