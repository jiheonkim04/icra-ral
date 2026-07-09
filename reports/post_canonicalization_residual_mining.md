# Post-Canonicalization Residual Mining

Final decision: `KILL_CANONICALIZATION_DOMINATED`

- training happened: `False`
- downloads/GPU/OpenVLA-OFT happened: `False` / `False` / `False`
- canonicalization residual size: `0.587661`
- canonicalization residual vs best non-oracle: `-0.013226`
- canonicalization clean-to-paraphrase delta: `-0.000748`
- largest residual subgroup: `{'name': 'gripper', 'value': 0.389255}`
- residual structured as method-worthy target/language failure: `False`
- standard LoRA/MLP already solves it within margin: `True`
- oracle/headroom exists: `False`

## Split Metrics

- mean-action: `{'clean': {'action_l2': 0.903848, 'translation_l2': 0.40424, 'rotation_l2': 0.074325, 'gripper_error': 0.75, 'gripper_accuracy': 0.75, 'sample_count': 60, 'per_dim_mae': [0.204955, 0.107845, 0.272665, 0.013426, 0.050147, 0.032784, 0.75]}, 'heldout_paraphrase': {'action_l2': 0.903848, 'translation_l2': 0.40424, 'rotation_l2': 0.074325, 'gripper_error': 0.75, 'gripper_accuracy': 0.75, 'sample_count': 360, 'per_dim_mae': [0.204955, 0.107845, 0.272665, 0.013426, 0.050147, 0.032784, 0.75]}, 'object_lexical': {'action_l2': 0.903848, 'translation_l2': 0.40424, 'rotation_l2': 0.074325, 'gripper_error': 0.75, 'gripper_accuracy': 0.75, 'sample_count': 60, 'per_dim_mae': [0.204955, 0.107845, 0.272665, 0.013426, 0.050147, 0.032784, 0.75]}}`
- MLP: `{'clean': {'action_l2': 0.619985, 'translation_l2': 0.352708, 'rotation_l2': 0.078822, 'gripper_error': 0.430371, 'gripper_accuracy': 0.866667, 'sample_count': 60, 'per_dim_mae': [0.179516, 0.119234, 0.215022, 0.020374, 0.052676, 0.038591, 0.430371]}, 'heldout_paraphrase': {'action_l2': 0.619985, 'translation_l2': 0.352708, 'rotation_l2': 0.078822, 'gripper_error': 0.430371, 'gripper_accuracy': 0.866667, 'sample_count': 360, 'per_dim_mae': [0.179516, 0.119234, 0.215022, 0.020374, 0.052676, 0.038591, 0.430371]}, 'object_lexical': {'action_l2': 0.619985, 'translation_l2': 0.352708, 'rotation_l2': 0.078822, 'gripper_error': 0.430371, 'gripper_accuracy': 0.866667, 'sample_count': 60, 'per_dim_mae': [0.179516, 0.119234, 0.215022, 0.020374, 0.052676, 0.038591, 0.430371]}}`
- ridge: `{'clean': {'action_l2': 0.584541, 'translation_l2': 0.346178, 'rotation_l2': 0.076878, 'gripper_error': 0.404054, 'gripper_accuracy': 0.933333, 'sample_count': 60, 'per_dim_mae': [0.180606, 0.126043, 0.203828, 0.017367, 0.051946, 0.037752, 0.404054]}}`
- standard LoRA: `{'clean': {'action_l2': 0.600887, 'translation_l2': 0.350954, 'rotation_l2': 0.078396, 'gripper_error': 0.402141, 'gripper_accuracy': 0.85, 'sample_count': 60, 'per_dim_mae': [0.178757, 0.13345, 0.203413, 0.019131, 0.051521, 0.038298, 0.402141]}, 'heldout_paraphrase': {'action_l2': 0.600887, 'translation_l2': 0.350954, 'rotation_l2': 0.078396, 'gripper_error': 0.402141, 'gripper_accuracy': 0.85, 'sample_count': 360, 'per_dim_mae': [0.178757, 0.13345, 0.203413, 0.019131, 0.051521, 0.038298, 0.402141]}, 'object_lexical': {'action_l2': 0.600887, 'translation_l2': 0.350954, 'rotation_l2': 0.078396, 'gripper_error': 0.402141, 'gripper_accuracy': 0.85, 'sample_count': 60, 'per_dim_mae': [0.178757, 0.13345, 0.203413, 0.019131, 0.051521, 0.038298, 0.402141]}}`
- canonicalization-only: `{'clean': {'action_l2': 0.588409, 'translation_l2': 0.339112, 'rotation_l2': 0.082532, 'gripper_error': 0.390587, 'gripper_accuracy': 0.85, 'sample_count': 60, 'per_dim_mae': [0.185176, 0.12119, 0.199332, 0.018726, 0.055345, 0.041226, 0.390587]}, 'heldout_paraphrase': {'action_l2': 0.587661, 'translation_l2': 0.338821, 'rotation_l2': 0.082683, 'gripper_error': 0.389255, 'gripper_accuracy': 0.85, 'sample_count': 360, 'per_dim_mae': [0.184885, 0.121454, 0.198728, 0.018692, 0.055534, 0.041326, 0.389255]}, 'object_lexical': {'action_l2': 0.587388, 'translation_l2': 0.338919, 'rotation_l2': 0.082509, 'gripper_error': 0.38967, 'gripper_accuracy': 0.85, 'sample_count': 60, 'per_dim_mae': [0.185073, 0.121093, 0.19913, 0.018692, 0.055389, 0.041164, 0.38967]}}`
- TG-7D failed reference: `{'clean': {'action_l2': 0.735738, 'translation_l2': 0.358227, 'rotation_l2': 0.087954, 'gripper_error': 0.572981, 'gripper_accuracy': 0.816667, 'sample_count': 60, 'per_dim_mae': [0.182124, 0.119239, 0.223456, 0.017877, 0.059756, 0.044794, 0.572981]}, 'heldout_paraphrase': {'action_l2': 0.740922, 'translation_l2': 0.358191, 'rotation_l2': 0.088013, 'gripper_error': 0.58099, 'gripper_accuracy': 0.816667, 'sample_count': 360, 'per_dim_mae': [0.18165, 0.119774, 0.223436, 0.017883, 0.059847, 0.044786, 0.58099]}, 'object_lexical': {'action_l2': 0.744749, 'translation_l2': 0.3577, 'rotation_l2': 0.088292, 'gripper_error': 0.584648, 'gripper_accuracy': 0.816667, 'sample_count': 60, 'per_dim_mae': [0.181575, 0.119274, 0.222921, 0.017822, 0.060051, 0.045009, 0.584648]}}`
- oracle target upper bound: `{'clean': {'action_l2': 0.724476, 'translation_l2': 0.399237, 'rotation_l2': 0.080637, 'gripper_error': 0.527888, 'gripper_accuracy': 0.766667, 'sample_count': 60, 'per_dim_mae': [0.1808, 0.190487, 0.22981, 0.018546, 0.053336, 0.039205, 0.527888]}, 'heldout_paraphrase': {'action_l2': 0.724674, 'translation_l2': 0.399472, 'rotation_l2': 0.080665, 'gripper_error': 0.52794, 'gripper_accuracy': 0.766667, 'sample_count': 360, 'per_dim_mae': [0.180846, 0.190674, 0.23007, 0.018571, 0.053341, 0.039211, 0.527941]}, 'object_lexical': {'action_l2': 0.725269, 'translation_l2': 0.399492, 'rotation_l2': 0.080648, 'gripper_error': 0.52857, 'gripper_accuracy': 0.766667, 'sample_count': 60, 'per_dim_mae': [0.180777, 0.19071, 0.229898, 0.018601, 0.053311, 0.039205, 0.52857]}}`

## Group Breakdowns

- clean instructions: `60` records
- paraphrase groups: `60` groups / `360` records
- object lexical groups: `10` groups / `60` records
- syntactic paraphrase groups: `9`
- counterfactual groups: `30` pairs / `30` records
- task-level groups: `{'open the middle drawer of the cabinet': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'open the top drawer and put the bowl inside': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'push the plate to the front of the stove': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'put the bowl on the plate': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'put the bowl on the stove': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'put the bowl on top of the cabinet': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'put the cream cheese in the bowl': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'put the wine bottle on the rack': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'put the wine bottle on top of the cabinet': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}, 'turn on the stove': {'heldout_records': 36, 'heldout_groups': 6, 'object_records': 6}}`

## Action-Dimension Groups

- translation: `{'canonicalization_l2': 0.338821, 'standard_lora_l2': 0.350954, 'tg7d_l2': 0.358191, 'canonical_minus_standard': -0.012133}`
- rotation: `{'canonicalization_l2': 0.082683, 'standard_lora_l2': 0.078396, 'tg7d_l2': 0.088013, 'canonical_minus_standard': 0.004287}`
- gripper: `{'canonicalization_error': 0.389255, 'standard_lora_error': 0.402141, 'tg7d_error': 0.58099, 'canonical_minus_standard': -0.012886}`

Evidence limit: per-example predictions were not archived, and this run did not retrain. Therefore per-paraphrase-group prediction metrics are not claimed.
