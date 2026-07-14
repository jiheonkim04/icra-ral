# EAC-VLA Runtime Queue Check

Date: `2026-07-15`

Proposal hash: `A89ED48AE9FD4D26A8DA9E3E987FACDBBD9F861D070AE135372A092A44581E4E`

Final decision: `EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED`

- closed-loop experiment happened: `False`
- training happened: `False`
- validation search happened: `False`
- confirmatory-test tuning happened: `False`
- runtime device: `NVIDIA GeForce RTX 5080`
- policy class: `SmolVLAPolicy`
- full postprocessed chunk shape: `[50, 7]`
- full chunk finite: `True`
- select-action matches chunk[0]: `True`
- select-action/chunk[0] max abs diff: `0.0`
- queue owner present: `True`
- queue length before select: `0`
- queue length after select: `49`
- queue prefix checks passed: `True`
- max prefix abs diff: `0.0`
- max queue-pop abs diff: `0.0`

Prefix preservation checks:

```json
[
  {
    "action_values_modified": false,
    "commitment_length": 1,
    "expected_prefix_sha256": "cbde4b485fa6d2f5b6e19f6eb03b0ca7f325905f7883439ae63bcd06900458be",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "cbde4b485fa6d2f5b6e19f6eb03b0ca7f325905f7883439ae63bcd06900458be",
    "prefix_shape": [
      1,
      7
    ],
    "queue_pop_max_abs_diff": 0.0
  },
  {
    "action_values_modified": false,
    "commitment_length": 2,
    "expected_prefix_sha256": "3c0d0d5882691474a5f639cc639a8929f80c4eb161968482e383d51fb01e4eeb",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "3c0d0d5882691474a5f639cc639a8929f80c4eb161968482e383d51fb01e4eeb",
    "prefix_shape": [
      2,
      7
    ],
    "queue_pop_max_abs_diff": 0.0
  },
  {
    "action_values_modified": false,
    "commitment_length": 4,
    "expected_prefix_sha256": "d8260f2b9de9d43b11f681c6d8cbe87ca01b3b4ddf450f7db6d3106fc4b5d85a",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "d8260f2b9de9d43b11f681c6d8cbe87ca01b3b4ddf450f7db6d3106fc4b5d85a",
    "prefix_shape": [
      4,
      7
    ],
    "queue_pop_max_abs_diff": 0.0
  },
  {
    "action_values_modified": false,
    "commitment_length": 8,
    "expected_prefix_sha256": "05e862c61e56114046fb94aadcb3294772b48a3d5b6809ba480f6386a9a407d3",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "05e862c61e56114046fb94aadcb3294772b48a3d5b6809ba480f6386a9a407d3",
    "prefix_shape": [
      8,
      7
    ],
    "queue_pop_max_abs_diff": 0.0
  },
  {
    "action_values_modified": false,
    "commitment_length": 16,
    "expected_prefix_sha256": "c2184aaf9d210aa55977a377bfd20e64841b6d991916674232b2fd0382f8e002",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "c2184aaf9d210aa55977a377bfd20e64841b6d991916674232b2fd0382f8e002",
    "prefix_shape": [
      16,
      7
    ],
    "queue_pop_max_abs_diff": 0.0
  },
  {
    "action_values_modified": false,
    "commitment_length": 50,
    "expected_prefix_sha256": "e53c3b84489bdc60ff91bb9c950e91cde95b64606ac87b8639721ca535007285",
    "prefix_max_abs_diff": 0.0,
    "prefix_sha256": "e53c3b84489bdc60ff91bb9c950e91cde95b64606ac87b8639721ca535007285",
    "prefix_shape": [
      50,
      7
    ],
    "queue_pop_max_abs_diff": 0.0
  }
]
```

Hard stop reasons:
- none

Next step: Proceed to bounded validation search under the frozen EAC preregistration.
