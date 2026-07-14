# FANG-VLA Validation Search

Date: `2026-07-14`

Proposal hash: `6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B`

Final decision: `VALIDATION_SEARCH_STOP_DESIGN_FAILURE`

- closed-loop experiment happened: `False`
- training happened: `True`
- train records: `6568`
- validation records: `4233`
- total configurations: `6`
- selected config: `fang_c01`
- selected score: `0.8804892856627703`
- selected mean delta L2: `0.015608571469783783`
- selected gate activation fraction: `1.0`
- selected action validity: `1.0`

Hard stop reasons:
- `selected config gate activates almost everywhere`

Configurations:

- `fang_c01` score `0.8804892856627703` delta `0.015608571469783783` gate `1.0` validity `1.0`
- `fang_c02` score `0.8590849364176393` delta `0.032732050865888596` gate `1.0` validity `1.0`
- `fang_c03` score `0.8300352483987808` delta `0.05597180128097534` gate `1.0` validity `1.0`
- `fang_c04` score `0.8794510309584439` delta `0.016439175233244896` gate `1.0` validity `1.0`
- `fang_c05` score `0.8601378291845322` delta `0.03188973665237427` gate `1.0` validity `1.0`
- `fang_c06` score `0.8364324798807502` delta `0.05085401609539986` gate `1.0` validity `1.0`

Next step: Do not roll out FANG; classify the design failure.
