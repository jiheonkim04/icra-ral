# CCIF-VLA Stage 0 Result

Final decision: `CCIF_STAGE_0_DESIGN_FAILURE`.

Rows: `4480 / 4480` model rows.
Unique observation rows: `640`.
Coarse-to-Control prior label: `coarse_to_control_continuous_proxy`.
Intent dimension: `31`; waypoints: `[9, 19, 34, 49]`.
Intent probe Huber: `0.40368824627421485`.
Task/phase mean intent Huber: `0.2552738050181288`.
Endpoint-only intent Huber: `0.42065329150014324`.
Base / prior / CCIF / no-intent / LoRA-proxy Huber: `0.0033407550543043956 / 0.010546990332363412 / 0.014732240917569977 / 0.014924647736380412 / 0.010476810826014676`.
CCIF minus prior relative / absolute Huber gain: `-0.3968194198836169 / -0.004185250585206565`.
CCIF minus ablation relative / absolute Huber gain: `0.012891883427266642 / 0.00019240681881043525`.
Identity max abs error: `0.0`.
Gradient finite/nonzero/ratio: `True / True / 19.910678643470327`.
Exceptions: `0`.

No simulator rollout, reward, success, done flag, validation search, or confirmatory identity was used.
