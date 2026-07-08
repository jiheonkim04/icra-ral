# SafeTrace-VLA Related Work Matrix

| Work | What it gives | Gap SafeTrace would need to solve | STATE 1 implication |
| --- | --- | --- | --- |
| SafeManip | LTLf temporal safety templates, symbolic traces, DFA monitors over privileged simulator state. | It evaluates temporal safety but does not optimize VLA policies from monitor-derived preferences. | Best anchor, but local SafeManip rollout assets are not present. |
| LIBERO-Safety | Official safety benchmark with physical and semantic safety suites, public code, public dataset path, and multi-model evaluation context. | It benchmarks safety and provides safe demonstrations; preference training from temporal violations is not established. | Clear official path, but not downloaded/installed in this bounded run. |
| ForesightSafety-VLA | Process-level metrics such as cumulative safety cost, risk exposure time, and safe/unsafe success quadrants. | Diagnostic benchmark, not a preference-training method. | Useful metric vocabulary; no local code/data path found. |
| SafeVLA / PACS / safety filters | Safety alignment or constrained control/filtering. | Must beat safety-only and stop/filter behavior while retaining utility. | Mandatory baselines. |
| VLA-Corrector / A2C2 | Action-chunk correction and adaptation. | Corrects chunks but does not directly optimize temporal safety monitors as preferences. | Mandatory action-correction context and baseline risk. |
| OpenVLA / OpenVLA-OFT / SmolVLA / pi0 / GR00T | Model context for VLA evaluation. | SafeTrace must generalize beyond one local proxy/model. | No full VLA fine-tuning or OpenVLA-OFT in STATE 0-1. |

