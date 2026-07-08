# SafeTrace-VLA Task Definition

Long title: Temporal Safety Preference Optimization for Vision-Language-Action Robot Policies.

Core claim to test: temporal safety monitors can convert rollout traces into preference signals that reduce temporal violations while preserving task utility.

Minimum novelty requirement: not only a benchmark wrapper and not only a runtime filter. The method must introduce temporal credit assignment or utility-preserving preference construction that beats safety-only, stop-on-risk, clipping, reward penalty, and generic DPO/preference baselines.

STATE 0-1 scope: source availability, observable temporal metric, preference-pair headroom, and tiny baseline comparison only. No OpenVLA-OFT, full fine-tuning, GPU, large download, or paper-grade claim.

