# Real Candidate-Generation Smoke Summary

This report describes the report-only synthesis step for the bounded real
candidate-generation smoke.

The summary reads:

```text
reports/real_candidate_generation_smoke_report.json
```

and writes ignored runtime reports:

```text
reports/real_candidate_generation_smoke_summary_report.json
reports/real_candidate_generation_smoke_summary_report.md
```

The summary performs no model import, model load, inference, training, rollout,
simulator execution, GPU job, download, OpenVLA-OFT execution, token access, or
paper-grade claim. Its only purpose is to decide whether the bounded smoke is
usable as engineering evidence for a later offline candidate-generation
comparison plan.
