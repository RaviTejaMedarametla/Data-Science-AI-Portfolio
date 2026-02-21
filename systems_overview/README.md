# Systems Overview

This section reframes the portfolio as a **research-grade AI systems program** centered on four integrated projects:

1. **Neural Network project** → represented by the student purchase modeling stack (`src/train.py`) and extended through hardware-aware experiments.
2. **Digit Classifier project** → represented as an embedded/edge classification case study under `case_studies/embedded_digit_classifier.md`.
3. **NBA Pipeline project** → represented by streaming and batch system design under `real_time_pipelines/` and `case_studies/sports_analytics_streaming.md`.
4. **Healthcare AI project** → represented by latency-sensitive edge deployment and monitoring in `case_studies/healthcare_edge_ai.md`.

## End-to-end lifecycle

The unified architecture follows:

`data -> preprocessing -> model -> optimization -> deployment -> monitoring`

See:
- `systems_overview/workflow.md`
- `real_time_pipelines/unified_pipeline.py`
- `deployment/onnx_workflow.md`
- `benchmarking/dashboard.py`

## Reproducibility stack

- Configured experiments in `config/experiments.yaml`
- Dataset manifests in `config/datasets.yaml`
- Artifact conventions in `config/artifacts.yaml`
- CI workflow in `.github/workflows/ci.yml`
