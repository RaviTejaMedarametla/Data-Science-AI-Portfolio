# AI Systems Engineering Portfolio (Hardware-Aware + Deployment-Ready)

This repository is organized as a **research-grade AI + hardware systems portfolio** designed for semiconductor AI, edge computing, and production ML roles.

## Identity and focus

I position this portfolio as an **AI Systems Engineer** with emphasis on:
- hardware-aware ML optimization,
- reproducible experiment design,
- deployment-oriented model engineering,
- monitoring and lifecycle operations.

## Portfolio architecture

- `systems_overview/` - integrated architecture and workflow narrative.
- `hardware_aware_ml/` - precision/memory/latency/energy trade-off experiments.
- `real_time_pipelines/` - batch + streaming inference workflows.
- `deployment/` - ONNX export and CPU inference pipelines for edge settings.
- `benchmarking/` - unified dashboard for latency, memory, energy, and accuracy.
- `case_studies/` - healthcare edge AI, sports streaming analytics, embedded classification.
- `config/` - experiments, datasets, and artifact tracking manifests.
- `src/` - production ML/API components preserved from the original portfolio.

## Existing projects integrated into one system

- **Neural network project analogue:** student predictive modeling lifecycle in `src/train.py`.
- **Digit classifier project analogue:** embedded classification deployment path in `case_studies/embedded_digit_classifier.md` + `deployment/`.
- **NBA pipeline project analogue:** streaming and warehouse-driven analytics patterns in `real_time_pipelines/` and `case_studies/sports_analytics_streaming.md`.
- **Healthcare AI project analogue:** edge-safe monitoring and latency-sensitive deployment in `case_studies/healthcare_edge_ai.md`.

## Guided walkthrough

### 1) Reproducible setup
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train baseline model and generate monitoring artifacts
```bash
python -m src.train
```

### 3) Run unified batch + streaming pipeline
```bash
python real_time_pipelines/unified_pipeline.py --mode both
```

### 4) Run hardware-aware trade-off simulation
```bash
python benchmarking/statistical_benchmark.py --runs 10 --batch-size 256
python hardware_aware_ml/tradeoff_experiments.py
```

### 5) Build benchmarking dashboard and trade-off plot
```bash
python benchmarking/dashboard.py
```

### 6) Export ONNX and test CPU inference
```bash
python deployment/export_onnx.py
python deployment/quantize_onnx.py
python deployment/parity_check.py --abs-tol 0.04 --mean-tol 0.01
python deployment/cpu_inference.py --backend sklearn
python deployment/cpu_inference.py --backend onnx
```

## Unified end-to-end workflow

`data -> preprocessing -> model -> optimization -> deployment -> monitoring`

- Data and feature engineering: `src/data.py`, `src/features.py`
- Model selection/calibration: `src/train.py`
- Online serving + drift checks: `src/api.py`
- Hardware optimization experiments: `hardware_aware_ml/tradeoff_experiments.py`
- Deployment/inference pathways: `deployment/`
- Benchmark dashboard: `benchmarking/dashboard.py`

## Reproducibility and research credibility

- **Artifact lineage:** run_id + dataset/config/model hashes in `artifacts/lineage.json`
- **Centralized experiment tracking:** `config/experiments.yaml`
- **Dataset versioning manifest:** `config/datasets.yaml`
- **Artifact logging registry:** `config/artifacts.yaml`
- **CI workflow:** `.github/workflows/ci.yml`

## CI + Docker

- CI validates compile/train/API smoke path.
- Dockerfile provides reproducible runtime packaging.
