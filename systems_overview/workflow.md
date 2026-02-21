# Unified AI + Hardware Workflow

## Pipeline stages

1. **Data**
   - Student, segmentation, and SQL-derived analytics datasets.
   - Versioned via `config/datasets.yaml`.
2. **Preprocessing**
   - Imputation, outlier clipping, scaling, one-hot encoding (`src/train.py`).
3. **Modeling**
   - Baseline classical models + calibrated thresholds (`src/train.py`).
   - Hardware-aware surrogate experiments (`hardware_aware_ml/tradeoff_experiments.py`).
4. **Optimization**
   - Precision and batch-size experiments.
   - Latency-throughput-energy benchmarking (`benchmarking/dashboard.py`).
5. **Deployment**
   - FastAPI online serving (`src/api.py`).
   - ONNX and CPU runners (`deployment/`).
6. **Monitoring**
   - Drift and retraining trigger endpoints (`src/api.py`).
   - Artifact logging and benchmark snapshots.

## Trade-off simulation design

- **Precision trade-off**: Float32 vs quantized/low-precision surrogate metrics.
- **Memory constraints**: Model artifact size + inference batch memory footprint.
- **Latency vs throughput**: Single request and micro-batch timing.
- **Energy vs accuracy**: Power proxy from runtime * utilization.

All outputs are stored under `artifacts/` for reproducible review.
