# Data Science + AI Portfolio (Production-Finetuned)

This repository contains:
- Student purchase prediction pipeline (`src/train.py`, `src/api.py`, `src/predict.py`)
- LangChain QA service (`src/qa_api.py`)
- Segmentation analytics (`src/segmentation_analysis.py`)
- SQL KPI analytics (`student_kpi_queries.sql`) and warehouse star schema (`warehouse_star_schema.sql`)

## Reproducible setup

```bash
pip install -r requirements.txt
python -m src.train
```

## Run APIs

Predictive API:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

QA API:
```bash
uvicorn src.qa_api:app --host 0.0.0.0 --port 8000
```

## Monitoring and drift controls

Prediction API endpoints:
- `GET /health`
- `GET /monitoring/drift`
- `GET /monitoring/retraining_trigger`
- `POST /predict`
- `POST /batch_predict`

Key operational behavior:
- Inference logging to JSONL (`monitoring.prediction_log_file` in `config.yaml`)
- Drift detection using z-score shifts versus training baseline
- Retraining trigger recommendation based on minimum sample gate + drift / class-rate shift

## CI + Docker

- CI workflow: `.github/workflows/ci.yml` (pins Python and runs warning-clean checks)
- Docker runtime: `Dockerfile`
