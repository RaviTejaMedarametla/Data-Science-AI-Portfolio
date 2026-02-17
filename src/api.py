from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import load_config
from src.features import FeatureConfig, add_engineered_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("purchase_api")


class PredictRequest(BaseModel):
    student_country: str = Field(..., min_length=2, max_length=64)
    days_on_platform: int = Field(..., ge=0)
    minutes_watched: float = Field(..., ge=0)
    courses_started: int = Field(..., ge=0)
    practice_exams_started: int = Field(..., ge=0)
    practice_exams_passed: int = Field(..., ge=0)
    minutes_spent_on_exams: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    predicted_purchase_probability: float
    predicted_purchase: int


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    ready: bool
    predictor_loaded: bool
    drift_baseline_loaded: bool


class DriftStatusResponse(BaseModel):
    samples_observed: int
    drift_score_max_abs_z: float
    drifted_features: List[str]
    predicted_positive_rate: float
    training_positive_rate: float
    should_retrain: bool
    reason: str
    recommended_action: str


_CONFIG: dict | None = None
_MODEL = None
_THRESHOLD: float | None = None
_FEATURE_CFG: FeatureConfig | None = None
_DRIFT_BASELINE: dict | None = None
_PREDICTION_LOG_PATH: Path | None = None

_MONITORING: Dict[str, object] = {
    "samples": 0,
    "feature_sums": {},
    "predicted_positive": 0,
}
_LOCK = Lock()


def _build_feature_config(config: dict) -> FeatureConfig:
    return FeatureConfig(
        epsilon=float(config["features"]["epsilon"]),
        minutes_watched_weight=float(config["features"]["engagement"]["minutes_watched_weight"]),
        days_on_platform_weight=float(config["features"]["engagement"]["days_on_platform_weight"]),
        courses_started_weight=float(config["features"]["engagement"]["courses_started_weight"]),
    )


def _compute_drift_status() -> DriftStatusResponse:
    if _DRIFT_BASELINE is None:
        return DriftStatusResponse(
            samples_observed=0,
            drift_score_max_abs_z=0.0,
            drifted_features=[],
            predicted_positive_rate=0.0,
            training_positive_rate=0.0,
            should_retrain=False,
            reason="baseline_not_loaded",
            recommended_action="run_training_to_generate_baseline",
        )

    monitoring_cfg = _CONFIG.get("monitoring", {}) if _CONFIG else {}
    min_samples = int(monitoring_cfg.get("drift_min_samples", 100))
    z_threshold = float(monitoring_cfg.get("drift_zscore_threshold", 3.0))
    min_drifted_features = int(monitoring_cfg.get("drift_min_features", 2))
    class_rate_shift = float(monitoring_cfg.get("class_rate_shift_threshold", 0.10))

    baseline_stats = _DRIFT_BASELINE.get("numeric_feature_stats", {})
    training_positive_rate = float(_DRIFT_BASELINE.get("training_positive_rate", 0.0))

    with _LOCK:
        samples = int(_MONITORING["samples"])
        feature_sums = dict(_MONITORING["feature_sums"])
        predicted_positive = int(_MONITORING["predicted_positive"])

    if samples == 0:
        return DriftStatusResponse(
            samples_observed=0,
            drift_score_max_abs_z=0.0,
            drifted_features=[],
            predicted_positive_rate=0.0,
            training_positive_rate=training_positive_rate,
            should_retrain=False,
            reason="no_predictions_observed",
            recommended_action="collect_inference_samples",
        )

    drifted_features: List[str] = []
    max_abs_z = 0.0
    for feature, s in feature_sums.items():
        if feature not in baseline_stats:
            continue
        current_mean = float(s) / samples
        base_mean = float(baseline_stats[feature]["mean"])
        base_std = max(float(baseline_stats[feature]["std"]), 1e-6)
        abs_z = abs((current_mean - base_mean) / base_std)
        max_abs_z = max(max_abs_z, abs_z)
        if abs_z >= z_threshold:
            drifted_features.append(feature)

    predicted_positive_rate = predicted_positive / samples
    class_shift = abs(predicted_positive_rate - training_positive_rate)

    should_retrain = False
    reason = "below_threshold"
    recommended_action = "continue_monitoring"

    if samples < min_samples:
        reason = "insufficient_samples"
        recommended_action = "collect_more_samples"
    else:
        if len(drifted_features) >= min_drifted_features:
            should_retrain = True
            reason = "feature_distribution_drift"
            recommended_action = "trigger_retraining_pipeline"
        elif class_shift >= class_rate_shift:
            should_retrain = True
            reason = "prediction_rate_shift"
            recommended_action = "trigger_retraining_pipeline"

    return DriftStatusResponse(
        samples_observed=samples,
        drift_score_max_abs_z=float(max_abs_z),
        drifted_features=sorted(drifted_features),
        predicted_positive_rate=float(predicted_positive_rate),
        training_positive_rate=float(training_positive_rate),
        should_retrain=should_retrain,
        reason=reason,
        recommended_action=recommended_action,
    )


def _log_prediction_row(raw_record: dict, probability: float, prediction: int) -> None:
    if _PREDICTION_LOG_PATH is None:
        return

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": _THRESHOLD,
        "predicted_purchase_probability": probability,
        "predicted_purchase": prediction,
        "features": raw_record,
    }

    try:
        _PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PREDICTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as exc:
        logger.warning("Could not write prediction log entry: %s", exc)


def load_artifacts() -> None:
    global _CONFIG, _MODEL, _THRESHOLD, _FEATURE_CFG, _DRIFT_BASELINE, _PREDICTION_LOG_PATH

    _CONFIG = load_config()
    artifacts_dir = Path(_CONFIG["artifacts"]["model_dir"])
    model_path = artifacts_dir / _CONFIG["artifacts"]["model_file"]
    threshold_path = artifacts_dir / _CONFIG["artifacts"]["threshold_file"]
    baseline_path = artifacts_dir / _CONFIG["artifacts"].get("drift_baseline_file", "drift_baseline.json")

    if not model_path.exists() or not threshold_path.exists():
        msg = (
            "Model artifacts are missing. Run `python -m src.train` first "
            "to generate model and threshold files."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    _MODEL = joblib.load(model_path)
    _THRESHOLD = float(threshold_path.read_text(encoding="utf-8").strip())
    _FEATURE_CFG = _build_feature_config(_CONFIG)

    if baseline_path.exists():
        _DRIFT_BASELINE = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        _DRIFT_BASELINE = None
        logger.warning("Drift baseline file not found at %s", baseline_path)

    prediction_log_file = _CONFIG.get("monitoring", {}).get("prediction_log_file", "artifacts/prediction_log.jsonl")
    _PREDICTION_LOG_PATH = Path(prediction_log_file)

    logger.info("Artifacts loaded successfully from %s", artifacts_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Student Purchase Prediction API", version="1.2.0", lifespan=lifespan)


def _predict_records(records: List[PredictRequest]) -> List[PredictResponse]:
    if _MODEL is None or _THRESHOLD is None or _FEATURE_CFG is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    data = pd.DataFrame([r.model_dump() for r in records])

    invalid_rows = data["practice_exams_passed"] > data["practice_exams_started"]
    if invalid_rows.any():
        raise HTTPException(
            status_code=422,
            detail="practice_exams_passed cannot exceed practice_exams_started.",
        )

    feat_df = add_engineered_features(data, _FEATURE_CFG)
    probs = _MODEL.predict_proba(feat_df)[:, 1]
    preds = (probs >= _THRESHOLD).astype(int)

    numeric_cols = feat_df.select_dtypes(include="number").columns.tolist()

    with _LOCK:
        for col in numeric_cols:
            _MONITORING["feature_sums"][col] = _MONITORING["feature_sums"].get(col, 0.0) + float(feat_df[col].sum())
        _MONITORING["samples"] = int(_MONITORING["samples"]) + len(feat_df)
        _MONITORING["predicted_positive"] = int(_MONITORING["predicted_positive"]) + int(preds.sum())

    outputs: List[PredictResponse] = []
    for raw, prob, pred in zip(data.to_dict(orient="records"), probs, preds):
        _log_prediction_row(raw, float(prob), int(pred))
        outputs.append(
            PredictResponse(
                predicted_purchase_probability=float(prob),
                predicted_purchase=int(pred),
            )
        )

    return outputs


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        ready=bool(_MODEL is not None and _THRESHOLD is not None and _FEATURE_CFG is not None),
        predictor_loaded=bool(_MODEL is not None),
        drift_baseline_loaded=bool(_DRIFT_BASELINE is not None),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    logger.info("Received /predict request")
    result = _predict_records([payload])[0]
    logger.info("/predict success | probability=%.4f | class=%d", result.predicted_purchase_probability, result.predicted_purchase)
    return result


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(payload: BatchPredictRequest) -> BatchPredictResponse:
    logger.info("Received /batch_predict request | records=%d", len(payload.records))
    results = _predict_records(payload.records)
    logger.info("/batch_predict success | records=%d", len(results))
    return BatchPredictResponse(predictions=results)


@app.get("/monitoring/drift", response_model=DriftStatusResponse)
def monitoring_drift() -> DriftStatusResponse:
    return _compute_drift_status()


@app.get("/monitoring/retraining_trigger", response_model=DriftStatusResponse)
def monitoring_retraining_trigger() -> DriftStatusResponse:
    return _compute_drift_status()
