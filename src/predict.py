from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data import load_config
from src.features import FeatureConfig, add_engineered_features


def build_feature_config(config: dict) -> FeatureConfig:
    return FeatureConfig(
        epsilon=float(config["features"]["epsilon"]),
        minutes_watched_weight=float(config["features"]["engagement"]["minutes_watched_weight"]),
        days_on_platform_weight=float(config["features"]["engagement"]["days_on_platform_weight"]),
        courses_started_weight=float(config["features"]["engagement"]["courses_started_weight"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run purchase predictions on new CSV data.")
    parser.add_argument("--input", required=True, help="Path to input CSV with raw feature columns.")
    parser.add_argument("--output", default="artifacts/predictions.csv", help="Where to save predictions CSV.")
    args = parser.parse_args()

    config = load_config()
    artifacts_dir = Path(config["artifacts"]["model_dir"])

    model = joblib.load(artifacts_dir / config["artifacts"]["model_file"])
    threshold = float((artifacts_dir / config["artifacts"]["threshold_file"]).read_text(encoding="utf-8").strip())

    raw_df = pd.read_csv(args.input)
    feat_df = add_engineered_features(raw_df, build_feature_config(config))

    probs = model.predict_proba(feat_df)[:, 1]
    preds = (probs >= threshold).astype(int)

    out = raw_df.copy()
    out["predicted_purchase_probability"] = probs
    out["predicted_purchase"] = preds
    out.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
