from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import joblib
import pandas as pd

from src.data import load_config, load_dataset, split_data


def batch_inference(model, X: pd.DataFrame, out_path: Path) -> None:
    probs = model.predict_proba(X)[:, 1]
    out = X.copy()
    out["score"] = probs
    out["prediction"] = (probs >= 0.5).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def stream_inference(model, X: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in X.iterrows():
            x1 = row.to_frame().T
            score = float(model.predict_proba(x1)[:, 1][0])
            record = {
                "timestamp": time.time(),
                "score": score,
                "prediction": int(score >= 0.5),
            }
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "stream", "both"], default="both")
    args = parser.parse_args()

    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]
    model = joblib.load(model_path)

    df = load_dataset(config)
    _, X_test, _, _ = split_data(df, config)
    X_slice = X_test.head(64).copy()

    if args.mode in {"batch", "both"}:
        batch_inference(model, X_slice, Path("artifacts/batch_inference.csv"))

    if args.mode in {"stream", "both"}:
        stream_inference(model, X_slice, Path("artifacts/stream_inference.jsonl"))


if __name__ == "__main__":
    main()
