from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_config, load_dataset, split_data


def _to_onnx_inputs(X: pd.DataFrame) -> dict:
    feeds = {}
    for col in X.columns:
        kind = X[col].dtype.kind
        if kind in {"i", "u", "b", "f"}:
            feeds[col] = X[col].to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            feeds[col] = X[col].astype(str).to_numpy().reshape(-1, 1)
    return feeds


def run_sklearn(model_path: Path, X) -> tuple[float, int]:
    model = joblib.load(model_path)
    t0 = time.perf_counter()
    _ = model.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - t0
    return elapsed, len(X)


def run_onnx(onnx_path: Path, X) -> tuple[float, int]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = _to_onnx_inputs(X)
    t0 = time.perf_counter()
    _ = sess.run(None, feeds)
    elapsed = time.perf_counter() - t0
    return elapsed, len(X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "onnx"], default="sklearn")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]

    df = load_dataset(config)
    _, X_test, _, _ = split_data(df, config)
    X = X_test.head(args.batch_size)

    if args.backend == "sklearn":
        elapsed, n = run_sklearn(model_path, X)
    else:
        elapsed, n = run_onnx(Path("artifacts/model.onnx"), X)

    print(
        {
            "backend": args.backend,
            "samples": n,
            "latency_ms_per_sample": (elapsed / max(n, 1)) * 1000,
            "throughput_samples_per_sec": n / elapsed if elapsed > 0 else 0.0,
        }
    )


if __name__ == "__main__":
    main()
