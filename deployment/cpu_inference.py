from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import joblib
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_config, load_dataset, split_data


def run_sklearn(model_path: Path, X) -> tuple[float, int]:
    model = joblib.load(model_path)
    t0 = time.perf_counter()
    _ = model.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - t0
    return elapsed, len(X)


def run_onnx(onnx_path: Path, X) -> tuple[float, int]:
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    x_arr = X.to_numpy(dtype=np.float32)
    t0 = time.perf_counter()
    _ = sess.run(None, {input_name: x_arr})
    elapsed = time.perf_counter() - t0
    return elapsed, len(X)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "onnx"], default="sklearn")
    args = parser.parse_args()

    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]

    df = load_dataset(config)
    _, X_test, _, _ = split_data(df, config)

    if args.backend == "sklearn":
        X = X_test.head(128)
        elapsed, n = run_sklearn(model_path, X)
    else:
        cols = json.loads(Path("artifacts/onnx_features.json").read_text(encoding="utf-8"))
        X = X_test[cols].fillna(X_test[cols].median()).head(128)
        elapsed, n = run_onnx(Path("artifacts/model.onnx"), X)

    print({"backend": args.backend, "samples": n, "latency_ms": (elapsed / n) * 1000})


if __name__ == "__main__":
    main()
