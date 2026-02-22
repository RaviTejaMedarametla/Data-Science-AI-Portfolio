from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_config, load_dataset, split_data
from deployment.cpu_inference import _to_onnx_inputs


def _extract_proba(outputs):
    for out in outputs:
        if isinstance(out, np.ndarray) and out.ndim >= 2 and out.shape[1] >= 2:
            return out[:, 1]
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return np.array([float(d.get(1, d.get("1", 0.0))) for d in out], dtype=float)
    raise RuntimeError("Could not find probability output in ONNX outputs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abs-tol", type=float, default=0.04)
    parser.add_argument("--mean-tol", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]
    onnx_path = Path("artifacts/model.onnx")

    model = joblib.load(model_path)
    df = load_dataset(config)
    _, X_test, _, _ = split_data(df, config)
    X = X_test.head(args.batch_size)

    sk = model.predict_proba(X)[:, 1]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onx_outputs = sess.run(None, _to_onnx_inputs(X))
    ox = _extract_proba(onx_outputs)

    abs_diff = np.abs(sk - ox)
    report = {
        "samples": int(len(sk)),
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "abs_tol": args.abs_tol,
        "mean_tol": args.mean_tol,
        "passed": bool(abs_diff.max() <= args.abs_tol and abs_diff.mean() <= args.mean_tol),
    }

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/parity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
