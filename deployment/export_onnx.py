from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_config, load_dataset, split_data


def _build_initial_types(df):
    initial_types = []
    for col in df.columns:
        kind = df[col].dtype.kind
        if kind in {"i", "u", "b", "f"}:
            initial_types.append((col, FloatTensorType([None, 1])))
        else:
            initial_types.append((col, StringTensorType([None, 1])))
    return initial_types


def main() -> None:
    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]
    if not model_path.exists():
        raise FileNotFoundError("Train first: python -m src.train")

    model = joblib.load(model_path)
    df = load_dataset(config)
    _, X_test, _, _ = split_data(df, config)

    initial_types = _build_initial_types(X_test)
    onx = to_onnx(model, initial_types=initial_types, target_opset=15)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.onnx").write_bytes(onx.SerializeToString())
    (out_dir / "onnx_features.json").write_text(json.dumps(X_test.columns.tolist(), indent=2), encoding="utf-8")

    threshold_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["threshold_file"]
    threshold = float(threshold_path.read_text(encoding="utf-8").strip())
    (out_dir / "onnx_metadata.json").write_text(
        json.dumps({"threshold": threshold, "model_source": str(model_path)}, indent=2),
        encoding="utf-8",
    )
    print("Wrote artifacts/model.onnx, onnx_features.json, onnx_metadata.json")


if __name__ == "__main__":
    main()
