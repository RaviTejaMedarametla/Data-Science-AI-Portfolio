from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from skl2onnx import to_onnx
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data import load_config, load_dataset, split_data


def main() -> None:
    config = load_config("config.yaml")
    df = load_dataset(config)
    X_train, X_test, y_train, _ = split_data(df, config)

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    X_train_num = X_train[num_cols].fillna(X_train[num_cols].median())
    X_test_num = X_test[num_cols].fillna(X_train[num_cols].median())

    model = LogisticRegression(max_iter=500)
    model.fit(X_train_num, y_train)

    onx = to_onnx(model, X_test_num.head(1).to_numpy(dtype=np.float32), target_opset=15)

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.onnx").write_bytes(onx.SerializeToString())
    (out_dir / "onnx_features.json").write_text(json.dumps(num_cols, indent=2), encoding="utf-8")
    print("Wrote artifacts/model.onnx and artifacts/onnx_features.json")


if __name__ == "__main__":
    main()
