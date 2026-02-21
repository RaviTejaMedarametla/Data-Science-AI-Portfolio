from __future__ import annotations

import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd

from src.data import load_config, load_dataset, split_data


def run() -> None:
    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]
    if not model_path.exists():
        raise FileNotFoundError("Train the model first: python -m src.train")

    model = joblib.load(model_path)
    df = load_dataset(config)
    _, X_test, _, y_test = split_data(df, config)
    X_test = X_test.head(8).copy()

    scenarios = [
        {"precision": "fp32", "batch_size": 1, "energy_scale": 1.00},
        {"precision": "int8_surrogate", "batch_size": 8, "energy_scale": 0.52},
    ]

    rows = []
    for s in scenarios:
        batch_size = s["batch_size"]
        t0 = time.perf_counter()
        probs = []
        for i in range(0, len(X_test), batch_size):
            xb = X_test.iloc[i : i + batch_size]
            p = model.predict_proba(xb)[:, 1]
            if s["precision"] != "fp32":
                p = np.round(p, 2)
            probs.extend(p.tolist())
        elapsed = time.perf_counter() - t0

        preds = (np.array(probs) >= 0.5).astype(int)
        acc = float((preds == y_test.iloc[: len(preds)].to_numpy()).mean())
        throughput = float(len(preds) / elapsed) if elapsed > 0 else 0.0
        latency_ms = float((elapsed / len(preds)) * 1000) if len(preds) else 0.0

        artifact_bytes = model_path.stat().st_size
        memory_mb = float(artifact_bytes / (1024 * 1024) + batch_size * 0.002)
        energy_mj = float(elapsed * s["energy_scale"] * 1000)

        rows.append(
            {
                "precision": s["precision"],
                "batch_size": batch_size,
                "accuracy": acc,
                "latency_ms_per_sample": latency_ms,
                "throughput_samples_per_sec": throughput,
                "memory_mb": memory_mb,
                "energy_mj_proxy": energy_mj,
            }
        )

    out_df = pd.DataFrame(rows)
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "hardware_tradeoffs.csv", index=False)

    summary = {
        "best_accuracy": out_df.sort_values("accuracy", ascending=False).iloc[0].to_dict(),
        "best_energy": out_df.sort_values("energy_mj_proxy", ascending=True).iloc[0].to_dict(),
        "best_latency": out_df.sort_values("latency_ms_per_sample", ascending=True).iloc[0].to_dict(),
    }
    (out_dir / "hardware_tradeoffs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    run()
