from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def run() -> None:
    runs_path = Path("artifacts/stat_benchmark_runs.csv")
    if not runs_path.exists():
        raise FileNotFoundError("Run python benchmarking/statistical_benchmark.py first")

    df = pd.read_csv(runs_path)
    grouped = (
        df.groupby("scenario", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            latency_ms_per_sample=("latency_ms_per_sample", "mean"),
            throughput_samples_per_sec=("throughput_samples_per_sec", "mean"),
            memory_mb=("memory_mb_measured", "mean"),
            energy_uj_measured=("energy_uj_measured", "mean"),
            cpu_percent_avg=("cpu_percent_avg", "mean"),
        )
        .fillna(0.0)
    )

    grouped["energy_mj_proxy"] = grouped["latency_ms_per_sample"] * grouped["cpu_percent_avg"]
    grouped["energy_delta_vs_measured"] = np.where(
        grouped["energy_uj_measured"] > 0,
        grouped["energy_mj_proxy"] - (grouped["energy_uj_measured"] / 1000),
        np.nan,
    )

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_dir / "hardware_tradeoffs.csv", index=False)

    summary = {
        "best_accuracy": grouped.sort_values("accuracy", ascending=False).iloc[0].to_dict(),
        "best_energy_proxy": grouped.sort_values("energy_mj_proxy", ascending=True).iloc[0].to_dict(),
        "best_latency": grouped.sort_values("latency_ms_per_sample", ascending=True).iloc[0].to_dict(),
    }
    (out_dir / "hardware_tradeoffs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    run()
