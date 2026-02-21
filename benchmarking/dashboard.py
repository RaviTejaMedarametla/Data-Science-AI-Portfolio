from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    source = Path("artifacts/hardware_tradeoffs.csv")
    if not source.exists():
        raise FileNotFoundError("Run python hardware_aware_ml/tradeoff_experiments.py first")

    df = pd.read_csv(source)
    df["composite_score"] = (
        0.4 * df["accuracy"]
        + 0.2 * (1 / (1 + df["latency_ms_per_sample"]))
        + 0.2 * (1 / (1 + df["memory_mb"]))
        + 0.2 * (1 / (1 + df["energy_mj_proxy"]))
    )

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "benchmark_dashboard.csv", index=False)

    summary = {
        "best_composite": df.sort_values("composite_score", ascending=False).iloc[0].to_dict(),
        "lowest_latency": df.sort_values("latency_ms_per_sample", ascending=True).iloc[0].to_dict(),
        "lowest_memory": df.sort_values("memory_mb", ascending=True).iloc[0].to_dict(),
        "lowest_energy": df.sort_values("energy_mj_proxy", ascending=True).iloc[0].to_dict(),
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    plt.scatter(df["latency_ms_per_sample"], df["accuracy"], s=120)
    for _, r in df.iterrows():
        plt.annotate(f"{r['precision']}-b{int(r['batch_size'])}", (r["latency_ms_per_sample"], r["accuracy"]))
    plt.xlabel("Latency (ms/sample)")
    plt.ylabel("Accuracy")
    plt.title("System Trade-off: Latency vs Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_tradeoff.png", dpi=180)


if __name__ == "__main__":
    main()
