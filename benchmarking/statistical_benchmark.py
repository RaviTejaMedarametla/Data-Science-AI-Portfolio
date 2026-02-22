from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deployment.cpu_inference import _to_onnx_inputs
from src.data import load_config, load_dataset, split_data

try:
    import psutil
except ImportError:  # optional
    psutil = None


def _rapl_uj() -> float | None:
    rapl = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    if rapl.exists():
        try:
            return float(rapl.read_text(encoding="utf-8").strip())
        except OSError:
            return None
    return None


def _bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(sample.mean()))
    alpha = 1 - confidence
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _run_once(session, X, threshold: float, y_true) -> dict:
    mem_before = psutil.Process(os.getpid()).memory_info().rss if psutil else None
    cpu_before = psutil.cpu_percent(interval=None) if psutil else None
    e0 = _rapl_uj()

    t0 = time.perf_counter()
    if isinstance(session, ort.InferenceSession):
        outputs = session.run(None, _to_onnx_inputs(X))
        probs = None
        for out in outputs:
            if isinstance(out, np.ndarray) and out.ndim >= 2 and out.shape[1] >= 2:
                probs = out[:, 1]
                break
            if isinstance(out, list) and out and isinstance(out[0], dict):
                probs = np.array([float(d.get(1, d.get("1", 0.0))) for d in out], dtype=float)
                break
        if probs is None:
            raise RuntimeError("No probability output from ONNX")
    else:
        probs = session.predict_proba(X)[:, 1]
    elapsed = time.perf_counter() - t0

    e1 = _rapl_uj()
    mem_after = psutil.Process(os.getpid()).memory_info().rss if psutil else None
    cpu_after = psutil.cpu_percent(interval=None) if psutil else None

    preds = (probs >= threshold).astype(int)
    return {
        "latency_ms_per_sample": (elapsed / max(len(X), 1)) * 1000,
        "throughput_samples_per_sec": len(X) / elapsed if elapsed > 0 else 0.0,
        "accuracy": float((preds == y_true).mean()),
        "memory_mb_measured": (mem_after - mem_before) / (1024 * 1024) if mem_before is not None and mem_after is not None else None,
        "cpu_percent_avg": np.mean([v for v in [cpu_before, cpu_after] if v is not None]) if psutil else None,
        "energy_uj_measured": (e1 - e0) if e0 is not None and e1 is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config("config.yaml")
    model_path = Path(config["artifacts"]["model_dir"]) / config["artifacts"]["model_file"]
    threshold = float((Path(config["artifacts"]["model_dir"]) / config["artifacts"]["threshold_file"]).read_text(encoding="utf-8"))

    df = load_dataset(config)
    _, X_test, _, y_test = split_data(df, config)
    X = X_test.head(args.batch_size)
    y = y_test.head(args.batch_size).to_numpy()

    scenarios = {
        "sklearn_fp32": joblib.load(model_path),
        "onnx_fp32": ort.InferenceSession("artifacts/model.onnx", providers=["CPUExecutionProvider"]),
        "onnx_int8": ort.InferenceSession("artifacts/model.int8.onnx", providers=["CPUExecutionProvider"]),
    }

    rows = []
    for scenario, sess in scenarios.items():
        for run_idx in range(args.runs):
            res = _run_once(sess, X, threshold, y)
            res["scenario"] = scenario
            res["run"] = run_idx
            rows.append(res)

    out = pd.DataFrame(rows)
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "stat_benchmark_runs.csv", index=False)

    summary = []
    for scenario, g in out.groupby("scenario"):
        record = {"scenario": scenario}
        for metric in ["latency_ms_per_sample", "throughput_samples_per_sec", "accuracy"]:
            vals = g[metric].to_numpy(dtype=float)
            lo, hi = _bootstrap_ci(vals)
            record[f"{metric}_mean"] = float(vals.mean())
            record[f"{metric}_ci95_low"] = lo
            record[f"{metric}_ci95_high"] = hi
        summary.append(record)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "stat_benchmark_summary.csv", index=False)

    compare_rows = []
    baseline = out[out["scenario"] == "onnx_fp32"]
    for scenario in ["onnx_int8", "sklearn_fp32"]:
        cur = out[out["scenario"] == scenario]
        for metric in ["latency_ms_per_sample", "throughput_samples_per_sec", "accuracy"]:
            a = baseline[metric].to_numpy(dtype=float)
            b = cur[metric].to_numpy(dtype=float)
            t_stat, p_val = stats.ttest_rel(a, b)
            diff = b - a
            effect = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
            compare_rows.append(
                {
                    "baseline": "onnx_fp32",
                    "scenario": scenario,
                    "metric": metric,
                    "mean_diff": float(diff.mean()),
                    "p_value": float(p_val),
                    "cohens_d_paired": effect,
                }
            )

    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out_dir / "statistical_comparisons.csv", index=False)

    report = {
        "runs": args.runs,
        "batch_size": args.batch_size,
        "hardware_telemetry": {
            "psutil_enabled": bool(psutil is not None),
            "rapl_enabled": _rapl_uj() is not None,
        },
    }
    (out_dir / "stat_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
