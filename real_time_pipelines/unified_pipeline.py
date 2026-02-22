from __future__ import annotations

import argparse
import asyncio
import json
import random
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


async def async_stream_simulation(model, X: pd.DataFrame, out_path: Path, jitter_ms: float, queue_max: int, workers: int) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
    dropped = 0
    produced = 0
    consumed = 0
    latencies = []

    async def producer():
        nonlocal dropped, produced
        for _, row in X.iterrows():
            produced += 1
            evt = {"t_ingest": time.perf_counter(), "row": row.to_dict()}
            try:
                queue.put_nowait(evt)
            except asyncio.QueueFull:
                dropped += 1
            await asyncio.sleep(max(0.0, random.gauss(jitter_ms / 1000.0, jitter_ms / 4000.0)))

    async def consumer():
        nonlocal consumed
        while True:
            evt = await queue.get()
            if evt is None:
                queue.task_done()
                break
            row_df = pd.DataFrame([evt["row"]])
            score = float(model.predict_proba(row_df)[:, 1][0])
            consumed += 1
            latencies.append((time.perf_counter() - evt["t_ingest"]) * 1000)
            record = {
                "timestamp": time.time(),
                "queue_latency_ms": latencies[-1],
                "score": score,
                "prediction": int(score >= 0.5),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            queue.task_done()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    producer_task = asyncio.create_task(producer())
    consumer_tasks = [asyncio.create_task(consumer()) for _ in range(workers)]

    await producer_task
    await queue.join()
    for _ in range(workers):
        await queue.put(None)
    await asyncio.gather(*consumer_tasks)

    metrics = {
        "produced": produced,
        "consumed": consumed,
        "dropped": dropped,
        "drop_rate": dropped / produced if produced else 0.0,
        "queue_latency_ms_p50": float(pd.Series(latencies).quantile(0.5)) if latencies else 0.0,
        "queue_latency_ms_p95": float(pd.Series(latencies).quantile(0.95)) if latencies else 0.0,
        "workers": workers,
        "queue_max": queue_max,
        "jitter_ms": jitter_ms,
    }
    Path("artifacts/streaming_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "stream", "async_stream", "both"], default="both")
    parser.add_argument("--jitter-ms", type=float, default=5.0)
    parser.add_argument("--queue-max", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
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

    if args.mode in {"async_stream", "both"}:
        asyncio.run(
            async_stream_simulation(
                model,
                X_slice,
                Path("artifacts/stream_inference_async.jsonl"),
                jitter_ms=args.jitter_ms,
                queue_max=args.queue_max,
                workers=args.workers,
            )
        )


if __name__ == "__main__":
    main()
