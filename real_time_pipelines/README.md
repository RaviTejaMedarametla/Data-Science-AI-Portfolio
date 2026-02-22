# Real-Time Pipelines

Unified batch + streaming inference orchestration.

Modes:

```bash
python real_time_pipelines/unified_pipeline.py --mode both
python real_time_pipelines/unified_pipeline.py --mode async_stream --jitter-ms 8 --queue-max 64 --workers 4
```

Artifacts:
- `artifacts/batch_inference.csv`
- `artifacts/stream_inference.jsonl`
- `artifacts/stream_inference_async.jsonl`
- `artifacts/streaming_metrics.json` (queue latency, drop rate, scaling parameters)
