# Hardware-Aware ML

This module now consumes repeated benchmark runs and measured telemetry to produce realistic system trade-offs:

- FP32 sklearn, FP32 ONNX, INT8 ONNX scenarios
- Latency, throughput, accuracy, memory, CPU utilization
- Optional RAPL energy counter integration and proxy-vs-measured comparison

Run sequence:

```bash
python benchmarking/statistical_benchmark.py --runs 10 --batch-size 256
python hardware_aware_ml/tradeoff_experiments.py
```

Outputs:
- `artifacts/hardware_tradeoffs.csv`
- `artifacts/hardware_tradeoffs_summary.json`
