# Hardware-Aware ML

This module captures semiconductor/edge-oriented AI engineering concerns:

- Precision trade-offs (FP32 vs reduced precision approximations)
- Memory constraints (artifact and runtime footprint)
- Latency vs throughput (single vs batch inference)
- Energy vs accuracy trade-offs (runtime-derived power proxy)

Run experiments:

```bash
python hardware_aware_ml/tradeoff_experiments.py
```

Outputs:
- `artifacts/hardware_tradeoffs.csv`
- `artifacts/hardware_tradeoffs_summary.json`
