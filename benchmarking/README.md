# Benchmarking

## Statistical benchmark framework

```bash
python benchmarking/statistical_benchmark.py --runs 10 --batch-size 256
```

Exports:
- `artifacts/stat_benchmark_runs.csv`
- `artifacts/stat_benchmark_summary.csv` (bootstrap 95% CI)
- `artifacts/statistical_comparisons.csv` (paired t-test + effect size)
- `artifacts/stat_benchmark_report.json`

## Dashboard

```bash
python benchmarking/dashboard.py
```

Outputs:
- `artifacts/benchmark_dashboard.csv`
- `artifacts/benchmark_summary.json`
- `artifacts/benchmark_tradeoff.png`
