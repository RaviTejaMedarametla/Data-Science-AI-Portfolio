# Embedded Digit Classifier

## Scenario
Digit recognition on memory-constrained edge hardware.

## System design decisions
- ONNX export with CPU-only runtime.
- Small batch-size inference path for deterministic latency.
- Accuracy-energy frontier tracked by `benchmarking/dashboard.py`.

## Deployment notes
- Reserve static memory footprint budgets.
- Track model binary size and peak resident memory for release gates.
