# Healthcare Edge AI

## Scenario
Low-latency risk scoring at point-of-care where cloud connectivity may be intermittent.

## System design
- Local CPU inference via ONNX Runtime.
- Conservative threshold tuning for recall-sensitive workflows.
- Drift monitoring and retraining triggers using API observability endpoints.

## Trade-offs
- Prioritize reliability and explainability over model complexity.
- Use quantized deployment when power envelope is constrained.
