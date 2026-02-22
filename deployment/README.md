# Deployment

Deployment-oriented assets for edge and CPU environments:

- `export_onnx.py` - exports the **exact production serving model artifact** (`artifacts/best_model.joblib`) to ONNX.
- `quantize_onnx.py` - post-training dynamic int8 quantization for ONNX Runtime.
- `cpu_inference.py` - CPU inference latency + throughput checks for sklearn / ONNX.
- `parity_check.py` - tolerance-based sklearn vs ONNX agreement validation (CI-gated).
- `onnx_workflow.md` - deployment workflow and edge packaging notes.
