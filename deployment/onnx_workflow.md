# ONNX Workflow

1. Train model:
```bash
python -m src.train
```
2. Export to ONNX:
```bash
python deployment/export_onnx.py
```
3. Run CPU inference:
```bash
python deployment/cpu_inference.py --backend sklearn
python deployment/cpu_inference.py --backend onnx
```

## Edge deployment notes

- Use ONNX Runtime CPU EP for x86 edge gateways.
- Quantize model when constrained by memory/energy budgets.
- Package with FastAPI or gRPC front-end depending on latency SLA.
