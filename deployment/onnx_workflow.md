# ONNX Workflow

1. Train production model (with lineage):
```bash
python -m src.train
```
2. Export to ONNX from production artifact:
```bash
python deployment/export_onnx.py
```
3. Quantize for edge constraints:
```bash
python deployment/quantize_onnx.py
```
4. Validate parity (must pass):
```bash
python deployment/parity_check.py --abs-tol 0.04 --mean-tol 0.01
```
5. Run CPU inference:
```bash
python deployment/cpu_inference.py --backend sklearn
python deployment/cpu_inference.py --backend onnx
```

## Edge deployment notes

- Use ONNX Runtime CPU EP for x86 edge gateways.
- Validate quantized-vs-fp32 accuracy drift before release.
- Parity failure is treated as release-blocking in CI.
