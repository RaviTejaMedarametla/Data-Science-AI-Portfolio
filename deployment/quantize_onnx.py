from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    source = Path("artifacts/model.onnx")
    target = Path("artifacts/model.int8.onnx")
    if not source.exists():
        raise FileNotFoundError("Missing artifacts/model.onnx. Run export first.")

    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
