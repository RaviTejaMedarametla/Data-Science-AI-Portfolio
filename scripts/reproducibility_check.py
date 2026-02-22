from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lineage_path = Path("artifacts/lineage.json")
    if not lineage_path.exists():
        raise FileNotFoundError("Missing artifacts/lineage.json. Run training first.")

    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": (Path(lineage["dataset"]["path"]), lineage["dataset"]["sha256"]),
        "config": (Path(lineage["config"]["path"]), lineage["config"]["sha256"]),
        "model": (Path(lineage["model"]["path"]), lineage["model"]["sha256"]),
        "threshold": (Path(lineage["threshold"]["path"]), lineage["threshold"]["sha256"]),
    }

    report = {"run_id": lineage.get("run_id"), "checks": {}}
    all_passed = True
    for name, (path, expected) in checks.items():
        actual = sha256(path)
        passed = actual == expected
        all_passed &= passed
        report["checks"][name] = {
            "path": str(path),
            "expected": expected,
            "actual": actual,
            "passed": passed,
        }

    report["passed"] = all_passed
    Path("artifacts/reproducibility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
