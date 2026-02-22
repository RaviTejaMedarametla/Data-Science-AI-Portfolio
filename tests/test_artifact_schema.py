from __future__ import annotations

import json
import unittest
from pathlib import Path


class TestArtifactSchema(unittest.TestCase):
    def test_lineage_schema(self):
        data = json.loads(Path("artifacts/lineage.json").read_text(encoding="utf-8"))
        for key in ["run_id", "dataset", "config", "model", "threshold"]:
            self.assertIn(key, data)
        self.assertIn("sha256", data["dataset"])
        self.assertIn("sha256", data["config"])
        self.assertIn("sha256", data["model"])

    def test_parity_schema(self):
        data = json.loads(Path("artifacts/parity_report.json").read_text(encoding="utf-8"))
        for key in ["samples", "max_abs_diff", "mean_abs_diff", "passed"]:
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
