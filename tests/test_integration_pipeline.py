from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


class TestIntegrationPipeline(unittest.TestCase):
    def test_full_pipeline_and_monitoring(self):
        subprocess.run(["python", "-m", "src.train"], check=True)
        subprocess.run(["python", "deployment/export_onnx.py"], check=True)
        subprocess.run(["python", "deployment/quantize_onnx.py"], check=True)
        subprocess.run(["python", "deployment/parity_check.py", "--batch-size", "128"], check=True)
        subprocess.run(["python", "benchmarking/statistical_benchmark.py", "--runs", "3", "--batch-size", "128"], check=True)
        subprocess.run(["python", "hardware_aware_ml/tradeoff_experiments.py"], check=True)
        subprocess.run(["python", "benchmarking/dashboard.py"], check=True)
        subprocess.run(["python", "real_time_pipelines/unified_pipeline.py", "--mode", "both"], check=True)
        subprocess.run(["python", "scripts/reproducibility_check.py"], check=True)

        expected = [
            "artifacts/best_model.joblib",
            "artifacts/model.onnx",
            "artifacts/model.int8.onnx",
            "artifacts/parity_report.json",
            "artifacts/statistical_comparisons.csv",
            "artifacts/streaming_metrics.json",
            "artifacts/reproducibility_report.json",
        ]
        for item in expected:
            self.assertTrue(Path(item).exists(), item)

    def test_drift_endpoint_trigger_logic(self):
        from fastapi.testclient import TestClient
        from src.api import app, _MONITORING

        with TestClient(app) as c:
            payload = {
                "student_country": "US",
                "days_on_platform": 100,
                "minutes_watched": 180.0,
                "courses_started": 3,
                "practice_exams_started": 2,
                "practice_exams_passed": 1,
                "minutes_spent_on_exams": 40.0,
            }
            for _ in range(60):
                r = c.post("/predict", json=payload)
                self.assertEqual(r.status_code, 200)

            drift = c.get("/monitoring/drift")
            self.assertEqual(drift.status_code, 200)
            body = drift.json()
            self.assertIn("should_retrain", body)
            self.assertGreaterEqual(body["samples_observed"], 60)


if __name__ == "__main__":
    unittest.main()
