from __future__ import annotations

import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app


class CsvMlApiTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("OPENAI_API_KEY", None)
        self.client = TestClient(app)

    def test_health_ready_and_metadata(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/ready").status_code, 200)
        metadata = self.client.get("/metadata").json()
        self.assertIn("profile_dataset", metadata["workflow_nodes"])
        self.assertIn("Deterministic fallback", metadata["execution_modes"])

    def test_workflow_returns_structured_response(self) -> None:
        response = self.client.post(
            "/workflow",
            json={
                "csv_text": Path("sample_data/patient_readmission_sample.csv").read_text(encoding="utf-8"),
                "target_column": "readmitted",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["execution_mode"], "Deterministic fallback")
        self.assertIn("request_id", payload)
        self.assertIn("profile_dataset", payload["completed_nodes"])
        self.assertIsNotNone(payload["readiness_review"])
        self.assertIn("model_recommendation", payload)

    def test_invalid_payload_is_rejected(self) -> None:
        self.assertEqual(self.client.post("/workflow", json={"csv_text": ""}).status_code, 422)
        self.assertEqual(self.client.post("/workflow", json={"csv_text": "a,b\n1,2", "target_column": "missing"}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
