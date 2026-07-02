from __future__ import annotations

import os
import unittest

import pandas as pd

from src.graph import run_dataset_review


class DatasetReviewWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("OPENAI_API_KEY", None)

    def test_sample_dataset_runs_in_deterministic_mode(self) -> None:
        df = pd.read_csv("sample_data/patient_readmission_sample.csv")
        state = run_dataset_review(df, None)

        self.assertEqual(state["execution_mode"], "Deterministic fallback")
        self.assertEqual(
            state["completed_nodes"],
            ["profile_dataset", "assess_data_quality", "recommend_models", "synthesize_report"],
        )
        self.assertIn("Dataset Review Report", state["report"])

    def test_binary_target_returns_classification_guidance(self) -> None:
        df = pd.read_csv("sample_data/patient_readmission_sample.csv")
        state = run_dataset_review(df, "readmitted")

        self.assertIn("Classifier", " ".join(state["recommendation"].model_families))


if __name__ == "__main__":
    unittest.main()
