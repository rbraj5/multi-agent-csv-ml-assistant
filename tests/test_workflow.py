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
            ["profile_dataset", "assess_data_quality", "review_readiness", "recommend_models", "synthesize_report"],
        )
        self.assertIn("Dataset Review Report", state["report"])

    def test_binary_target_returns_classification_guidance(self) -> None:
        df = pd.read_csv("sample_data/patient_readmission_sample.csv")
        state = run_dataset_review(df, "readmitted")

        self.assertIn("Classifier", " ".join(state["recommendation"].model_families))
        self.assertIn("readiness", state)
        self.assertTrue(state["readiness"].class_balance)

    def test_clean_dataset_without_target_skips_readiness_gate(self) -> None:
        df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "income": [40000, 50000, 60000],
                "score": [0.1, 0.2, 0.3],
            }
        )
        state = run_dataset_review(df, None)

        self.assertEqual(
            state["completed_nodes"],
            ["profile_dataset", "assess_data_quality", "recommend_models", "synthesize_report"],
        )
        self.assertNotIn("readiness", state)

    def test_potential_leakage_is_flagged(self) -> None:
        df = pd.DataFrame(
            {
                "feature": [1, 2, 3, 4],
                "previous_outcome_flag": [0, 1, 0, 1],
                "target": [0, 1, 0, 1],
            }
        )
        state = run_dataset_review(df, "target")

        self.assertTrue(state["readiness"].leakage_warnings)


if __name__ == "__main__":
    unittest.main()
