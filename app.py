from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.utils.multiclass import type_of_target


SAMPLE_PATH = Path("sample_data/patient_readmission_sample.csv")


@dataclass
class AgentResult:
    name: str
    output: str


def eda_analyst(df: pd.DataFrame) -> AgentResult:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    missing = df.isna().sum().sort_values(ascending=False)
    duplicate_count = int(df.duplicated().sum())

    lines = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols) or 'none'})",
        f"Categorical columns: {len(categorical_cols)} ({', '.join(categorical_cols) or 'none'})",
        f"Duplicate rows: {duplicate_count}",
    ]

    missing_items = missing[missing > 0]
    if missing_items.empty:
        lines.append("Missing values: none detected")
    else:
        lines.append("Missing values:")
        lines.extend([f"- {col}: {count}" for col, count in missing_items.items()])

    return AgentResult("EDA Analyst", "\n".join(lines))


def cleaning_advisor(df: pd.DataFrame) -> AgentResult:
    suggestions: list[str] = []

    if df.duplicated().any():
        suggestions.append("Remove duplicate rows before model training.")

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            suggestions.append(f"Fill missing values in `{col}` with median or use an imputer.")
        else:
            suggestions.append(f"Fill missing values in `{col}` with the most frequent value.")

    high_cardinality = [
        col
        for col in df.select_dtypes(exclude="number").columns
        if df[col].nunique(dropna=True) > max(10, len(df) * 0.5)
    ]
    for col in high_cardinality:
        suggestions.append(f"Review `{col}` because it has many unique categories.")

    if not suggestions:
        suggestions.append("No major cleaning issues found. Validate outliers before training.")

    return AgentResult("Data Cleaning Advisor", "\n".join(f"- {item}" for item in suggestions))


def model_recommender(df: pd.DataFrame, target_col: str | None) -> AgentResult:
    if not target_col:
        return AgentResult(
            "Model Recommender",
            "- Select a target column to receive task-specific model suggestions.\n"
            "- For numeric prediction, start with Linear Regression, Random Forest Regressor, and Gradient Boosting.\n"
            "- For category prediction, start with Logistic Regression, Random Forest, and Gradient Boosting.",
        )

    target = df[target_col].dropna()
    try:
        target_kind = type_of_target(target)
    except ValueError:
        target_kind = "unknown"

    feature_count = max(df.shape[1] - 1, 0)
    if target_kind in {"binary", "multiclass"}:
        recommendations = [
            "Logistic Regression as a baseline classifier",
            "Random Forest Classifier for non-linear relationships",
            "Gradient Boosting Classifier for stronger tabular performance",
        ]
        metric = "Use accuracy, F1-score, precision, recall, and ROC-AUC where suitable."
    elif target_kind in {"continuous", "continuous-multioutput"}:
        recommendations = [
            "Linear Regression as a baseline regressor",
            "Random Forest Regressor for robust tabular modelling",
            "Gradient Boosting Regressor for improved predictive performance",
        ]
        metric = "Use MAE, RMSE, and R2 score."
    else:
        recommendations = [
            "Clarify the target type before modelling",
            "Check whether the target should be cleaned, encoded, or transformed",
        ]
        metric = "Choose metrics after confirming the problem type."

    lines = [
        f"Target column: `{target_col}`",
        f"Detected target type: `{target_kind}`",
        f"Candidate feature columns: {feature_count}",
        "Suggested models:",
        *[f"- {item}" for item in recommendations],
        metric,
    ]
    return AgentResult("Model Recommender", "\n".join(lines))


def report_writer(results: list[AgentResult]) -> AgentResult:
    sections = ["# Dataset Review Report", ""]
    for result in results:
        sections.extend([f"## {result.name}", result.output, ""])
    sections.append("## Next Steps")
    sections.append("- Validate assumptions with domain knowledge.")
    sections.append("- Split data into train/test sets before model evaluation.")
    sections.append("- Track preprocessing and model metrics in a reproducible notebook.")
    return AgentResult("Report Writer", "\n".join(sections))


def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(SAMPLE_PATH)


def main() -> None:
    st.set_page_config(page_title="Multi-Agent CSV ML Assistant", layout="wide")
    st.title("Multi-Agent CSV ML Assistant")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    df = load_data(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    target_col = st.selectbox("Optional target column", [""] + df.columns.tolist())
    target_value = target_col or None

    results = [
        eda_analyst(df),
        cleaning_advisor(df),
        model_recommender(df, target_value),
    ]
    final_report = report_writer(results)

    for result in results:
        with st.expander(result.name, expanded=True):
            st.markdown(result.output)

    st.subheader(final_report.name)
    st.markdown(final_report.output)
    st.download_button("Download report", final_report.output, "dataset_review_report.md")


if __name__ == "__main__":
    main()
