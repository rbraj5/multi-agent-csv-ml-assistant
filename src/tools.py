from __future__ import annotations

import pandas as pd
from sklearn.utils.multiclass import type_of_target

from src.schemas import DataQualityAssessment, DatasetProfile, ModelRecommendation


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    missing = {
        column: int(count)
        for column, count in df.isna().sum().sort_values(ascending=False).items()
        if count > 0
    }

    return DatasetProfile(
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        duplicate_rows=int(df.duplicated().sum()),
        missing_values=missing,
    )


def assess_data_quality(df: pd.DataFrame, profile: DatasetProfile) -> DataQualityAssessment:
    recommendations: list[str] = []

    if profile.duplicate_rows:
        recommendations.append("Remove duplicate rows before model training.")

    for column, missing_count in profile.missing_values.items():
        if pd.api.types.is_numeric_dtype(df[column]):
            recommendations.append(f"Fill missing values in `{column}` with median or use an imputer.")
        else:
            recommendations.append(f"Fill missing values in `{column}` with the most frequent value.")

    high_cardinality = [
        col
        for col in df.select_dtypes(exclude="number").columns
        if df[col].nunique(dropna=True) > max(10, len(df) * 0.5)
    ]
    for column in high_cardinality:
        recommendations.append(f"Review `{column}` because it has many unique categories.")

    if not recommendations:
        recommendations.append("No major cleaning issues found. Validate outliers before training.")

    return DataQualityAssessment(
        recommendations=recommendations,
        high_cardinality_columns=high_cardinality,
    )


def recommend_models(df: pd.DataFrame, target_column: str | None) -> ModelRecommendation:
    if not target_column:
        return ModelRecommendation(
            target_column=None,
            target_type=None,
            candidate_feature_count=max(df.shape[1], 0),
            model_families=[
                "Linear Regression for numeric prediction",
                "Random Forest for robust tabular baselines",
                "Gradient Boosting for stronger tabular performance",
                "Logistic Regression for category prediction",
            ],
            metric_guidance="Select task-specific metrics after confirming the target column.",
        )

    target = df[target_column].dropna()
    try:
        target_kind = type_of_target(target)
    except ValueError:
        target_kind = "unknown"

    feature_count = max(df.shape[1] - 1, 0)
    if target_kind in {"binary", "multiclass"}:
        model_families = [
            "Logistic Regression as a baseline classifier",
            "Random Forest Classifier for non-linear relationships",
            "Gradient Boosting Classifier for stronger tabular performance",
        ]
        metric_guidance = "Use accuracy, F1-score, precision, recall, and ROC-AUC where suitable."
    elif target_kind in {"continuous", "continuous-multioutput"}:
        model_families = [
            "Linear Regression as a baseline regressor",
            "Random Forest Regressor for robust tabular modelling",
            "Gradient Boosting Regressor for improved predictive performance",
        ]
        metric_guidance = "Use MAE, RMSE, and R2 score."
    else:
        model_families = [
            "Clarify the target type before modelling",
            "Check whether the target should be cleaned, encoded, or transformed",
        ]
        metric_guidance = "Choose metrics after confirming the problem type."

    return ModelRecommendation(
        target_column=target_column,
        target_type=target_kind,
        candidate_feature_count=feature_count,
        model_families=model_families,
        metric_guidance=metric_guidance,
    )
