from __future__ import annotations

import pandas as pd
from sklearn.utils.multiclass import type_of_target

from src.schemas import DataQualityAssessment, DatasetProfile, ModelRecommendation, ReadinessReview


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
        unique_counts={column: int(df[column].nunique(dropna=True)) for column in df.columns},
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


def review_readiness(
    df: pd.DataFrame,
    profile: DatasetProfile,
    quality: DataQualityAssessment,
    target_column: str | None,
) -> ReadinessReview:
    risks: list[str] = []
    actions: list[str] = []
    leakage_warnings: list[str] = []
    class_balance: dict[str, int] = {}

    if profile.missing_values:
        risks.append("Dataset contains missing values.")
        actions.append("Resolve missing-value handling before model training.")
    if profile.duplicate_rows:
        risks.append("Dataset contains duplicate rows.")
        actions.append("Remove or justify duplicate records before evaluation.")
    if quality.high_cardinality_columns:
        risks.append("Dataset contains high-cardinality categorical columns.")
        actions.append("Review encoding strategy for high-cardinality fields.")

    if target_column:
        target = df[target_column].dropna()
        class_balance = {str(label): int(count) for label, count in target.value_counts().items()}
        if len(class_balance) == 2:
            minority = min(class_balance.values())
            majority = max(class_balance.values())
            if minority / max(majority, 1) < 0.25:
                risks.append("Binary target appears imbalanced.")
                actions.append("Use stratified splits and imbalance-aware metrics.")

        target_terms = {"target", "label", "outcome", "readmitted", "churned", "defaulted"}
        for column in df.columns:
            if column == target_column:
                continue
            lowered = column.lower()
            if any(term in lowered for term in target_terms):
                leakage_warnings.append(f"`{column}` may encode target-like information.")
        if leakage_warnings:
            risks.append("Potential target leakage indicators were found.")
            actions.append("Review target-like feature names before modelling.")
    else:
        risks.append("No target column selected.")
        actions.append("Select a target column before task-specific modelling.")

    status = "Ready for baseline modelling" if not risks else "Needs review before modelling"
    return ReadinessReview(
        status=status,
        risks=risks,
        required_actions=actions,
        class_balance=class_balance,
        leakage_warnings=leakage_warnings,
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
