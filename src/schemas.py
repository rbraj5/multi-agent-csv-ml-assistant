from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    row_count: int
    column_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    duplicate_rows: int
    missing_values: dict[str, int]


class DataQualityAssessment(BaseModel):
    recommendations: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)


class ModelRecommendation(BaseModel):
    target_column: str | None
    target_type: str | None
    candidate_feature_count: int
    model_families: list[str]
    metric_guidance: str


class DatasetReviewState(TypedDict, total=False):
    dataframe: object
    target_column: str | None
    profile: DatasetProfile
    quality: DataQualityAssessment
    recommendation: ModelRecommendation
    report: str
    execution_mode: str
    completed_nodes: list[str]
    warnings: list[str]
