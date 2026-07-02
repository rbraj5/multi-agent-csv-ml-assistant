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
    unique_counts: dict[str, int] = Field(default_factory=dict)


class DataQualityAssessment(BaseModel):
    recommendations: list[str] = Field(default_factory=list)
    high_cardinality_columns: list[str] = Field(default_factory=list)


class ModelRecommendation(BaseModel):
    target_column: str | None
    target_type: str | None
    candidate_feature_count: int
    model_families: list[str]
    metric_guidance: str


class ReadinessReview(BaseModel):
    status: str
    risks: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    class_balance: dict[str, int] = Field(default_factory=dict)
    leakage_warnings: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    node: str
    summary: str


class DatasetReviewState(TypedDict, total=False):
    dataframe: object
    target_column: str | None
    profile: DatasetProfile
    quality: DataQualityAssessment
    readiness: ReadinessReview
    recommendation: ModelRecommendation
    report: str
    execution_mode: str
    completed_nodes: list[str]
    trace_events: list[TraceEvent]
    warnings: list[str]
