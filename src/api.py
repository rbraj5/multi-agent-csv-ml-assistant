from __future__ import annotations

import logging
import os
from io import StringIO
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import run_dataset_review

PROJECT_NAME = "Multi-Agent CSV ML Assistant"
WORKFLOW_NODES = ["profile_dataset", "assess_data_quality", "review_readiness", "recommend_models", "synthesize_report"]

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("csv_ml_api")

app = FastAPI(
    title=PROJECT_NAME,
    version=os.getenv("APP_VERSION", "0.1.0"),
    description="Production-ready FastAPI wrapper for the LangGraph dataset review workflow.",
)


class HealthResponse(BaseModel):
    service: str
    version: str
    status: str
    environment: str


class MetadataResponse(BaseModel):
    service: str
    workflow_nodes: list[str]
    execution_modes: list[str]
    llm_configured: bool


class DatasetWorkflowRequest(BaseModel):
    csv_text: str = Field(..., min_length=1)
    target_column: str | None = None


class DatasetWorkflowResponse(BaseModel):
    request_id: str
    execution_mode: str
    completed_nodes: list[str]
    trace_events: list[dict]
    warnings: list[str]
    profile: dict
    quality_report: dict
    readiness_review: dict | None
    model_recommendation: dict
    report: str


@app.get("/health", response_model=HealthResponse, summary="Container health check")
def health() -> HealthResponse:
    return HealthResponse(
        service=PROJECT_NAME,
        version=os.getenv("APP_VERSION", "0.1.0"),
        status="healthy",
        environment=os.getenv("APP_ENV", "local"),
    )


@app.get("/ready", response_model=HealthResponse, summary="Runtime readiness check")
def ready() -> HealthResponse:
    return HealthResponse(
        service=PROJECT_NAME,
        version=os.getenv("APP_VERSION", "0.1.0"),
        status="ready",
        environment=os.getenv("APP_ENV", "local"),
    )


@app.get("/metadata", response_model=MetadataResponse, summary="Workflow metadata")
def metadata() -> MetadataResponse:
    return MetadataResponse(
        service=PROJECT_NAME,
        workflow_nodes=WORKFLOW_NODES,
        execution_modes=["Deterministic fallback", "LLM-assisted synthesis"],
        llm_configured=bool(os.getenv("OPENAI_API_KEY")),
    )


@app.post("/workflow", response_model=DatasetWorkflowResponse, summary="Run the LangGraph dataset review workflow")
def workflow(payload: DatasetWorkflowRequest) -> DatasetWorkflowResponse:
    request_id = str(uuid4())
    try:
        df = pd.read_csv(StringIO(payload.csv_text))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV input: {exc}") from exc

    if payload.target_column and payload.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"target_column '{payload.target_column}' was not found in the CSV")

    logger.info("running dataset workflow request_id=%s rows=%s", request_id, len(df))
    state = run_dataset_review(df, payload.target_column)
    return DatasetWorkflowResponse(
        request_id=request_id,
        execution_mode=state["execution_mode"],
        completed_nodes=state["completed_nodes"],
        trace_events=[event.model_dump() for event in state.get("trace_events", [])],
        warnings=state.get("warnings", []),
        profile=state["profile"].model_dump(),
        quality_report=state["quality"].model_dump(),
        readiness_review=state.get("readiness").model_dump() if state.get("readiness") else None,
        model_recommendation=state["recommendation"].model_dump(),
        report=state["report"],
    )
