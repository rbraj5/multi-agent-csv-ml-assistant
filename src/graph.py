from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.schemas import DatasetReviewState
from src.tools import assess_data_quality, profile_dataset, recommend_models


def _completed(state: DatasetReviewState, node: str) -> list[str]:
    return [*state.get("completed_nodes", []), node]


def _profile_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    return {
        "profile": profile_dataset(df),
        "completed_nodes": _completed(state, "profile_dataset"),
    }


def _quality_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    return {
        "quality": assess_data_quality(df, state["profile"]),
        "completed_nodes": _completed(state, "assess_data_quality"),
    }


def _recommend_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    return {
        "recommendation": recommend_models(df, state.get("target_column")),
        "completed_nodes": _completed(state, "recommend_models"),
    }


def _deterministic_report(state: DatasetReviewState) -> str:
    profile = state["profile"]
    quality = state["quality"]
    recommendation = state["recommendation"]

    sections = [
        "# Dataset Review Report",
        "",
        "## Dataset Profile",
        f"- Rows: {profile.row_count}",
        f"- Columns: {profile.column_count}",
        f"- Numeric columns: {len(profile.numeric_columns)} ({', '.join(profile.numeric_columns) or 'none'})",
        f"- Categorical columns: {len(profile.categorical_columns)} ({', '.join(profile.categorical_columns) or 'none'})",
        f"- Duplicate rows: {profile.duplicate_rows}",
        "",
        "## Data Quality",
    ]

    if profile.missing_values:
        sections.append("Missing values:")
        sections.extend(f"- {column}: {count}" for column, count in profile.missing_values.items())
    else:
        sections.append("- Missing values: none detected")

    sections.extend(["", "## Cleaning Recommendations"])
    sections.extend(f"- {item}" for item in quality.recommendations)
    sections.extend(["", "## Model Recommendation"])
    sections.append(f"- Target column: {recommendation.target_column or 'not selected'}")
    sections.append(f"- Target type: {recommendation.target_type or 'not selected'}")
    sections.append(f"- Candidate feature columns: {recommendation.candidate_feature_count}")
    sections.append("- Suggested model families:")
    sections.extend(f"  - {item}" for item in recommendation.model_families)
    sections.append(f"- Metric guidance: {recommendation.metric_guidance}")
    sections.extend(
        [
            "",
            "## Next Steps",
            "- Validate assumptions with domain knowledge.",
            "- Split data into train/test sets before model evaluation.",
            "- Track preprocessing and model metrics in a reproducible notebook.",
        ]
    )
    return "\n".join(sections)


def _llm_report(state: DatasetReviewState) -> tuple[str, list[str]]:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return _deterministic_report(state), []

    try:
        from langchain_openai import ChatOpenAI

        profile = state["profile"]
        quality = state["quality"]
        recommendation = state["recommendation"]
        model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        response = model.invoke(
            "Create a concise stakeholder-ready dataset review report. "
            "Use only the structured facts below and do not invent metrics.\n\n"
            f"Profile: {profile.model_dump()}\n"
            f"Quality: {quality.model_dump()}\n"
            f"Model recommendation: {recommendation.model_dump()}\n"
        )
        return str(response.content), []
    except Exception as exc:  # pragma: no cover - defensive fallback for local demos
        return _deterministic_report(state), [f"LLM synthesis failed; deterministic report used. Reason: {exc}"]


def _synthesis_node(state: DatasetReviewState) -> DatasetReviewState:
    report, warnings = _llm_report(state)
    execution_mode = "LLM-assisted synthesis" if os.getenv("OPENAI_API_KEY") and not warnings else "Deterministic fallback"
    return {
        "report": report,
        "warnings": [*state.get("warnings", []), *warnings],
        "execution_mode": execution_mode,
        "completed_nodes": _completed(state, "synthesize_report"),
    }


def build_graph():
    graph = StateGraph(DatasetReviewState)
    graph.add_node("profile_dataset", _profile_node)
    graph.add_node("assess_data_quality", _quality_node)
    graph.add_node("recommend_models", _recommend_node)
    graph.add_node("synthesize_report", _synthesis_node)
    graph.add_edge(START, "profile_dataset")
    graph.add_edge("profile_dataset", "assess_data_quality")
    graph.add_edge("assess_data_quality", "recommend_models")
    graph.add_edge("recommend_models", "synthesize_report")
    graph.add_edge("synthesize_report", END)
    return graph.compile()


def run_dataset_review(df: pd.DataFrame, target_column: str | None) -> DatasetReviewState:
    graph = build_graph()
    return graph.invoke(
        {
            "dataframe": df,
            "target_column": target_column,
            "completed_nodes": [],
            "warnings": [],
            "execution_mode": "Deterministic fallback",
        }
    )
