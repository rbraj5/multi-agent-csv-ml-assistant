from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.schemas import DatasetReviewState, TraceEvent
from src.tools import assess_data_quality, profile_dataset, recommend_models, review_readiness


def _completed(state: DatasetReviewState, node: str) -> list[str]:
    return [*state.get("completed_nodes", []), node]


def _trace(state: DatasetReviewState, node: str, summary: str) -> list[TraceEvent]:
    return [*state.get("trace_events", []), TraceEvent(node=node, summary=summary)]


def _profile_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    profile = profile_dataset(df)
    return {
        "profile": profile,
        "completed_nodes": _completed(state, "profile_dataset"),
        "trace_events": _trace(state, "profile_dataset", f"Profiled {profile.row_count} rows and {profile.column_count} columns."),
    }


def _quality_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    quality = assess_data_quality(df, state["profile"])
    return {
        "quality": quality,
        "completed_nodes": _completed(state, "assess_data_quality"),
        "trace_events": _trace(state, "assess_data_quality", f"Generated {len(quality.recommendations)} data-quality recommendation(s)."),
    }


def _readiness_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    readiness = review_readiness(df, state["profile"], state["quality"], state.get("target_column"))
    return {
        "readiness": readiness,
        "completed_nodes": _completed(state, "review_readiness"),
        "trace_events": _trace(state, "review_readiness", readiness.status),
    }


def _recommend_node(state: DatasetReviewState) -> DatasetReviewState:
    df = state["dataframe"]
    recommendation = recommend_models(df, state.get("target_column"))
    return {
        "recommendation": recommendation,
        "completed_nodes": _completed(state, "recommend_models"),
        "trace_events": _trace(state, "recommend_models", recommendation.metric_guidance),
    }


def _route_after_quality(state: DatasetReviewState) -> str:
    quality = state["quality"]
    profile = state["profile"]
    if profile.missing_values or profile.duplicate_rows or quality.high_cardinality_columns or state.get("target_column"):
        return "review_readiness"
    return "recommend_models"


def _deterministic_report(state: DatasetReviewState) -> str:
    profile = state["profile"]
    quality = state["quality"]
    readiness = state.get("readiness")
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
    if readiness:
        sections.extend(["", "## Readiness Gate"])
        sections.append(f"- Status: {readiness.status}")
        if readiness.class_balance:
            sections.append(f"- Target balance: {readiness.class_balance}")
        if readiness.risks:
            sections.append("- Risks:")
            sections.extend(f"  - {item}" for item in readiness.risks)
        if readiness.required_actions:
            sections.append("- Required actions:")
            sections.extend(f"  - {item}" for item in readiness.required_actions)
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
            f"Readiness: {state.get('readiness').model_dump() if state.get('readiness') else None}\n"
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
        "trace_events": _trace(state, "synthesize_report", f"Generated report in {execution_mode} mode."),
    }


def build_graph():
    graph = StateGraph(DatasetReviewState)
    graph.add_node("profile_dataset", _profile_node)
    graph.add_node("assess_data_quality", _quality_node)
    graph.add_node("review_readiness", _readiness_node)
    graph.add_node("recommend_models", _recommend_node)
    graph.add_node("synthesize_report", _synthesis_node)
    graph.add_edge(START, "profile_dataset")
    graph.add_edge("profile_dataset", "assess_data_quality")
    graph.add_conditional_edges(
        "assess_data_quality",
        _route_after_quality,
        {
            "review_readiness": "review_readiness",
            "recommend_models": "recommend_models",
        },
    )
    graph.add_edge("review_readiness", "recommend_models")
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
            "trace_events": [],
            "warnings": [],
            "execution_mode": "Deterministic fallback",
        }
    )
