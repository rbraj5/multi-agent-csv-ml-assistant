from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.graph import run_dataset_review


SAMPLE_PATH = Path("sample_data/patient_readmission_sample.csv")


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

    state = run_dataset_review(df, target_value)
    st.caption(f"Execution mode: {state['execution_mode']}")
    st.write("Completed graph nodes:", " -> ".join(state["completed_nodes"]))

    if state.get("warnings"):
        for warning in state["warnings"]:
            st.warning(warning)

    with st.expander("Dataset Profile", expanded=True):
        st.json(state["profile"].model_dump())

    with st.expander("Data Quality Assessment", expanded=True):
        st.json(state["quality"].model_dump())

    with st.expander("Model Recommendation", expanded=True):
        st.json(state["recommendation"].model_dump())

    st.subheader("Dataset Review Report")
    st.markdown(state["report"])
    st.download_button("Download report", state["report"], "dataset_review_report.md")


if __name__ == "__main__":
    main()
