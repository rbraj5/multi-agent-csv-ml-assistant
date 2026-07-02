# Multi-Agent CSV ML Assistant

Multi-Agent CSV ML Assistant is a Streamlit-based data review tool for rapid assessment of tabular datasets. The application coordinates four focused analysis stages to profile uploaded CSV files, identify data quality risks, recommend modelling approaches, and produce a stakeholder-ready Markdown report.

## Operational Context

The system supports early-stage machine learning discovery work where teams need a fast, repeatable review of a new dataset before committing to deeper modelling. It uses deterministic Python analysis for the core workflow, so it can run in local or controlled environments without external AI services.

## Agent Workflow

- **EDA Analyst:** profiles shape, data types, missing values, duplicates, and column composition.
- **Data Cleaning Advisor:** identifies practical cleaning actions before modelling.
- **Model Recommender:** suggests baseline model families based on the selected target type.
- **Report Writer:** consolidates findings into a downloadable Markdown report.

## Capabilities

- CSV upload with bundled sample data fallback
- Dataset preview and target-column selection
- Automated profiling for schema, missingness, duplicate rows, and target hints
- Cleaning recommendations for numeric and categorical fields
- Classification or regression model family suggestions
- Downloadable dataset review report for handoff or documentation

## Repository Structure

```text
multi-agent-csv-ml-assistant/
|-- app.py
|-- sample_data/
|   `-- patient_readmission_sample.csv
|-- screenshots/
|   `-- .gitkeep
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Local Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Usage Flow

1. Launch the Streamlit application.
2. Upload a CSV file or use the bundled sample dataset.
3. Select an optional target column.
4. Review each agent output.
5. Download the generated Markdown report.

## Notes

The `.env.example` file is reserved for future LLM-backed extensions. The current implementation runs without external services.
