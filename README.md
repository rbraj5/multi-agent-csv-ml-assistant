# Multi-Agent CSV ML Assistant

A simple Streamlit app that reviews a CSV dataset through four focused agents:

- EDA Analyst
- Data Cleaning Advisor
- Model Recommender
- Report Writer

The app is designed as a practical junior-level portfolio project. It runs in demo mode without any API keys and uses deterministic Python analysis for the core workflow.

## Features

- Upload a CSV file or use the bundled sample dataset
- Review dataset shape, column types, missing values, duplicates, and target hints
- Generate cleaning recommendations
- Suggest suitable machine learning model families
- Produce a downloadable Markdown report

## Project Structure

```text
multi-agent-csv-ml-assistant/
├── app.py
├── sample_data/
│   └── patient_readmission_sample.csv
├── screenshots/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Start the app.
2. Upload a CSV file or keep the default sample dataset.
3. Choose an optional target column.
4. Review each agent output.
5. Download the generated Markdown report.

## Notes

The `.env.example` file includes optional variables for future LLM integration, but the current app works without external services.

