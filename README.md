# Multi-Agent CSV ML Assistant

## Overview
Multi-Agent CSV ML Assistant is a production-ready local container service for early machine-learning dataset intake. It combines LangGraph orchestration, deterministic data-quality tools, Pydantic validation, FastAPI backend integration, Docker packaging, and optional LangChain/OpenAI report synthesis.

The deterministic path is the reliability baseline. Optional LLM synthesis is only used to write a stakeholder-ready report from structured dataset facts.

## Production Use Case
The service supports ML discovery and data-readiness review before teams commit to modelling. It profiles uploaded CSV content, checks missingness and duplicates, identifies high-cardinality fields, runs target-readiness checks, and recommends baseline model families with metric guidance.

## Architecture
- FastAPI exposes the LangGraph workflow as a service API with OpenAPI documentation at `/docs`.
- LangGraph controls state and conditional readiness routing.
- Deterministic pandas/scikit-learn tools calculate factual dataset outputs.
- Pydantic validates API requests and structures responses.
- Docker packages the API as a production-ready local container.
- Trace events, completed nodes, request IDs, and warnings provide lightweight observability.
- LangSmith tracing can be enabled through environment variables when needed.

## LangGraph Workflow
```text
profile_dataset -> assess_data_quality -> review_readiness? -> recommend_models -> synthesize_report
```

- `profile_dataset`: calculates dataset shape, data types, missing values, and duplicate rows.
- `assess_data_quality`: converts profile findings into cleaning recommendations.
- `review_readiness`: conditionally checks modelling risks such as missing values, duplicates, class imbalance, and target-leakage indicators.
- `recommend_models`: infers target type and recommends baseline model families.
- `synthesize_report`: produces a Markdown report through deterministic fallback or optional LangChain/OpenAI synthesis.

## API Usage
Run locally, then open the interactive API docs:

```powershell
http://localhost:8000/docs
```

Health and metadata:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metadata
```

Workflow request:

```powershell
$csv = Get-Content .\sample_data\patient_readmission_sample.csv -Raw
$body = @{ csv_text = $csv; target_column = "readmitted" } | ConvertTo-Json
curl -X POST http://localhost:8000/workflow -H "Content-Type: application/json" -d $body
```

The response includes `request_id`, `execution_mode`, `completed_nodes`, `trace_events`, `profile`, `quality_report`, `readiness_review`, `model_recommendation`, `report`, and `warnings`.

## Docker Run
Build and run the API container locally:

```powershell
docker build -t multi-agent-csv-ml-assistant:local .
docker run --rm -p 8000:8000 --env-file .env.example multi-agent-csv-ml-assistant:local
```

The Docker health check calls `/health`. The container starts the API with:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Local Streamlit/CLI Demo
The production API does not replace the Streamlit demo.

```powershell
streamlit run app.py
```

## Configuration
Use `.env.example` as the configuration template:

```text
APP_ENV=local
APP_VERSION=0.1.0
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LOG_LEVEL=INFO
```

No OpenAI or LangSmith configuration is required for deterministic operation. Do not commit secrets or cloud credentials.

## Testing
Run local checks:

```powershell
python -m compileall .
python -m unittest discover -s tests
docker build -t multi-agent-csv-ml-assistant:local .
```

The test suite includes workflow tests, FastAPI endpoint tests, and README contract tests that enforce the required documentation structure.

## Azure Container Apps Deployment Path
This repo is Azure Container Apps ready, but no cloud deployment is required for the local demo.

Example deployment path:

```powershell
az group create --name rg-agentic-ai-demo --location uksouth
az containerapp env create --name cae-agentic-ai-demo --resource-group rg-agentic-ai-demo --location uksouth
az containerapp create `
  --name csv-ml-assistant-api `
  --resource-group rg-agentic-ai-demo `
  --environment cae-agentic-ai-demo `
  --image <registry>/multi-agent-csv-ml-assistant:latest `
  --target-port 8000 `
  --ingress external `
  --env-vars APP_ENV=azure APP_VERSION=0.1.0 LOG_LEVEL=INFO
```

Configure secrets such as `OPENAI_API_KEY` through Azure Container Apps secret management, not in source control.

## Production Readiness Notes
- FastAPI backend available.
- Docker image buildable.
- Health and readiness endpoints available.
- Pydantic request/response validation.
- Deterministic fallback without API key.
- Optional LangChain/OpenAI synthesis.
- Trace events returned in API responses.
- CI validates compile/tests/Docker build.
- Azure Container Apps deployment path documented.

## Limitations and Next Steps
- Authentication and rate limiting are not implemented.
- CSV input is processed per request and is not persisted.
- Azure deployment commands are documented but not executed here.
- Future extensions would add data contracts, baseline model benchmarking, drift checks, experiment tracking, and durable workflow checkpointing.
