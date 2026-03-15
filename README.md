# Decision Intelligence Agent

Local FastAPI demo for deterministic analytics, governance-aware explanations, local bronze/silver/gold reporting, and pipeline inspection visuals.

## Demo concept

The app now behaves like a small local multi-agent workflow:

1. **Business Shareholder** uploads a dataset and states the business question.
2. **Data Analyst** profiles the data, validates quality, runs domain tools, and returns a governed explanation.
3. **Business Shareholder** decides whether to build a local data product and report.
4. **Data Engineer** materializes bronze, silver, and gold outputs on disk and in SQLite.
5. **Visualization Studio** renders charts from the engineered outputs, including 3D and flattened pipe views for the pipeline domain.

The app is local-first. It is not designed around a remote warehouse or a hosted orchestration stack.

## What the app does

- General domain: data quality, governance, profiling, schema-contract review, freshness checks, standardization review, duplicate-entity screening, anomaly detection, segmentation, correlations, regression, forecasting, and Bayesian A/B analysis.
- Finance domain: returns, risk, drawdown, volume spikes, portfolio optimization, and signal backtesting.
- Healthcare domain: readmission, cohort comparison, length of stay, survival, and treatment-effect workflows.
- Pipeline domain: point-cloud profiling, cleaning, cylinder fitting, deviation mapping, dent detection, ovality measurement, and local inspection visuals.
- RAG: optional `.txt` / `.md` context ingestion into Qdrant for grounded explanations.
- Engineering layer: local bronze/silver/gold outputs and a SQLite reporting database.

## Key local workflow

1. Run the app locally.
2. Open `http://localhost:8000/`.
3. Upload a dataset or point cloud.
4. Optionally upload governance/context files.
5. Ask the analyst what to do with the data.
6. Review the analyst answer and explanation layer.
7. Approve the build-report step.
8. Review artifact paths, SQLite tables, and final visuals.

## Main files

- [app/main.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/main.py): FastAPI entrypoint and static frontend host.
- [app/static/index.html](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/static/index.html): single-page local demo UI with agent stages.
- [app/api/demo_routes.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/api/demo_routes.py): upload analysis endpoint plus local report-building endpoint.
- [app/agent/orchestrator.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/agent/orchestrator.py): analysis orchestration and explanation-layer assembly.
- [app/agent/tools.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/agent/tools.py): registry and argument resolution for deterministic tools.
- [app/engineering/local_pipeline.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/engineering/local_pipeline.py): local bronze/silver/gold reporting pipeline and visualization payloads.

## Local setup

Use the local full profile when you want the richest demo, especially for pipeline work.

```powershell
.\scripts\setup_local.ps1
.\scripts\run_local.ps1
```

Open:
- `http://localhost:8000/`
- `http://localhost:8000/healthz`
- `http://localhost:8000/docs`

## Demo files

Suggested sample files live here:
- [coffee_quality_demo.csv](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/coffee_quality_demo.csv)
- [coffee_quality_context.txt](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/coffee_quality_context.txt)
- [sample_pipe.xyz](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/sample_pipe.xyz)
- [sample_pipeline_context.txt](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/sample_pipeline_context.txt)
- [finance_market_demo.csv](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/finance_market_demo.csv)
- [finance_market_context.txt](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/finance_market_context.txt)
- [healthcare_admissions_demo.csv](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/healthcare_admissions_demo.csv)
- [healthcare_context.txt](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/healthcare_context.txt)
- [pipeline_inspection_context.md](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/pipeline_inspection_context.md)
- [DEMO_FILE_GUIDE.md](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/files_for_demo/DEMO_FILE_GUIDE.md)

## Bronze / Silver / Gold

The engineering step creates local outputs under `data/demo_reports/<run_id>_report/`.

- **Bronze**: immutable copy of the uploaded raw file.
- **Silver**: cleaned or standardized working data, quality flags, and derived inspection data.
- **Gold**: SQLite reporting database plus summary tables and output files.

For tabular data, the gold layer focuses on fact/dimension-style reporting tables and quality summaries.
For pipeline data, the gold layer focuses on deviation, dent, and ovality tables plus visualization-ready inspection outputs.

## Notes

- The analyst layer is deterministic-first. The LLM does not compute metrics.
- The current demo is stronger on governed assessment and reporting than on auto-fixing the raw file.
- The pipeline visuals are rendered in the browser from locally generated plot payloads.






