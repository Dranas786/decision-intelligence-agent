# Code And Tools Guide

This document explains the major code files, how the request moves through the system, and why each tool exists.

## End-to-end request flow

1. The frontend in [app/static/index.html](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/static/index.html) gathers the dataset, optional context files, question, semantic config, and analysis params.
2. The upload endpoint in [app/api/demo_routes.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/api/demo_routes.py) saves the uploaded files locally under `data/demo_uploads/<run_id>/`.
3. Optional context files are ingested into the RAG layer.
4. The orchestrator in [app/agent/orchestrator.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/agent/orchestrator.py) loads the resource, asks the planner for a tool plan, executes deterministic tools, and builds the explanation layer.
5. The final answer is generated from grounded tool outputs plus retrieved RAG context.
6. If the user approves the engineering step, the local reporting module in [app/engineering/local_pipeline.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/engineering/local_pipeline.py) builds bronze/silver/gold artifacts and visualization payloads.
7. The frontend renders the engineer outputs and the visual report.

## Major files

### Core app files

- [app/main.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/main.py)
  What it does: creates the FastAPI app, configures CORS, mounts static files, serves `/`, and exposes `/healthz`.
  Why it exists: this is the root of the local web app.

- [app/static/index.html](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/static/index.html)
  What it does: single-page UI for the business-shareholder, analyst, engineer, and visualization stages.
  Why it exists: gives you a local interview-ready demo without a separate frontend build system.

- [app/api/demo_routes.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/api/demo_routes.py)
  What it does: receives uploads, saves files, ingests context into RAG, runs the analyst flow, and builds the local report on demand.
  Why it exists: this is the main demo-specific API surface.

- [app/api/routes.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/api/routes.py)
  What it does: exposes the lower-level `/v1/analyze` JSON API.
  Why it exists: useful for direct API testing and future integrations.

### Agent and orchestration

- [app/agent/orchestrator.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/agent/orchestrator.py)
  What it does: orchestrates the full analyst flow, including plan selection, tool execution, RAG retrieval, explanation-layer construction, and final answer assembly.
  Why it exists: keeps the execution flow centralized and deterministic.

- [app/agent/tools.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/agent/tools.py)
  What it does: maps tool names to Python functions, resolves arguments from semantic config and analysis params, and loads either tabular or point-cloud resources.
  Why it exists: this is the registry that makes the agent modular.

### LLM and RAG

- [app/llm/factory.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/llm/factory.py)
  What it does: provides the planner and answer generator, with Groq as the primary provider and rule-based fallback behavior.
  Why it exists: the model should choose and explain, not do the calculations.

- [app/llm/answer_builder.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/llm/answer_builder.py)
  What it does: turns grounded context into the final stakeholder-facing answer.
  Why it exists: separates computation from narrative.

- [app/rag/loaders.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/rag/loaders.py)
  What it does: loads `.txt` and `.md` files.
  Why it exists: RAG should work on the governance notes and background docs the user uploads.

- [app/rag/chunker.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/rag/chunker.py)
  What it does: splits documents into retrieval-sized chunks.
  Why it exists: embeddings and retrieval work better on smaller context blocks.

- [app/rag/embedder.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/rag/embedder.py)
  What it does: creates embeddings using either the local sentence-transformer path or the lightweight hash path.
  Why it exists: supports both local-full and lightweight environments.

- [app/rag/vector_store.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/rag/vector_store.py)
  What it does: writes and retrieves chunk vectors from Qdrant.
  Why it exists: this is the storage layer for context retrieval.

- [app/rag/service.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/rag/service.py)
  What it does: orchestrates ingest and retrieve operations.
  Why it exists: the rest of the app should not care about chunking or storage details.

### Engineering and reporting

- [app/engineering/local_pipeline.py](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/app/engineering/local_pipeline.py)
  What it does: creates local bronze/silver/gold outputs, SQLite tables, report summaries, and visualization payloads.
  Why it exists: this is the local data-engineering layer that follows the analyst step.

## Tool-by-tool guide

### General analyst tools

- `validate_dataset`
  What it does: checks completeness, duplicate rate, date conformity, categorical standardization opportunities, validity rules, sensitive-field signals, and human-review-required items.
  How it works: calculates missing-value share, duplicate share, parses date-like columns, looks for normalization opportunities, and flags invalid count-like fields.
  Why use it: this is the backbone of the governance and validity story.

- `profile_table`
  What it does: profiles the dataset shape, inferred types, likely keys, candidate business keys, grain, sensitive fields, and ambiguous fields.
  How it works: inspects dtypes, cardinality, duplicate rate, and key-like uniqueness patterns.
  Why use it: the analyst should understand the dataset before making recommendations.

- `detect_anomalies`
  What it does: screens a time-based metric for unexpected spikes or drops.
  How it works: uses rolling and z-score style anomaly logic.
  Why use it: useful for operational data and for finding events worth investigating.

- `segment_drivers`
  What it does: compares how segments contribute to the main metric.
  How it works: groups by a selected dimension and compares contribution changes.
  Why use it: helps explain business movement in a human-readable way.

- `scan_correlations`
  What it does: screens numeric features for likely relationships with a target metric.
  How it works: calculates ranked correlation-style evidence.
  Why use it: fast way to narrow candidate drivers.

- `fit_driver_regression`
  What it does: quantifies effect sizes against a target metric.
  How it works: fits a regression model using selected numeric features.
  Why use it: stronger than simple correlations when you need directional effect estimates.

- `forecast_metric`
  What it does: projects a metric forward in time.
  How it works: wraps a forecasting path over the chosen time series.
  Why use it: useful when the stakeholder asks for expected future movement.

- `bayesian_ab_test`
  What it does: evaluates a two-group uplift style comparison.
  How it works: compares outcome distributions between a control and treatment-like split.
  Why use it: useful when the data represents variants, tests, or experiments.

### Finance tools

- `calculate_returns`
  What it does: computes return series from price data.
  How it works: transforms ordered prices into return observations.
  Why use it: raw prices are less comparable than returns.

- `measure_risk`
  What it does: measures volatility and benchmark-aware risk.
  How it works: computes risk metrics from the return stream.
  Why use it: stakeholders usually want downside and variability context, not just performance.

- `measure_drawdown`
  What it does: quantifies downside peak-to-trough decline.
  How it works: compares the series against its running peak.
  Why use it: drawdown is often more intuitive than volatility.

- `detect_volume_spikes`
  What it does: highlights abnormal volume periods.
  How it works: uses z-score style screening on volume.
  Why use it: useful as a market attention or liquidity signal.

- `optimize_portfolio`
  What it does: proposes portfolio weights.
  How it works: applies portfolio optimization logic on price/return inputs.
  Why use it: useful when the stakeholder asks for allocation, not just diagnostics.

- `backtest_signal`
  What it does: tests a trading signal against price history.
  How it works: simulates signal-driven behavior on the time series.
  Why use it: lets the system move from descriptive to decision-oriented finance analysis.

### Healthcare tools

- `compute_readmission_rate`
  What it does: estimates readmission behavior over a chosen window.
  How it works: links patient records across time windows.
  Why use it: common operational KPI in admissions data.

- `compare_cohorts`
  What it does: compares outcomes between groups.
  How it works: aggregates outcome behavior by the cohort field.
  Why use it: useful for treatment group, site, or program comparisons.

- `analyze_length_of_stay`
  What it does: measures stay duration behavior.
  How it works: calculates time between admission and discharge.
  Why use it: core utilization metric for operations.

- `survival_risk_analysis`
  What it does: performs time-to-event style analysis.
  How it works: uses duration and event fields to summarize risk over time.
  Why use it: needed when timing of the event matters, not just whether it happened.

- `estimate_treatment_effect`
  What it does: estimates a treatment-outcome effect.
  How it works: compares treated and untreated patterns with the available features.
  Why use it: useful when the stakeholder wants intervention impact, not only description.

### Pipeline tools

- `profile_point_cloud`
  What it does: summarizes point count, bounds, density proxy, and normals.
  How it works: computes geometric bounds and scan metadata from the raw point cloud.
  Why use it: you need a basic scan health check before defect work.

- `clean_point_cloud`
  What it does: downsamples and removes outliers.
  How it works: uses voxel downsampling and neighbor-based outlier removal.
  Why use it: noisy clouds destabilize cylinder fitting and dent detection.

- `fit_pipe_cylinder`
  What it does: fits a nominal cylinder to the scan.
  How it works: estimates the main axis, projects points to a cross-sectional plane, and fits a circle to derive the radius and center.
  Why use it: dent and ovality detection require a nominal surface baseline.

- `compute_pipe_deviation_map`
  What it does: measures signed deviation from the nominal cylinder.
  How it works: converts each point to axial position, circumferential angle, and radial residual relative to the fitted radius.
  Why use it: this is the core defect surface map.

- `detect_pipe_dents`
  What it does: clusters inward deviations into dent candidates.
  How it works: thresholds negative deviations and groups nearby points with DBSCAN.
  Why use it: converts a surface defect field into discrete dent findings with depth and span.

- `measure_pipe_ovality`
  What it does: measures out-of-roundness along the pipe.
  How it works: slices the pipe axially and compares max versus min radius in each slice.
  Why use it: ovality is broader deformation and should be measured separately from dents.

## Engineering layer

### Bronze

What it does: keeps an immutable local copy of the uploaded file.
Why it exists: this preserves lineage and lets you show that the raw intake was not silently overwritten.

### Silver

What it does for tabular data: creates standardized working records with row-level quality flags.
What it does for pipeline data: stores cleaned geometry and deviation-style derived inspection data.
Why it exists: this is the layer where business logic and governance flags start becoming explicit.

### Gold

What it does: creates a local SQLite reporting database and final report artifacts.
Why it exists: gives the demo a real data-engineering handoff and a reporting-ready output layer.

## Visualization layer

### Tabular visuals

- Record quality status: shows how many rows are ready versus flagged.
- Missingness by field: highlights data quality hotspots.
- Metric by dimension: provides a business-facing rollup from the silver layer.

### Pipeline visuals

- 3D pipe scan: point cloud colored by deviation.
- Cross-section view: mid-pipe slice against the nominal circle.
- Unwrapped defect map: pipe surface laid out flat as axial position versus angle.
- Ovality profile: line view of slice-level ovality.
- Dent summary: bar chart of dent depths.

## Why the system is enterprise-friendly

- Deterministic tools calculate the numbers.
- The LLM is used for planning and explanation, not hidden computation.
- Governance, validation, and human review are treated as first-class outputs.
- The local engineering layer turns one-off analysis into reusable reporting artifacts.

## Recent Additions

- `audit_schema_contract`: checks required columns, expected type hints, and schema-version drift.
- `assess_freshness`: checks how recent the newest record is relative to warning and error thresholds.
- `audit_standardization`: finds label variants that should collapse into standard values in the silver layer.
- `detect_entity_collisions`: screens for duplicate-entity risk using normalized identifying keys.
- `/v1/demo/extra_info/<run_id>`: returns persisted analysis and report payloads without showing raw JSON on the main page.
- Pipeline report upgrades: 3D scan with dent centroids, unwrapped heatmap, axial defect profile, cross-section, ovality profile, and dent readout.

