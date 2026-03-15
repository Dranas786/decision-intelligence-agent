# Full System Architecture Guide

This document is the detailed explanation of how the whole project works.

It is written for someone who wants to understand:
- what every major file is doing
- how requests move through the system
- what each tool does
- how the RAG layer works
- how the LLM is used
- how the local engineering layer works
- how the pipeline 3D inspection flow works
- what gets saved to disk
- what the frontend is showing at each stage

This is the most complete architecture explanation in the repository.

## 1. What the application is

The application is a local **Decision Intelligence Agent**.

It is not just a chatbot.
It is a deterministic analytics and reporting workflow with an LLM on top.

The main idea is:
1. the user provides data
2. the system profiles and validates the data
3. the system selects domain-relevant analysis tools
4. the tools produce deterministic outputs
5. the system explains those outputs in business language
6. the user can then ask for a local data-engineering layer
7. the system builds bronze, silver, and gold outputs plus report visuals

There are four main domain modes:
- `general`
- `finance`
- `healthcare`
- `pipeline`

## 2. Core architectural principle

The most important architectural rule in this project is:

**the model does not do the calculations**

Instead:
- the planner decides which tools should run
- Python analytics functions do the real math and logic
- the final answer is written from grounded evidence

So the system is:
- LLM for planning and explanation
- deterministic code for analytics and engineering

That is why the project is more trustworthy than a pure prompt-based assistant.

## 3. High-level runtime map

The simplest mental model is this:

`Frontend -> FastAPI routes -> Orchestrator -> Tool registry -> Analytics modules -> RAG -> Final answer -> Optional engineering report -> Frontend visualizations`

In file terms:
- frontend: `app/static/index.html`
- web app entrypoint: `app/main.py`
- demo API: `app/api/demo_routes.py`
- core analysis orchestrator: `app/agent/orchestrator.py`
- tool registry and argument resolution: `app/agent/tools.py`
- analytics modules: `app/analytics/*`
- RAG modules: `app/rag/*`
- LLM planner and answer generation: `app/llm/*`
- local engineering and reporting: `app/engineering/local_pipeline.py`

## 4. Folder-by-folder map

### `app/`
Main application code.

### `app/api/`
FastAPI route definitions.

### `app/agent/`
The analysis orchestration layer.

### `app/analytics/`
Deterministic analytics functions for all domains.

### `app/llm/`
Planner and final-answer generation.

### `app/rag/`
Document ingestion, chunking, embedding, vector storage, retrieval, and prompt assembly.

### `app/engineering/`
The local bronze/silver/gold report builder.

### `app/static/`
The single-page frontend.

### `files_for_demo/`
Sample datasets and context documents.

### `docs/`
Human-readable project documentation.

### `scripts/`
Local setup and local run helpers.

### `data/`
Runtime-generated artifacts such as uploads, reports, and local Qdrant storage.

## 5. Main files in runtime order

The easiest way to understand the codebase is to read it in runtime order.

### `app/main.py`
This is the FastAPI entrypoint.

Responsibilities:
- create the FastAPI app
- configure CORS
- include the API routers
- mount static files
- serve the frontend at `/`
- expose `/healthz`

Why it matters:
- this is the root process that uvicorn runs
- if this file is broken, nothing else is reachable

### `app/static/index.html`
This is the local demo UI.

Responsibilities:
- present the business-shareholder workflow
- collect uploaded files, question, semantic config, and analysis params
- call the backend routes
- show analyst output
- ask for engineering approval
- render the final visualizations
- show chart guide modals

Important design note:
- this is a plain HTML/CSS/JS frontend
- it is served directly by FastAPI
- there is no separate frontend build system right now

### `app/api/demo_routes.py`
This is the main demo backend surface.

Responsibilities:
- accept uploads
- validate domain-specific file types
- save uploaded files into a run folder
- ingest optional context files into RAG
- call the analysis orchestrator
- persist the analysis payload
- expose sample demo packs
- trigger local report building
- expose extra run information

Important endpoints:
- `GET /v1/demo/sample-package/{domain}`
- `POST /v1/demo/analyze-upload`
- `POST /v1/demo/analyze-sample`
- `POST /v1/demo/build-report`
- `GET /v1/demo/extra_info/{run_id}`

### `app/agent/orchestrator.py`
This is the heart of the analysis stage.

Responsibilities:
- load the uploaded resource
- summarize the resource
- ask the planner for a tool plan
- enrich the plan with extra rule-based logic
- execute tools in order
- collect insights, charts, diagnostics, and structured evidence
- optionally retrieve RAG context
- build the explanation layer
- build the final answer input
- generate the final answer

This file is the bridge between:
- frontend intent
- planner decision-making
- deterministic tool execution
- final explanation

### `app/agent/tools.py`
This file is the registry and argument resolver.

Responsibilities:
- define what tool names exist
- load either a table or a point cloud depending on domain
- summarize the resource
- resolve arguments from semantic config and analysis params
- call the underlying analytics functions

This is what makes the system modular.
The orchestrator only needs tool names; this file knows how to run them.

## 6. Request lifecycle in detail

### 6.1 Analyst stage from uploaded files

The main user flow is:

1. The frontend sends a multipart request to `POST /v1/demo/analyze-upload`.
2. `demo_routes.py` validates the domain and file type.
3. A run folder is created under `data/demo_uploads/<run_id>/`.
4. The dataset file is copied there.
5. Optional `.txt` and `.md` files are copied there too.
6. If `use_rag` is true, those text files are ingested into the RAG system.
7. `run_agent(...)` is called from `app/agent/orchestrator.py`.
8. The orchestrator loads the dataset or point cloud.
9. The planner returns a tool plan.
10. The plan is enriched with deterministic rules.
11. The selected tools are executed.
12. An explanation layer is built.
13. Retrieved RAG context is optionally attached.
14. A final stakeholder-facing answer is generated.
15. The result is saved to `analysis_payload.json` in the run folder.
16. The frontend receives the response and renders the analyst workspace.

### 6.2 Sample-pack flow

The sample-pack flow is similar except the uploaded file is replaced with a preconfigured sample from `files_for_demo/`.

The route is `POST /v1/demo/analyze-sample`.

### 6.3 Engineering/report flow

Once the user approves the engineering step:

1. The frontend calls `POST /v1/demo/build-report`.
2. The request includes:
   - original dataset path
   - domain
   - semantic config
   - analysis params
   - full analysis result
   - visualization request
3. `build_local_report(...)` in `app/engineering/local_pipeline.py` runs.
4. The report builder creates a new folder under `data/demo_reports/<run_id>_report/`.
5. Bronze, silver, and gold subfolders are created.
6. Tabular or pipeline-specific local reporting outputs are built.
7. Visualization payloads are created.
8. The result is saved to `report_payload.json`.
9. The frontend receives the report and renders the engineer and visualization workspaces.

## 7. Resource types

The system handles two main resource types.

### Table resource
Used for:
- general
- finance
- healthcare

Loaded as:
- pandas DataFrame

Supported input formats:
- CSV
- Parquet for lower-level APIs

### Point-cloud resource
Used for:
- pipeline

Loaded as:
- `PointCloudData`

Supported input formats in the demo path:
- `.ply`
- `.pcd`
- `.xyz`
- `.xyzn`
- `.xyzrgb`

## 8. What `semantic_config` does

`semantic_config` tells the tool registry how to interpret columns or geometry settings.

Examples:
- `time_col`
- `primary_metric`
- `dimensions`
- `price_col`
- `patient_id_col`
- `units`
- `voxel_size`
- `axis_hint`
- `expected_radius`

This is important because the system should not guess everything from the raw data shape.

## 9. What `analysis_params` does

`analysis_params` tunes the analysis behavior.

Examples:
- anomaly z-score threshold
- rolling window
- forecast periods
- freshness thresholds
- duplicate-group settings
- dent threshold
- minimum cluster size
- slice spacing
- severity bands

Think of `semantic_config` as **meaning** and `analysis_params` as **behavior tuning**.

## 10. LLM layer

### `app/llm/factory.py`
This file provides the LLM client.

Responsibilities:
- choose planner and answer generation behavior
- use Groq if configured
- fall back to a rule-based planner when needed
- expose interfaces that the orchestrator can call

What the planner does:
- look at the question and domain
- choose relevant tools
- return a plan in a format the orchestrator expects

What the answer generator does:
- take grounded analytics and retrieved text context
- write a human-facing answer

### `app/llm/answer_builder.py`
This file converts structured tool output into grounded answer input.

Responsibilities:
- summarize analytics results
- combine explanation layer and retrieved RAG context
- prepare the text that the final answer generator uses

Critical architectural rule:
- the LLM is not doing regression, dent detection, risk, or forecasting math itself
- it is only planning and explaining

## 11. RAG architecture

The RAG system exists so uploaded notes, procedures, governance rules, or manuals can influence the final answer.

### `app/rag/loaders.py`
Loads `.txt` and `.md` documents.

### `app/rag/chunker.py`
Splits each document into smaller retrieval-sized chunks.

### `app/rag/embedder.py`
Creates embeddings.

It supports different runtime modes:
- lightweight hash embedding path
- sentence-transformer embedding path

Which path is preferred depends on profile and env settings.

### `app/rag/vector_store.py`
Stores and retrieves vectors in Qdrant.

Supports:
- remote Qdrant Cloud
- local persistent Qdrant path

### `app/rag/service.py`
High-level ingest and retrieve service used by the rest of the app.

### `app/rag/retriever.py`
Executes retrieval operations against the vector store.

### `app/rag/prompt_builder.py`
Builds the prompt context from retrieved chunks.

### `app/rag/factory.py`
Constructs the configured RAG service.

### `app/rag/rag_routes.py`
Exposes direct RAG routes outside the demo path.

### RAG flow end-to-end

1. context file uploaded
2. loader reads it
3. chunker splits it
4. embedder converts chunks into vectors
5. vector store writes them to Qdrant
6. later, the analysis question is embedded
7. similar chunks are retrieved
8. those chunks are added to the final answer context

## 12. Analytics modules

Each analytics file contains deterministic logic.

### `app/analytics/profiling.py`
Purpose:
- understand the basic structure of a table

Typical outputs:
- row count
- column count
- inferred types
- likely keys
- business key candidates
- grain
- sensitive fields
- ambiguous fields

Why it matters:
- almost all downstream reasoning depends on the system understanding what the dataset is

### `app/analytics/validation.py`
Purpose:
- enforce baseline quality checks

Typical outputs:
- missingness
- duplicate rate
- date parsing failures
- invalid values
- human review required
- sensitive field flags

Why it matters:
- this is the basis for the governance story

### `app/analytics/data_quality.py`
Purpose:
- governance-heavy quality tooling beyond the base validator

Tools here include:
- `audit_schema_contract`
- `assess_freshness`
- `audit_standardization`
- `detect_entity_collisions`

What each one does:

#### `audit_schema_contract`
Checks whether required fields are present and whether type expectations are being respected.
It also looks for schema-version drift if version-style columns exist.

Why it matters:
- downstream reporting breaks when schema assumptions drift silently

#### `assess_freshness`
Measures how old the newest valid timestamp is.

Why it matters:
- stale data can still be technically valid but operationally unusable

#### `audit_standardization`
Looks for label variants in categorical text columns.

Examples:
- `Downtown`
- `downtown`
- `Calgary DT`

Why it matters:
- inconsistent labels break grouping, dashboards, and governance consistency

#### `detect_entity_collisions`
Looks for duplicate-entity risk based on key-like text fields.

Why it matters:
- real business duplicates are often not exact row duplicates

### `app/analytics/anomalies.py`
Purpose:
- detect unusual movement in time-based metrics

Why it matters:
- operational and financial datasets often need spike or outlier review

### `app/analytics/segmentation.py`
Purpose:
- compare behavior by segment or dimension

Why it matters:
- this helps explain which groups are driving the metric

### `app/analytics/correlations.py`
Purpose:
- identify candidate relationships between numeric variables

Why it matters:
- useful for quick driver screening before regression

### `app/analytics/regression.py`
Purpose:
- estimate directional relationships against a target metric

Why it matters:
- stronger than simple pairwise correlation when you want effect-size directionality

### `app/analytics/forecasting.py`
Purpose:
- project a time-based metric forward

Why it matters:
- business users often ask where the metric is going, not only where it has been

### `app/analytics/bayes.py`
Purpose:
- run experiment-style Bayesian comparisons

Why it matters:
- gives the system a lightweight experimental analysis capability

### `app/analytics/finance.py`
Purpose:
- domain-specific finance calculations

Tools here:
- `calculate_returns`
- `measure_risk`
- `measure_drawdown`
- `detect_volume_spikes`
- `optimize_portfolio`
- `backtest_signal`

Why it matters:
- finance questions need domain-native calculations, not generic analytics alone

### `app/analytics/healthcare.py`
Purpose:
- healthcare-style patient and cohort analysis

Tools here:
- `compute_readmission_rate`
- `compare_cohorts`
- `analyze_length_of_stay`
- `survival_risk_analysis`
- `estimate_treatment_effect`

Why it matters:
- healthcare data has time windows, cohorts, durations, and interventions that general business tools do not model well

### `app/analytics/pipeline_3d.py`
Purpose:
- 3D inspection of isolated pipe-section scans

This is the most domain-specific analytics module in the project.

Main functions:
- `load_point_cloud`
- `profile_point_cloud`
- `clean_point_cloud`
- `fit_pipe_cylinder`
- `compute_pipe_deviation_map`
- `detect_pipe_dents`
- `measure_pipe_ovality`

#### Pipeline flow in detail

1. Load the point cloud.
2. Profile the cloud to check point count, bounds, normals, and general scan health.
3. Clean the cloud using downsampling and outlier removal.
4. Fit a nominal cylinder to represent the expected pipe shape.
5. Convert each point into:
   - axial position
   - angle around the pipe
   - radial deviation from the nominal cylinder
6. Threshold inward deviations to find dent candidate points.
7. Cluster those candidate points into dent events.
8. Measure each dent:
   - depth
   - axial span
   - circumferential span
   - area proxy
   - severity band
   - review priority
9. Slice the pipe along the axis to measure ovality.
10. Return all of that as structured evidence for engineering and visualization.

#### What a deviation map is
A deviation map is the difference between the actual scanned pipe surface and the fitted nominal cylinder.

Interpretation:
- negative residuals: inward dents
- positive residuals: outward bulges or surface lift
- near zero: surface close to nominal cylinder

#### What the dent detector is doing
The dent detector is not just classifying the whole pipe.
It is doing localized geometry analysis.

It works by:
- selecting points with enough inward deviation
- clustering nearby defect points
- discarding tiny noisy clusters
- computing physical metrics for each surviving cluster

That means the output is a set of measured defects, not a single yes/no label.

#### What ovality means here
Ovality is broader out-of-roundness.
A pipe can have low dent severity but still have global shape distortion.
That is why ovality is measured separately from localized dents.

## 13. Tool registry logic

`app/agent/tools.py` decides:
- how to load a resource
- how to summarize it
- how to resolve arguments for each tool
- which Python function to call

This file is where the domain-specific argument logic lives.

Examples:
- general tools resolve `time_col`, `primary_metric`, and `dimensions`
- finance tools resolve `price_col`, `volume_col`, `entity_col`
- healthcare tools resolve patient, admission, discharge, cohort, treatment, outcome fields
- pipeline tools resolve `units`, `voxel_size`, `axis_hint`, `expected_radius`, dent thresholds, and slice spacing

This is important because the same frontend can serve very different domains while the tool layer still stays deterministic.

## 14. Planner and plan enrichment

The planner creates an initial plan.
Then `orchestrator.py` enriches that plan with extra rules.

Examples:
- if the user asks about quality or governance, extra schema and freshness tools are added
- if the user asks about dents, dent detection is forced into the pipeline plan
- if the user asks about ovality, ovality measurement is added

So the final executed plan is a mix of:
- planner output
- safety and domain enrichment logic

This keeps the system practical and reduces the chance that the planner misses obvious required tools.

## 15. Explanation layer

One of the most important outputs is `explanation_layer`.

It is built in `app/agent/orchestrator.py`.

Its purpose is to translate raw tool outputs into a structured governance and decision-support summary.

Typical sections:
- dataset profile
- quality methodology
- quality findings
- actions taken
- governance notes
- human review required

Why it exists:
- interview demos and business stakeholders need more than raw metrics
- governance requires explicit explanation of risk, assumptions, and unresolved issues

## 16. Final answer generation

After tools run and the explanation layer is built:
- the orchestrator assembles a grounded answer input
- retrieved RAG chunks are added if available
- the answer builder creates the final narrative answer

Important rule:
- the final answer should explain the deterministic results
- it should not invent unsupported findings

## 17. Local engineering layer

The local engineering layer lives in `app/engineering/local_pipeline.py`.

Its purpose is to turn one-off analysis into a local data product.

### Output structure
It creates:
- `bronze/`
- `silver/`
- `gold/`

inside:
- `data/demo_reports/<run_id>_report/`

### Bronze layer
What it does:
- stores the raw input as an immutable copy

Why it exists:
- lineage
- traceability
- proof that raw intake was preserved

### Silver layer for tabular data
What it does:
- creates standardized records
- adds row-level quality flags
- parses dates
- normalizes selected text dimensions
- creates review-required flags
- writes `silver_records.csv`
- writes `quality_events.csv`

### Gold layer for tabular data
What it does:
- builds a local SQLite database
- writes dimension and fact-style tables
- writes governance and quality summary tables
- can write business rollups by dimension

Examples of gold-style tables:
- `dim_entity`
- `fact_records`
- `dq_summary`
- `dim_field_profile`
- `fact_quality_issues`
- `governance_review_queue`
- `freshness_summary`
- `standardization_candidates`
- `entity_collision_candidates`
- `gold_metric_by_dimension`

### Silver/gold logic for pipeline data
What it does:
- loads the point cloud again
- cleans it again for report-building consistency
- fits the cylinder
- computes deviations
- detects dents
- measures ovality
- writes structured pipeline fact tables
- constructs visualization payloads

Examples of pipeline reporting tables:
- `pipeline_summary`
- `fact_pipe_deviation`
- `fact_dent_events`
- `fact_ovality_slices`
- `fact_pipe_heatmap_bins`
- `fact_pipe_axial_profile`

### Why this layer matters
This is what makes the demo look like a data-engineering workflow instead of just an analytics page.

## 18. Visualization layer

The frontend renders report visualizations from payloads returned by the engineering layer.

The backend does not return pre-rendered images.
It returns chart definitions and supporting metadata.

### Tabular visuals
Examples:
- data quality status views
- missingness or issue distributions
- metric rollups by dimension
- governance and review summaries

### Pipeline visuals
Examples:
- 3D point-cloud view
- cross-section view
- unwrapped defect map
- axial defect profile
- ovality profile
- dent summary
- dent risk matrix

### Info-button guides
The frontend attaches a guide to many visualizations.

A guide contains:
- summary
- how to read it
- what to look for
- healthy example
- concerning example
- engineering significance

This is a major usability layer because many users do not automatically know how to interpret a defect map or ovality profile.

## 19. Frontend interaction model

The frontend now models four desks:
- Business Desk
- Analyst Desk
- Engineering Desk
- Visualization Studio

The active workspace below the desk scene changes depending on state.

The intended journey is:
1. business fills intake form
2. click send to analyst
3. analyst panel becomes useful
4. business approves engineering step
5. engineering panel shows artifacts and tables
6. visualization studio shows final charts

This is mostly a presentation layer over the same backend APIs.

## 20. Persistence and output files

### Upload runs
Stored under:
- `data/demo_uploads/<run_id>/`

Typical contents:
- uploaded dataset
- uploaded context files
- `analysis_payload.json`

### Report runs
Stored under:
- `data/demo_reports/<run_id>_report/`

Typical contents:
- bronze raw copy
- silver CSV outputs
- gold SQLite database
- `report_summary.json`
- `report_payload.json`

## 21. Profiles and runtime modes

`app/config.py` defines two application profiles:
- `hosted_free`
- `local_full`

### `hosted_free`
Purpose:
- lighter hosting mode

### `local_full`
Purpose:
- richer local demo mode
- better for full pipeline visuals and local embeddings

The profile influences defaults such as embedding behavior.

## 22. Setup and run scripts

### `scripts/setup_local.ps1`
Purpose:
- create `.venv`
- prepare local env file
- install the fuller local dependency set

### `scripts/run_local.ps1`
Purpose:
- load `.env`
- start uvicorn locally

These scripts exist so the project is easy to run as a local demo.

## 23. Demo files

The sample files in `files_for_demo/` exist to exercise different parts of the system.

### General
- `coffee_quality_demo.csv`
- `coffee_quality_context.txt`

### Finance
- `finance_market_demo.csv`
- `finance_market_context.txt`

### Healthcare
- `healthcare_admissions_demo.csv`
- `healthcare_context.txt`

### Pipeline
- `sample_pipe.xyz`
- `sample_pipeline_context.txt`
- `pipeline_multi_defect_demo.xyz`
- `pipeline_inspection_context.md`
- `pipeline_multi_defect_context.md`

The richer pipeline pack is designed to exercise:
- multiple dents
- ovality
- report visuals
- review-priority logic

## 24. How the business-user story maps to the code

### Business Shareholder
Frontend role in `app/static/index.html`.

What this role does:
- choose domain
- upload data
- write question
- approve the engineering step

### Data Analyst
Mostly represented by:
- `app/agent/orchestrator.py`
- `app/agent/tools.py`
- `app/analytics/*`
- `app/llm/*`
- `app/rag/*`

What this role does:
- profile
- validate
- analyze
- explain

### Data Engineer
Mostly represented by:
- `app/engineering/local_pipeline.py`

What this role does:
- materialize bronze/silver/gold
- build local report assets
- create reporting tables
- create visualization payloads

### Visualization Studio
Represented by:
- frontend rendering in `app/static/index.html`
- backend chart payload construction in `app/engineering/local_pipeline.py`

## 25. What the system does not currently do

Important limitations:
- it does not automatically rewrite and export a fully cleaned source-of-truth dataset for all domains
- it does not run on a remote data warehouse by default
- it does not use the LLM to write arbitrary transformation code on the fly
- it is local-first, not a fully deployed enterprise platform
- pipeline input assumes a single isolated pipe segment, not full-scene segmentation

## 26. How to read the code if you are learning it

Read in this order:
1. `app/main.py`
2. `app/static/index.html`
3. `app/api/demo_routes.py`
4. `app/agent/orchestrator.py`
5. `app/agent/tools.py`
6. `app/analytics/profiling.py`
7. `app/analytics/validation.py`
8. `app/analytics/data_quality.py`
9. domain analytics modules:
   - `finance.py`
   - `healthcare.py`
   - `pipeline_3d.py`
10. `app/rag/*`
11. `app/llm/*`
12. `app/engineering/local_pipeline.py`

That order follows the actual runtime flow and makes the architecture much easier to understand.

## 27. Summary

If you want one sentence for the whole architecture, it is this:

**The app is a local deterministic analytics and reporting system with an LLM used for tool planning and explanation, plus an optional local bronze/silver/gold engineering stage and a browser-based visualization workspace.**

If you want one sentence for the pipeline domain specifically, it is this:

**The pipeline flow takes an isolated point-cloud scan, fits a nominal cylinder, computes surface deviations, clusters inward defects into dent findings, measures ovality, and turns the result into engineering-facing visual evidence.**

If you want one sentence for the governance story, it is this:

**The system treats data quality, schema stability, freshness, standardization, sensitivity, and human review as first-class outputs rather than optional notes.**
