# Low-Level Code Reference

This document is the developer-oriented explanation of the codebase.

Unlike the recruiter-facing README, this file is intentionally low-level.
It focuses on:
- file responsibilities
- important functions and classes
- request/response shapes
- internal state passed between modules
- how tools are executed
- what the engineering layer writes

## 1. Reading order

If you want to understand the code without getting lost, read in this order:

1. `app/main.py`
2. `app/config.py`
3. `app/static/index.html`
4. `app/api/routes.py`
5. `app/api/demo_routes.py`
6. `app/agent/orchestrator.py`
7. `app/agent/tools.py`
8. `app/analytics/profiling.py`
9. `app/analytics/validation.py`
10. `app/analytics/data_quality.py`
11. domain analytics files:
    - `finance.py`
    - `healthcare.py`
    - `pipeline_3d.py`
12. `app/rag/*`
13. `app/llm/*`
14. `app/engineering/local_pipeline.py`

## 2. App bootstrap

### `app/main.py`

#### `app = FastAPI(...)`
Creates the web app.

#### `_cors_allowed_origins() -> list[str]`
Reads `CORS_ALLOWED_ORIGINS` from the environment.
Returns either:
- `[*]` if all origins are allowed
- a parsed list of comma-separated origins

#### `root() -> FileResponse`
Serves `app/static/index.html`.

#### `healthz() -> dict[str, str | bool]`
Returns a small runtime summary with:
- status
- active app profile
- whether Groq is configured
- embedding provider
- Qdrant target

## 3. Runtime configuration

### `app/config.py`

#### Constants
- `HOSTED_FREE_PROFILE`
- `LOCAL_FULL_PROFILE`

#### `app_profile() -> str`
Returns the active profile from `APP_PROFILE`.
Falls back to `hosted_free`.

#### `profile_default(...) -> str`
Utility helper that returns one default for hosted mode and another for local mode.
This is used to keep runtime defaults profile-aware.

## 4. API contracts

### `app/api/routes.py`
This is the lower-level JSON API.

#### `AnalyzeRequest`
Main request model for `/v1/analyze`.

Fields:
- `dataset_path`
- `question`
- `table_name`
- `domain`
- `semantic_config`
- `tool_whitelist`
- `analysis_params`
- `use_rag`
- `rag_limit`

#### `ExecutedTool`
Small structured record describing an executed tool.

#### `RetrievedChunkResponse`
Shape of a retrieved RAG chunk in the API response.

#### `AnalyzeResponse`
Main structured analysis response.

Important fields:
- `plan`
- `executed_tools`
- `insights`
- `insight_objects`
- `diagnostics`
- `charts`
- `rag_used`
- `retrieved_chunks`
- `rag_prompt`
- `combined_context`
- `grounded_answer_input`
- `final_answer`
- `analysis_brief`
- `explanation_layer`

#### `analyze(request)`
Thin wrapper around `run_agent(...)`.
Converts exceptions into HTTP 400 or 500.

### `app/api/demo_routes.py`
This is the richer local demo API.

#### Constants
- `UPLOAD_ROOT`
- `FILES_FOR_DEMO_ROOT`
- `PIPELINE_DATASET_EXTENSIONS`
- `TABULAR_DATASET_EXTENSIONS`
- `CONTEXT_FILE_EXTENSIONS`
- `ANALYSIS_PAYLOAD_NAME`
- `REPORT_PAYLOAD_NAME`

#### `BuildReportRequest`
Payload for the report-building step.
Includes the original analysis result so the engineering layer can reuse already-computed evidence.

#### `AnalyzeSampleRequest`
Payload for running a bundled sample pack.

#### `_safe_filename(name)`
Normalizes uploaded filenames for local storage.

#### `_dataset_extensions_for_domain(domain)`
Routes file-extension validation by domain.

#### `_json_default(value)`
Serialization helper for NumPy arrays, paths, and other non-basic types.

#### `_write_json(path, payload)`
Writes JSON payloads to disk.

#### `_read_json(path)`
Reads a stored JSON payload back from disk if it exists.

#### `_sample_package_catalog()`
Returns the hardcoded sample-pack registry.
Each pack contains:
- label
- dataset path
- context file paths
- question
- semantic config
- analysis params
- visualization request
- notes

#### `_get_sample_package(domain)`
Validates that a sample pack exists and that its files are present.

#### `_package_metadata(domain)`
Returns a frontend-friendly summary of the sample pack.

#### `_materialize_context_files(...)`
Copies bundled context files into the run folder and optionally ingests them into RAG.

#### `_run_analysis_and_persist(...)`
Calls `run_agent(...)`, attaches `demo_run` metadata, and writes `analysis_payload.json`.
This is the main helper used by both upload and sample analysis paths.

#### `sample_package(domain)`
Route: `GET /v1/demo/sample-package/{domain}`
Returns sample-pack metadata.

#### `analyze_upload(...)`
Route: `POST /v1/demo/analyze-upload`
This is the main upload endpoint.

Steps:
1. parse JSON strings from the form
2. validate domain
3. validate file type
4. create a run directory
5. save dataset
6. save accepted context files
7. optionally ingest context into RAG
8. call `_run_analysis_and_persist(...)`

#### `analyze_sample(request)`
Route: `POST /v1/demo/analyze-sample`
Copies files from `files_for_demo/` into a run folder, ingests context if enabled, and runs the same analysis path.

#### `build_report(request)`
Route: `POST /v1/demo/build-report`
Calls `build_local_report(...)`, stores `report_payload.json`, and returns the report object.

#### `extra_info(run_id)`
Route: `GET /v1/demo/extra_info/{run_id}`
Returns saved analysis/report payloads for a given run.

## 5. Orchestrator internals

### `app/agent/orchestrator.py`
This file controls the full analysis lifecycle.

#### `SUPPORTED_DOMAINS`
Defines the valid domains.

#### `TOOL_LABELS`
Maps tool names to human-readable action labels used in the explanation layer.

#### `_append_tool_if_missing(plan, tool_name, args=None)`
Utility to enrich a plan safely without duplicating a tool.

#### `_enrich_plan(plan, question, domain)`
Adds domain- and keyword-based tools after planning.

Examples:
- governance questions add schema, freshness, standardization, and collision tools
- dent questions add dent detection
- ovality questions add ovality measurement

#### `_tool_action_label(tool_name)`
Returns a human-readable label for a tool.

#### `_build_explanation_layer(...)`
Builds the governance and explanation object returned to the frontend.

Main output sections:
- `dataset_profile`
- `quality_methodology`
- `quality_findings`
- `actions_taken`
- `governance_notes`
- `human_review_required`

This function uses evidence from `insight_objects` to convert raw tool results into a business-usable narrative structure.

#### `_build_grounded_answer_input(...)`
Builds the text block used by the final answer generator.
It includes:
- question
- domain
- resource summary
- executed tools
- insights
- structured objects
- explanation layer
- retrieved chunks
- diagnostics

#### `_build_analysis_brief(...)`
Creates a compact summary object that the frontend or docs can use to summarize what happened.

#### `run_agent(...)`
Main entrypoint for the analyst stage.

High-level steps:
1. validate domain
2. load resource with `load_resource(...)`
3. summarize resource with `summarize_resource(...)`
4. ask LLM client for a plan
5. enrich plan
6. execute tools one by one
7. collect chart payloads, diagnostics, insights, and structured objects
8. optionally retrieve RAG chunks
9. build explanation layer
10. build grounded answer input
11. generate final answer
12. return a structured response dict

## 6. Tool registry internals

### `app/agent/tools.py`
This file is the execution switchboard.

#### Resource loading

##### `load_dataset(dataset_path)`
Supports CSV and Parquet.
Returns a pandas DataFrame.

##### `load_resource(dataset_path, domain=None)`
If `domain == pipeline`, it loads a point cloud.
Otherwise it loads a table.

##### `summarize_resource(resource)`
Returns a small metadata summary.
For tables:
- `resource_kind`
- `columns`
- `row_count`

For point clouds:
- `resource_kind`
- `point_count`
- `has_normals`
- `bounds`

#### Argument helpers

##### `_preferred_metric(...)`
Finds the most useful metric column from `semantic_config` or numeric columns.

##### `_preferred_segment(...)`
Finds the most useful segment/dimension column.

##### `_resolve_general_feature_cols(...)`
Chooses numeric feature columns for correlation and regression.

##### `_resolve_pipeline_args(...)`
Builds tool arguments for the pipeline domain.

Examples:
- `profile_point_cloud` gets units
- `clean_point_cloud` gets voxel size, outlier settings, and normal-estimation behavior
- `detect_pipe_dents` gets deviation threshold, min cluster points, and severity bands
- `measure_pipe_ovality` gets slice spacing and units

##### `_resolve_tabular_args(...)`
Builds tool arguments for all tabular tools.
This function is one of the most important in the file because it translates semantic config into actual runtime parameters.

#### Execution helpers

The `_run_*` functions are thin wrappers around the analytics modules.
They exist so the registry can manage shared state between tools.

Examples:
- `_run_profile`
- `_run_validate`
- `_run_schema_contract`
- `_run_freshness`
- `_run_standardization`
- `_run_entity_collisions`
- `_run_profile_point_cloud`
- `_run_clean_point_cloud`
- `_run_fit_pipe_cylinder`
- `_run_compute_pipe_deviation_map`
- `_run_detect_pipe_dents`
- `_run_measure_pipe_ovality`

Shared-state behavior is important in the pipeline domain.
For example:
- cleaned cloud can be reused
- fitted cylinder can be reused
- deviation map can be reused by dent and ovality steps

#### Public registry functions

##### `list_available_tools(domain=None, tool_whitelist=None)`
Returns tool metadata available for a domain.

##### `build_tool_step(...)`
Builds a normalized tool step object.

##### `execute_tool(...)`
The main dispatcher.
It resolves arguments, checks usability, runs the correct `_run_*` wrapper, updates shared state, and returns a structured tool result.

## 7. Analytics module reference

### `app/analytics/profiling.py`
Primary function:
- builds dataset structure summaries

Expected concerns handled:
- row and column counts
- inferred types
- likely keys
- grain
- sensitive field names
- ambiguous field names

### `app/analytics/validation.py`
Primary function:
- validates baseline data quality expectations

Expected concerns handled:
- required columns
- missingness
- duplicate rows
- parse failures
- obvious invalid values
- human review candidates

### `app/analytics/data_quality.py`
Primary functions:
- `audit_schema_contract`
- `assess_freshness`
- `audit_standardization`
- `detect_entity_collisions`

These are governance-focused tools and are central to the data-quality story.

### `app/analytics/anomalies.py`
Primary function:
- anomaly detection for time-based metrics

### `app/analytics/segmentation.py`
Primary function:
- segment-based metric analysis

### `app/analytics/correlations.py`
Primary function:
- ranked numeric relationship screening

### `app/analytics/regression.py`
Primary function:
- regression-based effect estimation

### `app/analytics/forecasting.py`
Primary function:
- forward metric projection

### `app/analytics/bayes.py`
Primary function:
- Bayesian experiment-style comparison

### `app/analytics/finance.py`
Primary functions:
- returns
- risk
- drawdown
- volume spikes
- portfolio optimization
- signal backtesting

### `app/analytics/healthcare.py`
Primary functions:
- readmission rate
- cohort comparison
- length of stay
- survival analysis
- treatment effect estimation

### `app/analytics/pipeline_3d.py`
This file is structurally different from the tabular modules because it works on geometry instead of rows.

Key concepts in this file:
- `PointCloudData`
- raw point loading
- cleaning/downsampling
- cylinder fit
- deviation map
- dent clustering
- ovality slicing

Important mental model:
- most tabular tools return summaries from columns
- pipeline tools return summaries from spatial geometry and derived surface coordinates

## 8. LLM internals

### `app/llm/factory.py`
This file contains the LLM strategy layer.

#### `RuleBasedPlanner`
Fallback planner when a cloud model is unavailable or intentionally not used.

#### `GroqClient`
Primary provider wrapper when Groq is configured.

#### `HybridLLMClient`
Combines provider behavior and fallback behavior into one app-facing client.

#### `get_llm_client()`
Returns the configured hybrid client.

Main design point:
- the orchestrator expects a client that can plan and generate
- the factory hides whether Groq or fallback logic is active

### `app/llm/answer_builder.py`

#### `_build_structured_fallback(combined_context)`
Fallback text-generation path if the main generation path is unavailable.

#### `build_final_answer(...)`
Top-level answer assembly function.

This function should always work from already-grounded evidence.

## 9. RAG internals

### `app/rag/schemas.py`
Typed models for ingestion and retrieval objects.

### `app/rag/loaders.py`
Loads raw text documents from disk.

### `app/rag/chunker.py`
Splits documents into chunks.

### `app/rag/embedder.py`
Embeds each chunk.

Important runtime idea:
- local full mode can use sentence-transformers
- lighter mode can use a hash embedding path

### `app/rag/vector_store.py`
Wrapper around Qdrant.

Responsibilities:
- ensure collection exists
- upsert vectors
- search vectors

### `app/rag/retriever.py`
Retrieval helper sitting above the store.

### `app/rag/prompt_builder.py`
Formats retrieved chunks into prompt-ready context.

### `app/rag/service.py`
High-level RAG interface.

This is what the app actually uses.
The rest of the app usually does not call the embedder or store directly.

### `app/rag/factory.py`
Builds the configured RAG stack.

### `app/rag/rag_routes.py`
Direct RAG endpoints for lower-level or separate testing.

## 10. Engineering/report builder internals

### `app/engineering/local_pipeline.py`
This is the biggest file after the orchestrator because it handles local reporting outputs.

#### `_tool_evidence_map(analysis_result)`
Builds a map from tool name to evidence payload so the report layer can reuse analyst results without re-parsing the whole response each time.

#### `build_local_report(...)`
Top-level engineering/report entrypoint.

Responsibilities:
- create report folders
- route to tabular or pipeline report builder
- attach artifact root and agent story metadata

#### `_build_tabular_report(...)`
Tabular reporting pipeline.

Main work:
- copy raw dataset to bronze
- create silver DataFrame with flags and normalized columns
- write silver CSVs
- build profile, quality, governance, standardization, collision, and freshness tables
- create SQLite gold database
- build report summary and report markdown
- build tabular visualization payloads

#### `_build_dim_entity(df)`
Builds entity dimension table.

#### `_build_fact_records(df, dim_entity, time_col)`
Builds row-level fact table.

#### `_build_dim_time(parsed_dates)`
Builds a small time dimension.

#### `_build_tabular_quality_summary(...)`
Creates gold-style quality summary table.

#### `_build_field_profile(...)`
Creates field-level metadata summary.

#### `_build_quality_issue_facts(...)`
Creates issue-level table describing quality problems.

#### `_build_governance_review_queue(...)`
Turns explanation-layer human review items into a queue-style table.

#### `_flatten_standardization_candidates(...)`
Turns standardization evidence into a table.

#### `_flatten_collision_candidates(...)`
Turns duplicate-entity evidence into a table.

#### `_build_freshness_table(...)`
Turns freshness evidence into a reporting table.

#### `_build_pipeline_report(...)`
Pipeline reporting pipeline.

Main work:
- copy raw point cloud to bronze
- load point cloud
- clean point cloud
- fit cylinder
- compute deviations
- detect dents
- measure ovality
- build pipeline summary tables
- build pipeline visuals
- build report markdown and report summary

#### `_build_pipeline_summary_table(...)`
Creates one summary table for pipe-level scan health and defect statistics.

#### `_build_tabular_visualizations(...)`
Builds browser-ready visualization payloads for tabular reporting.

#### `_build_pipeline_visualizations(...)`
Builds browser-ready payloads for pipeline inspection views.

#### `_build_pipe_heatmap_bins(...)`
Creates axial-angle bins for the unwrapped defect map.

#### `_build_pipe_axial_profile(...)`
Creates the defect profile along the pipe axis.

#### `_visual_guide(...)`
Common helper to attach guide metadata to a visualization.

#### `_tabular_visual_guide(viz_id)`
Returns a guide object for tabular charts.

#### `_pipeline_visual_guide(viz_id, units, context=None)`
Returns a guide object for pipeline charts.

#### `_plotly_bar(...)` and `_plotly_layout(...)`
Small chart payload helpers.

#### `_sample_points(...)`
Downsamples point arrays for frontend plotting performance.

## 11. Frontend low-level behavior

### `app/static/index.html`
This file now does a lot more than a normal static form.

Main frontend state:
- `lastAnalysis`
- `lastRequest`
- `samplePackage`
- `currentGuideViz`
- `currentStage`
- `currentPanel`

Important frontend helpers:

#### `setStage(stageIndex)`
Updates:
- stage pill text
- narrative text
- scene brief
- stakeholder position in the desk scene
- actor caption

#### `setAgentState(key, state, status, note)`
Updates one desk card in the scene.

#### `setDeskAvailability(panelId, enabled)`
Enables or disables stage navigation and desk buttons.

#### `activatePanel(panelId, options={})`
Switches the visible lower workspace panel.
This is how the same page behaves like multiple workspaces without navigating away.

#### `updateFormForDomain()`
Updates file-extension hints and default semantic config by domain.

#### `parseResponsePayload(response)`
Safely handles both JSON and plain-text error responses.

#### `importSamplePackage()`
Loads sample pack metadata from the backend and pre-fills the intake form.

#### `guideIllustration(type)`
Returns the inline SVG illustration used in the chart guide modal.

#### `openGuide(viz)` and `closeGuide()`
Open and close the visualization guide modal.

#### `formatExplanationLayer(layer)`
Converts the explanation layer into readable text for the analyst panel.

#### `renderVisualizations(visualizations)`
Builds visualization cards and calls Plotly for chart rendering.

#### `runAnalysis()`
Frontend handler for the analyst step.

Main work:
- gather files and inputs
- parse semantic config and analysis params
- call either `analyze-upload` or `analyze-sample`
- update desk states
- update analyst workspace outputs

#### `buildReport()`
Frontend handler for the engineering step.

Main work:
- send analysis result plus visualization request to `build-report`
- update desk states
- update engineering and visualization workspaces

## 12. Data structures that matter most

### `insights`
List of short, human-readable findings.

### `insight_objects`
Structured evidence objects per tool.
This is where the detailed machine-readable outputs live.

### `diagnostics`
Warnings, skipped-tool reasons, and fit-quality caveats.

### `charts` or visualization payloads
Browser-renderable plot definitions.

### `explanation_layer`
Governance and explanation summary.

### `demo_run`
Run metadata added by the demo route.
Includes:
- `run_id`
- `dataset_path`
- `dataset_kind`
- `ingested_files`
- `extra_info_url`
- `sample_label`

## 13. What gets reused between layers

The system does not start from zero at each stage.

Examples of reuse:
- the engineering stage receives the whole `analysis_result`
- the report builder reuses `insight_objects` evidence
- the pipeline stage reuses fitted geometry logic and report-specific derivations
- the frontend reuses `lastAnalysis` and `lastRequest` when building the report

## 14. What to debug first when something breaks

### If the app does not open
Start with:
- `app/main.py`
- uvicorn startup
- `.env`

### If upload fails
Start with:
- `app/api/demo_routes.py`
- file extension validation
- JSON parsing of semantic config and analysis params

### If a tool is skipped unexpectedly
Start with:
- `app/agent/tools.py`
- `_resolve_args(...)`
- `_args_are_usable(...)`

### If the final answer is weak
Start with:
- `app/agent/orchestrator.py`
- `app/llm/answer_builder.py`
- RAG retrieval evidence

### If the report is weak
Start with:
- `app/engineering/local_pipeline.py`
- visualization payload builders
- evidence extracted from `insight_objects`

### If pipeline results look wrong
Start with:
- `app/analytics/pipeline_3d.py`
- cylinder fit quality
- deviation threshold
- cluster size threshold
- slice spacing

## 15. Short developer summary

At the lowest level, the project is organized like this:

- FastAPI receives the request
- demo routes store files and normalize inputs
- the orchestrator asks for a plan and runs tools
- the tool registry resolves arguments and dispatches analytics
- analytics modules produce deterministic evidence
- the explanation layer turns evidence into governed findings
- the LLM writes a final answer from grounded context
- the local engineering layer turns results into bronze/silver/gold artifacts and chart payloads
- the frontend renders those outputs in a staged workspace

That is the real code path behind the demo.
