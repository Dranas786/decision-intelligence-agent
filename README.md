# Decision Intelligence Agent

FastAPI demo application for deterministic analytics, retrieval-augmented answers, and pipeline point-cloud inspection.

## What it does

- Upload a CSV for general, finance, or healthcare analysis.
- Upload a `PLY`, `PCD`, or `XYZ` point cloud for pipeline dent and ovality inspection.
- Upload optional `.txt` or `.md` context files for RAG.
- Store document vectors in Qdrant.
- Generate a grounded final answer using analytics results plus retrieved document context.

## Hosted demo architecture

- Frontend: static page served from `/` and configurable to call any hosted API base URL.
- Backend: FastAPI app in `app/main.py`.
- Analytics orchestration: `app/agent/orchestrator.py`.
- RAG: chunking, embeddings, Qdrant storage, retrieval, prompt building under `app/rag/`.
- LLM: Groq when configured, rule-based fallback when not configured.
- Vector store: remote Qdrant Cloud via `QDRANT_URL` and `QDRANT_API_KEY`, or local persistent storage via `QDRANT_PATH`.

## Environment variables

Copy `.env.example` to `.env` and set the values you need.

Required for hosted Qdrant:
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`

Recommended for hosted API access:
- `CORS_ALLOWED_ORIGINS=https://your-frontend-domain.com`
- `PORT=8000`

Optional for LLM answers:
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_PLANNER_MODEL`
- `GROQ_ANSWER_MODEL`

Optional for embeddings and local fallback:
- `EMBEDDING_MODEL_NAME`
- `QDRANT_PATH`
- `RAG_ALLOWED_ROOTS`

## Local run

```bash
pip install .
uvicorn app.main:app --reload
```

Open:
- `http://localhost:8000/` for the demo UI
- `http://localhost:8000/docs` for the API docs
- `http://localhost:8000/healthz` for a health check

## Docker run

Build and run:

```bash
docker build -t decision-intelligence-agent .
docker run --env-file .env -p 8000:8000 decision-intelligence-agent
```

## Qdrant Cloud setup

1. Create a Qdrant Cloud cluster.
2. Copy the HTTPS endpoint into `QDRANT_URL`.
3. Copy the API key into `QDRANT_API_KEY`.
4. Set a collection name in `QDRANT_COLLECTION_NAME`.
5. Start the app and check `/healthz`.

Example:

```env
QDRANT_URL=https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333
QDRANT_API_KEY=YOUR_QDRANT_API_KEY
QDRANT_COLLECTION_NAME=decision_intelligence_demo
```

## Demo flow

1. Open the frontend.
2. Set the API base URL if the frontend and API are hosted on different domains.
3. Upload a dataset or point cloud.
4. Optionally upload `.txt` or `.md` RAG context files.
5. Ask a question.
6. Run the analysis and review the final answer plus the structured JSON response.
