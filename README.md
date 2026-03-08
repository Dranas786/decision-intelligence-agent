# Decision Intelligence Agent

FastAPI demo application for deterministic analytics, retrieval-augmented answers, and lightweight pipeline point-cloud inspection.

## What it does

- Upload a CSV for general, finance, or healthcare analysis.
- Upload an `XYZ` point cloud for free-tier pipeline dent and ovality inspection.
- Upload optional `.txt` or `.md` context files for RAG.
- Store document vectors in Qdrant.
- Generate a grounded final answer using analytics results plus retrieved document context.

## Free-tier deployment profile

The default deployment is optimized for low-memory hosts.

- Frontend: static page served from `/`.
- Backend: FastAPI app in `app/main.py`.
- RAG: uses a lightweight hash-based embedder by default.
- Vector store: remote Qdrant Cloud via `QDRANT_URL` and `QDRANT_API_KEY`.
- LLM: Groq when configured, rule-based fallback when not configured.
- Pipeline support on free-tier: `XYZ` uploads work without the optional `open3d` package.

Heavy features are optional extras, not part of the base deploy:
- `rag-local`: installs `sentence-transformers`
- `pipeline`: installs `open3d` for `PLY` / `PCD`
- `advanced-analytics`: installs `prophet` and `pymc`

## Environment variables

Copy `.env.example` to `.env` and set the values you need.

Required for hosted Qdrant:
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`

Recommended for hosted API access:
- `CORS_ALLOWED_ORIGINS=https://your-frontend-domain.com`
- `PORT=8000`

Recommended for free-tier RAG:
- `EMBEDDING_PROVIDER=hash`
- `EMBEDDING_VECTOR_SIZE=384`

Optional for local semantic embeddings:
- `EMBEDDING_PROVIDER=sentence-transformer`
- `EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`

Optional for LLM answers:
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_PLANNER_MODEL`
- `GROQ_ANSWER_MODEL`

Optional local fallback settings:
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

## Optional heavier installs

If you want local semantic embeddings:

```bash
pip install .[rag-local]
```

If you want `PLY` / `PCD` pipeline support:

```bash
pip install .[pipeline]
```

If you want the heavier forecasting / Bayesian extras:

```bash
pip install .[advanced-analytics]
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
EMBEDDING_PROVIDER=hash
EMBEDDING_VECTOR_SIZE=384
```

## Demo flow

1. Open the frontend.
2. Set the API base URL if the frontend and API are hosted on different domains.
3. Upload a dataset or an `XYZ` point cloud.
4. Optionally upload `.txt` or `.md` RAG context files.
5. Ask a question.
6. Run the analysis and review the final answer plus the structured JSON response.
