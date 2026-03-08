# Decision Intelligence Agent

FastAPI demo application for deterministic analytics, retrieval-augmented answers, and lightweight pipeline point-cloud inspection.

## What it does

- Upload a CSV for general, finance, or healthcare analysis.
- Upload an `XYZ` point cloud for free-tier pipeline dent and ovality inspection.
- Upload optional `.txt` or `.md` context files for RAG.
- Store document vectors in Qdrant.
- Generate a grounded final answer using analytics results plus retrieved document context.

## Deployment profiles

The repo now supports two explicit profiles driven by `APP_PROFILE`.

### `hosted_free`

This is the default profile for low-memory hosts.

- Frontend: static page served from `/`.
- Backend: FastAPI app in `app/main.py`.
- RAG: uses a lightweight hash-based embedder by default.
- Vector store: remote Qdrant Cloud via `QDRANT_URL` and `QDRANT_API_KEY`.
- LLM: Groq when configured, rule-based fallback when not configured.
- Pipeline support on free-tier: `XYZ` uploads work without the optional `open3d` package.

### `local_full`

This profile is for your local machine when you want the better demo setup.

- Installs the heavier extras.
- Uses local semantic embeddings by default.
- Keeps `open3d` available for richer pipeline work.
- Uses the same frontend and backend structure, just with more capabilities enabled.

Heavy features are optional extras, not part of the base deploy:
- `rag-local`: installs `sentence-transformers`
- `pipeline`: installs `open3d` for `PLY` / `PCD`
- `advanced-analytics`: installs `prophet` and `pymc`

## Environment files

- [.env.example](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/.env.example) is the hosted free template.
- [.env.local.example](C:/Users/drana/Desktop/Projects/Main_practice_folder/decision-intelligence-agent/.env.local.example) is the local full template.

## Local full setup

Use the bootstrap scripts when you want the full local profile with `open3d`, semantic embeddings, and heavier analytics extras.

```powershell
.\scripts\setup_local.ps1
.\scripts\run_local.ps1
```

What the setup script does:
- copies `.env.local.example` to `.env` if needed
- creates `.venv`
- upgrades `pip`
- installs `.[pipeline,rag-local,advanced-analytics,finance,healthcare]`

What the run script does:
- loads variables from `.env`
- starts `uvicorn app.main:app --reload`

## Hosted free setup

Use `.env.example` values on your host.

Recommended hosted free env values:

```env
APP_PROFILE=hosted_free
PORT=8000
CORS_ALLOWED_ORIGINS=*
EMBEDDING_PROVIDER=hash
EMBEDDING_VECTOR_SIZE=384
QDRANT_URL=https://YOUR-CLUSTER-ID.cloud.qdrant.io:6333
QDRANT_API_KEY=YOUR_QDRANT_API_KEY
QDRANT_COLLECTION_NAME=decision_intelligence_demo
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

## Optional heavier installs

If you want local semantic embeddings only:

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

## Minimal local run

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
