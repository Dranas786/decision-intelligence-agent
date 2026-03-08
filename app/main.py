from fastapi import FastAPI

from app.api.routes import router as api_router
from app.rag.rag_routes import router as rag_router


app = FastAPI(
    title="AI Decision Intelligence Agent",
    description="Agent that analyzes structured datasets and surfaces actionable insights.",
    version="0.1.0",
)


app.include_router(api_router)
app.include_router(rag_router)


@app.get("/")
def root():
    return {
        "message": "AI Decision Intelligence Agent is running",
        "docs": "/docs",
    }