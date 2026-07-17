import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI
from src.api.routers.query import router as query_router

app = FastAPI(
    title="🏛️ Vietnamese Legal RAG - Serving Layer",
    description="API serving layer with keyword-based routing for Vietnamese legal documents.",
    version="1.0.0"
)

# Include query router
app.include_router(query_router)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Vietnamese Legal RAG API serving layer!"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
