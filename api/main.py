"""FastAPI app exposing the AI brain system pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from brain.controller import thinking_pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT_DIR / "ui" / "index.html"

app = FastAPI(
    title="AI Brain System API",
    version="0.6.0",
    description="HTTP interface for the AI brain system with persistent managed memory.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    """Input payload for the thinking endpoint."""

    problem: str = Field(..., min_length=3, description="Problem statement to process.")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the browser UI."""
    return INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    """Simple health endpoint for local development."""
    return {"status": "ok"}


@app.post("/think")
def think(query: Query) -> dict[str, object]:
    """Run the thinking pipeline for a single problem."""
    return {"result": thinking_pipeline(query.problem)}
