"""FastAPI app exposing the AI brain system pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import AliasChoices, BaseModel, Field, field_validator

from brain.controller import thinking_pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT_DIR / "ui" / "index.html"

app = FastAPI(
    title="AI Brain System API",
    version="0.7.0",
    description="HTTP interface for the AI brain system with persistent memory and multi-agent deliberate reasoning.",
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

    problem: str = Field(
        ...,
        validation_alias=AliasChoices("problem", "query"),
        description="Problem statement to process.",
    )
    mode: str = Field(
        default="chat",
        description="Response mode. Use deep_research for more research-style answers.",
    )

    @field_validator("problem")
    @classmethod
    def validate_problem(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Please enter a question.")
        return cleaned

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        if normalized not in {"chat", "deep_research"}:
            raise ValueError("Mode must be 'chat' or 'deep_research'.")
        return normalized


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
    try:
        return {"result": thinking_pipeline(query.problem, response_mode=query.mode)}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
