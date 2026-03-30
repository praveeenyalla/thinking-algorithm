from pydantic import BaseModel, Field

from app.models import CognitiveResponse


class SolveRequest(BaseModel):
    problem: str = Field(..., min_length=5, description="Problem statement to analyze.")


class SolveResponse(CognitiveResponse):
    pass


class HealthResponse(BaseModel):
    status: str
    llm_mode: str
    total_runs: int
