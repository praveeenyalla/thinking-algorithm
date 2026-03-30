from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Distributed Cognitive AI System",
    version="0.1.0",
    description="Phase 1 distributed cognitive AI architecture with Brain, Heart, agents, communication bus, and memory.",
)

app.include_router(router)
