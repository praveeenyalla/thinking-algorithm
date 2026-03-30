from functools import lru_cache

from fastapi import APIRouter

from app.heart.controller import HeartController
from app.schemas import HealthResponse, SolveRequest, SolveResponse

router = APIRouter()


@lru_cache(maxsize=1)
def get_controller() -> HeartController:
    return HeartController.build_default()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    controller = get_controller()
    snapshot = controller.memory_snapshot()
    return HealthResponse(
        status="ok",
        llm_mode=controller.llm_mode,
        total_runs=snapshot.total_runs,
    )


@router.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    controller = get_controller()
    return SolveResponse(**controller.run(request.problem).model_dump())
