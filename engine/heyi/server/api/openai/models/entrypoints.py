from __future__ import annotations

from fastapi import APIRouter

from heyi.server.config import Config

from .schema import ListModelsResponse, Model

router = APIRouter()


@router.get("/models", response_model=ListModelsResponse, tags=["Models"])
def list_models() -> ListModelsResponse:
    """
    List models
    """
    return ListModelsResponse.model_construct(
        data=[
            Model.model_construct(
                id=Config().model_name,
            )
        ]
    )
