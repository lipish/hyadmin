from fastapi import APIRouter

# from .endpoints.chat import router as chat_router
from .entrypoints import router as entrypoints_router

router = APIRouter(prefix='/anthropic/v1')

router.include_router(entrypoints_router)
