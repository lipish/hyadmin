from fastapi import APIRouter

from .chat.entrypoints import router as chat_router
from .models.entrypoints import router as models_router

router = APIRouter(prefix='/v1')

router.include_router(chat_router)
router.include_router(models_router)
