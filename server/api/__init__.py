from fastapi import APIRouter

from .openai import router as openai_router
from .anthropic import router as anthropic_router
from .status import router as status_router
from .lic import router as lic_router
router = APIRouter()
router.include_router(openai_router)
router.include_router(anthropic_router)
router.include_router(status_router)
router.include_router(lic_router)
