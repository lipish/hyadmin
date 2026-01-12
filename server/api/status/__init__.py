from fastapi import APIRouter

from .entrypoints import router as status_router

router = APIRouter(prefix="/status")

router.include_router(status_router)
