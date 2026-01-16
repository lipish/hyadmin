from fastapi import APIRouter

from .entrypoints import router as lic_router

router = APIRouter(prefix="/lic")

router.include_router(lic_router)
