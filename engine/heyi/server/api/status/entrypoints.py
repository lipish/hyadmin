from fastapi import APIRouter

from heyi.engine import Engine
from heyi.licmgr import licmgr_flags, heyi_sn

from .schema import StatusResponse

router = APIRouter()

@router.get("/status", response_model=StatusResponse, tags=["Status"])
def status() -> StatusResponse:
    licmgr_state = licmgr_flags.get_state()
    licmgr_error = licmgr_flags.licensing_error
    status = Engine().get_status()
    return StatusResponse.model_construct(
        heyi_sn=heyi_sn,
        licmgr_state=licmgr_state, 
        licmgr_error=licmgr_error,
        **status
    )
