from pydantic import BaseModel
from typing import Dict, Literal, List, Optional

class RequestStatus(BaseModel):
    state: Literal[
        "PENDING",
        "PREFILLING",
        "LPREFILLING",
        "DECODING",
        "FINISHED",
        "CANCELLED"
    ]
    ttft: float
    avg_tbt: float
    p95_tbt: float
    throughput: float
    completion_tokens: int
    prompt_tokens: int
    cache_hit_tokens: int
    total_tokens: int


class StatusResponse(BaseModel):
    heyi_sn: str = ""
    licmgr_state: Literal[
        "LICENSE_INVALID",
        "SN_INVALID",
        "COLLECTING_FAILED",
        "LICENSING",
        "LICENSING_FAILED",
        "LICENSED",
    ]
    licmgr_error: str
    engine_state: Literal[
        "INIT", "BOOTING", "RUNNING", "LPREFILLING", "ERROR"
    ]
    request_counters: Optional[Dict[Literal["pending", "prefilling", "decoding"], int]] = None
    decode_throughput: Optional[float] = None
    config: Optional[Dict] = None
    requests: Optional[List[RequestStatus]] = None
