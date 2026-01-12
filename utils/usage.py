from pydantic import BaseModel

class Usage(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0
    total_tokens: int = 0
