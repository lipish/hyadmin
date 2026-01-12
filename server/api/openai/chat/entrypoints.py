from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterable
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from heyi.engine import Engine
from heyi.server.config import Config
from heyi.utils.log import logger
from heyi.utils.request import Request as EngineRequest
from heyi.utils.usage import Usage

from .tool_call_utils import extract_tool_calls
from .schema import (
    ChatCompletionStreamResponseDelta,
    Choice1,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CompletionUsage,
    PromptTokensDetails,
    Choice,
    ChatCompletionResponseMessage,
    ModelIds,
)

router = APIRouter()


def streamer(request: CreateChatCompletionRequest, req: EngineRequest):
    async def streamer():
        try:
            output_buffer = ""
            tool_mode = False
            async for res in req.stream.generator():

                chunk = CreateChatCompletionStreamResponse.model_construct(
                    id="stream-resp-" + uuid4().hex,
                    choices=[],
                    created=int(time.time()),
                    model=request.model,
                )

                if isinstance(res, Usage):
                    chunk.usage = CompletionUsage.model_construct(
                        prompt_tokens=res.prompt_tokens - res.cache_hit_tokens,
                        completion_tokens=res.completion_tokens,
                        total_tokens=res.total_tokens - res.cache_hit_tokens,
                        prompt_tokens_details=PromptTokensDetails.model_construct(
                            cached_tokens=res.cache_hit_tokens
                        ),
                    )
                    yield chunk
                    continue

                token, reason = res if isinstance(res, tuple) else (res, None)

                output_buffer += token
                tool_call_begin_at, tool_call_end_at, tool_calls = extract_tool_calls(
                    output_buffer, request
                )

                if tool_call_begin_at != -1:
                    assert not tool_mode

                    tool_mode = True
                    content, output_buffer = (
                        output_buffer[:tool_call_begin_at],
                        output_buffer[tool_call_begin_at:],
                    )
                    if content:
                        chunk.choices = [
                            Choice1.model_construct(
                                delta=ChatCompletionStreamResponseDelta.model_construct(
                                    content=content,
                                    role="assistant",
                                ),
                                finish_reason="tool_calls",
                                index=0,
                            )
                        ]
                        yield chunk
                    continue

                if tool_mode:
                    if tool_call_end_at != -1:
                        chunk.choices = [
                            Choice1.model_construct(
                                delta=ChatCompletionStreamResponseDelta.model_construct(
                                    tool_calls=tool_calls,
                                    role="tool",
                                ),
                                finish_reason="tool_calls",
                                index=0,
                            )
                        ]
                        yield chunk
                        return
                else:
                    chunk.choices = [
                        Choice1.model_construct(
                            delta=ChatCompletionStreamResponseDelta.model_construct(
                                content=token,
                                role="assistant",
                            ),
                            finish_reason=reason,
                            index=0,
                        )
                    ]
                    yield chunk

        except asyncio.CancelledError:
            logger.warning("Inference cancelled")
            req.cancel()
        finally:
            if not tool_mode:
                chunk.choices = [
                    Choice1.model_construct(
                        index=0,
                        delta=ChatCompletionStreamResponseDelta.model_construct(),
                        finish_reason="stop",
                    )
                ]
                yield chunk

    return streamer()


async def to_sse(events: AsyncIterable[Any]) -> AsyncIterable[str]:
    async for e in events:
        if isinstance(e, str):
            yield e
        elif isinstance(e, BaseModel):
            yield f"data: {e.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post(
    "/chat/completions", response_model=CreateChatCompletionResponse, tags=["Chat"]
)
async def create_chat_completion(
    body: CreateChatCompletionRequest,
):
    """
    Create chat completion
    """
    req_id = uuid4().hex

    if isinstance(body.model, str):
        model_id = ModelIds.from_name(body.model)
        if not model_id or model_id != ModelIds.from_name(Config().model_name):
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"The model `{body.model}` does not match or is invalid.",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    }
                },
            )
        else:
            body.model = model_id

    messages = [m.model_dump() for m in body.messages]

    generation_config = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_logprobs,
        "max_new_tokens": body.max_completion_tokens or body.max_new_tokens or body.max_tokens,
    }

    if body.tools is not None:
        tools = [t.model_dump() for t in body.tools]
    else:
        tools = None

    req = Engine().submit(req_id, messages, generation_config, tools)

    if body.stream:
        return StreamingResponse(
            to_sse(streamer(body, req)), media_type="text/event-stream"
        )

    # NON-STREAMING
    output_buffer = ""
    tool_mode = False

    async for res in req.stream.generator():
        if isinstance(res, Usage):
            usage = CompletionUsage.model_construct(
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                total_tokens=res.total_tokens,
                prompt_tokens_details=PromptTokensDetails.model_construct(
                    cached_tokens=res.cache_hit_tokens
                ),
            )
            continue

        token, reason = res if isinstance(res, tuple) else (res, None)
        output_buffer += token

        tool_call_begin_at, tool_call_end_at, tool_calls = extract_tool_calls(
            output_buffer, body
        )

        if tool_call_begin_at != -1 and not tool_mode:
            tool_mode = True
            content, output_buffer = (
                output_buffer[:tool_call_begin_at],
                output_buffer[tool_call_begin_at:],
            )
            continue

        if tool_mode:
            if tool_call_end_at != -1:
                break

    if not tool_mode:
        tool_calls = None
        finish_reason = reason
        content = output_buffer
    else:
        finish_reason = "tool_calls"

    return CreateChatCompletionResponse.model_construct(
        id="resp-" + req_id,
        choices=[
            Choice.model_construct(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionResponseMessage.model_construct(
                    content=content,
                    tool_calls=tool_calls,
                    role="assistant",
                ),
            )
        ],
        created=int(time.time()),
        model=body.model,
        usage=usage,
    )
