from __future__ import annotations

import asyncio
from typing import Any, AsyncIterable, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from heyi.engine import Engine
from heyi.server.config import Config
from heyi.utils.log import logger
from heyi.utils.request import Request as EngineRequest
from heyi.utils.usage import Usage as HeyiUsage

from ..openai.chat.tool_call_utils import extract_tool_calls
from .schema import (
    CreateMessagesRequest,
    Message,
    MessageContentText,
    MessageContentToolUse,
    MessagesResponse,
    MessageStart,
    MessageDelta,
    MessageStop,
    ContentBlockStart,
    ContentBlockDelta,
    ContentBlockStop,
    TextDelta,
    InputJSONDelta,
    ContentBlockStartText,
    ContentBlockStartToolUse,
    Usage,
    ModelIds,
)

router = APIRouter()


async def to_sse(events: AsyncIterable[Any]) -> AsyncIterable[str]:
    async for e in events:
        if isinstance(e, str):
            # print(e)
            yield e
        elif isinstance(e, BaseModel):
            yield f"event: {e.type}\ndata: {e.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


def streamer(request: CreateMessagesRequest, req: EngineRequest):
    async def _streamer():
        output_buffer = ""
        tool_mode = False
        cb_idx = 0 # index of current content block
        reason = None
        usage = None

        message_started = False

        try:
            async for res in req.stream.generator():
                if isinstance(res, HeyiUsage):
                    usage = Usage.model_construct(
                        cache_creation_input_tokens = res.prompt_tokens - res.cache_hit_tokens,
                        cache_read_input_tokens = res.cache_hit_tokens,
                        input_tokens = res.prompt_tokens,
                        output_tokens = res.completion_tokens
                    )
                    continue

                # import pdb; pdb.set_trace()
                if not message_started:
                    message_started = True
                    msg_start = MessageStart.model_construct(
                        message=MessagesResponse.model_construct(
                            id="msg_" + uuid4().hex,
                            model=request.model,
                            content=[],
                            usage=usage,
                        )
                    )
                    yield msg_start

                    yield ContentBlockStart.model_construct(
                        index=cb_idx, content_block=ContentBlockStartText.model_construct(text="")
                    )


                # if isinstance(res, RawUsage):
                #     chunk.usage = CompletionUsage(
                #         prompt_tokens=res.prefill_count,
                #         completion_tokens=res.decode_count,
                #         total_tokens=res.prefill_count + res.decode_count,
                #     )
                #     if create.return_speed:
                #         chunk.usage.prefill_time = res.prefill_time
                #         chunk.usage.decode_time = res.decode_time
                #     yield chunk
                #     continue


                token, reason = res if isinstance(res, tuple) else (res, None)

                output_buffer += token
                tool_call_begin_at, tool_call_end_at, tool_calls = extract_tool_calls(
                    output_buffer, request
                )

                if tool_call_begin_at != -1 and not tool_mode:
                    tool_mode = True
                    content, output_buffer = (
                        output_buffer[:tool_call_begin_at],
                        output_buffer[tool_call_begin_at:],
                    )
                    if content:
                        yield ContentBlockDelta.model_construct(index=cb_idx, delta = TextDelta.model_construct(text=content))
                        yield ContentBlockStop.model_construct(index=cb_idx)
                    continue

                if tool_mode:
                    if tool_call_end_at != -1:

                        yield ContentBlockStop.model_construct(index=cb_idx)

                        cb_idx += 1
                        yield ContentBlockStart.model_construct(index=cb_idx, content_block=ContentBlockStartToolUse.model_construct(
                            id="toolu_" + uuid4().hex,
                            name=tool_calls[0].function.name,
                            input={},
                        ))

                        yield ContentBlockDelta.model_construct(
                            index=cb_idx,
                            delta=InputJSONDelta.model_construct(
                                partial_json=tool_calls[0].function.arguments
                            )
                        )

                        yield ContentBlockStop.model_construct(
                            index=cb_idx
                        )
                        return
                else:
                    yield ContentBlockDelta.model_construct(index=cb_idx, delta = TextDelta.model_construct(text=output_buffer))
                    output_buffer = ""

        except asyncio.CancelledError:
            logger.warning("Inference cancelled")
            req.cancel()
        finally:
            if not tool_mode:
                yield ContentBlockDelta.model_construct(index=cb_idx, delta=TextDelta.model_construct(text=""))
                yield ContentBlockStop.model_construct(index=cb_idx)
                yield MessageDelta.model_construct(
                    delta = MessagesResponse.model_construct(
                        stop_reason=reason,
                    ),
                    usage=usage
                )
            yield MessageStop.model_construct()

    return _streamer()


def system_message_dump_to_openai(system: None | str | MessageContentText | List[MessageContentText]):
    if not system:
        return None
    
    system_msg_content = None
    if isinstance(system, str):
        system_msg_content = system
    elif isinstance(system, MessageContentText):
        system_msg_content = system.text
    else: 
        system_msg_content = ""
        for m in system:
            system_msg_content += m.text

    return {"role": "system", "content": system_msg_content}


def messages_dump_to_openai(messages: List[Message]):
    res = []
    for m in messages:
        res += m.dump_openai()
    return res


@router.post(
    "/messages", 
    # response_model=MessagesResponse, 
    tags=["Messages"]
)
async def create_messages(
    # request: Request,
    body: CreateMessagesRequest,
):
    """
    Create messages
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

    system_message = system_message_dump_to_openai(body.system)
    messages = messages_dump_to_openai(body.messages)
    if system_message:
        messages = [system_message] + messages

    generation_config = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "top_k": body.top_k,
        "max_new_tokens": body.max_tokens,
    }

    if body.tools is not None:
        tools = [t.dump_openai() for t in body.tools]
    else:
        tools = None

    req = Engine().submit(req_id, messages, generation_config, tools)

    if body.stream:
        logger.warning(f"<{req_id}> STREAMING")
        # raise KeyError
        return StreamingResponse(
            to_sse(streamer(body, req)), media_type="text/event-stream"
        )
    logger.warning(f"<{req_id}> NON-STREAMING")


    # NON-STREAMING
    output_buffer = ""
    tool_mode = False

    async for res in req.stream.generator():
        if isinstance(res, HeyiUsage):
            usage = Usage.model_construct(
                cache_creation_input_tokens = res.prompt_tokens - res.cache_hit_tokens,
                cache_read_input_tokens = res.cache_hit_tokens,
                input_tokens = res.prompt_tokens,
                output_tokens = res.completion_tokens
            )
            continue

        # if isinstance(res, RawUsage):
        #     usage = CompletionUsage(
        #         prompt_tokens=res.prefill_count,
        #         completion_tokens=res.decode_count,
        #         total_tokens=res.prefill_count + res.decode_count,
        #     )
        #     if create.return_speed:
        #         usage.prefill_time = res.prefill_time
        #         usage.decode_time = res.decode_time
        #     continue

        token, reason = res if isinstance(res, tuple) else (res, None)
        output_buffer += token

        tool_call_begin_at, tool_call_end_at, tool_calls = extract_tool_calls(
            output_buffer, body
        )

        if tool_call_begin_at != -1:
            tool_mode = True
            content, output_buffer = (
                output_buffer[:tool_call_begin_at],
                output_buffer[tool_call_begin_at:],
            )
            continue

        if tool_mode:
            if tool_call_end_at != -1:
                logger.warning(f"OUTPUT BUFFER: {output_buffer}")
                logger.warning(f"TOOL CALLS: {tool_calls}")
                break

    if tool_mode: 
        tool_use_contents = [
            MessageContentToolUse.model_construct(
                id="toolu_" + uuid4().hex,
                input=tc.function.arguments,
                name=tc.function.name,
            ) for tc in tool_calls
        ]
        stop_reason = "tool_use"

    else:
        tool_use_contents = []
        content = output_buffer
        stop_reason = reason


    return MessagesResponse.model_construct(
        id="msg_resp_" + req_id,
        content=[
            MessageContentText.model_construct(
                text=content
            ),
            *tool_use_contents
        ],
        model=body.model,
        stop_reason=stop_reason,
        usage=usage
    )
