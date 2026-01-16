from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union, Dict

from pydantic import BaseModel, Field, RootModel

from ..common.schema import ModelIds


class MessageContentText(BaseModel):
    text: str = Field(...)
    type: Literal["text"] = "text"


class MessageContentThinking(BaseModel):
    thinking: str = Field(...)
    type: Literal["thinking"] = "thinking"


class MessageContentToolUse(BaseModel):
    id: str = ""
    input: Dict = Field(...)
    name: str = ""
    type: Literal["tool_use"] = "tool_use"


class MessageContentToolResult(BaseModel):
    tool_use_id: str = ""
    type: Literal["tool_result"] = "tool_result"
    content: Optional[str | Dict] = None
    is_error: Optional[bool] = False

class MessageContent(RootModel[Any]):
    root: Union[
        MessageContentText,
        MessageContentThinking,
        MessageContentToolUse,
        MessageContentToolResult,
    ]


class Message(BaseModel):
    content: Union[str, MessageContent, List[MessageContent]] = Field(...)
    role: Literal["user", "assistant"] = Field(
        ..., description="The role of the messages author."
    )
    def dump_openai(self):
        def _dump_content(content: str | MessageContent):
            if isinstance(content, str):
                return {"role": self.role, "content": content}
            content = content.root
            if isinstance(content, MessageContentText):
                return {"role": self.role, "content": content.text}
            elif isinstance(content, MessageContentThinking):
                return {"role": self.role, "content": content.thinking}
            elif isinstance(content, MessageContentToolUse):
                return {
                    "role": self.role,
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": content.input,
                            },
                        }
                    ],
                }
            elif isinstance(content, MessageContentToolResult):
                return {"role": "tool", "content": str(content.content)}
            else:
                raise TypeError(f"invalid content type: {type(content)}")

        if isinstance(self.content, list):
            res = [_dump_content(c) for c in self.content]
        else:
            res = [_dump_content(self.content)]
        return res


class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool", "none"] = Field(...)
    name: Optional[str] = Field(...)
    disable_parallel_tool_use: Optional[bool] = Field(...)


class Tool(BaseModel):
    name: str = Field(...)
    description: Optional[str] = Field(...)
    input_schema: Dict = Field(...)
    def dump_openai(self):
        """
        Convert Anthropic API tool format to OpenAI API tool format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class CreateMessagesRequest(BaseModel):
    model: Union[ModelIds, str] = Field(
        ...
    )
    messages: List[Message] = Field(
        ...,
        min_length=1,
    )
    max_tokens: Optional[int] = Field(
        None,
        description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).\n",
    )

    stream: Optional[bool] = Field(
        False,
        description="If set to true, the model response data will be streamed to the client\nas it is generated using [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).\nSee the [Streaming section below](https://platform.openai.com/docs/api-reference/chat/streaming)\nfor more information, along with the [streaming responses](https://platform.openai.com/docs/guides/streaming-responses)\nguide for more information on how to handle the streaming events.\n",
    )

    system: Optional[str | MessageContentText | List[MessageContentText]] = None

    temperature: Annotated[
        Optional[float],
        Field(
            description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.\nWe generally recommend altering this or `top_p` but not both.\n",
            ge=0.0,
            le=1.0,
        ),
    ] = 1

    thinking: Optional[bool] = Field(False)

    tool_choice: Optional[ToolChoice] = Field(None)

    tools: Optional[List[Tool]] = Field(None)

    top_k: Annotated[
        Optional[int],
        Field(
            description="An integer between 0 and 20 specifying the number of most likely tokens to\nreturn at each token position, each with an associated log probability.\n",
            ge=0,
        ),
    ] = None
    top_p: Annotated[
        Optional[float],
        Field(
            description="An alternative to sampling with temperature, called nucleus sampling,\nwhere the model considers the results of the tokens with top_p probability\nmass. So 0.1 means only the tokens comprising the top 10% probability mass\nare considered.\n\nWe generally recommend altering this or `temperature` but not both.\n",
            ge=0.0,
            le=1.0,
        ),
    ] = 1
    beta: Optional[bool] = None


class Usage(BaseModel):
    cache_creation_input_tokens: Optional[int] = Field(
        ...,
        description="The number of input tokens used to create the cache entry."
    )
    cache_read_input_tokens: Optional[int] = Field(
        ...,
        description="The number of input tokens read from the cache."
    )
    input_tokens: int = Field(
        ...,
        description="The number of input tokens which were used."
    )
    output_tokens: int = Field(
        ...,
        description="The number of output tokens which were used."
    )


class MessagesResponse(BaseModel):
    id: str = Field(...)
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[Union[
        MessageContentText,
        MessageContentThinking,
        MessageContentToolUse,
    ]] = Field(...)
    model: str = Field(...)
    stop_reason: Optional[Literal[
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "refusal"
    ]] = None

    usage: Usage = Field(...)

class MessageStart(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessagesResponse = Field(...)

class MessageDeltaDelta(BaseModel):
    stop_reason: Optional[Literal[
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "refusal"
    ]] = None
    stop_sequence: Optional[str] = None

class MessageDelta(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: MessagesResponse = Field(...)
    usage: Optional[Usage] = Field(...)

class MessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"

class ContentBlockStartText(BaseModel):
    type: Literal["text"] = "text"
    text: str = Field("")

class ContentBlockStartToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(...)
    name: str = Field(...)
    input: Dict = Field({})

class ContentBlockStart(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int = Field(...)
    content_block: Union[ContentBlockStartText, ContentBlockStartToolUse] = Field(...)

class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str = Field(...)

class InputJSONDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str = Field(...)

class ContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int = Field(...)
    delta: Union[TextDelta, InputJSONDelta] = Field(...)

class ContentBlockStop(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int = Field(...)
    


