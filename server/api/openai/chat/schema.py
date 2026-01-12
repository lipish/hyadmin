from __future__ import annotations

from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, RootModel

from ...common.schema import ModelIds


class FunctionParameters(RootModel[Any]):
    root: Any


class FunctionObject(BaseModel):
    description: Optional[str] = Field(
        None,
        description="A description of what the function does, used by the model to choose when and how to call the function.",
    )
    name: str = Field(
        ...,
        description="The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.",
    )
    parameters: Optional[FunctionParameters] = None
    strict: Optional[bool] = Field(
        False,
        description="Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the `parameters` field. Only a subset of JSON Schema is supported when `strict` is `true`. Learn more about Structured Outputs in the [function calling guide](https://platform.openai.com/docs/guides/function-calling).",
    )


class ChatCompletionTool(BaseModel):
    type: Literal["function"] = Field(
        ...,
        description="The type of the tool. Currently, only `function` is supported.",
    )
    function: FunctionObject


class ChatCompletionRequestMessage(BaseModel):
    content: str = Field(...)
    role: Literal["developer", "system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the messages author."
    )
    name: Optional[str] = Field(
        None,
        description="An optional name for the participant. Provides the model information to differentiate between participants of the same role.",
    )


class CreateChatCompletionRequest(BaseModel):
    messages: List[ChatCompletionRequestMessage] = Field(
        ...,
        description="A list of messages comprising the conversation so far. Depending on the\n[model](https://platform.openai.com/docs/models) you use, different message types (modalities) are\nsupported, like [text](https://platform.openai.com/docs/guides/text-generation),\n[images](https://platform.openai.com/docs/guides/vision), and [audio](https://platform.openai.com/docs/guides/audio).\n",
        min_length=1,
    )
    model: Union[ModelIds, str] = Field(
        ...,
        description="Model ID used to generate the response, like `gpt-4o` or `o3`. OpenAI\noffers a wide range of models with different capabilities, performance\ncharacteristics, and price points. Refer to the [model guide](https://platform.openai.com/docs/models)\nto browse and compare available models.\n",
    )

    max_completion_tokens: Annotated[
        Optional[int],
        Field(
            description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).\n",
            ge=0,
        ),
    ] = None

    max_new_tokens: Annotated[
        Optional[int],
        Field(
            description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).\n",
            ge=0,
        ),
    ] = None

    max_tokens: Annotated[
        Optional[int],
        Field(
            description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).\n",
            ge=0,
        ),
    ] = None

    top_logprobs: Annotated[
        Optional[int],
        Field(
            description="An integer between 0 and 20 specifying the number of most likely tokens to\nreturn at each token position, each with an associated log probability.\n",
            ge=0,
        ),
    ] = None

    top_k: Annotated[
        Optional[int],
        Field(
            description="An integer between 0 and 20 specifying the number of most likely tokens to\nreturn at each token position, each with an associated log probability.\n",
            ge=0,
        ),
    ] = None

    temperature: Annotated[
        Optional[float],
        Field(
            description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.\nWe generally recommend altering this or `top_p` but not both.\n",
            ge=0.0,
            le=2.0,
        ),
    ] = 1
    top_p: Annotated[
        Optional[float],
        Field(
            description="An alternative to sampling with temperature, called nucleus sampling,\nwhere the model considers the results of the tokens with top_p probability\nmass. So 0.1 means only the tokens comprising the top 10% probability mass\nare considered.\n\nWe generally recommend altering this or `temperature` but not both.\n",
            ge=0.0,
            le=1.0,
        ),
    ] = 1

    stream: Optional[bool] = Field(
        False,
        description="If set to true, the model response data will be streamed to the client\nas it is generated using [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).\nSee the [Streaming section below](https://platform.openai.com/docs/api-reference/chat/streaming)\nfor more information, along with the [streaming responses](https://platform.openai.com/docs/guides/streaming-responses)\nguide for more information on how to handle the streaming events.\n",
    )
    logprobs: Optional[bool] = Field(
        False,
        description="Whether to return log probabilities of the output tokens or not. If true,\nreturns the log probabilities of each output token returned in the\n`content` of `message`.\n",
    )
    tools: Optional[List[ChatCompletionTool]] = Field(
        None,
        description="A list of tools the model may call. You can provide either\n[custom tools](https://platform.openai.com/docs/guides/function-calling#custom-tools) or\n[function tools](https://platform.openai.com/docs/guides/function-calling).\n",
    )
    tool_choice: Optional[Literal["none", "auto", "required"]] = None
    parallel_tool_calls: Optional[bool] = True


class Function(BaseModel):
    name: Optional[str] = Field(None, description="The name of the function to call.")
    arguments: Optional[str] = Field(
        None,
        description="The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.",
    )


class ChatCompletionMessageToolCall(BaseModel):
    id: str = Field(..., description="The ID of the tool call.")
    type: Literal["function"] = Field(
        ...,
        description="The type of the tool. Currently, only `function` is supported.",
    )
    function: Function = Field(..., description="The function that the model called.")


class ChatCompletionResponseMessage(BaseModel):
    content: str = Field(..., description="The contents of the message.")
    refusal: str = Field(..., description="The refusal message generated by the model.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    role: Literal["assistant"] = Field(
        ..., description="The role of the author of this message."
    )


class Choice(BaseModel):
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ] = Field(
        ...,
        description="The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,\n`length` if the maximum number of tokens specified in the request was reached,\n`content_filter` if content was omitted due to a flag from our content filters,\n`tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function.\n",
    )
    index: int = Field(
        ..., description="The index of the choice in the list of choices."
    )
    message: ChatCompletionResponseMessage


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: Annotated[
        Optional[int],
        Field(
            description="When using Predicted Outputs, the number of tokens in the\nprediction that appeared in the completion.\n"
        ),
    ] = 0
    audio_tokens: Annotated[
        Optional[int], Field(description="Audio input tokens generated by the model.")
    ] = 0
    reasoning_tokens: Annotated[
        Optional[int], Field(description="Tokens generated by the model for reasoning.")
    ] = 0
    rejected_prediction_tokens: Annotated[
        Optional[int],
        Field(
            description="When using Predicted Outputs, the number of tokens in the\nprediction that did not appear in the completion. However, like\nreasoning tokens, these tokens are still counted in the total\ncompletion tokens for purposes of billing, output, and context window\nlimits.\n"
        ),
    ] = 0


class PromptTokensDetails(BaseModel):
    audio_tokens: Annotated[
        Optional[int], Field(description="Audio input tokens present in the prompt.")
    ] = 0
    cached_tokens: Annotated[
        Optional[int], Field(description="Cached tokens present in the prompt.")
    ] = 0


class CompletionUsage(BaseModel):
    completion_tokens: Annotated[
        int, Field(description="Number of tokens in the generated completion.")
    ]
    prompt_tokens: Annotated[int, Field(description="Number of tokens in the prompt.")]
    total_tokens: Annotated[
        int,
        Field(
            description="Total number of tokens used in the request (prompt + completion)."
        ),
    ]
    completion_tokens_details: Annotated[
        Optional[CompletionTokensDetails],
        Field(description="Breakdown of tokens used in a completion."),
    ] = None
    prompt_tokens_details: Annotated[
        Optional[PromptTokensDetails],
        Field(description="Breakdown of tokens used in the prompt."),
    ] = None


class CreateChatCompletionResponse(BaseModel):
    id: str = Field(..., description="A unique identifier for the chat completion.")
    choices: List[Choice] = Field(
        ...,
        description="A list of chat completion choices. Can be more than one if `n` is greater than 1.",
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the chat completion was created.",
    )
    model: str = Field(..., description="The model used for the chat completion.")
    object: Literal["chat.completion"] = Field(
        ..., description="The object type, which is always `chat.completion`."
    )
    usage: Optional[CompletionUsage] = None


class ChatCompletionMessageToolCallChunk(BaseModel):
    index: int
    id: Optional[str] = Field(None, description="The ID of the tool call.")
    type: Optional[Literal["function"]] = Field(
        None,
        description="The type of the tool. Currently, only `function` is supported.",
    )
    function: Optional[Function] = None


class ChatCompletionStreamResponseDelta(BaseModel):
    content: Optional[str] = Field(
        None, description="The contents of the chunk message."
    )
    tool_calls: Optional[List[ChatCompletionMessageToolCallChunk]] = None
    role: Optional[Literal["developer", "system", "user", "assistant", "tool"]] = Field(
        None, description="The role of the author of this message."
    )
    refusal: Optional[str] = Field(
        None, description="The refusal message generated by the model."
    )


class Choice1(BaseModel):
    delta: ChatCompletionStreamResponseDelta
    finish_reason: Literal[
        "stop", "length", "tool_calls", "content_filter", "function_call"
    ] = Field(
        ...,
        description="The reason the model stopped generating tokens. This will be `stop` if the model hit a natural stop point or a provided stop sequence,\n`length` if the maximum number of tokens specified in the request was reached,\n`content_filter` if content was omitted due to a flag from our content filters,\n`tool_calls` if the model called a tool, or `function_call` (deprecated) if the model called a function.\n",
    )
    index: int = Field(
        ..., description="The index of the choice in the list of choices."
    )


class CreateChatCompletionStreamResponse(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the chat completion. Each chunk has the same ID.",
    )
    choices: List[Choice1] = Field(
        ...,
        description='A list of chat completion choices. Can contain more than one elements if `n` is greater than 1. Can also be empty for the\nlast chunk if you set `stream_options: {"include_usage": true}`.\n',
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has the same timestamp.",
    )
    model: str = Field(..., description="The model to generate the completion.")
    object: Literal["chat.completion.chunk"] = Field(
        ..., description="The object type, which is always `chat.completion.chunk`."
    )
    usage: Optional[CompletionUsage] = Field(
        None,
        description='An optional field that will only be present when you set\n`stream_options: {"include_usage": true}` in your request. When present, it\ncontains a null value **except for the last chunk** which contains the\ntoken usage statistics for the entire request.\n\n**NOTE:** If the stream is interrupted or cancelled, you may not\nreceive the final usage chunk which contains the total token usage for\nthe request.\n',
    )
