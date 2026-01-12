import json
import re
from typing import List, NamedTuple, Tuple
from uuid import uuid4

from .schema import (
    ChatCompletionMessageToolCall,
    CreateChatCompletionRequest,
    Function,
    ModelIds,
)

ToolCall = NamedTuple("ToolCall", [("name", str), ("arguments", str)])


def deepseek_v31_tool_parser(s: str) -> Tuple[int, int, List[ToolCall]]:
    pattern = r"<｜tool▁call▁begin｜>(?P<function_name>.*?)<｜tool▁sep｜>(?P<function_arguments>.*?)<｜tool▁call▁end｜>"
    botc_token, eotc_token = (r"<｜tool▁calls▁begin｜>", r"<｜tool▁calls▁end｜>")
    tool_calls = [ToolCall(*match) for match in re.findall(pattern, s, re.DOTALL)]
    return (s.find(botc_token), s.find(eotc_token), tool_calls)


def kimi_k2_tool_parser(s: str) -> Tuple[int, int, List[ToolCall]]:
    botc_token, eotc_token = (
        r"<|tool_calls_section_begin|>",
        r"<|tool_calls_section_end|>",
    )

    if botc_token not in s:
        # No tool calls
        return -1, -1, []

    # pattern = r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>"

    # tool_calls_sections = re.findall(pattern, s, re.DOTALL)

    # Extract multiple tool calls
    func_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*?)\s*<\|tool_call_end\|>"
    tool_calls = []
    for match in re.findall(func_call_pattern, s, re.DOTALL):
        function_id, function_args = match
        # function_id: functions.get_weather:0
        function_name = function_id.split(".")[1].split(":")[0]
        tool_calls.append(ToolCall(function_name, function_args))
    return (s.find(botc_token), s.find(eotc_token), tool_calls)


def qwen3_moe_tool_parser(s: str) -> Tuple[int, int, List[ToolCall]]:
    botc_token, eotc_token = r"<tool_call>", r"</tool_call>"
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    matches = tool_call_regex.findall(s)
    tool_calls: List[ToolCall] = []
    for match in matches:
        tool_call_obj = json.loads(match)
        name = tool_call_obj.get("name")
        arguments = json.dumps(tool_call_obj.get("arguments"))
        tool_calls.append(ToolCall(name, arguments))
    return (s.find(botc_token), s.find(eotc_token), tool_calls)


def qwen3_coder_tool_parser(s: str) -> Tuple[int, int, List[ToolCall]]:
    tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    function_regex = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
    parameter_regex = re.compile(
        r"<parameter=(.*?)>(.*?)</parameter>",
        re.DOTALL,
    )
    tool_call_ranges = tool_call_regex.findall(s)

    tool_calls = []
    for tc_range in tool_call_ranges:
        for name, func_range in function_regex.findall(tc_range):
            arguments = {}
            for param_name, param_content in parameter_regex.findall(func_range):
                arguments[param_name] = param_content
            tc = ToolCall(name, json.dumps(arguments))
            tool_calls.append(tc)

    botc_token, eotc_token = r"<tool_call>", r"</tool_call>"
    return (s.find(botc_token), s.find(eotc_token), tool_calls)


# "GLM-4.5": re.compile(r"<tool_call>(?P<function_name>.+?)(?:\n<arg_key>(?P<arg_key>.+?)<\/arg_key>\s*<arg_value>(?P<arg_value>.+?)<\/arg_value>)+<\/tool_call>"),


def extract_tool_calls(
    output_buffer: str, req: CreateChatCompletionRequest
) -> Tuple[int, int, List[ChatCompletionMessageToolCall]]:

    if req.model in [ModelIds.DeepSeek_R1, ModelIds.DeepSeek_V3, ModelIds.DeepSeek_V31]:
        parser = deepseek_v31_tool_parser
    elif req.model in [ModelIds.Kimi_K2]:
        parser = kimi_k2_tool_parser
    elif req.model in [ModelIds.Qwen3_235B_A22B, ModelIds.Qwen3_30B_A3B]:
        parser = qwen3_moe_tool_parser
    elif req.model in [ModelIds.Qwen3_Coder_480B_A35B, ModelIds.Qwen3_Coder_30B_A3B]:
        parser = qwen3_coder_tool_parser
    else:
        parser = qwen3_moe_tool_parser

    botc_at, eotc_at, tc_matches = parser(output_buffer)

    tool_calls = [
        ChatCompletionMessageToolCall(
            id="tool-" + uuid4().hex,
            type="function",
            function=Function(name=name, arguments=arguments),
        )
        for name, arguments in tc_matches
    ]

    return (botc_at, eotc_at, tool_calls)
