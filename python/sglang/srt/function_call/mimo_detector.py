# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ast
import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


def _normalize_tool_name_for_lookup(tool_name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", tool_name.casefold())


def _get_param_type(func_name: str, param_name: str, tools: List[Tool]) -> str:
    """Get parameter type from tool schema."""
    for tool in tools:
        if tool.function.name == func_name:
            props = tool.function.parameters.get("properties", {})
            if param_name in props:
                return props[param_name].get("type", "string")
    return "string"


def _convert_param_value(
    param_value: str, param_name: str, func_name: str, tools: List[Tool]
) -> Any:
    """
    Convert parameter value based on its type in the schema.
    Adapted from vllm-project/vllm (vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py)
    """
    param_value = html.unescape(param_value)

    # Handle null value for any type
    if param_value.lower() == "null":
        return None

    param_type = _get_param_type(func_name, param_name, tools)

    if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("integer")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        try:
            return int(param_value)
        except (ValueError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not an "
                "integer in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        try:
            float_param_value = float(param_value)
            return (
                float_param_value
                if float_param_value - int(float_param_value) != 0
                else int(float_param_value)
            )
        except (ValueError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a float "
                "in tool '%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
            return param_value
    elif param_type in ["boolean", "bool", "binary"]:
        param_value = param_value.lower()
        if param_value not in ["true", "false"]:
            logger.warning(
                "Parsed value '%s' of parameter '%s' is not a boolean "
                "(`true` or `false`) in tool '%s', degenerating to "
                "false.",
                param_value,
                param_name,
                func_name,
            )
        return param_value == "true"
    else:
        if (
            param_type in ["object", "array", "arr"]
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                param_value = json.loads(param_value)
                return param_value
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "Parsed value '%s' of parameter '%s' cannot be "
                    "parsed with json.loads in tool '%s', will try "
                    "other methods to parse it.",
                    param_value,
                    param_name,
                    func_name,
                )
        try:
            param_value = ast.literal_eval(param_value)  # safer
        except (ValueError, SyntaxError, TypeError):
            logger.warning(
                "Parsed value '%s' of parameter '%s' cannot be "
                "converted via Python `ast.literal_eval()` in tool "
                "'%s', degenerating to string.",
                param_value,
                param_name,
                func_name,
            )
        return param_value


class MiMoDetector(BaseFormatDetector):
    """
    Detector for MiMo function call format.

    Format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>pwd && ls</parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.func_regex = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
        self.param_regex = re.compile(
            r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL
        )
        self.function_token = "<function="
        self.parameter_token = "<parameter="
        self.end_function_token = "</function>"
        self.end_parameter_token = "</parameter>"
        self._reset_streaming_tool_state()

    def _get_normalized_tool_names(self, tools: List[Tool]) -> Dict[str, str]:
        normalized_names = {}
        ambiguous_names = set()
        for tool in tools:
            tool_name = tool.function.name
            if not tool_name:
                continue

            normalized_name = _normalize_tool_name_for_lookup(tool_name)
            if (
                normalized_name in normalized_names
                and normalized_names[normalized_name] != tool_name
            ):
                ambiguous_names.add(normalized_name)
                continue

            normalized_names[normalized_name] = tool_name

        for normalized_name in ambiguous_names:
            normalized_names.pop(normalized_name, None)
        return normalized_names

    def _resolve_tool_name(
        self,
        func_name: str,
        tools: List[Tool],
        tool_indices: Optional[Dict[str, int]] = None,
    ) -> str:
        tool_indices = tool_indices or self._get_tool_indices(tools)
        if func_name in tool_indices:
            return func_name

        normalized_name = _normalize_tool_name_for_lookup(func_name)
        return self._get_normalized_tool_names(tools).get(normalized_name, func_name)

    def _tool_call_items_from_parsed(
        self, parsed: Dict[str, Any], tool_indices: Dict[str, int]
    ) -> List[ToolCallItem]:
        func_name = parsed.get("name")
        if not func_name:
            return []

        return [
            ToolCallItem(
                tool_index=tool_indices.get(func_name, -1),
                name=func_name,
                parameters=json.dumps(parsed.get("parameters", {}), ensure_ascii=False),
            )
        ]

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse complete text for tool calls."""
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx]
        tool_indices = self._get_tool_indices(tools)

        calls = []

        for match in self.tool_call_regex.finditer(text):
            tool_call_body = match.group(1)

            parsed = self._parse_tool_call(tool_call_body, tools)

            if parsed:
                func_name = parsed.get("name")
                if func_name not in tool_indices:
                    logger.debug("Forwarding unknown MiMo function: %s", func_name)
                calls.extend(self._tool_call_items_from_parsed(parsed, tool_indices))

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Parse MiMo tool-call markup incrementally.

        MiMo emits XML-ish tags that can be split across arbitrary chunks.  The
        streaming API should not expose those tags as content, but it also
        should not wait for the full ``</tool_call>`` block before exposing the
        tool name and argument bytes.
        """
        self._buffer += new_text
        normal_text = ""
        calls = []

        while self._buffer:
            if not self._streaming_in_tool_call:
                start = self._buffer.find(self.bot_token)
                if start == -1:
                    keep = self._partial_suffix_len(self._buffer, [self.bot_token])
                    emit_len = len(self._buffer) - keep
                    normal_text += self._buffer[:emit_len]
                    self._buffer = self._buffer[emit_len:]
                    break

                normal_text += self._buffer[:start]
                self._raw_tool_call_buffer = self.bot_token
                self._buffer = self._buffer[start + len(self.bot_token) :]
                self._streaming_in_tool_call = True
                continue

            if not self._streaming_known_tool_call:
                marker_idx, marker = self._find_first_marker(
                    self._buffer, [self.function_token, self.eot_token]
                )
                if marker_idx == -1:
                    keep = self._partial_suffix_len(
                        self._buffer, [self.function_token, self.eot_token]
                    )
                    consume_len = len(self._buffer) - keep
                    self._raw_tool_call_buffer += self._buffer[:consume_len]
                    self._buffer = self._buffer[consume_len:]
                    break

                self._raw_tool_call_buffer += self._buffer[:marker_idx]
                self._buffer = self._buffer[marker_idx:]

                if marker == self.eot_token:
                    self._raw_tool_call_buffer += self.eot_token
                    normal_text += self._raw_tool_call_buffer
                    self._buffer = self._buffer[len(self.eot_token) :]
                    self._reset_streaming_tool_state()
                    continue

                close = self._buffer.find(">", len(self.function_token))
                if close == -1:
                    break

                self._raw_tool_call_buffer += self._buffer[: close + 1]
                raw_func_name = self._buffer[len(self.function_token) : close].strip()
                self._buffer = self._buffer[close + 1 :]
                self._start_streaming_function(raw_func_name, tools, calls)
                continue

            if self._streaming_param_name is not None:
                marker_idx, marker = self._find_first_marker(
                    self._buffer,
                    [
                        self.end_parameter_token,
                        self.end_function_token,
                        self.eot_token,
                    ],
                )
                if marker_idx == -1:
                    keep = self._partial_suffix_len(
                        self._buffer,
                        [
                            self.end_parameter_token,
                            self.end_function_token,
                            self.eot_token,
                        ],
                    )
                    consume_len = len(self._buffer) - keep
                    value_text = self._buffer[:consume_len]
                    self._raw_tool_call_buffer += value_text
                    self._append_streaming_param_value(value_text, calls)
                    self._buffer = self._buffer[consume_len:]
                    break

                value_text = self._buffer[:marker_idx]
                self._raw_tool_call_buffer += value_text
                self._append_streaming_param_value(value_text, calls)
                self._buffer = self._buffer[marker_idx:]

                if marker == self.end_parameter_token:
                    self._raw_tool_call_buffer += self.end_parameter_token
                    self._buffer = self._buffer[len(self.end_parameter_token) :]
                    self._finish_streaming_param(tools, calls)
                else:
                    # Malformed but common enough during constrained decoding:
                    # close the parameter implicitly and let the outer-state
                    # branch consume </function> or </tool_call>.
                    self._finish_streaming_param(tools, calls)
                continue

            marker_idx, marker = self._find_first_marker(
                self._buffer,
                [self.parameter_token, self.end_function_token, self.eot_token],
            )
            if marker_idx == -1:
                keep = self._partial_suffix_len(
                    self._buffer,
                    [self.parameter_token, self.end_function_token, self.eot_token],
                )
                consume_len = len(self._buffer) - keep
                self._raw_tool_call_buffer += self._buffer[:consume_len]
                self._buffer = self._buffer[consume_len:]
                break

            self._raw_tool_call_buffer += self._buffer[:marker_idx]
            self._buffer = self._buffer[marker_idx:]

            if marker == self.parameter_token:
                close = self._buffer.find(">", len(self.parameter_token))
                if close == -1:
                    break

                self._raw_tool_call_buffer += self._buffer[: close + 1]
                param_name = self._buffer[len(self.parameter_token) : close].strip()
                self._buffer = self._buffer[close + 1 :]
                self._start_streaming_param(param_name, tools, calls)
                continue

            if marker == self.end_function_token:
                self._raw_tool_call_buffer += self.end_function_token
                self._buffer = self._buffer[len(self.end_function_token) :]
                self._finish_streaming_arguments(tools, calls)
                continue

            self._raw_tool_call_buffer += self.eot_token
            self._buffer = self._buffer[len(self.eot_token) :]
            self._finish_streaming_tool_call(tools, calls)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    @staticmethod
    def _find_first_marker(text: str, markers: Sequence[str]) -> Tuple[int, str]:
        best_idx = -1
        best_marker = ""
        for marker in markers:
            idx = text.find(marker)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_marker = marker
        return best_idx, best_marker

    @staticmethod
    def _partial_suffix_len(text: str, markers: Sequence[str]) -> int:
        keep = 0
        for marker in markers:
            max_len = min(len(text), len(marker) - 1)
            for length in range(1, max_len + 1):
                if marker.startswith(text[-length:]):
                    keep = max(keep, length)
        return keep

    @staticmethod
    def _json_string_fragment(text: str) -> str:
        return json.dumps(text, ensure_ascii=False)[1:-1]

    @staticmethod
    def _streams_as_string(param_type: str) -> bool:
        return param_type in ["string", "str", "text", "varchar", "char", "enum"]

    def _reset_streaming_tool_state(self) -> None:
        self._streaming_in_tool_call = False
        self._streaming_known_tool_call = False
        self._streaming_func_name = ""
        self._streaming_param_name = None
        self._streaming_param_value = ""
        self._streaming_param_as_string = True
        self._streaming_json_args_started = False
        self._streaming_json_args_closed = False
        self._streaming_json_param_open = False
        self._raw_tool_call_buffer = ""

    def _ensure_streaming_tool_slot(self, func_name: str) -> int:
        if self.current_tool_id == -1:
            self.current_tool_id = 0

        while len(self.prev_tool_call_arr) <= self.current_tool_id:
            self.prev_tool_call_arr.append({})
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }
        self.streamed_args_for_tool[self.current_tool_id] = ""
        return self.current_tool_id

    def _append_streaming_arguments_chunk(self, chunk: str, calls: List[Any]) -> None:
        if not chunk:
            return

        tool_index = self.current_tool_id
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")
        self.streamed_args_for_tool[tool_index] += chunk
        calls.append(ToolCallItem(tool_index=tool_index, parameters=chunk))

    def _start_streaming_function(
        self, func_name: str, tools: List[Tool], calls: List[Any]
    ) -> None:
        tool_indices = self._get_tool_indices(tools)
        func_name = self._resolve_tool_name(func_name, tools, tool_indices)
        if func_name not in tool_indices:
            logger.debug("Forwarding unknown MiMo function: %s", func_name)

        tool_index = self._ensure_streaming_tool_slot(func_name)
        self._streaming_known_tool_call = True
        self._streaming_func_name = func_name
        self.current_tool_name_sent = True
        calls.append(ToolCallItem(tool_index=tool_index, name=func_name, parameters=""))

    def _start_streaming_param(
        self, param_name: str, tools: List[Tool], calls: List[Any]
    ) -> None:
        if self._streaming_json_args_closed:
            return

        self._streaming_param_name = param_name
        self._streaming_param_value = ""
        param_type = _get_param_type(self._streaming_func_name, param_name, tools)
        self._streaming_param_as_string = self._streams_as_string(param_type)
        self._streaming_json_param_open = self._streaming_param_as_string

        arguments = self.prev_tool_call_arr[self.current_tool_id]["arguments"]
        arguments[param_name] = ""

        prefix = ", " if self._streaming_json_args_started else "{"
        prefix += f"{json.dumps(param_name, ensure_ascii=False)}: "
        if self._streaming_param_as_string:
            prefix += '"'
        self._streaming_json_args_started = True
        self._append_streaming_arguments_chunk(prefix, calls)

    def _append_streaming_param_value(self, value_text: str, calls: List[Any]) -> None:
        if not value_text or self._streaming_param_name is None:
            return

        self._streaming_param_value += value_text
        if self._streaming_param_as_string:
            self.prev_tool_call_arr[self.current_tool_id]["arguments"][
                self._streaming_param_name
            ] = self._streaming_param_value
            self._append_streaming_arguments_chunk(
                self._json_string_fragment(value_text), calls
            )

    def _finish_streaming_param(self, tools: List[Tool], calls: List[Any]) -> None:
        if self._streaming_param_name is None:
            return

        param_name = self._streaming_param_name
        param_value = self._streaming_param_value

        if self._streaming_param_as_string:
            self.prev_tool_call_arr[self.current_tool_id]["arguments"][param_name] = (
                param_value
            )
            if self._streaming_json_param_open:
                self._append_streaming_arguments_chunk('"', calls)
        else:
            converted_value = _convert_param_value(
                param_value, param_name, self._streaming_func_name, tools
            )
            self.prev_tool_call_arr[self.current_tool_id]["arguments"][param_name] = (
                converted_value
            )
            self._append_streaming_arguments_chunk(
                json.dumps(converted_value, ensure_ascii=False), calls
            )

        self._streaming_param_name = None
        self._streaming_param_value = ""
        self._streaming_param_as_string = True
        self._streaming_json_param_open = False

    def _finish_streaming_arguments(self, tools: List[Tool], calls: List[Any]) -> None:
        if self._streaming_json_args_closed:
            return

        if self._streaming_param_name is not None:
            self._finish_streaming_param(tools, calls)

        if self._streaming_json_args_started:
            self._append_streaming_arguments_chunk("}", calls)
        else:
            self._append_streaming_arguments_chunk("{}", calls)

        self._streaming_json_args_closed = True

    def _finish_streaming_tool_call(self, tools: List[Tool], calls: List[Any]) -> None:
        self._finish_streaming_arguments(tools, calls)
        self.current_tool_name_sent = False
        if self.current_tool_id == -1:
            self.current_tool_id = 0
        self.current_tool_id += 1
        self._reset_streaming_tool_state()

    def _parse_tool_call(
        self, tool_call_body: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """
        Parse content inside <tool_call>...</tool_call>.

        Structure:
            tool_call_body contains: <function=name>...params...</function>
        """
        # Match complete <function=name>body</function> block
        func_match = self.func_regex.search(tool_call_body)
        if not func_match:
            return None

        func_name = func_match.group(1).strip()
        func_name = self._resolve_tool_name(func_name, tools)
        func_body = func_match.group(2)

        params = {}
        for param_match in self.param_regex.finditer(func_body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2)
            params[param_name] = _convert_param_value(
                param_value, param_name, func_name, tools
            )

        return {"name": func_name, "parameters": params}

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
