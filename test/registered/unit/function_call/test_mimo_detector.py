"""Unit tests for MiMoDetector streaming behavior."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mimo_detector import MiMoDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestMiMoDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            )
        ]
        self.detector = MiMoDetector()

    def _tool_call_text(self):
        return (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>tetris</parameter>\n"
            "<parameter=limit>3</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

    def _stream_chunks(self, chunks, detector=None):
        detector = detector or self.detector
        normal_text = []
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text.append(result.normal_text)
            calls.extend(result.calls)
        return "".join(normal_text), calls

    def _arguments_from_calls(self, calls):
        return "".join(call.parameters for call in calls if not call.name)

    def _arguments_by_tool(self, calls):
        arguments = {}
        for call in calls:
            if call.name:
                continue
            arguments.setdefault(call.tool_index, "")
            arguments[call.tool_index] += call.parameters
        return arguments

    def test_streams_normal_text_after_completed_tool_call(self):
        tool_call = (
            "<tool_call>\n"
            "<function=search>\n"
            "<parameter=query>tetris</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = self.detector.parse_streaming_increment(tool_call, self.tools)
        self.assertGreater(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(
            json.loads(self._arguments_from_calls(result.calls)), {"query": "tetris"}
        )

        result = self.detector.parse_streaming_increment(" normal text", self.tools)
        self.assertEqual(result.normal_text, " normal text")
        self.assertEqual(result.calls, [])
        self.assertEqual(self.detector._buffer, "")

    def test_keeps_only_partial_tool_start_marker(self):
        self.detector.current_tool_id = 1

        result = self.detector.parse_streaming_increment("<", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])
        self.assertEqual(self.detector._buffer, "<")

        result = self.detector.parse_streaming_increment("style>", self.tools)
        self.assertEqual(result.normal_text, "<style>")
        self.assertEqual(result.calls, [])
        self.assertEqual(self.detector._buffer, "")

    def test_streams_tool_call_before_end_tag(self):
        result = self.detector.parse_streaming_increment(
            "<tool_call>\n<function=search>\n<parameter=query>tet", self.tools
        )

        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(self._arguments_from_calls(result.calls), '{"query": "tet')

    def test_handles_split_tags_without_leaking_markup(self):
        normal_text, calls = self._stream_chunks(
            [
                "Let me. <tool_",
                "call>\n<function=se",
                "arch>\n<parameter=que",
                "ry>te",
                "tris</par",
                "ameter>\n</function>\n</tool_call> trailing",
            ]
        )

        self.assertEqual(normal_text, "Let me.  trailing")
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(
            json.loads(self._arguments_from_calls(calls)), {"query": "tetris"}
        )

    def test_converts_non_string_parameters(self):
        normal_text, calls = self._stream_chunks(
            [
                "<tool_call><function=search>",
                "<parameter=query>tetris</parameter>",
                "<parameter=limit>3</parameter>",
                "</function></tool_call>",
            ]
        )

        self.assertEqual(normal_text, "")
        self.assertEqual(
            json.loads(self._arguments_from_calls(calls)),
            {"query": "tetris", "limit": 3},
        )

    def test_streams_multiple_tool_calls_with_escaped_text(self):
        normal_text, calls = self._stream_chunks(
            [
                "<tool_call><function=search>",
                "<parameter=query>cat << 'EOF'\nprint(\"hi\")\nEOF</parameter>",
                "</function></tool_call>",
                "<tool_call><function=search>",
                "<parameter=query>second</parameter>",
                "</function></tool_call>",
            ]
        )

        argument_by_tool = self._arguments_by_tool(calls)
        self.assertEqual(normal_text, "")
        self.assertEqual(
            [call.name for call in calls if call.name], ["search", "search"]
        )
        self.assertEqual(
            json.loads(argument_by_tool[0]),
            {"query": "cat << 'EOF'\nprint(\"hi\")\nEOF"},
        )
        self.assertEqual(json.loads(argument_by_tool[1]), {"query": "second"})

    def test_unknown_tool_call_is_forwarded_streaming(self):
        tool_call = (
            "<tool_call><function=missing>"
            "<parameter=query>tetris</parameter>"
            "</function></tool_call>"
        )

        result = self.detector.parse_streaming_increment(tool_call, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls[0].name, "missing")
        self.assertEqual(
            json.loads(self._arguments_from_calls(result.calls)), {"query": "tetris"}
        )

    def test_corrects_server_style_tool_name_case_non_streaming(self):
        tool_call = (
            "<tool_call><function=Search>"
            "<parameter=query>tetris</parameter>"
            "</function></tool_call>"
        )

        result = self.detector.detect_and_parse(tool_call, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(json.loads(result.calls[0].parameters), {"query": "tetris"})

    def test_corrects_server_style_tool_name_case_streaming(self):
        normal_text, calls = self._stream_chunks(
            [
                "<tool_call><function=Search>",
                "<parameter=query>tetris</parameter>",
                "</function></tool_call>",
            ]
        )

        self.assertEqual(normal_text, "")
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].tool_index, 0)
        self.assertEqual(
            json.loads(self._arguments_from_calls(calls)), {"query": "tetris"}
        )

    def test_forwards_unknown_tool_call_non_streaming(self):
        tool_call = (
            "<tool_call><function=Grep>"
            "<parameter=pattern>needle</parameter>"
            "</function></tool_call>"
        )

        result = self.detector.detect_and_parse(tool_call, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "Grep")
        self.assertEqual(result.calls[0].tool_index, -1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"pattern": "needle"})

    def test_forwards_unknown_tool_call_streaming(self):
        normal_text, calls = self._stream_chunks(
            [
                "<tool_call><function=Grep>",
                "<parameter=pattern>needle</parameter>",
                "</function></tool_call>",
            ]
        )

        self.assertEqual(normal_text, "")
        self.assertEqual(calls[0].name, "Grep")
        self.assertEqual(calls[0].tool_index, 0)
        self.assertEqual(
            json.loads(self._arguments_from_calls(calls)), {"pattern": "needle"}
        )

    def test_single_character_chunks_parse_tool_call(self):
        """One-character chunks should not leak or miss any MiMo tags."""
        normal_text, calls = self._stream_chunks(
            "prefix " + self._tool_call_text() + " trailing",
            detector=MiMoDetector(),
        )

        self.assertEqual(normal_text, "prefix  trailing")
        self.assertEqual([call.name for call in calls if call.name], ["search"])
        self.assertEqual(
            json.loads(self._arguments_from_calls(calls)),
            {"query": "tetris", "limit": 3},
        )

    def test_every_two_chunk_split_point_parse_tool_call(self):
        """Every possible two-chunk split should parse the same tool call."""
        full_text = "prefix " + self._tool_call_text() + " trailing"
        for split_at in range(1, len(full_text)):
            with self.subTest(split_at=split_at):
                normal_text, calls = self._stream_chunks(
                    [full_text[:split_at], full_text[split_at:]],
                    detector=MiMoDetector(),
                )

                self.assertEqual(normal_text, "prefix  trailing")
                self.assertEqual([call.name for call in calls if call.name], ["search"])
                self.assertEqual(
                    json.loads(self._arguments_from_calls(calls)),
                    {"query": "tetris", "limit": 3},
                )

    def test_every_structural_marker_split_point_parse_tool_call(self):
        """Explicitly split inside every structural marker MiMo understands."""
        tool_call = self._tool_call_text()
        full_text = "prefix " + tool_call + " trailing"
        markers = [
            "<tool_call>",
            "<function=",
            "<parameter=",
            "</parameter>",
            "</function>",
            "</tool_call>",
        ]

        for marker in markers:
            search_at = 0
            while True:
                marker_at = full_text.find(marker, search_at)
                if marker_at == -1:
                    break
                search_at = marker_at + len(marker)

                for split_inside_marker in range(1, len(marker)):
                    split_at = marker_at + split_inside_marker
                    with self.subTest(marker=marker, split_at=split_at):
                        normal_text, calls = self._stream_chunks(
                            [full_text[:split_at], full_text[split_at:]],
                            detector=MiMoDetector(),
                        )

                        self.assertEqual(normal_text, "prefix  trailing")
                        self.assertEqual(
                            [call.name for call in calls if call.name], ["search"]
                        )
                        self.assertEqual(
                            json.loads(self._arguments_from_calls(calls)),
                            {"query": "tetris", "limit": 3},
                        )
