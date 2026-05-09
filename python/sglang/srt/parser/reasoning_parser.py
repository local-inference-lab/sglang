from typing import Dict, List, Optional, Tuple, Type

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.parser.harmony_parser import HarmonyParser


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
        thinks_internally: bool = False,
        reasoning_default: str = "always",
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.think_excluded_tokens = think_excluded_tokens
        self.tool_start_token = tool_start_token
        self.force_reasoning = force_reasoning
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.thinks_internally = thinks_internally
        self.reasoning_default = reasoning_default

        self._buffer = ""
        self._reasoning_accumulator = ""
        self.stripped_think_start = False
        self.think_start_self_label = ""

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        think_start_text = self.think_start_token + self.think_start_self_label
        end_idx = text.find(self.think_end_token)
        tool_idx = (
            text.find(self.tool_start_token)
            if self.tool_start_token is not None
            else -1
        )

        if tool_idx != -1 and (end_idx == -1 or tool_idx < end_idx):
            # Treat tool-call markup as ending reasoning when it appears before
            # the explicit reasoning end token.
            reasoning_text = text[:tool_idx].replace(think_start_text, "").strip()
            normal_text = text[tool_idx:]
            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )

        if end_idx == -1 and self.think_end_token not in self.previous_content:
            # Assume reasoning was truncated before end token
            return StreamingParseResult(
                reasoning_text=text.replace(think_start_text, "").strip()
            )

        # Extract reasoning content
        if end_idx != -1:
            reasoning_text = text[:end_idx].replace(think_start_text, "").strip()
            normal_text = text[end_idx + len(self.think_end_token) :].strip()

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text
            )
        else:
            # think_end_token is in self.previous_content for continue_final_message=True case
            return StreamingParseResult(
                normal_text=text.replace(think_start_text, "").strip()
            )

    @staticmethod
    def _find_first_marker(text: str, markers: list[str]) -> tuple[int, str] | None:
        """Return the earliest full marker occurrence in text."""
        first_idx = -1
        first_marker = ""
        for marker in markers:
            if not marker:
                continue
            idx = text.find(marker)
            if idx == -1:
                continue
            if first_idx == -1 or idx < first_idx:
                first_idx = idx
                first_marker = marker
        if first_idx == -1:
            return None
        return first_idx, first_marker

    @staticmethod
    def _partial_suffix_len(text: str, markers: list[str]) -> int:
        """Return the longest suffix that could become one of the markers."""
        max_suffix_len = 0
        for marker in markers:
            if not marker:
                continue
            max_len = min(len(text), len(marker) - 1)
            for suffix_len in range(1, max_len + 1):
                if text.endswith(marker[:suffix_len]):
                    max_suffix_len = max(max_suffix_len, suffix_len)
        return max_suffix_len

    def _reasoning_markers(self) -> list[str]:
        markers = [
            self.think_end_token,
            self.think_start_token + self.think_start_self_label,
        ]
        if self.tool_start_token:
            markers.append(self.tool_start_token)
        return markers

    def _normal_markers(self) -> list[str]:
        markers = [self.think_end_token]
        if not self.stripped_think_start:
            markers.append(self.think_start_token + self.think_start_self_label)
        return markers

    def _append_reasoning(self, text: str, reasoning_parts: list[str]) -> None:
        if not text:
            return
        if self.stream_reasoning:
            reasoning_parts.append(text)
        else:
            self._reasoning_accumulator += text

    def _flush_reasoning_accumulator(self, reasoning_parts: list[str]) -> None:
        if self.stream_reasoning or not self._reasoning_accumulator:
            return
        reasoning_parts.append(self._reasoning_accumulator)
        self._reasoning_accumulator = ""

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        normal_parts: list[str] = []
        reasoning_parts: list[str] = []
        think_start_text = self.think_start_token + self.think_start_self_label

        while self._buffer:
            if self._in_reasoning:
                markers = self._reasoning_markers()
                marker_match = self._find_first_marker(self._buffer, markers)
                if marker_match is None:
                    suffix_len = self._partial_suffix_len(self._buffer, markers)
                    safe_len = len(self._buffer) - suffix_len
                    self._append_reasoning(self._buffer[:safe_len], reasoning_parts)
                    self._buffer = self._buffer[safe_len:]
                    break

                marker_idx, marker = marker_match
                self._append_reasoning(self._buffer[:marker_idx], reasoning_parts)

                if marker == think_start_text:
                    self.stripped_think_start = True
                    self._buffer = self._buffer[marker_idx + len(marker) :]
                    continue

                if marker == self.think_end_token:
                    self._buffer = self._buffer[marker_idx + len(marker) :]
                    self._in_reasoning = False
                    self._flush_reasoning_accumulator(reasoning_parts)
                    continue

                # Tool-call markers switch the remainder of the current chunk to
                # normal text so the function-call parser receives the raw marker.
                normal_parts.append(self._buffer[marker_idx:])
                self._buffer = ""
                self._in_reasoning = False
                self._flush_reasoning_accumulator(reasoning_parts)
                break

            markers = self._normal_markers()
            marker_match = (
                self._find_first_marker(self._buffer, [think_start_text])
                if not self.stripped_think_start
                else None
            )
            if marker_match is not None:
                marker_idx, marker = marker_match
                normal_parts.append(self._buffer[:marker_idx])
                self.stripped_think_start = True
                self._in_reasoning = True
                self._buffer = self._buffer[marker_idx + len(marker) :]
                continue

            suffix_len = self._partial_suffix_len(self._buffer, markers)
            safe_len = len(self._buffer) - suffix_len
            normal_parts.append(self._buffer[:safe_len])
            self._buffer = self._buffer[safe_len:]
            break

        return StreamingParseResult(
            normal_text="".join(normal_parts),
            reasoning_text="".join(reasoning_parts),
        )


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Supported models:
      - DeepSeek-R1: Always generates thinking content without <think> start tag
      - DeepSeek-R1-0528: Generates thinking content with <think> start tag

    Format patterns:
      - DeepSeek-R1: "I need to think about this...</think>The answer is 42."
      - DeepSeek-R1-0528: "<think>I need to think about this...</think>The answer is 42."

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class KimiK2Detector(BaseReasoningFormatDetector):
    """
    Detector for Kimi K2 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Kimi K2 can switch from reasoning to tool-call section with
    `<|tool_calls_section_begin|>` before emitting `</think>`.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        think_excluded_tokens = [
            "<think>",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_section_end|>",
            "<|tool_call_end|>",
            "[EOS]",
            "<|im_end|>",
            "<|end_header_id|>",
            "[EOT]",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<|tool_calls_section_begin|>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="thinking",
        )


class MiMoReasoningDetector(BaseReasoningFormatDetector):
    """
    Detector for MiMo reasoning with native `<tool_call>` interruption.

    MiMo can switch from reasoning to its native tool-call grammar before it
    emits `</think>`, so the reasoning parser must pass `<tool_call>` through
    as normal text for the function-call parser.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="explicit_enable_thinking",
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    Detector for GLM-4.5 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    GLM-4.5 uses `<tool_call>` as the tool start token to switch from reasoning mode to normal mode.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<eop>",
            "<|user|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
        )


class GptOssDetector(BaseReasoningFormatDetector):
    """
    Detector for T4-style reasoning format (GPT-OSS), using the HarmonyParser.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<|channel|>analysis<|message|>",
            "<|end|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.parser = HarmonyParser()

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        events = self.parser.parse(text)
        # Flush the buffer for one-shot parsing
        events += self.parser.parse("")

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)
        # Tool call events preserve raw text with structural markers

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        events = self.parser.parse(new_text)

        reasoning_text = "".join(
            [e.content for e in events if e.event_type == "reasoning"]
        )
        normal_parts = []
        for e in events:
            if e.event_type == "normal":
                normal_parts.append(e.content)
            elif e.event_type == "tool_call":
                # Use raw_text to preserve structural markers for function call detector
                normal_parts.append(e.raw_text if e.raw_text else e.content)
        normal_text = "".join(normal_parts)

        return StreamingParseResult(
            normal_text=normal_text,
            reasoning_text=reasoning_text,
        )


class MiniMaxAppendThinkDetector(BaseReasoningFormatDetector):
    """
    Append `<think>` token to the beginning of the text.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        # scheduler.py need `reasoning_parser.detector.think_end_token`
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )
        self.is_first_chunk = False

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        if not self.is_first_chunk:
            self.is_first_chunk = True
            new_text = self.think_start_token + new_text
        return StreamingParseResult(normal_text=new_text)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        return StreamingParseResult(normal_text=self.think_start_token + text)


class Nemotron3Detector(BaseReasoningFormatDetector):
    """
    Detector for Nemotron3 model.
    Uses the same reasoning format as DeepSeek-R1: (<think>)*(.*)</think>

    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="enable_thinking",
        )
        self._force_nonempty_content = force_nonempty_content

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        ret = super().detect_and_parse(text)
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret


class MistralDetector(BaseReasoningFormatDetector):
    """
    Detector for Mistral models with reasoning (e.g., Mistral-Small-4-119B-2603).
    Assumes reasoning format:
      [THINK]reasoning content[/THINK]answer

    Reasoning is optional — it only appears when reasoning_effort="high" is set.
    When reasoning_effort="none", the model outputs directly without thinking tokens.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "[THINK]",
            "[/THINK]",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="mistral",
        )


class HunyuanDetector(BaseReasoningFormatDetector):
    """
    Detector for Hunyuan models (e.g., tencent/Hunyuan-A13B-Instruct).

    Like Glm45Detector but uses ``<tool_calls>`` (plural) as the tool start token.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_calls>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
        )


class Gemma4Detector(BaseReasoningFormatDetector):
    """Gemma4 reasoning detector."""

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
    ):
        super().__init__(
            "<|channel>",
            "<channel|>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="explicit_enable_thinking",
        )
        self.think_start_self_label = "thought\n"


class _DeepSeekV3Detector(Qwen3Detector):
    """DeepSeek-V3 reuses Qwen3 tokens but requires explicit thinking=True to enable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_thinking"


class _MimoDetector(Qwen3Detector):
    """MIMO reuses Qwen3 tokens but requires explicit enable_thinking=True to enable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_enable_thinking"


class _PoolsideV1Detector(Qwen3Detector):
    """Poolside v1 (Laguna-XS.2) reuses Qwen3 <think> tokens but the HF chat template
    defaults `enable_thinking=False`; reasoning is opt-in via `enable_thinking=True`."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reasoning_default = "explicit_enable_thinking"


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: Dict[str, Type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "deepseek-v3": _DeepSeekV3Detector,
        "deepseek-v4": _DeepSeekV3Detector,
        "glm45": Glm45Detector,
        "hunyuan": HunyuanDetector,
        "gpt-oss": GptOssDetector,
        "kimi": KimiDetector,
        "kimi_k2": KimiK2Detector,
        "poolside_v1": _PoolsideV1Detector,
        "mimo": MiMoReasoningDetector,
        "qwen3": Qwen3Detector,
        "qwen3-thinking": Qwen3Detector,
        "minimax": Qwen3Detector,
        "minimax-append-think": MiniMaxAppendThinkDetector,
        "step3": DeepSeekR1Detector,
        "step3p5": DeepSeekR1Detector,
        "mistral": MistralDetector,
        "nemotron_3": Nemotron3Detector,
        "interns1": Qwen3Detector,
        "gemma4": Gemma4Detector,
    }

    def __init__(
        self,
        model_type: Optional[str] = None,
        stream_reasoning: bool = True,
        force_reasoning: Optional[bool] = None,
        request: ChatCompletionRequest = None,
    ):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Special cases where we override force_reasoning
        if model_type.lower() in {
            "qwen3-thinking",
            "gpt-oss",
            "minimax",
        }:
            force_reasoning = True

        # Only pass force_reasoning if explicitly set, let detectors use their defaults
        kwargs = {"stream_reasoning": stream_reasoning}
        if force_reasoning is not None:
            kwargs["force_reasoning"] = force_reasoning

        if (
            request is not None
            and isinstance(request, ChatCompletionRequest)
            and request.continue_final_message
            and request.messages[-1].role == "assistant"
        ):
            kwargs["continue_final_message"] = True
            kwargs["previous_content"] = request.messages[-1].content

        chat_template_kwargs = getattr(request, "chat_template_kwargs", None) or {}
        if chat_template_kwargs.get("force_nonempty_content") is True:
            kwargs["force_nonempty_content"] = True

        self.detector = detector_class(**kwargs)

    def parse_non_stream(self, full_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(
        self, chunk_text: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
