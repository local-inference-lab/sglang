"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest tests.test_serving_completions_unit -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import json
import unittest
from http import HTTPStatus
from typing import Optional
from unittest.mock import AsyncMock, Mock

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    merge_preferred_sampling_params,
)
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="stage-a-test-cpu")


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = (
            None  # Set to None to avoid template processing
        )


class ServingCompletionTestCase(unittest.TestCase):
    """Bundle all prompt/echo tests in one TestCase."""

    # ---------- shared test fixtures ----------
    def setUp(self):
        # build the mock TokenizerManager once for every test
        tm = Mock(spec=TokenizerManager)

        tm.tokenizer = Mock()
        tm.tokenizer.encode.return_value = [1, 2, 3, 4]
        tm.tokenizer.decode.return_value = "decoded text"
        tm.tokenizer.bos_token_id = 1

        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(enable_cache_report=False)

        tm.generate_request = AsyncMock()
        tm.create_abort_task = Mock()

        self.template_manager = _MockTemplateManager()
        self.sc = OpenAIServingCompletion(tm, self.template_manager)
        self.fastapi_request = Mock(spec=Request)

    # ---------- prompt-handling ----------
    def test_single_string_prompt(self):
        req = CompletionRequest(model="x", prompt="Hello world", max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.text, "Hello world")

    def test_single_token_ids_prompt(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3, 4], max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.input_ids, [1, 2, 3, 4])

    # ---------- echo-handling ----------
    def test_echo_with_string_prompt_streaming(self):
        req = CompletionRequest(model="x", prompt="Hello", max_tokens=1, echo=True)
        self.assertEqual(self.sc._get_echo_text(req, 0), "Hello")

    def test_echo_with_list_of_strings_streaming(self):
        req = CompletionRequest(
            model="x", prompt=["A", "B"], max_tokens=1, echo=True, n=1
        )
        self.assertEqual(self.sc._get_echo_text(req, 0), "A")
        self.assertEqual(self.sc._get_echo_text(req, 1), "B")

    def test_echo_with_token_ids_streaming(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3], max_tokens=1, echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded_prompt"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded_prompt")

    def test_echo_with_multiple_token_ids_streaming(self):
        req = CompletionRequest(
            model="x", prompt=[[1, 2], [3, 4]], max_tokens=1, echo=True, n=1
        )
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded")

    def test_prepare_echo_prompts_non_streaming(self):
        # single string
        req = CompletionRequest(model="x", prompt="Hi", echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi"])

        # list of strings
        req = CompletionRequest(model="x", prompt=["Hi", "Yo"], echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi", "Yo"])

        # token IDs
        req = CompletionRequest(model="x", prompt=[1, 2, 3], echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["decoded"])

    # ---------- response_format handling ----------
    def test_response_format_json_object(self):
        """Test that response_format json_object is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        sampling_params = self.sc._build_sampling_params(req)
        self.assertEqual(sampling_params["json_schema"], '{"type": "object"}')

    def test_response_format_json_schema(self):
        """Test that response_format json_schema is correctly processed in sampling params."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema},
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The schema should be converted to string by convert_json_schema_to_str
        self.assertIn("json_schema", sampling_params)
        self.assertIsInstance(sampling_params["json_schema"], str)

    def test_response_format_structural_tag(self):
        """Test that response_format structural_tag is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate structured output:",
            max_tokens=100,
            response_format={
                "type": "structural_tag",
                "structures": [{"begin": "<data>", "end": "</data>"}],
                "triggers": ["<data>"],
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The structural_tag should be processed
        self.assertIn("structural_tag", sampling_params)
        self.assertIsInstance(sampling_params["structural_tag"], str)

    def test_response_format_none(self):
        """Test that no response_format doesn't add extra constraints."""
        req = CompletionRequest(model="x", prompt="Generate text:", max_tokens=100)
        sampling_params = self.sc._build_sampling_params(req)
        # Should not have json_schema or structural_tag from response_format
        # (but might have json_schema from the legacy json_schema field)
        self.assertIsNone(sampling_params.get("structural_tag"))

    def test_entropy_penalty_sampling_params(self):
        """Test completion entropy_penalty fields are forwarded."""
        req = CompletionRequest(
            model="x",
            prompt="Generate text:",
            max_tokens=100,
            entropy_penalty=1.5,
            entropy_penalty_min_len=8,
            entropy_penalty_max_len=64,
            entropy_penalty_window=512,
            entropy_penalty_max_penalty=4.0,
            entropy_penalty_min_repetitions=3,
        )
        sampling_params = self.sc._build_sampling_params(req)
        self.assertEqual(sampling_params["entropy_penalty"], 1.5)
        self.assertEqual(sampling_params["entropy_penalty_min_len"], 8)
        self.assertEqual(sampling_params["entropy_penalty_max_len"], 64)
        self.assertEqual(sampling_params["entropy_penalty_window"], 512)
        self.assertEqual(sampling_params["entropy_penalty_max_penalty"], 4.0)
        self.assertEqual(sampling_params["entropy_penalty_min_repetitions"], 3)

    def test_thinking_end_logit_boost_sampling_params(self):
        """Test completion thinking boost fields are forwarded."""
        req = CompletionRequest(
            model="x",
            prompt="Generate text:",
            max_tokens=100,
            thinking_end_logit_boost=2.5,
            thinking_end_logit_boost_start=24,
            thinking_end_logit_boost_ramp=128,
            thinking_start_token_id=11,
            thinking_end_token_id=12,
        )
        sampling_params = self.sc._build_sampling_params(req)
        self.assertEqual(sampling_params["thinking_end_logit_boost"], 2.5)
        self.assertEqual(sampling_params["thinking_end_logit_boost_start"], 24)
        self.assertEqual(sampling_params["thinking_end_logit_boost_ramp"], 128)
        self.assertEqual(sampling_params["thinking_start_token_id"], 11)
        self.assertEqual(sampling_params["thinking_end_token_id"], 12)

    def test_omitted_entropy_params_do_not_override_preferred_defaults(self):
        req = CompletionRequest(model="x", prompt="Generate text:", max_tokens=100)
        sampling_params = self.sc._build_sampling_params(req)

        preferred = {
            "entropy_penalty": 0.8,
            "entropy_penalty_min_len": 16,
            "entropy_penalty_max_len": 512,
            "entropy_penalty_window": 8192,
            "entropy_penalty_max_penalty": 12.0,
            "entropy_penalty_min_repetitions": 2,
        }
        merged = merge_preferred_sampling_params(preferred, sampling_params)

        self.assertEqual(merged["entropy_penalty"], 0.8)
        self.assertEqual(merged["entropy_penalty_max_penalty"], 12.0)
        self.assertEqual(merged["entropy_penalty_min_repetitions"], 2)

    def test_omitted_thinking_boost_params_do_not_override_preferred_defaults(self):
        req = CompletionRequest(model="x", prompt="Generate text:", max_tokens=100)
        sampling_params = self.sc._build_sampling_params(req)

        preferred = {
            "thinking_end_logit_boost": 2.5,
            "thinking_end_logit_boost_start": 32,
            "thinking_end_logit_boost_ramp": 256,
            "thinking_start_token_id": 11,
            "thinking_end_token_id": 12,
        }
        merged = merge_preferred_sampling_params(preferred, sampling_params)

        self.assertEqual(merged["thinking_end_logit_boost"], 2.5)
        self.assertEqual(merged["thinking_end_logit_boost_start"], 32)
        self.assertEqual(merged["thinking_end_logit_boost_ramp"], 256)
        self.assertEqual(merged["thinking_start_token_id"], 11)
        self.assertEqual(merged["thinking_end_token_id"], 12)

    def test_preferred_merge_preserves_non_entropy_none_values(self):
        preferred = {
            "entropy_penalty": 0.8,
            "entropy_penalty_max_penalty": 12.0,
            "entropy_penalty_min_repetitions": 2,
            "max_new_tokens": 4096,
            "temperature": 0.2,
            "logit_bias": {"52592": -1e9, "33763": -1e9},
        }
        sampling_params = {
            "entropy_penalty": None,
            "entropy_penalty_max_penalty": None,
            "entropy_penalty_min_repetitions": None,
            "max_new_tokens": None,
            "temperature": None,
            "logit_bias": None,
        }
        merged = merge_preferred_sampling_params(preferred, sampling_params)

        self.assertEqual(merged["entropy_penalty"], 0.8)
        self.assertEqual(merged["entropy_penalty_max_penalty"], 12.0)
        self.assertEqual(merged["entropy_penalty_min_repetitions"], 2)
        self.assertIsNone(merged["max_new_tokens"])
        self.assertIsNone(merged["temperature"])
        self.assertEqual(merged["logit_bias"], {"52592": -1e9, "33763": -1e9})

    def test_preferred_logit_bias_is_forced_when_request_has_bias(self):
        preferred = {
            "logit_bias": {"52592": -1e9, "33763": -1e9},
        }
        sampling_params = {
            "max_new_tokens": None,
            "logit_bias": {"123": -3.0, "52592": 0.0},
        }
        merged = merge_preferred_sampling_params(preferred, sampling_params)

        self.assertIsNone(merged["max_new_tokens"])
        self.assertEqual(
            merged["logit_bias"],
            {"123": -3.0, "52592": -1e9, "33763": -1e9},
        )

    def test_thought_loop_preset_as_preferred_sampling_params(self):
        preferred = {
            "entropy_penalty": 2.0,
            "entropy_penalty_min_len": 16,
            "entropy_penalty_max_len": 512,
            "entropy_penalty_window": 8192,
            "entropy_penalty_max_penalty": 512.0,
            "entropy_penalty_min_repetitions": 2,
            "logit_bias": {"52592": -1e9, "33763": -1e9},
        }
        merged = merge_preferred_sampling_params(
            preferred,
            {
                "max_new_tokens": None,
                "entropy_penalty": None,
                "logit_bias": None,
            },
        )

        self.assertEqual(merged["entropy_penalty"], 2.0)
        self.assertEqual(merged["entropy_penalty_min_len"], 16)
        self.assertEqual(merged["entropy_penalty_max_len"], 512)
        self.assertEqual(merged["entropy_penalty_window"], 8192)
        self.assertEqual(merged["entropy_penalty_max_penalty"], 512.0)
        self.assertEqual(merged["entropy_penalty_min_repetitions"], 2)
        self.assertIsNone(merged["max_new_tokens"])
        self.assertEqual(merged["logit_bias"], {"52592": -1e9, "33763": -1e9})

    def test_preferred_entropy_can_be_overridden_by_explicit_request_value(self):
        preferred = {
            "entropy_penalty": 2.0,
            "entropy_penalty_min_len": 16,
            "entropy_penalty_max_len": 512,
            "logit_bias": {"52592": -1e9, "33763": -1e9},
        }
        merged = merge_preferred_sampling_params(
            preferred,
            {
                "max_new_tokens": None,
                "entropy_penalty": 0.5,
                "logit_bias": {"123": -3.0, "52592": 0.0},
            },
        )

        self.assertEqual(merged["entropy_penalty"], 0.5)
        self.assertEqual(merged["entropy_penalty_min_len"], 16)
        self.assertEqual(merged["entropy_penalty_max_len"], 512)
        self.assertIsNone(merged["max_new_tokens"])
        self.assertEqual(
            merged["logit_bias"],
            {"123": -3.0, "52592": -1e9, "33763": -1e9},
        )

    def test_logprobs_false_non_streaming(self):
        """Test that logprobs=False doesn't cause KeyError in non-streaming response."""
        req = CompletionRequest(
            model="x", prompt="Hello", max_tokens=10, logprobs=False
        )

        mock_ret = [
            {
                "text": " world",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "finish_reason": {"type": "stop"},
                    "weight_version": "v1",
                },
            }
        ]

        response = self.sc._build_completion_response(req, mock_ret, 1234567890)

        self.assertEqual(len(response.choices), 1)
        self.assertEqual(response.choices[0].text, " world")
        self.assertEqual(len(response.choices[0].logprobs.top_logprobs), 0)

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort(*args, **kwargs):
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = _mock_generate_abort

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
        )

        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream():
            chunks = []
            try:
                async for chunk in self.sc._generate_completion_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
            except Exception as e:
                print(f"Error during stream iteration: {e}")
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk, and possibly a role chunk
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    def test_non_streaming_cached_tokens_details_emits_sglext(self):
        """Test that non-streaming completion responses emit cached token details in sglext."""

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            return_cached_tokens_details=True,
        )
        ret = [
            {
                "text": "Cached response",
                "meta_info": {
                    "id": "cmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.sc._build_completion_response(req, ret, 1234567890)

        self.assertIsNotNone(response.sglext)
        self.assertEqual(
            response.sglext.cached_tokens_details.model_dump(exclude_none=True),
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    def test_streaming_cached_tokens_details_emits_sglext(self):
        """Test that streaming completion responses emit cached token details in sglext."""

        async def _mock_generate_with_cached_tokens_details(*args, **kwargs):
            yield {
                "text": "Cached response",
                "meta_info": {
                    "id": "cmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = (
            _mock_generate_with_cached_tokens_details
        )

        req = CompletionRequest(
            model="x",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
            return_cached_tokens_details=True,
        )

        adapted_request, _ = self.sc._convert_to_internal_request(req)

        async def run_stream():
            chunks = []
            async for chunk in self.sc._generate_completion_stream(
                adapted_request, req, self.fastapi_request
            ):
                chunks.append(chunk)
            return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        sglext_chunks = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk[len("data: ") :])
            if "sglext" in data:
                sglext_chunks.append(data)

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["choices"], [])
        self.assertEqual(
            sglext_chunks[0]["sglext"]["cached_tokens_details"],
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
