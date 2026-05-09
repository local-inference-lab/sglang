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
"""Tests for OpenAI API protocol models"""

import unittest
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    Function,
    ModelCard,
    ModelList,
    ResponsesRequest,
    Tool,
    UsageInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


class TestModelCard(unittest.TestCase):
    """Test ModelCard protocol model"""

    def test_model_card_serialization(self):
        """Test model card JSON serialization"""
        card = ModelCard(id="test-model", max_model_len=4096)
        data = card.model_dump()
        self.assertEqual(data["id"], "test-model")
        self.assertEqual(data["object"], "model")
        self.assertEqual(data["max_model_len"], 4096)


class TestModelList(unittest.TestCase):
    """Test ModelList protocol model"""

    def test_empty_model_list(self):
        """Test empty model list creation"""
        model_list = ModelList()
        self.assertEqual(model_list.object, "list")
        self.assertEqual(len(model_list.data), 0)

    def test_model_list_with_cards(self):
        """Test model list with model cards"""
        cards = [
            ModelCard(id="model-1"),
            ModelCard(id="model-2", max_model_len=2048),
        ]
        model_list = ModelList(data=cards)
        self.assertEqual(len(model_list.data), 2)
        self.assertEqual(model_list.data[0].id, "model-1")
        self.assertEqual(model_list.data[1].id, "model-2")


class TestCompletionRequest(unittest.TestCase):
    """Test CompletionRequest protocol model"""

    def test_basic_completion_request(self):
        """Test basic completion request"""
        request = CompletionRequest(model="test-model", prompt="Hello world")
        self.assertEqual(request.model, "test-model")
        self.assertEqual(request.prompt, "Hello world")
        self.assertEqual(request.max_tokens, 16)  # default
        self.assertEqual(request.temperature, 1.0)  # default
        self.assertEqual(request.n, 1)  # default
        self.assertFalse(request.stream)  # default
        self.assertFalse(request.echo)  # default

    def test_completion_request_sglang_extensions(self):
        """Test completion request with SGLang-specific extensions"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello",
            top_k=50,
            min_p=0.1,
            repetition_penalty=1.1,
            regex=r"\d+",
            json_schema='{"type": "object"}',
            entropy_penalty=1.5,
            entropy_penalty_min_len=8,
            entropy_penalty_max_len=64,
            entropy_penalty_window=512,
            entropy_penalty_max_penalty=4.0,
            entropy_penalty_min_repetitions=3,
            thinking_end_logit_boost=2.5,
            thinking_end_logit_boost_start=32,
            thinking_end_logit_boost_ramp=256,
            thinking_start_token_id=11,
            thinking_end_token_id=12,
            lora_path="/path/to/lora",
        )
        self.assertEqual(request.top_k, 50)
        self.assertEqual(request.min_p, 0.1)
        self.assertEqual(request.repetition_penalty, 1.1)
        self.assertEqual(request.regex, r"\d+")
        self.assertEqual(request.json_schema, '{"type": "object"}')
        self.assertEqual(request.entropy_penalty, 1.5)
        self.assertEqual(request.entropy_penalty_min_len, 8)
        self.assertEqual(request.entropy_penalty_max_len, 64)
        self.assertEqual(request.entropy_penalty_window, 512)
        self.assertEqual(request.entropy_penalty_max_penalty, 4.0)
        self.assertEqual(request.entropy_penalty_min_repetitions, 3)
        self.assertEqual(request.thinking_end_logit_boost, 2.5)
        self.assertEqual(request.thinking_end_logit_boost_start, 32)
        self.assertEqual(request.thinking_end_logit_boost_ramp, 256)
        self.assertEqual(request.thinking_start_token_id, 11)
        self.assertEqual(request.thinking_end_token_id, 12)
        self.assertEqual(request.lora_path, "/path/to/lora")

    def test_completion_request_validation_errors(self):
        """Test completion request validation errors"""
        with self.assertRaises(ValidationError):
            CompletionRequest()  # missing required fields

        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model")  # missing prompt


class TestChatCompletionRequest(unittest.TestCase):
    """Test ChatCompletionRequest protocol model"""

    def test_basic_chat_completion_request(self):
        """Test basic chat completion request"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(model="test-model", messages=messages)
        self.assertEqual(request.model, "test-model")
        self.assertEqual(len(request.messages), 1)
        self.assertEqual(request.messages[0].role, "user")
        self.assertEqual(request.messages[0].content, "Hello")
        self.assertEqual(request.temperature, None)  # default
        self.assertFalse(request.stream)  # default
        self.assertEqual(request.tool_choice, "none")  # default when no tools

    def test_sampling_param_build(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.8,
            max_tokens=150,
            min_tokens=5,
            top_p=0.9,
            entropy_penalty=1.25,
            entropy_penalty_min_len=12,
            entropy_penalty_max_len=96,
            entropy_penalty_window=1024,
            entropy_penalty_max_penalty=6.0,
            entropy_penalty_min_repetitions=4,
            thinking_end_logit_boost=3.0,
            thinking_end_logit_boost_start=24,
            thinking_end_logit_boost_ramp=128,
            thinking_start_token_id=21,
            thinking_end_token_id=22,
            stop=["</s>"],
        )
        params = req.to_sampling_params(["</s>"], {}, None)
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["max_new_tokens"], 150)
        self.assertEqual(params["min_new_tokens"], 5)
        self.assertEqual(params["stop"], ["</s>"])
        self.assertEqual(params["entropy_penalty"], 1.25)
        self.assertEqual(params["entropy_penalty_min_len"], 12)
        self.assertEqual(params["entropy_penalty_max_len"], 96)
        self.assertEqual(params["entropy_penalty_window"], 1024)
        self.assertEqual(params["entropy_penalty_max_penalty"], 6.0)
        self.assertEqual(params["entropy_penalty_min_repetitions"], 4)
        self.assertEqual(params["thinking_end_logit_boost"], 3.0)
        self.assertEqual(params["thinking_end_logit_boost_start"], 24)
        self.assertEqual(params["thinking_end_logit_boost_ramp"], 128)
        self.assertEqual(params["thinking_start_token_id"], 21)
        self.assertEqual(params["thinking_end_token_id"], 22)

    def test_sampling_param_build_uses_entropy_model_defaults(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
        )
        params = req.to_sampling_params(
            [],
            {
                "entropy_penalty": 0.75,
                "entropy_penalty_min_len": 7,
                "entropy_penalty_max_len": 63,
                "entropy_penalty_window": 511,
                "entropy_penalty_max_penalty": 3.5,
                "entropy_penalty_min_repetitions": 5,
                "thinking_end_logit_boost": 1.5,
                "thinking_end_logit_boost_start": 9,
                "thinking_end_logit_boost_ramp": 99,
                "thinking_start_token_id": 31,
                "thinking_end_token_id": 32,
            },
            None,
        )
        self.assertEqual(params["entropy_penalty"], 0.75)
        self.assertEqual(params["entropy_penalty_min_len"], 7)
        self.assertEqual(params["entropy_penalty_max_len"], 63)
        self.assertEqual(params["entropy_penalty_window"], 511)
        self.assertEqual(params["entropy_penalty_max_penalty"], 3.5)
        self.assertEqual(params["entropy_penalty_min_repetitions"], 5)
        self.assertEqual(params["thinking_end_logit_boost"], 1.5)
        self.assertEqual(params["thinking_end_logit_boost_start"], 9)
        self.assertEqual(params["thinking_end_logit_boost_ramp"], 99)
        self.assertEqual(params["thinking_start_token_id"], 31)
        self.assertEqual(params["thinking_end_token_id"], 32)

    def test_sampling_param_build_uses_uniform_internal_defaults(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
        )
        params = req.to_sampling_params([], {}, None)
        self.assertEqual(params["entropy_penalty"], 0.0)
        self.assertEqual(params["entropy_penalty_min_len"], 16)
        self.assertEqual(params["entropy_penalty_max_len"], 256)
        self.assertEqual(params["entropy_penalty_window"], 8192)
        self.assertEqual(params["entropy_penalty_max_penalty"], 8.0)
        self.assertEqual(params["entropy_penalty_min_repetitions"], 1)
        self.assertEqual(params["thinking_end_logit_boost"], 0.0)
        self.assertEqual(params["thinking_end_logit_boost_start"], 0)
        self.assertEqual(params["thinking_end_logit_boost_ramp"], 256)
        self.assertIsNone(params["thinking_start_token_id"])
        self.assertIsNone(params["thinking_end_token_id"])

    def test_chat_preferred_sampling_params_override_generation_defaults(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.4,
            top_p=0.5,
            logit_bias={"7": 1.0},
        )
        params = req.to_sampling_params(
            ["template-stop"],
            {
                "temperature": 0.9,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.1,
                "logit_bias": {"7": -2.0},
                "max_new_tokens": 100,
                "min_new_tokens": 2,
                "sampling_seed": 1,
                "stop": ["generation-stop"],
            },
            {
                "temperature": 0.7,
                "top_p": 0.6,
                "repetition_penalty": 1.2,
                "presence_penalty": 0.2,
                "logit_bias": {"7": -1.0},
                "max_tokens": 50,
                "min_tokens": 3,
                "seed": 7,
                "stop": ["preferred-stop"],
            },
            None,
        )

        self.assertEqual(params["temperature"], 0.4)
        self.assertEqual(params["top_p"], 0.5)
        self.assertEqual(params["repetition_penalty"], 1.2)
        self.assertEqual(params["presence_penalty"], 0.2)
        self.assertEqual(params["logit_bias"], {"7": 1.0})
        self.assertEqual(params["max_new_tokens"], 50)
        self.assertEqual(params["min_new_tokens"], 3)
        self.assertEqual(params["sampling_seed"], 7)
        self.assertEqual(params["stop"], ["preferred-stop"])

    def test_chat_request_object_values_override_preferred_sampling_params(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
        )
        req.skip_special_tokens = False

        params = req.to_sampling_params(
            [],
            {},
            {"skip_special_tokens": True},
            None,
        )

        self.assertFalse(params["skip_special_tokens"])

    def test_chat_completion_tool_choice_validation(self):
        """Test tool choice validation logic"""
        messages = [{"role": "user", "content": "Hello"}]

        # No tools, tool_choice should default to "none"
        request1 = ChatCompletionRequest(model="test-model", messages=messages)
        self.assertEqual(request1.tool_choice, "none")

        # With tools, tool_choice should default to "auto"
        tools = [
            {
                "type": "function",
                "function": {"name": "test_func", "description": "Test function"},
            }
        ]
        request2 = ChatCompletionRequest(
            model="test-model", messages=messages, tools=tools
        )
        self.assertEqual(request2.tool_choice, "auto")

    def test_chat_completion_sglang_extensions(self):
        """Test chat completion with SGLang extensions"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            top_k=40,
            min_p=0.05,
            entropy_penalty=1.5,
            entropy_penalty_min_len=8,
            entropy_penalty_max_len=64,
            entropy_penalty_window=512,
            entropy_penalty_max_penalty=4.0,
            entropy_penalty_min_repetitions=3,
            thinking_end_logit_boost=2.0,
            thinking_end_logit_boost_start=16,
            thinking_end_logit_boost_ramp=96,
            thinking_start_token_id=41,
            thinking_end_token_id=42,
            separate_reasoning=False,
            stream_reasoning=False,
            chat_template_kwargs={"custom_param": "value"},
        )
        self.assertEqual(request.top_k, 40)
        self.assertEqual(request.min_p, 0.05)
        self.assertEqual(request.entropy_penalty, 1.5)
        self.assertEqual(request.entropy_penalty_min_len, 8)
        self.assertEqual(request.entropy_penalty_max_len, 64)
        self.assertEqual(request.entropy_penalty_window, 512)
        self.assertEqual(request.entropy_penalty_max_penalty, 4.0)
        self.assertEqual(request.entropy_penalty_min_repetitions, 3)
        self.assertEqual(request.thinking_end_logit_boost, 2.0)
        self.assertEqual(request.thinking_end_logit_boost_start, 16)
        self.assertEqual(request.thinking_end_logit_boost_ramp, 96)
        self.assertEqual(request.thinking_start_token_id, 41)
        self.assertEqual(request.thinking_end_token_id, 42)
        self.assertFalse(request.separate_reasoning)
        self.assertFalse(request.stream_reasoning)
        self.assertEqual(request.chat_template_kwargs, {"custom_param": "value"})

    def test_responses_request_entropy_sampling_params(self):
        request = ResponsesRequest(
            model="test-model",
            input="Hello",
            entropy_penalty=1.5,
            entropy_penalty_min_len=8,
            entropy_penalty_max_len=64,
            entropy_penalty_window=512,
            entropy_penalty_max_penalty=4.0,
            entropy_penalty_min_repetitions=3,
            thinking_end_logit_boost=2.25,
            thinking_end_logit_boost_start=20,
            thinking_end_logit_boost_ramp=80,
            thinking_start_token_id=51,
            thinking_end_token_id=52,
        )
        params = request.to_sampling_params(default_max_tokens=100)
        self.assertEqual(params["entropy_penalty"], 1.5)
        self.assertEqual(params["entropy_penalty_min_len"], 8)
        self.assertEqual(params["entropy_penalty_max_len"], 64)
        self.assertEqual(params["entropy_penalty_window"], 512)
        self.assertEqual(params["entropy_penalty_max_penalty"], 4.0)
        self.assertEqual(params["entropy_penalty_min_repetitions"], 3)
        self.assertEqual(params["thinking_end_logit_boost"], 2.25)
        self.assertEqual(params["thinking_end_logit_boost_start"], 20)
        self.assertEqual(params["thinking_end_logit_boost_ramp"], 80)
        self.assertEqual(params["thinking_start_token_id"], 51)
        self.assertEqual(params["thinking_end_token_id"], 52)

    def test_responses_request_leaves_omitted_entropy_for_server_defaults(self):
        request = ResponsesRequest(model="test-model", input="Hello")
        params = request.to_sampling_params(default_max_tokens=100)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_p"], 1.0)
        self.assertEqual(params["repetition_penalty"], 1.0)
        self.assertIsNone(params["entropy_penalty"])
        self.assertIsNone(params["entropy_penalty_min_len"])
        self.assertIsNone(params["entropy_penalty_max_len"])
        self.assertIsNone(params["entropy_penalty_window"])
        self.assertIsNone(params["entropy_penalty_max_penalty"])
        self.assertIsNone(params["entropy_penalty_min_repetitions"])
        self.assertIsNone(params["thinking_end_logit_boost"])
        self.assertIsNone(params["thinking_end_logit_boost_start"])
        self.assertIsNone(params["thinking_end_logit_boost_ramp"])
        self.assertIsNone(params["thinking_start_token_id"])
        self.assertIsNone(params["thinking_end_token_id"])

    def test_chat_completion_reasoning_effort(self):
        """Test chat completion with reasoning effort"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            reasoning={
                "enabled": True,
                "reasoning_effort": "high",
            },
        )
        self.assertEqual(request.reasoning_effort, "high")
        self.assertEqual(
            request.chat_template_kwargs,
            {"thinking": True, "enable_thinking": True},
        )

    def test_chat_completion_reasoning_effort_none(self):
        """Test reasoning_effort='none' disables thinking"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            reasoning_effort="none",
        )
        self.assertEqual(request.reasoning_effort, "none")
        self.assertFalse(request.chat_template_kwargs.get("thinking"))
        self.assertFalse(request.chat_template_kwargs.get("enable_thinking"))

    def test_chat_completion_reasoning_effort_none_from_reasoning_dict(self):
        """Test reasoning_effort='none' via nested reasoning dict"""
        messages = [{"role": "user", "content": "Hello"}]
        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            reasoning={"effort": "none"},
        )
        self.assertEqual(request.reasoning_effort, "none")
        self.assertFalse(request.chat_template_kwargs.get("thinking"))
        self.assertFalse(request.chat_template_kwargs.get("enable_thinking"))

    def test_chat_completion_json_format(self):
        """Test chat completion json format"""
        transcript = "Good morning! It's 7:00 AM, and I'm just waking up. Today is going to be a busy day, "
        "so let's get started. First, I need to make a quick breakfast. I think I'll have some "
        "scrambled eggs and toast with a cup of coffee. While I'm cooking, I'll also check my "
        "emails to see if there's anything urgent."

        messages = [
            {
                "role": "system",
                "content": "The following is a voice message transcript. Only answer in JSON.",
            },
            {
                "role": "user",
                "content": transcript,
            },
        ]

        class VoiceNote(BaseModel):
            title: str = Field(description="A title for the voice note")
            summary: str = Field(
                description="A short one sentence summary of the voice note."
            )
            strict: Optional[bool] = True
            actionItems: List[str] = Field(
                description="A list of action items from the voice note"
            )

        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            top_k=40,
            min_p=0.05,
            separate_reasoning=False,
            stream_reasoning=False,
            chat_template_kwargs={"custom_param": "value"},
            response_format={
                "type": "json_schema",
                "schema": VoiceNote.model_json_schema(),
            },
        )
        res_format = request.response_format
        json_format = res_format.json_schema
        name = json_format.name
        schema = json_format.schema_
        strict = json_format.strict
        self.assertEqual(name, "VoiceNote")
        self.assertEqual(strict, True)
        self.assertNotIn("strict", schema["properties"])

        request = ChatCompletionRequest(
            model="test-model",
            messages=messages,
            top_k=40,
            min_p=0.05,
            separate_reasoning=False,
            stream_reasoning=False,
            chat_template_kwargs={"custom_param": "value"},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "VoiceNote",
                    "schema": VoiceNote.model_json_schema(),
                    "strict": True,
                },
            },
        )
        res_format = request.response_format
        json_format = res_format.json_schema
        name = json_format.name
        schema = json_format.schema_
        strict = json_format.strict
        self.assertEqual(name, "VoiceNote")
        self.assertEqual(strict, True)


class TestModelSerialization(unittest.TestCase):
    """Test model serialization with hidden states"""

    def test_hidden_states_excluded_when_none(self):
        """Test that None hidden_states are excluded with exclude_none=True"""
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop",
            hidden_states=None,
        )

        response = ChatCompletionResponse(
            id="test-id",
            model="test-model",
            choices=[choice],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )

        # Test exclude_none serialization (should exclude None hidden_states)
        data = response.model_dump(exclude_none=True)
        self.assertNotIn("hidden_states", data["choices"][0])

    def test_hidden_states_included_when_not_none(self):
        """Test that non-None hidden_states are included"""
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop",
            hidden_states=[0.1, 0.2, 0.3],
        )

        response = ChatCompletionResponse(
            id="test-id",
            model="test-model",
            choices=[choice],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )

        # Test exclude_none serialization (should include non-None hidden_states)
        data = response.model_dump(exclude_none=True)
        self.assertIn("hidden_states", data["choices"][0])
        self.assertEqual(data["choices"][0]["hidden_states"], [0.1, 0.2, 0.3])


class TestFunctionDeferLoading(unittest.TestCase):
    """Test defer_loading field behavior on Function/Tool."""

    def test_function_defaults_preserve_strict(self):
        """strict must default to False and be present in dumps so downstream
        code (function_call_parser, chat templates) sees the expected shape."""
        f = Function(name="foo")
        data = f.model_dump()
        self.assertEqual(data["name"], "foo")
        self.assertEqual(data["strict"], False)
        self.assertNotIn("defer_loading", data)

    def test_function_defer_loading_true_serialized(self):
        f = Function(name="foo", defer_loading=True)
        data = f.model_dump()
        self.assertTrue(data["defer_loading"])
        self.assertEqual(data["strict"], False)

    def test_function_defer_loading_false_serialized(self):
        """defer_loading=False is an explicit value and must be preserved."""
        f = Function(name="foo", defer_loading=False)
        data = f.model_dump()
        self.assertIn("defer_loading", data)
        self.assertFalse(data["defer_loading"])

    def test_tool_level_defer_loading_propagates_to_function(self):
        """defer_loading at the Tool level should propagate to Function."""
        tool = Tool(
            type="function",
            defer_loading=True,
            function={"name": "search_db"},
        )
        self.assertTrue(tool.function.defer_loading)
        data = tool.model_dump()
        self.assertTrue(data["function"]["defer_loading"])

    def test_function_level_defer_loading_wins_over_tool_level(self):
        """Explicit function-level value is preserved when both set."""
        tool = Tool(
            type="function",
            defer_loading=True,
            function={"name": "search_db", "defer_loading": False},
        )
        self.assertFalse(tool.function.defer_loading)

    def test_tool_reference_content_part_accepted(self):
        """Chat completion should accept tool_reference content on tool-role
        messages (GLM-specific extension consumed by the chat template)."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "tool_reference", "name": "search_db"},
                    {"type": "text", "text": "ok"},
                ],
            },
        ]
        request = ChatCompletionRequest(model="test-model", messages=messages)
        parts = request.messages[0].content
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0].type, "tool_reference")
        self.assertEqual(parts[0].name, "search_db")
        self.assertEqual(parts[1].type, "text")


class TestValidationEdgeCases(unittest.TestCase):
    """Test edge cases and validation scenarios"""

    def test_invalid_tool_choice_type(self):
        """Test invalid tool choice type"""
        messages = [{"role": "user", "content": "Hello"}]
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="test-model", messages=messages, tool_choice=123
            )

    def test_negative_token_limits(self):
        """Test negative token limits"""
        with self.assertRaises(ValidationError):
            CompletionRequest(model="test-model", prompt="Hello", max_tokens=-1)

    def test_model_serialization_roundtrip(self):
        """Test that models can be serialized and deserialized"""
        original_request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        # Serialize to dict
        data = original_request.model_dump()

        # Deserialize back
        restored_request = ChatCompletionRequest(**data)

        self.assertEqual(restored_request.model, original_request.model)
        self.assertEqual(restored_request.temperature, original_request.temperature)
        self.assertEqual(restored_request.max_tokens, original_request.max_tokens)
        self.assertEqual(len(restored_request.messages), len(original_request.messages))


if __name__ == "__main__":
    unittest.main(verbosity=2)
