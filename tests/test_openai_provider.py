"""Regression tests for native OpenAI provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import Config
from src.models.gemini import GeminiModel
from src.models.openai import OpenAIModel
from src.models.openrouter import OpenRouterModel
from src.models import get_model
from src.utils.response import extract_response_text


class OpenAIProviderTests(unittest.TestCase):
    def test_get_model_dispatches_to_gemini_wrapper_with_override(self) -> None:
        with patch("src.models.GeminiModel") as mock_gemini:
            sentinel = object()
            mock_gemini.return_value = sentinel
            selected = get_model(provider="gemini", model_name="gemini-2.5-pro")
            mock_gemini.assert_called_once_with(model_name="gemini-2.5-pro")
            self.assertIs(selected, sentinel)

    def test_get_model_dispatches_to_openai_wrapper(self) -> None:
        with patch("src.models.OpenAIModel") as mock_openai:
            sentinel = object()
            mock_openai.return_value = sentinel
            selected = get_model(provider="openai", model_name="gpt-4.1-mini")
            mock_openai.assert_called_once_with(model_name="gpt-4.1-mini")
            self.assertIs(selected, sentinel)

    def test_get_model_name_for_provider_reads_selected_provider_slot(self) -> None:
        original_gemini = Config.GEMINI_MODEL
        original_openai = Config.OPENAI_MODEL
        original_openrouter = Config.OPENROUTER_MODEL
        try:
            Config.GEMINI_MODEL = "gemini-2.5-flash"
            Config.OPENAI_MODEL = "gpt-4.1-mini"
            Config.OPENROUTER_MODEL = "openrouter/custom-model"
            self.assertEqual(Config.get_model_name_for_provider("gemini"), "gemini-2.5-flash")
            self.assertEqual(Config.get_model_name_for_provider("openai"), "gpt-4.1-mini")
            self.assertEqual(Config.get_model_name_for_provider("openrouter"), "openrouter/custom-model")
        finally:
            Config.GEMINI_MODEL = original_gemini
            Config.OPENAI_MODEL = original_openai
            Config.OPENROUTER_MODEL = original_openrouter

    def test_get_active_model_name_uses_openai_model(self) -> None:
        original_provider = Config.LLM_PROVIDER
        original_model = Config.OPENAI_MODEL
        try:
            Config.LLM_PROVIDER = "openai"
            Config.OPENAI_MODEL = "gpt-4.1"
            self.assertEqual(Config.get_active_model_name(), "gpt-4.1")
        finally:
            Config.LLM_PROVIDER = original_provider
            Config.OPENAI_MODEL = original_model

    def test_validate_requires_openai_api_key(self) -> None:
        original_provider = Config.LLM_PROVIDER
        original_key = Config.OPENAI_API_KEY
        try:
            Config.LLM_PROVIDER = "openai"
            Config.OPENAI_API_KEY = None
            with self.assertRaises(ValueError):
                Config.validate()
            Config.OPENAI_API_KEY = "test-key"
            Config.validate()
        finally:
            Config.LLM_PROVIDER = original_provider
            Config.OPENAI_API_KEY = original_key

    def test_validate_requires_explicit_openrouter_api_key(self) -> None:
        original_provider = Config.LLM_PROVIDER
        original_openrouter_key = Config.OPENROUTER_API_KEY
        original_openai_key = Config.OPENAI_API_KEY
        try:
            Config.LLM_PROVIDER = "openrouter"
            Config.OPENAI_API_KEY = "openai-key"
            Config.OPENROUTER_API_KEY = None
            with self.assertRaises(ValueError):
                Config.validate()
            Config.OPENROUTER_API_KEY = "openrouter-key"
            Config.validate()
        finally:
            Config.LLM_PROVIDER = original_provider
            Config.OPENROUTER_API_KEY = original_openrouter_key
            Config.OPENAI_API_KEY = original_openai_key

    @patch("src.models.openai.ChatOpenAI")
    def test_openai_wrapper_sets_timeout_and_output_cap(self, mock_chat_openai) -> None:
        original_timeout = Config.LLM_REQUEST_TIMEOUT_SEC
        original_max_output = Config.LLM_MAX_OUTPUT_TOKENS
        original_key = Config.OPENAI_API_KEY
        try:
            Config.LLM_REQUEST_TIMEOUT_SEC = 45
            Config.LLM_MAX_OUTPUT_TOKENS = 2048
            Config.OPENAI_API_KEY = "test-key"
            OpenAIModel(model_name="gpt-4.1-mini")
        finally:
            Config.LLM_REQUEST_TIMEOUT_SEC = original_timeout
            Config.LLM_MAX_OUTPUT_TOKENS = original_max_output
            Config.OPENAI_API_KEY = original_key

        mock_chat_openai.assert_called_once()
        kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 2048)
        self.assertEqual(kwargs["request_timeout"], 45)

    @patch("src.models.openrouter.ChatOpenAI")
    def test_openrouter_wrapper_sets_timeout_and_output_cap(self, mock_chat_openai) -> None:
        original_timeout = Config.LLM_REQUEST_TIMEOUT_SEC
        original_max_output = Config.LLM_MAX_OUTPUT_TOKENS
        original_key = Config.OPENROUTER_API_KEY
        try:
            Config.LLM_REQUEST_TIMEOUT_SEC = 60
            Config.LLM_MAX_OUTPUT_TOKENS = 3072
            Config.OPENROUTER_API_KEY = "test-key"
            OpenRouterModel(model_name="google/gemini-2.5-flash")
        finally:
            Config.LLM_REQUEST_TIMEOUT_SEC = original_timeout
            Config.LLM_MAX_OUTPUT_TOKENS = original_max_output
            Config.OPENROUTER_API_KEY = original_key

        mock_chat_openai.assert_called_once()
        kwargs = mock_chat_openai.call_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 3072)
        self.assertEqual(kwargs["request_timeout"], 60)

    @patch("src.models.gemini.ChatGoogleGenerativeAI")
    def test_gemini_wrapper_sets_timeout_and_output_cap(self, mock_gemini) -> None:
        original_timeout = Config.LLM_REQUEST_TIMEOUT_SEC
        original_max_output = Config.LLM_MAX_OUTPUT_TOKENS
        original_key = Config.GEMINI_API_KEY
        try:
            Config.LLM_REQUEST_TIMEOUT_SEC = 75
            Config.LLM_MAX_OUTPUT_TOKENS = 1024
            Config.GEMINI_API_KEY = "test-key"
            GeminiModel(model_name="gemini-2.5-flash")
        finally:
            Config.LLM_REQUEST_TIMEOUT_SEC = original_timeout
            Config.LLM_MAX_OUTPUT_TOKENS = original_max_output
            Config.GEMINI_API_KEY = original_key

        mock_gemini.assert_called_once()
        kwargs = mock_gemini.call_args.kwargs
        self.assertEqual(kwargs["max_output_tokens"], 1024)
        self.assertEqual(kwargs["timeout"], 75)

    def test_extract_response_text_truncates_pathological_outputs(self) -> None:
        original_limit = Config.LLM_MAX_RESPONSE_CHARS
        try:
            Config.LLM_MAX_RESPONSE_CHARS = 120
            response = type("Resp", (), {"content": "x" * 500})()
            text = extract_response_text(response)
        finally:
            Config.LLM_MAX_RESPONSE_CHARS = original_limit

        self.assertLessEqual(len(text), 120)
        self.assertIn("Response truncated in UI", text)


if __name__ == "__main__":
    unittest.main()
