"""Regression tests for native OpenAI provider wiring."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import Config
from src.models import get_model


class OpenAIProviderTests(unittest.TestCase):
    def test_get_model_dispatches_to_openai_wrapper(self) -> None:
        with patch("src.models.OpenAIModel") as mock_openai:
            sentinel = object()
            mock_openai.return_value = sentinel
            selected = get_model(provider="openai", model_name="gpt-4.1-mini")
            mock_openai.assert_called_once_with(model_name="gpt-4.1-mini")
            self.assertIs(selected, sentinel)

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


if __name__ == "__main__":
    unittest.main()
