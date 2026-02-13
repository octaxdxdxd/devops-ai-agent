"""OpenRouter LLM wrapper (OpenAI-compatible endpoint) for LangChain."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from ..config import Config


class OpenRouterModel:
    """Wrapper for OpenRouter chat models via ChatOpenAI."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or Config.OPENROUTER_MODEL

        default_headers: dict[str, str] = {}
        if Config.OPENROUTER_SITE_URL:
            default_headers["HTTP-Referer"] = Config.OPENROUTER_SITE_URL
        if Config.OPENROUTER_APP_NAME:
            default_headers["X-Title"] = Config.OPENROUTER_APP_NAME

        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENROUTER_API_KEY,
            openai_api_base=Config.OPENROUTER_BASE_URL,
            default_headers=default_headers or None,
        )

    def get_llm(self):
        return self.llm

    def get_llm_with_tools(self, tools: list):
        return self.llm.bind_tools(tools)
