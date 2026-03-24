"""Native OpenAI chat-model wrapper for LangChain."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from ..config import Config


class OpenAIModel:
    """Wrapper for OpenAI chat models via ChatOpenAI."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or Config.OPENAI_MODEL
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=Config.TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY,
            openai_api_base=Config.OPENAI_BASE_URL,
        )

    def get_llm(self):
        return self.llm

    def get_llm_with_tools(self, tools: list):
        return self.llm.bind_tools(tools)
