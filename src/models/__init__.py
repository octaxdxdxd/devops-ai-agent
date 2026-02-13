"""
LLM models package
"""
from .gemini import GeminiModel
from .openrouter import OpenRouterModel

from ..config import Config

__all__ = ['GeminiModel', 'OpenRouterModel', 'get_model']


def get_model(provider: str | None = None, model_name: str | None = None):
	"""Return a configured model wrapper for the selected provider."""
	selected = (provider or Config.LLM_PROVIDER or 'gemini').lower()
	if selected == 'openrouter':
		return OpenRouterModel(model_name=model_name)
	return GeminiModel()
