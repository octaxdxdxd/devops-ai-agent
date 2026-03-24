"""
Google Gemini LLM wrapper
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import Config


class GeminiModel:
    """Wrapper for Google Gemini LLM"""
    
    def __init__(self, model_name: str | None = None):
        """Initialize the Gemini model"""
        self.model_name = model_name or Config.GEMINI_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.LLM_MAX_OUTPUT_TOKENS,
            timeout=Config.LLM_REQUEST_TIMEOUT_SEC,
        )
    
    def get_llm(self):
        """Get the LLM instance"""
        return self.llm
    
    def get_llm_with_tools(self, tools: list):
        """Get LLM with tools bound"""
        return self.llm.bind_tools(tools)
