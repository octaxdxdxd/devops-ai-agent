"""Configuration management for the AI Ops agent."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # LLM Provider
    # Supported values: 'gemini', 'openrouter'
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini').lower()
    
    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    # OpenRouter (aliases: OPENAI_API_KEY / OPENAI_MODEL for convenience)
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL') or os.getenv('OPENAI_MODEL') or 'arcee-ai/trinity-large-preview:free'
    OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
    # Optional headers OpenRouter recommends
    OPENROUTER_SITE_URL = os.getenv('OPENROUTER_SITE_URL')
    OPENROUTER_APP_NAME = os.getenv('OPENROUTER_APP_NAME')
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
    
    # Legacy path setting retained for backward compatibility (not used for local log tools anymore)
    LOG_DIRECTORY = os.getenv('LOG_DIRECTORY', 'logs')
    
    # Kubernetes Configuration
    K8S_KUBECONFIG = os.getenv('K8S_KUBECONFIG', '')
    K8S_CONTEXT = os.getenv('K8S_CONTEXT', '')
    K8S_DEFAULT_NAMESPACE = os.getenv('K8S_DEFAULT_NAMESPACE', 'production')
    K8S_DRY_RUN = os.getenv('K8S_DRY_RUN', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    K8S_REQUEST_TIMEOUT_SEC = int(os.getenv('K8S_REQUEST_TIMEOUT_SEC', '20'))
    K8S_OUTPUT_MAX_CHARS = int(os.getenv('K8S_OUTPUT_MAX_CHARS', '12000'))
    
    # Agent Configuration
    MAX_ITERATIONS = 5
    MAX_TOOL_CALLS_PER_TURN = int(os.getenv('MAX_TOOL_CALLS_PER_TURN', '12'))
    MAX_DUPLICATE_TOOL_CALLS = int(os.getenv('MAX_DUPLICATE_TOOL_CALLS', '2'))
    VERBOSE = True

    # Tracing (structured JSONL)
    TRACE_ENABLED = os.getenv('TRACE_ENABLED', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    TRACE_DIR = os.getenv('TRACE_DIR', os.path.join(LOG_DIRECTORY, 'traces'))
    TRACE_MAX_FIELD_CHARS = int(os.getenv('TRACE_MAX_FIELD_CHARS', '2000'))
    TRACE_REDACT = os.getenv('TRACE_REDACT', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    
    @classmethod
    def is_k8s_configured(cls) -> bool:
        """Check if Kubernetes is configured"""
        # Provider-agnostic: if kubectl can reach any configured context, tools can run.
        # This supports EKS/AKS/GKE as long as kubeconfig auth is already set up.
        return True
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if cls.LLM_PROVIDER == 'gemini':
            if not cls.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found. "
                    "Please set it in .env file or environment variables."
                )
        elif cls.LLM_PROVIDER == 'openrouter':
            if not cls.OPENROUTER_API_KEY:
                raise ValueError(
                    "OpenRouter API key not found. "
                    "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env or environment variables."
                )
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {cls.LLM_PROVIDER!r}. Use 'gemini' or 'openrouter'."
            )
        
        # Local log directory is no longer required for normal operation.

    @classmethod
    def get_active_model_name(cls) -> str:
        if cls.LLM_PROVIDER == 'openrouter':
            return cls.OPENROUTER_MODEL
        return cls.GEMINI_MODEL
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for the agent"""
        prompt_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'system_prompt.txt'
        )
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(
                f"System prompt file not found: {prompt_file}\n"
                "Please ensure system_prompt.txt exists in the project root."
            )
