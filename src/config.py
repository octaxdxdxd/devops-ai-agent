"""Lean configuration for the rebuilt AI Ops backend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_text(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_text(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env_text(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env_text(name, str(default)))


class Config:
    """Centralized runtime configuration.

    The new backend keeps only the values that materially affect runtime behavior.
    A few legacy UI-facing flags remain as fixed compatibility constants.
    """

    SUPPORTED_LLM_PROVIDERS = ("gemini", "openai", "openrouter")

    LOG_DIRECTORY = _env_text("LOG_DIRECTORY", "logs")

    # Model/runtime selection
    LLM_PROVIDER = _env_text("LLM_PROVIDER", "gemini").lower()
    TEMPERATURE = _env_float("TEMPERATURE", 0.05)
    LLM_REQUEST_TIMEOUT_SEC = _env_float("LLM_REQUEST_TIMEOUT_SEC", 120.0)
    LLM_MAX_OUTPUT_TOKENS = _env_int("LLM_MAX_OUTPUT_TOKENS", 6144)
    LLM_MAX_RESPONSE_CHARS = _env_int("LLM_MAX_RESPONSE_CHARS", 32000)
    MAX_CHAT_HISTORY_MESSAGES = _env_int("MAX_CHAT_HISTORY_MESSAGES", 14)

    # Provider credentials / defaults
    GEMINI_API_KEY = _env_text("GEMINI_API_KEY")
    GEMINI_MODEL = _env_text("GEMINI_MODEL", "gemini-2.5-flash")

    OPENAI_API_KEY = _env_text("OPENAI_API_KEY")
    OPENAI_MODEL = _env_text("OPENAI_MODEL", "gpt-4.1-mini")
    OPENAI_BASE_URL = _env_text("OPENAI_BASE_URL", "https://api.openai.com/v1")

    OPENROUTER_API_KEY = _env_text("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = _env_text("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    OPENROUTER_BASE_URL = _env_text("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_SITE_URL = _env_text("OPENROUTER_SITE_URL")
    OPENROUTER_APP_NAME = _env_text("OPENROUTER_APP_NAME")

    # LLM retry/backoff
    LLM_RETRY_ON_RATE_LIMIT = _env_bool("LLM_RETRY_ON_RATE_LIMIT", True)
    LLM_RETRY_MAX_ATTEMPTS = _env_int("LLM_RETRY_MAX_ATTEMPTS", 3)
    LLM_RETRY_BASE_DELAY_SEC = _env_float("LLM_RETRY_BASE_DELAY_SEC", 1.0)
    LLM_RETRY_MAX_DELAY_SEC = _env_float("LLM_RETRY_MAX_DELAY_SEC", 8.0)

    # Investigation loop
    AGENT_MAX_STEPS = _env_int("AGENT_MAX_STEPS", 6)
    AGENT_MAX_ACTIONS_PER_STEP = _env_int("AGENT_MAX_ACTIONS_PER_STEP", 3)

    # Kubernetes / Helm
    K8S_KUBECONFIG = _env_text("K8S_KUBECONFIG")
    K8S_CONTEXT = _env_text("K8S_CONTEXT")
    K8S_DEFAULT_NAMESPACE = _env_text("K8S_DEFAULT_NAMESPACE")
    K8S_DRY_RUN = _env_bool("K8S_DRY_RUN", False)
    K8S_REQUEST_TIMEOUT_SEC = _env_int("K8S_REQUEST_TIMEOUT_SEC", 30)
    TOOL_STRUCTURED_OUTPUT_MAX_CHARS = _env_int("TOOL_STRUCTURED_OUTPUT_MAX_CHARS", 50000)
    HELM_TIMEOUT_SEC = _env_int("HELM_TIMEOUT_SEC", 45)

    # AWS
    AWS_CLI_DRY_RUN = _env_bool("AWS_CLI_DRY_RUN", False)
    AWS_CLI_TIMEOUT_SEC = _env_int("AWS_CLI_TIMEOUT_SEC", 45)
    AWS_CLI_PROFILE = _env_text("AWS_CLI_PROFILE")
    AWS_CLI_DEFAULT_REGION = _env_text("AWS_CLI_DEFAULT_REGION")
    AWS_CLI_AUTO_REGION_FANOUT_MAX = _env_int("AWS_CLI_AUTO_REGION_FANOUT_MAX", 8)
    AWS_CLI_FALLBACK_REGIONS = _env_text(
        "AWS_CLI_FALLBACK_REGIONS",
        "us-east-1,us-east-2,us-west-2,eu-west-1,eu-central-1,ap-southeast-1,ap-northeast-1",
    )

    # Tracing
    TRACE_ENABLED = _env_bool("TRACE_ENABLED", False)
    TRACE_DIR = _env_text("TRACE_DIR", str(Path(LOG_DIRECTORY) / "traces"))
    TRACE_MAX_FIELD_CHARS = _env_int("TRACE_MAX_FIELD_CHARS", 4000)
    TRACE_REDACT = _env_bool("TRACE_REDACT", True)

    # Compatibility flags for the existing UI. These are fixed by design now.
    COMMAND_SAFETY_POSTURE = "approval-gated-full-access"
    K8S_CLI_ALLOW_ALL_READ = True
    K8S_CLI_ALLOW_ALL_WRITE = True
    AWS_CLI_ENABLED = True
    AWS_CLI_ALLOW_ALL_READ = True
    AWS_CLI_ALLOW_ALL_WRITE = True

    # The old background-autonomy subsystem is gone; keep the UI compatible without
    # spending tokens on idle scans unless explicitly triggered.
    AUTONOMY_ENABLED = False
    AUTONOMY_SCAN_ON_USER_TURN = False
    AUTONOMY_NAMESPACE = "all"
    AUTONOMY_RECENT_MINUTES = 30
    AUTONOMY_BACKGROUND_SCAN_INTERVAL_SEC = 300
    AUTONOMY_ALERT_CACHE_MAX_AGE_SEC = 600
    ALERT_PENDING_GRACE_MINUTES = 15
    ALERT_CRITICAL_EVENT_MIN_COUNT = 3

    @classmethod
    def is_k8s_configured(cls) -> bool:
        return bool(shutil.which("kubectl"))

    @classmethod
    def validate(cls) -> None:
        provider = str(cls.LLM_PROVIDER or "gemini").strip().lower()
        if provider not in cls.SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {provider!r}. Use one of {', '.join(cls.SUPPORTED_LLM_PROVIDERS)}."
            )

        if provider == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env or environment variables.")
        if provider == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found. Set it in .env or environment variables.")
        if provider == "openrouter" and not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found. Set it in .env or environment variables.")

    @classmethod
    def get_active_model_name(cls) -> str:
        return cls.get_model_name_for_provider(cls.LLM_PROVIDER)

    @classmethod
    def get_model_name_for_provider(cls, provider: str | None) -> str:
        selected = str(provider or cls.LLM_PROVIDER or "gemini").strip().lower()
        if selected == "openai":
            return cls.OPENAI_MODEL
        if selected == "openrouter":
            return cls.OPENROUTER_MODEL
        return cls.GEMINI_MODEL

    @classmethod
    def set_runtime_model_selection(cls, provider: str, model_name: str | None = None) -> str:
        selected_provider = str(provider or cls.LLM_PROVIDER or "gemini").strip().lower()
        if selected_provider not in cls.SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {selected_provider!r}. Use one of {', '.join(cls.SUPPORTED_LLM_PROVIDERS)}."
            )

        selected_model = str(model_name or "").strip() or cls.get_model_name_for_provider(selected_provider)
        cls.LLM_PROVIDER = selected_provider
        if selected_provider == "openai":
            cls.OPENAI_MODEL = selected_model
        elif selected_provider == "openrouter":
            cls.OPENROUTER_MODEL = selected_model
        else:
            cls.GEMINI_MODEL = selected_model
        return selected_model

    @classmethod
    def get_system_prompt(cls) -> str:
        prompt_file = Path(__file__).resolve().parent.parent / "system_prompt.txt"
        try:
            return prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValueError(
                f"System prompt file not found: {prompt_file}\n"
                "Please ensure system_prompt.txt exists in the project root."
            ) from exc
