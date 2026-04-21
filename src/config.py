"""Central application configuration loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


class Config:
    """Singleton-style config populated once from environment."""

    SUPPORTED_LLM_PROVIDERS = ("gemini", "openai", "openrouter")

    # ── LLM ──────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = _env("LLM_PROVIDER", "openrouter")
    TEMPERATURE: float = _env_float("TEMPERATURE", 0.1)
    LLM_MAX_OUTPUT_TOKENS: int = _env_int("LLM_MAX_OUTPUT_TOKENS", 4096)
    LLM_REQUEST_TIMEOUT_SEC: int = _env_int("LLM_REQUEST_TIMEOUT_SEC", 120)

    # Gemini
    GEMINI_API_KEY: str = _env("GEMINI_API_KEY")
    GEMINI_MODEL: str = _env("GEMINI_MODEL", "gemini-2.5-flash")

    # OpenAI
    OPENAI_API_KEY: str = _env("OPENAI_API_KEY")
    OPENAI_MODEL: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL: str | None = _env("OPENAI_BASE_URL") or None

    # OpenRouter
    OPENROUTER_API_KEY: str = _env("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = _env("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    OPENROUTER_BASE_URL: str = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_SITE_URL: str = _env("OPENROUTER_SITE_URL")
    OPENROUTER_APP_NAME: str = _env("OPENROUTER_APP_NAME", "AIOps Agent")

    # ── Kubernetes ───────────────────────────────────────────────────────
    K8S_CONTEXT: str = _env("K8S_CONTEXT")
    K8S_DEFAULT_NAMESPACE: str = _env("K8S_DEFAULT_NAMESPACE", "default")
    K8S_CLI_ALLOW_ALL_READ: bool = _env_bool("K8S_CLI_ALLOW_ALL_READ", True)
    K8S_CLI_ALLOW_ALL_WRITE: bool = _env_bool("K8S_CLI_ALLOW_ALL_WRITE", False)
    K8S_ALLOWED_NAMESPACES: str = _env("K8S_ALLOWED_NAMESPACES")
    K8S_BLOCKED_NAMESPACES: str = _env("K8S_BLOCKED_NAMESPACES")

    # ── AWS ──────────────────────────────────────────────────────────────
    AWS_CLI_ENABLED: bool = _env_bool("AWS_CLI_ENABLED", True)
    AWS_CLI_ALLOW_ALL_READ: bool = _env_bool("AWS_CLI_ALLOW_ALL_READ", True)
    AWS_CLI_ALLOW_ALL_WRITE: bool = _env_bool("AWS_CLI_ALLOW_ALL_WRITE", False)
    AWS_CLI_PROFILE: str = _env("AWS_CLI_PROFILE")
    AWS_CLI_DEFAULT_REGION: str = _env("AWS_CLI_DEFAULT_REGION")
    AWS_ALLOWED_REGIONS: str = _env("AWS_ALLOWED_REGIONS")
    AWS_BLOCKED_REGIONS: str = _env("AWS_BLOCKED_REGIONS")

    # ── Safety ───────────────────────────────────────────────────────────
    COMMAND_SAFETY_POSTURE: str = _env("COMMAND_SAFETY_POSTURE", "approval_required")

    # ── Tracing ──────────────────────────────────────────────────────────
    TRACE_ENABLED: bool = _env_bool("TRACE_ENABLED", True)
    TRACE_DIR: str = _env("TRACE_DIR", "traces")

    # ── Autonomy ─────────────────────────────────────────────────────────
    AUTONOMY_ENABLED: bool = _env_bool("AUTONOMY_ENABLED", True)
    AUTONOMY_SCAN_ON_USER_TURN: bool = _env_bool("AUTONOMY_SCAN_ON_USER_TURN", True)
    AUTONOMY_BACKGROUND_SCAN_INTERVAL_SEC: int = _env_int("AUTONOMY_BACKGROUND_SCAN_INTERVAL_SEC", 300)
    AUTONOMY_ALERT_CACHE_MAX_AGE_SEC: int = _env_int("AUTONOMY_ALERT_CACHE_MAX_AGE_SEC", 600)
    AUTONOMY_NAMESPACE: str = _env("AUTONOMY_NAMESPACE")
    AUTONOMY_RECENT_MINUTES: int = _env_int("AUTONOMY_RECENT_MINUTES", 15)

    # ── Alerts ───────────────────────────────────────────────────────────
    ALERT_PENDING_GRACE_MINUTES: int = _env_int("ALERT_PENDING_GRACE_MINUTES", 5)
    ALERT_CRITICAL_EVENT_MIN_COUNT: int = _env_int("ALERT_CRITICAL_EVENT_MIN_COUNT", 3)

    # ── Chat ─────────────────────────────────────────────────────────────
    MAX_CHAT_HISTORY_MESSAGES: int = _env_int("MAX_CHAT_HISTORY_MESSAGES", 14)

    # ── Agent ────────────────────────────────────────────────────────────
    DIAGNOSE_MAX_STEPS: int = _env_int("DIAGNOSE_MAX_STEPS", 15)
    DIAGNOSE_CHECKPOINT_STEP: int = _env_int("DIAGNOSE_CHECKPOINT_STEP", 7)
    LOOKUP_MAX_STEPS: int = _env_int("LOOKUP_MAX_STEPS", 8)
    LOOKUP_CHECKPOINT_STEP: int = _env_int("LOOKUP_CHECKPOINT_STEP", 3)
    EXPLAIN_MAX_STEPS: int = _env_int("EXPLAIN_MAX_STEPS", 10)
    ACTION_MAX_STEPS: int = _env_int("ACTION_MAX_STEPS", 8)

    @classmethod
    def validate(cls) -> None:
        provider = cls.LLM_PROVIDER.lower()
        if provider not in cls.SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}. Choose from {cls.SUPPORTED_LLM_PROVIDERS}")
        if provider == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when using the Gemini provider")
        if provider == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when using the OpenAI provider")
        if provider == "openrouter" and not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required when using the OpenRouter provider")
        posture = str(cls.COMMAND_SAFETY_POSTURE or "approval_required").strip().lower()
        if posture not in {"approval_required", "read_only"}:
            raise ValueError("COMMAND_SAFETY_POSTURE must be 'approval_required' or 'read_only'")

    @classmethod
    def get_active_model_name(cls) -> str:
        return cls.get_model_name_for_provider(cls.LLM_PROVIDER)

    @classmethod
    def get_model_name_for_provider(cls, provider: str) -> str:
        p = (provider or cls.LLM_PROVIDER).lower()
        if p == "openai":
            return cls.OPENAI_MODEL
        if p == "openrouter":
            return cls.OPENROUTER_MODEL
        return cls.GEMINI_MODEL

    @classmethod
    def set_runtime_model_selection(cls, provider: str, model_name: str = "") -> str:
        p = provider.lower()
        if p not in cls.SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"Unsupported provider: {p}")
        cls.LLM_PROVIDER = p
        name = model_name.strip() or cls.get_model_name_for_provider(p)
        if p == "gemini":
            cls.GEMINI_MODEL = name
        elif p == "openai":
            cls.OPENAI_MODEL = name
        elif p == "openrouter":
            cls.OPENROUTER_MODEL = name
        return name
