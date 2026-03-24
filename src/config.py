"""Configuration management for the AI Ops agent."""
import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    SUPPORTED_LLM_PROVIDERS = ('gemini', 'openai', 'openrouter')

    # LLM Provider
    # Supported values: 'gemini', 'openrouter', 'openai'
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini').lower()
    
    # API Configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    # Native OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    # OpenRouter
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'arcee-ai/trinity-large-preview:free')
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
    K8S_DEFAULT_NAMESPACE = os.getenv('K8S_DEFAULT_NAMESPACE', 'default')
    K8S_DRY_RUN = os.getenv('K8S_DRY_RUN', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    K8S_REQUEST_TIMEOUT_SEC = int(os.getenv('K8S_REQUEST_TIMEOUT_SEC', '20'))
    K8S_OUTPUT_MAX_CHARS = int(os.getenv('K8S_OUTPUT_MAX_CHARS', '12000'))
    K8S_CLI_ALLOW_ALL_READ = os.getenv('K8S_CLI_ALLOW_ALL_READ', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    K8S_CLI_ALLOW_ALL_WRITE = os.getenv('K8S_CLI_ALLOW_ALL_WRITE', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    K8S_CLI_READONLY_VERBS = os.getenv(
        'K8S_CLI_READONLY_VERBS',
        'get,describe,logs,top,api-resources,api-versions,cluster-info,version,config,explain,auth,events,diff,wait',
    )
    K8S_CLI_WRITE_ALLOWLIST_VERBS = os.getenv(
        'K8S_CLI_WRITE_ALLOWLIST_VERBS',
        'apply,create,delete,edit,patch,replace,scale,rollout,set,cordon,uncordon,drain,taint,label,annotate,autoscale',
    )

    # AWS CLI tools
    AWS_CLI_ENABLED = os.getenv('AWS_CLI_ENABLED', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_DRY_RUN = os.getenv('AWS_CLI_DRY_RUN', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_TIMEOUT_SEC = int(os.getenv('AWS_CLI_TIMEOUT_SEC', '30'))
    AWS_CLI_PROFILE = os.getenv('AWS_CLI_PROFILE', '').strip()
    AWS_CLI_DEFAULT_REGION = os.getenv('AWS_CLI_DEFAULT_REGION', '').strip()
    AWS_CLI_AUDIT_LOG = os.getenv('AWS_CLI_AUDIT_LOG', os.path.join(LOG_DIRECTORY, 'aws_cli_audit.jsonl'))
    AWS_CLI_AUTO_REGION_RETRY = os.getenv('AWS_CLI_AUTO_REGION_RETRY', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_PROMPT_FOR_REGION_ON_EMPTY = os.getenv('AWS_CLI_PROMPT_FOR_REGION_ON_EMPTY', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_REQUIRE_DEFAULT_REGION_FOR_REGIONAL = os.getenv('AWS_CLI_REQUIRE_DEFAULT_REGION_FOR_REGIONAL', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_AUTO_REGION_FANOUT_MAX = int(os.getenv('AWS_CLI_AUTO_REGION_FANOUT_MAX', '6'))
    AWS_CLI_ALLOW_ALL_READ = os.getenv('AWS_CLI_ALLOW_ALL_READ', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_ALLOW_ALL_WRITE = os.getenv('AWS_CLI_ALLOW_ALL_WRITE', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_ENFORCE_BLOCKLIST = os.getenv('AWS_CLI_ENFORCE_BLOCKLIST', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AWS_CLI_FALLBACK_REGIONS = os.getenv(
        'AWS_CLI_FALLBACK_REGIONS',
        'us-east-1,us-east-2,us-west-2,eu-west-1,eu-central-1,ap-southeast-1,ap-northeast-1',
    )
    AWS_CLI_READONLY_ALLOWLIST = os.getenv(
        'AWS_CLI_READONLY_ALLOWLIST',
        (
            'sts:get-caller-identity,*:list*,*:get*,*:describe*,*:head*,'
            '*:batch-get*,*:lookup*,*:query*,*:search*,*:select*,'
            'cloudwatch:generate-query,logs:filter-log-events,logs:start-query,logs:get-query-results'
        ),
    )
    AWS_CLI_WRITE_ALLOWLIST = os.getenv(
        'AWS_CLI_WRITE_ALLOWLIST',
        (
            'autoscaling:resume-processes,autoscaling:suspend-processes,autoscaling:set-desired-capacity,'
            'autoscaling:update-auto-scaling-group,autoscaling:start-instance-refresh,autoscaling:cancel-instance-refresh,'
            'autoscaling:terminate-instance-in-auto-scaling-group,eks:update-nodegroup-config,eks:update-nodegroup-version,'
            'eks:update-cluster-config,eks:update-cluster-version,eks:update-addon,eks:create-addon,eks:delete-addon,'
            'ecs:update-service,ecs:run-task,ecs:start-task,ecs:stop-task,ecs:execute-command,'
            'ec2:reboot-instances,ec2:start-instances,ec2:stop-instances,ec2:terminate-instances,ec2:create-tags,ec2:delete-tags,'
            'rds:reboot-db-instance,rds:start-db-instance,rds:stop-db-instance,rds:modify-db-instance,'
            'elasticloadbalancing:register-instances-with-load-balancer,elasticloadbalancing:deregister-instances-from-load-balancer,'
            'elbv2:register-targets,elbv2:deregister-targets,elbv2:modify-target-group,elbv2:modify-listener,elbv2:modify-rule,'
            'lambda:update-function-configuration,lambda:update-function-code,lambda:publish-version,lambda:put-function-concurrency,'
            'route53:change-resource-record-sets,route53:create-health-check,route53:update-health-check,route53:delete-health-check,'
            'cloudwatch:put-metric-alarm,cloudwatch:delete-alarms,cloudwatch:disable-alarm-actions,cloudwatch:enable-alarm-actions,'
            'logs:put-retention-policy,logs:delete-retention-policy,logs:put-subscription-filter,logs:delete-subscription-filter,'
            'ssm:send-command,ssm:start-automation-execution,ssm:start-session,ssm:cancel-command,'
            'sns:publish,sns:subscribe,sns:unsubscribe,sqs:send-message,sqs:purge-queue,s3:put-object,s3:delete-object'
        ),
    ).strip()
    AWS_CLI_BLOCKLIST = os.getenv('AWS_CLI_BLOCKLIST', 'iam:*,organizations:*,account:*').strip()
    
    # Agent Configuration
    MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', '8'))
    MAX_TOOL_CALLS_PER_TURN = int(os.getenv('MAX_TOOL_CALLS_PER_TURN', '24'))
    MAX_DUPLICATE_TOOL_CALLS = int(os.getenv('MAX_DUPLICATE_TOOL_CALLS', '3'))
    MAX_SEMANTIC_DUPLICATE_TOOL_CALLS = int(os.getenv('MAX_SEMANTIC_DUPLICATE_TOOL_CALLS', '2'))
    MAX_CHAT_HISTORY_MESSAGES = int(os.getenv('MAX_CHAT_HISTORY_MESSAGES', '10'))
    AGENT_TOOL_RESULT_MAX_CHARS = int(os.getenv('AGENT_TOOL_RESULT_MAX_CHARS', '2500'))
    INCIDENT_STATE_MAX_EVIDENCE = int(os.getenv('INCIDENT_STATE_MAX_EVIDENCE', '8'))
    INCIDENT_STATE_MAX_CACHE_ENTRIES = int(os.getenv('INCIDENT_STATE_MAX_CACHE_ENTRIES', '24'))
    AGENT_ENABLE_DIRECT_READ_ROUTER = os.getenv('AGENT_ENABLE_DIRECT_READ_ROUTER', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AGENT_ENABLE_INTENT_TOOL_FILTER = os.getenv('AGENT_ENABLE_INTENT_TOOL_FILTER', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    COMMAND_SAFETY_POSTURE = os.getenv('COMMAND_SAFETY_POSTURE', 'powerful').strip().lower()
    DEEP_INITIAL_INVESTIGATION = os.getenv('DEEP_INITIAL_INVESTIGATION', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    LLM_RETRY_ON_RATE_LIMIT = os.getenv('LLM_RETRY_ON_RATE_LIMIT', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    LLM_RETRY_MAX_ATTEMPTS = int(os.getenv('LLM_RETRY_MAX_ATTEMPTS', '3'))
    LLM_RETRY_BASE_DELAY_SEC = float(os.getenv('LLM_RETRY_BASE_DELAY_SEC', '1.0'))
    LLM_RETRY_MAX_DELAY_SEC = float(os.getenv('LLM_RETRY_MAX_DELAY_SEC', '8.0'))
    LLM_REQUEST_TIMEOUT_SEC = float(os.getenv('LLM_REQUEST_TIMEOUT_SEC', '90'))
    LLM_MAX_OUTPUT_TOKENS = int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '4096'))
    LLM_MAX_RESPONSE_CHARS = int(os.getenv('LLM_MAX_RESPONSE_CHARS', '24000'))
    VERBOSE = True

    # Tracing (structured JSONL)
    TRACE_ENABLED = os.getenv('TRACE_ENABLED', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    TRACE_DIR = os.getenv('TRACE_DIR', os.path.join(LOG_DIRECTORY, 'traces'))
    TRACE_MAX_FIELD_CHARS = int(os.getenv('TRACE_MAX_FIELD_CHARS', '2000'))
    TRACE_REDACT = os.getenv('TRACE_REDACT', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

    # Autonomous Hybrid Monitoring
    AUTONOMY_ENABLED = os.getenv('AUTONOMY_ENABLED', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AUTONOMY_SCAN_ON_USER_TURN = os.getenv('AUTONOMY_SCAN_ON_USER_TURN', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    AUTONOMY_NAMESPACE = os.getenv('AUTONOMY_NAMESPACE', 'all')
    AUTONOMY_RECENT_MINUTES = int(os.getenv('AUTONOMY_RECENT_MINUTES', '30'))
    AUTONOMY_BASELINE_MINUTES = int(os.getenv('AUTONOMY_BASELINE_MINUTES', '180'))
    AUTONOMY_RESTART_HEAVY_THRESHOLD = int(os.getenv('AUTONOMY_RESTART_HEAVY_THRESHOLD', '3'))
    AUTONOMY_STATE_FILE = os.getenv('AUTONOMY_STATE_FILE', os.path.join(LOG_DIRECTORY, 'autonomy_state.json'))

    # Simple alert monitor thresholds
    ALERT_PENDING_GRACE_MINUTES = int(os.getenv('ALERT_PENDING_GRACE_MINUTES', '15'))
    ALERT_CRITICAL_EVENT_MIN_COUNT = int(os.getenv('ALERT_CRITICAL_EVENT_MIN_COUNT', '3'))
    ALERT_MIN_CONFIDENCE = int(os.getenv('ALERT_MIN_CONFIDENCE', '75'))
    ALERT_REPEAT_MINUTES = int(os.getenv('ALERT_REPEAT_MINUTES', '120'))

    # Alerts and notification channels
    ALERT_COOLDOWN_MINUTES = int(os.getenv('ALERT_COOLDOWN_MINUTES', '30'))
    ALERT_SLACK_WEBHOOK = os.getenv('ALERT_SLACK_WEBHOOK', '').strip()
    ALERT_TEAMS_WEBHOOK = os.getenv('ALERT_TEAMS_WEBHOOK', '').strip()

    ALERT_EMAIL_ENABLED = os.getenv('ALERT_EMAIL_ENABLED', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    ALERT_EMAIL_FROM = os.getenv('ALERT_EMAIL_FROM', '').strip()
    ALERT_EMAIL_TO = os.getenv('ALERT_EMAIL_TO', '').strip()
    ALERT_SMTP_HOST = os.getenv('ALERT_SMTP_HOST', '').strip()
    ALERT_SMTP_PORT = int(os.getenv('ALERT_SMTP_PORT', '587'))
    ALERT_SMTP_USER = os.getenv('ALERT_SMTP_USER', '').strip()
    ALERT_SMTP_PASSWORD = os.getenv('ALERT_SMTP_PASSWORD', '').strip()
    ALERT_SMTP_USE_TLS = os.getenv('ALERT_SMTP_USE_TLS', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    
    @classmethod
    def is_k8s_configured(cls) -> bool:
        """Check if Kubernetes is configured"""
        return bool(shutil.which("kubectl"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if cls.LLM_PROVIDER == 'gemini':
            if not cls.GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found. "
                    "Please set it in .env file or environment variables."
                )
        elif cls.LLM_PROVIDER == 'openai':
            if not cls.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found. "
                    "Please set it in .env file or environment variables."
                )
        elif cls.LLM_PROVIDER == 'openrouter':
            if not cls.OPENROUTER_API_KEY:
                raise ValueError(
                    "OpenRouter API key not found. "
                    "Set OPENROUTER_API_KEY in .env or environment variables."
                )
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {cls.LLM_PROVIDER!r}. Use 'gemini', 'openai', or 'openrouter'."
            )
        
        # Local log directory is no longer required for normal operation.

    @classmethod
    def get_active_model_name(cls) -> str:
        return cls.get_model_name_for_provider(cls.LLM_PROVIDER)

    @classmethod
    def get_model_name_for_provider(cls, provider: str | None) -> str:
        selected = str(provider or cls.LLM_PROVIDER or 'gemini').strip().lower()
        if selected == 'openai':
            return cls.OPENAI_MODEL
        if selected == 'openrouter':
            return cls.OPENROUTER_MODEL
        return cls.GEMINI_MODEL

    @classmethod
    def set_runtime_model_selection(cls, provider: str, model_name: str | None = None) -> str:
        """Apply an in-memory provider/model selection for the current app session."""
        selected_provider = str(provider or cls.LLM_PROVIDER or 'gemini').strip().lower()
        if selected_provider not in cls.SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {selected_provider!r}. "
                "Use 'gemini', 'openai', or 'openrouter'."
            )

        selected_model = str(model_name or '').strip() or cls.get_model_name_for_provider(selected_provider)

        cls.LLM_PROVIDER = selected_provider
        if selected_provider == 'openai':
            cls.OPENAI_MODEL = selected_model
        elif selected_provider == 'openrouter':
            cls.OPENROUTER_MODEL = selected_model
        else:
            cls.GEMINI_MODEL = selected_model

        return selected_model
    
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
