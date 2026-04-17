# AIOps Agent

AIOps assistant for Kubernetes and AWS investigation, diagnostics, and approval-gated remediation.

## Prerequisites

- Python 3.11+
- `pip`
- Optional: configured `kubectl` context for Kubernetes operations
- Optional: configured AWS credentials for AWS operations
- One LLM provider API key

## Required Environment Variables

The app validates the selected LLM provider on startup.

At minimum, set one provider and its matching API key:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
```

Other common options:

```env
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Or use Gemini instead
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your_key_here
# GEMINI_MODEL=gemini-2.5-flash

# Or use OpenAI instead
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_key_here
# OPENAI_MODEL=gpt-4o-mini

# Optional runtime settings
AWS_CLI_DEFAULT_REGION=us-east-1
K8S_DEFAULT_NAMESPACE=default
TRACE_ENABLED=true
```

Start from the example file:

```bash
cp .env.example .env
```

Then edit `.env` and fill in the provider/API key you want to use.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Start the Project

Run the Streamlit app from the repository root:

```bash
streamlit run app.py
```

By default, Streamlit runs on:

```text
http://localhost:8501
```

Open that URL in your browser.

## Optional AWS and Kubernetes Setup

For Kubernetes features, make sure `kubectl` works before starting the app:

```bash
kubectl config current-context
kubectl get ns
```

For AWS features, make sure your AWS credentials are available:

```bash
aws sts get-caller-identity
```

If you use a named profile, set it in `.env`:

```env
AWS_CLI_PROFILE=your-profile
AWS_CLI_DEFAULT_REGION=us-east-1
```