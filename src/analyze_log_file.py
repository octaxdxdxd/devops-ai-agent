import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OPENAI_API_KEY not found in environment variables.")
    exit(1)

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

with open('logs/sample_app.log', 'r') as file:
    logs = file.read()

prompt = f"""You are an expert DevOps engineer analyzing application logs.

Analyze these logs and provide:
1. A summary of what happened
2. The root cause of any issues
3. Recommended actions to fix the problems

Logs:
{logs}

Provide a clear, structed analysis."""

print("Analyzing logs with AI...")
print("-" * 50)

response = client.responses.create(
    model = "meta-llama/llama-3.3-70b-instruct:free",
    input = prompt,
)

print(response.output_text)

print("-" * 50)
print("Log analysis complete.")