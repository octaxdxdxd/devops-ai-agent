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

sample_log = """
2024-10-21 14:23:45 ERROR Database connection failed
2024-10-21 14:23:46 WARN Retry attempt 1 of 3
2024-10-21 14:23:48 ERROR Database connection failed
2024-10-21 14:23:49 WARN Retry attempt 2 of 3
2024-10-21 14:23:51 ERROR Database connection failed
2024-10-21 14:23:52 ERROR Maximum retries reached
"""

prompt = f"""You are a DevOps engineer analyzing application logs.
Analyze this log and explain what's happening:

{sample_log}

Provide a brief analysis and suggest what might be wrong."""

print("Analyzing logs with AI...")
print("-" * 50)

response = client.responses.create(
  model="meta-llama/llama-3.3-70b-instruct:free",
  input=prompt,
)

print(response.output_text)

print("-" * 50)
print("Setup successful! Your environment is ready.")