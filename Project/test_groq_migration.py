"""
test_groq_migration.py
======================
Standalone integration test to verify the Groq Cloud API migration.

This script:
  1. Loads GROQ_API_KEY from the .env file
  2. Initializes an OpenAI client pointed at https://api.groq.com/openai/v1
  3. Sends a basic prompt to llama-3.3-70b-versatile
  4. Validates authentication and response

Run:
    cd Project
    uv run python test_groq_migration.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"
errors = 0

print("=" * 60)
print("  GROQ CLOUD MIGRATION — INTEGRATION TEST")
print("=" * 60)

# ── Test 1: API Key loaded ──────────────────────────────────────────────
print("\n[1/4] Checking GROQ_API_KEY...")
api_key = os.getenv("GROQ_API_KEY", "")
if not api_key or api_key == "your-groq-api-key-here" or api_key == "gsk_yourKeyHere":
    print(f"  {FAIL}  GROQ_API_KEY is not set or is placeholder")
    print("         Get a FREE key at: https://console.groq.com/keys")
    print("         Then add it to .env: GROQ_API_KEY=gsk_your_key_here")
    sys.exit(1)
else:
    print(f"  {PASS}  GROQ_API_KEY is configured (length: {len(api_key)})")

# ── Test 2: OpenAI client initializes ───────────────────────────────────
print("\n[2/4] Initializing OpenAI client with Groq base URL...")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    print(f"  {PASS}  Client initialized (base_url=https://api.groq.com/openai/v1)")
except Exception as e:
    print(f"  {FAIL}  Failed to initialize client: {e}")
    sys.exit(1)

# ── Test 3: Basic API call ──────────────────────────────────────────────
print("\n[3/4] Sending test prompt to llama-3.3-70b-versatile...")
start_time = time.time()
try:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
            {"role": "user", "content": "Say 'Hello from Groq!' and nothing else."},
        ],
        temperature=0.1,
        max_tokens=50,
    )
    elapsed = time.time() - start_time

    # Validate response structure
    assert response.choices, "No choices in response"
    content = response.choices[0].message.content
    assert content and len(content) > 0, "Empty response content"

    print(f"  {PASS}  API call succeeded ({elapsed:.2f}s)")
    print(f"         Model: {response.model}")
    print(f"         Response: {content.strip()}")
    if response.usage:
        print(f"         Tokens — prompt: {response.usage.prompt_tokens}, "
              f"completion: {response.usage.completion_tokens}, "
              f"total: {response.usage.total_tokens}")

except Exception as e:
    elapsed = time.time() - start_time
    err_str = str(e)
    print(f"  {FAIL}  API call failed ({elapsed:.2f}s)")
    print(f"         Error: {err_str[:300]}")
    if "401" in err_str:
        print("         → 401 Unauthorized: Check your GROQ_API_KEY")
    elif "400" in err_str:
        print("         → 400 Bad Request: Check model name or payload format")
    elif "404" in err_str:
        print("         → 404 Not Found: Model may not be available")
    errors += 1

# ── Test 4: GroqClient wrapper works ────────────────────────────────────
print("\n[4/4] Testing GroqClient wrapper from src/llm_client.py...")
try:
    from src.llm_client import GroqClient
    groq = GroqClient(api_key=api_key)
    assert groq.model_name == "llama-3.3-70b-versatile", f"Unexpected default model: {groq.model_name}"
    print(f"  {PASS}  GroqClient initialized — default model: {groq.model_name}")
except Exception as e:
    print(f"  {FAIL}  GroqClient initialization failed: {e}")
    errors += 1

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors == 0:
    print(f"  \033[92m🎉 ALL TESTS PASSED — Groq Cloud migration is working!\033[0m")
else:
    print(f"  \033[91m⚠️  {errors} test(s) failed — see above\033[0m")
print("=" * 60 + "\n")

sys.exit(1 if errors > 0 else 0)
