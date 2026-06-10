"""Quick test: run the agent investigation against Groq to verify no 400 errors."""
import sys, os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.agent import LogAnalysisAgent

test_log = str(PROJECT_ROOT / "test_app.log")
print(f"Testing agent investigation on: {test_log}")

def on_tool(name, args, result):
    args_short = {k: v for k, v in args.items() if k != "file_path"}
    print(f"  → Tool: {name}({args_short}) — status: {result.get('status', 'ok')}")

try:
    agent = LogAnalysisAgent()
    result = agent.investigate(test_log, on_tool_call=on_tool)
    
    print(f"\n✅ SUCCESS — {result['iterations']} iteration(s), {len(result['tool_calls_log'])} tool call(s)")
    print(f"   Model: {result['model']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Root cause: {result['root_cause'][:120]}...")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    sys.exit(1)
