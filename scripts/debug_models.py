import asyncio
import os
import json
from pathlib import Path
from eval_runner import call_prime

# Mock example prompt
PROMPT = """You are a Python data analyst expert.
Given a DataFrame and a question, write pandas code to answer it.

Rules:
- The DataFrame is already loaded as `df`
- Store your final answer in a variable called `result`
- Only use pandas (pd) and numpy (np) - already imported
- Write minimal, clean code
- Do not print anything

Put your code inside <code></code> tags.

Example:
<code>
result = df['column'].sum()
</code>

Question: Divide customers into 4 day quartiles. What is the subscription percentage (0-100) in the lower-middle (25-50%) (Q2) quartile?
"""

async def debug_model(model):
    print(f"\n{'='*80}")
    print(f"DEBUGGING MODEL: {model}")
    print(f"{'='*80}")
    
    try:
        print("Sending request...")
        # Format prompt as list of messages
        messages = [{"role": "user", "content": PROMPT}]
        response = await call_prime(messages, model)
        print("\n--- RAW RESPONSE START ---")
        print(response)
        print("--- RAW RESPONSE END ---\n")
        
        # Check for code tags
        if "<code>" in response and "</code>" in response:
            print("✅ Found <code> tags")
        else:
            print("❌ No <code> tags found")
            
        # Check for markdown code blocks
        if "```python" in response or "```" in response:
            print("⚠️ Found markdown code blocks (fallback)")
            
    except Exception as e:
        print(f"❌ Error calling model: {e}")

async def main():
    models = [
        "qwen/qwen-2.5-72b-instruct"
    ]
    
    for model in models:
        await debug_model(model)

if __name__ == "__main__":
    asyncio.run(main())
