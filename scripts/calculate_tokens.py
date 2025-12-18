import json
import pandas as pd
from pathlib import Path
import collections
import io

# Setup paths
BASE_DIR = Path("environments/codeblue-analytics")
DATA_DIR = BASE_DIR / "data"
JSONL_PATH = DATA_DIR / "questions" / "bank" / "v2_hard.jsonl"
CSV_PATH = DATA_DIR / "datasets" / "bank" / "data.csv"

# System Prompt
SYSTEM_PROMPT = '''You are a Python data analyst expert.
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
</code>'''

def calculate_tokens():
    # Load DataFrame for context
    df = pd.read_csv(CSV_PATH)
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()
    df_head = df.head().to_markdown()
    
    context = f"""
DataFrame Info:
{df_info}

First 5 rows:
{df_head}
"""

    total_input_chars = 0
    total_output_chars = 0
    question_count = 0
    
    templates = collections.Counter()
    levels = collections.Counter()
    
    print(f"Analyzing {JSONL_PATH}...")
    
    with open(JSONL_PATH, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_count += 1
            
            # Metadata
            templates[data.get('template', 'unknown')] += 1
            levels[data.get('level', 'unknown')] += 1
            
            # Input: System Prompt + Context + Question
            prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {data['question']}"
            total_input_chars += len(prompt)
            
            # Output: Gold Code
            total_output_chars += len(data['gold_code'])
            
    # Approximation: 4 chars = 1 token
    avg_input_tokens = (total_input_chars / 4.0) / question_count
    avg_output_tokens = (total_output_chars / 4.0) / question_count
    
    print(f"\nAnalysis of {question_count} questions:")
    print(f"Average Input Tokens: {avg_input_tokens:.1f}")
    print(f"Average Output Tokens (Gold Code): {avg_output_tokens:.1f}")
    print(f"Total Input Tokens (Benchmark Run): {int(avg_input_tokens * question_count)}")
    print(f"Total Output Tokens (Benchmark Run - Gold): {int(avg_output_tokens * question_count)}")
    
    print("\nTemplate Distribution:")
    for t, c in templates.most_common():
        print(f"  - {t}: {c}")
        
    print("\nLevel Distribution:")
    for l, c in levels.most_common():
        print(f"  - {l}: {c}")

if __name__ == "__main__":
    calculate_tokens()
