#!/usr/bin/env python3
"""
Unified Eval Runner for CodeBlue: Data Analytics

Supports both local evaluation and Prime Intellect evaluation with a toggle.

Usage:
    # Local evaluation
    python scripts/eval_runner.py --split eval --backend local --model gpt-4o-mini

    # Prime Intellect evaluation
    python scripts/eval_runner.py --split eval --backend prime --model deepseek/deepseek-chat

    # List available splits and models
    python scripts/eval_runner.py --list-splits
    python scripts/eval_runner.py --list-models
"""
import sys
sys.path.insert(0, 'environments/codeblue-analytics')

import argparse
import asyncio
import json
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import environment functions
from codeblue_analytics import (
    load_environment,
    load_dataframes,
    load_split_config,
    load_questions_registry,
    extract_code,
    compare_outputs,
    get_configs_dir,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Load API keys from config file
CONFIG_PATH = Path.home() / ".pandas-rlvr.env"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Model configurations
MODELS = {
    # Prime Intellect models (use exact model IDs from `prime inference models`)
    "prime": {
        "deepseek/deepseek-chat": {"name": "DeepSeek Chat", "provider": "prime"},
        "deepseek/deepseek-r1": {"name": "DeepSeek R1", "provider": "prime"},
        "meta-llama/llama-3.1-70b-instruct": {"name": "Llama 3.1 70B", "provider": "prime"},
        "meta-llama/llama-4-scout": {"name": "Llama 4 Scout", "provider": "prime"},
        "google/gemini-2.0-flash-001": {"name": "Gemini 2.0 Flash", "provider": "prime"},
        "openai/gpt-4o-mini": {"name": "GPT-4o Mini (Prime)", "provider": "prime"},
        "anthropic/claude-3-haiku-20240307": {"name": "Claude 3 Haiku (Prime)", "provider": "prime"},
        "qwen/qwen3-235b-a22b": {"name": "Qwen 3 235B", "provider": "prime"},
        "mistralai/mistral-large-2411": {"name": "Mistral Large", "provider": "prime"},
    },
    # Direct API models
    "local": {
        "gpt-4o": {"name": "GPT-4o", "provider": "openai"},
        "gpt-4o-mini": {"name": "GPT-4o Mini", "provider": "openai"},
        "gpt-4-turbo": {"name": "GPT-4 Turbo", "provider": "openai"},
        "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "provider": "openai"},
        "claude-sonnet-4-20250514": {"name": "Claude Sonnet 4", "provider": "anthropic"},
        "claude-sonnet-4-5-20250929": {"name": "Claude Sonnet 4.5", "provider": "anthropic"},
        "claude-3-haiku-20240307": {"name": "Claude 3 Haiku", "provider": "anthropic"},
        "gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "provider": "google"},
        "gemini-2.0-flash": {"name": "Gemini 2.0 Flash", "provider": "google"},
    }
}


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    question_id: str
    question: str
    exec_success: bool
    correct: bool
    generated_code: Optional[str]
    result: Any
    expected: Any
    error: Optional[str] = None
    model_id: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = None
    metrics: Dict[str, Any] = None
    prompt: List[Dict] = None


# ============================================================
# API CLIENTS
# ============================================================

async def call_openai(prompt: List[Dict], model: str) -> str:
    """Call OpenAI API."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    data = {"model": model, "messages": prompt}
    
    # Handle O1 and GPT-5 specific parameter requirements
    if model.startswith("o1") or "gpt-5" in model:
        data["max_completion_tokens"] = 5000 # O1/GPT-5 use this instead of max_tokens
        # O1/GPT-5 often don't support temperature or require it to be 1 (default)
    else:
        data["max_tokens"] = 500
        
    result = subprocess.run(
        ['curl', '-s', 'https://api.openai.com/v1/chat/completions',
         '-H', f'Authorization: Bearer {api_key}',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(data)],
        capture_output=True, text=True
    )
    resp = json.loads(result.stdout)
    if 'error' in resp:
        raise ValueError(f"OpenAI API error: {resp['error']}")
    return resp['choices'][0]['message']['content']


async def call_anthropic(prompt: List[Dict], model: str) -> str:
    """Call Anthropic API."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    # Anthropic requires system prompt as separate field
    system_msg = ""
    messages = []
    for m in prompt:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            messages.append({"role": m["role"], "content": m["content"]})

    data = {"model": model, "max_tokens": 500, "system": system_msg, "messages": messages}
    result = subprocess.run(
        ['curl', '-s', 'https://api.anthropic.com/v1/messages',
         '-H', f'x-api-key: {api_key}',
         '-H', 'anthropic-version: 2023-06-01',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(data)],
        capture_output=True, text=True
    )
    resp = json.loads(result.stdout)
    if 'error' in resp:
        raise ValueError(f"Anthropic API error: {resp['error']}")
    if 'content' not in resp:
        raise ValueError(f"Unexpected response: {resp}")
    return resp['content'][0]['text']


async def call_prime(prompt: List[Dict], model: str) -> str:
    """Call Prime Intellect API (OpenAI-compatible)."""
    api_key = os.environ.get('PRIME_API_KEY')
    if not api_key:
        raise ValueError("PRIME_API_KEY not set")

    data = {"model": model, "max_tokens": 500, "messages": prompt}
    result = subprocess.run(
        ['curl', '-s', 'https://api.pinference.ai/api/v1/chat/completions',
         '-H', f'Authorization: Bearer {api_key}',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(data)],
        capture_output=True, text=True
    )
    resp = json.loads(result.stdout)
    if 'error' in resp:
        raise ValueError(f"Prime API error: {resp['error']}")
    if 'choices' not in resp:
        raise ValueError(f"Unexpected response: {resp}")
    return resp['choices'][0]['message']['content']


async def call_openai_completion(prompt: List[Dict], model: str) -> str:
    """Call OpenAI Legacy Completion API (for Codex/Instruct models)."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    # Flatten prompt for completion API
    full_prompt = ""
    for m in prompt:
        if m["role"] == "system":
            full_prompt += f"# {m['content']}\n\n"
        elif m["role"] == "user":
            full_prompt += f"# {m['content']}\n\n"
        elif m["role"] == "assistant":
            full_prompt += f"{m['content']}\n\n"
    
    full_prompt += "```python\n" # Start code block to guide completion

    data = {
        "model": model, 
        "prompt": full_prompt, 
        "max_tokens": 500,
        "temperature": 0,
        "stop": ["```"]
    }
    
    result = subprocess.run(
        ['curl', '-s', 'https://api.openai.com/v1/completions',
         '-H', f'Authorization: Bearer {api_key}',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps(data)],
        capture_output=True, text=True
    )
    resp = json.loads(result.stdout)
    if 'error' in resp:
        raise ValueError(f"OpenAI API error: {resp['error']}")
    return resp['choices'][0]['text']


async def call_model(prompt: List[Dict], model: str, backend: str) -> str:
    """Route model call to appropriate API."""
    if backend == "prime":
        return await call_prime(prompt, model)
    else:
        # Determine provider from model name
        if 'codex' in model or 'instruct' in model: # Heuristic for completion models
             return await call_openai_completion(prompt, model)
        elif 'gpt' in model:
            return await call_openai(prompt, model)
        elif 'claude' in model:
            return await call_anthropic(prompt, model)
        else:
            raise ValueError(f"Unknown model provider for: {model}")


# ============================================================
# EVALUATION LOGIC
# ============================================================

async def evaluate_single(
    example: Dict,
    model: str,
    backend: str,
    dataframes: Dict,
    verbose: bool = False
) -> EvalResult:
    """Evaluate a single example."""
    import pandas as pd
    import numpy as np
    import time
    from datetime import datetime

    start_time = time.time()
    timestamp = datetime.now().isoformat()

    question_id = example['info']['question_id']
    df_name = example['info']['df_name']
    df = dataframes[df_name]
    expected = json.loads(example['info']['expected_output_json'])
    output_type = example['info']['output_type']
    
    # Extract metadata
    metadata = {
        'level': example['info'].get('level', 'unknown'),
        'template': example['info'].get('template', 'unknown'),
        'insight_type': example['info'].get('insight_type', 'unknown'),
        'df_name': df_name
    }

    # Extract question from prompt
    user_msg = example['prompt'][1]['content']
    question = user_msg.split('Question: ')[-1].strip() if 'Question: ' in user_msg else 'N/A'

    try:
        # Call model
        response = await call_model(example['prompt'], model, backend)
        code = extract_code(response)

        if code is None:
            return EvalResult(
                question_id=question_id,
                question=question,
                exec_success=False,
                correct=False,
                generated_code=None,
                result=None,
                expected=expected,
                error="No code extracted from response",
                model_id=model,
                timestamp=timestamp,
                metadata=metadata,
                metrics={'execution_time': time.time() - start_time},
                prompt=example['prompt']
            )

        # Execute code
        local_vars = {'df': df.copy(), 'pd': pd, 'np': np}
        exec(code, {'pd': pd, 'np': np}, local_vars)
        result = local_vars.get('result')

        if result is None:
            return EvalResult(
                question_id=question_id,
                question=question,
                exec_success=True,
                correct=False,
                generated_code=code,
                result=None,
                expected=expected,
                error="No 'result' variable found",
                model_id=model,
                timestamp=timestamp,
                metadata=metadata,
                metrics={'execution_time': time.time() - start_time},
                prompt=example['prompt']
            )

        # Check correctness
        correct = compare_outputs(result, expected, output_type)
        
        return EvalResult(
            question_id=question_id,
            question=question,
            exec_success=True,
            correct=correct,
            generated_code=code,
            result=result,
            expected=expected,
            error=None,
            model_id=model,
            timestamp=timestamp,
            metadata=metadata,
            metrics={'execution_time': time.time() - start_time},
            prompt=example['prompt']
        )

    except Exception as e:
        return EvalResult(
            question_id=question_id,
            question=question,
            exec_success=False,
            correct=False,
            generated_code=None,
            result=None,
            expected=expected,
            error=str(e),
            model_id=model,
            timestamp=timestamp,
            metadata=metadata,
            metrics={'execution_time': time.time() - start_time},
            prompt=example['prompt']
        )



async def run_evaluation(
    split: str,
    model: str,
    backend: str,
    num_examples: int = -1,
    verbose: bool = False
) -> List[EvalResult]:
    """Run evaluation on a split."""

    print("=" * 80)
    print(f"EVALUATION: {model}")
    print(f"Backend: {backend.upper()}")
    print(f"Split: {split}")
    print("=" * 80)

    # Load environment and data
    print("\nLoading environment...")
    env = load_environment(split=split)
    dataframes = load_dataframes()
    dataset = env.dataset

    # Get examples
    examples = [dataset[i] for i in range(len(dataset))]
    if num_examples > 0:
        examples = examples[:num_examples]

    print(f"Evaluating {len(examples)} examples\n")

    # Run evaluation
    results = []
    for i, example in enumerate(examples):
        print(f"[{i+1}/{len(examples)}] {example['info']['question_id']}...", end=" ")

        result = await evaluate_single(example, model, backend, dataframes, verbose)
        results.append(result)

        # Handle potential Series/Array types for boolean check
        is_correct = result.correct
        if hasattr(is_correct, 'all'): # pandas Series or numpy array
            is_correct = bool(is_correct.all())
        
        if is_correct:
            print("PASS")
        else:
            print("FAIL")
            if verbose or not result.exec_success:
                print(f"    Question: {result.question}")
                print(f"    Error: {result.error}")
                if result.generated_code:
                    print(f"    Code: {result.generated_code}")
                print(f"    Got: {result.result}")
                print(f"    Expected: {result.expected}")

    # Summary
    print("\n" + "=" * 80)
    perfect = sum(1 for r in results if r.correct)
    exec_ok = sum(1 for r in results if r.exec_success)
    print(f"Execution Success: {exec_ok}/{len(results)} ({100*exec_ok/len(results):.0f}%)")
    print(f"Correct: {perfect}/{len(results)} ({100*perfect/len(results):.0f}%)")
    print("=" * 80)

    return results


def run_prime_eval(split: str, model: str):
    """Run evaluation using Prime's `prime env eval` command."""
    print("=" * 80)
    print(f"PRIME EVAL: {model}")
    print(f"Split: {split}")
    print("=" * 80)
    print("\nRunning `prime env eval`...\n")

    cmd = ['prime', 'env', 'eval', 'codeblue-analytics', '-m', model, '-s', split]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


# ============================================================
# CLI
# ============================================================

def list_splits():
    """List available splits."""
    print("\nAvailable Splits:")
    print("-" * 40)

    # List config-based splits
    configs_dir = get_configs_dir() / "splits"
    if configs_dir.exists():
        for config_file in sorted(configs_dir.glob("*.yaml")):
            split_name = config_file.stem
            try:
                config = load_split_config(split_name)
                desc = config.get('description', 'No description')
                print(f"  {split_name:20} - {desc}")
            except Exception:
                print(f"  {split_name:20} - (error loading config)")

    # List question sets (can be used directly as splits)
    print("\nQuestion Sets (can be used as splits):")
    print("-" * 40)
    registry = load_questions_registry()
    for qset_name, qset_info in registry.get('question_sets', {}).items():
        count = qset_info.get('count', '?')
        difficulty = qset_info.get('difficulty', '?')
        print(f"  {qset_name:30} - {count} questions, {difficulty}")


def list_models():
    """List available models."""
    print("\nLocal Models (direct API):")
    print("-" * 40)
    for model_id, info in MODELS['local'].items():
        print(f"  {model_id:35} - {info['name']}")

    print("\nPrime Intellect Models:")
    print("-" * 40)
    for model_id, info in MODELS['prime'].items():
        print(f"  {model_id:35} - {info['name']}")


def save_results(results: List[EvalResult], output_path: str):
    """Save evaluation results to JSON file."""
    import pandas as pd
    import numpy as np
    from dataclasses import asdict

    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    output_file = Path(output_path)
    data = [asdict(r) for r in results]
    
    with open(output_file, 'w') as f:
        json.dump(data, f, cls=CustomEncoder, indent=2)
    
    print(f"\nResults saved to {output_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified eval runner for CodeBlue: Data Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local evaluation with GPT-4o-mini
  python scripts/eval_runner.py --split eval --backend local --model gpt-4o-mini

  # Prime evaluation with DeepSeek
  python scripts/eval_runner.py --split eval --backend prime --model deepseek/deepseek-chat

  # Use Prime's built-in eval command
  python scripts/eval_runner.py --split eval --backend prime-native --model deepseek/deepseek-chat

  # List available options
  python scripts/eval_runner.py --list-splits
  python scripts/eval_runner.py --list-models
        """
    )

    parser.add_argument('--split', default='eval', help='Split to evaluate (default: eval)')
    parser.add_argument('--backend', choices=['local', 'prime', 'prime-native'], default='local',
                        help='Evaluation backend: local (direct API), prime (Prime API), prime-native (prime env eval)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model to evaluate')
    parser.add_argument('--output', help='Path to save results JSON')
    parser.add_argument('--num-examples', type=int, default=-1, help='Number of examples (-1 for all)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--list-splits', action='store_true', help='List available splits')
    parser.add_argument('--list-models', action='store_true', help='List available models')

    args = parser.parse_args()

    if args.list_splits:
        list_splits()
        return

    if args.list_models:
        list_models()
        return

    # Run evaluation
    if args.backend == 'prime-native':
        # Use Prime's built-in eval command
        run_prime_eval(args.split, args.model)
    else:
        # Use our unified runner
        results = asyncio.run(run_evaluation(
            split=args.split,
            model=args.model,
            backend=args.backend,
            num_examples=args.num_examples,
            verbose=args.verbose
        ))
        
        if args.output:
            save_results(results, args.output)


if __name__ == '__main__':
    main()
