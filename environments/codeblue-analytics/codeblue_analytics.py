"""
CodeBlue: Dataanalytics - RLVR Environment

Train models to generate Pandas code from natural language questions.
"""

import verifiers as vf
from datasets import Dataset
import pandas as pd
import numpy as np
import json
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# CONSTANTS
# ============================================================

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

# ============================================================
# REGISTRY LOADERS
# ============================================================

def get_data_dir() -> Path:
    """Get the data directory relative to this file."""
    return Path(__file__).parent / "data"


def get_configs_dir() -> Path:
    """Get the configs directory relative to this file."""
    return Path(__file__).parent / "configs"


def load_dataset_registry() -> Dict:
    """Load the datasets registry."""
    registry_path = get_data_dir() / "datasets" / "registry.yaml"
    with open(registry_path) as f:
        return yaml.safe_load(f)


def load_questions_registry() -> Dict:
    """Load the questions registry."""
    registry_path = get_data_dir() / "questions" / "registry.yaml"
    with open(registry_path) as f:
        return yaml.safe_load(f)


def load_split_config(split: str) -> Dict:
    """Load a split configuration."""
    # Check if it's a predefined split config
    config_path = get_configs_dir() / "splits" / f"{split}.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Backward compatibility: treat as legacy split name
    # Map old names to new structure
    legacy_mapping = {
        "heart_eval": {"include": [{"question_set": "heart/v1_basic"}]},
        "shopping_eval": {"include": [{"question_set": "shopping/v1_basic"}]},
    }
    if split in legacy_mapping:
        return legacy_mapping[split]

    raise ValueError(f"Unknown split: {split}")


# ============================================================
# DATA LOADERS
# ============================================================

def load_dataframes() -> Dict[str, pd.DataFrame]:
    """Load all DataFrames from the datasets registry."""
    registry = load_dataset_registry()
    data_dir = get_data_dir() / "datasets"

    dfs = {}
    for dataset_name, dataset_info in registry.get("datasets", {}).items():
        csv_path = data_dir / dataset_info["file"]
        if csv_path.exists():
            dfs[dataset_name] = pd.read_csv(csv_path)
    return dfs


def load_questions_from_set(question_set: str, sample: Optional[int] = None) -> List[Dict]:
    """Load questions from a question set (e.g., 'shopping/v1_basic')."""
    registry = load_questions_registry()

    if question_set not in registry.get("question_sets", {}):
        raise ValueError(f"Unknown question set: {question_set}")

    set_info = registry["question_sets"][question_set]
    filepath = get_data_dir() / "questions" / set_info["file"]

    questions = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    # Optional sampling
    if sample and sample < len(questions):
        import random
        questions = random.sample(questions, sample)

    return questions


def load_questions(split: str) -> List[Dict]:
    """Load questions based on split configuration."""
    config = load_split_config(split)

    all_questions = []
    for include_item in config.get("include", []):
        question_set = include_item.get("question_set")
        sample = include_item.get("sample")

        questions = load_questions_from_set(question_set, sample)
        all_questions.extend(questions)

    # Apply global settings
    settings = config.get("settings", {})
    max_questions = settings.get("max_questions")
    if max_questions and len(all_questions) > max_questions:
        all_questions = all_questions[:max_questions]

    if settings.get("shuffle"):
        import random
        random.shuffle(all_questions)

    return all_questions


# Legacy function for backward compatibility
def load_questions_legacy(split: str) -> List[Dict]:
    """Load questions from legacy JSONL file (backward compatibility)."""
    filepath = get_data_dir() / "questions" / f"{split}.jsonl"
    if not filepath.exists():
        return load_questions(split)

    questions = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_df_info(df: pd.DataFrame, df_name: str) -> str:
    """Generate DataFrame info string for prompt."""
    info = f"DataFrame: `df` ({df_name})\n"
    info += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
    info += "Columns and types:\n"
    for col, dtype in df.dtypes.items():
        info += f"  - {col}: {dtype}\n"
    info += f"\nFirst 3 rows:\n{df.head(3).to_markdown(index=False)}"
    return info


def extract_code(response: str) -> str | None:
    """Extract code from <code></code> tags or markdown blocks."""
    # Handle both string and list of messages
    if isinstance(response, list):
        response = response[-1].get("content", "")

    # Try <code> tags first
    match = re.search(r'<code>(.*?)</code>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try markdown code blocks
    match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r'```\n(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def execute_code(code: str, df: pd.DataFrame) -> tuple[bool, Any, str]:
    """
    Execute code safely and return result.
    Returns: (success, result, error_message)
    """
    if code is None:
        return False, None, "No code extracted"

    # Safety check - forbidden operations
    forbidden = ['import os', 'import sys', 'subprocess', 'eval(', 'exec(',
                 '__import__', 'open(', 'file(', 'compile(', 'globals(', 'locals(']
    for f in forbidden:
        if f in code:
            return False, None, f"Forbidden operation: {f}"

    # Allow safe builtins
    safe_builtins = {
        "len": len, "sum": sum, "min": min, "max": max, "abs": abs,
        "round": round, "sorted": sorted, "range": range,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
    }

    try:
        local_vars = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, {"__builtins__": safe_builtins}, local_vars)
        result = local_vars.get("result")
        if result is None:
            return False, None, "No 'result' variable found"
        return True, result, ""
    except Exception as e:
        return False, None, str(e)


def compare_outputs(result: Any, expected: Any, output_type: str) -> bool:
    """Compare result to expected output."""
    try:
        if output_type == "scalar":
            # If result is a Series/DataFrame when scalar expected, it's wrong
            if isinstance(result, (pd.Series, pd.DataFrame)):
                return False
            if isinstance(result, (int, float, np.number)) and isinstance(expected, (int, float)):
                return abs(float(result) - float(expected)) < 0.01
            # Ensure comparison returns a scalar boolean
            comparison = result == expected
            if isinstance(comparison, (pd.Series, pd.DataFrame, np.ndarray)):
                return False
            return bool(comparison)

        elif output_type == "series":
            # Convert result to Series if it's a dict (LLM returned .to_dict())
            if isinstance(result, dict):
                result = pd.Series(result)
            # Convert expected to Series if it's a dict
            if isinstance(expected, dict):
                # Convert dict keys to match result index dtype
                if len(result) > 0 and isinstance(result.index[0], (int, np.integer)):
                    expected = {int(k): v for k, v in expected.items()}
                expected = pd.Series(expected)
            # Sort both by index and compare values with tolerance
            result = result.sort_index()
            expected = expected.sort_index()
            if not result.index.equals(expected.index):
                return False
            # Check values with tolerance for floats
            for idx in result.index:
                r_val, e_val = result[idx], expected[idx]
                if isinstance(r_val, (float, np.floating)) or isinstance(e_val, (float, np.floating)):
                    if abs(float(r_val) - float(e_val)) >= 0.01:
                        return False
                elif r_val != e_val:
                    return False
            return True

        elif output_type == "dataframe":
            if isinstance(expected, dict):
                expected = pd.DataFrame(expected)
            return result.equals(expected)

        elif output_type == "string":
            # String comparison - convert both to string and compare
            return str(result) == str(expected)

        # Fallback for unknown types - direct comparison
        return result == expected
    except Exception:
        return False


# ============================================================
# REWARD FUNCTIONS
# ============================================================

# Global cache for DataFrames (loaded once per environment)
_DATAFRAMES_CACHE = {}


def execution_reward(completion: str, info: Dict, **kwargs) -> float:
    """Reward for code that executes without error."""
    code = extract_code(completion)
    # Load DataFrame on-demand
    df_name = info["df_name"]
    if df_name not in _DATAFRAMES_CACHE:
        _DATAFRAMES_CACHE.update(load_dataframes())
    df = _DATAFRAMES_CACHE[df_name]

    success, _, _ = execute_code(code, df)
    return 1.0 if success else 0.0


def correctness_reward(completion: str, info: Dict, **kwargs) -> float:
    """Reward for code that produces correct output."""
    code = extract_code(completion)
    # Load DataFrame on-demand
    df_name = info["df_name"]
    if df_name not in _DATAFRAMES_CACHE:
        _DATAFRAMES_CACHE.update(load_dataframes())
    df = _DATAFRAMES_CACHE[df_name]

    # Deserialize expected output from JSON
    expected = json.loads(info["expected_output_json"])
    output_type = info["output_type"]

    success, result, _ = execute_code(code, df)
    if not success:
        return 0.0

    return 1.0 if compare_outputs(result, expected, output_type) else 0.0


# ============================================================
# MAIN ENVIRONMENT
# ============================================================

def load_environment(
    split: str = "eval",
    **kwargs
) -> vf.Environment:
    """
    Load the CodeBlue: Dataanalytics environment.

    Args:
        split: Split name - can be:
            - Predefined: "dev", "eval", "full"
            - Legacy: "heart_eval", "shopping_eval"
            - Custom question set: "shopping/v1_basic"

    Returns:
        vf.SingleTurnEnv configured for pandas code generation
    """

    # Load data
    dataframes = load_dataframes()

    # Try new registry-based loading, fall back to legacy
    try:
        questions = load_questions(split)
    except (FileNotFoundError, ValueError):
        questions = load_questions_legacy(split)

    # Build dataset
    dataset_records = []
    for q in questions:
        # Support both old and new schema
        df_name = q.get("dataset") or q.get("df_name")
        df = dataframes[df_name]
        df_info = get_df_info(df, df_name)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{df_info}\n\nQuestion: {q['question']}"}
        ]

        # Extract metadata (new schema) or use defaults (old schema)
        metadata = q.get("metadata", {})
        difficulty = metadata.get("difficulty") or "medium"

        # Map difficulty to level for backward compat
        level_map = {"easy": 1, "medium": 3, "hard": 5}
        level = q.get("level") or level_map.get(difficulty, 3)

        category = metadata.get("computation_type") or q.get("category", "other")

        dataset_records.append({
            "prompt": prompt,
            "answer": q["gold_code"],
            "info": {
                "question_id": q["id"],
                "df_name": df_name,
                "level": level,
                "category": category,
                "expected_output_json": json.dumps(q["expected_output"]),
                "output_type": q["output_type"],
            }
        })

    dataset = Dataset.from_list(dataset_records)

    # Create rubric
    rubric = vf.Rubric(
        funcs=[execution_reward, correctness_reward],
        weights=[0.2, 0.8]
    )

    # Create environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric
    )
