# Runbook: codeblue-analytics

## Purpose
**One-shot evaluation benchmark** for LLMs on pandas data analysis tasks. Unlike codeblue-env (multi-turn with tools), this tests single-response code generation across multiple datasets.

- **172 questions** across 5 datasets
- **No tools** - model generates code in one shot
- **Exact match** verification via code execution

## Prerequisites
- Python 3.10+
- API keys in `~/.pandas-rlvr.env`:
```bash
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
PRIME_API_KEY=...
```

## Quick Start

```bash
cd codeblue-analytics

# Run evaluation on a split
python scripts/eval_runner.py --split bank/v2_hard --backend local --model gpt-4o-mini

# List available splits
python scripts/eval_runner.py --list-splits

# List available models
python scripts/eval_runner.py --list-models
```

## Datasets

| Dataset | Rows | Columns | Domain |
|---------|------|---------|--------|
| **bank** | 750,000 | 18 | Finance - marketing campaigns |
| **shopping** | 3,900 | 18 | E-commerce - purchase patterns |
| **heart** | 918 | 12 | Healthcare - clinical records |
| **loan** | 32,581 | 12 | Finance - loan applications |
| **road** | 12,316 | 21 | Safety - accident data |

## Question Splits

### Easy Splits
| Split | Questions | Skills |
|-------|-----------|--------|
| `shopping/v1_basic` | 6 | duplicated, isnull, value_counts |
| `heart/v1_basic` | 4 | groupby, mean, value_counts |

### Mixed Difficulty
| Split | Questions | Skills |
|-------|-----------|--------|
| `bank/v1_validated` | 18 | groupby, filtering, percentage |
| `shopping/v1_mined` | 7 | groupby, mode, idxmax/min |
| `shopping/v2_validated` | 16 | groupby, mean, describe |
| `loan/v1_mined` | 40 | groupby, aggregation |
| `loan/v1_validated` | 22 | groupby, mean, value_counts |
| `road/v1_validated` | 10 | groupby, corr, idxmax |

### Hard Split (L3-L5)
| Split | Questions | Level Distribution |
|-------|-----------|-------------------|
| `bank/v2_hard` | 49 | L5: 41%, L4: 35%, L3: 24% |

**Total:** 172 questions

## Difficulty Levels

| Level | Complexity | Example |
|-------|------------|---------|
| L3 | Comparisons & rankings | "Which job categories have subscription rates above 15%?" |
| L4 | Binned analysis (pd.qcut) | "What is the subscription percentage in Q2 balance quartile?" |
| L5 | Multi-hop chains | "For the month with lowest subscription, what's the avg age?" |

## Running Evaluations

### Local Backend (OpenAI/Anthropic)
```bash
python scripts/eval_runner.py \
  --split bank/v2_hard \
  --backend local \
  --model gpt-4o-mini
```

### Prime Backend
```bash
python scripts/eval_runner.py \
  --split bank/v2_hard \
  --backend prime \
  --model deepseek/deepseek-chat
```

## Analysis Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `analyze_failures.py` | Analyze failed questions | `python scripts/analyze_failures.py results/<file>.json` |
| `analyze_by_level.py` | Breakdown by difficulty | `python scripts/analyze_by_level.py results/<file>.json` |
| `summarize_results.py` | Generate summary | `python scripts/summarize_results.py` |
| `run_benchmark_suite.py` | Run multiple models | `python scripts/run_benchmark_suite.py` |
| `calculate_tokens.py` | Token usage stats | `python scripts/calculate_tokens.py` |
| `rerun_failed.py` | Retry failed questions | `python scripts/rerun_failed.py results/<file>.json` |

## Comparison: codeblue-analytics vs codeblue-env

| Feature | codeblue-analytics | codeblue-env |
|---------|-------------------|--------------|
| **Mode** | One-shot | Multi-turn |
| **Tools** | None | run_python, save_note, clarify, submit_answer |
| **Datasets** | 5 (bank, shopping, heart, loan, road) | 1 (bank) |
| **Framework** | Custom eval_runner | verifiers (vf-eval) |
| **Use Case** | Code generation benchmark | Agent/RL training |
| **Questions** | 172 | 50+ (L4-L6) |

## Key Files

| File | Purpose |
|------|---------|
| `scripts/eval_runner.py` | Main evaluation runner |
| `scripts/analyze_*.py` | Analysis scripts |
| `scripts/run_benchmark_suite.py` | Batch benchmarking |
| `environments/codeblue-analytics/` | Environment module |
| `results/` | Evaluation results |
| `README.md` | Full documentation |

## Output Format

Results saved to `results/` as JSON:
```json
{
  "model": "gpt-4o-mini",
  "split": "bank/v2_hard",
  "total": 49,
  "correct": 35,
  "accuracy": 0.714,
  "results": [
    {
      "question_id": "bank_001",
      "correct": true,
      "expected": 14.23,
      "actual": 14.23
    }
  ]
}
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `API key not found` | Missing key | Add to `~/.pandas-rlvr.env` |
| `Split not found` | Invalid split name | Run `--list-splits` |
| Code execution error | Sandbox issue | Check Python version |
| Low accuracy | Model struggles | Try larger model |
