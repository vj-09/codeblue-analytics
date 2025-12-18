# CodeBlue Analytics

A [Prime Intellect](https://www.primeintellect.ai/) verifiers environment for evaluating language models on data analytics tasks - generating Pandas code from natural language questions.

## Overview

CodeBlue Analytics benchmarks LLM capabilities in translating natural language queries into executable Pandas code across diverse real-world datasets. Models are evaluated on their ability to correctly analyze tabular data and return accurate results.

## Features

- **Multi-domain datasets** - Banking, loans, shopping, road safety, healthcare
- **Difficulty levels** - Easy, medium, hard question splits
- **Automatic verification** - Code execution and result validation
- **Extensible** - Easy to add new datasets and questions

## Installation

```bash
# Using Prime CLI (recommended)
prime env install codeblue/codeblue-analytics

# Or clone directly
git clone https://github.com/vj-09/codeblue-analytics
cd codeblue-analytics
uv sync
```

## Quick Start

```bash
# Run evaluation on a split
uv run vf-eval codeblue-analytics --split bank_v2_hard

# Run with specific model
uv run vf-eval codeblue-analytics --split shopping_eval --model gpt-4o

# Run benchmark suite
python scripts/run_benchmark_suite.py
```

## Datasets

| Dataset | Rows | Domain | Description |
|---------|------|--------|-------------|
| bank | 45K | Finance | Bank marketing campaign data |
| loan | 100K | Finance | Loan application records |
| shopping | 3.9K | Retail | Customer shopping behavior |
| road | 307K | Safety | UK road accident data |
| heart | 918 | Health | Heart disease indicators |

## Question Splits

| Split | Questions | Difficulty |
|-------|-----------|------------|
| bank_v1_validated | 20 | Mixed |
| bank_v2_hard | 15 | Hard |
| loan_v1_validated | 20 | Mixed |
| shopping_v1_validated | 20 | Mixed |
| shopping_v2_validated | 15 | Hard |
| road_v1_validated | 20 | Mixed |

## Benchmark Results (bank_v2_hard)

| Model | Accuracy |
|-------|----------|
| Claude Opus 4.5 | - |
| GPT-5.1 | - |
| Gemini 2.5 Flash | - |
| Gemini 3 Pro | - |
| Grok 4 Fast | - |
| DeepSeek Chat | - |
| Qwen3 Max | - |
| Qwen 2.5 72B | - |
| Llama 3.1 70B | - |
| Intellect 3 | - |

## Project Structure

```
codeblue-analytics/
├── environments/
│   └── codeblue-analytics/
│       ├── codeblue_analytics.py    # Main environment loader
│       ├── data/
│       │   ├── datasets/            # CSV data files
│       │   └── questions/           # JSONL question sets
│       └── configs/
│           └── splits/              # Evaluation split configs
├── scripts/
│   ├── eval_runner.py               # Single model evaluation
│   ├── run_benchmark_suite.py       # Multi-model benchmarking
│   ├── analyze_failures.py          # Error analysis
│   └── summarize_results.py         # Results aggregation
├── results/                         # Benchmark outputs
└── README.md
```

## Adding New Questions

1. Create questions in JSONL format:
```json
{"question": "What is the average age?", "answer": 42.5, "level": "easy"}
```

2. Add to `data/questions/{dataset}/`

3. Register in `data/questions/registry.yaml`

## Publishing

```bash
cd environments/codeblue-analytics
prime env push
```

## Scripts

| Script | Description |
|--------|-------------|
| `eval_runner.py` | Run evaluation for a single model |
| `run_benchmark_suite.py` | Run full benchmark across models |
| `analyze_failures.py` | Analyze model failure patterns |
| `summarize_results.py` | Generate results summary |
| `rank_questions.py` | Rank questions by difficulty |

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

---

Built with [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/prime)
