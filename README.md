# CodeBlue: Dataanalytics

A Prime Intellect verifiers environment for training language models to generate Pandas code from natural language questions.

## Structure

```
codeblue-analytics/
├── environments/
│   └── codeblue-analytics/           # The verifiers environment
│       ├── codeblue_analytics.py     # Main load_environment()
│       ├── pyproject.toml            # Dependencies
│       ├── README.md                 # Documentation
│       ├── data/
│       │   ├── dataframes/           # CSV datasets
│       │   └── questions/            # JSONL eval sets
│       └── reports/                  # Evaluation results
├── pyproject.toml
└── README.md
```

## Quick Start

```bash
# Install
prime env install codeblue/codeblue-analytics

# Run evaluation
uv run vf-eval codeblue-analytics --split heart_eval

# Run specific split
uv run vf-eval codeblue-analytics --split shopping_eval --model gpt-4o
```

## Datasets

| Name | Rows | Description |
|------|------|-------------|
| heart | 918 | Heart disease records |
| shopping | 3,900 | Shopping behavior |

## Evaluation Splits

| Split | Questions |
|-------|-----------|
| heart_eval | 4 |
| shopping_eval | 5 |

## Benchmark Results

| Model | heart_eval | shopping_eval |
|-------|------------|---------------|
| Claude Haiku | 100% | 100% |
| GPT-4o | 100% | 100% |
| Claude Sonnet 4 | 100% | 100% |
| Claude Opus 4 | 0% | 100% |
| GPT-4 Turbo | 100% | 100% |

## Publishing

```bash
cd environments/codeblue-analytics
prime env push
```

## License

MIT
