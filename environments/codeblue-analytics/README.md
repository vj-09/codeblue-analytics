# CodeBlue: Dataanalytics

**ID**: `codeblue-analytics`
**Description**: Single-turn pandas data analysis with code execution and exact-match verification
**Task Type**: SingleTurnEnv with code parser

## Overview

This environment tests LLM ability to write correct pandas code for data analysis tasks. Models receive a DataFrame schema and natural language question, then must produce executable Python code that yields the exact expected output.

## Datasets

| Name | Rows | Columns | Description |
|------|------|---------|-------------|
| `heart` | 918 | 12 | Heart disease clinical records |
| `shopping` | 3,900 | 18 | Customer shopping behavior |

## Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| `execution_reward` | 0.2 | Code executes without error |
| `correctness_reward` | 0.8 | Output matches expected result exactly |

## Splits

| Split | Questions | Description |
|-------|-----------|-------------|
| `heart_eval` | 4 | Heart disease specific |
| `shopping_eval` | 5 | Shopping behavior specific |

## Question Categories

- `groupby_sum` - Aggregation with sum
- `groupby_mean` - Aggregation with mean
- `filter_count` - Filtering and counting
- `value_counts` - Distribution analysis
- `pivot_table` - Pivot operations

## Output Types

- `scalar` - Single numeric/string value
- `series` - Pandas Series with index
- `dataframe` - Pandas DataFrame

## Quickstart

```bash
# Install
prime env install codeblue/codeblue-analytics

# Run evaluation
uv run vf-eval codeblue-analytics --split heart_eval

# Run specific split
uv run vf-eval codeblue-analytics --split shopping_eval
```

## Example

**Input:**
```
DataFrame: shopping (3900 rows x 18 columns)
Columns: Customer ID, Age, Gender, Category, Purchase Amount (USD), ...

Question: What is the average purchase amount for each season?
```

**Expected Code:**
```python
result = df.groupby('Season')['Purchase Amount (USD)'].mean()
```

**Expected Output:**
```
Season
Fall      61.556923
Spring    58.737738
Summer    58.405236
Winter    60.357364
Name: Purchase Amount (USD), dtype: float64
```

## Benchmark Results

| Model | heart_eval | shopping_eval |
|-------|------------|---------------|
| Claude Haiku | 100% | 100% |
| GPT-4o | 100% | 100% |
| Claude Sonnet 4 | 100% | 100% |
| Claude Opus 4 | 0% | 100% |
| GPT-4 Turbo | 100% | 100% |

*Note: Opus 4 fails heart_eval due to over-engineering (returns DataFrame instead of Series, multiplies rates by 100)*

## Code Format

Models should generate code wrapped in `<code></code>` tags:

```
<code>
result = df['total_amount'].sum()
</code>
```

**Requirements:**
- DataFrame is pre-loaded as `df`
- Store final answer in variable `result`
- Only pandas (pd) and numpy (np) are available
- No imports, prints, or file operations

## License

MIT
