# CodeBlue: DataAnalytics

![Version](https://img.shields.io/badge/version-0.1.8-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-blue)

**High-quality RLVR environment for training and evaluating LLMs on pandas data analysis tasks**

A comprehensive benchmark for evaluating language models on real-world data analysis tasks with exact-match verification. Features 172 curated questions across 5 datasets with difficulty levels from basic operations to complex multi-hop analytical queries.

## 🎯 Key Features

- **172 High-Quality Questions** across 9 question sets with validated gold code
- **5 Real-World Datasets** (banking, healthcare, e-commerce, loans, road safety)
- **Multi-Level Difficulty** (L3-L5): From simple aggregations to complex multi-hop chains
- **23 Question Templates** covering business analytics patterns
- **Exact Match Verification** with code execution sandbox
- **Comprehensive Evaluation** splits for training and testing

---

## 📊 Datasets

| Dataset | Rows | Columns | Domain | Description |
|---------|------|---------|--------|-------------|
| **bank** | 750,000 | 18 | Finance | Bank marketing campaign with customer demographics and subscription outcomes |
| **shopping** | 3,900 | 18 | E-commerce | Customer shopping behavior with purchase patterns and demographics |
| **heart** | 918 | 12 | Healthcare | Heart disease clinical records with patient health indicators |
| **loan** | 32,581 | 12 | Finance | Loan applications with approval status and financial metrics |
| **road** | 12,316 | 21 | Safety | Road accident data with severity indicators and conditions |

---

## 📚 Question Sets & Splits

### Easy Splits (Basic Operations)
| Split | Questions | Difficulty | Skills | Description |
|-------|-----------|------------|--------|-------------|
| `shopping/v1_basic` | 6 | Easy | duplicated, isnull, value_counts | Data quality and basic counts |
| `heart/v1_basic` | 4 | Easy | groupby, mean, value_counts | Simple aggregations |

### Mixed Difficulty Splits
| Split | Questions | Difficulty | Skills | Description |
|-------|-----------|------------|--------|-------------|
| `bank/v1_validated` | 18 | Mixed | groupby, filtering, percentage | Marketing analytics |
| `shopping/v1_mined` | 7 | Mixed | groupby, mode, idxmax/min | Shopping behavior patterns |
| `shopping/v2_validated` | 16 | Mixed | groupby, mean, describe | Enhanced shopping analysis |
| `loan/v1_mined` | 40 | Mixed | groupby, aggregation | Mined from notebooks |
| `loan/v1_validated` | 22 | Mixed | groupby, mean, value_counts | Validated loan analytics |
| `road/v1_validated` | 10 | Mixed | groupby, corr, idxmax | Road safety diagnostics |

### Hard Split (L3-L5 Multi-hop)
| Split | Questions | Difficulty | Level Distribution | Description |
|-------|-----------|------------|-------------------|-------------|
| `bank/v2_hard` | 49 | Hard | L5: 41%, L4: 35%, L3: 24% | Template-generated complex analytics |

**Total:** 172 questions across 9 splits

---

## 🎓 Difficulty Levels & Templates

### L3 Templates (24.5%) - Comparisons & Rankings
Simple 1-2 step operations focusing on comparisons and filtering:
- Conversion rate comparisons between segments
- Top/bottom performing segments
- Segments above/below thresholds
- Size-based rankings

**Example:** *"Which job categories have subscription rates above 15%?"*

### L4 Templates (34.7%) - Binned Analysis & Cross-Metrics
Quartile-based analysis and cross-metric operations:
- Quartile conversion analysis (pd.qcut)
- Segment breakdown tables
- Cross-segment performance metrics
- Metric-based breakdowns

**Example:** *"Divide customers into 4 balance quartiles. What is the subscription percentage in Q2?"*

### L5 Templates (40.8%) - Multi-hop Chains & Complex Analytics
Complex multi-step reasoning requiring 3+ operations:
- Chain conversion (find extrema → analyze subset)
- Opportunity segment identification
- Nested extrema analysis
- Anomaly detection (high metric + low conversion)
- Impact potential scoring

**Example:** *"Find the job with lowest subscription rate. Within that group, what is the highest average day?"*

**23 Unique Templates** generate questions programmatically with slot-filling from dataset schemas.

---

## 🏆 Benchmark Results (bank/v2_hard - 49 Hard Questions)

| Rank | Model | Execution Success | Correct Answers | Accuracy |
|------|-------|------------------|-----------------|----------|
| 🥇 | **Qwen3 Max** | 49/49 (100%) | 37/49 | **75.5%** |
| 🥈 | **Claude Opus 4.5** | 49/49 (100%) | 35/49 | **71.4%** |
| 🥉 | **GPT-5.1** | 49/49 (100%) | 32/49 | **65.3%** |
| 4 | **Grok 4 Fast** | 48/49 (98%) | 30/49 | **61.2%** |
| 5 | **Qwen 2.5 72B Instruct** | 47/49 (96%) | 28/49 | **57.1%** |
| 6 | **Gemini 2.5 Flash** | 48/49 (98%) | 25/49 | **51.0%** |
| 7 | **DeepSeek Chat** | 48/49 (98%) | 24/49 | **49.0%** |
| 8 | **Llama 3.1 70B Instruct** | 47/49 (96%) | 20/49 | **40.8%** |
| 9 | **Gemini 3 Pro Preview** | 11/49 (22%) | 9/49 | **18.4%** |
| 10 | **Prime Intellect Intellect-3** | 6/49 (12%) | 3/49 | **6.1%** |
| 11 | **Gemini 2.5 Pro** | 0/49 (0%) | 0/49 | **0%** |

**Key Findings:**
- ✅ **Top performers** (Qwen3 Max, Claude Opus 4.5) achieve 71-76% accuracy on complex multi-hop analytics
- ⚠️ **Mid-tier models** (GPT-5.1, Grok 4, Qwen 2.5) achieve 57-65% with near-perfect execution
- ❌ **Gemini models struggle** with code generation (18.4% down to 0% for 2.5 Pro)
- 📊 **Execution success** is high (95-100%) for most models - main challenge is correctness

*Full benchmark results and per-question analysis available in `results/bank_v2_hard/`*

---

## 🚀 Quickstart

### Installation
```bash
# Install environment from Prime Hub
prime env install codeblue/codeblue-analytics

# Or install locally
cd environments/codeblue-analytics
pip install -e .
```

### Running Evaluations
```bash
# Run easy evaluation (10 questions)
uv run vf-eval codeblue-analytics --split eval

# Run hard benchmark (49 questions)
uv run vf-eval codeblue-analytics --split bank/v2_hard

# Run specific dataset evaluations
uv run vf-eval codeblue-analytics --split heart/v1_basic
uv run vf-eval codeblue-analytics --split shopping/v2_validated
uv run vf-eval codeblue-analytics --split loan/v1_validated
```

### Using Custom Models
```bash
# Using eval_runner.py with Prime backend
python scripts/eval_runner.py \
  --split bank/v2_hard \
  --backend prime \
  --model google/gemini-2.5-flash \
  --output results/my_eval.json

# Using OpenAI models (local backend)
python scripts/eval_runner.py \
  --split bank/v2_hard \
  --backend local \
  --model gpt-4o \
  --output results/gpt4o_eval.json
```

---

## 📖 Task Format

### Input Structure
Models receive:
1. **System Prompt** - Instructions for code generation
2. **DataFrame Schema** - Column names, types, and sample rows
3. **Natural Language Question** - Business analytics query

### Expected Output
Models must generate Python code wrapped in `<code></code>` tags:

```python
<code>
result = df.groupby('job')['y'].apply(lambda x: (x == 1).mean() * 100).round(2)
</code>
```

### Requirements
- ✅ DataFrame pre-loaded as `df`
- ✅ Store final answer in `result` variable
- ✅ Only pandas (pd) and numpy (np) available
- ❌ No imports, prints, or file I/O
- ❌ No external libraries

---

## 💡 Example Task

**Question:** *"Show the subscription percentage breakdown by job. Return a DataFrame with columns [job, count, rate] where rate is percentage (0-100), sorted by rate descending."*

**DataFrame Context:**
```
Dataset: bank (750,000 rows × 18 columns)
Columns: id, age, job, marital, education, balance, y, ...
Sample:
  job         | y
  ------------|---
  technician  | 0
  blue-collar | 0
  management  | 1
```

**Expected Gold Code:**
```python
breakdown = df.groupby('job').agg(
    count=('y', 'size'),
    rate=('y', lambda x: round((x == 1).mean() * 100, 2))
).sort_values('rate', ascending=False)
result = breakdown.reset_index()
```

**Expected Output:**
```python
[
  {'job': 'student', 'count': 11767, 'rate': 34.08},
  {'job': 'retired', 'count': 35185, 'rate': 24.62},
  {'job': 'unemployed', 'count': 17634, 'rate': 17.98},
  ...
]
```

---

## 🎯 Evaluation Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| **Execution Success** | 0.3 | Code executes without errors in sandbox |
| **Correctness** | 0.7 | Output matches expected result exactly |

**Scoring:**
- ✅ **Perfect (1.0)**: Code executes AND output matches exactly
- ⚠️ **Partial (0.3)**: Code executes BUT wrong output
- ❌ **Failed (0.0)**: Code fails to execute OR no code extracted

---

## 📁 Repository Structure

```
environments/codeblue-analytics/
├── codeblue_analytics.py          # Main environment implementation
├── pyproject.toml                  # Package configuration (v0.1.7)
├── data/
│   ├── datasets/                   # CSV files for all datasets
│   │   ├── bank.csv               # 750k banking records
│   │   ├── shopping.csv           # 3.9k shopping records
│   │   ├── heart.csv              # 918 health records
│   │   ├── loan.csv               # 32k loan records
│   │   └── road.csv               # 12k accident records
│   └── questions/                  # Question sets (JSONL)
│       ├── registry.yaml          # Question set metadata
│       ├── bank/v2_hard.jsonl     # 49 hard questions
│       ├── shopping/              # Shopping questions
│       ├── heart/                 # Heart questions
│       ├── loan/                  # Loan questions
│       └── road/                  # Road questions
├── configs/splits/                 # Evaluation split configs
└── .prime/                         # Prime environment metadata

scripts/
├── eval_runner.py                  # Unified eval runner (Prime + local)
├── analyze_failures.py             # Failure analysis by template/level
├── analyze_by_level.py             # Performance breakdown by L3/L4/L5
├── run_benchmark_suite.py          # Multi-model benchmark runner
└── summarize_results.py            # Results aggregation

qa-gen/                             # Question generation pipeline
└── templates/__init__.py           # 23 question templates (L3-L5)
```

---

## 🔧 Advanced Usage

### Creating Custom Splits
Create a YAML file in `configs/splits/`:
```yaml
# configs/splits/my_custom_split.yaml
include:
  - question_set: bank/v2_hard
    filters:
      level: ["L5"]  # Only L5 questions
  - question_set: shopping/v2_validated
```

Run with:
```bash
uv run vf-eval codeblue-analytics --split my_custom_split
```

### Analyzing Results
```bash
# Analyze failures by template and level
python scripts/analyze_failures.py results/my_eval.json

# Compare performance across difficulty levels
python scripts/analyze_by_level.py results/my_eval.json

# Aggregate results from multiple runs
python scripts/summarize_results.py results/bank_v2_hard/*.json
```

### Question Generation
The environment includes a template-based QA generation pipeline:
```bash
cd qa-gen
python run.py --dataset bank --output-dir output/
```

See `qa-gen/templates/__init__.py` for 23 template definitions.

---

## 📊 Dataset Statistics

| Metric | bank | shopping | heart | loan | road |
|--------|------|----------|-------|------|------|
| **Rows** | 750,000 | 3,900 | 918 | 32,581 | 12,316 |
| **Columns** | 18 | 18 | 12 | 12 | 21 |
| **Target Column** | y (binary) | - | DEATH_EVENT | Status | Accident_Severity |
| **Groupby Cols** | job, marital, month | Category, Season | Sex, Smoking | Purpose, Gender | Weather, Road_Type |
| **Numeric Cols** | age, balance, day | Age, Purchase Amount | Age, Ejection_Fraction | Amount, Income | Speed_limit |
| **Question Sets** | 2 (67 total) | 3 (29 total) | 1 (4 total) | 2 (62 total) | 1 (10 total) |

---

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Adding new datasets with domain-specific analytics
- Creating new question templates for different analytical patterns
- Improving evaluation metrics and verification logic
- Extending to multi-turn analytical conversations

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Links

- **Prime Environment Hub**: [codeblue/codeblue-analytics](https://app.primeintellect.ai/dashboard/environments/codeblue/codeblue-analytics)
- **Install**: `prime env install codeblue/codeblue-analytics`
- **Version**: 0.1.8
- **Last Updated**: 2025-12-14

---

## 📧 Citation

If you use this environment in your research, please cite:

```bibtex
@software{codeblue_analytics_2025,
  title = {CodeBlue DataAnalytics: A Comprehensive RLVR Environment for Pandas Code Generation},
  author = {CodeBlue Team},
  year = {2025},
  version = {0.1.8},
  url = {https://app.primeintellect.ai/dashboard/environments/codeblue/codeblue-analytics}
}
```

---

**Built with ❤️ for the RLVR community**
