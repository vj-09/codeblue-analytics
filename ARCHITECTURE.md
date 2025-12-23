# codeblue-analytics Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CODEBLUE-ANALYTICS ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PURPOSE: One-shot code generation benchmark (NO tools, NO multi-turn)      │
│                                                                             │
│  INPUT                                                                       │
│  ═════                                                                       │
│                                                                             │
│  ┌────────────────┐     ┌────────────────┐                                  │
│  │  Question      │     │  Dataset CSV   │                                  │
│  │  Registry      │     │                │                                  │
│  │                │     │ • bank.csv     │                                  │
│  │ • 172 Qs       │     │ • shopping.csv │                                  │
│  │ • 5 datasets   │     │ • heart.csv    │                                  │
│  │ • 10+ splits   │     │ • loan.csv     │                                  │
│  │                │     │ • road.csv     │                                  │
│  └───────┬────────┘     └───────┬────────┘                                  │
│          │                      │                                            │
│          └──────────┬───────────┘                                            │
│                     │                                                        │
│                     ▼                                                        │
│  EVAL RUNNER (scripts/eval_runner.py)                                        │
│  ════════════════════════════════════                                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │  │
│  │  │ Load Split  │───>│ For Each Q  │───>│ Call LLM    │              │  │
│  │  │ (questions) │    │             │    │ (one-shot)  │              │  │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘              │  │
│  │                                               │                       │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐              │  │
│  │  │ Compare     │<───│ Execute     │<───│ Extract     │              │  │
│  │  │ Outputs     │    │ Code        │    │ Code        │              │  │
│  │  └──────┬──────┘    └─────────────┘    └─────────────┘              │  │
│  │         │                                                             │  │
│  │         ▼                                                             │  │
│  │  ┌─────────────┐                                                      │  │
│  │  │ Save Result │                                                      │  │
│  │  │ (JSON)      │                                                      │  │
│  │  └─────────────┘                                                      │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  BACKENDS                                                                    │
│  ════════                                                                    │
│                                                                             │
│  ┌────────────────────┐    ┌────────────────────┐                          │
│  │  LOCAL              │    │  PRIME             │                          │
│  │                     │    │                    │                          │
│  │  • OpenAI API       │    │  • Prime Intellect │                          │
│  │  • Anthropic API    │    │  • Cloud inference │                          │
│  │  • Google API       │    │  • Many models     │                          │
│  │                     │    │                    │                          │
│  │  Requires API keys  │    │  Requires prime    │                          │
│  │  in ~/.pandas-      │    │  CLI login         │                          │
│  │  rlvr.env           │    │                    │                          │
│  └────────────────────┘    └────────────────────┘                          │
│                                                                             │
│  OUTPUT                                                                      │
│  ══════                                                                      │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  results/<model>_<split>_<timestamp>.json                           │    │
│  │  {                                                                  │    │
│  │    "model": "gpt-4o-mini",                                         │    │
│  │    "split": "bank/v2_hard",                                        │    │
│  │    "total": 49,                                                    │    │
│  │    "correct": 35,                                                  │    │
│  │    "accuracy": 0.714,                                              │    │
│  │    "results": [...]                                                │    │
│  │  }                                                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow (One-Shot Evaluation)

```
Question + Dataset
        │
        ▼
┌───────────────────┐
│  Build Prompt     │     System: "You are a data analyst..."
│                   │     User: "Question: <question>"
│                   │           "df.head():\n<preview>"
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Call LLM         │     Single API call
│  (ONE response)   │     No tools, no iteration
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Extract Code     │     Parse ```python ... ```
│                   │     or raw code block
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Execute Code     │     exec() in sandbox
│                   │     with df, pd, np
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Compare Output   │     result == expected?
│                   │     (with tolerance)
└─────────┬─────────┘
          │
          ▼
     correct / incorrect
```

## Key Files

| File | Purpose | Location |
|------|---------|----------|
| `scripts/eval_runner.py` | Main evaluation runner | scripts/ |
| `scripts/analyze_failures.py` | Analyze failed questions | scripts/ |
| `scripts/analyze_by_level.py` | Breakdown by difficulty | scripts/ |
| `scripts/run_benchmark_suite.py` | Run multiple models | scripts/ |
| `scripts/rerun_failed.py` | Retry failed questions | scripts/ |
| `scripts/summarize_results.py` | Generate summary | scripts/ |
| `environments/codeblue-analytics/` | Environment module | environments/ |
| `results/` | Evaluation outputs | results/ |

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `eval_runner.py` | Run evaluation | `--split bank/v2_hard --backend local --model gpt-4o-mini` |
| `analyze_failures.py` | Debug failures | `results/<file>.json` |
| `analyze_by_level.py` | Level breakdown | `results/<file>.json` |
| `run_benchmark_suite.py` | Batch run | (runs multiple models) |
| `summarize_results.py` | Summary stats | (aggregates results/) |
| `rerun_failed.py` | Retry failures | `results/<file>.json` |
| `calculate_tokens.py` | Token usage | `results/<file>.json` |

## Question Splits

| Split | Questions | Difficulty | Dataset |
|-------|-----------|------------|---------|
| `bank/v2_hard` | 49 | L3-L5 | bank |
| `bank/v1_validated` | 18 | Mixed | bank |
| `shopping/v1_basic` | 6 | Easy | shopping |
| `shopping/v2_validated` | 16 | Mixed | shopping |
| `heart/v1_basic` | 4 | Easy | heart |
| `loan/v1_mined` | 40 | Mixed | loan |
| `loan/v1_validated` | 22 | Mixed | loan |
| `road/v1_validated` | 10 | Mixed | road |

---

## Comparison: codeblue-analytics vs codeblue-env

| Feature | codeblue-analytics | codeblue-env |
|---------|-------------------|--------------|
| **Mode** | One-shot | Multi-turn |
| **Tools** | None | run_python, save_note, etc. |
| **Datasets** | 5 (bank, shopping, heart, loan, road) | 1 (bank) |
| **Questions** | 172 | 50+ (L4-L6) |
| **Framework** | Custom eval_runner | verifiers (vf-eval) |
| **Use Case** | Code generation benchmark | Agent/RL training |
| **Scoring** | Binary correct/incorrect | 4 metrics (max 4.0) |

---

## Update Guidelines (Don't Break Cross-Functional Flows)

### CRITICAL: Do NOT Change (Shared with qa-gen)

| Component | Why | Location |
|-----------|-----|----------|
| Question format in registry | qa-gen outputs this | environments/codeblue-analytics/ |
| Expected output format | Validation depends on it | compare_outputs() |

### Safe to Change

| Component | Notes |
|-----------|-------|
| Backend configurations | Just API endpoints |
| Model list | Add/remove supported models |
| Analysis scripts | Post-processing only |
| Result JSON format | Only affects analysis |
| Prompt template | Internal to eval_runner |

### This Repo is STANDALONE

Unlike codeblue-env and codeblue-ensemble (which are tightly coupled), codeblue-analytics is mostly independent:
- Does NOT interact with ensemble
- Does NOT use verifiers framework
- Questions come from qa-gen but can be manually created

---

## Testing After Changes

```bash
# 1. List available splits
python scripts/eval_runner.py --list-splits

# 2. Run single split
python scripts/eval_runner.py \
  --split bank/v2_hard \
  --backend local \
  --model gpt-4o-mini

# 3. Analyze results
python scripts/analyze_failures.py results/<file>.json

# 4. Run full benchmark suite
python scripts/run_benchmark_suite.py
```

---

## Maintaining This Document

**UPDATE THIS DOC AT END OF EACH WORKING SESSION**

Before ending a session where you modified codeblue-analytics:

1. **Check for new splits**: Did you add new question splits?
   - If yes, add to "Question Splits" table

2. **Check for new models**: Did you add new model configurations?
   - If yes, verify in eval_runner.py MODELS dict

3. **Check for new scripts**: Did you add new analysis scripts?
   - If yes, add to "Scripts Reference" table

4. **Test evaluation**: Run a quick eval to verify changes work

### Changelog

| Date | Change | Impact |
|------|--------|--------|
| 2024-12-23 | Initial architecture doc | - |

---

## Cross-Repo Dependencies

```
┌──────────────────┐
│     qa-gen       │
│                  │
│  Generates       │
│  questions       │
└────────┬─────────┘
         │
         │ questions.json
         ▼
┌──────────────────┐
│ codeblue-        │
│ analytics        │
│                  │
│ (Standalone)     │
│ No runtime deps  │
└──────────────────┘

Note: codeblue-analytics does NOT interact with:
- codeblue-env (different evaluation mode)
- codeblue-ensemble (no multi-turn, no voting)
```
