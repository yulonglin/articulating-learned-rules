Look in specs/

# Pipeline Overview

This project tests the hypothesis: **Can LLMs achieve high accuracy on classification tasks but fail to articulate the rule they're using?**

## Three-Step Research Pipeline

### Step 1: Rule Generation & Curation

#### 1a. Brainstorm Rules (`brainstorm_rules.py`)
- Generates diverse classification rules using LLMs (GPT-4.1-nano, Claude Haiku 4.5)
- Uses multiple prompt strategies: syntactic_simple, pattern_complex, semantic, statistical
- Each rule includes: name, natural language articulation, category, examples, expected difficulty
- **Output:** JSONL file with ~100 candidate rules

#### 1b. Curate Rules (`curate_rules.py`)
- Deduplicates rules (exact and similarity-based clustering)
- Assesses implementability: programmatic vs LLM-needed vs complex
- Scores quality based on articulation clarity and example consistency
- Selects diverse, high-quality subset balanced across categories/difficulties
- **Output:** Curated JSONL with metadata (implementability, similarity_cluster, quality_score)

#### 1c. Generate Datasets (`generate_datasets.py`)
- Creates balanced labeled datasets for each curated rule
- Supports both programmatic generators (e.g., syntactic rules) and LLM-based generation
- Ensures balanced positive/negative cases with edge case coverage
- **Output:** Per-rule datasets with consistent structure

### Step 2: Test Learnability (Research Step 1)

#### Test (`test_learnability.py`)
- Tests if LLMs can learn rules from few-shot examples (5, 10, 20 shots)
- **Direct classification WITHOUT chain-of-thought reasoning**
- Goal: >90% accuracy on held-out examples
- **Output:** Per-rule JSONL + summary YAML with accuracy metrics

#### Analyze (`analyze_learnability.py`)
- Filters for rules with >90% accuracy
- Identifies which rules are learnable vs difficult
- **Output:** `summary_complete.yaml` with per-rule, per-model, per-few-shot metrics

#### Enrich (`enrich_rules_with_learnability.py`)
- Adds model-specific `min_few_shot_required` metadata to rules
- Filters to only include learnable rules (≥90% accuracy for at least one model)
- **Output:** `curated_rules_learnable.jsonl` with learnability metadata per model

### Step 3: Test Articulation (Research Step 2)

#### Multiple Choice (`test_articulation_mc.py`)
- Tests if model can identify correct rule from multiple options
- **CoT reasoning allowed** (unlike learnability test)
- Automatically uses `min_few_shot_required` per model from enriched rules

#### Free-form (`test_articulation_freeform.py`)
- Tests if model can articulate the rule in natural language
- Uses LLM judges and functional scoring to evaluate articulations
- Automatically uses `min_few_shot_required` per model from enriched rules

#### Analyze (`analyze_articulation_freeform.py`)
- Aggregates articulation results
- Identifies rules where model learns but cannot articulate

### Step 4: Test Faithfulness (Research Step 3)

#### Test (`test_faithfulness.py`)
- Generates counterfactual test cases based on articulated rules
- Checks if model's classifications match its articulated rule's predictions
- Tests whether articulations actually explain behavior

## Infrastructure

- **`runner.py`:** Reproducible experiment execution framework with CLI args, JSONL output, logging, checkpointing, git tracking
- **`api_caller.py`:** Async LLM API calls with caching (15min short cache, persistent cache)
- **`model_registry.py`:** Central model configuration
- **`utils.py`:** Shared utilities (JSONL I/O, random seeds, etc.)
- **`analyze.py`:** Quick diagnostics
- **`create_visualizations.py`:** Generates figures for results

## Key Design Principles

1. **Reproducibility:** CLI args, random seeds, git commits, timestamps tracked
2. **Caching:** API response caching to avoid redundant calls
3. **JSONL outputs:** Streamable, appendable format for results
4. **Async execution:** Concurrent API calls with rate limiting
5. **Critical constraint:** Learnability tests use NO CoT, articulation tests allow CoT

## Workflow Example

```bash
# 1. Generate and curate rules
python -m src.brainstorm_rules --output-path out/rules/raw_rules.jsonl
python -m src.curate_rules --input-path out/rules/raw_rules.jsonl --output-path out/rules/curated_rules.jsonl

# 2. Generate datasets
python -m src.generate_datasets --rules-file out/rules/curated_rules.jsonl --output-dir data/datasets

# 3. Test learnability
python -m src.test_learnability --rules-file out/rules/curated_rules.jsonl --datasets-dir data/datasets --output-dir out/learnability

# 3a. Analyze learnability and enrich rules
python -m src.analyze_learnability --results-dir out/learnability --output-summary out/learnability/summary_complete.yaml
python -m src.enrich_rules_with_learnability \
  --rules-file data/processed/list-of-rules/curated_rules_generated.jsonl \
  --learnability-summary out/learnability/summary_complete.yaml \
  --output-file out/rules/curated_rules_learnable.jsonl

# 4. Test articulation (uses enriched rules with min_few_shot_required per model)
python -m src.test_articulation_freeform --rules-file out/rules/curated_rules_learnable.jsonl --datasets-dir data/datasets --output-dir out/articulation

# 5. Test faithfulness
python -m src.test_faithfulness --rules-file out/rules/curated_rules.jsonl --articulations-file out/articulation/results.jsonl --output-dir out/faithfulness
```