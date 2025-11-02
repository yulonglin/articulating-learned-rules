# List of Rules

This directory contains rule definitions used in the articulation experiments.

## Files

### `curated_rules_generated.jsonl`
- **Count:** 38 rules
- **Description:** All curated rules from brainstorming and curation phase (Step 1a-1b)
- **Source:** Generated via `src/brainstorm_rules.py` + `src/curate_rules.py`
- **Includes:** Both learnable and non-learnable rules

### `learnable_rules.jsonl`
- **Count:** 31 rules
- **Description:** Filtered subset of rules that achieved ≥90% accuracy in learnability tests
- **Source:** Filtered from `curated_rules_generated.jsonl` based on `experiments/learnability/summary.yaml`
- **Generated:** 2025-10-31 via `tmp/filter_learnable_rules.py`
- **Purpose:** Use this file for articulation testing (Step 2) to focus on rules that models can learn

## Rule Structure

Each JSONL entry contains:
```json
{
    "rule_id": "unique_identifier",
    "rule_name": "human_readable_name",
    "articulation": "Natural language description of the rule",
    "category": "syntactic|semantic|statistical",
    "examples": [
        {"input": "example text", "label": true},
        ...
    ],
    "expected_difficulty": "easy|medium|hard",
    "source_model": "model_that_generated_this_rule",
    "timestamp": "ISO_8601_timestamp",
    "prompt_strategy": "strategy_used",
    "implementability": "programmatic|llm_needed|complex",
    "similarity_cluster": "cluster_id",
    "selection_reason": "why_this_rule_was_selected",
    "quality_score": 0.0-1.0,
    "semantic_validation": null
}
```

## Usage

### For articulation testing (Step 2):
```bash
# Use learnable_rules.jsonl to test only rules that models can learn
python -m src.test_articulation_mc \
  --rules-file data/processed/list-of-rules/learnable_rules.jsonl \
  ...

python -m src.test_articulation_freeform \
  --rules-file data/processed/list-of-rules/learnable_rules.jsonl \
  ...
```

### For full rule exploration:
```bash
# Use curated_rules_generated.jsonl to test all rules
python -m src.test_learnability \
  --rules-file data/processed/list-of-rules/curated_rules_generated.jsonl \
  ...
```

## Learnability Tiers

The 31 learnable rules are organized into three tiers:

### Tier 1: Highly Learnable (7 rules)
Perfect (100%) accuracy achieved across most configurations

### Tier 2: Strongly Learnable (17 rules)
90%+ accuracy with moderate shots (10-50 examples)

### Tier 3: Marginally Learnable (7 rules)
90%+ accuracy only at high shots (50-100 examples)

See `experiments/learnability/learnable_rules.txt` for the complete list organized by tier.
