# Can Language Models Learn Rules They Cannot Articulate?

This repository contains code and data for evaluating whether large language models can learn classification rules they cannot faithfully articulate. We test 31 learnable rules across pattern-based, semantic, and statistical categories with GPT-4.1-nano and Claude Haiku 4.5.

**Paper:** [Link to paper]
**Code & Data:** https://github.com/yulonglin/articulating-learned-rules

## Key Findings

1. **Dataset artifact overfitting undermines rule learning claims**: Models achieve perfect classification (100%) while learning completely wrong rules. For example, articulating "contains letter 's'" for a rule about consecutive repeated characters. 16 rule-model pairs show classification >90% but multiple-choice articulation <60%, with gaps reaching 62-71% that *increase* with more examples.

2. **High functional accuracy masks unfaithful explanations**: Models achieve 85-90% accuracy when using their own articulations to classify, yet these predict only 73% of counterfactual classifications (51% without few-shot context).

3. **Post-hoc rationalization is widespread**: Several rules demonstrate high articulation quality (>85%) but low faithfulness (~50%), indicating persuasive but unfaithful explanations.

4. **Statistical rules show notable faithfulness gaps**: Despite 89% functional accuracy on statistical rules, models show lower faithfulness—likely reflecting known difficulties with counting and numerical reasoning.

## Research Question

**Do models genuinely understand the rules they apply, or do they merely exploit statistical patterns without explicit knowledge?**

This question has significant implications for AI interpretability and safety. If models perform well while holding incorrect beliefs about their reasoning, their natural language explanations may be unreliable guides to actual behavior.

## Three-Step Evaluation Pipeline

### Step 1: Learnability Testing
- Test whether models can learn binary classification rules from few-shot examples (5, 10, 20, 50, 100 shots)
- **Critical constraint**: Direct classification WITHOUT chain-of-thought reasoning
- Rules achieving ≥90% accuracy are considered "learnable"

### Step 2: Articulation Testing
- Test whether models can explicitly state learned rules in natural language
- **Two evaluation modes**:
  - **Multiple-choice**: Identify correct rule from options (CoT reasoning allowed)
  - **Free-form**: Generate rule articulation, evaluated via:
    - LLM judge (semantic equivalence)
    - Cosine similarity
    - Functional accuracy (can the articulation classify new examples?)

### Step 3: Faithfulness Testing
- Assess whether articulated rules actually explain model behavior
- Generate ~20 counterfactual test cases per rule designed to discriminate the articulation
- Compare model predictions (with few-shot context) vs articulation predictions
- Faithfulness score = % of cases where predictions match

## Installation

```bash
# Clone the repository
git clone https://github.com/yulonglin/articulating-learned-rules.git
cd articulating-learned-rules

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt

# Set up API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## Quick Start

```bash
# 1. Generate and curate rules
python -m src.brainstorm_rules --output-path out/rules/raw_rules.jsonl
python -m src.curate_rules \
  --input-path out/rules/raw_rules.jsonl \
  --output-path out/rules/curated_rules.jsonl

# 2. Generate datasets for rules
python -m src.generate_datasets \
  --rules-file out/rules/curated_rules.jsonl \
  --output-dir data/datasets

# 3. Test learnability (Step 1)
python -m src.test_learnability \
  --rules-file out/rules/curated_rules.jsonl \
  --datasets-dir data/datasets \
  --output-dir out/learnability

# 3a. Analyze and enrich rules with learnability metadata
python -m src.analyze_learnability \
  --results-dir out/learnability \
  --output-summary out/learnability/summary_complete.yaml

python -m src.enrich_rules_with_learnability \
  --rules-file out/rules/curated_rules.jsonl \
  --learnability-summary out/learnability/summary_complete.yaml \
  --output-file out/rules/curated_rules_learnable.jsonl

# 4. Test articulation (Step 2)
# Multiple-choice
python -m src.test_articulation_mc \
  --rules-file out/rules/curated_rules_learnable.jsonl \
  --datasets-dir data/datasets \
  --output-dir out/articulation_mc

# Free-form
python -m src.test_articulation_freeform \
  --rules-file out/rules/curated_rules_learnable.jsonl \
  --datasets-dir data/datasets \
  --output-dir out/articulation_freeform

# 5. Test faithfulness (Step 3)
python -m src.test_faithfulness \
  --rules-file out/rules/curated_rules_learnable.jsonl \
  --articulations-file out/articulation_freeform/results.jsonl \
  --output-dir out/faithfulness

# 6. Generate visualizations
python -m src.create_visualizations \
  --learnability-dir out/learnability \
  --articulation-dir out/articulation_freeform \
  --faithfulness-dir out/faithfulness \
  --output-dir paper/figures
```

## Project Structure

```
articulating-learned-rules/
├── src/                          # Source code
│   ├── brainstorm_rules.py      # Generate candidate rules
│   ├── curate_rules.py          # Deduplicate and filter rules
│   ├── generate_datasets.py     # Create balanced datasets
│   ├── test_learnability.py     # Step 1: Test rule learning
│   ├── analyze_learnability.py  # Analyze learnability results
│   ├── enrich_rules_with_learnability.py  # Add learnability metadata
│   ├── test_articulation_mc.py  # Step 2: Multiple-choice articulation
│   ├── test_articulation_freeform.py  # Step 2: Free-form articulation
│   ├── test_faithfulness.py     # Step 3: Counterfactual testing
│   ├── create_visualizations.py # Generate paper figures
│   ├── api_caller.py            # Async LLM API calls with caching
│   ├── model_registry.py        # Model configurations
│   └── utils.py                 # Shared utilities
├── data/                        # Datasets
│   ├── datasets/                # Generated rule datasets
│   └── processed/               # Processed data
├── out/                         # Experimental outputs
│   ├── learnability/            # Step 1 results
│   ├── articulation_mc/         # Step 2 MC results
│   ├── articulation_freeform/   # Step 2 free-form results
│   └── faithfulness/            # Step 3 results
├── paper/                       # LaTeX paper and figures
│   ├── main.tex
│   └── figures/
├── specs/                       # Project specifications
└── README.md
```

## Rule Dataset

We evaluate 31 learnable rules across three categories:

**Pattern-based (n=17)**: Character/token patterns and structural rules
- Palindromes, digit patterns, alternating case, URLs, hyphenated words
- Repeated characters, quotation depth, Fibonacci word lengths

**Semantic (n=8)**: Meaning-based rules
- Complaints, urgency, financial topics, emotional expression
- First-person vs third-person perspective, negation presence

**Statistical (n=6)**: Numeric properties
- Word length variance, Shannon entropy, character ratios
- Punctuation density, unique character ratios

## Models Tested

- **GPT-4.1-nano** (gpt-4.1-nano-2025-04-14)
- **Claude Haiku 4.5** (claude-haiku-4-5-20251001)

All experiments use temperature=0.0 for deterministic outputs (except data generation).

## Key Results

### Learnability
- 31 of 50 candidate rules (71%) achieved ≥90% accuracy
- 94% agreement between models on which rules are learnable
- Pattern-based: 85% learnable, Semantic: 89% learnable, Statistical: 50% learnable

### Articulation
- Functional accuracy: 85-90% (models can use their own articulations)
- Semantic agreement with ground truth: 49-56% (less informative due to multiple valid articulations)
- Dataset artifact overfitting: 16 rule-model pairs show classification >90% but MC articulation <60%

### Faithfulness
- Overall faithfulness: 72.8% (averaged across 5/10/20-shot contexts)
- Zero-shot faithfulness: 51% (near-random, showing articulations require context)
- Several rules show high functional accuracy (>85%) but low faithfulness (~50%)

## Citation

```bibtex
@misc{lin2025articulating,
  title={Can Language Models Learn Rules They Cannot Articulate? Evaluating the Learnability-Articulation Gap in LLMs},
  author={Lin, Yulong},
  howpublished={\url{https://github.com/yulonglin/articulating-learned-rules}},
  year={2025}
}
```

## Implications

Our findings demonstrate that:

1. **High accuracy ≠ correct rule learning**: Models can achieve perfect classification while learning spurious correlations
2. **Functional accuracy ≠ faithful explanation**: Operational success doesn't guarantee that articulations accurately describe decision processes
3. **Rigorous validation is essential**: Counterfactual testing is necessary to assess explanation faithfulness
4. **Context matters**: Articulations require few-shot context to be operationalizable, and even then show significant faithfulness gaps

These results have important implications for AI interpretability and safety, suggesting that model-generated explanations require rigorous validation before being trusted as faithful accounts of reasoning.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work represents approximately 16 hours of focused research effort. The research infrastructure was designed following best practices for reproducibility and transparency in AI safety research.
