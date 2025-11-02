# Learnability Analysis Summary

## Overall Statistics

- **5-shot accuracy:** 82.8%
- **100-shot accuracy:** 92.7%
- **Improvement:** +9.9%

## Model Comparison (100-shot)

- **Claude Haiku 4.5:** 97.7%
- **GPT-4.1-nano:** 87.7%

## Category Breakdown (100-shot)

- **Pattern-based:** 92.7% average | 17/17 rules ≥90%
- **Semantic:** 91.4% average | 8/8 rules ≥90%
- **Statistical:** 94.7% average | 5/5 rules ≥90%

## Failed Rules (<90% for both models at all shot counts)

- None! All rules achieved ≥90% for at least one model/shot combination

## Model Agreement on Task Difficulty (100-shot)

- **Pearson correlation:** r = 0.612 (p = 1.0e-03)
- **Spearman correlation:** ρ = 0.659 (p = 1.0e-03)
- **Interpretation:** Moderate agreement - some differences in which rules are challenging

## Key Insights

1. **Monotonic improvement:** Accuracy increases consistently with more examples
2. **Claude advantage:** Outperforms GPT across nearly all categories
3. **Category difficulty:** Pattern-based rules generally easier, semantic/statistical more challenging
4. **Optimal shot count:** 50-100 examples recommended for reliable performance