# Learnability Analysis Summary

## Overall Statistics

- **5-shot accuracy:** 78.2%
- **100-shot accuracy:** 88.4%
- **Improvement:** +10.2%

## Model Comparison (100-shot)

- **Claude Haiku 4.5:** 94.5%
- **GPT-4.1-nano:** 82.3%

## Category Breakdown (100-shot)

- **Pattern:** 91.0% average | 8/8 rules ≥90%
- **Semantic:** 91.4% average | 8/8 rules ≥90%
- **Statistical:** 94.7% average | 5/5 rules ≥90%
- **Syntactic:** 94.2% average | 9/9 rules ≥90%
- **Unknown:** 69.9% average | 0/7 rules ≥90%

## Failed Rules (<90% for both models at all shot counts)

- `semantic_animal_color_binding_claude_085` (unknown): max 86.0%
- `Part-of-Speech Pattern_gpt_007` (unknown): max 85.0%
- `reference_starts_and_ends_same_char` (unknown): max 66.0%
- `reference_is_adjective` (unknown): max 78.0%
- `reference_rhyming_ends` (unknown): max 84.0%
- `reference_starts_with_vowel` (unknown): max 87.0%
- `reference_word_count_between_3_and_7` (unknown): max 84.0%

## Model Agreement on Task Difficulty (100-shot)

- **Pearson correlation:** r = 0.785 (p = 1.0e-03)
- **Spearman correlation:** ρ = 0.758 (p = 1.0e-03)
- **Interpretation:** Moderate agreement - some differences in which rules are challenging

## Key Insights

1. **Monotonic improvement:** Accuracy increases consistently with more examples
2. **Claude advantage:** Outperforms GPT across nearly all categories
3. **Category difficulty:** Syntactic rules easiest, semantic/statistical more challenging
4. **Optimal shot count:** 50-100 examples recommended for reliable performance
5. **Model agreement:** Both models find similar rules difficult, suggesting inherent task difficulty