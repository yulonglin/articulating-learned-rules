# Learnability Analysis Summary

**Date:** 2025-10-31
**Analysis of:** experiments/learnability/summary.yaml
**Total Rules Tested:** 38
**Models:** gpt-4.1-nano-2025-04-14, claude-haiku-4-5-20251001
**Few-shot Configurations:** 5, 10, 20, 50, 100 examples

## Overview

- **Learnable Rules (≥90%):** 31 (81.6%)
- **Non-Learnable Rules (<90%):** 7 (18.4%)

## Learnable Rules by Tier

### Tier 1: Highly Learnable (100% accuracy) - 7 rules

| Rule ID | Category | Best Accuracy | Notes |
|---------|----------|---------------|-------|
| digit_surrounded_by_letters_claude_003 | Syntactic | 100% | Perfect across all shots/models |
| PalindromeCheck_gpt_007 | Syntactic | 100% | Perfect from 10-shot onwards |
| nested_quotation_depth_claude_078 | Syntactic | 100% | Perfect across all shots/models |
| Numeric Pattern_gpt_004 | Syntactic | 100% | Perfect across most configurations |
| reference_is_anagram_of_list | Syntactic | 100% | Perfect across all shots/models |
| word_length_variance_low_claude_002 | Statistical | 100% | Claude perfect from 20-shot, GPT 98%+ |
| word_length_variance_high_claude_002 | Statistical | 100% | Near-perfect across all shots/models |

### Tier 2: Strongly Learnable (90%+ with 10-50 shots) - 17 rules

| Rule ID | Category | Best Accuracy | Shots Needed |
|---------|----------|---------------|--------------|
| contains_multiple_exclamation_marks_claude_003 | Syntactic | 100% | 10-shot |
| contains_consecutive_repeated_characters_claude_009 | Syntactic | 100% | 50-shot |
| contains_digit_pattern_gpt_005 | Syntactic | 100% | 20-shot (Claude) |
| contains_multiple_punctuation_marks_claude_004 | Syntactic | 100% | 10-shot (GPT) |
| contains_hyphenated_word_claude_009 | Syntactic | 100% | 20-shot (Claude) |
| alternating_case_words_claude_000 | Syntactic | 100% | 20-shot (Claude) |
| positive_product_review_gpt_000 | Semantic | 97% | 10-shot (Claude) |
| urgent_intent_gpt_001 | Semantic | 100% | 5-shot |
| complaint_statement_gpt_003 | Semantic | 99% | 5-shot |
| financial_or_money_related_gpt_009 | Semantic | 100% | 10-shot |
| emotional_expression_gpt_005 | Semantic | 100% | 10-shot (Claude) |
| entropy_threshold_low_claude_001 | Statistical | 100% | 50-shot (Claude) |
| PresenceOfURL_gpt_006 | Syntactic | 100% | 5-shot (Claude) |
| unique_character_ratio_claude_009 | Statistical | 100% | 10-shot (Claude) |
| all_caps_gpt_000 | Syntactic | 96% | 10-shot (Claude) |
| Repeated Punctuation_gpt_003 | Syntactic | 98% | 50-shot (Claude) |
| word_count_less_than_5_gpt_004 | Statistical | 94% | 10-shot (Claude) |

### Tier 3: Marginally Learnable (90%+ only at 50-100 shots) - 7 rules

| Rule ID | Category | Best Accuracy | Shots Needed |
|---------|----------|---------------|--------------|
| symmetric_word_pattern_claude_002 | Syntactic | 93% | 100-shot (Claude) |
| digit_to_letter_ratio_claude_004 | Statistical | 91% | 100-shot (Claude) |
| punctuation_density_high_claude_004 | Statistical | 97% | 100-shot (Claude) |
| word_length_fibonacci_claude_084 | Statistical | 99% | 50-shot (Claude) |
| reference_third_person_perspective | Semantic | 95% | 10-shot (Claude) |
| reference_negation_presence | Semantic | 90% | 100-shot (Claude) |
| reference_first_person_perspective | Semantic | 97% | 100-shot (Claude) |

## Non-Learnable Rules (<90% across all configurations) - 7 rules

| Rule ID | Category | Best Accuracy | Notes |
|---------|----------|---------------|-------|
| semantic_animal_color_binding_claude_085 | Semantic | 86% | Semantic binding task |
| Part-of-Speech Pattern_gpt_007 | Syntactic | 85% | Complex POS pattern |
| reference_starts_and_ends_same_char | Syntactic | 66% | Structural pattern |
| reference_is_adjective | Syntactic | 78% | POS identification |
| reference_rhyming_ends | Phonetic | 84% | Rhyme detection |
| reference_starts_with_vowel | Syntactic | 87% | First character pattern |
| reference_word_count_between_3_and_7 | Statistical | 84% | Counting task |

## Key Insights

1. **Syntactic rules are highly learnable:** Most syntactic pattern rules achieve 90%+ accuracy
2. **Semantic rules vary:** Some semantic rules (sentiment, urgency) are learnable, others (POS patterns, semantic bindings) are not
3. **Statistical rules mixed:** Simple statistical rules work well, complex ones (entropy, ratios) need more examples
4. **Model differences:** Claude Haiku generally reaches 90%+ with fewer shots than GPT-4.1-nano
5. **Reference-based rules struggle:** Rules requiring structural analysis of word properties (rhyming, POS, vowels) consistently underperform

## Next Steps

Use `data/processed/list-of-rules/learnable_rules.jsonl` (31 rules) for articulation testing:

```bash
# Multiple-choice articulation
python -m src.test_articulation_mc \
  --rules-file data/processed/list-of-rules/learnable_rules.jsonl \
  --datasets-dir experiments/datasets \
  --output-dir experiments/articulation_mc

# Free-form articulation
python -m src.test_articulation_freeform \
  --rules-file data/processed/list-of-rules/learnable_rules.jsonl \
  --datasets-dir experiments/datasets \
  --output-dir experiments/articulation_freeform
```

## Files Generated

- `experiments/learnability/learnable_rules.txt` - Plain text list of 31 learnable rule IDs
- `data/processed/list-of-rules/learnable_rules.jsonl` - Filtered rules file for articulation testing
- `data/processed/list-of-rules/README.md` - Documentation for rule files
