# Research Log

## [Timestamp: 2025-10-31 (current session)]

**Activity:** Analysis of Learnability Experiment Results (Step 1 Complete)

**Description & Status:**
Analyzed learnability test results to identify rules achieving ≥90% accuracy. This completes Step 1 of the research pipeline and identifies candidate rules for articulation testing (Step 2). Status: **Complete**.

**Commands Run:**
- Examined existing learnability results:
  ```bash
  ls experiments/learnability/
  cat experiments/learnability/summary.yaml
  ```
- Generated filtered learnable rules file:
  ```bash
  python -m src.enrich_rules_with_learnability \
    --rules-file data/processed/list-of-rules/curated_rules_generated.jsonl \
    --learnability-summary out/learnability/summary_complete.yaml \
    --output-file out/rules/curated_rules_learnable.jsonl
  # Output: Filtered 38 rules to 31 learnable rules
  # Output written to: out/rules/curated_rules_learnable.jsonl
  ```
- Verified output:
  ```bash
  wc -l out/rules/curated_rules_learnable.jsonl
  # 31 out/rules/curated_rules_learnable.jsonl
  ```

**Files and Outputs Examined/Generated:**
- **Input:** `experiments/learnability/summary.yaml` - Aggregated learnability metrics across all rules, models, and few-shot configurations
- **Outputs Generated:**
  - `research_log.md` - This research log entry documenting learnable rules
  - `experiments/learnability/learnable_rules.txt` - Plain text list of 31 learnable rule IDs organized by tier
  - `out/rules/curated_rules_learnable.jsonl` - Filtered JSONL with full metadata and learnability scores for 31 learnable rules (from 38 total)
  - `data/processed/list-of-rules/README.md` - Documentation for rule files and usage instructions
  - `src/enrich_rules_with_learnability.py` - Script used to enrich rules with learnability metadata

**Key Results:**

### Learnability Summary Statistics

Analyzed 44 total rules tested across:
- Models: `gpt-4.1-nano-2025-04-14`, `claude-haiku-4-5-20251001`
- Few-shot configurations: 5, 10, 20, 50, 100 examples
- Test set: 100 held-out examples per rule
- Parse rate: 100% (all responses parseable)

### Learnable Rules (31 total, ≥90% accuracy achievable)

#### Tier 1: Highly Learnable (100% accuracy achieved)
1. `digit_surrounded_by_letters_claude_003` - Perfect across all shots
2. `PalindromeCheck_gpt_007` - Perfect from 10-shot onwards
3. `nested_quotation_depth_claude_078` - Perfect across all shots
4. `Numeric Pattern_gpt_004` - Perfect in many configurations
5. `reference_is_anagram_of_list` - Perfect across all shots
6. `word_length_variance_low_claude_002` - Perfect from 20-shot onwards (Claude) and 98%+ (GPT)
7. `word_length_variance_high_claude_002` - Perfect/near-perfect across all shots

#### Tier 2: Strongly Learnable (90%+ with moderate shots, 10-50)
8. `contains_multiple_exclamation_marks_claude_003` - 94-100% from 10-shot onwards
9. `contains_consecutive_repeated_characters_claude_009` - 100% at 50+ shots
10. `contains_digit_pattern_gpt_005` - 91-100% from 20-shot onwards (Claude)
11. `contains_multiple_punctuation_marks_claude_004` - 92-100% across shots
12. `contains_hyphenated_word_claude_009` - 95-100% from 20-shot onwards (Claude)
13. `alternating_case_words_claude_000` - 96-100% from 20-shot onwards (Claude)
14. `positive_product_review_gpt_000` - 90-97% from 50-shot onwards
15. `urgent_intent_gpt_001` - 95-100% from 5-shot onwards
16. `complaint_statement_gpt_003` - 91-99% across all shots
17. `financial_or_money_related_gpt_009` - 90-100% from 10-shot onwards
18. `emotional_expression_gpt_005` - 91-100% from 10-shot onwards
19. `entropy_threshold_low_claude_001` - 92-100% from 50-shot onwards
20. `PresenceOfURL_gpt_006` - 96-100% across all shots
21. `unique_character_ratio_claude_009` - 92-100% from 10-shot onwards
22. `all_caps_gpt_000` - 90-96% from 10-shot onwards (Claude)
23. `Repeated Punctuation_gpt_003` - 90-98% from 20-shot onwards (Claude)
24. `word_count_less_than_5_gpt_004` - 90-94% at various shots (Claude)

#### Tier 3: Marginally Learnable (90%+ only at high shots, 50-100)
25. `symmetric_word_pattern_claude_002` - 93% at 100-shot (Claude)
26. `digit_to_letter_ratio_claude_004` - 91% at 100-shot (Claude)
27. `punctuation_density_high_claude_004` - 90-97% from 10-50+ shots
28. `word_length_fibonacci_claude_084` - 95-99% from 20-shot onwards (Claude)
29. `reference_third_person_perspective` - 95% at various shots (Claude)
30. `reference_negation_presence` - 90% at 100-shot (Claude)
31. `reference_first_person_perspective` - 97% at 100-shot (Claude)

#### Not Learnable (<90% across all configurations)
13 rules failed to reach 90% accuracy:
- `semantic_animal_color_binding_claude_085`
- `Part-of-Speech Pattern_gpt_007`
- `reference_starts_and_ends_same_char`
- `reference_is_adjective`
- `reference_rhyming_ends`
- `reference_starts_with_vowel`
- `reference_word_count_between_3_and_7`
- Plus 6 others below threshold

**Outcome:**

✅ **31 rules identified as learnable** (≥90% accuracy achievable)
- These rules are ready for articulation testing (Step 2)
- Models can perform classification task with high accuracy
- Next step: Test if models can articulate the learned rules

❌ **13 rules not learnable** (<90% accuracy)
- These rules are unsuitable for articulation testing
- Models struggle with the classification task itself
- Should be excluded from further analysis

**Key Patterns Observed:**
1. **Syntactic rules** (palindromes, digit patterns, punctuation) are highly learnable
2. **Semantic rules** (animal-color binding, POS patterns) are less learnable
3. **Claude Haiku** generally achieves 90%+ with fewer shots than GPT-4.1-nano
4. **Reference-based rules** (rhyming, starts with vowel) are particularly difficult

**Next Steps:**

1. **Run articulation tests on learnable rules:**
   ```bash
   # Multiple-choice articulation test
   python -m src.test_articulation_mc \
     --rules-file data/curated_rules.jsonl \
     --datasets-dir experiments/datasets \
     --output-dir experiments/articulation_mc \
     --few-shot-count 10 \
     --num-test-cases 5 \
     --distractor-strategy mixed \
     --models gpt-4.1-nano-2025-04-14 claude-haiku-4-5-20251001

   # Free-form articulation test
   python -m src.test_articulation_freeform \
     --rules-file data/curated_rules.jsonl \
     --datasets-dir experiments/datasets \
     --output-dir experiments/articulation_freeform \
     --few-shot-count 10 \
     --prompt-variations simple cot explicit \
     --evaluation-methods keyword rouge llm_judge functional \
     --models gpt-4.1-nano-2025-04-14 claude-haiku-4-5-20251001
   ```

2. **Use enriched learnable rules file:**
   - Use `out/rules/curated_rules_learnable.jsonl` containing only the 31 learnable rules with model-specific learnability metadata
   - This reduces API costs and experiment runtime, and allows articulation tests to use `min_few_shot_required` per model

3. **Expected articulation test outcomes:**
   - **Hypothesis:** Some rules will be learnable but not articulable
   - **Measurement:** MC accuracy, free-form evaluation scores
   - **Success criteria:** Clear cases where learnability ≥90% but articulation <70%

**Blockers:** None

**Reflection:**

This analysis confirms that Step 1 (learnability testing) was successful. The distribution of learnable vs. non-learnable rules is reasonable (~70% learnable), providing sufficient candidates for articulation testing while also identifying genuinely difficult rules.

The strong performance on syntactic rules vs. weak performance on semantic/reference-based rules aligns with expectations about what patterns are easier for LLMs to learn from few-shot examples.

The research question "Can LLMs learn rules but fail to articulate them?" can now be tested with 31 learnable rules. The diversity of rule types (syntactic, semantic, statistical) provides good coverage for identifying potential dissociations between learning and articulation.

---
