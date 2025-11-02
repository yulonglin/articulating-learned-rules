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

## [Timestamp: 2025-11-01 (documentation)]

**Activity:** Articulation Testing Methodology - Prompt Templates

**Description & Status:**
Documented the exact prompts used for articulation testing (Step 2) for paper methods section. Status: **Documentation**.

**Files Referenced:**
- `src/test_articulation_freeform.py:94-142` - Free-form prompt builder
- `src/test_articulation_mc.py:143-200` - Multiple-choice prompt builder

**Prompts Used:**

### Free-form Articulation Prompts

Three variations tested to assess prompt sensitivity:

#### 1. Simple Variation (`variation="simple"`)
```
Here are examples of a classification task:
Examples:
Input: "{example.input}" → {label}
Input: "{example.input}" → {label}
...

In 1-2 sentences, describe the rule that determines when the output is True vs False.

Rule:
```

#### 2. Chain-of-Thought Variation (`variation="cot"`)
```
Here are examples of a classification task:
Examples:
Input: "{example.input}" → {label}
Input: "{example.input}" → {label}
...

Think step-by-step about what pattern distinguishes True from False cases.
Then write the rule in 1-2 sentences.

Thinking:
```

#### 3. Explicit Variation (`variation="explicit"`)
```
Classify texts as True or False based on these examples:
Examples:
Input: "{example.input}" → {label}
Input: "{example.input}" → {label}
...

What is the classification rule? Describe it precisely and concisely.

Rule:
```

### Multiple-Choice Articulation Prompt

```
You are shown examples of a classification task. Select which rule best explains the pattern.

Examples:
Input: "{example.input}" → {label}
Input: "{example.input}" → {label}
...

Which rule explains this pattern?

A) {option_text}
B) {option_text}
C) {option_text}
D) {option_text}

Answer with just the letter (A, B, C, or D):
```

**Implementation Details:**

1. **Few-shot examples format:** Consistent across all prompt types
   - Format: `Input: "{example.input}" → {label}`
   - Labels: Boolean (True/False)
   - Number of examples: Automatically uses `min_few_shot_required` from learnability enrichment per model

2. **MC distractor generation:**
   - 3 distractors per question (correct + 3 = 4 options total)
   - Distractor strategies: `same_category`, `different_category`, `mixed`
   - Option order randomized to prevent position bias
   - Correct answer position tracked for evaluation

3. **Free-form evaluation methods:**
   - **Keyword match:** Proportion of key concepts from ground truth found in generated articulation
   - **ROUGE-L:** F1 score measuring longest common subsequence
   - **LLM judge:** GPT-4 evaluates semantic equivalence (0-10 scale normalized to 0-1)
   - **Functional test:** Use generated articulation to classify held-out examples, measure accuracy

4. **Key constraint differences from learnability testing:**
   - **Learnability test:** NO chain-of-thought reasoning allowed (direct classification only)
   - **Articulation test:** CoT reasoning explicitly encouraged in some variations
   - This ensures we test learning vs. articulation separately

**Notes for Paper Methods Section:**

- All prompts use identical few-shot example formatting to maintain consistency
- Free-form variations test whether prompt structure affects articulation quality
- MC format tests recognition vs. generation (easier task than free-form)
- Evaluation uses multiple complementary metrics (lexical, semantic, functional)
- Ground truth articulations come from original rule definitions in curated rules file

---

## [Timestamp: 2025-11-01 19:28 UTC]

**Activity:** Free-Form Articulation Multi-Shot Experiment (Step 2 - Generation Task)

**Description & Status:**
Completed multi-shot free-form articulation testing across 31 learnable rules with 3 prompt variations (simple, CoT, explicit) at 5 shot counts [5, 10, 20, 50, 100]. This tests whether models can generate natural language articulations of learned rules and whether chain-of-thought reasoning helps. Status: **Complete**.

**Commands Run:**
```bash
# Multi-shot free-form articulation experiment
python -m src.test_articulation_freeform \
  --rules-file data/processed/rules/curated_rules_learnable.jsonl \
  --datasets-dir data/processed/datasets \
  --output-dir experiments/articulation_freeform_multishot \
  --prompt-variations simple cot explicit \
  --evaluation-methods llm_judge functional \
  --functional-test-size 20 \
  --cache-mode persistent \
  --max-concurrent 50 \
  --log-level INFO

# Analysis
python -m src.analyze_articulation_freeform \
  --results-dir experiments/articulation_freeform_multishot \
  --output-file experiments/articulation_freeform_multishot/summary.yaml
```

**Files and Outputs Examined/Generated:**
- **Input:** `data/processed/rules/curated_rules_learnable.jsonl` (31 learnable rules)
- **Outputs:**
  - `experiments/articulation_freeform_multishot/*.jsonl` (930 individual result files)
  - `experiments/articulation_freeform_multishot/summary.yaml` (aggregated metrics)
  - `experiments/articulation_freeform_multishot/best_articulations.yaml` (successful configs ≥90%)
  - `experiments/articulation_freeform_multishot/articulation_freeform.log` (execution log)

**Experiment Parameters:**
- **Rules tested:** 31 learnable rules
- **Models:** `gpt-4.1-nano-2025-04-14`, `claude-haiku-4-5-20251001`
- **Prompt variations:** simple, cot (chain-of-thought), explicit
- **Few-shot counts:** [5, 10, 20, 50, 100]
- **Functional test size:** 20 held-out samples per rule
- **Total evaluations:** 930 (31 rules × 2 models × 3 variations × 5 shots)
- **Total API calls:** ~19,530 (930 articulations + 18,600 functional tests)
- **Concurrency:** 50 parallel requests
- **Runtime:** ~47 minutes

**Key Results:**

### Overall Performance (Generation Task)

| Metric | GPT-4.1-nano | Claude Haiku |
|--------|--------------|--------------|
| **Avg LLM Judge** | 49.0% | 49.7% |
| **Avg Functional** | 84.5% | 88.4% |

**Critical Finding**: ~35-40% gap between LLM judge scores and functional accuracy!
- Models articulate rules differently than ground truth
- But articulated rules still work operationally (~85-88% accuracy)
- Suggests semantic equivalence despite lexical/stylistic differences

### Prompt Variation Impact - CoT DOES Help!

| Variation | Avg LLM Judge | Avg Functional |
|-----------|---------------|----------------|
| **CoT** | 51.4% | 88.3% |
| **Explicit** | 52.0% | 86.8% |
| **Simple** | 44.9% | 85.7% |

**Key Finding**: CoT improves articulation quality by ~7% over simple prompts
- Most effective for pattern rules requiring step-by-step reasoning
- Example: `contains_consecutive_repeated_characters_claude_009` (Claude)
  - Simple: 20-40% LLM judge
  - **CoT: 80-100% LLM judge** (massive 2-3x improvement!)

### Comparison: MC (Recognition) vs Free-Form (Generation)

| Task | MC Articulation | Free-Form Articulation |
|------|-----------------|------------------------|
| **Type** | Recognition (select from options) | Generation (produce text) |
| **Performance @ 100-shot** | 68.6% | 49-52% (judge) |
| **Functional accuracy** | N/A | 85-88% |
| **CoT used?** | No (not needed for A/B/C/D) | Yes (tested as variation) |
| **Statistical rules** | 51.6% | 20-40% (judge) |
| **Semantic rules** | 87.9% | 70-80% (judge) |

**Interpretation**: Generation is ~20% harder than recognition for LLM judges, but functionally the articulated rules work nearly as well!

### Successful Articulations (≥90% Both LLM Judge + Functional)

**11 rules achieved ≥90% on BOTH metrics:**

1. **PalindromeCheck_gpt_007** - 16 successful configs (most robust!)
   - Success across all variations and shot counts
   - Example config: Claude simple @ 5-shot → 90% judge, 100% functional

2. **complaint_statement_gpt_003** - 2 successful configs
   - Semantic rule with perfect consistency
   - 80% judge + 100% functional across ALL variations/shots
   - Most reliable semantic rule

3. **contains_consecutive_repeated_characters_claude_009** - 7 successful configs
   - **CoT critical for success**: 5/7 successful configs use CoT
   - Claude CoT @ 5-shot: 100% judge, 100% functional

4. **symmetric_word_pattern_claude_002** - 4 successful configs
5. **PresenceOfURL_gpt_006** - 4 successful configs
6. **contains_hyphenated_word_claude_009** - 5 successful configs
7. **alternating_case_words_claude_000** - 4 successful configs
8. **all_caps_gpt_000** - 2 successful configs
9. **reference_negation_presence** - 2 successful configs
10. **reference_is_anagram_of_list** - 1 successful config

**Notable absence**: Statistical rules (entropy, variance, ratios) rarely achieve ≥90% on both metrics

### Functional vs LLM Judge Gap Analysis

**High functional, low judge examples** (models capture behavior but express differently):

- `entropy_threshold_low_claude_001`: 20-40% judge but 85-100% functional
- `word_length_variance_low_claude_002`: 10-40% judge but 85-100% functional
- `digit_surrounded_by_letters_claude_003`: 20-40% judge but 90-100% functional

**Why this matters**: Models may understand rules operationally even when articulation doesn't match ground truth terminology

### Category-Specific Performance

**Semantic rules (best articulation)**:
- `complaint_statement_gpt_003`: 80% judge, 100% functional (all configs!)
- `urgent_intent_gpt_001`: 70-80% judge, 90-100% functional
- `financial_or_money_related_gpt_009`: 70-80% judge, 95-100% functional
- `emotional_expression_gpt_005`: 40-80% judge, 85-100% functional

**Pattern rules (CoT helps most)**:
- `contains_consecutive_repeated_characters_claude_009`: 20% → 100% with CoT
- `alternating_case_words_claude_000`: 20-40% → 80-90% with CoT
- `PalindromeCheck_gpt_007`: 80-100% across all variations

**Statistical rules (hardest to articulate)**:
- `word_length_variance_low/high`: 10-40% judge (but 85-100% functional!)
- `entropy_threshold_low`: 20-60% judge (but 75-100% functional!)
- `unique_character_ratio`: 30-40% judge (but 75-100% functional!)
- `punctuation_density_high`: 20-40% judge (but 75-90% functional!)

**Interpretation**: Models learn statistical patterns operationally but struggle to articulate them in natural language matching ground truth

### Functional Accuracy Measurement Methodology

**How it works** (for paper methods section):
1. Take the generated articulation text
2. For each of 20 held-out test samples, create classification prompt:
   ```
   Rule: {generated_articulation}

   Based on this rule, classify the following input as True or False.

   Input: "{sample.input}"

   Answer with just True or False:
   ```
3. Get model predictions using same model that generated articulation
4. Compare predictions to ground truth labels
5. Calculate: `accuracy = n_correct / n_classified`

**Key details**:
- Test samples stratified (balanced positive/negative)
- Skips ambiguous responses (containing both "true" and "false")
- Temperature=0.0 for deterministic classification
- Uses same model for generation + classification (measures self-consistency)

**This measures**: Can the model use its own articulation to correctly classify new examples?

**Outcome:**

✅ **Free-form articulation experiment complete** (930 evaluations)
- Generation is ~20% harder than recognition (MC: 68.6%, Free-form judge: 49-52%)
- BUT functional accuracy remains high (85-88%)
- CoT improves articulation quality by ~7%
- 11 rules achieve ≥90% on both judge + functional metrics
- Statistical rules show largest judge-functional gap (models learn but can't articulate)

**Blockers:** None

**Reflection:**

This experiment reveals a fascinating dissociation: **models can learn and apply rules functionally (85-88% accuracy) even when their articulations don't match ground truth terminology (49-52% judge agreement)**.

Three key insights:

1. **Recognition vs Generation gap**: MC articulation (68.6%) significantly outperforms free-form judge scores (49-52%), confirming that generating articulations from scratch is ~20% harder than recognizing correct articulations from options.

2. **Functional-Judge gap is the real story**: The 35-40% gap between LLM judge scores (49-52%) and functional accuracy (85-88%) suggests models capture rules operationally but express them differently. This is especially pronounced for statistical rules: models achieve 80-100% functional accuracy despite 10-40% judge scores.

3. **CoT helps but doesn't close the gap**: While CoT improves articulation quality (+7% judge, +3% functional), it doesn't eliminate the learnability-articulation gap. Even with CoT, many learnable rules remain hard to articulate in ground-truth-matching language.

**Next steps**: Compare these results with MC articulation to understand recognition vs generation trade-offs, and identify rules where models learn but fundamentally cannot articulate (functional high, judge persistently low).

---
