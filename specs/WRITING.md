# Writing Guide for Paper

Write LaTeX in `paper/` directory, which you can compile to PDF afterwards.

## Sources to Reference

1. **Research log:** `research_log.md` - Primary source for findings, chronological experiments
2. **Figures:** `experiments/*/figures/` and `out/figures/` - Visualizations
3. **Specs:** `specs/RESEARCH_SPEC.md` - Original research questions and methodology
4. **Presentation advice:** https://www.lesswrong.com/posts/i3b9uQfjJjJkwZF4f/tips-on-empirical-research-slides

## Key Findings to Highlight (All Confirmed in Research Log)

### 1. **Learnability → Articulation → Faithfulness Pipeline**
- **Location:** `research_log.md:695-1080` (faithfulness section)
- **Finding:** Models can learn rules (90%+ accuracy) and articulate them functionally (85-88%), but articulations are only ~70% faithful (multi-shot) or ~51% (zero-shot)
- **Implication:** Articulations are often post-hoc rationalizations, not faithful explanations

### 2. **Degrading Performance with More Examples**
- **Location:** `research_log.md` (needs investigation)
- **Finding:** Some rules show worse performance as few-shot count increases
- **Action needed:** Examine datasets for quality issues, identify patterns

### 3. **Syntactic vs Semantic Rule Articulation**
- **Location:** `research_log.md:400-450` (category analysis)
- **Finding:** Statistical rules (entropy, variance) harder to articulate than syntactic (palindrome, patterns)
  - Palindrome: 16 successful configs (most robust)
  - Entropy rules: 20-40% LLM judge despite 85-100% functional accuracy
- **Note:** Original hypothesis was "syntactic harder" but evidence shows **statistical** rules are actually harder to articulate

### 4. **Model Agreement (Haiku 4.5 vs GPT-4.1-nano)**
- **Location:** `research_log.md:113-117, 325-328`
- **Finding:** Strong agreement across models
  - Claude Haiku 4.5: 88.4% avg functional accuracy
  - GPT-4.1-nano: 84.5% avg functional accuracy
  - Both ~49-50% LLM judge scores

### 5. **Functional Accuracy vs LLM Judge Gap**
- **Location:** `research_log.md:330-333, 389-398`
- **Critical finding:** ~35-40% gap between LLM judge (49%) and functional accuracy (85-88%)
- **Interpretation:** Models articulate rules differently than ground truth but still work operationally
- **V1 issue:** Initial dataset lacked diversity, causing LLM judge and cosine similarity to fail
- **V3 solution:** Diversity optimization improved dataset quality

### 6. **End-to-End Pipeline Documentation**
- **Location:** `research_log.md:167-272` (prompt templates)
- **Components to explain:**
  - Rule generation & curation (brainstorm → curate → generate datasets)
  - Learnability testing (NO CoT, direct classification)
  - Articulation testing (CoT allowed, 3 prompt variations)
  - Faithfulness testing (counterfactual, consistency, functional, cross-context)
  - Full prompts documented in research log lines 176-240

### 7. **Category Trends: Learnability vs Articulation**
- **Location:** `research_log.md:400-450`
- **Interesting pattern:** Need to verify if any category shows opposite trends
- **Known patterns:**
  - Semantic rules: High MC accuracy (87.9%), moderate judge scores (70-80%)
  - Statistical rules: Low judge scores (20-40%), high functional (85-100%)

### 8. **Chain-of-Thought Impact**
- **Location:** `research_log.md:335-348`
- **Finding:** CoT improves articulation by ~7% over simple prompts
- **Example:** `contains_consecutive_repeated_characters_claude_009`
  - Simple: 20-40% LLM judge
  - CoT: 80-100% LLM judge (2-3x improvement!)

## Post-Draft Cleanup Tasks

After completing first draft:
- [ ] Organize figures by paper section
- [ ] Clean up experiment output directories
- [ ] Archive failed/intermediate runs
- [ ] Update config files for reproducibility
- [ ] Verify all referenced datasets exist and are documented

