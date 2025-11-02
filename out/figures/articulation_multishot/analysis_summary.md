# Multi-Shot Articulation Analysis: Learnability vs Articulation

## Overall Comparison (100-shot)

- **Mean Learnability:** 88.4%
- **Mean Articulation:** 68.6%
- **Gap:** +19.8% (positive = easier to learn than articulate)

## Category Breakdown (100-shot)

- **Pattern:** Learn=91.0%, Artic=64.7%, Gap=+26.3%
- **Semantic:** Learn=91.4%, Artic=87.9%, Gap=+3.5%
- **Statistical:** Learn=94.7%, Artic=51.6%, Gap=+43.1%
- **Syntactic:** Learn=94.2%, Artic=67.3%, Gap=+26.9%
- **Unknown:** Learn=69.9%, Artic=nan%, Gap=+nan%

## Gap Trends Across Shot Counts

- **Pattern:** +16.4% → +12.5% → +20.3% → +24.1% → +26.3%
- **Semantic:** -4.9% → -0.4% → -0.1% → +1.1% → +3.5%
- **Statistical:** +33.8% → +40.3% → +40.7% → +42.8% → +43.1%
- **Syntactic:** +6.1% → +22.4% → +24.9% → +26.6% → +26.9%
- **Unknown:** +nan% → +nan% → +nan% → +nan% → +nan%

## Articulation Degradation (5-shot → 100-shot)

Found 6 rule-model combinations where articulation degrades >10%:

- `contains_multiple_exclamation_marks_claude_003` (Claude): 76.0% → 34.0% (Δ=+42.0%)
- `symmetric_word_pattern_claude_002` (Claude): 73.0% → 31.0% (Δ=+42.0%)
- `contains_consecutive_repeated_characters_claude_009` (Claude): 62.0% → 29.0% (Δ=+33.0%)
- `reference_negation_presence` (Claude): 60.0% → 32.0% (Δ=+28.0%)
- `Repeated Punctuation_gpt_003` (Claude): 89.0% → 70.0% (Δ=+19.0%)

## Model Agreement on Articulation (100-shot)

- **Pearson correlation (n=19 rules with both models):** r = 0.773
- **Interpretation:** Strong agreement - models find similar rules hard to articulate

## Key Insights

1. **Performance gap persists:** Learnability consistently exceeds articulation across all shot counts
2. **Statistical rules hardest:** Largest gap for statistical rules (learnable but inarticulate)
3. **More examples ≠ better articulation:** Some rules show degrading articulation with more shots
4. **Semantic rules exception:** Smallest gap - models can articulate semantic rules better
5. **CoT doesn't close the gap:** Despite CoT in articulation test, gap remains substantial