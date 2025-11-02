# Learnability vs Functional Accuracy Analysis

## Overall Comparison (100-shot)

- **Mean Learnability:** 88.4%
- **Mean Functional (COT):** 89.1%
- **Mean Functional (Explicit):** 88.1%
- **Mean Functional (Simple):** 89.2%
- **Mean Functional (all variations):** 88.8%
- **Gap:** -0.4% (positive = easier to learn than to articulate functionally)

## Category Breakdown (100-shot)

- **Pattern:** Learn=91.0%, Functional=93.2%, Gap=-2.2%
- **Semantic:** Learn=91.4%, Functional=90.5%, Gap=+0.9%
- **Statistical:** Learn=94.7%, Functional=87.2%, Gap=+7.5%
- **Syntactic:** Learn=94.2%, Functional=84.6%, Gap=+9.6%
- **Unknown:** Learn=69.9%, Functional=nan%, Gap=+nan%

## Prompt Variation Performance (100-shot)

- **COT:** 89.1% (gaps: patt=-8.2%, sema=+3.1%, stat=+9.1%, synt=+10.4%)
- **Explicit:** 88.1% (gaps: patt=+0.9%, sema=-1.1%, stat=+9.8%, synt=+9.5%)
- **Simple:** 89.2% (gaps: patt=+0.7%, sema=+0.6%, stat=+3.8%, synt=+8.9%)

## Gap Trends Across Shot Counts (averaged across variations)

- **Pattern:** +5.5% → -1.6% → -3.4% → -3.4% → -2.2%
- **Semantic:** -5.1% → +0.2% → -3.8% → -2.4% → +0.9%
- **Statistical:** +6.7% → +7.1% → +4.0% → +7.5% → +7.5%
- **Syntactic:** +0.1% → +5.1% → +6.0% → +4.8% → +9.6%
- **Unknown:** +nan% → +nan% → +nan% → +nan% → +nan%

## Functional Accuracy Degradation (5-shot → 100-shot)

Found 3 rule-model combinations where functional accuracy degrades >10%:

- `nested_quotation_depth_claude_078` (Claude): 100.0% → 50.0% (Δ=+50.0%)
- `entropy_threshold_low_claude_001` (GPT): 100.0% → 72.0% (Δ=+28.0%)
- `contains_multiple_punctuation_marks_claude_004` (Claude): 100.0% → 76.3% (Δ=+23.7%)

## Model Agreement on Functional Accuracy (100-shot)

- **Pearson correlation (n=19 rules with both models):** r = -0.077
- **Interpretation:** Weak agreement - models differ in which rules produce functional articulations

## Key Insights

1. **Functional accuracy measures whether articulated rules actually work for classification**
2. **Gap between learnability and functional:** Models can classify but struggle to articulate working rules
3. **Best prompt variation:** Simple achieves highest functional accuracy (89.2%)
4. **Hardest category:** Syntactic (gap=+9.6%) - models learn but can't articulate functional rules
5. **Easiest category:** Pattern (gap=-2.2%) - functional articulations work better