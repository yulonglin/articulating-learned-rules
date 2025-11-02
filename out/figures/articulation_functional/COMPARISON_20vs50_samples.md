# Functional Test Size Comparison: 20 vs 50 Samples

## Motivation
The original functional tests used only 20 samples, which may not expose edge cases where articulated rules fail. Increasing to 50 samples provides more robust evaluation.

## Overall Results (100-shot)

### 20 Samples (Original)
- **Mean Learnability:** 88.4%
- **Mean Functional (all variations):** 89.5%
- **Gap:** -1.2% (functional > learnability)

### 50 Samples (New)
- **Mean Learnability:** 88.4%
- **Mean Functional (all variations):** 88.8%
- **Gap:** -0.4% (functional ≈ learnability)

## Key Changes

**Functional Accuracy Decreased:** 89.5% → 88.8% (-0.7%)
- With more test samples, articulated rules fail more often
- Gap narrowed from -1.2% to -0.4%
- Functional accuracy now nearly equal to learnability

## Category-Specific Changes (100-shot)

| Category | Functional@20 | Functional@50 | Change |
|----------|--------------|--------------|--------|
| **Pattern** | 93.1% | 93.2% | +0.1% |
| **Semantic** | 90.1% | 90.5% | +0.4% |
| **Statistical** | 89.1% | 87.2% | -1.9% ⚠️ |
| **Syntactic** | 86.3% | 84.6% | -1.7% ⚠️ |

**Statistical and Syntactic rules** showed the largest drops, suggesting these categories have more edge cases that the 20-sample tests missed.

## Gap Changes by Category (100-shot)

| Category | Gap@20 | Gap@50 | Change |
|----------|--------|--------|--------|
| **Pattern** | -2.1% | -2.2% | -0.1% |
| **Semantic** | +1.2% | +0.9% | -0.3% |
| **Statistical** | +5.6% | +7.5% | +1.9% ⚠️ |
| **Syntactic** | +7.9% | +9.6% | +1.7% ⚠️ |

**Interpretation:**
- **Pattern & Semantic:** Stable - articulated rules generalize well
- **Statistical & Syntactic:** Growing gap - models learn these rules but struggle to articulate them in a way that works on diverse test cases

## Degrading Rules

### 20 Samples
- 3 rules with >10% degradation (5-shot → 100-shot)

### 50 Samples  
- 3 rules with >10% degradation (same count)
- But degradation is worse:
  - `nested_quotation_depth_claude_078`: 100.0% → 50.0% (Δ=+50.0%)
  - `entropy_threshold_low_claude_001`: 100.0% → 72.0% (Δ=+28.0%)
  - `contains_multiple_punctuation_marks_claude_004`: 100.0% → 76.3% (Δ=+23.7%)

## Conclusions

1. **50 samples is more realistic**: Functional accuracy dropped by 0.7%, revealing that 20 samples was too easy

2. **Statistical & Syntactic rules are hardest**: Largest drops and growing gaps suggest these are genuinely hard to articulate functionally

3. **Pattern rules are easiest**: Functional accuracy actually improved slightly, suggesting articulations generalize well

4. **Gap is now minimal overall**: -0.4% gap means functional ≈ learnability, but this masks category-specific patterns

5. **Dataset diversity matters**: Your initial concern was correct - base datasets need more diverse edge cases to properly evaluate functional accuracy

## Recommendation

The 50-sample functional test is more robust and should be used going forward. The narrowing gap suggests articulated rules work reasonably well overall, but category-specific analysis reveals that **Statistical and Syntactic rules remain challenging** for models to articulate in a functionally accurate way.
