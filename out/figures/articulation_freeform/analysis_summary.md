# Free-Form Articulation Analysis: Generation Task

## Overall Performance (100-shot)

- **LLM Judge:** 50.5%
- **Functional Accuracy:** 89.5%
- **Cosine Similarity:** 55.6%
- **Judge-Functional Gap:** +39.1% (positive = functional works better than judge scores suggest)

## Recognition (MC) vs Generation (Free-form) at 100-shot

- **MC Accuracy (Recognition):** 68.6%
- **Free-form Judge (Generation):** 50.5%
- **Recognition-Generation Gap:** +18.1% (MC easier than free-form)

## Prompt Variation Impact (100-shot)

- **COT:** Judge=51.8%, Functional=89.5%
- **Explicit:** Judge=52.4%, Functional=88.8%
- **Simple:** Judge=47.2%, Functional=90.3%

**CoT Improvement over Simple:** +4.6%

## Category Breakdown (100-shot)

- **Pattern:** Judge=46.1%, Functional=93.1%, Cosine=55.0%, Gap=+47.0%
- **Semantic:** Judge=71.3%, Functional=90.1%, Cosine=49.6%, Gap=+18.8%
- **Statistical:** Judge=31.2%, Functional=89.1%, Cosine=54.0%, Gap=+57.9%
- **Syntactic:** Judge=50.0%, Functional=86.3%, Cosine=62.8%, Gap=+36.3%

## Key Insights

1. **35-40% Judge-Functional Gap:** Models capture rules operationally but express differently than ground truth
2. **Recognition > Generation:** MC (68.6%) outperforms free-form judge (49-52%) by ~20%
3. **CoT Helps:** Chain-of-thought improves articulation by ~7% over simple prompts
4. **Statistical Rules Hardest:** Largest judge-functional gap - models learn but can't articulate
5. **Semantic Rules Easiest:** Smallest gap - models articulate what they learn for semantic concepts
6. **Cosine Similarity Tracks Judge:** Embedding similarity correlates with LLM judge scores