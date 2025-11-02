# Research Analysis Visualization Guide

This guide explains the **research-focused** visualizations (not the descriptive KDE plots).

## Context: Core Research Questions

From `specs/RESEARCH_SPEC.md`, the goal is to test:

**Step 1 → Step 2**: Can LLMs **learn** rules (>90% accuracy) but **fail to articulate** them?
**Step 2 → Step 3**: Do **good articulations** actually **faithfully explain** behavior?

## The 4 Key Figures

### Figure Q1: Learnability vs Articulation
**File:** `research_q1_learnability_vs_articulation.png`

**Question:** Can models learn rules but fail to articulate them?

**What to look for:**
- **Points ABOVE diagonal** = "Knowing without knowing" (high learning, low articulation)
- **Points ON diagonal** = Learning and articulation scale together
- **Red shaded region** = High learning (>90%) but poor articulation (<90%)

**Interpretation:**
- If many points are above diagonal → Evidence for implicit knowledge without explicit understanding
- If points cluster on diagonal → No dissociation between learning and articulation
- Annotated points show the biggest "articulation gaps"

**Why this matters:**
This is the core phenomenon from THOUGHTS.md "phase transitions" - where accuracy plateaus before articulation improves.

---

### Figure Q2: Articulation vs Faithfulness
**File:** `research_q2_articulation_vs_faithfulness.png`

**Question:** Do good articulations faithfully explain behavior?

**What to look for:**
- **Points BELOW diagonal** = Unfaithful articulation (good explanation, but doesn't predict behavior)
- **Points ON diagonal** = Articulation quality predicts faithfulness
- **Red shaded region** = High articulation quality (>80%) but low faithfulness (<50%)

**Interpretation:**
- Points below diagonal suggest **post-hoc rationalization** (model explains well but that's not the actual rule it uses)
- Points on diagonal suggest **faithful explanations** (articulation genuinely explains behavior)
- This tests RESEARCH_SPEC.md Step 3: "Does the articulation explain counterfactual behavior?"

**Why this matters:**
From RESEARCH_SPEC.md: "The rule should explain counterfactually to a human what the model would do otherwise."
Points below diagonal violate this - the model says one rule but behaves according to a different one.

---

### Figure Q3: Learnability vs Faithfulness
**File:** `research_q3_learnability_vs_faithfulness.png`

**Question:** Do easily-learned rules have faithful articulations?

**What to look for:**
- **High learn, Low faithful** (red shaded region) = Model learned well but articulations don't explain behavior
- **Correlation strength** = Does learning difficulty predict faithfulness difficulty?

**Interpretation:**
- Weak correlation → Learning and faithfulness are independent
- Strong correlation → Rules that are easy to learn also have faithful articulations
- Points in red region → Problematic cases where model performs well but can't explain faithfully

**Why this matters:**
Tests whether the quality of implicit learning (Step 1) predicts the quality of explicit explanation (Step 3).

---

### Figure: Case Study Quadrants
**File:** `research_case_study_quadrants.png`

**Question:** What are the interesting patterns across the full space?

**Four quadrants:**

1. **High learn, High articulate** (Green, bottom-right)
   - Expected/ideal case
   - Model learns and can explain

2. **High learn, Low articulate** (Red, top-right)
   - **"KNOWING WITHOUT KNOWING"** ← Most interesting!
   - Evidence that model uses implicit knowledge it can't articulate
   - These are your best case studies for the phenomenon

3. **Low learn, High articulate** (Orange, bottom-left)
   - Suspicious/concerning
   - Model explains well but didn't actually learn
   - Possible spurious correlations or overfitting to surface features

4. **Low learn, Low articulate** (Gray, top-left)
   - Expected/uninteresting
   - Model didn't learn, can't explain

**What to do with this:**
- Count how many rules fall into each quadrant
- Focus qualitative analysis on quadrant 2 (red)
- Investigate quadrant 3 (orange) cases for spurious features

---

## How To Interpret Your Results

### Strong Evidence for "Knowing Without Knowing":
- Many points above diagonal in Q1
- Quadrant 2 (red) has >5 cases
- Examples: Rules with high learnability but <70% articulation

### Evidence for Post-Hoc Rationalization:
- Points below diagonal in Q2
- High functional accuracy (>85%) but low faithfulness (<60%)
- Model says rule R but counterfactual tests show it uses rule R'

### Null Results (Still Interesting!):
- Points cluster on diagonals in Q1, Q2, Q3 → Learning, articulation, and faithfulness all scale together
- This suggests: No dissociation between implicit and explicit knowledge (at least for these rules)

---

## Comparison to Original Faithfulness Figures

**KDE plots (fig1a-c, fig2):**
- ❌ Don't answer causal questions
- ❌ Don't show individual rules
- ✅ Good for: "What's the distribution of faithfulness?"
- **Verdict:** Less useful for research questions, more for descriptive statistics

**fig4 (functional vs faithfulness):**
- ✅ This IS useful - similar to Research Q2
- But doesn't include learnability metrics

**fig6 (correlation heatmap):**
- ✅ Useful summary of relationships
- But doesn't show individual cases or quadrants

**Research figures (Q1-Q3, quadrants):**
- ✅ Directly answer the core research questions
- ✅ Show individual rules (can identify case studies)
- ✅ Highlight interesting regions (knowing without knowing, post-hoc rationalization)

---

## What To Report

### For Paper/Report:

**Essential figures:**
1. Research Q1 (learnability vs articulation) - shows main phenomenon
2. Case Study Quadrants - summarizes all patterns
3. One or two example rules from "knowing without knowing" quadrant with qualitative analysis

**Optional figures:**
- Research Q2 if you find evidence of unfaithful articulations
- Research Q3 if there's an interesting correlation pattern
- Original fig4, fig6, fig7 as supplementary material

**Tables to include:**
- Count of rules in each quadrant
- List of top 5 "knowing without knowing" cases with their metrics
- Example articulations from faithful vs unfaithful cases

---

## Next Steps for Analysis

1. **Identify case studies**: Look at annotated points in Q1 and red quadrant
2. **Qualitative analysis**: For top "knowing without knowing" cases, examine:
   - What's the actual rule?
   - What did the model articulate?
   - Why might the gap exist? (e.g., character-level rules, XOR patterns, statistical rules)
3. **Check for patterns**: Do certain rule categories fall into "knowing without knowing" more often?
4. **Test faithfulness**: For high-articulation cases, verify if counterfactuals actually work
