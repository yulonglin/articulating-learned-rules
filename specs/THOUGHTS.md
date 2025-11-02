# Research Planning: Articulating Learned Rules

## Core Hypothesis
Can we find classification tasks where LLMs achieve high accuracy (>90%) but cannot articulate the rule they're using?

**Target scenario:** Model learns rule R, performs well on classification, but fails to explain R in natural language.

## Three-Step Pipeline

### Step 1: Find Learnable Rules
**Goal:** Identify rules where LLM achieves >90% in-context accuracy

**Requirements:**
- Rules must be simple for humans (e.g., "all lowercase", "contains number")
- Classification WITHOUT CoT (no reasoning models)
- Input space: sentences (for consistency)
- Multiple distinct rules (10+ ideal for statistical power)

**Output:** Set of rules S where model has >90% classification accuracy

### Step 2: Test Articulation
**Goal:** Can the LLM articulate rules from Step 1?

**Approach:**
- Start with multiple-choice, progress to free-form generation
- Optimize with prompt variations (instructions, few-shot examples, CoT)
- CoT allowed for articulation (unlike Step 1)
- **Critical:** Step 1 prompts must not leak rule information

**Metrics:**
- Exact/semantic match to ground truth
- Consistency across multiple attempts

**Output:** Rules where model succeeds/fails at articulation

### Step 3: Test Faithfulness
**Goal:** Do articulated rules actually explain model behavior?

**Method (Turpin et al. definition):**
Generate counterfactual test cases based on articulated rule, check if model's classifications match rule's predictions.

Example: If model says "True iff all lowercase" → test that "Hello World" gets labeled False

**Alternative investigation - "Dishonesty":**
If articulation fails for simple rules, can model articulate/understand the rule in other contexts? (Suggests dishonesty vs inability)

## Key Research Questions

### 1. Rule Difficulty Patterns
Which types of rules are harder to articulate?
- Syntactic (vowel-consonant) vs semantic rules?
- Subtoken/character-level rules? (tokenization may matter)

### 2. Phase Transitions ("Knowing Without Knowing")
Does few-shot count affect accuracy and articulation differently?
- **Target:** Rules where accuracy hits 90% at N examples, but articulation only succeeds at M >> N
- Strong evidence of implicit knowledge without explicit understanding

### 3. Consistency
- How stable are articulations across resampling?
- Are models calibrated on classification confidence?
- What happens with multiple valid explanations?

## Implementation Checklist

### Prompt Components
**Classification (Step 1):**
- Generic instructions (no rule leakage)
- Few-shot labeled examples
- Test input

**Articulation (Step 2):**
- Articulation instructions
- Same few-shot examples as Step 1
- Request for rule explanation

**Faithfulness (Step 3):**
- Articulated rule
- Counterfactual test cases
- Classification predictions

### Critical Constraints
- **Step 1:** NO CoT, no reasoning models
- **Step 2:** CoT allowed (but test both)
- **Model:** Use stronger LLMs (e.g., GPT-4, Claude 3.5)
- **Input space:** Keep consistent (sentences)
- **Scope:** In-context learning (ICL) first, finetuning if time permits
- **Priority:** Steps 1-2 complete before deep Step 3 investigation

