"""
Create composite rule definitions from pairs of existing rules.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def load_rules(rules_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load rules from JSONL file into dict keyed by rule_id."""
    rules = {}
    with open(rules_path) as f:
        for line in f:
            rule = json.loads(line)
            rules[rule["rule_id"]] = rule
    return rules


def create_composite_rule(
    rule_a: Dict[str, Any],
    rule_b: Dict[str, Any],
    operator: str,
    composite_id: str
) -> Dict[str, Any]:
    """Create a composite rule from two base rules and an operator."""

    # Create articulation
    if operator == "AND":
        articulation = (
            f"The input is labeled True if it satisfies BOTH of the following conditions: "
            f"(1) {rule_a['articulation']} "
            f"(2) {rule_b['articulation']}"
        )
        expected_difficulty = "hard"  # Composition increases difficulty
    elif operator == "OR":
        articulation = (
            f"The input is labeled True if it satisfies AT LEAST ONE of the following conditions: "
            f"(1) {rule_a['articulation']} "
            f"(2) {rule_b['articulation']}"
        )
        expected_difficulty = "moderate"  # OR is usually easier than AND
    else:
        raise ValueError(f"Unknown operator: {operator}")

    # Determine composite category
    cats = sorted([rule_a["category"], rule_b["category"]])
    if cats[0] == cats[1]:
        composite_category = cats[0]
    else:
        composite_category = f"{cats[0]}_x_{cats[1]}"

    # Create composite rule
    composite_rule = {
        "rule_id": composite_id,
        "rule_name": f"{rule_a['rule_name']}_{operator}_{rule_b['rule_name']}",
        "articulation": articulation,
        "category": f"compositional_{composite_category}",
        "composition_type": operator,
        "base_rules": {
            "rule_a": {
                "rule_id": rule_a["rule_id"],
                "articulation": rule_a["articulation"],
                "category": rule_a["category"]
            },
            "rule_b": {
                "rule_id": rule_b["rule_id"],
                "articulation": rule_b["articulation"],
                "category": rule_b["category"]
            }
        },
        "examples": [],  # Will be filled during dataset generation
        "expected_difficulty": expected_difficulty,
        "source_model": "composite",
        "timestamp": datetime.now().isoformat(),
        "prompt_strategy": "compositional",
        "implementability": "composite",  # Can be derived from base rules
        "quality_score": 1.0,
    }

    return composite_rule


def main():
    parser = argparse.ArgumentParser(description="Create composite rule definitions")
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=Path("data/processed/rules/curated_rules_learnable.jsonl"),
        help="Path to learnable rules file"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/processed/rules/composite_rules_pilot.jsonl"),
        help="Path to output composite rules file"
    )
    args = parser.parse_args()

    # Load existing rules
    print(f"Loading rules from {args.rules_file}...")
    rules = load_rules(args.rules_file)

    # Define the 5 pairs based on faithfulness analysis
    rule_pairs = [
        # Pair 1: Syntactic + Syntactic
        ("contains_multiple_punctuation_marks_claude_004", "contains_hyphenated_word_claude_009"),
        # Pair 2: Semantic + Semantic
        ("positive_product_review_gpt_000", "urgent_intent_gpt_001"),
        # Pair 3: Syntactic + Semantic
        ("contains_multiple_punctuation_marks_claude_004", "positive_product_review_gpt_000"),
        # Pair 4: Pattern + Statistical
        ("Numeric Pattern_gpt_004", "digit_to_letter_ratio_claude_004"),
        # Pair 5: Syntactic + Semantic (variant)
        ("contains_hyphenated_word_claude_009", "positive_product_review_gpt_000"),
    ]

    composite_rules = []

    # Generate composite rules for each pair with AND/OR operators
    for i, (rule_a_id, rule_b_id) in enumerate(rule_pairs, 1):
        rule_a = rules[rule_a_id]
        rule_b = rules[rule_b_id]

        # Create AND composite
        and_id = f"comp_pair{i:02d}_AND"
        and_rule = create_composite_rule(rule_a, rule_b, "AND", and_id)
        composite_rules.append(and_rule)

        # Create OR composite
        or_id = f"comp_pair{i:02d}_OR"
        or_rule = create_composite_rule(rule_a, rule_b, "OR", or_id)
        composite_rules.append(or_rule)

        print(f"\nPair {i}:")
        print(f"  Base A: {rule_a_id} ({rule_a['category']})")
        print(f"  Base B: {rule_b_id} ({rule_b['category']})")
        print(f"  Created: {and_id}, {or_id}")

    # Save composite rules
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        for rule in composite_rules:
            f.write(json.dumps(rule) + "\n")

    print(f"\n{'='*80}")
    print(f"Created {len(composite_rules)} composite rules")
    print(f"Saved to: {args.output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
