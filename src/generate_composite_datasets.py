"""
Generate datasets for composite rules by re-labeling base dataset examples.

Uses LLM evaluation to determine which examples from base datasets satisfy
each component rule, then creates balanced composite datasets.
"""
import json
import argparse
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from src.api_caller import Message, create_caller, CacheMode
from src.model_registry import DEFAULT_TEST_MODEL
from src.utils import set_random_seed


async def evaluate_text_against_rule(
    text: str,
    rule_articulation: str,
    caller
) -> bool:
    """Use LLM to evaluate if text satisfies a rule."""
    prompt = f"""Does the following text match this rule?

Rule: {rule_articulation}

Text: {text}

Answer ONLY with 'true' or 'false' (lowercase, no quotes)."""

    messages = [Message(role="user", content=prompt)]
    response = await caller.call(messages)

    # Parse response
    content = response.content.strip().lower()
    if "true" in content and "false" not in content:
        return True
    elif "false" in content and "true" not in content:
        return False
    else:
        # Ambiguous - default to first occurrence
        if content.startswith("true"):
            return True
        return False


async def relabel_examples_for_composite(
    examples: List[Dict[str, Any]],
    rule_a_articulation: str,
    rule_b_articulation: str,
    caller,
    max_examples: int = 400  # Evaluate more than needed so we can sample
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Re-label examples according to both base rules.
    Returns dict with keys: "A+B+", "A+B-", "A-B+", "A-B-"

    PARALLELIZED: Batches all API calls and runs them concurrently.
    """
    print(f"  Evaluating {len(examples)} examples against both rules...")

    # Limit to max_examples for efficiency
    if len(examples) > max_examples:
        examples = random.sample(examples, max_examples)

    # PARALLELIZATION: Create all evaluation tasks upfront
    # We need 2 evaluations per example (rule A and rule B)
    # Run them all in parallel with aggressive batching
    print(f"  Creating {len(examples) * 2} parallel API call tasks...")

    tasks = []
    for example in examples:
        text = example["input"]
        # Each task is a tuple of (eval_a_task, eval_b_task, original_example)
        tasks.append((
            evaluate_text_against_rule(text, rule_a_articulation, caller),
            evaluate_text_against_rule(text, rule_b_articulation, caller),
            example
        ))

    # Run ALL evaluations in parallel (both rules for all examples)
    # This should be ~800 concurrent calls for 400 examples
    all_evals = []
    for task_a, task_b, example in tasks:
        all_evals.append(task_a)
        all_evals.append(task_b)

    print(f"  Running {len(all_evals)} API calls in parallel...")
    results = await asyncio.gather(*all_evals)

    # Parse results back into (label_a, label_b) pairs
    categorized = defaultdict(list)
    for i, (_, _, example) in enumerate(tasks):
        label_a = results[i * 2]      # Every even index is rule A
        label_b = results[i * 2 + 1]  # Every odd index is rule B

        # Categorize
        if label_a and label_b:
            category = "A+B+"
        elif label_a and not label_b:
            category = "A+B-"
        elif not label_a and label_b:
            category = "A-B+"
        else:
            category = "A-B-"

        categorized[category].append({
            "input": example["input"],
            "label_a": label_a,
            "label_b": label_b,
            "category": category
        })

    # Print distribution
    for cat in ["A+B+", "A+B-", "A-B+", "A-B-"]:
        print(f"    {cat}: {len(categorized[cat])} examples")

    return categorized


async def generate_composite_dataset(
    rule: Dict[str, Any],
    datasets_dir: Path,
    caller,
    num_examples: int = 200,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate dataset for a composite rule by re-labeling base dataset examples.
    """
    random.seed(seed)

    rule_a = rule["base_rules"]["rule_a"]
    rule_b = rule["base_rules"]["rule_b"]
    operator = rule["composition_type"]

    print(f"\nGenerating dataset for {rule['rule_id']}")
    print(f"  Base A: {rule_a['rule_id']}")
    print(f"  Base B: {rule_b['rule_id']}")
    print(f"  Operator: {operator}")

    # Load base datasets
    def get_dataset_path(rule_id: str) -> Path:
        candidates = [
            datasets_dir / f"{rule_id}_v3.jsonl",
            datasets_dir / f"{rule_id}.jsonl",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"No dataset found for rule {rule_id}")

    dataset_a_path = get_dataset_path(rule_a["rule_id"])
    dataset_b_path = get_dataset_path(rule_b["rule_id"])

    # Load examples
    def load_dataset(path: Path) -> List[Dict]:
        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    examples_a = load_dataset(dataset_a_path)
    examples_b = load_dataset(dataset_b_path)

    print(f"  Loaded {len(examples_a)} from A, {len(examples_b)} from B")

    # Combine and deduplicate by input text
    all_examples = {ex["input"]: ex for ex in examples_a + examples_b}.values()
    all_examples = list(all_examples)

    print(f"  Combined: {len(all_examples)} unique examples")

    # Re-label according to both rules
    categorized = await relabel_examples_for_composite(
        all_examples,
        rule_a["articulation"],
        rule_b["articulation"],
        caller
    )

    # Sample balanced dataset
    # Target: 25% from each category
    samples_per_category = num_examples // 4

    composite_examples = []

    for category in ["A+B+", "A+B-", "A-B+", "A-B-"]:
        available = categorized[category]

        if len(available) == 0:
            print(f"    WARNING: No examples for {category}, skipping")
            continue

        # Sample up to target
        n_samples = min(samples_per_category, len(available))
        sampled = random.sample(available, n_samples)

        # Assign composite label based on operator
        for item in sampled:
            label_a = item["label_a"]
            label_b = item["label_b"]

            if operator == "AND":
                composite_label = label_a and label_b
            elif operator == "OR":
                composite_label = label_a or label_b
            else:
                raise ValueError(f"Unknown operator: {operator}")

            composite_examples.append({
                "input": item["input"],
                "label": composite_label,
                "rule_id": rule["rule_id"],
                "metadata": {
                    "base_a_label": label_a,
                    "base_b_label": label_b,
                    "category": category,
                    "operator": operator,
                    "version": "compositional_v1"
                }
            })

    # Shuffle
    random.shuffle(composite_examples)

    print(f"  Generated {len(composite_examples)} composite examples")

    # Verify balance
    pos_count = sum(1 for ex in composite_examples if ex["label"])
    neg_count = len(composite_examples) - pos_count
    print(f"  Final balance: {pos_count} positive, {neg_count} negative")

    return composite_examples


async def main():
    parser = argparse.ArgumentParser(description="Generate composite rule datasets")
    parser.add_argument(
        "--composite-rules",
        type=Path,
        default=Path("data/processed/rules/composite_rules_pilot.jsonl"),
        help="Path to composite rules file"
    )
    parser.add_argument(
        "--base-datasets-dir",
        type=Path,
        default=Path("data/datasets_v3"),
        help="Directory containing base rule datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets_compositionality"),
        help="Output directory for composite datasets"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Target number of examples per composite dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEST_MODEL,
        help="Model for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    set_random_seed(args.seed)

    # Create API caller for evaluation
    caller = create_caller(
        model=args.model,
        temperature=0.0,
        cache_mode=CacheMode.PERSISTENT
    )

    # Load composite rules
    print(f"Loading composite rules from {args.composite_rules}...")
    rules = []
    with open(args.composite_rules) as f:
        for line in f:
            rules.append(json.loads(line))
    print(f"Loaded {len(rules)} composite rules")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets for each composite rule
    for rule in rules:
        try:
            dataset = await generate_composite_dataset(
                rule,
                args.base_datasets_dir,
                caller,
                num_examples=args.num_examples,
                seed=args.seed
            )

            # Save dataset
            output_path = args.output_dir / f"{rule['rule_id']}.jsonl"
            with open(output_path, "w") as f:
                for example in dataset:
                    f.write(json.dumps(example) + "\n")

            print(f"  Saved to {output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"Dataset generation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
