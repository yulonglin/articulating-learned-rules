"""
Augment existing datasets by adding more examples to rules that are close to the threshold.

This script reads existing datasets, identifies rules that need more examples,
and generates only the needed examples to reach the target.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from src.generate_datasets import LLMGenerator, LLMEvaluator, EVAL_FUNCTIONS, LLM_EVAL_RULES
from src.utils import load_jsonl
from src.data_models import Rule, DatasetSample


async def augment_rule_dataset(
    rule: Rule,
    dataset_path: Path,
    target_positive: int = 100,
    target_negative: int = 100,
    max_attempts: int = 10
) -> int:
    """Augment a dataset by adding more examples to reach target counts.

    Returns:
        Number of new examples added
    """
    # Load existing dataset
    existing_samples = load_jsonl(dataset_path)

    # Count current labels
    pos_count = sum(1 for s in existing_samples if s.get("label"))
    neg_count = len(existing_samples) - pos_count

    needed_pos = max(0, target_positive - pos_count)
    needed_neg = max(0, target_negative - neg_count)

    if needed_pos == 0 and needed_neg == 0:
        print(f"  {rule.rule_id}: Already meets target ({pos_count}+/{neg_count}-)")
        return 0

    print(f"  {rule.rule_id}: Need {needed_pos} more positive, {needed_neg} more negative (currently {pos_count}+/{neg_count}-)")

    # Setup generator and evaluator
    generator = LLMGenerator(rule=rule, random_seed=42, max_concurrent=50)

    needs_llm_eval = rule.rule_name in LLM_EVAL_RULES
    if needs_llm_eval:
        evaluator = LLMEvaluator(rule=rule)
    else:
        eval_fn = EVAL_FUNCTIONS.get(rule.rule_name)
        if eval_fn is None:
            print(f"    ERROR: No evaluation function for {rule.rule_name}")
            return 0

    async def evaluate_input(text: str) -> bool:
        if needs_llm_eval:
            return await evaluator.evaluate(text)
        return eval_fn(text)

    # Collect existing texts to avoid duplicates
    existing_texts = {s["input"].lower().strip() for s in existing_samples}

    new_samples = []

    # Generate needed positive examples
    if needed_pos > 0:
        for attempt in range(max_attempts):
            batch_size = min(needed_pos - len([s for s in new_samples if s["label"]]), 20)
            if batch_size <= 0:
                break

            candidates = await generator.generate_batch_v3(
                batch_size=batch_size,
                target_label=True,
                batch_type="diversity",
                temperature=0.8
            )

            for candidate in candidates:
                if candidate.lower().strip() in existing_texts:
                    continue

                try:
                    actual = await evaluate_input(candidate)
                    if actual:
                        new_samples.append({
                            "input": candidate,
                            "label": True,
                            "rule_id": rule.rule_id,
                            "metadata": {"augmented": True}
                        })
                        existing_texts.add(candidate.lower().strip())

                        if len([s for s in new_samples if s["label"]]) >= needed_pos:
                            break
                except Exception:
                    continue

    # Generate needed negative examples
    if needed_neg > 0:
        for attempt in range(max_attempts):
            batch_size = min(needed_neg - len([s for s in new_samples if not s["label"]]), 20)
            if batch_size <= 0:
                break

            candidates = await generator.generate_batch_v3(
                batch_size=batch_size,
                target_label=False,
                batch_type="diversity",
                temperature=0.8
            )

            for candidate in candidates:
                if candidate.lower().strip() in existing_texts:
                    continue

                try:
                    actual = await evaluate_input(candidate)
                    if not actual:
                        new_samples.append({
                            "input": candidate,
                            "label": False,
                            "rule_id": rule.rule_id,
                            "metadata": {"augmented": True}
                        })
                        existing_texts.add(candidate.lower().strip())

                        if len([s for s in new_samples if not s["label"]]) >= needed_neg:
                            break
                except Exception:
                    continue

    # Append new samples to dataset
    if new_samples:
        with open(dataset_path, 'a') as f:
            for sample in new_samples:
                f.write(json.dumps(sample) + '\n')

        new_pos = len([s for s in new_samples if s["label"]])
        new_neg = len(new_samples) - new_pos
        print(f"    Added {new_pos} positive, {new_neg} negative samples")

    return len(new_samples)


async def main():
    parser = argparse.ArgumentParser(description="Augment datasets to reach target counts")
    parser.add_argument("--rules-file", type=str, required=True)
    parser.add_argument("--datasets-dir", type=str, required=True)
    parser.add_argument("--min-positive", type=int, default=100)
    parser.add_argument("--min-negative", type=int, default=100)
    parser.add_argument("--max-gap", type=int, default=10, help="Only augment if within this gap of target")

    args = parser.parse_args()

    rules_file = Path(args.rules_file)
    datasets_dir = Path(args.datasets_dir)

    # Load all rules
    rules = []
    with open(rules_file) as f:
        for line in f:
            if line.strip():
                rules.append(Rule(**json.loads(line)))

    print(f"Loaded {len(rules)} rules from {rules_file}")
    print(f"Target: {args.min_positive} positive, {args.min_negative} negative")
    print(f"Only augmenting rules within {args.max_gap} of target\n")

    # Process each rule
    total_added = 0
    rules_augmented = 0

    for rule in rules:
        # Find dataset file
        dataset_path = datasets_dir / f"{rule.rule_id}_v3.jsonl"
        if not dataset_path.exists():
            dataset_path = datasets_dir / f"{rule.rule_id}.jsonl"

        if not dataset_path.exists():
            continue

        # Check current counts
        existing_samples = load_jsonl(dataset_path)
        pos_count = sum(1 for s in existing_samples if s.get("label"))
        neg_count = len(existing_samples) - pos_count

        pos_gap = args.min_positive - pos_count
        neg_gap = args.min_negative - neg_count

        # Only augment if close to target
        if (0 < pos_gap <= args.max_gap) or (0 < neg_gap <= args.max_gap):
            added = await augment_rule_dataset(
                rule, dataset_path, args.min_positive, args.min_negative
            )
            if added > 0:
                total_added += added
                rules_augmented += 1

    print(f"\n{'='*60}")
    print(f"Augmented {rules_augmented} rules")
    print(f"Added {total_added} total examples")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
