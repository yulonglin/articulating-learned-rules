"""
Filter rules to only those with successfully generated balanced datasets.

Reads generated datasets and filters to rules with sufficient balanced data:
- At least 100 positive examples
- At least 100 negative examples

Outputs filtered rules to curated_rules_generated_v3.jsonl.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def count_labels(dataset_path: Path) -> Dict[str, int]:
    """Count positive and negative examples in a dataset."""
    pos_count = 0
    neg_count = 0

    with open(dataset_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("label"):
                pos_count += 1
            else:
                neg_count += 1

    return {"positive": pos_count, "negative": neg_count}


def main():
    parser = argparse.ArgumentParser(
        description="Filter rules to only those with successfully generated balanced datasets"
    )
    parser.add_argument(
        "--rules-file",
        type=str,
        required=True,
        help="Path to curated rules JSONL file"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        required=True,
        help="Directory containing generated datasets"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output path for filtered rules"
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=100,
        help="Minimum number of positive examples required (default: 100)"
    )
    parser.add_argument(
        "--min-negative",
        type=int,
        default=100,
        help="Minimum number of negative examples required (default: 100)"
    )

    args = parser.parse_args()

    rules_file = Path(args.rules_file)
    datasets_dir = Path(args.datasets_dir)
    output_file = Path(args.output_file)

    # Load all rules
    rules = []
    with open(rules_file) as f:
        for line in f:
            if line.strip():
                rules.append(json.loads(line))

    print(f"Loaded {len(rules)} rules from {rules_file}")
    print(f"Filtering for rules with ≥{args.min_positive} positive and ≥{args.min_negative} negative examples\n")

    # Check each rule's dataset
    filtered_rules = []
    for rule in rules:
        rule_id = rule["rule_id"]

        # Try different naming patterns
        possible_paths = [
            datasets_dir / f"{rule_id}_v3.jsonl",
            datasets_dir / f"{rule_id}.jsonl",
        ]

        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break

        if not dataset_path:
            print(f"❌ {rule_id}: Dataset not found")
            continue

        # Count labels
        counts = count_labels(dataset_path)
        pos = counts["positive"]
        neg = counts["negative"]

        if pos >= args.min_positive and neg >= args.min_negative:
            print(f"✅ {rule_id}: {pos} positive, {neg} negative")
            filtered_rules.append(rule)
        else:
            print(f"❌ {rule_id}: {pos} positive, {neg} negative (insufficient)")

    # Save filtered rules
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for rule in filtered_rules:
            f.write(json.dumps(rule) + '\n')

    print(f"\n{'='*60}")
    print(f"Filtered {len(filtered_rules)}/{len(rules)} rules")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
