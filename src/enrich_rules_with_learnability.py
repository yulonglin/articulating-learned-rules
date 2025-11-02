"""
Enrich curated rules with learnability metadata.

Reads learnability test results and adds model-specific min_few_shot_required
data to each rule. Filters to only include rules that are learnable by at least
one model (≥90% accuracy).

Input:
- curated_rules_generated.jsonl: Original curated rules
- summary_complete.yaml: Learnability results from analyze_learnability.py

Output:
- curated_rules_learnable.jsonl: Enriched rules with learnability metadata

The output adds a `learnability` field to each rule:
{
  "rule_id": "...",
  ...existing fields...,
  "learnability": {
    "gpt-4.1-nano-2025-04-14": {
      "min_few_shot_required": 10,
      "best_accuracy": 0.95
    },
    "claude-haiku-4-5-20251001": {
      "min_few_shot_required": 5,
      "best_accuracy": 0.92
    }
  }
}
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils import load_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_few_shot_count(few_shot_key: str) -> int:
    """Extract numeric few-shot count from key like 'few_shot_10'."""
    return int(few_shot_key.replace("few_shot_", ""))


def find_min_few_shot(
    model_results: dict[str, dict[str, Any]],
    threshold: float,
    logger: logging.Logger,
) -> tuple[Optional[int], Optional[float]]:
    """
    Find minimum few-shot count where accuracy >= threshold.

    Args:
        model_results: Dict mapping few_shot_X keys to metrics
        threshold: Accuracy threshold (e.g., 0.90)
        logger: Logger instance for warnings

    Returns:
        Tuple of (min_few_shot_required, best_accuracy) or (None, None)
    """
    # Sort by few-shot count
    sorted_results = sorted(
        model_results.items(),
        key=lambda x: extract_few_shot_count(x[0])
    )

    min_few_shot = None
    best_accuracy = 0.0

    for few_shot_key, metrics in sorted_results:
        accuracy = metrics.get("accuracy")
        if accuracy is None:
            logger.warning(
                f"Missing 'accuracy' in metrics for {few_shot_key}, skipping"
            )
            continue

        few_shot_count = extract_few_shot_count(few_shot_key)

        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Find minimum few-shot where threshold is met
        if accuracy >= threshold and min_few_shot is None:
            min_few_shot = few_shot_count

    if min_few_shot is None:
        return None, None

    return min_few_shot, best_accuracy


def enrich_rules(
    rules_path: Path,
    learnability_summary_path: Path,
    output_path: Path,
    threshold: float = 0.90,
) -> None:
    """
    Enrich rules with learnability metadata.

    Args:
        rules_path: Path to curated_rules_generated.jsonl
        learnability_summary_path: Path to summary_complete.yaml
        output_path: Path to output enriched rules
        threshold: Accuracy threshold for learnability
    """
    logger.info("=" * 80)
    logger.info("Enriching rules with learnability metadata")
    logger.info("=" * 80)
    logger.info(f"Rules file: {rules_path}")
    logger.info(f"Learnability summary: {learnability_summary_path}")
    logger.info(f"Threshold: {threshold:.1%}")
    logger.info(f"Output: {output_path}")

    # Load rules
    rules = load_jsonl(rules_path)
    logger.info(f"Loaded {len(rules)} rules")

    # Load learnability summary
    with learnability_summary_path.open("r") as f:
        learnability_summary = yaml.safe_load(f)
    logger.info(f"Loaded learnability data for {len(learnability_summary)} rules")

    # Enrich rules
    enriched_rules = []
    skipped_rules = []

    for rule in rules:
        rule_id = rule["rule_id"]

        if rule_id not in learnability_summary:
            skipped_rules.append(rule_id)
            logger.debug(f"Skipping {rule_id}: No learnability data")
            continue

        # Extract learnability metadata for this rule
        learnability = {}
        rule_data = learnability_summary[rule_id]

        for model, model_results in rule_data.items():
            min_few_shot, best_accuracy = find_min_few_shot(
                model_results, threshold, logger
            )

            if min_few_shot is not None:
                learnability[model] = {
                    "min_few_shot_required": min_few_shot,
                    "best_accuracy": round(best_accuracy, 4),
                }

        # Only include rules learnable by at least one model
        if learnability:
            rule["learnability"] = learnability
            enriched_rules.append(rule)
            logger.debug(
                f"✓ {rule_id}: learnable by {len(learnability)} model(s)"
            )
        else:
            skipped_rules.append(rule_id)
            logger.debug(f"Skipping {rule_id}: Not learnable by any model")

    # Save enriched rules
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for rule in enriched_rules:
            f.write(json.dumps(rule) + "\n")

    # Summary
    logger.info("=" * 80)
    logger.info(f"Enriched {len(enriched_rules)} rules")
    logger.info(f"Skipped {len(skipped_rules)} rules (not learnable or no data)")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 80)

    # Print sample
    if enriched_rules:
        logger.info("\nSample enriched rule:")
        sample = enriched_rules[0]
        logger.info(f"  Rule ID: {sample['rule_id']}")
        logger.info(f"  Learnability: {json.dumps(sample['learnability'], indent=4)}")

    # Print skipped rules if any
    if skipped_rules and len(skipped_rules) <= 10:
        logger.info(f"\nSkipped rules: {', '.join(skipped_rules)}")
    elif skipped_rules:
        logger.info(f"\nSkipped {len(skipped_rules)} rules (first 5): {', '.join(skipped_rules[:5])}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enrich curated rules with learnability metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--rules-file",
        type=Path,
        required=True,
        help="Path to curated_rules_generated.jsonl",
    )
    parser.add_argument(
        "--learnability-summary",
        type=Path,
        required=True,
        help="Path to summary_complete.yaml from analyze_learnability.py",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output enriched rules file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Accuracy threshold for learnability",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level
    logger.setLevel(getattr(logging, args.log_level.upper()))

    # Run enrichment
    enrich_rules(
        rules_path=args.rules_file,
        learnability_summary_path=args.learnability_summary,
        output_path=args.output_file,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
