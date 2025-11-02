"""
Analyze learnability test results and identify learnable rules.

Reads all JSONL result files and creates:
- Complete summary with statistics
- List of learnable rules (>90% accuracy)
- Comparison of model performance
"""

import argparse
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from src.model_registry import DEFAULT_MULTI_MODEL_LIST
from src.utils import load_jsonl


class LearnabilityResult(BaseModel):
    """Result for a single test sample."""
    rule_id: str
    model: str
    few_shot_count: int
    test_input: str
    true_label: bool
    predicted_label: Optional[bool]
    raw_response: str
    correct: Optional[bool]
    parse_error: bool = False


def analyze_experiment(jsonl_path: Path) -> dict[str, Any]:
    """Analyze a single experiment JSONL file."""
    results_dicts = load_jsonl(jsonl_path)
    results = [LearnabilityResult(**r) for r in results_dicts]

    n_total = len(results)
    n_correct = sum(1 for r in results if r.correct)
    n_parseable = sum(1 for r in results if not r.parse_error)

    accuracy = n_correct / n_total if n_total > 0 else 0.0
    parse_rate = n_parseable / n_total if n_total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "n_correct": n_correct,
        "n_total": n_total,
        "parse_rate": round(parse_rate, 4),
        "n_parseable": n_parseable,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze learnability results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing JSONL result files"
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Output summary YAML file"
    )
    parser.add_argument(
        "--learnable-threshold",
        type=float,
        default=0.90,
        help="Accuracy threshold for learnable rules"
    )

    args = parser.parse_args()

    # Find all result files
    result_files = sorted(args.results_dir.glob("*.jsonl"))
    print(f"Found {len(result_files)} result files")

    # Parse filenames and organize results
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for result_file in result_files:
        # Parse filename: {rule_id}_{model}_{few_shot_count}shot.jsonl
        filename = result_file.stem
        parts = filename.rsplit("_", 1)

        if len(parts) != 2 or not parts[1].endswith("shot"):
            print(f"Warning: Skipping file with unexpected format: {filename}")
            continue

        few_shot_str = parts[1].replace("shot", "")
        try:
            few_shot_count = int(few_shot_str)
        except ValueError:
            print(f"Warning: Could not parse few-shot count from {filename}")
            continue

        # Everything before the last underscore contains rule_id and model
        # We need to find where the model name starts
        prefix = parts[0]

        # Known model names
        models = DEFAULT_MULTI_MODEL_LIST

        rule_id = None
        model = None

        for m in models:
            if prefix.endswith(f"_{m}"):
                model = m
                rule_id = prefix[:-len(m)-1]
                break

        if not rule_id or not model:
            print(f"Warning: Could not parse rule_id and model from {filename}")
            continue

        # Analyze experiment
        stats = analyze_experiment(result_file)

        # Store in summary
        if rule_id not in summary:
            summary[rule_id] = {}
        if model not in summary[rule_id]:
            summary[rule_id][model] = {}

        summary[rule_id][model][f"few_shot_{few_shot_count}"] = stats

    # Save complete summary
    output_file = args.output_summary or args.results_dir / "summary_complete.yaml"
    with output_file.open("w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Complete summary saved to {output_file}")

    # Analyze learnability
    print("\n" + "=" * 80)
    print("LEARNABILITY ANALYSIS")
    print("=" * 80)

    learnable_rules = {}

    for rule_id, model_results in sorted(summary.items()):
        print(f"\n{rule_id}:")

        rule_learnable = False
        best_accuracy = 0.0
        best_model = None
        best_few_shot = None

        for model, few_shot_results in sorted(model_results.items()):
            print(f"  {model}:")

            for few_shot_key, metrics in sorted(few_shot_results.items()):
                accuracy = metrics["accuracy"]
                parse_rate = metrics["parse_rate"]

                status = "✓ LEARNABLE" if accuracy >= args.learnable_threshold else "✗"

                print(
                    f"    {few_shot_key}: {accuracy:.1%} "
                    f"({metrics['n_correct']}/{metrics['n_total']}) "
                    f"[parse: {parse_rate:.1%}] {status}"
                )

                if accuracy >= args.learnable_threshold:
                    rule_learnable = True

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_few_shot = few_shot_key

        if rule_learnable:
            learnable_rules[rule_id] = {
                "best_accuracy": best_accuracy,
                "best_model": best_model,
                "best_few_shot": best_few_shot,
            }

    # Summary statistics
    print("\n" + "=" * 80)
    print(f"LEARNABLE RULES: {len(learnable_rules)}/{len(summary)}")
    print("=" * 80)

    for rule_id, info in sorted(learnable_rules.items()):
        print(
            f"✓ {rule_id}: {info['best_accuracy']:.1%} "
            f"({info['best_model']}, {info['best_few_shot']})"
        )

    # Save learnable rules list
    learnable_file = args.results_dir / "learnable_rules.yaml"
    with learnable_file.open("w") as f:
        yaml.dump(learnable_rules, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Learnable rules saved to {learnable_file}")


if __name__ == "__main__":
    main()
