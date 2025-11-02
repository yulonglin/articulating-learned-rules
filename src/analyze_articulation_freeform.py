"""
Analyze free-form articulation test results.

Reads all JSONL result files and creates:
- Complete summary with LLM judge + functional test metrics
- Model comparison (gpt-4.1-nano vs claude-haiku-4-5)
- Prompt variation comparison (simple vs cot vs explicit)
- Rule-level analysis
"""

import argparse
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from src.utils import load_jsonl


class ArticulationResult(BaseModel):
    """Result for a single articulation test."""
    rule_id: str
    model: str
    few_shot_count: int
    prompt_variation: str
    ground_truth_articulation: str
    generated_articulation: str
    raw_response: str
    keyword_match_score: Optional[float] = None
    llm_judge_score: Optional[float]
    llm_judge_reasoning: Optional[str] = None
    functional_test_accuracy: float
    functional_test_details: dict[str, Any]
    parse_error: bool = False


def analyze_experiment(jsonl_path: Path) -> dict[str, Any]:
    """Analyze a single articulation experiment."""
    results_dicts = load_jsonl(jsonl_path)

    if not results_dicts:
        return None

    # Load single result (each file has one articulation test)
    result = ArticulationResult(**results_dicts[0])

    return {
        "rule_id": result.rule_id,
        "model": result.model,
        "prompt_variation": result.prompt_variation,
        "llm_judge": result.llm_judge_score,
        "functional_accuracy": result.functional_test_accuracy,
        "n_classified": result.functional_test_details.get("n_classified", 0),
        "n_correct": result.functional_test_details.get("n_correct", 0),
        "n_total": result.functional_test_details.get("n_total", 0),
        "parse_error": result.parse_error,
    }


def aggregate_results(results_dir: Path) -> dict[str, Any]:
    """Aggregate all articulation results."""
    results_by_rule = {}
    results_by_model = {}
    results_by_prompt = {}
    all_results = []

    # Process all JSONL files
    for jsonl_file in sorted(results_dir.glob("*_freeform.jsonl")):
        result = analyze_experiment(jsonl_file)
        if result is None:
            continue

        rule_id = result["rule_id"]
        model = result["model"]
        prompt_var = result["prompt_variation"]

        # Store in all_results
        all_results.append(result)

        # Aggregate by rule
        if rule_id not in results_by_rule:
            results_by_rule[rule_id] = {}
        if model not in results_by_rule[rule_id]:
            results_by_rule[rule_id][model] = {}
        results_by_rule[rule_id][model][prompt_var] = {
            "llm_judge": result["llm_judge"],
            "functional_accuracy": result["functional_accuracy"],
        }

        # Aggregate by model
        if model not in results_by_model:
            results_by_model[model] = {
                "llm_judge_scores": [],
                "functional_scores": [],
            }
        results_by_model[model]["llm_judge_scores"].append(result["llm_judge"])
        results_by_model[model]["functional_scores"].append(result["functional_accuracy"])

        # Aggregate by prompt
        if prompt_var not in results_by_prompt:
            results_by_prompt[prompt_var] = {
                "llm_judge_scores": [],
                "functional_scores": [],
            }
        results_by_prompt[prompt_var]["llm_judge_scores"].append(result["llm_judge"])
        results_by_prompt[prompt_var]["functional_scores"].append(result["functional_accuracy"])

    # Compute averages
    for model in results_by_model:
        results_by_model[model]["avg_llm_judge"] = round(
            sum(results_by_model[model]["llm_judge_scores"]) / len(results_by_model[model]["llm_judge_scores"]), 3
        )
        results_by_model[model]["avg_functional"] = round(
            sum(results_by_model[model]["functional_scores"]) / len(results_by_model[model]["functional_scores"]), 3
        )
        del results_by_model[model]["llm_judge_scores"]
        del results_by_model[model]["functional_scores"]

    for prompt_var in results_by_prompt:
        results_by_prompt[prompt_var]["avg_llm_judge"] = round(
            sum(results_by_prompt[prompt_var]["llm_judge_scores"]) / len(results_by_prompt[prompt_var]["llm_judge_scores"]), 3
        )
        results_by_prompt[prompt_var]["avg_functional"] = round(
            sum(results_by_prompt[prompt_var]["functional_scores"]) / len(results_by_prompt[prompt_var]["functional_scores"]), 3
        )
        del results_by_prompt[prompt_var]["llm_judge_scores"]
        del results_by_prompt[prompt_var]["functional_scores"]

    return {
        "n_experiments": len(all_results),
        "n_rules": len(results_by_rule),
        "by_rule": results_by_rule,
        "by_model": results_by_model,
        "by_prompt": results_by_prompt,
    }


def identify_best_articulations(results_dir: Path, threshold: float = 0.9) -> dict[str, Any]:
    """Identify rules with successful articulations (high functional + LLM judge)."""
    best_articulations = {}

    for jsonl_file in sorted(results_dir.glob("*_freeform.jsonl")):
        result = analyze_experiment(jsonl_file)
        if result is None:
            continue

        rule_id = result["rule_id"]
        model = result["model"]
        prompt_var = result["prompt_variation"]

        # Success criteria: high functional accuracy AND high LLM judge score
        if result["functional_accuracy"] >= threshold and result["llm_judge"] >= threshold:
            if rule_id not in best_articulations:
                best_articulations[rule_id] = []

            best_articulations[rule_id].append({
                "model": model,
                "prompt_variation": prompt_var,
                "llm_judge": result["llm_judge"],
                "functional_accuracy": result["functional_accuracy"],
            })

    return best_articulations


def main():
    parser = argparse.ArgumentParser(description="Analyze free-form articulation results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/articulation_freeform"),
        help="Directory containing JSONL result files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("experiments/articulation_freeform/summary.yaml"),
        help="Output summary file",
    )
    parser.add_argument(
        "--best-threshold",
        type=float,
        default=0.9,
        help="Threshold for identifying best articulations",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Analyzing Free-Form Articulation Results")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print()

    # Aggregate all results
    summary = aggregate_results(args.results_dir)

    print(f"Total experiments: {summary['n_experiments']}")
    print(f"Total rules tested: {summary['n_rules']}")
    print()

    # Model comparison
    print("MODEL COMPARISON:")
    print("-" * 80)
    for model, metrics in summary["by_model"].items():
        print(f"\n{model}:")
        print(f"  Avg LLM judge:        {metrics['avg_llm_judge']:.1%}")
        print(f"  Avg functional:       {metrics['avg_functional']:.1%}")
    print()

    # Prompt comparison
    print("PROMPT VARIATION COMPARISON:")
    print("-" * 80)
    for prompt_var, metrics in summary["by_prompt"].items():
        print(f"\n{prompt_var}:")
        print(f"  Avg LLM judge:        {metrics['avg_llm_judge']:.1%}")
        print(f"  Avg functional:       {metrics['avg_functional']:.1%}")
    print()

    # Identify best articulations
    best = identify_best_articulations(args.results_dir, threshold=args.best_threshold)

    print(f"SUCCESSFUL ARTICULATIONS (≥{args.best_threshold:.0%} functional + LLM judge):")
    print("-" * 80)
    if best:
        for rule_id, configs in best.items():
            print(f"\n{rule_id}: {len(configs)} successful configs")
            for config in configs:
                print(f"  - {config['model']}, {config['prompt_variation']}: "
                      f"LLM={config['llm_judge']:.0%}, Func={config['functional_accuracy']:.0%}")
    else:
        print("No articulations met both thresholds.")
    print()

    # Save summary
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"Summary saved to: {args.output_file}")

    # Save best articulations
    best_file = args.output_file.parent / "best_articulations.yaml"
    with open(best_file, "w") as f:
        yaml.dump(best, f, default_flow_style=False, sort_keys=False)

    print(f"Best articulations saved to: {best_file}")


if __name__ == "__main__":
    main()
