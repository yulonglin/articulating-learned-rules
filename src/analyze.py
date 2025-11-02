"""
Analysis utilities for experiment results.

Provides functions to load, analyze, and visualize experiment outputs.
"""

from pathlib import Path
from typing import Any, Optional

from src.utils import compute_accuracy, extract_yes_no, load_jsonl


def analyze_classification_results(
    results_path: Path,
    label_key: str = "true_label",
    output_key: str = "output",
    normalize_fn: callable = extract_yes_no,
) -> dict[str, Any]:
    """
    Analyze classification experiment results.

    Args:
        results_path: Path to results.jsonl file
        label_key: Key in metadata containing true label
        output_key: Key containing model output
        normalize_fn: Function to normalize outputs before comparison

    Returns:
        Dictionary with analysis metrics
    """
    # Load results
    results = load_jsonl(results_path)

    if not results:
        return {"error": "No results found"}

    # Extract predictions and labels
    predictions = [normalize_fn(r[output_key]) for r in results]
    labels = [normalize_fn(r["metadata"][label_key]) for r in results]

    # Compute accuracy
    accuracy = compute_accuracy(predictions, labels, normalize_fn=lambda x: x)

    # Count correct/incorrect
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    incorrect = len(predictions) - correct

    # Token statistics
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_tokens = total_prompt_tokens + total_completion_tokens

    # Caching statistics
    cached_count = sum(1 for r in results if r["cached"])
    cache_rate = cached_count / len(results) if results else 0

    # Model info
    models = set(r["model"] for r in results)

    analysis = {
        "total_samples": len(results),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "tokens": {
            "prompt": total_prompt_tokens,
            "completion": total_completion_tokens,
            "total": total_tokens,
            "avg_per_sample": total_tokens / len(results) if results else 0,
        },
        "caching": {
            "cached_count": cached_count,
            "uncached_count": len(results) - cached_count,
            "cache_rate": cache_rate,
        },
        "models": list(models),
    }

    return analysis


def print_analysis(analysis: dict[str, Any]) -> None:
    """
    Pretty print analysis results.

    Args:
        analysis: Analysis dictionary from analyze_classification_results
    """
    print("=" * 60)
    print("EXPERIMENT ANALYSIS")
    print("=" * 60)
    print()

    print(f"Total Samples: {analysis['total_samples']}")
    print(f"Correct: {analysis['correct']}")
    print(f"Incorrect: {analysis['incorrect']}")
    print(f"Accuracy: {analysis['accuracy']:.2%}")
    print()

    print("Token Usage:")
    print(f"  Prompt tokens: {analysis['tokens']['prompt']:,}")
    print(f"  Completion tokens: {analysis['tokens']['completion']:,}")
    print(f"  Total tokens: {analysis['tokens']['total']:,}")
    print(f"  Avg per sample: {analysis['tokens']['avg_per_sample']:.1f}")
    print()

    print("Caching:")
    print(f"  Cached: {analysis['caching']['cached_count']}")
    print(f"  Uncached: {analysis['caching']['uncached_count']}")
    print(f"  Cache rate: {analysis['caching']['cache_rate']:.1%}")
    print()

    print(f"Models: {', '.join(analysis['models'])}")
    print()


def get_failed_samples(
    results_path: Path,
    label_key: str = "true_label",
    output_key: str = "output",
    normalize_fn: callable = extract_yes_no,
) -> list[dict[str, Any]]:
    """
    Get list of samples where prediction didn't match label.

    Args:
        results_path: Path to results.jsonl file
        label_key: Key in metadata containing true label
        output_key: Key containing model output
        normalize_fn: Function to normalize outputs

    Returns:
        List of failed sample dictionaries
    """
    results = load_jsonl(results_path)

    failed = []
    for r in results:
        prediction = normalize_fn(r[output_key])
        label = normalize_fn(r["metadata"][label_key])

        if prediction != label:
            failed.append(
                {
                    "sample_id": r["sample_id"],
                    "input": r["input"],
                    "prediction": prediction,
                    "label": label,
                    "raw_output": r[output_key],
                    "metadata": r["metadata"],
                }
            )

    return failed


def print_failed_samples(failed_samples: list[dict[str, Any]]) -> None:
    """
    Pretty print failed samples.

    Args:
        failed_samples: List from get_failed_samples
    """
    if not failed_samples:
        print("No failed samples!")
        return

    print("=" * 60)
    print(f"FAILED SAMPLES ({len(failed_samples)})")
    print("=" * 60)
    print()

    for sample in failed_samples:
        print(f"Sample ID: {sample['sample_id']}")
        print(f"Input: {sample['input']}")
        print(f"Prediction: {sample['prediction']}")
        print(f"True Label: {sample['label']}")
        print(f"Raw Output: {sample['raw_output']}")
        print()


def compare_experiments(
    experiment_paths: list[Path],
    names: Optional[list[str]] = None,
) -> dict[str, dict[str, Any]]:
    """
    Compare multiple experiment results.

    Args:
        experiment_paths: List of paths to results.jsonl files
        names: Optional names for experiments (defaults to filenames)

    Returns:
        Dictionary mapping experiment names to analysis results
    """
    if names is None:
        names = [p.parent.name for p in experiment_paths]

    comparisons = {}
    for name, path in zip(names, experiment_paths):
        comparisons[name] = analyze_classification_results(path)

    return comparisons


def print_comparison(comparisons: dict[str, dict[str, Any]]) -> None:
    """
    Pretty print experiment comparison.

    Args:
        comparisons: Dictionary from compare_experiments
    """
    print("=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    print()

    # Print header
    exp_names = list(comparisons.keys())
    print(f"{'Metric':<30}", end="")
    for name in exp_names:
        print(f"{name:>20}", end="")
    print()
    print("-" * (30 + 20 * len(exp_names)))

    # Accuracy
    print(f"{'Accuracy':<30}", end="")
    for name in exp_names:
        acc = comparisons[name]["accuracy"]
        print(f"{acc:>19.1%}", end="")
    print()

    # Samples
    print(f"{'Total Samples':<30}", end="")
    for name in exp_names:
        samples = comparisons[name]["total_samples"]
        print(f"{samples:>20,}", end="")
    print()

    # Tokens
    print(f"{'Total Tokens':<30}", end="")
    for name in exp_names:
        tokens = comparisons[name]["tokens"]["total"]
        print(f"{tokens:>20,}", end="")
    print()

    # Cache rate
    print(f"{'Cache Rate':<30}", end="")
    for name in exp_names:
        rate = comparisons[name]["caching"]["cache_rate"]
        print(f"{rate:>19.1%}", end="")
    print()


# Example usage
def main():
    """Example usage of analysis functions."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.analyze <results.jsonl>")
        print("\nExample:")
        print("  python -m src.analyze experiments/20251026_test/results.jsonl")
        sys.exit(1)

    results_path = Path(sys.argv[1])

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    # Run analysis
    analysis = analyze_classification_results(results_path)
    print_analysis(analysis)

    # Show failed samples
    failed = get_failed_samples(results_path)
    if failed:
        print()
        print_failed_samples(failed)


if __name__ == "__main__":
    main()
