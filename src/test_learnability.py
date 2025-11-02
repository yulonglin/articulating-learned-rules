"""
Test learnability of classification rules via few-shot learning (Step 1).

This script tests whether LLMs can learn classification rules from few-shot
examples WITHOUT chain-of-thought reasoning. It evaluates:
- Whether the model achieves >90% accuracy on held-out examples
- How performance varies with few-shot count (5, 10, 20 examples)
- How performance varies across different models

Key requirements from RESEARCH_SPEC.md Step 1:
- In-context learning (few-shot examples + instructions only)
- NO CoT reasoning - direct classification only
- Target: >90% accuracy on held-out in-distribution examples
- Simple classification rules that are easy to articulate

Output Format:
- Per-rule JSONL: {rule_id}_{model}_{few_shot_count}.jsonl
- Summary YAML: summary.yaml with accuracy metrics
"""

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from tqdm.asyncio import tqdm as async_tqdm

from src.api_caller import CacheMode, Message, create_caller
from src.model_registry import DEFAULT_MULTI_MODEL_LIST
from src.utils import load_jsonl, set_random_seed


# ============================================================================
# Data Models
# ============================================================================


class DatasetSample(BaseModel):
    """Single sample from a dataset."""
    input: str
    label: bool
    rule_id: str
    metadata: dict[str, Any] = {}


class Rule(BaseModel):
    """Classification rule definition."""
    rule_id: str
    rule_name: str
    articulation: str
    category: str
    examples: list[dict[str, Any]]
    expected_difficulty: str


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


@dataclass
class LearnabilityConfig:
    """Configuration for learnability test."""
    rules_file: Path
    datasets_dir: Path
    few_shot_counts: list[int]
    models: list[str]
    test_size: int
    output_dir: Path
    random_seed: int
    cache_mode: CacheMode
    cache_dir: Path
    max_concurrent: int
    temperature: float
    max_tokens: int
    log_level: str


# ============================================================================
# Prompt Building
# ============================================================================


def build_few_shot_prompt(
    few_shot_examples: list[DatasetSample],
    test_input: str,
) -> str:
    """
    Build few-shot classification prompt WITHOUT chain-of-thought.

    Args:
        few_shot_examples: List of few-shot examples
        test_input: Input to classify

    Returns:
        Formatted prompt string
    """
    prompt_parts = [
        "You will classify text inputs as True or False based on the examples below.",
        "Respond with ONLY 'True' or 'False' - no explanations, reasoning, or other text.",
        "",
        "Examples:",
    ]

    # Add few-shot examples
    for example in few_shot_examples:
        prompt_parts.append(f'Input: "{example.input}"')
        prompt_parts.append(f"Output: {str(example.label)}")
        prompt_parts.append("")

    # Add test input
    prompt_parts.append("Now classify this input. Return ONLY 'True' or 'False', and nothing else:")
    prompt_parts.append(f'Input: "{test_input}"')
    prompt_parts.append("Output:")

    return "\n".join(prompt_parts)


# ============================================================================
# Response Parsing
# ============================================================================


def parse_boolean_response(response: str) -> Optional[bool]:
    """
    Parse boolean classification response.

    Handles variations like: "True", "true", "TRUE", "False", "false", etc.

    Args:
        response: Model response text

    Returns:
        True/False if parseable, None if unparseable
    """
    response_clean = response.strip().lower()

    # Direct match
    if response_clean == "true":
        return True
    if response_clean == "false":
        return False

    # Check first word
    first_word = response_clean.split()[0] if response_clean.split() else ""
    if first_word == "true":
        return True
    if first_word == "false":
        return False

    # Check if contains true/false (but not both)
    has_true = "true" in response_clean
    has_false = "false" in response_clean

    if has_true and not has_false:
        return True
    if has_false and not has_true:
        return False

    # Unable to parse
    return None


# ============================================================================
# Data Splitting
# ============================================================================


def stratified_split(
    samples: list[DatasetSample],
    few_shot_count: int,
    test_size: int,
    random_seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    """
    Split dataset into few-shot and test sets with stratification.

    Ensures ~50/50 positive/negative balance in both sets.

    Args:
        samples: All dataset samples
        few_shot_count: Number of few-shot examples
        test_size: Number of test samples
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (few_shot_samples, test_samples)
    """
    # Separate by label
    positive_samples = [s for s in samples if s.label]
    negative_samples = [s for s in samples if not s.label]

    # Set random seed
    rng = random.Random(random_seed)

    # Shuffle
    rng.shuffle(positive_samples)
    rng.shuffle(negative_samples)

    # Calculate splits (aim for 50/50)
    few_shot_positive = few_shot_count // 2
    few_shot_negative = few_shot_count - few_shot_positive

    test_positive = test_size // 2
    test_negative = test_size - test_positive

    # Check we have enough samples
    if len(positive_samples) < few_shot_positive + test_positive:
        raise ValueError(
            f"Not enough positive samples: need {few_shot_positive + test_positive}, "
            f"have {len(positive_samples)}"
        )
    if len(negative_samples) < few_shot_negative + test_negative:
        raise ValueError(
            f"Not enough negative samples: need {few_shot_negative + test_negative}, "
            f"have {len(negative_samples)}"
        )

    # Split
    few_shot_samples = (
        positive_samples[:few_shot_positive] +
        negative_samples[:few_shot_negative]
    )

    test_samples = (
        positive_samples[few_shot_positive:few_shot_positive + test_positive] +
        negative_samples[few_shot_negative:few_shot_negative + test_negative]
    )

    # Shuffle to mix positive/negative
    rng.shuffle(few_shot_samples)
    rng.shuffle(test_samples)

    return few_shot_samples, test_samples


# ============================================================================
# Evaluation
# ============================================================================


async def evaluate_rule(
    rule: Rule,
    model: str,
    few_shot_count: int,
    config: LearnabilityConfig,
    logger: logging.Logger,
) -> list[LearnabilityResult]:
    """
    Evaluate learnability for a single rule with given few-shot count.

    Args:
        rule: Rule to evaluate
        model: Model name
        few_shot_count: Number of few-shot examples
        config: Configuration
        logger: Logger instance

    Returns:
        List of LearnabilityResult objects
    """
    start_time = time.time()
    logger.info(
        f"Evaluating {rule.rule_id} with {model} and {few_shot_count} few-shot examples"
    )

    # Load dataset
    dataset_path = config.datasets_dir / f"{rule.rule_id}.jsonl"
    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        return []

    dataset_dicts = load_jsonl(dataset_path)
    dataset_samples = [DatasetSample(**d) for d in dataset_dicts]

    logger.info(f"Loaded {len(dataset_samples)} samples from {dataset_path.name}")

    # Split into few-shot and test
    try:
        few_shot_samples, test_samples = stratified_split(
            samples=dataset_samples,
            few_shot_count=few_shot_count,
            test_size=config.test_size,
            random_seed=config.random_seed,
        )
    except ValueError as e:
        logger.error(f"Failed to split dataset for {rule.rule_id}: {e}")
        return []

    logger.info(
        f"Split: {len(few_shot_samples)} few-shot, {len(test_samples)} test "
        f"(pos: {sum(1 for s in test_samples if s.label)}, "
        f"neg: {sum(1 for s in test_samples if not s.label)})"
    )

    # Create API caller
    caller = create_caller(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        cache_mode=config.cache_mode,
        cache_dir=config.cache_dir,
        max_concurrent=config.max_concurrent,
    )

    # Run classification on test samples
    results = []

    for test_sample in async_tqdm(
        test_samples,
        desc=f"{rule.rule_id[:20]}... | {model} | {few_shot_count}-shot",
        leave=False,
    ):
        # Build prompt
        prompt = build_few_shot_prompt(few_shot_samples, test_sample.input)
        messages = [Message(role="user", content=prompt)]

        # Call API
        response = await caller.call(messages)

        # Parse response
        predicted_label = parse_boolean_response(response.content)
        parse_error = predicted_label is None

        # Determine correctness
        if predicted_label is not None:
            correct = predicted_label == test_sample.label
        else:
            correct = None

        # Create result
        result = LearnabilityResult(
            rule_id=rule.rule_id,
            model=model,
            few_shot_count=few_shot_count,
            test_input=test_sample.input,
            true_label=test_sample.label,
            predicted_label=predicted_label,
            raw_response=response.content,
            correct=correct,
            parse_error=parse_error,
        )

        results.append(result)

    # Log summary
    n_correct = sum(1 for r in results if r.correct)
    n_parseable = sum(1 for r in results if not r.parse_error)
    accuracy = n_correct / len(results) if results else 0.0
    parse_rate = n_parseable / len(results) if results else 0.0

    elapsed_time = time.time() - start_time

    logger.info(
        f"{rule.rule_id} | {model} | {few_shot_count}-shot: "
        f"accuracy={accuracy:.2%} ({n_correct}/{len(results)}), "
        f"parse_rate={parse_rate:.2%}, "
        f"time={elapsed_time:.1f}s"
    )

    return results


# ============================================================================
# Main Runner
# ============================================================================


async def run_learnability_tests(config: LearnabilityConfig) -> None:
    """
    Run learnability tests for all rules, models, and few-shot counts.

    Args:
        config: Configuration for learnability tests
    """
    # Setup logging
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = config.output_dir / "learnability.log"
    logger = logging.getLogger("learnability")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logger.info("=" * 80)
    logger.info("Starting Learnability Tests (Step 1)")
    logger.info("=" * 80)
    logger.info(f"Rules file: {config.rules_file}")
    logger.info(f"Datasets directory: {config.datasets_dir}")
    logger.info(f"Few-shot counts: {config.few_shot_counts}")
    logger.info(f"Models: {config.models}")
    logger.info(f"Test size: {config.test_size}")
    logger.info(f"Random seed: {config.random_seed}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("=" * 80)

    # Load rules
    rules_dicts = load_jsonl(config.rules_file)
    rules = [Rule(**r) for r in rules_dicts]
    logger.info(f"Loaded {len(rules)} rules from {config.rules_file.name}")

    # Set random seed
    set_random_seed(config.random_seed)

    # Run evaluations in parallel
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    # Initialize summary structure
    for rule in rules:
        summary[rule.rule_id] = {}
        for model in config.models:
            summary[rule.rule_id][model] = {}

    # Create all experiment tasks
    tasks = []
    task_metadata = []

    for rule in rules:
        for model in config.models:
            for few_shot_count in config.few_shot_counts:
                # Check if output already exists
                output_file = (
                    config.output_dir /
                    f"{rule.rule_id}_{model}_{few_shot_count}shot.jsonl"
                )
                if output_file.exists():
                    logger.info(f"Skipping {output_file.name} (already exists)")
                    continue

                task = evaluate_rule(
                    rule=rule,
                    model=model,
                    few_shot_count=few_shot_count,
                    config=config,
                    logger=logger,
                )
                tasks.append(task)
                task_metadata.append({
                    "rule": rule,
                    "model": model,
                    "few_shot_count": few_shot_count,
                })

    logger.info(f"Running {len(tasks)} experiments in parallel...")

    # Run all experiments in parallel
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for results, metadata in zip(all_results, task_metadata):
        rule = metadata["rule"]
        model = metadata["model"]
        few_shot_count = metadata["few_shot_count"]

        # Handle exceptions
        if isinstance(results, Exception):
            logger.error(
                f"Error in {rule.rule_id} | {model} | {few_shot_count}-shot: {results}"
            )
            continue

        if not results:
            logger.warning(
                f"No results for {rule.rule_id} | {model} | {few_shot_count}-shot"
            )
            continue

        # Save per-rule results
        output_file = (
            config.output_dir /
            f"{rule.rule_id}_{model}_{few_shot_count}shot.jsonl"
        )
        with output_file.open("w") as f:
            for result in results:
                f.write(result.model_dump_json() + "\n")

        logger.info(f"Saved results to {output_file.name}")

        # Compute summary statistics
        n_total = len(results)
        n_correct = sum(1 for r in results if r.correct)
        n_parseable = sum(1 for r in results if not r.parse_error)
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        parse_rate = n_parseable / n_total if n_total > 0 else 0.0

        summary[rule.rule_id][model][f"few_shot_{few_shot_count}"] = {
            "accuracy": round(accuracy, 4),
            "n_correct": n_correct,
            "n_total": n_total,
            "parse_rate": round(parse_rate, 4),
            "n_parseable": n_parseable,
        }

    # Save summary
    summary_file = config.output_dir / "summary.yaml"
    with summary_file.open("w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 80)
    logger.info(f"Summary saved to {summary_file}")
    logger.info("Learnability tests complete!")
    logger.info("=" * 80)

    # Print summary table
    print("\n" + "=" * 80)
    print("LEARNABILITY TEST SUMMARY")
    print("=" * 80 + "\n")

    for rule_id, model_results in summary.items():
        print(f"\n{rule_id}:")
        for model, few_shot_results in model_results.items():
            print(f"  {model}:")
            for few_shot_key, metrics in few_shot_results.items():
                accuracy = metrics["accuracy"]
                status = "✓" if accuracy >= 0.90 else "✗"
                print(
                    f"    {few_shot_key}: {accuracy:.1%} "
                    f"({metrics['n_correct']}/{metrics['n_total']}) {status}"
                )

    print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for learnability tests."""
    parser = argparse.ArgumentParser(
        description="Test learnability of classification rules (Step 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--rules-file",
        type=Path,
        required=True,
        help="JSONL file with curated rules",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("experiments/datasets"),
        help="Directory with generated datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/learnability"),
        help="Output directory for results",
    )

    # Experiment parameters
    parser.add_argument(
        "--few-shot-counts",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50, 100, 150],
        help="List of few-shot counts to test",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MULTI_MODEL_LIST,
        help="Models to test",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Number of test samples per rule",
    )

    # Reproducibility
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # API configuration
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="short",
        choices=["none", "short", "one_week", "two_weeks", "persistent"],
        help="Cache mode: none, short (15min), one_week (1 week), two_weeks (2 weeks), persistent",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Directory for cache files",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum completion tokens",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Create config
    config = LearnabilityConfig(
        rules_file=args.rules_file,
        datasets_dir=args.datasets_dir,
        few_shot_counts=args.few_shot_counts,
        models=args.models,
        test_size=args.test_size,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        cache_mode=CacheMode(args.cache_mode),
        cache_dir=args.cache_dir,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_level=args.log_level,
    )

    # Run tests
    asyncio.run(run_learnability_tests(config))


if __name__ == "__main__":
    main()
