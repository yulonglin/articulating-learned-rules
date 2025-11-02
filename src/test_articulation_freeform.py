"""
Test articulation via free-form generation (Step 2: Articulation - Free-form).

This script tests whether LLMs can articulate classification rules they've learned
by generating natural language descriptions. It evaluates:
- Ability to articulate rules in free-form natural language
- Effect of different prompt variations (simple, CoT, explicit)
- Correctness using multiple evaluation methods

Key requirements from RESEARCH_SPEC.md Step 2:
- Show few-shot examples to establish the pattern
- Ask for natural language articulation of the rule
- Evaluate correctness via: keyword overlap, ROUGE-L, LLM-as-judge, and functional testing
- Test multiple prompt variations systematically

Output Format:
- Per-test JSONL: {rule_id}_{model}_{variation}_freeform.jsonl
- Summary YAML: summary_freeform.yaml with accuracy metrics
"""

import argparse
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from rouge_score import rouge_scorer
from tqdm.asyncio import tqdm as async_tqdm

from src.api_caller import CacheMode, Message, create_caller
from src.model_registry import DEFAULT_JUDGE_MODEL, DEFAULT_MULTI_MODEL_LIST
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


class FreeFormArticulationResult(BaseModel):
    """Result for a single free-form articulation test."""
    rule_id: str
    model: str
    few_shot_count: int
    prompt_variation: str
    ground_truth_articulation: str
    generated_articulation: str
    raw_response: str

    # Evaluation scores
    keyword_match_score: Optional[float] = None
    rouge_l_score: Optional[float] = None
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None
    functional_test_accuracy: Optional[float] = None
    functional_test_details: Optional[dict[str, Any]] = None

    parse_error: bool = False


@dataclass
class FreeFormArticulationConfig:
    """Configuration for free-form articulation test."""
    rules_file: Path
    datasets_dir: Path
    few_shot_count: int
    prompt_variations: list[str]
    models: list[str]
    evaluation_methods: list[str]  # "keyword", "rouge", "llm_judge", "functional"
    judge_model: str  # Model to use for LLM-as-judge
    functional_test_size: int  # Number of samples for functional testing
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


def build_freeform_prompt(
    few_shot_examples: list[DatasetSample],
    variation: str,
) -> str:
    """
    Build free-form articulation prompt with specified variation.

    Args:
        few_shot_examples: List of few-shot examples
        variation: Prompt variation ("simple", "cot", "explicit")

    Returns:
        Formatted prompt string
    """
    # Build examples section
    examples_parts = ["Examples:"]
    for example in few_shot_examples:
        examples_parts.append(f'Input: "{example.input}" → {str(example.label)}')
    examples_text = "\n".join(examples_parts)

    if variation == "simple":
        prompt = f"""Here are examples of a classification task:
{examples_text}

In 1-2 sentences, describe the rule that determines when the output is True vs False.

Rule:"""

    elif variation == "cot":
        prompt = f"""Here are examples of a classification task:
{examples_text}

Think step-by-step about what pattern distinguishes True from False cases.
Then write the rule in 1-2 sentences.

Thinking:"""

    elif variation == "explicit":
        prompt = f"""Classify texts as True or False based on these examples:
{examples_text}

What is the classification rule? Describe it precisely and concisely.

Rule:"""

    else:
        raise ValueError(f"Unknown prompt variation: {variation}")

    return prompt


# ============================================================================
# Response Parsing
# ============================================================================


def extract_rule_from_response(response: str, variation: str) -> str:
    """
    Extract articulated rule from model response.

    Args:
        response: Model response text
        variation: Prompt variation used

    Returns:
        Extracted rule text
    """
    response_clean = response.strip()

    if variation == "cot":
        # For CoT, extract text after "Rule:" if present
        if "Rule:" in response_clean:
            parts = response_clean.split("Rule:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        # Otherwise return the last paragraph
        paragraphs = [p.strip() for p in response_clean.split("\n\n") if p.strip()]
        if paragraphs:
            return paragraphs[-1]

    # For other variations, return the full response
    return response_clean


def evaluate_keyword_match(
    generated: str,
    ground_truth: str,
) -> float:
    """
    Evaluate using keyword matching.

    Extracts key concepts from ground truth and checks if they appear in generated.

    Args:
        generated: Generated articulation
        ground_truth: Ground truth articulation

    Returns:
        Score: proportion of keywords found (0.0 to 1.0)
    """
    # Extract potential keywords from ground truth
    # Remove common words and extract meaningful terms
    stopwords = {
        "the", "is", "if", "and", "or", "a", "an", "as", "in", "on", "at",
        "to", "for", "of", "with", "from", "by", "true", "false", "labeled",
        "input", "output", "text", "it", "are", "be", "all", "any"
    }

    # Simple tokenization and filtering
    gt_words = re.findall(r'\b\w+\b', ground_truth.lower())
    keywords = [w for w in gt_words if w not in stopwords and len(w) > 2]

    if not keywords:
        return 0.0

    # Check how many keywords appear in generated
    gen_lower = generated.lower()
    found_keywords = sum(1 for kw in keywords if kw in gen_lower)

    return found_keywords / len(keywords)


def evaluate_rouge_l(
    generated: str,
    ground_truth: str,
) -> float:
    """
    Evaluate using ROUGE-L (Longest Common Subsequence).

    ROUGE-L measures the longest common subsequence between generated and ground truth,
    capturing both content overlap and word order.

    Args:
        generated: Generated articulation
        ground_truth: Ground truth articulation

    Returns:
        Score: ROUGE-L F1 score (0.0 to 1.0)
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, generated)

    # Return F1 score (harmonic mean of precision and recall)
    return scores['rougeL'].fmeasure


async def evaluate_llm_judge(
    generated: str,
    ground_truth: str,
    judge_model: str,
    config: FreeFormArticulationConfig,
) -> tuple[float, str]:
    """
    Evaluate using LLM-as-judge.

    Args:
        generated: Generated articulation
        ground_truth: Ground truth articulation
        judge_model: Model to use as judge
        config: Configuration

    Returns:
        Tuple of (score, reasoning)
    """
    # Create judge prompt
    judge_prompt = f"""You are evaluating whether two rule descriptions are equivalent.

Ground Truth Rule:
{ground_truth}

Generated Rule:
{generated}

Do these two rules describe the same classification logic? Consider:
1. Do they identify the same key features or patterns?
2. Would they produce the same classifications on most inputs?
3. Are the core concepts equivalent, even if phrasing differs?

Provide your evaluation in this format:
Score: [0-10, where 10 = perfectly equivalent, 0 = completely different]
Reasoning: [Brief explanation of your score]

Evaluation:"""

    # Call judge model
    caller = create_caller(
        model=judge_model,
        temperature=0.0,
        max_tokens=500,
        cache_mode=config.cache_mode,
        cache_dir=config.cache_dir,
        max_concurrent=config.max_concurrent,
    )

    messages = [Message(role="user", content=judge_prompt)]
    response = await caller.call(messages)

    # Parse score and reasoning
    response_text = response.content
    score = 0.0
    reasoning = response_text

    # Try to extract score
    score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
    if score_match:
        score_raw = float(score_match.group(1))
        score = score_raw / 10.0  # Normalize to 0-1

    # Try to extract reasoning
    reasoning_match = re.search(r'Reasoning:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    return score, reasoning


async def evaluate_functional(
    generated_articulation: str,
    rule: Rule,
    test_samples: list[DatasetSample],
    model: str,
    config: FreeFormArticulationConfig,
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate using functional testing.

    Uses the generated articulation to classify held-out examples
    and measures accuracy against ground truth labels.

    Args:
        generated_articulation: Generated rule articulation
        rule: Original rule (for reference)
        test_samples: Held-out test samples
        model: Model to use for classification
        config: Configuration

    Returns:
        Tuple of (accuracy, details_dict)
    """
    # Create classification prompt using generated rule
    caller = create_caller(
        model=model,
        temperature=0.0,
        max_tokens=100,
        cache_mode=config.cache_mode,
        cache_dir=config.cache_dir,
        max_concurrent=config.max_concurrent,
    )

    predictions = []
    true_labels = []

    for sample in test_samples:
        classify_prompt = f"""Rule: {generated_articulation}

Based on this rule, classify the following input as True or False.

Input: "{sample.input}"

Answer with just True or False:"""

        messages = [Message(role="user", content=classify_prompt)]
        response = await caller.call(messages)

        # Parse response
        response_clean = response.content.strip().lower()
        if "true" in response_clean and "false" not in response_clean:
            predicted = True
        elif "false" in response_clean and "true" not in response_clean:
            predicted = False
        else:
            # Ambiguous, skip this sample
            continue

        predictions.append(predicted)
        true_labels.append(sample.label)

    # Calculate accuracy
    if not predictions:
        return 0.0, {
            "n_total": len(test_samples),
            "n_classified": 0,
            "n_correct": 0,
            "n_skipped": len(test_samples),
            "accuracy": 0.0,
        }

    n_correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    accuracy = n_correct / len(predictions)

    details = {
        "n_total": len(test_samples),
        "n_classified": len(predictions),
        "n_correct": n_correct,
        "n_skipped": len(test_samples) - len(predictions),
        "accuracy": accuracy,
    }

    return accuracy, details


# ============================================================================
# Data Splitting
# ============================================================================


def stratified_split(
    samples: list[DatasetSample],
    few_shot_count: int,
    test_count: int,
    random_seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    """
    Get stratified few-shot examples and test set (balanced positive/negative).

    Args:
        samples: All dataset samples
        few_shot_count: Number of few-shot examples
        test_count: Number of test samples
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

    test_positive = test_count // 2
    test_negative = test_count - test_positive

    # Check we have enough samples
    total_positive_needed = few_shot_positive + test_positive
    total_negative_needed = few_shot_negative + test_negative

    if len(positive_samples) < total_positive_needed:
        raise ValueError(
            f"Not enough positive samples: need {total_positive_needed}, "
            f"have {len(positive_samples)}"
        )
    if len(negative_samples) < total_negative_needed:
        raise ValueError(
            f"Not enough negative samples: need {total_negative_needed}, "
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


async def evaluate_rule_freeform(
    rule: Rule,
    model: str,
    variation: str,
    config: FreeFormArticulationConfig,
    logger: logging.Logger,
) -> FreeFormArticulationResult:
    """
    Evaluate free-form articulation for a single rule with given prompt variation.

    Args:
        rule: Rule to evaluate
        model: Model name
        variation: Prompt variation to use
        config: Configuration
        logger: Logger instance

    Returns:
        FreeFormArticulationResult object
    """
    logger.info(
        f"Evaluating {rule.rule_id} with {model} using {variation} prompt"
    )

    # Load dataset
    dataset_path = config.datasets_dir / f"{rule.rule_id}.jsonl"
    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        return None

    dataset_dicts = load_jsonl(dataset_path)
    dataset_samples = [DatasetSample(**d) for d in dataset_dicts]

    logger.info(f"Loaded {len(dataset_samples)} samples from {dataset_path.name}")

    # Split into few-shot and test
    try:
        few_shot_samples, test_samples = stratified_split(
            samples=dataset_samples,
            few_shot_count=config.few_shot_count,
            test_count=config.functional_test_size,
            random_seed=config.random_seed,
        )
    except ValueError as e:
        logger.error(f"Failed to split dataset: {e}")
        return None

    # Build prompt
    prompt = build_freeform_prompt(few_shot_samples, variation)
    messages = [Message(role="user", content=prompt)]

    # Create API caller
    caller = create_caller(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        cache_mode=config.cache_mode,
        cache_dir=config.cache_dir,
        max_concurrent=config.max_concurrent,
    )

    # Call API
    response = await caller.call(messages)

    # Extract rule
    generated_articulation = extract_rule_from_response(response.content, variation)

    # Create result object
    result = FreeFormArticulationResult(
        rule_id=rule.rule_id,
        model=model,
        few_shot_count=config.few_shot_count,
        prompt_variation=variation,
        ground_truth_articulation=rule.articulation,
        generated_articulation=generated_articulation,
        raw_response=response.content,
    )

    # Run evaluations
    if "keyword" in config.evaluation_methods:
        result.keyword_match_score = evaluate_keyword_match(
            generated_articulation, rule.articulation
        )
        logger.info(f"  Keyword match: {result.keyword_match_score:.2f}")

    if "rouge" in config.evaluation_methods:
        result.rouge_l_score = evaluate_rouge_l(
            generated_articulation, rule.articulation
        )
        logger.info(f"  ROUGE-L: {result.rouge_l_score:.2f}")

    if "llm_judge" in config.evaluation_methods:
        score, reasoning = await evaluate_llm_judge(
            generated_articulation,
            rule.articulation,
            config.judge_model,
            config,
        )
        result.llm_judge_score = score
        result.llm_judge_reasoning = reasoning
        logger.info(f"  LLM judge: {result.llm_judge_score:.2f}")

    if "functional" in config.evaluation_methods:
        accuracy, details = await evaluate_functional(
            generated_articulation,
            rule,
            test_samples,
            model,
            config,
        )
        result.functional_test_accuracy = accuracy
        result.functional_test_details = details
        logger.info(
            f"  Functional test: {result.functional_test_accuracy:.2%} "
            f"({details['n_correct']}/{details['n_classified']})"
        )

    return result


# ============================================================================
# Main Runner
# ============================================================================


async def run_freeform_articulation_tests(config: FreeFormArticulationConfig) -> None:
    """
    Run free-form articulation tests for all rules, models, and prompt variations.

    Args:
        config: Configuration for free-form articulation tests
    """
    # Setup logging
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = config.output_dir / "articulation_freeform.log"
    logger = logging.getLogger("articulation_freeform")
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
    logger.info("Starting Free-Form Articulation Tests (Step 2)")
    logger.info("=" * 80)
    logger.info(f"Rules file: {config.rules_file}")
    logger.info(f"Datasets directory: {config.datasets_dir}")
    logger.info(f"Few-shot count: {config.few_shot_count}")
    logger.info(f"Prompt variations: {config.prompt_variations}")
    logger.info(f"Evaluation methods: {config.evaluation_methods}")
    logger.info(f"Models: {config.models}")
    logger.info(f"Judge model: {config.judge_model}")
    logger.info(f"Random seed: {config.random_seed}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("=" * 80)

    # Load rules
    rules_dicts = load_jsonl(config.rules_file)
    rules = [Rule(**r) for r in rules_dicts]
    logger.info(f"Loaded {len(rules)} rules from {config.rules_file.name}")

    # Set random seed
    set_random_seed(config.random_seed)

    # Run evaluations
    summary: dict[str, dict[str, dict[str, Any]]] = {}

    for rule in rules:
        summary[rule.rule_id] = {}

        for model in config.models:
            summary[rule.rule_id][model] = {}

            for variation in config.prompt_variations:
                # Run evaluation
                result = await evaluate_rule_freeform(
                    rule=rule,
                    model=model,
                    variation=variation,
                    config=config,
                    logger=logger,
                )

                if result is None:
                    logger.warning(
                        f"No result for {rule.rule_id} | {model} | {variation}"
                    )
                    continue

                # Save result
                output_file = (
                    config.output_dir /
                    f"{rule.rule_id}_{model}_{variation}_freeform.jsonl"
                )
                with output_file.open("w") as f:
                    f.write(result.model_dump_json() + "\n")

                logger.info(f"Saved result to {output_file.name}")

                # Compute summary statistics
                metrics = {
                    "generated_articulation": result.generated_articulation,
                }

                if result.keyword_match_score is not None:
                    metrics["keyword_match"] = round(result.keyword_match_score, 4)
                if result.llm_judge_score is not None:
                    metrics["llm_judge"] = round(result.llm_judge_score, 4)
                if result.functional_test_accuracy is not None:
                    metrics["functional_accuracy"] = round(
                        result.functional_test_accuracy, 4
                    )

                summary[rule.rule_id][model][variation] = metrics

    # Save summary
    summary_file = config.output_dir / "summary_freeform.yaml"
    with summary_file.open("w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 80)
    logger.info(f"Summary saved to {summary_file}")
    logger.info("Free-form articulation tests complete!")
    logger.info("=" * 80)

    # Print summary table
    print("\n" + "=" * 80)
    print("FREE-FORM ARTICULATION TEST SUMMARY")
    print("=" * 80 + "\n")

    for rule_id, model_results in summary.items():
        print(f"\n{rule_id}:")
        for model, variation_results in model_results.items():
            print(f"  {model}:")
            for variation, metrics in variation_results.items():
                print(f"    {variation}:")
                for metric_name, metric_value in metrics.items():
                    if metric_name != "generated_articulation":
                        print(f"      {metric_name}: {metric_value}")

    print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for free-form articulation tests."""
    parser = argparse.ArgumentParser(
        description="Test articulation via free-form generation (Step 2)",
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
        default=Path("experiments/articulation_freeform"),
        help="Output directory for results",
    )

    # Experiment parameters
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=10,
        help="Number of few-shot examples to show",
    )
    parser.add_argument(
        "--prompt-variations",
        type=str,
        nargs="+",
        default=["simple", "cot", "explicit"],
        choices=["simple", "cot", "explicit"],
        help="Prompt variations to test",
    )
    parser.add_argument(
        "--evaluation-methods",
        type=str,
        nargs="+",
        default=["keyword", "rouge", "llm_judge", "functional"],
        choices=["keyword", "rouge", "llm_judge", "functional"],
        help="Evaluation methods to use (keyword=keyword matching, rouge=ROUGE-L F1, llm_judge=LLM-as-judge, functional=functional testing)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="Model to use for LLM-as-judge evaluation",
    )
    parser.add_argument(
        "--functional-test-size",
        type=int,
        default=20,
        help="Number of samples for functional testing",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MULTI_MODEL_LIST,
        help="Models to test",
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
        choices=["none", "short", "persistent"],
        help="Cache mode: none, short (15min), persistent",
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
        default=500,
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
    config = FreeFormArticulationConfig(
        rules_file=args.rules_file,
        datasets_dir=args.datasets_dir,
        few_shot_count=args.few_shot_count,
        prompt_variations=args.prompt_variations,
        models=args.models,
        evaluation_methods=args.evaluation_methods,
        judge_model=args.judge_model,
        functional_test_size=args.functional_test_size,
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
    asyncio.run(run_freeform_articulation_tests(config))


if __name__ == "__main__":
    main()
