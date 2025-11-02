"""
Test articulation via multiple-choice (Step 2: Articulation - MC).

This script tests whether LLMs can articulate classification rules they've learned
by selecting the correct rule from multiple choices. It evaluates:
- Ability to identify the correct rule among distractors
- Effect of distractor difficulty (same category vs different categories)
- Performance across different models

Key requirements from RESEARCH_SPEC.md Step 2:
- Show few-shot examples to establish the pattern
- Present actual rule + 3 distractors in randomized order
- Parse A/B/C/D selection
- Calculate articulation accuracy

Output Format:
- Per-test JSONL: {rule_id}_{model}_mc.jsonl
- Summary YAML: summary_mc.yaml with accuracy metrics
"""

import argparse
import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from tqdm.asyncio import tqdm as async_tqdm

from src.api_caller import CacheMode, Message, create_caller
from src.model_registry import DEFAULT_MULTI_MODEL_LIST
from src.data_models import DatasetSample, Rule
from src.utils import load_jsonl, set_random_seed
from src.embedding_cache import EmbeddingCache, cosine_similarity, batch_cosine_similarity


# ============================================================================
# Data Models
# ============================================================================


class MCArticulationResult(BaseModel):
    """Result for a single multiple-choice test."""
    rule_id: str
    model: str
    few_shot_count: int
    test_index: int
    correct_articulation: str
    distractors: list[str]
    option_order: list[int]  # Maps options to indices (correct=0, distractors=1,2,3)
    correct_option: str  # A, B, C, or D
    predicted_option: Optional[str]
    raw_response: str
    correct: Optional[bool]
    parse_error: bool = False


@dataclass
class MCArticulationConfig:
    """Configuration for MC articulation test."""
    rules_file: Path
    datasets_dir: Path
    few_shot_count: int
    num_test_cases: int
    models: list[str]
    output_dir: Path
    random_seed: int
    cache_mode: CacheMode
    cache_dir: Path
    max_concurrent: int
    temperature: float
    max_tokens: int
    log_level: str
    distractor_strategy: str  # "same_category", "different_category", "mixed", "similarity_filtered"
    distractor_pool_file: Optional[Path] = None  # Optional separate rule pool for distractors
    similarity_min: float = 0.3  # Min similarity for similarity_filtered strategy
    similarity_max: float = 0.7  # Max similarity for similarity_filtered strategy
    llm_generated_distractors: int = 0  # Number of LLM-generated distractors (0-3)


# ============================================================================
# Distractor Generation
# ============================================================================


def generate_distractors(
    target_rule: Rule,
    all_rules: list[Rule],
    num_distractors: int,
    strategy: str,
    random_seed: int,
) -> list[str]:
    """
    Generate distractor articulations for multiple-choice.

    Args:
        target_rule: The rule being tested
        all_rules: All available rules to sample from
        num_distractors: Number of distractors to generate
        strategy: "same_category", "different_category", or "mixed"
        random_seed: Random seed for reproducibility

    Returns:
        List of distractor articulation strings
    """
    rng = random.Random(random_seed)

    # Filter out the target rule
    other_rules = [r for r in all_rules if r.rule_id != target_rule.rule_id]

    if not other_rules:
        raise ValueError("Not enough rules to generate distractors")

    if strategy == "same_category":
        # Select from same category
        candidates = [r for r in other_rules if r.category == target_rule.category]
        if len(candidates) < num_distractors:
            # Fall back to all rules if not enough in same category
            candidates = other_rules
    elif strategy == "different_category":
        # Select from different categories
        candidates = [r for r in other_rules if r.category != target_rule.category]
        if len(candidates) < num_distractors:
            # Fall back to all rules if not enough in different categories
            candidates = other_rules
    else:  # "mixed"
        candidates = other_rules

    if len(candidates) < num_distractors:
        raise ValueError(
            f"Not enough candidate rules: need {num_distractors}, have {len(candidates)}"
        )

    # Sample distractors
    selected = rng.sample(candidates, num_distractors)
    return [r.articulation for r in selected]


def generate_distractors_similarity_filtered(
    target_rule: Rule,
    all_rules: list[Rule],
    num_distractors: int,
    strategy: str,
    random_seed: int,
    similarity_min: float = 0.3,
    similarity_max: float = 0.7,
    embedding_cache: Optional[EmbeddingCache] = None,
) -> list[str]:
    """
    Generate distractor articulations using embedding-based similarity filtering.

    Filters distractors to be:
    - Not too similar (confusingly close to target)
    - Not too dissimilar (obviously wrong)

    Args:
        target_rule: The rule being tested
        all_rules: All available rules to sample from
        num_distractors: Number of distractors to generate
        strategy: Base strategy ("same_category", "different_category", "mixed")
        random_seed: Random seed for reproducibility
        similarity_min: Minimum cosine similarity (default 0.3)
        similarity_max: Maximum cosine similarity (default 0.7)
        embedding_cache: EmbeddingCache instance (created if None)

    Returns:
        List of distractor articulation strings
    """
    rng = random.Random(random_seed)

    # Create embedding cache if not provided
    if embedding_cache is None:
        embedding_cache = EmbeddingCache()

    # Filter out the target rule
    other_rules = [r for r in all_rules if r.rule_id != target_rule.rule_id]

    if not other_rules:
        raise ValueError("Not enough rules to generate distractors")

    # Apply base strategy filter
    if strategy == "same_category":
        candidates = [r for r in other_rules if r.category == target_rule.category]
        if len(candidates) < num_distractors:
            candidates = other_rules
    elif strategy == "different_category":
        candidates = [r for r in other_rules if r.category != target_rule.category]
        if len(candidates) < num_distractors:
            candidates = other_rules
    else:  # "mixed" or "similarity_filtered"
        candidates = other_rules

    # Get target embedding
    target_emb = embedding_cache.get_embedding(target_rule.articulation)

    # Get candidate embeddings
    candidate_embs = embedding_cache.get_embeddings_batch([r.articulation for r in candidates])

    # Calculate similarities
    similarities = batch_cosine_similarity(target_emb, candidate_embs)

    # Filter by similarity range
    valid_candidates = []
    for rule, sim in zip(candidates, similarities):
        if similarity_min <= sim <= similarity_max:
            valid_candidates.append((rule, sim))

    # If not enough valid candidates, relax thresholds
    if len(valid_candidates) < num_distractors:
        # Gradually relax thresholds
        relaxed_min = max(0.0, similarity_min - 0.1)
        relaxed_max = min(1.0, similarity_max + 0.1)

        valid_candidates = []
        for rule, sim in zip(candidates, similarities):
            if relaxed_min <= sim <= relaxed_max:
                valid_candidates.append((rule, sim))

    # Still not enough? Use all candidates
    if len(valid_candidates) < num_distractors:
        valid_candidates = [(rule, sim) for rule, sim in zip(candidates, similarities)]

    if len(valid_candidates) < num_distractors:
        raise ValueError(
            f"Not enough candidate rules: need {num_distractors}, have {len(valid_candidates)}"
        )

    # Sample distractors (prefer mid-range similarities)
    # Weight by inverse distance from mid-range
    mid_sim = (similarity_min + similarity_max) / 2
    weights = [1.0 / (abs(sim - mid_sim) + 0.1) for _, sim in valid_candidates]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted random sample
    selected_indices = rng.choices(
        range(len(valid_candidates)),
        weights=weights,
        k=num_distractors
    )

    # Ensure uniqueness
    selected_indices = list(set(selected_indices))
    while len(selected_indices) < num_distractors:
        # Add more if deduplication removed some
        new_idx = rng.choices(range(len(valid_candidates)), weights=weights, k=1)[0]
        if new_idx not in selected_indices:
            selected_indices.append(new_idx)

    selected = [valid_candidates[i][0] for i in selected_indices[:num_distractors]]
    return [r.articulation for r in selected]


async def generate_llm_distractor(
    target_rule: Rule,
    model: str,
    temperature: float,
    cache_mode: CacheMode,
    cache_dir: Path,
) -> str:
    """
    Generate a distractor articulation using an LLM.

    Prompts the LLM to create a similar but distinctly different rule.

    Args:
        target_rule: The rule being tested
        model: LLM model to use for generation
        temperature: Sampling temperature
        cache_mode: API cache mode
        cache_dir: Cache directory

    Returns:
        Generated distractor articulation string
    """
    prompt = f"""Given this classification rule:

"{target_rule.articulation}"

Generate a similar but distinctly different classification rule. The new rule should:
1. Be for the same type of input (text strings)
2. Be plausible and well-defined
3. Have a different classification criterion than the original
4. Be at a similar level of complexity

Only output the new rule articulation, nothing else."""

    caller = create_caller(
        model=model,
        temperature=temperature,
        max_tokens=200,
        cache_mode=cache_mode,
        cache_dir=cache_dir,
        max_concurrent=1,
    )

    messages = [Message(role="user", content=prompt)]
    response = await caller.call(messages)

    return response.content.strip()


# ============================================================================
# Prompt Building
# ============================================================================


def build_mc_prompt(
    few_shot_examples: list[DatasetSample],
    correct_articulation: str,
    distractors: list[str],
    random_seed: int,
) -> tuple[str, list[int], str]:
    """
    Build multiple-choice articulation prompt.

    Args:
        few_shot_examples: List of few-shot examples
        correct_articulation: The correct rule articulation
        distractors: List of distractor articulations
        random_seed: Random seed for randomizing option order

    Returns:
        Tuple of (prompt_text, option_order, correct_option_letter)
        - option_order maps A/B/C/D to indices (0=correct, 1-3=distractors)
        - correct_option_letter is 'A', 'B', 'C', or 'D'
    """
    rng = random.Random(random_seed)

    # Create options list
    all_options = [correct_articulation] + distractors

    # Randomize order
    option_indices = list(range(len(all_options)))
    rng.shuffle(option_indices)

    # Find correct option letter
    correct_idx = option_indices.index(0)
    correct_letter = chr(ord('A') + correct_idx)

    # Build prompt
    prompt_parts = [
        "You are shown examples of a classification task. Select which rule best explains the pattern.",
        "",
        "Examples:",
    ]

    # Add few-shot examples
    for example in few_shot_examples:
        prompt_parts.append(f'Input: "{example.input}" → {str(example.label)}')

    prompt_parts.append("")
    prompt_parts.append("Which rule explains this pattern?")
    prompt_parts.append("")

    # Add options
    for i, idx in enumerate(option_indices):
        option_letter = chr(ord('A') + i)
        option_text = all_options[idx]
        prompt_parts.append(f"{option_letter}) {option_text}")

    prompt_parts.append("")
    prompt_parts.append("Answer with just the letter (A, B, C, or D):")

    return "\n".join(prompt_parts), option_indices, correct_letter


# ============================================================================
# Response Parsing
# ============================================================================


def parse_mc_response(response: str) -> Optional[str]:
    """
    Parse multiple-choice response to extract selected option.

    Handles variations like: "A", "a", "A)", "The answer is A", etc.

    Args:
        response: Model response text

    Returns:
        Selected option letter (A/B/C/D) or None if unparseable
    """
    response_clean = response.strip().upper()

    # Direct match
    if response_clean in ["A", "B", "C", "D"]:
        return response_clean

    # Match with parenthesis or period
    if response_clean.startswith(("A)", "B)", "C)", "D)", "A.", "B.", "C.", "D.")):
        return response_clean[0]

    # Check first character
    if response_clean and response_clean[0] in ["A", "B", "C", "D"]:
        return response_clean[0]

    # Extract first occurrence of A/B/C/D
    for char in response_clean:
        if char in ["A", "B", "C", "D"]:
            return char

    # Unable to parse
    return None


# ============================================================================
# Data Splitting
# ============================================================================


def stratified_split(
    samples: list[DatasetSample],
    few_shot_count: int,
    random_seed: int,
) -> list[DatasetSample]:
    """
    Get stratified few-shot examples (balanced positive/negative).

    Args:
        samples: All dataset samples
        few_shot_count: Number of few-shot examples
        random_seed: Random seed for reproducibility

    Returns:
        List of few-shot samples
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

    # Check we have enough samples
    if len(positive_samples) < few_shot_positive:
        raise ValueError(
            f"Not enough positive samples: need {few_shot_positive}, "
            f"have {len(positive_samples)}"
        )
    if len(negative_samples) < few_shot_negative:
        raise ValueError(
            f"Not enough negative samples: need {few_shot_negative}, "
            f"have {len(negative_samples)}"
        )

    # Split
    few_shot_samples = (
        positive_samples[:few_shot_positive] +
        negative_samples[:few_shot_negative]
    )

    # Shuffle to mix positive/negative
    rng.shuffle(few_shot_samples)

    return few_shot_samples


# ============================================================================
# Evaluation
# ============================================================================


async def evaluate_rule_mc(
    rule: Rule,
    all_rules: list[Rule],
    model: str,
    config: MCArticulationConfig,
    logger: logging.Logger,
) -> list[MCArticulationResult]:
    """
    Evaluate MC articulation for a single rule.

    Args:
        rule: Rule to evaluate
        all_rules: All rules (for generating distractors)
        model: Model name
        config: Configuration
        logger: Logger instance

    Returns:
        List of MCArticulationResult objects
    """
    logger.info(
        f"Evaluating MC articulation for {rule.rule_id} with {model}"
    )

    # Check learnability - skip if model not learnable for this rule
    if rule.learnability is not None:
        if model not in rule.learnability:
            logger.info(
                f"Skipping {rule.rule_id} | {model}: Not in learnability data"
            )
            return []

        # Test articulation at multiple few-shot counts (like learnability)
        # This lets us see if articulation improves with more examples
        few_shot_counts = [5, 10, 20, 50, 100]
        logger.info(
            f"Testing articulation at multiple few-shot counts: {few_shot_counts}"
        )
    else:
        # No learnability data, use config default
        few_shot_counts = [config.few_shot_count]
        logger.debug(f"No learnability data, using few_shot_counts: {few_shot_counts}")

    # Load dataset
    dataset_path = config.datasets_dir / f"{rule.rule_id}.jsonl"
    if not dataset_path.exists():
        logger.warning(f"Dataset not found: {dataset_path}")
        return []

    dataset_dicts = load_jsonl(dataset_path)
    dataset_samples = [DatasetSample(**d) for d in dataset_dicts]

    logger.info(f"Loaded {len(dataset_samples)} samples from {dataset_path.name}")

    # Create API caller
    caller = create_caller(
        model=model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        cache_mode=config.cache_mode,
        cache_dir=config.cache_dir,
        max_concurrent=config.max_concurrent,
    )

    # Initialize embedding cache if using similarity filtering
    embedding_cache = None
    if config.distractor_strategy == "similarity_filtered":
        embedding_cache = EmbeddingCache(cache_dir=config.cache_dir / "embeddings")

    # Run multiple test cases with different random seeds
    results = []

    for test_idx in range(config.num_test_cases):
        # Use different seed for each test case
        test_seed = config.random_seed + test_idx

        # Generate distractors ONCE per test case (reuse across all few-shot counts)
        try:
            # Determine number of each type of distractor
            num_llm_distractors = config.llm_generated_distractors
            num_pool_distractors = 3 - num_llm_distractors

            distractors = []

            # Generate pool-based distractors
            if num_pool_distractors > 0:
                if config.distractor_strategy == "similarity_filtered":
                    pool_distractors = generate_distractors_similarity_filtered(
                        target_rule=rule,
                        all_rules=all_rules,
                        num_distractors=num_pool_distractors,
                        strategy="mixed",
                        random_seed=test_seed,
                        similarity_min=config.similarity_min,
                        similarity_max=config.similarity_max,
                        embedding_cache=embedding_cache,
                    )
                else:
                    pool_distractors = generate_distractors(
                        target_rule=rule,
                        all_rules=all_rules,
                        num_distractors=num_pool_distractors,
                        strategy=config.distractor_strategy,
                        random_seed=test_seed,
                    )
                distractors.extend(pool_distractors)

            # Generate LLM distractors
            if num_llm_distractors > 0:
                for _ in range(num_llm_distractors):
                    llm_distractor = await generate_llm_distractor(
                        target_rule=rule,
                        model=model,
                        temperature=0.7,  # Higher temp for diversity
                        cache_mode=config.cache_mode,
                        cache_dir=config.cache_dir,
                    )
                    distractors.append(llm_distractor)

        except ValueError as e:
            logger.error(f"Failed to generate distractors: {e}")
            continue

        # Test with multiple few-shot counts using SAME distractors
        for few_shot_count in few_shot_counts:
            # Get few-shot examples for this count
            try:
                few_shot_samples = stratified_split(
                    samples=dataset_samples,
                    few_shot_count=few_shot_count,
                    random_seed=test_seed,
                )
            except ValueError as e:
                logger.error(f"Failed to get few-shot samples (n={few_shot_count}): {e}")
                continue

            # Build prompt
            prompt, option_order, correct_option = build_mc_prompt(
                few_shot_examples=few_shot_samples,
                correct_articulation=rule.articulation,
                distractors=distractors,  # REUSE same distractors
                random_seed=test_seed,
            )

            messages = [Message(role="user", content=prompt)]

            # Call API
            response = await caller.call(messages)

            # Parse response
            predicted_option = parse_mc_response(response.content)
            parse_error = predicted_option is None

            # Determine correctness
            if predicted_option is not None:
                correct = predicted_option == correct_option
            else:
                correct = None

            # Create result
            result = MCArticulationResult(
                rule_id=rule.rule_id,
                model=model,
                few_shot_count=few_shot_count,
                test_index=test_idx,
                correct_articulation=rule.articulation,
                distractors=distractors,
                option_order=option_order,
                correct_option=correct_option,
                predicted_option=predicted_option,
                raw_response=response.content,
                correct=correct,
                parse_error=parse_error,
            )

            results.append(result)

            logger.debug(
                f"Test {test_idx}, n={few_shot_count}: correct={correct_option}, "
                f"predicted={predicted_option}, correct={correct}"
            )

    # Log summary
    n_correct = sum(1 for r in results if r.correct)
    n_parseable = sum(1 for r in results if not r.parse_error)
    accuracy = n_correct / len(results) if results else 0.0
    parse_rate = n_parseable / len(results) if results else 0.0

    logger.info(
        f"{rule.rule_id} | {model}: "
        f"accuracy={accuracy:.2%} ({n_correct}/{len(results)}), "
        f"parse_rate={parse_rate:.2%}"
    )

    return results


# ============================================================================
# Main Runner
# ============================================================================


async def run_mc_articulation_tests(config: MCArticulationConfig) -> None:
    """
    Run MC articulation tests for all rules and models.

    Args:
        config: Configuration for MC articulation tests
    """
    # Setup logging
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = config.output_dir / "articulation_mc.log"
    logger = logging.getLogger("articulation_mc")
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
    logger.info("Starting Multiple-Choice Articulation Tests (Step 2)")
    logger.info("=" * 80)
    logger.info(f"Rules file: {config.rules_file}")
    logger.info(f"Datasets directory: {config.datasets_dir}")
    logger.info(f"Few-shot count: {config.few_shot_count}")
    logger.info(f"Test cases per rule: {config.num_test_cases}")
    logger.info(f"Distractor strategy: {config.distractor_strategy}")
    if config.distractor_strategy == "similarity_filtered":
        logger.info(f"Similarity range: [{config.similarity_min}, {config.similarity_max}]")
    if config.llm_generated_distractors > 0:
        logger.info(f"LLM-generated distractors: {config.llm_generated_distractors}")
    if config.distractor_pool_file:
        logger.info(f"Distractor pool file: {config.distractor_pool_file}")
    logger.info(f"Models: {config.models}")
    logger.info(f"Random seed: {config.random_seed}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("=" * 80)

    # Load rules to test
    rules_dicts = load_jsonl(config.rules_file)
    rules = [Rule(**r) for r in rules_dicts]
    logger.info(f"Loaded {len(rules)} rules to test from {config.rules_file.name}")

    # Load distractor pool (if separate from test rules)
    if config.distractor_pool_file:
        distractor_pool_dicts = load_jsonl(config.distractor_pool_file)
        distractor_pool = [Rule(**r) for r in distractor_pool_dicts]
        logger.info(f"Loaded {len(distractor_pool)} rules as distractor pool from {config.distractor_pool_file.name}")
    else:
        distractor_pool = rules
        logger.info(f"Using test rules as distractor pool")

    if len(distractor_pool) < 4:
        logger.error("Need at least 4 rules in distractor pool to generate distractors")
        return

    # Set random seed
    set_random_seed(config.random_seed)

    # Run evaluations in parallel across all rules and models
    async def evaluate_and_save(rule: Rule, model: str) -> tuple[str, str, dict[str, Any]]:
        """Evaluate a single rule-model combination and save results."""
        results = await evaluate_rule_mc(
            rule=rule,
            all_rules=distractor_pool,
            model=model,
            config=config,
            logger=logger,
        )

        if not results:
            logger.warning(f"No results for {rule.rule_id} | {model}")
            return rule.rule_id, model, {}

        # Save per-rule results
        output_file = config.output_dir / f"{rule.rule_id}_{model}_mc.jsonl"
        with output_file.open("w") as f:
            for result in results:
                f.write(result.model_dump_json() + "\n")

        logger.info(f"Saved results to {output_file.name}")

        # Compute summary statistics per few-shot count
        stats_by_shot = {}
        for few_shot_count in sorted(set(r.few_shot_count for r in results)):
            shot_results = [r for r in results if r.few_shot_count == few_shot_count]
            n_total = len(shot_results)
            n_correct = sum(1 for r in shot_results if r.correct)
            n_parseable = sum(1 for r in shot_results if not r.parse_error)
            accuracy = n_correct / n_total if n_total > 0 else 0.0
            parse_rate = n_parseable / n_total if n_total > 0 else 0.0

            stats_by_shot[few_shot_count] = {
                "accuracy": round(accuracy, 4),
                "n_correct": n_correct,
                "n_total": n_total,
                "parse_rate": round(parse_rate, 4),
                "n_parseable": n_parseable,
            }

        return rule.rule_id, model, stats_by_shot

    # Create all tasks (rule × model combinations)
    tasks = [
        evaluate_and_save(rule, model)
        for rule in rules
        for model in config.models
    ]

    logger.info(f"Running {len(tasks)} evaluations in parallel...")

    # Run all evaluations in parallel
    all_results = await asyncio.gather(*tasks)

    # Build summary from results
    summary: dict[str, dict[str, Any]] = {}
    for rule_id, model, stats in all_results:
        if stats:  # Skip empty results
            if rule_id not in summary:
                summary[rule_id] = {}
            summary[rule_id][model] = stats

    # Save summary
    summary_file = config.output_dir / "summary_mc.yaml"
    with summary_file.open("w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 80)
    logger.info(f"Summary saved to {summary_file}")
    logger.info("MC articulation tests complete!")
    logger.info("=" * 80)

    # Print summary table
    print("\n" + "=" * 80)
    print("MULTIPLE-CHOICE ARTICULATION TEST SUMMARY")
    print("=" * 80 + "\n")

    for rule_id, model_results in summary.items():
        print(f"\n{rule_id}:")
        for model, stats_by_shot in model_results.items():
            # Print results for each few-shot count
            for shot_count, metrics in sorted(stats_by_shot.items()):
                accuracy = metrics["accuracy"]
                status = "✓" if accuracy >= 0.80 else "✗"
                print(
                    f"  {model} ({shot_count}-shot): {accuracy:.1%} "
                    f"({metrics['n_correct']}/{metrics['n_total']}) {status}"
                )

    print("\n" + "=" * 80)


# ============================================================================
# CLI
# ============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for MC articulation tests."""
    parser = argparse.ArgumentParser(
        description="Test articulation via multiple-choice (Step 2)",
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
        default=Path("experiments/articulation_mc"),
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
        "--num-test-cases",
        type=int,
        default=5,
        help="Number of MC questions per rule",
    )
    parser.add_argument(
        "--distractor-strategy",
        type=str,
        default="mixed",
        choices=["same_category", "different_category", "mixed", "similarity_filtered"],
        help="Strategy for generating distractors",
    )
    parser.add_argument(
        "--distractor-pool-file",
        type=Path,
        default=None,
        help="Optional JSONL file with rules to use as distractor pool (if different from test rules)",
    )
    parser.add_argument(
        "--similarity-min",
        type=float,
        default=0.3,
        help="Minimum similarity for similarity_filtered strategy (0-1)",
    )
    parser.add_argument(
        "--similarity-max",
        type=float,
        default=0.7,
        help="Maximum similarity for similarity_filtered strategy (0-1)",
    )
    parser.add_argument(
        "--llm-generated-distractors",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Number of LLM-generated distractors (0-3)",
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
    config = MCArticulationConfig(
        rules_file=args.rules_file,
        datasets_dir=args.datasets_dir,
        few_shot_count=args.few_shot_count,
        num_test_cases=args.num_test_cases,
        models=args.models,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        cache_mode=CacheMode(args.cache_mode),
        cache_dir=args.cache_dir,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        log_level=args.log_level,
        distractor_strategy=args.distractor_strategy,
        distractor_pool_file=args.distractor_pool_file,
        similarity_min=args.similarity_min,
        similarity_max=args.similarity_max,
        llm_generated_distractors=args.llm_generated_distractors,
    )

    # Run tests
    asyncio.run(run_mc_articulation_tests(config))


if __name__ == "__main__":
    main()
