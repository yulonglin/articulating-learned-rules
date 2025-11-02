"""
Rule curation for selecting high-quality, diverse, implementable rules.

Performs:
1. Deduplication (exact and similarity-based)
2. Implementability assessment (programmatic vs LLM-needed)
3. Quality scoring (articulation clarity, example consistency, etc.)
4. Diversity-based selection (category balance, difficulty mix)

Output: Curated JSONL with metadata fields added
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RuleExample(BaseModel):
    """Single example for a classification rule."""

    input: str
    label: bool


class ClassificationRule(BaseModel):
    """Structured classification rule with metadata."""

    rule_id: str
    rule_name: str
    articulation: str
    category: Literal["syntactic", "pattern", "semantic", "statistical"]
    examples: list[RuleExample]
    expected_difficulty: Literal["easy", "moderate", "hard"]
    source_model: str
    timestamp: str
    prompt_strategy: str


class CuratedRule(BaseModel):
    """Classification rule with curation metadata."""

    # Original fields
    rule_id: str
    rule_name: str
    articulation: str
    category: Literal["syntactic", "pattern", "semantic", "statistical"]
    examples: list[RuleExample]
    expected_difficulty: Literal["easy", "moderate", "hard"]
    source_model: str
    timestamp: str
    prompt_strategy: str

    # Curation metadata
    implementability: Literal["programmatic", "llm_needed", "complex"]
    similarity_cluster: Optional[str]
    selection_reason: str
    quality_score: float = Field(ge=0.0, le=1.0)


@dataclass
class CurationStats:
    """Statistics from the curation process."""

    total_rules: int
    exact_duplicates_removed: int
    similar_rules_clustered: int
    total_clusters: int
    implementable_rules: int
    llm_needed_rules: int
    complex_rules: int
    final_selected: int
    category_distribution: dict[str, int]
    difficulty_distribution: dict[str, int]


def load_rules(input_path: Path) -> list[ClassificationRule]:
    """Load rules from JSONL file."""
    rules = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            rules.append(ClassificationRule(**data))
    logger.info(f"Loaded {len(rules)} rules from {input_path}")
    return rules


def remove_exact_duplicates(rules: list[ClassificationRule]) -> tuple[list[ClassificationRule], int]:
    """Remove rules with duplicate rule_name, keeping first occurrence."""
    seen_names = set()
    unique_rules = []
    duplicates = 0

    for rule in rules:
        if rule.rule_name not in seen_names:
            seen_names.add(rule.rule_name)
            unique_rules.append(rule)
        else:
            duplicates += 1
            logger.debug(f"Duplicate rule_name: {rule.rule_name}")

    logger.info(f"Removed {duplicates} exact duplicates by rule_name")
    return unique_rules, duplicates


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_normalized_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity score (0=different, 1=identical)."""
    distance = compute_levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)


def extract_keywords(text: str) -> set[str]:
    """Extract keywords from text (simple word extraction)."""
    # Simple tokenization - split on whitespace and punctuation
    import re

    words = re.findall(r"\b\w+\b", text.lower())
    # Filter out common stopwords
    stopwords = {"the", "is", "are", "if", "it", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    return {w for w in words if w not in stopwords and len(w) > 2}


def compute_keyword_overlap(s1: str, s2: str) -> float:
    """Compute keyword overlap between two strings."""
    kw1 = extract_keywords(s1)
    kw2 = extract_keywords(s2)
    if not kw1 or not kw2:
        return 0.0
    intersection = len(kw1 & kw2)
    union = len(kw1 | kw2)
    return intersection / union if union > 0 else 0.0


def cluster_similar_rules(
    rules: list[ClassificationRule], similarity_threshold: float = 0.75
) -> dict[str, list[ClassificationRule]]:
    """
    Cluster rules by similarity of articulation.

    Uses simple greedy clustering based on Levenshtein distance + keyword overlap.
    Returns dict mapping cluster_id to list of rules in that cluster.
    """
    clusters: dict[str, list[ClassificationRule]] = {}
    cluster_id_counter = 0

    for rule in rules:
        # Try to find existing cluster with high similarity
        best_cluster = None
        best_similarity = 0.0

        for cluster_id, cluster_rules in clusters.items():
            # Compare with representative (first) rule in cluster
            representative = cluster_rules[0]
            text_sim = compute_normalized_similarity(rule.articulation, representative.articulation)
            kw_sim = compute_keyword_overlap(rule.articulation, representative.articulation)
            combined_sim = 0.6 * text_sim + 0.4 * kw_sim

            if combined_sim > best_similarity:
                best_similarity = combined_sim
                best_cluster = cluster_id

        # Add to existing cluster or create new one
        if best_similarity >= similarity_threshold and best_cluster is not None:
            clusters[best_cluster].append(rule)
            logger.debug(
                f"Added '{rule.rule_name}' to cluster {best_cluster} "
                f"(similarity={best_similarity:.2f})"
            )
        else:
            new_cluster_id = f"cluster_{cluster_id_counter}"
            clusters[new_cluster_id] = [rule]
            cluster_id_counter += 1

    logger.info(
        f"Clustered {len(rules)} rules into {len(clusters)} clusters "
        f"(threshold={similarity_threshold})"
    )

    # Log cluster sizes
    cluster_sizes = Counter(len(cluster) for cluster in clusters.values())
    logger.info(f"Cluster size distribution: {dict(cluster_sizes)}")

    return clusters


def assess_implementability(rule: ClassificationRule) -> Literal["programmatic", "llm_needed", "complex"]:
    """
    Assess whether a rule can be implemented programmatically or needs LLM.

    - programmatic: Can implement with simple functions (regex, string ops, etc.)
    - llm_needed: Requires semantic understanding (sentiment, topic classification)
    - complex: Very hard to implement or ambiguous
    """
    articulation = rule.articulation.lower()
    category = rule.category

    # Syntactic/pattern rules are usually programmatic
    if category in ["syntactic", "pattern"]:
        # Check for ambiguous terms
        ambiguous_terms = ["generally", "usually", "often", "mostly", "somewhat"]
        if any(term in articulation for term in ambiguous_terms):
            return "complex"

        # Check for linguistic understanding needed
        linguistic_terms = ["noun", "verb", "adjective", "adverb", "part of speech", "synonym", "antonym"]
        if any(term in articulation for term in linguistic_terms):
            return "llm_needed"

        # Check for complex phonetic/linguistic features
        complex_terms = ["phonetic", "syllable", "morpheme", "homophone"]
        if any(term in articulation for term in complex_terms):
            return "llm_needed"

        return "programmatic"

    # Semantic rules typically need LLM
    if category == "semantic":
        # Check for simple keyword-based rules
        simple_semantic = ["contains", "mentions", "includes word"]
        if any(term in articulation for term in simple_semantic):
            return "programmatic"
        return "llm_needed"

    # Statistical rules can be complex
    if category == "statistical":
        # Check for well-defined statistical measures
        stats_terms = [
            "ratio",
            "percentage",
            "count",
            "length",
            "frequency",
            "average",
            "variance",
            "entropy",
        ]
        if any(term in articulation for term in stats_terms):
            # Check if bounds are specified
            if any(
                char in articulation
                for char in ["<", ">", "between", "more than", "less than", "exceeds"]
            ):
                return "programmatic"
            return "complex"
        return "complex"

    return "llm_needed"


def check_example_consistency(rule: ClassificationRule) -> float:
    """
    Check if examples are consistent with articulation.
    Returns score 0.0-1.0 (higher = more consistent).

    Simple heuristics:
    - Check if positive/negative examples are balanced
    - Check if examples have reasonable diversity
    - Check if all examples have labels
    """
    if not rule.examples or len(rule.examples) < 3:
        return 0.3  # Too few examples

    # Check label balance (should have both true and false)
    labels = [ex.label for ex in rule.examples]
    num_true = sum(labels)
    num_false = len(labels) - num_true

    if num_true == 0 or num_false == 0:
        return 0.4  # Imbalanced labels

    # Check input diversity (should not all be the same)
    inputs = [ex.input for ex in rule.examples]
    unique_inputs = len(set(inputs))
    if unique_inputs < len(inputs):
        return 0.6  # Some duplicate inputs

    # Check if inputs have reasonable length variation
    input_lengths = [len(inp) for inp in inputs]
    if max(input_lengths) - min(input_lengths) < 3:
        return 0.7  # All inputs very similar length

    return 1.0  # Looks good


def compute_quality_score(rule: ClassificationRule) -> float:
    """
    Compute overall quality score for a rule.

    Factors:
    - Articulation clarity (length, specificity)
    - Example consistency
    - Number of examples
    - Difficulty alignment with category
    """
    score = 0.0

    # Articulation clarity (30%)
    articulation_length = len(rule.articulation)
    if 50 <= articulation_length <= 300:
        score += 0.3
    elif 30 <= articulation_length < 50 or 300 < articulation_length <= 500:
        score += 0.2
    else:
        score += 0.1

    # Example consistency (40%)
    consistency_score = check_example_consistency(rule)
    score += 0.4 * consistency_score

    # Number of examples (15%)
    num_examples = len(rule.examples)
    if num_examples >= 5:
        score += 0.15
    elif num_examples >= 4:
        score += 0.12
    elif num_examples >= 3:
        score += 0.08

    # Category-difficulty alignment (15%)
    # Syntactic/pattern should mostly be easy/moderate
    # Semantic/statistical can be harder
    if rule.category in ["syntactic", "pattern"]:
        if rule.expected_difficulty in ["easy", "moderate"]:
            score += 0.15
        else:
            score += 0.08
    else:  # semantic, statistical
        if rule.expected_difficulty in ["moderate", "hard"]:
            score += 0.15
        else:
            score += 0.10

    return min(1.0, score)


def select_best_from_cluster(cluster_rules: list[ClassificationRule]) -> ClassificationRule:
    """Select the best rule from a cluster based on quality score."""
    if len(cluster_rules) == 1:
        return cluster_rules[0]

    # Compute quality scores and select best
    scored_rules = [(rule, compute_quality_score(rule)) for rule in cluster_rules]
    best_rule, best_score = max(scored_rules, key=lambda x: x[1])

    logger.debug(
        f"Selected '{best_rule.rule_name}' from cluster of {len(cluster_rules)} "
        f"(score={best_score:.2f})"
    )

    return best_rule


def select_diverse_rules(
    clusters: dict[str, list[ClassificationRule]],
    target_count: int,
    category_balance: str = "balanced",
) -> list[tuple[ClassificationRule, str, str]]:
    """
    Select diverse rules from clusters.

    Returns list of (rule, cluster_id, selection_reason) tuples.

    Strategy:
    1. Ensure minimum representation from each category
    2. Within categories, select by quality score and difficulty mix
    3. Fill remaining slots with highest-quality rules
    """
    # First, select best rule from each cluster
    cluster_representatives: list[tuple[ClassificationRule, str, float]] = []
    for cluster_id, cluster_rules in clusters.items():
        best_rule = select_best_from_cluster(cluster_rules)
        quality = compute_quality_score(best_rule)
        cluster_representatives.append((best_rule, cluster_id, quality))

    # If we have fewer clusters than target, return all
    if len(cluster_representatives) <= target_count:
        logger.info(f"Selecting all {len(cluster_representatives)} cluster representatives")
        return [
            (rule, cluster_id, "cluster_representative")
            for rule, cluster_id, _ in cluster_representatives
        ]

    # Group by category
    by_category: dict[str, list[tuple[ClassificationRule, str, float]]] = defaultdict(list)
    for rule, cluster_id, quality in cluster_representatives:
        by_category[rule.category].append((rule, cluster_id, quality))

    # Sort each category by quality
    for category in by_category:
        by_category[category].sort(key=lambda x: x[2], reverse=True)

    # Determine target counts per category
    categories = ["syntactic", "pattern", "semantic", "statistical"]
    if category_balance == "balanced":
        base_per_category = target_count // len(categories)
        remainder = target_count % len(categories)
        target_per_category = {cat: base_per_category for cat in categories}
        # Distribute remainder to categories with most rules
        category_counts = [(cat, len(by_category.get(cat, []))) for cat in categories]
        category_counts.sort(key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            target_per_category[category_counts[i][0]] += 1
    else:
        # Proportional to available rules
        total_available = len(cluster_representatives)
        target_per_category = {
            cat: max(1, int(target_count * len(by_category.get(cat, [])) / total_available))
            for cat in categories
        }

    logger.info(f"Target per category: {target_per_category}")

    # Select rules from each category
    selected: list[tuple[ClassificationRule, str, str]] = []

    for category in categories:
        category_rules = by_category.get(category, [])
        target = target_per_category[category]

        if not category_rules:
            logger.warning(f"No rules available for category: {category}")
            continue

        # Ensure difficulty diversity within category
        by_difficulty: dict[str, list[tuple[ClassificationRule, str, float]]] = defaultdict(list)
        for rule, cluster_id, quality in category_rules:
            by_difficulty[rule.expected_difficulty].append((rule, cluster_id, quality))

        # Select from each difficulty level
        difficulties = ["easy", "moderate", "hard"]
        selected_from_category = []

        # Round-robin selection across difficulties
        difficulty_indices = {d: 0 for d in difficulties}
        while len(selected_from_category) < target:
            added_this_round = False
            for difficulty in difficulties:
                if len(selected_from_category) >= target:
                    break
                idx = difficulty_indices[difficulty]
                if idx < len(by_difficulty[difficulty]):
                    rule, cluster_id, quality = by_difficulty[difficulty][idx]
                    reason = f"{category}_{difficulty}_representative"
                    selected_from_category.append((rule, cluster_id, reason))
                    difficulty_indices[difficulty] += 1
                    added_this_round = True

            # If we couldn't add any rules this round, break
            if not added_this_round:
                break

        selected.extend(selected_from_category)
        logger.info(f"Selected {len(selected_from_category)} rules from category: {category}")

    # If we're under target, add highest-quality remaining rules
    if len(selected) < target_count:
        selected_cluster_ids = {cluster_id for _, cluster_id, _ in selected}
        remaining = [
            (rule, cluster_id, quality)
            for rule, cluster_id, quality in cluster_representatives
            if cluster_id not in selected_cluster_ids
        ]
        remaining.sort(key=lambda x: x[2], reverse=True)

        needed = target_count - len(selected)
        for rule, cluster_id, quality in remaining[:needed]:
            selected.append((rule, cluster_id, f"high_quality_filler (score={quality:.2f})"))
            logger.debug(f"Added filler rule: {rule.rule_name} (quality={quality:.2f})")

    logger.info(f"Final selection: {len(selected)} rules")
    return selected


def curate_rules(
    input_path: Path,
    output_path: Path,
    target_count: int = 35,
    similarity_threshold: float = 0.75,
    category_balance: str = "balanced",
) -> CurationStats:
    """
    Main curation pipeline.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output curated JSONL file
        target_count: Target number of rules to select
        similarity_threshold: Threshold for similarity clustering (0.0-1.0)
        category_balance: "balanced" or "proportional"

    Returns:
        CurationStats object with statistics
    """
    # Load rules
    rules = load_rules(input_path)
    total_rules = len(rules)

    # Remove exact duplicates
    rules, exact_dupes = remove_exact_duplicates(rules)

    # Cluster similar rules
    clusters = cluster_similar_rules(rules, similarity_threshold)
    similar_rules_clustered = len(rules) - len(clusters)

    # Select diverse rules
    selected_tuples = select_diverse_rules(clusters, target_count, category_balance)

    # Create curated rules with metadata
    curated_rules: list[CuratedRule] = []
    implementability_counts = Counter()

    for rule, cluster_id, selection_reason in selected_tuples:
        implementability = assess_implementability(rule)
        implementability_counts[implementability] += 1

        quality_score = compute_quality_score(rule)

        curated_rule = CuratedRule(
            **rule.model_dump(),
            implementability=implementability,
            similarity_cluster=cluster_id,
            selection_reason=selection_reason,
            quality_score=quality_score,
        )
        curated_rules.append(curated_rule)

    # Sort by quality score descending
    curated_rules.sort(key=lambda r: r.quality_score, reverse=True)

    # Save to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rule in curated_rules:
            f.write(rule.model_dump_json() + "\n")

    logger.info(f"Saved {len(curated_rules)} curated rules to {output_path}")

    # Compute statistics
    category_dist = Counter(r.category for r in curated_rules)
    difficulty_dist = Counter(r.expected_difficulty for r in curated_rules)

    stats = CurationStats(
        total_rules=total_rules,
        exact_duplicates_removed=exact_dupes,
        similar_rules_clustered=similar_rules_clustered,
        total_clusters=len(clusters),
        implementable_rules=implementability_counts["programmatic"],
        llm_needed_rules=implementability_counts["llm_needed"],
        complex_rules=implementability_counts["complex"],
        final_selected=len(curated_rules),
        category_distribution=dict(category_dist),
        difficulty_distribution=dict(difficulty_dist),
    )

    return stats


def print_summary(stats: CurationStats, output_path: Path) -> None:
    """Print curation summary statistics."""
    print("\n" + "=" * 80)
    print("RULE CURATION SUMMARY")
    print("=" * 80)
    print(f"\nInput Statistics:")
    print(f"  Total rules loaded:          {stats.total_rules}")
    print(f"  Exact duplicates removed:    {stats.exact_duplicates_removed}")
    print(f"  Similar rules clustered:     {stats.similar_rules_clustered}")
    print(f"  Total clusters formed:       {stats.total_clusters}")

    print(f"\nImplementability Assessment:")
    print(f"  Programmatic:                {stats.implementable_rules}")
    print(f"  LLM needed:                  {stats.llm_needed_rules}")
    print(f"  Complex:                     {stats.complex_rules}")

    print(f"\nFinal Selection:")
    print(f"  Total rules selected:        {stats.final_selected}")

    print(f"\n  Category distribution:")
    for category, count in sorted(stats.category_distribution.items()):
        pct = 100 * count / stats.final_selected
        print(f"    {category:20s}: {count:3d} ({pct:5.1f}%)")

    print(f"\n  Difficulty distribution:")
    for difficulty, count in sorted(stats.difficulty_distribution.items()):
        pct = 100 * count / stats.final_selected
        print(f"    {difficulty:20s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nOutput: {output_path}")
    print("=" * 80 + "\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Curate brainstormed rules for quality, diversity, and implementability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with brainstormed rules",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file (default: tmp/curated_rules_TIMESTAMP.jsonl)",
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=35,
        help="Target number of rules to select",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for clustering (0.0-1.0)",
    )

    parser.add_argument(
        "--category-balance",
        choices=["balanced", "proportional"],
        default="balanced",
        help="How to balance categories: balanced (equal) or proportional (by availability)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"tmp/curated_rules_{timestamp}.jsonl")

    # Run curation
    stats = curate_rules(
        input_path=args.input,
        output_path=args.output,
        target_count=args.target_count,
        similarity_threshold=args.similarity_threshold,
        category_balance=args.category_balance,
    )

    # Print summary
    print_summary(stats, args.output)


if __name__ == "__main__":
    main()
