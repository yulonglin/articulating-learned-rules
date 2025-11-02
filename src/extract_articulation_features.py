"""
Extract Linguistic Features from Articulations

Analyzes articulations for linguistic markers:
- Hedging language (uncertainty markers)
- Confidence markers
- Specificity indicators (quantifiers, examples)
- Complexity metrics

Outputs JSONL with features per articulation.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np


def setup_logging(log_file: Path) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Linguistic marker dictionaries
HEDGING_WORDS = [
    "might", "may", "could", "possibly", "perhaps", "probably", "likely",
    "seems", "appears", "suggests", "tends", "often", "sometimes", "usually",
    "generally", "typically", "approximately", "roughly", "around", "about",
    "somewhat", "fairly", "relatively", "quite", "rather", "mostly", "largely"
]

CONFIDENCE_WORDS = [
    "always", "never", "must", "definitely", "certainly", "clearly", "obviously",
    "undoubtedly", "unquestionably", "absolutely", "consistently", "invariably",
    "necessarily", "surely", "indeed", "evidently", "manifestly", "decidedly",
    "all", "every", "none", "only", "exactly", "precisely", "strictly"
]

UNCERTAINTY_PHRASES = [
    "not sure", "unclear", "ambiguous", "uncertain", "unsure", "difficult to",
    "hard to", "challenging to", "complex", "nuanced", "subtle"
]

SPECIFICITY_MARKERS = {
    "quantifiers": [
        r"\bat least\b", r"\bmore than\b", r"\bless than\b", r"\bfewer than\b",
        r"\bgreater than\b", r"\bexactly\b", r"\bprecisely\b", r"\bbetween\b",
        r"\d+", r"\d+%", r"\d+\.\d+"
    ],
    "examples": [
        r"\be\.g\.", r"\bfor example\b", r"\bsuch as\b", r"\blike\b", r"\bincluding\b",
        r"\bspecifically\b", r"\bin particular\b"
    ],
    "conditionals": [
        r"\bif\b", r"\bwhen\b", r"\bunless\b", r"\bonly if\b", r"\bprovided\b",
        r"\bgiven\b", r"\bassuming\b"
    ]
}


def count_pattern_matches(text: str, patterns: List[str], case_sensitive: bool = False) -> int:
    """Count matches for a list of patterns (words or regexes)."""
    count = 0
    flags = 0 if case_sensitive else re.IGNORECASE

    for pattern in patterns:
        if pattern.startswith("\\b") or pattern.startswith("\\d"):
            # Regex pattern
            count += len(re.findall(pattern, text, flags=flags))
        else:
            # Simple word matching
            count += len(re.findall(r'\b' + re.escape(pattern) + r'\b', text, flags=re.IGNORECASE))

    return count


def extract_linguistic_features(articulation: str) -> Dict:
    """Extract linguistic features from an articulation."""
    # Basic stats
    words = articulation.split()
    word_count = len(words)
    char_count = len(articulation)

    # Normalize for scoring (per 100 words)
    norm_factor = word_count / 100 if word_count > 0 else 1

    # Hedging markers
    hedging_count = count_pattern_matches(articulation, HEDGING_WORDS)
    hedging_score = hedging_count / norm_factor if norm_factor else 0

    # Confidence markers
    confidence_count = count_pattern_matches(articulation, CONFIDENCE_WORDS)
    confidence_score = confidence_count / norm_factor if norm_factor else 0

    # Uncertainty phrases
    uncertainty_count = count_pattern_matches(articulation, UNCERTAINTY_PHRASES)
    uncertainty_score = uncertainty_count / norm_factor if norm_factor else 0

    # Specificity markers
    quantifier_count = count_pattern_matches(articulation, SPECIFICITY_MARKERS["quantifiers"])
    example_count = count_pattern_matches(articulation, SPECIFICITY_MARKERS["examples"])
    conditional_count = count_pattern_matches(articulation, SPECIFICITY_MARKERS["conditionals"])

    specificity_score = (quantifier_count + example_count + conditional_count) / norm_factor if norm_factor else 0

    # Structural complexity
    sentence_count = len(re.findall(r'[.!?]+', articulation))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count

    # Nested structure markers (parentheses, quotes, clauses)
    nested_markers = len(re.findall(r'[\(\)\[\]\{\}"\']', articulation))
    clause_markers = count_pattern_matches(articulation, [r'\,', r'\;', r'\band\b', r'\bor\b', r'\bbut\b'])

    complexity_score = (avg_sentence_length / 20) + (nested_markers / 10) + (clause_markers / word_count * 10)

    # Combined scores
    total_uncertainty = hedging_score + uncertainty_score
    total_certainty = confidence_score

    # Net certainty (positive = confident, negative = uncertain)
    net_certainty = total_certainty - total_uncertainty

    return {
        # Raw counts
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,

        # Hedging & uncertainty
        "hedging_count": hedging_count,
        "uncertainty_count": uncertainty_count,
        "hedging_score": round(hedging_score, 3),
        "uncertainty_score": round(uncertainty_score, 3),
        "total_uncertainty": round(total_uncertainty, 3),

        # Confidence
        "confidence_count": confidence_count,
        "confidence_score": round(confidence_score, 3),

        # Specificity
        "quantifier_count": quantifier_count,
        "example_count": example_count,
        "conditional_count": conditional_count,
        "specificity_score": round(specificity_score, 3),

        # Complexity
        "avg_sentence_length": round(avg_sentence_length, 2),
        "nested_markers": nested_markers,
        "clause_markers": clause_markers,
        "complexity_score": round(complexity_score, 3),

        # Summary metrics
        "net_certainty": round(net_certainty, 3),  # Positive = confident, negative = uncertain
    }


def process_faithfulness_file(file_path: Path, logger: logging.Logger) -> Dict:
    """Process a faithfulness result file and extract features."""
    try:
        with open(file_path) as f:
            data = json.load(f)

        articulation = data.get("generated_articulation", "")
        if not articulation:
            return None

        features = extract_linguistic_features(articulation)

        return {
            "rule_id": data.get("rule_id"),
            "model": data.get("model"),
            "few_shot_count": data.get("few_shot_count"),
            "articulation": articulation,
            **features,
            "faithfulness_metrics": {
                "counterfactual_faithfulness": data.get("counterfactual_faithfulness"),
                "consistency_score": data.get("consistency_score"),
                "cross_context_match_score": data.get("cross_context_match_score"),
                "functional_accuracy": data.get("functional_accuracy")
            }
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract linguistic features from articulations")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing faithfulness results"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSONL file (default: results_dir/linguistic_features.jsonl)"
    )
    args = parser.parse_args()

    # Setup output
    if args.output_file is None:
        args.output_file = args.results_dir / "linguistic_features.jsonl"

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.output_file.parent / f"{args.output_file.stem}.log"

    logger = setup_logging(log_file)
    logger.info("=" * 80)
    logger.info("Linguistic Feature Extraction")
    logger.info("=" * 80)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output file: {args.output_file}")

    # Find all faithfulness result files
    result_files = list(args.results_dir.glob("*_faithfulness.jsonl"))
    logger.info(f"Found {len(result_files)} result files")

    # Extract features
    all_features = []
    for file_path in result_files:
        features = process_faithfulness_file(file_path, logger)
        if features:
            all_features.append(features)

    logger.info(f"Extracted features from {len(all_features)} articulations")

    # Save features
    with open(args.output_file, 'w') as f:
        for features in all_features:
            f.write(json.dumps(features) + '\n')

    logger.info(f"Saved features to {args.output_file}")

    # Compute summary statistics
    if all_features:
        logger.info("\n" + "=" * 80)
        logger.info("Summary Statistics")
        logger.info("=" * 80)

        # Aggregate scores
        hedging_scores = [f["hedging_score"] for f in all_features]
        confidence_scores = [f["confidence_score"] for f in all_features]
        specificity_scores = [f["specificity_score"] for f in all_features]
        complexity_scores = [f["complexity_score"] for f in all_features]
        net_certainty_scores = [f["net_certainty"] for f in all_features]

        logger.info(f"\nHedging score: mean={np.mean(hedging_scores):.3f}, std={np.std(hedging_scores):.3f}")
        logger.info(f"Confidence score: mean={np.mean(confidence_scores):.3f}, std={np.std(confidence_scores):.3f}")
        logger.info(f"Specificity score: mean={np.mean(specificity_scores):.3f}, std={np.std(specificity_scores):.3f}")
        logger.info(f"Complexity score: mean={np.mean(complexity_scores):.3f}, std={np.std(complexity_scores):.3f}")
        logger.info(f"Net certainty: mean={np.mean(net_certainty_scores):.3f}, std={np.std(net_certainty_scores):.3f}")

        # Distribution
        high_hedging = sum(1 for s in hedging_scores if s > 5)
        high_confidence = sum(1 for s in confidence_scores if s > 5)
        high_specificity = sum(1 for s in specificity_scores if s > 10)

        logger.info(f"\nHigh hedging (>5): {high_hedging}/{len(all_features)} ({100*high_hedging/len(all_features):.1f}%)")
        logger.info(f"High confidence (>5): {high_confidence}/{len(all_features)} ({100*high_confidence/len(all_features):.1f}%)")
        logger.info(f"High specificity (>10): {high_specificity}/{len(all_features)} ({100*high_specificity/len(all_features):.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("Feature extraction complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
