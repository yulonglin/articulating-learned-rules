"""
Analyze Relationship Between Linguistic Features and Faithfulness

Correlates extracted linguistic features with faithfulness metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


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


def load_features(features_file: Path) -> List[Dict]:
    """Load linguistic features from JSONL."""
    features = []
    with open(features_file) as f:
        for line in f:
            features.append(json.loads(line))
    return features


def compute_correlation(x_vals: List[float], y_vals: List[float], logger: logging.Logger) -> Dict:
    """Compute correlation statistics."""
    # Filter out None values
    valid_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None]

    if len(valid_pairs) < 3:
        return {"error": "Insufficient valid data points"}

    x_vals, y_vals = zip(*valid_pairs)

    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)
    spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)

    return {
        "n": len(valid_pairs),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mean_x": float(np.mean(x_vals)),
        "mean_y": float(np.mean(y_vals))
    }


def create_scatter_with_regression(
    x_vals: List[float],
    y_vals: List[float],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """Create scatter plot with regression line."""
    valid_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None]
    if len(valid_pairs) < 3:
        logger.warning(f"Insufficient data for {title}")
        return

    x_vals, y_vals = zip(*valid_pairs)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, alpha=0.6)

    # Regression line
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'y={z[0]:.4f}x+{z[1]:.4f}')

    # Correlation
    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\nPearson r={pearson_r:.3f}, p={pearson_p:.3e}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scatter plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze linguistic features vs faithfulness")
    parser.add_argument(
        "--features-file",
        type=Path,
        required=True,
        help="JSONL file with linguistic features"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as features file parent)"
    )
    args = parser.parse_args()

    # Setup
    if args.output_dir is None:
        args.output_dir = args.features_file.parent

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(args.output_dir / "linguistic_analysis.log")
    logger.info("=" * 80)
    logger.info("Linguistic Features vs Faithfulness Analysis")
    logger.info("=" * 80)
    logger.info(f"Features file: {args.features_file}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load features
    features = load_features(args.features_file)
    logger.info(f"Loaded {len(features)} feature sets")

    # Analyze correlations
    logger.info("\n" + "=" * 80)
    logger.info("Correlation Analysis")
    logger.info("=" * 80)

    linguistic_vars = [
        ("hedging_score", "Hedging Score"),
        ("confidence_score", "Confidence Score"),
        ("specificity_score", "Specificity Score"),
        ("complexity_score", "Complexity Score"),
        ("net_certainty", "Net Certainty"),
        ("word_count", "Word Count")
    ]

    faithfulness_metrics = [
        ("counterfactual_faithfulness", "Counterfactual Faithfulness"),
        ("consistency_score", "Consistency Score"),
        ("cross_context_match_score", "Cross-Context Match"),
        ("functional_accuracy", "Functional Accuracy")
    ]

    correlations = {}

    for ling_var, ling_label in linguistic_vars:
        logger.info(f"\n{ling_label}:")
        correlations[ling_var] = {}

        for faith_metric, faith_label in faithfulness_metrics:
            # Extract values
            x_vals = [f[ling_var] for f in features]
            y_vals = [f["faithfulness_metrics"][faith_metric] for f in features]

            # Compute correlation
            corr = compute_correlation(x_vals, y_vals, logger)

            if "error" not in corr:
                logger.info(f"  {faith_label}: Pearson r={corr['pearson_r']:.3f}, p={corr['pearson_p']:.3e}")
                correlations[ling_var][faith_metric] = corr

                # Create scatter plot for significant correlations
                if corr['pearson_p'] < 0.05:
                    create_scatter_with_regression(
                        x_vals,
                        y_vals,
                        ling_label,
                        faith_label,
                        f"{ling_label} vs {faith_label}",
                        figures_dir / f"{ling_var}_vs_{faith_metric}.png",
                        logger
                    )

    # Save correlation matrix
    correlation_matrix = {}
    for ling_var, _ in linguistic_vars:
        correlation_matrix[ling_var] = {}
        for faith_metric, _ in faithfulness_metrics:
            if ling_var in correlations and faith_metric in correlations[ling_var]:
                correlation_matrix[ling_var][faith_metric] = {
                    "r": correlations[ling_var][faith_metric]["pearson_r"],
                    "p": correlations[ling_var][faith_metric]["pearson_p"]
                }

    summary_path = args.output_dir / "linguistic_correlations.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(correlation_matrix, f, default_flow_style=False)
    logger.info(f"\nSaved correlation summary: {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
