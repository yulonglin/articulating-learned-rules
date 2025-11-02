"""
Retroactive Length Analysis for Faithfulness Results

Analyzes the relationship between text length and faithfulness metrics
in existing faithfulness experiment results.

Outputs:
- Length statistics (YAML)
- Correlation analysis
- Visualizations (scatter plots, distributions)
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

from src.utils import load_jsonl


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


def compute_text_stats(text: str) -> Dict[str, int]:
    """Compute length statistics for a text."""
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0
    }


def analyze_faithfulness_file(file_path: Path, logger: logging.Logger) -> Dict:
    """Analyze a single faithfulness result file."""
    try:
        with open(file_path) as f:
            data = json.load(f)

        # Extract metadata
        rule_id = data.get("rule_id", "unknown")
        model = data.get("model", "unknown")
        few_shot_count = data.get("few_shot_count", 0)

        # Articulation stats
        articulation = data.get("generated_articulation", "")
        articulation_stats = compute_text_stats(articulation)

        # Counterfactual test stats
        counterfactual_tests = data.get("counterfactual_tests", [])
        test_lengths = []
        faithful_test_lengths = []
        unfaithful_test_lengths = []

        for test in counterfactual_tests:
            test_input = test.get("test_input", "")
            stats_dict = compute_text_stats(test_input)
            test_lengths.append(stats_dict)

            if test.get("faithful", False):
                faithful_test_lengths.append(stats_dict)
            else:
                unfaithful_test_lengths.append(stats_dict)

        # Aggregate metrics
        faithfulness_metrics = {
            "counterfactual_faithfulness": data.get("counterfactual_faithfulness"),
            "consistency_score": data.get("consistency_score"),
            "cross_context_match_score": data.get("cross_context_match_score"),
            "functional_accuracy": data.get("functional_accuracy")
        }

        return {
            "rule_id": rule_id,
            "model": model,
            "few_shot_count": few_shot_count,
            "articulation_stats": articulation_stats,
            "test_lengths": test_lengths,
            "faithful_test_lengths": faithful_test_lengths,
            "unfaithful_test_lengths": unfaithful_test_lengths,
            "num_tests": len(counterfactual_tests),
            "num_faithful": len(faithful_test_lengths),
            "num_unfaithful": len(unfaithful_test_lengths),
            "faithfulness_metrics": faithfulness_metrics,
            "file_path": str(file_path)
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def compute_correlations(data_points: List[Tuple[float, float]], logger: logging.Logger) -> Dict:
    """Compute correlation statistics between two variables."""
    if len(data_points) < 3:
        return {"error": "Insufficient data points"}

    x_vals, y_vals = zip(*data_points)

    # Filter out None values
    valid_points = [(x, y) for x, y in data_points if x is not None and y is not None]
    if len(valid_points) < 3:
        return {"error": "Insufficient valid data points"}

    x_vals, y_vals = zip(*valid_points)

    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)
    spearman_r, spearman_p = stats.spearmanr(x_vals, y_vals)

    return {
        "n": len(valid_points),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mean_x": float(np.mean(x_vals)),
        "mean_y": float(np.mean(y_vals)),
        "std_x": float(np.std(x_vals)),
        "std_y": float(np.std(y_vals))
    }


def create_scatter_plot(
    data_points: List[Tuple[float, float]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """Create scatter plot with regression line."""
    # Filter valid points
    valid_points = [(x, y) for x, y in data_points if x is not None and y is not None]
    if len(valid_points) < 3:
        logger.warning(f"Insufficient data for {title}")
        return

    x_vals, y_vals = zip(*valid_points)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_vals, y_vals, alpha=0.6)

    # Add regression line
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'y={z[0]:.4f}x+{z[1]:.4f}')

    # Compute correlation
    pearson_r, pearson_p = stats.pearsonr(x_vals, y_vals)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\nPearson r={pearson_r:.3f}, p={pearson_p:.3e}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scatter plot: {output_path}")


def create_distribution_comparison(
    faithful_lengths: List[float],
    unfaithful_lengths: List[float],
    x_label: str,
    title: str,
    output_path: Path,
    logger: logging.Logger
):
    """Create overlaid histograms comparing faithful vs unfaithful test lengths."""
    if not faithful_lengths or not unfaithful_lengths:
        logger.warning(f"Insufficient data for {title}")
        return

    plt.figure(figsize=(10, 6))

    plt.hist(faithful_lengths, bins=20, alpha=0.6, label=f'Faithful (n={len(faithful_lengths)})', color='green')
    plt.hist(unfaithful_lengths, bins=20, alpha=0.6, label=f'Unfaithful (n={len(unfaithful_lengths)})', color='red')

    # Add KDE
    if len(faithful_lengths) > 1:
        from scipy.stats import gaussian_kde
        kde_faithful = gaussian_kde(faithful_lengths)
        x_range = np.linspace(min(faithful_lengths + unfaithful_lengths),
                            max(faithful_lengths + unfaithful_lengths), 200)
        plt.plot(x_range, kde_faithful(x_range) * len(faithful_lengths) *
                (max(faithful_lengths) - min(faithful_lengths)) / 20,
                'g-', linewidth=2, alpha=0.8)

    if len(unfaithful_lengths) > 1:
        from scipy.stats import gaussian_kde
        kde_unfaithful = gaussian_kde(unfaithful_lengths)
        x_range = np.linspace(min(faithful_lengths + unfaithful_lengths),
                            max(faithful_lengths + unfaithful_lengths), 200)
        plt.plot(x_range, kde_unfaithful(x_range) * len(unfaithful_lengths) *
                (max(unfaithful_lengths) - min(unfaithful_lengths)) / 20,
                'r-', linewidth=2, alpha=0.8)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(faithful_lengths, unfaithful_lengths)

    plt.xlabel(x_label)
    plt.ylabel('Count')
    plt.title(f"{title}\nt-test: t={t_stat:.3f}, p={p_value:.3e}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze length effects in faithfulness results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/faithfulness_multishot"),
        help="Directory containing faithfulness results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/faithfulness_multishot/length_analysis"),
        help="Output directory for analysis results"
    )
    args = parser.parse_args()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    logger = setup_logging(args.output_dir / "length_analysis.log")
    logger.info("=" * 80)
    logger.info("Length Analysis for Faithfulness Results")
    logger.info("=" * 80)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Find all faithfulness result files
    result_files = list(args.results_dir.glob("*_faithfulness.jsonl"))
    logger.info(f"Found {len(result_files)} result files")

    # Analyze each file
    all_analyses = []
    for file_path in result_files:
        analysis = analyze_faithfulness_file(file_path, logger)
        if analysis:
            all_analyses.append(analysis)

    logger.info(f"Successfully analyzed {len(all_analyses)} files")

    # Aggregate statistics
    logger.info("\n" + "=" * 80)
    logger.info("Computing aggregate statistics")
    logger.info("=" * 80)

    # Articulation length vs faithfulness
    articulation_word_count_vs_faithfulness = []
    articulation_char_count_vs_faithfulness = []

    # Test example length distributions
    all_faithful_word_counts = []
    all_unfaithful_word_counts = []
    all_faithful_char_counts = []
    all_unfaithful_char_counts = []

    # Per-test correlations
    test_word_count_vs_faithful = []

    for analysis in all_analyses:
        cf_faithfulness = analysis["faithfulness_metrics"]["counterfactual_faithfulness"]
        if cf_faithfulness is not None:
            articulation_word_count_vs_faithfulness.append(
                (analysis["articulation_stats"]["word_count"], cf_faithfulness)
            )
            articulation_char_count_vs_faithfulness.append(
                (analysis["articulation_stats"]["char_count"], cf_faithfulness)
            )

        # Aggregate test lengths
        for test_stats in analysis["faithful_test_lengths"]:
            all_faithful_word_counts.append(test_stats["word_count"])
            all_faithful_char_counts.append(test_stats["char_count"])

        for test_stats in analysis["unfaithful_test_lengths"]:
            all_unfaithful_word_counts.append(test_stats["word_count"])
            all_unfaithful_char_counts.append(test_stats["char_count"])

        # Individual test correlations (within-rule)
        for test_stats in analysis["test_lengths"]:
            test_word_count_vs_faithful.append(test_stats["word_count"])

    # Compute correlations
    logger.info("\nCorrelation: Articulation Length vs Counterfactual Faithfulness")
    corr_artic_word = compute_correlations(articulation_word_count_vs_faithfulness, logger)
    logger.info(f"  Word count: Pearson r={corr_artic_word.get('pearson_r', 'N/A'):.3f}, "
                f"p={corr_artic_word.get('pearson_p', 'N/A'):.3e}")

    corr_artic_char = compute_correlations(articulation_char_count_vs_faithfulness, logger)
    logger.info(f"  Char count: Pearson r={corr_artic_char.get('pearson_r', 'N/A'):.3f}, "
                f"p={corr_artic_char.get('pearson_p', 'N/A'):.3e}")

    # Test length distributions
    logger.info("\nTest Example Length Distributions:")
    logger.info(f"  Faithful tests: {len(all_faithful_word_counts)} examples")
    logger.info(f"    Word count: mean={np.mean(all_faithful_word_counts):.1f}, "
                f"std={np.std(all_faithful_word_counts):.1f}")
    logger.info(f"  Unfaithful tests: {len(all_unfaithful_word_counts)} examples")
    logger.info(f"    Word count: mean={np.mean(all_unfaithful_word_counts):.1f}, "
                f"std={np.std(all_unfaithful_word_counts):.1f}")

    # Statistical test
    if all_faithful_word_counts and all_unfaithful_word_counts:
        t_stat, p_value = stats.ttest_ind(all_faithful_word_counts, all_unfaithful_word_counts)
        logger.info(f"\n  t-test (faithful vs unfaithful word counts): t={t_stat:.3f}, p={p_value:.3e}")

    # Create visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Creating visualizations")
    logger.info("=" * 80)

    # 1. Articulation word count vs faithfulness
    create_scatter_plot(
        articulation_word_count_vs_faithfulness,
        "Articulation Word Count",
        "Counterfactual Faithfulness",
        "Articulation Length vs Faithfulness",
        figures_dir / "articulation_words_vs_faithfulness.png",
        logger
    )

    # 2. Articulation char count vs faithfulness
    create_scatter_plot(
        articulation_char_count_vs_faithfulness,
        "Articulation Character Count",
        "Counterfactual Faithfulness",
        "Articulation Length (chars) vs Faithfulness",
        figures_dir / "articulation_chars_vs_faithfulness.png",
        logger
    )

    # 3. Test length distribution comparison
    create_distribution_comparison(
        all_faithful_word_counts,
        all_unfaithful_word_counts,
        "Test Example Word Count",
        "Faithful vs Unfaithful Test Example Lengths",
        figures_dir / "test_length_distribution_words.png",
        logger
    )

    create_distribution_comparison(
        all_faithful_char_counts,
        all_unfaithful_char_counts,
        "Test Example Character Count",
        "Faithful vs Unfaithful Test Example Lengths (chars)",
        figures_dir / "test_length_distribution_chars.png",
        logger
    )

    # Save summary statistics
    summary = {
        "num_files_analyzed": len(all_analyses),
        "articulation_length_vs_faithfulness": {
            "word_count": corr_artic_word,
            "char_count": corr_artic_char
        },
        "test_length_distributions": {
            "faithful": {
                "word_count": {
                    "n": len(all_faithful_word_counts),
                    "mean": float(np.mean(all_faithful_word_counts)) if all_faithful_word_counts else None,
                    "std": float(np.std(all_faithful_word_counts)) if all_faithful_word_counts else None,
                    "min": float(np.min(all_faithful_word_counts)) if all_faithful_word_counts else None,
                    "max": float(np.max(all_faithful_word_counts)) if all_faithful_word_counts else None
                },
                "char_count": {
                    "n": len(all_faithful_char_counts),
                    "mean": float(np.mean(all_faithful_char_counts)) if all_faithful_char_counts else None,
                    "std": float(np.std(all_faithful_char_counts)) if all_faithful_char_counts else None
                }
            },
            "unfaithful": {
                "word_count": {
                    "n": len(all_unfaithful_word_counts),
                    "mean": float(np.mean(all_unfaithful_word_counts)) if all_unfaithful_word_counts else None,
                    "std": float(np.std(all_unfaithful_word_counts)) if all_unfaithful_word_counts else None,
                    "min": float(np.min(all_unfaithful_word_counts)) if all_unfaithful_word_counts else None,
                    "max": float(np.max(all_unfaithful_word_counts)) if all_unfaithful_word_counts else None
                },
                "char_count": {
                    "n": len(all_unfaithful_char_counts),
                    "mean": float(np.mean(all_unfaithful_char_counts)) if all_unfaithful_char_counts else None,
                    "std": float(np.std(all_unfaithful_char_counts)) if all_unfaithful_char_counts else None
                }
            }
        }
    }

    if all_faithful_word_counts and all_unfaithful_word_counts:
        t_stat, p_value = stats.ttest_ind(all_faithful_word_counts, all_unfaithful_word_counts)
        summary["test_length_distributions"]["t_test_word_count"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value)
        }

    summary_path = args.output_dir / "length_statistics.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    logger.info(f"\nSaved summary statistics: {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Length analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
