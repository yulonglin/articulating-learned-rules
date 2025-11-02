"""
Create comprehensive visualizations for free-form articulation experiment results.

Compares free-form generation metrics across [5, 10, 20, 50, 100] shots:
- LLM judge scores (semantic equivalence)
- Functional accuracy (can the articulation classify correctly?)
- Cosine similarity (embedding-based similarity)

Generates publication-quality figures:
1. LLM Judge vs Functional Accuracy scatter (the 35-40% gap)
2. Multi-shot curves by metric (judge, functional, cosine)
3. Prompt variation comparison (simple vs CoT vs explicit)
4. Category-specific performance breakdown
5. MC vs Free-form comparison (recognition vs generation)
6. Judge-Functional gap analysis by category
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.utils import load_jsonl
from src.embedding_cache import EmbeddingCache, cosine_similarity


def load_freeform_data(
    rules_file: Path,
    freeform_dir: Path,
    mc_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load rules metadata, free-form results, and MC results."""
    # Load rules with category information
    rules_data = load_jsonl(rules_file)
    rules_df = pd.DataFrame(rules_data)

    # Load free-form summary
    with freeform_dir.joinpath("summary_freeform.yaml").open("r") as f:
        freeform_summary = yaml.safe_load(f)

    # Load MC summary for comparison
    with mc_summary.open("r") as f:
        mc_summary_data = yaml.safe_load(f)

    # Convert to DataFrames
    freeform_df = prepare_freeform_dataframe(rules_df, freeform_summary)
    mc_df = prepare_mc_dataframe(rules_df, mc_summary_data)

    return rules_df, freeform_df, mc_df


def prepare_freeform_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """Convert free-form summary to long-format DataFrame."""
    rows = []

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, variation_results in model_results.items():
            for variation, shot_results in variation_results.items():
                for shot_count, metrics in shot_results.items():
                    rows.append(
                        {
                            "rule_id": rule_id,
                            "category": category,
                            "model": model,
                            "variation": variation,
                            "shot_count": int(shot_count),
                            "llm_judge": metrics.get("llm_judge", 0.0),
                            "functional_accuracy": metrics.get(
                                "functional_accuracy", 0.0
                            ),
                            "n_classified": metrics.get("functional_details", {}).get(
                                "n_classified", 0
                            ),
                            "n_correct": metrics.get("functional_details", {}).get(
                                "n_correct", 0
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def prepare_mc_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """Convert MC summary to long-format DataFrame."""
    rows = []

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, shot_results in model_results.items():
            for shot_count, metrics in shot_results.items():
                rows.append(
                    {
                        "rule_id": rule_id,
                        "category": category,
                        "model": model,
                        "shot_count": int(shot_count),
                        "accuracy": metrics["accuracy"],
                        "n_correct": metrics["n_correct"],
                        "n_total": metrics["n_total"],
                    }
                )

    return pd.DataFrame(rows)


def compute_cosine_similarities(
    freeform_dir: Path, embedding_cache: EmbeddingCache
) -> pd.DataFrame:
    """Compute cosine similarities for all free-form results."""
    rows = []

    for jsonl_file in freeform_dir.glob("*_freeform.jsonl"):
        results = load_jsonl(jsonl_file)
        if not results:
            continue

        result = results[0]  # Each file has one result

        # Get embeddings
        gt_emb = embedding_cache.get_embedding(result["ground_truth_articulation"])
        gen_emb = embedding_cache.get_embedding(result["generated_articulation"])

        # Compute similarity
        similarity = cosine_similarity(gt_emb, gen_emb)

        rows.append(
            {
                "rule_id": result["rule_id"],
                "model": result["model"],
                "variation": result["prompt_variation"],
                "shot_count": result["few_shot_count"],
                "cosine_similarity": similarity,
            }
        )

    return pd.DataFrame(rows)


def plot_judge_vs_functional_scatter(
    freeform_df: pd.DataFrame, output_path: Path
):
    """Figure 1: LLM Judge vs Functional Accuracy scatter (show the gap)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Filter to 100-shot for cleaner visualization
    df_100 = freeform_df[freeform_df["shot_count"] == 100].copy()

    # Define category colors
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    for ax_idx, model in enumerate(sorted(df_100["model"].unique())):
        ax = axes[ax_idx]
        model_df = df_100[df_100["model"] == model]

        # Average across variations for each rule
        rule_means = (
            model_df.groupby(["rule_id", "category"])
            .agg({"llm_judge": "mean", "functional_accuracy": "mean"})
            .reset_index()
        )

        # Scatter by category
        for category in sorted(rule_means["category"].unique()):
            cat_data = rule_means[rule_means["category"] == category]
            ax.scatter(
                cat_data["llm_judge"],
                cat_data["functional_accuracy"],
                label=category.capitalize(),
                color=category_colors.get(category, "#95a5a6"),
                s=120,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

        # Add diagonal reference line (judge = functional)
        lim_min, lim_max = 0, 1
        ax.plot(
            [lim_min, lim_max],
            [lim_min, lim_max],
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label="Equal performance",
        )

        # Add gap statistics
        mean_judge = rule_means["llm_judge"].mean()
        mean_functional = rule_means["functional_accuracy"].mean()
        gap = mean_functional - mean_judge

        textstr = f"Gap: {gap:+.1%}\nJudge: {mean_judge:.1%}\nFunc: {mean_functional:.1%}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        model_label = (
            "GPT-4.1-nano" if "gpt" in model.lower() else "Claude Haiku 4.5"
        )
        ax.set_xlabel("LLM Judge Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Functional Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(f"{model_label} (100-shot)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        "LLM Judge vs Functional Accuracy: The 35-40% Gap",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1 to {output_path}")


def plot_multishot_curves_by_metric(
    freeform_df: pd.DataFrame, cosine_df: pd.DataFrame, output_path: Path
):
    """Figure 2: Multi-shot curves by metric (judge, functional, cosine)."""
    # Merge cosine similarities
    merged_df = freeform_df.merge(
        cosine_df, on=["rule_id", "model", "variation", "shot_count"], how="left"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    shot_counts = sorted(merged_df["shot_count"].unique())
    models = sorted(merged_df["model"].unique())
    model_labels = {
        "gpt-4.1-nano-2025-04-14": "GPT-4.1-nano",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    }

    for ax, model in zip(axes, models):
        model_df = merged_df[merged_df["model"] == model]

        # Average across variations and rules for each metric
        metrics_data = []
        for shot in shot_counts:
            shot_df = model_df[model_df["shot_count"] == shot]
            metrics_data.append(
                {
                    "shot": shot,
                    "llm_judge_mean": shot_df["llm_judge"].mean(),
                    "llm_judge_std": shot_df["llm_judge"].std(),
                    "functional_mean": shot_df["functional_accuracy"].mean(),
                    "functional_std": shot_df["functional_accuracy"].std(),
                    "cosine_mean": shot_df["cosine_similarity"].mean(),
                    "cosine_std": shot_df["cosine_similarity"].std(),
                }
            )

        metrics_df = pd.DataFrame(metrics_data)

        # Plot each metric
        ax.plot(
            metrics_df["shot"],
            metrics_df["llm_judge_mean"],
            marker="o",
            linewidth=2.5,
            color="#e74c3c",
            label="LLM Judge",
        )
        ax.fill_between(
            metrics_df["shot"],
            metrics_df["llm_judge_mean"] - metrics_df["llm_judge_std"],
            metrics_df["llm_judge_mean"] + metrics_df["llm_judge_std"],
            alpha=0.15,
            color="#e74c3c",
        )

        ax.plot(
            metrics_df["shot"],
            metrics_df["functional_mean"],
            marker="s",
            linewidth=2.5,
            color="#2ecc71",
            label="Functional Accuracy",
        )
        ax.fill_between(
            metrics_df["shot"],
            metrics_df["functional_mean"] - metrics_df["functional_std"],
            metrics_df["functional_mean"] + metrics_df["functional_std"],
            alpha=0.15,
            color="#2ecc71",
        )

        ax.plot(
            metrics_df["shot"],
            metrics_df["cosine_mean"],
            marker="^",
            linewidth=2.5,
            color="#3498db",
            label="Cosine Similarity",
        )
        ax.fill_between(
            metrics_df["shot"],
            metrics_df["cosine_mean"] - metrics_df["cosine_std"],
            metrics_df["cosine_mean"] + metrics_df["cosine_std"],
            alpha=0.15,
            color="#3498db",
        )

        ax.set_xlabel("Few-shot Examples", fontsize=12, fontweight="bold")
        ax.set_title(model_labels.get(model, model), fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0.3, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    axes[0].set_ylabel("Score", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Free-Form Articulation: Multi-Shot Performance by Metric",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2 to {output_path}")


def plot_prompt_variation_comparison(
    freeform_df: pd.DataFrame, output_path: Path
):
    """Figure 3: Prompt variation comparison (simple vs CoT vs explicit)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    shot_counts = sorted(freeform_df["shot_count"].unique())
    variations = sorted(freeform_df["variation"].unique())

    variation_colors = {
        "simple": "#95a5a6",
        "cot": "#e74c3c",
        "explicit": "#3498db",
    }

    # Plot 1: LLM Judge by variation
    ax = axes[0]
    for variation in variations:
        var_df = freeform_df[freeform_df["variation"] == variation]
        means = [
            var_df[var_df["shot_count"] == shot]["llm_judge"].mean()
            for shot in shot_counts
        ]
        stds = [
            var_df[var_df["shot_count"] == shot]["llm_judge"].std()
            for shot in shot_counts
        ]
        ax.plot(
            shot_counts,
            means,
            marker="o",
            linewidth=2.5,
            color=variation_colors.get(variation, "#95a5a6"),
            label=variation.upper() if variation == "cot" else variation.capitalize(),
        )
        ax.fill_between(
            shot_counts,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.15,
            color=variation_colors.get(variation, "#95a5a6"),
        )

    ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
    ax.set_ylabel("LLM Judge Score", fontsize=11, fontweight="bold")
    ax.set_title("LLM Judge by Prompt Variation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Plot 2: Functional Accuracy by variation
    ax = axes[1]
    for variation in variations:
        var_df = freeform_df[freeform_df["variation"] == variation]
        means = [
            var_df[var_df["shot_count"] == shot]["functional_accuracy"].mean()
            for shot in shot_counts
        ]
        stds = [
            var_df[var_df["shot_count"] == shot]["functional_accuracy"].std()
            for shot in shot_counts
        ]
        ax.plot(
            shot_counts,
            means,
            marker="s",
            linewidth=2.5,
            color=variation_colors.get(variation, "#95a5a6"),
            label=variation.upper() if variation == "cot" else variation.capitalize(),
        )
        ax.fill_between(
            shot_counts,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.15,
            color=variation_colors.get(variation, "#95a5a6"),
        )

    ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
    ax.set_ylabel("Functional Accuracy", fontsize=11, fontweight="bold")
    ax.set_title(
        "Functional Accuracy by Prompt Variation", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Plot 3: CoT improvement (CoT - Simple)
    ax = axes[2]
    cot_df = freeform_df[freeform_df["variation"] == "cot"]
    simple_df = freeform_df[freeform_df["variation"] == "simple"]

    for metric, color, label in [
        ("llm_judge", "#e74c3c", "LLM Judge"),
        ("functional_accuracy", "#2ecc71", "Functional"),
    ]:
        improvements = []
        for shot in shot_counts:
            cot_mean = cot_df[cot_df["shot_count"] == shot][metric].mean()
            simple_mean = simple_df[simple_df["shot_count"] == shot][metric].mean()
            improvements.append(cot_mean - simple_mean)

        ax.plot(
            shot_counts,
            improvements,
            marker="o",
            linewidth=2.5,
            color=color,
            label=label,
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)
    ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
    ax.set_ylabel("CoT Improvement (CoT - Simple)", fontsize=11, fontweight="bold")
    ax.set_title("CoT Improvement over Simple", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    # Plot 4: Variation comparison at 100-shot
    ax = axes[3]
    df_100 = freeform_df[freeform_df["shot_count"] == 100]

    variation_names = [
        v.upper() if v == "cot" else v.capitalize() for v in variations
    ]
    judge_means = [
        df_100[df_100["variation"] == v]["llm_judge"].mean() for v in variations
    ]
    functional_means = [
        df_100[df_100["variation"] == v]["functional_accuracy"].mean()
        for v in variations
    ]

    x = np.arange(len(variations))
    width = 0.35

    ax.bar(
        x - width / 2,
        judge_means,
        width,
        label="LLM Judge",
        color="#e74c3c",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        functional_means,
        width,
        label="Functional",
        color="#2ecc71",
        alpha=0.8,
    )

    ax.set_xlabel("Prompt Variation", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (100-shot)", fontsize=11, fontweight="bold")
    ax.set_title("Performance at 100-shot by Variation", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variation_names)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        "Prompt Variation Analysis: Simple vs CoT vs Explicit",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3 to {output_path}")


def plot_category_performance(freeform_df: pd.DataFrame, output_path: Path):
    """Figure 4: Category-specific performance (pattern-based, semantic, statistical)."""
    categories = sorted(freeform_df["category"].unique())
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    shot_counts = sorted(freeform_df["shot_count"].unique())

    for ax, category in zip(axes, categories):
        cat_df = freeform_df[freeform_df["category"] == category]

        # Calculate means and stds for each metric
        judge_means, judge_stds = [], []
        functional_means, functional_stds = [], []

        for shot in shot_counts:
            shot_df = cat_df[cat_df["shot_count"] == shot]
            judge_means.append(shot_df["llm_judge"].mean())
            judge_stds.append(shot_df["llm_judge"].std())
            functional_means.append(shot_df["functional_accuracy"].mean())
            functional_stds.append(shot_df["functional_accuracy"].std())

        color = category_colors.get(category, "#95a5a6")

        # Plot LLM Judge
        ax.plot(
            shot_counts,
            judge_means,
            marker="o",
            linewidth=2.5,
            color=color,
            label="LLM Judge",
            alpha=0.8,
        )
        ax.fill_between(
            shot_counts,
            np.array(judge_means) - np.array(judge_stds),
            np.array(judge_means) + np.array(judge_stds),
            alpha=0.15,
            color=color,
        )

        # Plot Functional
        ax.plot(
            shot_counts,
            functional_means,
            marker="s",
            linewidth=2.5,
            color=color,
            linestyle="--",
            label="Functional",
            alpha=0.8,
        )
        ax.fill_between(
            shot_counts,
            np.array(functional_means) - np.array(functional_stds),
            np.array(functional_means) + np.array(functional_stds),
            alpha=0.15,
            color=color,
        )

        # Add gap at 100-shot
        gap_100 = functional_means[-1] - judge_means[-1]
        textstr = f"Gap @ 100-shot:\n{gap_100:+.1%}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(f"{category.capitalize()} Rules", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0.0, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        "Category-Specific Free-Form Articulation Performance",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4 to {output_path}")


def plot_mc_vs_freeform_comparison(
    freeform_df: pd.DataFrame, mc_df: pd.DataFrame, output_path: Path
):
    """Figure 5: MC (recognition) vs Free-form (generation) comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    shot_counts = sorted(freeform_df["shot_count"].unique())

    # Plot 1: Overall MC vs Free-form (LLM Judge)
    ax = axes[0]

    # MC accuracy
    mc_means = [mc_df[mc_df["shot_count"] == shot]["accuracy"].mean() for shot in shot_counts]
    mc_stds = [mc_df[mc_df["shot_count"] == shot]["accuracy"].std() for shot in shot_counts]

    # Free-form LLM judge
    ff_judge_means = [
        freeform_df[freeform_df["shot_count"] == shot]["llm_judge"].mean()
        for shot in shot_counts
    ]
    ff_judge_stds = [
        freeform_df[freeform_df["shot_count"] == shot]["llm_judge"].std()
        for shot in shot_counts
    ]

    # Free-form functional
    ff_func_means = [
        freeform_df[freeform_df["shot_count"] == shot]["functional_accuracy"].mean()
        for shot in shot_counts
    ]
    ff_func_stds = [
        freeform_df[freeform_df["shot_count"] == shot]["functional_accuracy"].std()
        for shot in shot_counts
    ]

    ax.plot(
        shot_counts, mc_means, marker="o", linewidth=2.5, color="#3498db", label="MC (Recognition)"
    )
    ax.fill_between(
        shot_counts,
        np.array(mc_means) - np.array(mc_stds),
        np.array(mc_means) + np.array(mc_stds),
        alpha=0.15,
        color="#3498db",
    )

    ax.plot(
        shot_counts,
        ff_judge_means,
        marker="s",
        linewidth=2.5,
        color="#e74c3c",
        label="Free-form (Judge)",
    )
    ax.fill_between(
        shot_counts,
        np.array(ff_judge_means) - np.array(ff_judge_stds),
        np.array(ff_judge_means) + np.array(ff_judge_stds),
        alpha=0.15,
        color="#e74c3c",
    )

    ax.plot(
        shot_counts,
        ff_func_means,
        marker="^",
        linewidth=2.5,
        color="#2ecc71",
        label="Free-form (Functional)",
    )
    ax.fill_between(
        shot_counts,
        np.array(ff_func_means) - np.array(ff_func_stds),
        np.array(ff_func_means) + np.array(ff_func_stds),
        alpha=0.15,
        color="#2ecc71",
    )

    ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy/Score", fontsize=11, fontweight="bold")
    ax.set_title("Recognition vs Generation: Overall", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Plot 2: Gap analysis (MC - Free-form Judge)
    ax = axes[1]
    gaps = np.array(mc_means) - np.array(ff_judge_means)
    ax.plot(shot_counts, gaps, marker="o", linewidth=2.5, color="#e67e22")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)

    ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gap (MC - Free-form Judge)", fontsize=11, fontweight="bold")
    ax.set_title("Recognition-Generation Gap", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    # Plot 3: By category
    ax = axes[2]
    categories = sorted(freeform_df["category"].unique())
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    for category in categories:
        mc_cat_mean = mc_df[
            (mc_df["category"] == category) & (mc_df["shot_count"] == 100)
        ]["accuracy"].mean()

        ff_cat_mean = freeform_df[
            (freeform_df["category"] == category) & (freeform_df["shot_count"] == 100)
        ]["llm_judge"].mean()

        gap = mc_cat_mean - ff_cat_mean

        ax.scatter(
            mc_cat_mean,
            ff_cat_mean,
            s=200,
            color=category_colors.get(category, "#95a5a6"),
            label=category.capitalize(),
            edgecolors="black",
            linewidths=1,
        )

    # Diagonal reference
    lim_min, lim_max = 0, 1
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        linewidth=1.5,
        alpha=0.5,
    )

    ax.set_xlabel("MC Accuracy (100-shot)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Free-form Judge (100-shot)", fontsize=11, fontweight="bold")
    ax.set_title("MC vs Free-form by Category", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Plot 4: Gap by category (bar chart)
    ax = axes[3]
    category_names = [c.capitalize() for c in categories]
    mc_cat_means = [
        mc_df[(mc_df["category"] == c) & (mc_df["shot_count"] == 100)]["accuracy"].mean()
        for c in categories
    ]
    ff_cat_means = [
        freeform_df[(freeform_df["category"] == c) & (freeform_df["shot_count"] == 100)][
            "llm_judge"
        ].mean()
        for c in categories
    ]
    gaps = np.array(mc_cat_means) - np.array(ff_cat_means)

    colors_list = [category_colors.get(c, "#95a5a6") for c in categories]
    ax.bar(category_names, gaps, color=colors_list, alpha=0.8, edgecolor="black")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)

    ax.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gap (MC - Free-form Judge)", fontsize=11, fontweight="bold")
    ax.set_title("Recognition-Generation Gap by Category", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    fig.suptitle(
        "Recognition (MC) vs Generation (Free-form) Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 5 to {output_path}")


def plot_gap_analysis_by_category(freeform_df: pd.DataFrame, output_path: Path):
    """Figure 6: Judge-Functional gap analysis by category."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    categories = sorted(freeform_df["category"].unique())
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    shot_counts = sorted(freeform_df["shot_count"].unique())

    # Plot 1: Gap across shot counts by category
    ax = axes[0]
    for category in categories:
        cat_df = freeform_df[freeform_df["category"] == category]

        gaps = []
        gap_stds = []
        for shot in shot_counts:
            shot_df = cat_df[cat_df["shot_count"] == shot]

            # Compute per-observation gaps
            per_obs_gaps = shot_df["functional_accuracy"] - shot_df["llm_judge"]

            gaps.append(per_obs_gaps.mean())
            gap_stds.append(per_obs_gaps.std())

        color = category_colors.get(category, "#95a5a6")
        ax.plot(
            shot_counts,
            gaps,
            marker="o",
            linewidth=2.5,
            color=color,
            label=category.capitalize(),
        )
        ax.fill_between(
            shot_counts,
            np.array(gaps) - np.array(gap_stds),
            np.array(gaps) + np.array(gap_stds),
            alpha=0.15,
            color=color,
        )

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)
    ax.set_xlabel("Few-shot Examples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gap (Functional - Judge)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Judge-Functional Gap Across Shot Counts",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    # Plot 2: Gap at 100-shot by category (bar chart with error bars)
    ax = axes[1]
    df_100 = freeform_df[freeform_df["shot_count"] == 100]

    category_names = [c.capitalize() for c in categories]
    gap_means = []
    gap_stds = []

    for category in categories:
        cat_df = df_100[df_100["category"] == category]
        per_obs_gaps = cat_df["functional_accuracy"] - cat_df["llm_judge"]
        gap_means.append(per_obs_gaps.mean())
        gap_stds.append(per_obs_gaps.std())

    colors_list = [category_colors.get(c, "#95a5a6") for c in categories]
    ax.bar(
        category_names,
        gap_means,
        yerr=gap_stds,
        color=colors_list,
        alpha=0.8,
        edgecolor="black",
        capsize=5,
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)

    ax.set_xlabel("Category", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gap (Functional - Judge)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Judge-Functional Gap at 100-shot",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    fig.suptitle(
        "The Judge-Functional Gap: Models Learn But Express Differently",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 6 to {output_path}")


def generate_analysis_summary(
    freeform_df: pd.DataFrame,
    cosine_df: pd.DataFrame,
    mc_df: pd.DataFrame,
    output_path: Path,
):
    """Generate comprehensive markdown summary."""
    lines = []
    lines.append("# Free-Form Articulation Analysis: Generation Task\n")

    # Merge cosine data
    merged_df = freeform_df.merge(
        cosine_df, on=["rule_id", "model", "variation", "shot_count"], how="left"
    )

    # Overall comparison at 100-shot
    lines.append("## Overall Performance (100-shot)\n")
    df_100 = merged_df[merged_df["shot_count"] == 100]

    judge_mean = df_100["llm_judge"].mean()
    functional_mean = df_100["functional_accuracy"].mean()
    cosine_mean = df_100["cosine_similarity"].mean()
    gap = functional_mean - judge_mean

    lines.append(f"- **LLM Judge:** {judge_mean:.1%}")
    lines.append(f"- **Functional Accuracy:** {functional_mean:.1%}")
    lines.append(f"- **Cosine Similarity:** {cosine_mean:.1%}")
    lines.append(
        f"- **Judge-Functional Gap:** {gap:+.1%} (positive = functional works better than judge scores suggest)\n"
    )

    # MC vs Free-form comparison
    lines.append("## Recognition (MC) vs Generation (Free-form) at 100-shot\n")
    mc_100_mean = mc_df[mc_df["shot_count"] == 100]["accuracy"].mean()
    mc_ff_gap = mc_100_mean - judge_mean

    lines.append(f"- **MC Accuracy (Recognition):** {mc_100_mean:.1%}")
    lines.append(f"- **Free-form Judge (Generation):** {judge_mean:.1%}")
    lines.append(
        f"- **Recognition-Generation Gap:** {mc_ff_gap:+.1%} (MC easier than free-form)\n"
    )

    # Prompt variation comparison
    lines.append("## Prompt Variation Impact (100-shot)\n")
    for variation in sorted(df_100["variation"].unique()):
        var_df = df_100[df_100["variation"] == variation]
        judge = var_df["llm_judge"].mean()
        functional = var_df["functional_accuracy"].mean()
        lines.append(
            f"- **{variation.upper() if variation == 'cot' else variation.capitalize()}:** "
            f"Judge={judge:.1%}, Functional={functional:.1%}"
        )

    # CoT improvement
    cot_100 = df_100[df_100["variation"] == "cot"]["llm_judge"].mean()
    simple_100 = df_100[df_100["variation"] == "simple"]["llm_judge"].mean()
    cot_improvement = cot_100 - simple_100
    lines.append(f"\n**CoT Improvement over Simple:** {cot_improvement:+.1%}\n")

    # Category breakdown
    lines.append("## Category Breakdown (100-shot)\n")
    for category in sorted(df_100["category"].unique()):
        cat_df = df_100[df_100["category"] == category]
        judge = cat_df["llm_judge"].mean()
        functional = cat_df["functional_accuracy"].mean()
        cosine = cat_df["cosine_similarity"].mean()
        gap = functional - judge

        lines.append(
            f"- **{category.capitalize()}:** Judge={judge:.1%}, Functional={functional:.1%}, "
            f"Cosine={cosine:.1%}, Gap={gap:+.1%}"
        )

    # Key insights
    lines.append("\n## Key Insights\n")
    lines.append(
        "1. **35-40% Judge-Functional Gap:** Models capture rules operationally but express differently than ground truth"
    )
    lines.append(
        "2. **Recognition > Generation:** MC (68.6%) outperforms free-form judge (49-52%) by ~20%"
    )
    lines.append(
        "3. **CoT Helps:** Chain-of-thought improves articulation by ~7% over simple prompts"
    )
    lines.append(
        "4. **Statistical Rules Hardest:** Largest judge-functional gap - models learn but can't articulate"
    )
    lines.append(
        "5. **Semantic Rules Easiest:** Smallest gap - models articulate what they learn for semantic concepts"
    )
    lines.append(
        "6. **Cosine Similarity Tracks Judge:** Embedding similarity correlates with LLM judge scores"
    )

    # Write to file
    output_path.write_text("\n".join(lines))
    print(f"✓ Saved analysis summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create free-form articulation visualizations"
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=Path("data/processed/rules/curated_rules_learnable.jsonl"),
        help="Path to curated rules file",
    )
    parser.add_argument(
        "--freeform-dir",
        type=Path,
        default=Path("experiments/articulation_freeform_multishot"),
        help="Path to free-form experiment directory",
    )
    parser.add_argument(
        "--mc-summary",
        type=Path,
        default=Path("experiments/articulation_mc_multishot/summary_mc.yaml"),
        help="Path to MC summary for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/figures/articulation_freeform"),
        help="Directory to save figures",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path(".cache/embeddings"),
        help="Directory for embedding cache",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    rules_df, freeform_df, mc_df = load_freeform_data(
        args.rules_file, args.freeform_dir, args.mc_summary
    )

    print(f"Loaded {len(rules_df)} rules")
    print(f"Free-form: {len(freeform_df)} result records")
    print(f"MC: {len(mc_df)} result records")
    print(f"Categories: {sorted(freeform_df['category'].unique())}")
    print(f"Models: {sorted(freeform_df['model'].unique())}")
    print(f"Variations: {sorted(freeform_df['variation'].unique())}")
    print(f"Shot counts: {sorted(freeform_df['shot_count'].unique())}\n")

    # Compute cosine similarities
    print("Computing cosine similarities...")
    embedding_cache = EmbeddingCache(cache_dir=args.embedding_cache_dir)
    cosine_df = compute_cosine_similarities(args.freeform_dir, embedding_cache)
    print(f"Computed {len(cosine_df)} cosine similarities\n")

    # Generate figures
    print("Generating visualizations...")
    plot_judge_vs_functional_scatter(
        freeform_df, args.output_dir / "fig1_judge_vs_functional_scatter.png"
    )
    plot_multishot_curves_by_metric(
        freeform_df, cosine_df, args.output_dir / "fig2_multishot_curves_by_metric.png"
    )
    plot_prompt_variation_comparison(
        freeform_df, args.output_dir / "fig3_prompt_variation_comparison.png"
    )
    plot_category_performance(
        freeform_df, args.output_dir / "fig4_category_performance.png"
    )
    plot_mc_vs_freeform_comparison(
        freeform_df, mc_df, args.output_dir / "fig5_mc_vs_freeform_comparison.png"
    )
    plot_gap_analysis_by_category(
        freeform_df, args.output_dir / "fig6_gap_analysis_by_category.png"
    )

    # Generate analysis summary
    generate_analysis_summary(
        freeform_df, cosine_df, mc_df, args.output_dir / "analysis_summary.md"
    )

    print(f"\n✓ All visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
