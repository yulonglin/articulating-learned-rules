"""
Create comprehensive visualizations comparing learnability vs functional accuracy from free-form articulation.

Compares learnability curves vs functional accuracy curves across [5, 10, 20, 50, 100] shots.
Shows all 3 prompt variations (simple, CoT, explicit) where informative.

Generates 6 publication-quality figures:
1. Learnability vs Functional Accuracy curves (by model, all variations shown)
2. Category-specific comparison (4 subplots, averaged across variations)
3. Learnability-Functional gap across shot counts (by category and variation)
4. Rules where functional accuracy degrades with more examples
5. Model agreement on functional accuracy
6. Rule-level heatmap comparison (learn vs functional at 100-shot)
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


def load_data(
    rules_file: Path,
    learnability_summary: Path,
    freeform_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load rules metadata, learnability results, and freeform functional accuracy."""
    # Load rules with category information
    rules_data = load_jsonl(rules_file)
    rules_df = pd.DataFrame(rules_data)

    # Load learnability summary (keys: "few_shot_5", "few_shot_10", etc.)
    with learnability_summary.open("r") as f:
        learn_summary = yaml.safe_load(f)

    # Load freeform summary (keys: rule -> model -> variation -> shot -> metrics)
    with freeform_summary.open("r") as f:
        freeform_summary_data = yaml.safe_load(f)

    # Convert to long-format DataFrames
    learn_df = prepare_learnability_dataframe(rules_df, learn_summary)
    functional_df = prepare_functional_dataframe(rules_df, freeform_summary_data)

    return rules_df, learn_df, functional_df


def prepare_learnability_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """Convert learnability summary to long-format DataFrame with category info."""
    rows = []

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, shot_results in model_results.items():
            for shot_key, metrics in shot_results.items():
                # Extract shot count (e.g., "few_shot_5" -> 5)
                shot_count = int(str(shot_key).replace("few_shot_", ""))

                rows.append(
                    {
                        "rule_id": rule_id,
                        "category": category,
                        "model": model,
                        "shot_count": shot_count,
                        "accuracy": metrics["accuracy"],
                        "n_correct": metrics["n_correct"],
                        "n_total": metrics["n_total"],
                    }
                )

    return pd.DataFrame(rows)


def prepare_functional_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """Convert freeform summary to long-format DataFrame with functional accuracy."""
    rows = []

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, variation_results in model_results.items():
            for variation, shot_results in variation_results.items():
                for shot_key, metrics in shot_results.items():
                    shot_count = int(shot_key)

                    rows.append(
                        {
                            "rule_id": rule_id,
                            "category": category,
                            "model": model,
                            "variation": variation,
                            "shot_count": shot_count,
                            "functional_accuracy": metrics.get("functional_accuracy", 0.0),
                            "llm_judge": metrics.get("llm_judge", 0.0),
                            "n_correct": metrics.get("functional_details", {}).get("n_correct", 0),
                            "n_total": metrics.get("functional_details", {}).get("n_total", 0),
                        }
                    )

    return pd.DataFrame(rows)


def plot_learn_vs_functional_curves(
    learn_df: pd.DataFrame, functional_df: pd.DataFrame, output_path: Path
):
    """Figure 1: Learnability vs Functional Accuracy curves (by model, all variations)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    models = sorted(learn_df["model"].unique())
    model_labels = {
        "gpt-4.1-nano-2025-04-14": "GPT-4.1-nano",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    }

    shot_counts = sorted(learn_df["shot_count"].unique())
    variations = sorted(functional_df["variation"].unique())

    variation_colors = {
        "simple": "#95a5a6",
        "cot": "#e74c3c",
        "explicit": "#3498db",
    }

    for ax, model in zip(axes, models):
        learn_model = learn_df[learn_df["model"] == model]

        # Calculate learnability means and stds
        learn_means = []
        learn_stds = []
        for shot in shot_counts:
            shot_df = learn_model[learn_model["shot_count"] == shot]
            learn_means.append(shot_df["accuracy"].mean())
            learn_stds.append(shot_df["accuracy"].std())

        learn_means = np.array(learn_means)
        learn_stds = np.array(learn_stds)

        # Plot learnability
        ax.plot(
            shot_counts,
            learn_means,
            marker="o",
            linewidth=3,
            color="#2ecc71",
            label="Learnability (classification)",
            zorder=10,
        )
        ax.fill_between(
            shot_counts,
            learn_means - learn_stds,
            learn_means + learn_stds,
            alpha=0.15,
            color="#2ecc71",
        )

        # Plot functional accuracy for each variation
        functional_model = functional_df[functional_df["model"] == model]

        for variation in variations:
            var_df = functional_model[functional_model["variation"] == variation]

            var_means = []
            var_stds = []
            for shot in shot_counts:
                shot_df = var_df[var_df["shot_count"] == shot]
                var_means.append(shot_df["functional_accuracy"].mean())
                var_stds.append(shot_df["functional_accuracy"].std())

            var_means = np.array(var_means)
            var_stds = np.array(var_stds)

            var_label = variation.upper() if variation == "cot" else variation.capitalize()
            ax.plot(
                shot_counts,
                var_means,
                marker="s",
                linewidth=2.5,
                color=variation_colors.get(variation, "#95a5a6"),
                label=f"Functional ({var_label})",
                linestyle="--",
            )
            ax.fill_between(
                shot_counts,
                var_means - var_stds,
                var_means + var_stds,
                alpha=0.1,
                color=variation_colors.get(variation, "#95a5a6"),
            )

        # Add 90% threshold
        ax.axhline(y=0.90, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)

        ax.set_xlabel("Few-shot Examples", fontsize=12, fontweight="bold")
        ax.set_title(model_labels.get(model, model), fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0.4, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    axes[0].set_ylabel("Mean Accuracy", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Learnability vs Functional Accuracy: All Prompt Variations",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1 to {output_path}")


def plot_category_comparison(
    learn_df: pd.DataFrame, functional_df: pd.DataFrame, output_path: Path
):
    """Figure 2: Category-specific comparison (averaged across variations for clarity)."""
    categories = sorted(learn_df["category"].unique())
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    shot_counts = sorted(learn_df["shot_count"].unique())

    for ax, category in zip(axes, categories):
        # Filter data for this category
        learn_cat = learn_df[learn_df["category"] == category]
        functional_cat = functional_df[functional_df["category"] == category]

        # Calculate learnability means
        learn_means = []
        learn_stds = []
        for shot in shot_counts:
            shot_df = learn_cat[learn_cat["shot_count"] == shot]
            learn_means.append(shot_df["accuracy"].mean())
            learn_stds.append(shot_df["accuracy"].std())

        # Calculate functional means (averaged across variations)
        functional_means = []
        functional_stds = []
        for shot in shot_counts:
            shot_df = functional_cat[functional_cat["shot_count"] == shot]
            functional_means.append(shot_df["functional_accuracy"].mean())
            functional_stds.append(shot_df["functional_accuracy"].std())

        learn_means = np.array(learn_means)
        learn_stds = np.array(learn_stds)
        functional_means = np.array(functional_means)
        functional_stds = np.array(functional_stds)

        color = category_colors.get(category, "#95a5a6")

        # Plot learnability
        ax.plot(
            shot_counts,
            learn_means,
            marker="o",
            linewidth=2.5,
            color=color,
            label="Learnability",
            alpha=0.8,
        )
        ax.fill_between(
            shot_counts,
            learn_means - learn_stds,
            learn_means + learn_stds,
            alpha=0.1,
            color=color,
        )

        # Plot functional
        ax.plot(
            shot_counts,
            functional_means,
            marker="s",
            linewidth=2.5,
            color=color,
            linestyle="--",
            label="Functional (avg)",
            alpha=0.8,
        )
        ax.fill_between(
            shot_counts,
            functional_means - functional_stds,
            functional_means + functional_stds,
            alpha=0.1,
            color=color,
        )

        # Add 90% threshold
        ax.axhline(y=0.90, color="gray", linestyle=":", linewidth=1.5, alpha=0.4)

        # Add gap at 100-shot
        gap_100 = learn_means[-1] - functional_means[-1]
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
        ax.set_ylabel("Mean Accuracy", fontsize=11, fontweight="bold")
        ax.set_title(f"{category.capitalize()} Rules", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0.3, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        "Category-Specific: Learnability vs Functional Accuracy",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2 to {output_path}")


def plot_gap_across_shots(
    learn_df: pd.DataFrame, functional_df: pd.DataFrame, output_path: Path
):
    """Figure 3: Learnability-Functional gap across shot counts (by category and variation)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    categories = sorted(learn_df["category"].unique())
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    shot_counts = sorted(learn_df["shot_count"].unique())
    variations = sorted(functional_df["variation"].unique())

    variation_styles = {
        "simple": "-",
        "cot": "--",
        "explicit": ":",
    }

    for ax, variation in zip(axes, variations):
        for category in categories:
            learn_cat = learn_df[learn_df["category"] == category]
            functional_cat = functional_df[
                (functional_df["category"] == category)
                & (functional_df["variation"] == variation)
            ]

            # Compute per-rule gaps
            gap_means = []
            gap_stds = []
            for shot in shot_counts:
                learn_shot = learn_cat[learn_cat["shot_count"] == shot]
                functional_shot = functional_cat[functional_cat["shot_count"] == shot]

                # Merge on rule_id and model
                merged = learn_shot.merge(
                    functional_shot,
                    on=["rule_id", "model"],
                    suffixes=("_learn", "_func"),
                )

                # Compute per-observation gaps
                per_rule_gaps = merged["accuracy"] - merged["functional_accuracy"]

                gap_means.append(per_rule_gaps.mean())
                gap_stds.append(per_rule_gaps.std())

            gap_means = np.array(gap_means)
            gap_stds = np.array(gap_stds)

            color = category_colors.get(category, "#95a5a6")
            linestyle = variation_styles.get(variation, "-")

            ax.plot(
                shot_counts,
                gap_means,
                marker="o",
                linewidth=2.5,
                color=color,
                linestyle=linestyle,
                label=category.capitalize(),
            )

            # Add error bars (±1 SD)
            ax.fill_between(
                shot_counts,
                gap_means - gap_stds,
                gap_means + gap_stds,
                alpha=0.15,
                color=color,
            )

        # Add zero line
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)

        ax.set_xlabel("Few-shot Examples", fontsize=12, fontweight="bold")
        var_label = variation.upper() if variation == "cot" else variation.capitalize()
        ax.set_title(f"{var_label} Prompt", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    axes[0].set_ylabel("Gap (Learnability - Functional)", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Learnability-Functional Gap Across Shot Counts by Prompt Variation",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3 to {output_path}")


def plot_degrading_functional_rules(functional_df: pd.DataFrame, output_path: Path):
    """Figure 4: Rules where functional accuracy degrades with more examples."""
    # Find rules where functional at 100-shot < functional at 5-shot
    # Use best variation per rule for this analysis
    best_variation_df = (
        functional_df.groupby(["rule_id", "model", "shot_count"])
        .agg({"functional_accuracy": "max", "variation": "first", "category": "first"})
        .reset_index()
    )

    shot_5 = best_variation_df[best_variation_df["shot_count"] == 5].set_index(
        ["rule_id", "model"]
    )
    shot_100 = best_variation_df[best_variation_df["shot_count"] == 100].set_index(
        ["rule_id", "model"]
    )

    # Calculate degradation
    degradation = shot_5["functional_accuracy"] - shot_100["functional_accuracy"]
    degradation = degradation[degradation > 0.1]  # Filter to >10% degradation

    if len(degradation) == 0:
        print("⚠ No rules show significant functional accuracy degradation (>10%)")
        return

    # Get top degrading rules
    top_degrading = degradation.nlargest(min(10, len(degradation)))
    n_plots = len(top_degrading)

    # Dynamically determine subplot layout
    if n_plots <= 3:
        n_rows, n_cols = 1, n_plots
    elif n_plots <= 6:
        n_rows, n_cols = 2, 3
    elif n_plots <= 8:
        n_rows, n_cols = 2, 4
    else:
        n_rows, n_cols = 2, 5

    # Plot functional curves for these rules
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    shot_counts = sorted(functional_df["shot_count"].unique())

    for ax, (rule_model, deg_value) in zip(axes, top_degrading.items()):
        rule_id, model = rule_model
        rule_data = best_variation_df[
            (best_variation_df["rule_id"] == rule_id)
            & (best_variation_df["model"] == model)
        ]

        # Get category for color
        category = rule_data["category"].iloc[0]
        category_colors = {
            "syntactic": "#2ecc71",
            "pattern": "#e67e22",
            "semantic": "#9b59b6",
            "statistical": "#f39c12",
        }
        color = category_colors.get(category, "#95a5a6")

        # Plot functional curve
        accuracies = [
            rule_data[rule_data["shot_count"] == shot]["functional_accuracy"].values[0]
            for shot in shot_counts
        ]

        ax.plot(shot_counts, accuracies, marker="o", linewidth=2, color=color)
        ax.set_title(
            f"{rule_id[:30]}...\n({model.split('-')[0]}) Δ={deg_value:.1%}",
            fontsize=8,
        )
        ax.set_xlabel("Shots", fontsize=8)
        ax.set_ylabel("Functional Accuracy", fontsize=8)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    # Update title to reflect actual number
    title = f"Rules Where Functional Accuracy DEGRADES with More Examples ({n_plots} total)"
    if n_plots == 10:
        title = "Top 10 Rules Where Functional Accuracy DEGRADES with More Examples"

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4 to {output_path}")


def plot_model_agreement_functional(functional_df: pd.DataFrame, output_path: Path):
    """Figure 5: Model agreement on functional accuracy (100-shot, best variation)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get 100-shot data with best variation per rule-model
    df_100 = functional_df[functional_df["shot_count"] == 100].copy()
    best_variation_df = (
        df_100.groupby(["rule_id", "model", "category"])
        .agg({"functional_accuracy": "max"})
        .reset_index()
    )

    pivot = best_variation_df.pivot_table(
        index=["category", "rule_id"], columns="model", values="functional_accuracy"
    ).reset_index()

    gpt_col = "gpt-4.1-nano-2025-04-14"
    claude_col = "claude-haiku-4-5-20251001"

    # Filter to only rules with data from both models
    pivot_filtered = pivot.dropna(subset=[gpt_col, claude_col])

    gpt_acc = pivot_filtered[gpt_col].values
    claude_acc = pivot_filtered[claude_col].values
    categories = pivot_filtered["category"].values

    # Calculate correlation
    correlation = np.corrcoef(gpt_acc, claude_acc)[0, 1]

    # Scatter plot
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    for category in sorted(set(categories)):
        mask = categories == category
        ax.scatter(
            gpt_acc[mask],
            claude_acc[mask],
            label=category.capitalize(),
            color=category_colors.get(category, "#95a5a6"),
            s=120,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Add diagonal reference line
    lim_min = 0
    lim_max = 1
    ax.plot(
        [lim_min, lim_max],
        [lim_min, lim_max],
        "k--",
        linewidth=1.5,
        alpha=0.5,
        label="Equal performance",
    )

    # Add correlation text
    textstr = f"Pearson r = {correlation:.3f}"
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel(
        "GPT-4.1-nano Functional Accuracy (100-shot)", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel(
        "Claude Haiku 4.5 Functional Accuracy (100-shot)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(
        "Model Agreement on Functional Accuracy (Best Variation)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 5 to {output_path}")


def plot_heatmap_comparison(
    learn_df: pd.DataFrame, functional_df: pd.DataFrame, output_path: Path
):
    """Figure 6: Side-by-side heatmaps (learnability vs functional at 100-shot)."""
    # Filter to 100-shot
    learn_100 = learn_df[learn_df["shot_count"] == 100].copy()
    functional_100 = functional_df[functional_df["shot_count"] == 100].copy()

    # Get best variation for functional
    functional_best = (
        functional_100.groupby(["category", "rule_id"])
        .agg({"functional_accuracy": "max"})
        .reset_index()
    )

    # Average across models for simplicity
    learn_pivot = (
        learn_100.groupby(["category", "rule_id"])["accuracy"].mean().reset_index()
    )

    # Merge
    merged = learn_pivot.merge(
        functional_best, on=["category", "rule_id"], how="inner"
    )
    merged["gap"] = merged["accuracy"] - merged["functional_accuracy"]

    # Sort by gap (descending)
    merged = merged.sort_values(["category", "gap"], ascending=[True, False])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 14))

    # Prepare data matrices
    learn_matrix = merged.set_index(["category", "rule_id"])["accuracy"]
    functional_matrix = merged.set_index(["category", "rule_id"])["functional_accuracy"]
    gap_matrix = merged.set_index(["category", "rule_id"])["gap"]

    # Plot learnability heatmap
    sns.heatmap(
        learn_matrix.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0.4,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title("Learnability (100-shot)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Rule ID", fontsize=11, fontweight="bold")

    # Plot functional heatmap
    sns.heatmap(
        functional_matrix.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.4,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
        linewidths=0.5,
        ax=axes[1],
    )
    axes[1].set_title("Functional (100-shot, best var)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    # Plot gap heatmap
    sns.heatmap(
        gap_matrix.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=0.8,
        cbar_kws={"label": "Gap (Learn - Func)"},
        linewidths=0.5,
        ax=axes[2],
    )
    axes[2].set_title("Gap (Learn - Func)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("")

    # Format rule labels
    rule_labels = [rule_id for _, rule_id in merged[["category", "rule_id"]].values]
    for ax in axes:
        ax.set_yticklabels(rule_labels, fontsize=7)

    fig.suptitle(
        "Rule-Level Comparison: Learnability vs Functional Accuracy (100-shot)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 6 to {output_path}")


def generate_comprehensive_analysis(
    learn_df: pd.DataFrame, functional_df: pd.DataFrame, output_path: Path
):
    """Generate comprehensive markdown summary comparing learnability and functional accuracy."""
    lines = []
    lines.append("# Learnability vs Functional Accuracy Analysis\n")

    # Overall comparison at 100-shot
    lines.append("## Overall Comparison (100-shot)\n")
    learn_100_mean = learn_df[learn_df["shot_count"] == 100]["accuracy"].mean()
    functional_100 = functional_df[functional_df["shot_count"] == 100]

    lines.append(f"- **Mean Learnability:** {learn_100_mean:.1%}")

    # Per-variation functional means
    for variation in sorted(functional_100["variation"].unique()):
        var_mean = functional_100[functional_100["variation"] == variation][
            "functional_accuracy"
        ].mean()
        var_label = variation.upper() if variation == "cot" else variation.capitalize()
        lines.append(f"- **Mean Functional ({var_label}):** {var_mean:.1%}")

    functional_100_mean = functional_100["functional_accuracy"].mean()
    gap_100 = learn_100_mean - functional_100_mean

    lines.append(f"- **Mean Functional (all variations):** {functional_100_mean:.1%}")
    lines.append(
        f"- **Gap:** {gap_100:+.1%} (positive = easier to learn than to articulate functionally)\n"
    )

    # Category breakdown at 100-shot
    lines.append("## Category Breakdown (100-shot)\n")
    for category in sorted(learn_df["category"].unique()):
        learn_cat = learn_df[
            (learn_df["category"] == category) & (learn_df["shot_count"] == 100)
        ]["accuracy"].mean()

        functional_cat = functional_100[functional_100["category"] == category][
            "functional_accuracy"
        ].mean()

        gap_cat = learn_cat - functional_cat

        lines.append(
            f"- **{category.capitalize()}:** Learn={learn_cat:.1%}, Functional={functional_cat:.1%}, Gap={gap_cat:+.1%}"
        )

    # Variation comparison
    lines.append("\n## Prompt Variation Performance (100-shot)\n")
    for variation in sorted(functional_100["variation"].unique()):
        var_df = functional_100[functional_100["variation"] == variation]
        var_mean = var_df["functional_accuracy"].mean()
        var_label = variation.upper() if variation == "cot" else variation.capitalize()

        # Gap per category
        category_gaps = []
        for category in sorted(var_df["category"].unique()):
            learn_cat = learn_df[
                (learn_df["category"] == category) & (learn_df["shot_count"] == 100)
            ]["accuracy"].mean()
            func_cat = var_df[var_df["category"] == category][
                "functional_accuracy"
            ].mean()
            category_gaps.append(f"{category[:4]}={learn_cat - func_cat:+.1%}")

        lines.append(f"- **{var_label}:** {var_mean:.1%} (gaps: {', '.join(category_gaps)})")

    # Gap trends across shot counts
    lines.append("\n## Gap Trends Across Shot Counts (averaged across variations)\n")
    shot_counts = sorted(learn_df["shot_count"].unique())
    for category in sorted(learn_df["category"].unique()):
        gaps = []
        for shot in shot_counts:
            learn_mean = learn_df[
                (learn_df["category"] == category) & (learn_df["shot_count"] == shot)
            ]["accuracy"].mean()
            functional_mean = functional_df[
                (functional_df["category"] == category)
                & (functional_df["shot_count"] == shot)
            ]["functional_accuracy"].mean()
            gaps.append(learn_mean - functional_mean)

        # Format gaps as string
        gaps_str = " → ".join([f"{g:+.1%}" for g in gaps])
        lines.append(f"- **{category.capitalize()}:** {gaps_str}")

    # Rules where functional degrades
    lines.append("\n## Functional Accuracy Degradation (5-shot → 100-shot)\n")
    best_variation_df = (
        functional_df.groupby(["rule_id", "model", "shot_count"])
        .agg({"functional_accuracy": "max"})
        .reset_index()
    )

    shot_5 = best_variation_df[best_variation_df["shot_count"] == 5].set_index(
        ["rule_id", "model"]
    )
    shot_100 = best_variation_df[best_variation_df["shot_count"] == 100].set_index(
        ["rule_id", "model"]
    )
    degradation = shot_5["functional_accuracy"] - shot_100["functional_accuracy"]
    degradation = degradation[degradation > 0.1]  # >10% degradation

    if len(degradation) > 0:
        top_5 = degradation.nlargest(5)
        lines.append(
            f"Found {len(degradation)} rule-model combinations where functional accuracy degrades >10%:\n"
        )
        for (rule_id, model), deg in top_5.items():
            acc_5 = shot_5.loc[(rule_id, model), "functional_accuracy"]
            acc_100 = shot_100.loc[(rule_id, model), "functional_accuracy"]
            model_short = "GPT" if "gpt" in model.lower() else "Claude"
            lines.append(
                f"- `{rule_id}` ({model_short}): {acc_5:.1%} → {acc_100:.1%} (Δ={deg:+.1%})"
            )
    else:
        lines.append(
            "- No significant degradation found (all rules improve or stay flat)"
        )

    # Model agreement
    lines.append("\n## Model Agreement on Functional Accuracy (100-shot)\n")
    best_variation_100 = (
        functional_100.groupby(["rule_id", "model"])
        .agg({"functional_accuracy": "max"})
        .reset_index()
    )
    pivot = best_variation_100.pivot_table(
        index="rule_id", columns="model", values="functional_accuracy"
    )

    # Filter to only rules with data from both models
    pivot_filtered = pivot.dropna(
        subset=["gpt-4.1-nano-2025-04-14", "claude-haiku-4-5-20251001"]
    )

    if len(pivot_filtered) > 0:
        gpt_acc = pivot_filtered["gpt-4.1-nano-2025-04-14"].values
        claude_acc = pivot_filtered["claude-haiku-4-5-20251001"].values
        correlation = np.corrcoef(gpt_acc, claude_acc)[0, 1]

        lines.append(
            f"- **Pearson correlation (n={len(pivot_filtered)} rules with both models):** r = {correlation:.3f}"
        )
        if correlation > 0.7:
            lines.append(
                "- **Interpretation:** Strong agreement - models find similar rules hard to articulate functionally"
            )
        elif correlation > 0.4:
            lines.append(
                "- **Interpretation:** Moderate agreement - some differences in functional difficulty"
            )
        else:
            lines.append(
                "- **Interpretation:** Weak agreement - models differ in which rules produce functional articulations"
            )
    else:
        lines.append("- No rules tested on both models for comparison")

    # Key insights
    lines.append("\n## Key Insights\n")
    lines.append(
        "1. **Functional accuracy measures whether articulated rules actually work for classification**"
    )
    lines.append(
        "2. **Gap between learnability and functional:** Models can classify but struggle to articulate working rules"
    )

    # Determine best variation
    best_var = None
    best_mean = 0
    for variation in sorted(functional_100["variation"].unique()):
        var_mean = functional_100[functional_100["variation"] == variation][
            "functional_accuracy"
        ].mean()
        if var_mean > best_mean:
            best_mean = var_mean
            best_var = variation

    best_var_label = best_var.upper() if best_var == "cot" else best_var.capitalize()
    lines.append(
        f"3. **Best prompt variation:** {best_var_label} achieves highest functional accuracy ({best_mean:.1%})"
    )

    # Category patterns
    category_gaps_100 = []
    for category in sorted(functional_100["category"].unique()):
        learn_cat = learn_df[
            (learn_df["category"] == category) & (learn_df["shot_count"] == 100)
        ]["accuracy"].mean()
        func_cat = functional_100[functional_100["category"] == category][
            "functional_accuracy"
        ].mean()
        gap = learn_cat - func_cat
        category_gaps_100.append((category, gap))

    category_gaps_100.sort(key=lambda x: x[1], reverse=True)
    hardest_cat, hardest_gap = category_gaps_100[0]
    easiest_cat, easiest_gap = category_gaps_100[-1]

    lines.append(
        f"4. **Hardest category:** {hardest_cat.capitalize()} (gap={hardest_gap:+.1%}) - models learn but can't articulate functional rules"
    )
    lines.append(
        f"5. **Easiest category:** {easiest_cat.capitalize()} (gap={easiest_gap:+.1%}) - functional articulations work better"
    )

    # Write to file
    output_path.write_text("\n".join(lines))
    print(f"✓ Saved comprehensive analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create functional accuracy visualizations comparing learnability and functional accuracy"
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=Path("data/processed/rules/curated_rules_learnable.jsonl"),
        help="Path to curated rules file",
    )
    parser.add_argument(
        "--learnability-summary",
        type=Path,
        default=Path("experiments/learnability/summary.yaml"),
        help="Path to learnability summary YAML",
    )
    parser.add_argument(
        "--freeform-summary",
        type=Path,
        default=Path("experiments/articulation_freeform_multishot/summary_freeform.yaml"),
        help="Path to freeform articulation summary YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/figures/articulation_functional"),
        help="Directory to save figures",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    rules_df, learn_df, functional_df = load_data(
        args.rules_file, args.learnability_summary, args.freeform_summary
    )

    print(f"Loaded {len(rules_df)} rules")
    print(f"Learnability: {len(learn_df)} result records")
    print(f"Functional: {len(functional_df)} result records")
    print(f"Categories: {sorted(learn_df['category'].unique())}")
    print(f"Models: {sorted(learn_df['model'].unique())}")
    print(f"Variations: {sorted(functional_df['variation'].unique())}")
    print(f"Shot counts: {sorted(learn_df['shot_count'].unique())}\n")

    # Generate figures
    print("Generating visualizations...")
    plot_learn_vs_functional_curves(
        learn_df, functional_df, args.output_dir / "fig1_learn_vs_functional_curves.png"
    )
    plot_category_comparison(
        learn_df, functional_df, args.output_dir / "fig2_category_comparison.png"
    )
    plot_gap_across_shots(
        learn_df, functional_df, args.output_dir / "fig3_gap_across_shots.png"
    )
    plot_degrading_functional_rules(
        functional_df, args.output_dir / "fig4_degrading_functional.png"
    )
    plot_model_agreement_functional(
        functional_df, args.output_dir / "fig5_model_agreement.png"
    )
    plot_heatmap_comparison(
        learn_df, functional_df, args.output_dir / "fig6_heatmap_comparison.png"
    )

    # Generate comprehensive analysis
    generate_comprehensive_analysis(
        learn_df, functional_df, args.output_dir / "analysis_summary.md"
    )

    print(f"\n✓ All visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
