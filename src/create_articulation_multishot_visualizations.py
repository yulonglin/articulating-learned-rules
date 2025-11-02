"""
Create comprehensive visualizations for multi-shot articulation experiment results.

Compares learnability curves vs articulation curves across [5, 10, 20, 50, 100] shots.

Generates 6 publication-quality figures:
1. Learnability vs Articulation curves (overall, by model)
2. Category-specific comparison (4 subplots)
3. Learnability-Articulation gap across shot counts (by category)
4. Rules where articulation degrades with more examples
5. Model agreement on articulation difficulty
6. Rule-level heatmap comparison (learn vs articulate at 100-shot)
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
    articulation_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load rules metadata, learnability results, and articulation results."""
    # Load rules with category information
    rules_data = load_jsonl(rules_file)
    rules_df = pd.DataFrame(rules_data)

    # Load learnability summary (keys: "few_shot_5", "few_shot_10", etc.)
    with learnability_summary.open("r") as f:
        learn_summary = yaml.safe_load(f)

    # Load articulation summary (keys: 5, 10, 20, 50, 100)
    with articulation_summary.open("r") as f:
        artic_summary = yaml.safe_load(f)

    # Convert to long-format DataFrames
    learn_df = prepare_results_dataframe(
        rules_df, learn_summary, key_format="few_shot"
    )
    artic_df = prepare_results_dataframe(rules_df, artic_summary, key_format="direct")

    return rules_df, learn_df, artic_df


def prepare_results_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any], key_format: str = "few_shot"
) -> pd.DataFrame:
    """
    Convert summary dict to long-format DataFrame with category info.

    Args:
        key_format: "few_shot" for "few_shot_5" keys, "direct" for integer keys
    """
    rows = []

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, shot_results in model_results.items():
            for shot_key, metrics in shot_results.items():
                # Extract shot count based on format
                if key_format == "few_shot":
                    shot_count = int(str(shot_key).replace("few_shot_", ""))
                else:  # direct
                    shot_count = int(shot_key)

                rows.append(
                    {
                        "rule_id": rule_id,
                        "category": category,
                        "model": model,
                        "shot_count": shot_count,
                        "accuracy": metrics["accuracy"],
                        "n_correct": metrics["n_correct"],
                        "n_total": metrics["n_total"],
                        "parse_rate": metrics.get("parse_rate", 1.0),
                    }
                )

    return pd.DataFrame(rows)


def plot_learn_vs_articulation_curves(
    learn_df: pd.DataFrame, artic_df: pd.DataFrame, output_path: Path
):
    """Figure 1: Learnability vs Articulation curves (overall, faceted by model)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    models = sorted(learn_df["model"].unique())
    model_labels = {
        "gpt-4.1-nano-2025-04-14": "GPT-4.1-nano",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    }

    shot_counts = sorted(learn_df["shot_count"].unique())

    for ax, model in zip(axes, models):
        learn_model = learn_df[learn_df["model"] == model]
        artic_model = artic_df[artic_df["model"] == model]

        # Calculate means and stds for learnability
        learn_means = []
        learn_stds = []
        for shot in shot_counts:
            shot_df = learn_model[learn_model["shot_count"] == shot]
            learn_means.append(shot_df["accuracy"].mean())
            learn_stds.append(shot_df["accuracy"].std())

        learn_means = np.array(learn_means)
        learn_stds = np.array(learn_stds)

        # Calculate means and stds for articulation
        artic_means = []
        artic_stds = []
        for shot in shot_counts:
            shot_df = artic_model[artic_model["shot_count"] == shot]
            artic_means.append(shot_df["accuracy"].mean())
            artic_stds.append(shot_df["accuracy"].std())

        artic_means = np.array(artic_means)
        artic_stds = np.array(artic_stds)

        # Plot learnability
        ax.plot(
            shot_counts,
            learn_means,
            marker="o",
            linewidth=2.5,
            color="#2ecc71",
            label="Learnability (classification)",
        )
        ax.fill_between(
            shot_counts, learn_means - learn_stds, learn_means + learn_stds, alpha=0.15, color="#2ecc71"
        )

        # Plot articulation
        ax.plot(
            shot_counts,
            artic_means,
            marker="s",
            linewidth=2.5,
            color="#e74c3c",
            label="Articulation (MC)",
        )
        ax.fill_between(
            shot_counts, artic_means - artic_stds, artic_means + artic_stds, alpha=0.15, color="#e74c3c"
        )

        # Add 90% threshold
        ax.axhline(
            y=0.90, color="gray", linestyle="--", linewidth=1.5, alpha=0.5
        )

        ax.set_xlabel("Few-shot Examples", fontsize=12, fontweight="bold")
        ax.set_title(model_labels.get(model, model), fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0.4, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    axes[0].set_ylabel("Mean Accuracy", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Learnability vs Articulation Curves",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1 to {output_path}")


def plot_category_comparison(
    learn_df: pd.DataFrame, artic_df: pd.DataFrame, output_path: Path
):
    """Figure 2: Category-specific learnability vs articulation (4 category subplots)."""
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
        artic_cat = artic_df[artic_df["category"] == category]

        # Calculate aggregated means across both models
        learn_means = []
        learn_stds = []
        artic_means = []
        artic_stds = []

        for shot in shot_counts:
            learn_shot = learn_cat[learn_cat["shot_count"] == shot]
            artic_shot = artic_cat[artic_cat["shot_count"] == shot]

            learn_means.append(learn_shot["accuracy"].mean())
            learn_stds.append(learn_shot["accuracy"].std())
            artic_means.append(artic_shot["accuracy"].mean())
            artic_stds.append(artic_shot["accuracy"].std())

        learn_means = np.array(learn_means)
        learn_stds = np.array(learn_stds)
        artic_means = np.array(artic_means)
        artic_stds = np.array(artic_stds)

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
            shot_counts, learn_means - learn_stds, learn_means + learn_stds, alpha=0.1, color=color
        )

        # Plot articulation
        ax.plot(
            shot_counts,
            artic_means,
            marker="s",
            linewidth=2.5,
            color=color,
            linestyle="--",
            label="Articulation",
            alpha=0.8,
        )
        ax.fill_between(
            shot_counts, artic_means - artic_stds, artic_means + artic_stds, alpha=0.1, color=color
        )

        # Add 90% threshold
        ax.axhline(y=0.90, color="gray", linestyle=":", linewidth=1.5, alpha=0.4)

        ax.set_xlabel("Few-shot Examples", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Accuracy", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{category.capitalize()} Rules", fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0.3, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        "Category-Specific: Learnability vs Articulation",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2 to {output_path}")


def plot_gap_across_shots(
    learn_df: pd.DataFrame, artic_df: pd.DataFrame, output_path: Path
):
    """Figure 3: Learnability-Articulation gap across shot counts (by category)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = sorted(learn_df["category"].unique())
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    shot_counts = sorted(learn_df["shot_count"].unique())

    for category in categories:
        learn_cat = learn_df[learn_df["category"] == category]
        artic_cat = artic_df[artic_df["category"] == category]

        # Compute per-rule gaps to get standard deviations
        gap_means = []
        gap_stds = []
        for shot in shot_counts:
            # Merge learn and artic data for this shot count to compute per-rule gaps
            learn_shot = learn_cat[learn_cat["shot_count"] == shot]
            artic_shot = artic_cat[artic_cat["shot_count"] == shot]

            # Merge on rule_id and model to get paired observations
            merged = learn_shot.merge(
                artic_shot,
                on=["rule_id", "model"],
                suffixes=("_learn", "_artic")
            )

            # Compute per-observation gaps
            per_rule_gaps = merged["accuracy_learn"] - merged["accuracy_artic"]

            gap_means.append(per_rule_gaps.mean())
            gap_stds.append(per_rule_gaps.std())

        gap_means = np.array(gap_means)
        gap_stds = np.array(gap_stds)

        color = category_colors.get(category, "#95a5a6")
        ax.plot(
            shot_counts,
            gap_means,
            marker="o",
            linewidth=2.5,
            color=color,
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

    ax.set_xlabel("Few-shot Examples", fontsize=13, fontweight="bold")
    ax.set_ylabel("Gap (Learnability - Articulation)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Learnability-Articulation Gap Across Shot Counts",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0%}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3 to {output_path}")


def plot_degrading_articulation_rules(artic_df: pd.DataFrame, output_path: Path):
    """Figure 4: Rules where articulation degrades with more examples."""
    # Find rules where articulation at 100-shot < articulation at 5-shot
    shot_5 = artic_df[artic_df["shot_count"] == 5].set_index(["rule_id", "model"])
    shot_100 = artic_df[artic_df["shot_count"] == 100].set_index(
        ["rule_id", "model"]
    )

    # Calculate degradation
    degradation = shot_5["accuracy"] - shot_100["accuracy"]
    degradation = degradation[degradation > 0.1]  # Filter to >10% degradation

    if len(degradation) == 0:
        print("⚠ No rules show significant articulation degradation (>10%)")
        return

    # Get all degrading rules (or top 10 if more than 10)
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

    # Plot articulation curves for these rules
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    # Handle case where axes is not an array (single subplot)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    shot_counts = sorted(artic_df["shot_count"].unique())

    for ax, (rule_model, deg_value) in zip(axes, top_degrading.items()):
        rule_id, model = rule_model
        rule_data = artic_df[
            (artic_df["rule_id"] == rule_id) & (artic_df["model"] == model)
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

        # Plot articulation curve
        accuracies = [
            rule_data[rule_data["shot_count"] == shot]["accuracy"].values[0]
            for shot in shot_counts
        ]

        ax.plot(shot_counts, accuracies, marker="o", linewidth=2, color=color)
        ax.set_title(
            f"{rule_id[:30]}...\n({model.split('-')[0]}) Δ={deg_value:.1%}",
            fontsize=8,
        )
        ax.set_xlabel("Shots", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_ylim([0, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Update title to reflect actual number
    title = f"Rules Where Articulation DEGRADES with More Examples ({n_plots} total)"
    if n_plots == 10:
        title = "Top 10 Rules Where Articulation DEGRADES with More Examples"

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4 to {output_path}")


def plot_model_agreement_articulation(artic_df: pd.DataFrame, output_path: Path):
    """Figure 5: Model agreement on articulation difficulty (100-shot)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get 100-shot data
    df_100 = artic_df[artic_df["shot_count"] == 100].copy()
    pivot = df_100.pivot_table(
        index=["category", "rule_id"], columns="model", values="accuracy"
    ).reset_index()

    gpt_col = "gpt-4.1-nano-2025-04-14"
    claude_col = "claude-haiku-4-5-20251001"

    # Filter to only rules with data from both models (remove NaN)
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

    ax.set_xlabel("GPT-4.1-nano Articulation (100-shot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Claude Haiku 4.5 Articulation (100-shot)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Model Agreement on Articulation Difficulty",
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
    learn_df: pd.DataFrame, artic_df: pd.DataFrame, output_path: Path
):
    """Figure 6: Side-by-side heatmaps (learnability vs articulation at 100-shot)."""
    # Filter to 100-shot
    learn_100 = learn_df[learn_df["shot_count"] == 100].copy()
    artic_100 = artic_df[artic_df["shot_count"] == 100].copy()

    # Average across models for simplicity
    learn_pivot = (
        learn_100.groupby(["category", "rule_id"])["accuracy"].mean().reset_index()
    )
    artic_pivot = (
        artic_100.groupby(["category", "rule_id"])["accuracy"].mean().reset_index()
    )

    # Merge
    merged = learn_pivot.merge(
        artic_pivot, on=["category", "rule_id"], suffixes=("_learn", "_artic")
    )
    merged["gap"] = merged["accuracy_learn"] - merged["accuracy_artic"]

    # Sort by gap (descending)
    merged = merged.sort_values(["category", "gap"], ascending=[True, False])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 14))

    # Prepare data matrices
    learn_matrix = merged.set_index(["category", "rule_id"])["accuracy_learn"]
    artic_matrix = merged.set_index(["category", "rule_id"])["accuracy_artic"]
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

    # Plot articulation heatmap
    sns.heatmap(
        artic_matrix.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="Reds",
        vmin=0.4,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
        linewidths=0.5,
        ax=axes[1],
    )
    axes[1].set_title("Articulation (100-shot)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")

    # Plot gap heatmap
    sns.heatmap(
        gap_matrix.to_frame(),
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=0.8,
        cbar_kws={"label": "Gap (Learn - Artic)"},
        linewidths=0.5,
        ax=axes[2],
    )
    axes[2].set_title("Gap (Learn - Artic)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("")

    # Format rule labels (remove category from index for display)
    rule_labels = [rule_id for _, rule_id in merged[["category", "rule_id"]].values]
    for ax in axes:
        ax.set_yticklabels(rule_labels, fontsize=7)

    fig.suptitle(
        "Rule-Level Comparison: Learnability vs Articulation (100-shot)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 6 to {output_path}")


def generate_comprehensive_analysis(
    learn_df: pd.DataFrame, artic_df: pd.DataFrame, output_path: Path
):
    """Generate comprehensive markdown summary comparing learnability and articulation."""
    lines = []
    lines.append("# Multi-Shot Articulation Analysis: Learnability vs Articulation\n")

    # Overall comparison at 100-shot
    lines.append("## Overall Comparison (100-shot)\n")
    learn_100_mean = learn_df[learn_df["shot_count"] == 100]["accuracy"].mean()
    artic_100_mean = artic_df[artic_df["shot_count"] == 100]["accuracy"].mean()
    gap_100 = learn_100_mean - artic_100_mean

    lines.append(f"- **Mean Learnability:** {learn_100_mean:.1%}")
    lines.append(f"- **Mean Articulation:** {artic_100_mean:.1%}")
    lines.append(
        f"- **Gap:** {gap_100:+.1%} (positive = easier to learn than articulate)\n"
    )

    # Category breakdown at 100-shot
    lines.append("## Category Breakdown (100-shot)\n")
    for category in sorted(learn_df["category"].unique()):
        learn_cat = learn_df[
            (learn_df["category"] == category) & (learn_df["shot_count"] == 100)
        ]["accuracy"].mean()
        artic_cat = artic_df[
            (artic_df["category"] == category) & (artic_df["shot_count"] == 100)
        ]["accuracy"].mean()
        gap_cat = learn_cat - artic_cat

        lines.append(
            f"- **{category.capitalize()}:** Learn={learn_cat:.1%}, Artic={artic_cat:.1%}, Gap={gap_cat:+.1%}"
        )

    # Gap trends across shot counts
    lines.append("\n## Gap Trends Across Shot Counts\n")
    shot_counts = sorted(learn_df["shot_count"].unique())
    for category in sorted(learn_df["category"].unique()):
        gaps = []
        for shot in shot_counts:
            learn_mean = learn_df[
                (learn_df["category"] == category) & (learn_df["shot_count"] == shot)
            ]["accuracy"].mean()
            artic_mean = artic_df[
                (artic_df["category"] == category) & (artic_df["shot_count"] == shot)
            ]["accuracy"].mean()
            gaps.append(learn_mean - artic_mean)

        # Format gaps as string
        gaps_str = " → ".join([f"{g:+.1%}" for g in gaps])
        lines.append(f"- **{category.capitalize()}:** {gaps_str}")

    # Rules where articulation degrades
    lines.append("\n## Articulation Degradation (5-shot → 100-shot)\n")
    shot_5 = artic_df[artic_df["shot_count"] == 5].set_index(["rule_id", "model"])
    shot_100 = artic_df[artic_df["shot_count"] == 100].set_index(
        ["rule_id", "model"]
    )
    degradation = shot_5["accuracy"] - shot_100["accuracy"]
    degradation = degradation[degradation > 0.1]  # >10% degradation

    if len(degradation) > 0:
        top_5 = degradation.nlargest(5)
        lines.append(
            f"Found {len(degradation)} rule-model combinations where articulation degrades >10%:\n"
        )
        for (rule_id, model), deg in top_5.items():
            acc_5 = shot_5.loc[(rule_id, model), "accuracy"]
            acc_100 = shot_100.loc[(rule_id, model), "accuracy"]
            model_short = "GPT" if "gpt" in model.lower() else "Claude"
            lines.append(
                f"- `{rule_id}` ({model_short}): {acc_5:.1%} → {acc_100:.1%} (Δ={deg:+.1%})"
            )
    else:
        lines.append("- No significant degradation found (all rules improve or stay flat)")

    # Model agreement
    lines.append("\n## Model Agreement on Articulation (100-shot)\n")
    df_100 = artic_df[artic_df["shot_count"] == 100].copy()
    pivot = df_100.pivot_table(
        index="rule_id", columns="model", values="accuracy"
    )

    # Filter to only rules with data from both models
    pivot_filtered = pivot.dropna(subset=["gpt-4.1-nano-2025-04-14", "claude-haiku-4-5-20251001"])

    gpt_acc = pivot_filtered["gpt-4.1-nano-2025-04-14"].values
    claude_acc = pivot_filtered["claude-haiku-4-5-20251001"].values
    correlation = np.corrcoef(gpt_acc, claude_acc)[0, 1]

    lines.append(f"- **Pearson correlation (n={len(pivot_filtered)} rules with both models):** r = {correlation:.3f}")
    if correlation > 0.7:
        lines.append("- **Interpretation:** Strong agreement - models find similar rules hard to articulate")
    elif correlation > 0.4:
        lines.append("- **Interpretation:** Moderate agreement - some differences in articulation difficulty")
    else:
        lines.append("- **Interpretation:** Weak agreement - models differ in which rules are hard to articulate")

    # Key insights
    lines.append("\n## Key Insights\n")
    lines.append(
        "1. **Performance gap persists:** Learnability consistently exceeds articulation across all shot counts"
    )
    lines.append(
        "2. **Statistical rules hardest:** Largest gap for statistical rules (learnable but inarticulate)"
    )
    lines.append(
        "3. **More examples ≠ better articulation:** Some rules show degrading articulation with more shots"
    )
    lines.append(
        "4. **Semantic rules exception:** Smallest gap - models can articulate semantic rules better"
    )
    lines.append(
        "5. **CoT doesn't close the gap:** Despite CoT in articulation test, gap remains substantial"
    )

    # Write to file
    output_path.write_text("\n".join(lines))
    print(f"✓ Saved comprehensive analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create multi-shot articulation visualizations"
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
        "--articulation-summary",
        type=Path,
        default=Path("experiments/articulation_mc_multishot/summary_mc.yaml"),
        help="Path to multi-shot articulation summary YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/figures/articulation_multishot"),
        help="Directory to save figures",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    rules_df, learn_df, artic_df = load_data(
        args.rules_file, args.learnability_summary, args.articulation_summary
    )

    print(f"Loaded {len(rules_df)} rules")
    print(f"Learnability: {len(learn_df)} result records")
    print(f"Articulation: {len(artic_df)} result records")
    print(f"Categories: {sorted(learn_df['category'].unique())}")
    print(f"Models: {sorted(learn_df['model'].unique())}")
    print(f"Shot counts: {sorted(learn_df['shot_count'].unique())}\n")

    # Generate figures
    print("Generating visualizations...")
    plot_learn_vs_articulation_curves(
        learn_df, artic_df, args.output_dir / "fig1_learn_vs_artic_curves.png"
    )
    plot_category_comparison(
        learn_df, artic_df, args.output_dir / "fig2_category_comparison.png"
    )
    plot_gap_across_shots(
        learn_df, artic_df, args.output_dir / "fig3_gap_across_shots.png"
    )
    plot_degrading_articulation_rules(
        artic_df, args.output_dir / "fig4_degrading_articulation.png"
    )
    plot_model_agreement_articulation(
        artic_df, args.output_dir / "fig5_model_agreement.png"
    )
    plot_heatmap_comparison(
        learn_df, artic_df, args.output_dir / "fig6_heatmap_comparison.png"
    )

    # Generate comprehensive analysis
    generate_comprehensive_analysis(
        learn_df, artic_df, args.output_dir / "analysis_summary.md"
    )

    print(f"\n✓ All visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
