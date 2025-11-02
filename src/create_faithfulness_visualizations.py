"""
Create comprehensive visualizations for faithfulness experiment results.

Tests whether articulated rules from Step 2 faithfully explain model behavior from Step 1.

Generates 10 publication-quality figures:
1a-c. Distribution variants (overlaid KDE, separate KDE, violin+KDE)
2. Model-specific distributions (GPT vs Claude)
3. Category comparison (bar chart)
4. Functional accuracy vs faithfulness (scatter)
5. Cross-context improvement (scatter)
6. Metric correlation heatmap
7. Rule-level faithfulness heatmap
8. Per-category boxplots
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
    faithfulness_summary: Path,
    articulation_summary: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load rules metadata, faithfulness results, and articulation results."""
    # Load rules with category information
    rules_data = load_jsonl(rules_file)
    rules_df = pd.DataFrame(rules_data)

    # Load faithfulness summary
    with faithfulness_summary.open("r") as f:
        faith_summary = yaml.safe_load(f)

    # Load articulation summary (for functional accuracy and LLM judge scores)
    with articulation_summary.open("r") as f:
        artic_summary = yaml.safe_load(f)

    # Convert to DataFrames
    faith_df = prepare_faithfulness_dataframe(rules_df, faith_summary)
    artic_df = prepare_articulation_dataframe(rules_df, artic_summary)

    # Merge faithfulness and articulation data
    merged_df = pd.merge(
        faith_df,
        artic_df,
        on=["rule_id", "model", "category"],
        how="left",
        suffixes=("", "_artic"),
    )

    return rules_df, faith_df, merged_df


def prepare_faithfulness_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """Convert faithfulness summary to long-format DataFrame.

    Handles both formats:
    - Old: {rule_id: {model: {metrics}}}
    - New: {rule_id: {model: {5shot: {metrics}, 10shot: {metrics}, ...}}}
    """
    rows = []

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, data in model_results.items():
            # Check if data has shot count structure (nested dicts with keys like "5shot")
            if isinstance(data, dict) and any(
                key.endswith("shot") for key in data.keys()
            ):
                # New format: nested shot counts
                for shot_key, metrics in data.items():
                    if isinstance(metrics, dict):
                        # Extract shot count from key (e.g., "5shot" -> 5)
                        shot_count = int(shot_key.replace("shot", ""))
                        rows.append(
                            {
                                "rule_id": rule_id,
                                "category": category,
                                "model": model,
                                "shot_count": shot_count,
                                "counterfactual_faithfulness": metrics.get("counterfactual_faithfulness"),
                                "consistency_score": metrics.get("consistency_score"),
                                "cross_context_match": metrics.get("cross_context_match"),
                                "functional_accuracy": metrics.get("functional_accuracy"),
                            }
                        )
            else:
                # Old format: flat metrics
                rows.append(
                    {
                        "rule_id": rule_id,
                        "category": category,
                        "model": model,
                        "shot_count": None,  # Unknown for old format
                        "counterfactual_faithfulness": data.get("counterfactual_faithfulness"),
                        "consistency_score": data.get("consistency_score"),
                        "cross_context_match": data.get("cross_context_match"),
                        "functional_accuracy": data.get("functional_accuracy"),
                    }
                )

    return pd.DataFrame(rows)


def prepare_articulation_dataframe(
    rules_df: pd.DataFrame, summary: dict[str, Any]
) -> pd.DataFrame:
    """
    Extract best articulation metrics (highest functional accuracy) from summary.

    The articulation summary has nested structure by variation and shot count.
    We need to find the best performing configuration per rule-model.
    """
    rows = []

    # Handle both formats: with or without "by_rule" wrapper
    if "by_rule" in summary:
        summary = summary["by_rule"]

    for rule_id, model_results in summary.items():
        # Get category for this rule
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model, variations in model_results.items():
            # Find best configuration across all variations and shot counts
            best_func_acc = -1
            best_llm_judge = -1

            for variation, metrics_list in variations.items():
                # metrics_list is a list of dicts, one per shot count
                if isinstance(metrics_list, list):
                    for metrics in metrics_list:
                        func_acc = metrics.get("functional_accuracy", 0) or 0
                        llm_judge = metrics.get("llm_judge", 0) or 0

                        if (
                            func_acc > best_func_acc
                            or (func_acc == best_func_acc and llm_judge > best_llm_judge)
                        ):
                            best_func_acc = func_acc
                            best_llm_judge = llm_judge

            rows.append(
                {
                    "rule_id": rule_id,
                    "category": category,
                    "model": model,
                    "best_functional_accuracy": best_func_acc if best_func_acc >= 0 else None,
                    "best_llm_judge_score": best_llm_judge if best_llm_judge >= 0 else None,
                }
            )

    return pd.DataFrame(rows)


def plot_distributions_overlaid_kde(df: pd.DataFrame, output_path: Path):
    """Figure 1a: Overlaid KDE plots for all 3 metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))

    metrics = [
        ("counterfactual_faithfulness", "Counterfactual Faithfulness", "#e74c3c"),
        ("consistency_score", "Consistency Score", "#3498db"),
        ("cross_context_match", "Cross-Context Match", "#2ecc71"),
    ]

    for metric, label, color in metrics:
        data = df[metric].dropna()
        if len(data) > 0:
            sns.kdeplot(
                data=data,
                label=label,
                color=color,
                linewidth=2.5,
                alpha=0.7,
                ax=ax,
            )

    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Faithfulness Metrics (Overlaid KDE)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1a to {output_path}")


def plot_distributions_separate_kde(df: pd.DataFrame, output_path: Path):
    """Figure 1b: Separate KDE subplots for each metric."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("counterfactual_faithfulness", "Counterfactual Faithfulness", "#e74c3c"),
        ("consistency_score", "Consistency Score", "#3498db"),
        ("cross_context_match", "Cross-Context Match", "#2ecc71"),
    ]

    for ax, (metric, label, color) in zip(axes, metrics):
        data = df[metric].dropna()
        if len(data) > 0:
            sns.kdeplot(
                data=data,
                color=color,
                linewidth=2.5,
                fill=True,
                alpha=0.3,
                ax=ax,
            )

            # Add mean line
            mean_val = data.mean()
            ax.axvline(
                x=mean_val,
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"Mean: {mean_val:.2%}",
            )

        ax.set_xlabel("Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim([0, 1])

    fig.suptitle(
        "Distribution of Faithfulness Metrics (Separate KDE)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1b to {output_path}")


def plot_distributions_violin_kde(df: pd.DataFrame, output_path: Path):
    """Figure 1c: Combined violin + KDE plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("counterfactual_faithfulness", "Counterfactual\nFaithfulness", "#e74c3c"),
        ("consistency_score", "Consistency\nScore", "#3498db"),
        ("cross_context_match", "Cross-Context\nMatch", "#2ecc71"),
    ]

    for ax, (metric, label, color) in zip(axes, metrics):
        data = df[[metric]].dropna()
        if len(data) > 0:
            # Violin plot
            parts = ax.violinplot(
                [data[metric].values],
                positions=[0],
                widths=0.7,
                showmeans=True,
                showmedians=True,
            )

            # Color the violin
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.3)

            # Overlay KDE on the side
            ax2 = ax.twinx()
            sns.kdeplot(
                data=data[metric],
                color=color,
                linewidth=2.5,
                ax=ax2,
            )
            ax2.set_ylabel("")
            ax2.set_yticks([])

        ax.set_ylabel("Score", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, linestyle=":", axis="y")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.suptitle(
        "Distribution of Faithfulness Metrics (Violin + KDE)",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1c to {output_path}")


def plot_model_distributions(df: pd.DataFrame, output_path: Path):
    """Figure 2: Model-specific distributions (GPT vs Claude overlaid KDE)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("counterfactual_faithfulness", "Counterfactual Faithfulness"),
        ("consistency_score", "Consistency Score"),
        ("cross_context_match", "Cross-Context Match"),
    ]

    models = sorted(df["model"].unique())
    model_labels = {
        "gpt-4.1-nano-2025-04-14": "GPT-4.1-nano",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    }
    model_colors = {
        "gpt-4.1-nano-2025-04-14": "#e74c3c",
        "claude-haiku-4-5-20251001": "#3498db",
    }

    for ax, (metric, label) in zip(axes, metrics):
        for model in models:
            data = df[df["model"] == model][metric].dropna()
            if len(data) > 0:
                sns.kdeplot(
                    data=data,
                    label=model_labels.get(model, model),
                    color=model_colors.get(model, "#95a5a6"),
                    linewidth=2.5,
                    alpha=0.7,
                    ax=ax,
                )

        ax.set_xlabel("Score", fontsize=11, fontweight="bold")
        ax.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim([0, 1])

    fig.suptitle(
        "Model Comparison: Faithfulness Distributions",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2 to {output_path}")


def plot_category_comparison(df: pd.DataFrame, output_path: Path):
    """Figure 3: Faithfulness by category (grouped bar chart)."""
    categories = sorted(df["category"].unique())
    category_colors = {
        "syntactic": "#2ecc71",
        "pattern": "#e67e22",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    # Calculate means and stds by category
    metrics = [
        ("counterfactual_faithfulness", "Counterfactual"),
        ("consistency_score", "Consistency"),
        ("cross_context_match", "Cross-Context"),
    ]

    data_for_plot = []
    for category in categories:
        cat_df = df[df["category"] == category]
        for metric, metric_label in metrics:
            values = cat_df[metric].dropna()
            if len(values) > 0:
                data_for_plot.append(
                    {
                        "Category": category.capitalize(),
                        "Metric": metric_label,
                        "Mean": values.mean(),
                        "Std": values.std(),
                    }
                )

    plot_df = pd.DataFrame(data_for_plot)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(categories))
    width = 0.25

    for i, (metric, metric_label) in enumerate(metrics):
        metric_df = plot_df[plot_df["Metric"] == metric_label]
        means = [
            metric_df[metric_df["Category"] == cat.capitalize()]["Mean"].values[0]
            if len(metric_df[metric_df["Category"] == cat.capitalize()]) > 0
            else 0
            for cat in categories
        ]
        stds = [
            metric_df[metric_df["Category"] == cat.capitalize()]["Std"].values[0]
            if len(metric_df[metric_df["Category"] == cat.capitalize()]) > 0
            else 0
            for cat in categories
        ]

        colors = [category_colors.get(cat, "#95a5a6") for cat in categories]
        ax.bar(
            x + i * width,
            means,
            width,
            label=metric_label,
            yerr=stds,
            capsize=5,
            alpha=0.8,
            color=colors,
            edgecolor="black",
            linewidth=1,
        )

    ax.set_xlabel("Rule Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Faithfulness by Category (Error bars show std dev)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([cat.capitalize() for cat in categories])
    ax.legend(fontsize=11, title="Metric")
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3 to {output_path}")


def plot_functional_vs_faithfulness(df: pd.DataFrame, output_path: Path):
    """Figure 4: Functional accuracy vs counterfactual faithfulness (scatter)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    categories = sorted(df["category"].unique())
    category_colors = {
        "syntactic": "#2ecc71",
        "pattern": "#e67e22",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    models = sorted(df["model"].unique())
    model_markers = {
        "gpt-4.1-nano-2025-04-14": "o",
        "claude-haiku-4-5-20251001": "s",
    }

    # Plot scatter for each category and model
    for category in categories:
        for model in models:
            subset = df[(df["category"] == category) & (df["model"] == model)]
            subset = subset.dropna(
                subset=["best_functional_accuracy", "counterfactual_faithfulness"]
            )

            if len(subset) > 0:
                ax.scatter(
                    subset["best_functional_accuracy"],
                    subset["counterfactual_faithfulness"],
                    color=category_colors.get(category, "#95a5a6"),
                    marker=model_markers.get(model, "o"),
                    s=100,
                    alpha=0.6,
                    edgecolors="black",
                    linewidths=1,
                    label=f"{category.capitalize()} ({model.split('-')[0].upper()})"
                    if model == models[0]
                    else None,
                )

    # Diagonal line (perfect correlation)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect correlation")

    # Annotate outliers (high functional, low faithfulness)
    outliers = df[
        (df["best_functional_accuracy"] > 0.9)
        & (df["counterfactual_faithfulness"] < 0.4)
    ]
    for _, row in outliers.iterrows():
        ax.annotate(
            row["rule_id"][:20] + "...",
            (row["best_functional_accuracy"], row["counterfactual_faithfulness"]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Functional Accuracy (Step 2)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Counterfactual Faithfulness (Step 3)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Functional Accuracy vs Faithfulness\n(Points below diagonal = unfaithful despite high accuracy)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4 to {output_path}")


def plot_cross_context_improvement(df: pd.DataFrame, output_path: Path):
    """Figure 5: Cross-context match vs direct articulation LLM judge score."""
    fig, ax = plt.subplots(figsize=(12, 10))

    categories = sorted(df["category"].unique())
    category_colors = {
        "syntactic": "#2ecc71",
        "pattern": "#e67e22",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    for category in categories:
        subset = df[df["category"] == category]
        subset = subset.dropna(subset=["best_llm_judge_score", "cross_context_match"])

        if len(subset) > 0:
            ax.scatter(
                subset["best_llm_judge_score"],
                subset["cross_context_match"],
                color=category_colors.get(category, "#95a5a6"),
                s=100,
                alpha=0.6,
                edgecolors="black",
                linewidths=1,
                label=category.capitalize(),
            )

    # Diagonal line (no improvement)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="No improvement")

    # Shade region where cross-context helps
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color="green", label="Cross-context helps")

    ax.set_xlabel("Direct Articulation LLM Judge Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cross-Context Match Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cross-Context Improvement Over Direct Articulation\n(Points above diagonal = cross-context framing helps)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 5 to {output_path}")


def plot_metric_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """Figure 6: Correlation heatmap between all metrics."""
    # Select relevant columns
    corr_cols = [
        "counterfactual_faithfulness",
        "consistency_score",
        "cross_context_match",
        "best_functional_accuracy",
        "best_llm_judge_score",
    ]

    corr_labels = {
        "counterfactual_faithfulness": "Counterfactual\nFaithfulness",
        "consistency_score": "Consistency\nScore",
        "cross_context_match": "Cross-Context\nMatch",
        "best_functional_accuracy": "Functional\nAccuracy",
        "best_llm_judge_score": "LLM Judge\nScore",
    }

    # Compute correlation matrix
    corr_df = df[corr_cols].corr()

    # Rename for better labels
    corr_df = corr_df.rename(columns=corr_labels, index=corr_labels)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        vmin=-1,
        vmax=1,
    )

    ax.set_title(
        "Correlation Between Faithfulness and Articulation Metrics",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 6 to {output_path}")


def plot_rule_level_heatmap(df: pd.DataFrame, output_path: Path):
    """Figure 7: Rule-level faithfulness heatmap (rules × models).

    Aggregates across shot counts by taking the mean.
    """
    # Aggregate across shot counts (if present) by taking mean
    agg_df = (
        df.groupby(["rule_id", "model"])["counterfactual_faithfulness"]
        .mean()
        .reset_index()
    )

    # Pivot data for heatmap
    pivot_df = agg_df.pivot(
        index="rule_id",
        columns="model",
        values="counterfactual_faithfulness",
    )

    # Sort by mean faithfulness
    pivot_df["mean"] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values("mean")
    pivot_df = pivot_df.drop("mean", axis=1)

    # Rename columns
    model_labels = {
        "gpt-4.1-nano-2025-04-14": "GPT-4.1-nano",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    }
    pivot_df = pivot_df.rename(columns=model_labels)

    fig, ax = plt.subplots(figsize=(10, max(12, len(pivot_df) * 0.3)))

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": "Counterfactual Faithfulness"},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rule", fontsize=12, fontweight="bold")
    ax.set_title(
        "Rule-Level Counterfactual Faithfulness\n(Sorted by mean faithfulness)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # Rotate y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 7 to {output_path}")


def plot_category_boxplots(df: pd.DataFrame, output_path: Path):
    """Figure 8: Per-category boxplots for counterfactual faithfulness."""
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = sorted(df["category"].unique())
    category_colors = {
        "syntactic": "#2ecc71",
        "pattern": "#e67e22",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    # Prepare data for boxplot
    data_by_category = [
        df[df["category"] == cat]["counterfactual_faithfulness"].dropna().values
        for cat in categories
    ]

    bp = ax.boxplot(
        data_by_category,
        labels=[cat.capitalize() for cat in categories],
        patch_artist=True,
        widths=0.6,
        showfliers=True,
    )

    # Color boxes
    for patch, category in zip(bp["boxes"], categories):
        patch.set_facecolor(category_colors.get(category, "#95a5a6"))
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, category in enumerate(categories):
        cat_data = df[df["category"] == category]["counterfactual_faithfulness"].dropna()
        x = np.random.normal(i + 1, 0.04, size=len(cat_data))
        ax.scatter(
            x,
            cat_data.values,
            alpha=0.4,
            s=50,
            color=category_colors.get(category, "#95a5a6"),
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("Rule Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Counterfactual Faithfulness", fontsize=12, fontweight="bold")
    ax.set_title(
        "Distribution of Counterfactual Faithfulness by Category",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(alpha=0.3, linestyle=":", axis="y")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 8 to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create visualizations for faithfulness experiment results"
    )
    parser.add_argument(
        "--faithfulness-summary",
        type=Path,
        required=True,
        help="Path to faithfulness summary YAML file",
    )
    parser.add_argument(
        "--articulation-summary",
        type=Path,
        required=True,
        help="Path to articulation summary YAML file (for functional accuracy)",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        required=True,
        help="Path to rules JSONL file (for category information)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Creating Faithfulness Visualizations")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    rules_df, faith_df, merged_df = load_data(
        args.rules_file,
        args.faithfulness_summary,
        args.articulation_summary,
    )
    print(f"  Loaded {len(rules_df)} rules")
    print(f"  Loaded {len(faith_df)} faithfulness results")
    print(f"  Merged {len(merged_df)} results with articulation data")

    print("\nGenerating figures...")

    # Figure 1: Distribution variants
    plot_distributions_overlaid_kde(
        faith_df, args.output_dir / "fig1a_distributions_overlaid_kde.png"
    )
    plot_distributions_separate_kde(
        faith_df, args.output_dir / "fig1b_distributions_separate_kde.png"
    )
    plot_distributions_violin_kde(
        faith_df, args.output_dir / "fig1c_distributions_violin_kde.png"
    )

    # Figure 2: Model distributions
    plot_model_distributions(faith_df, args.output_dir / "fig2_model_distributions.png")

    # Figure 3: Category comparison
    plot_category_comparison(faith_df, args.output_dir / "fig3_category_comparison.png")

    # Figure 4: Functional vs faithfulness
    plot_functional_vs_faithfulness(
        merged_df, args.output_dir / "fig4_functional_vs_faithfulness.png"
    )

    # Figure 5: Cross-context improvement
    plot_cross_context_improvement(
        merged_df, args.output_dir / "fig5_cross_context_improvement.png"
    )

    # Figure 6: Correlation heatmap
    plot_metric_correlation_heatmap(
        merged_df, args.output_dir / "fig6_metric_correlation.png"
    )

    # Figure 7: Rule-level heatmap
    plot_rule_level_heatmap(faith_df, args.output_dir / "fig7_rule_level_heatmap.png")

    # Figure 8: Category boxplots
    plot_category_boxplots(faith_df, args.output_dir / "fig8_category_boxplots.png")

    print("\n" + "=" * 80)
    print("All visualizations created successfully!")
    print(f"Saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
