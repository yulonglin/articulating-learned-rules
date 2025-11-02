"""
Generate research-focused visualizations answering core questions:
1. Learnability vs Articulation: Can models learn but not articulate?
2. Articulation vs Faithfulness: Do good articulations faithfully explain behavior?
3. Learnability vs Faithfulness: Do easily-learned rules have faithful articulations?

These address the core research questions from RESEARCH_SPEC.md.
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


def load_all_data(
    rules_file: Path,
    learnability_summary: Path,
    articulation_summary: Path,
    faithfulness_summary: Path,
) -> pd.DataFrame:
    """Load and merge all three datasets into a single DataFrame."""
    # Load rules metadata
    rules_data = load_jsonl(rules_file)
    rules_df = pd.DataFrame(rules_data)

    # Load YAML summaries
    with learnability_summary.open("r") as f:
        learn_data = yaml.safe_load(f)

    with articulation_summary.open("r") as f:
        artic_data = yaml.safe_load(f)
        # Handle nested structure
        if "by_rule" in artic_data:
            artic_data = artic_data["by_rule"]

    with faithfulness_summary.open("r") as f:
        faith_data = yaml.safe_load(f)

    # Build combined dataframe
    rows = []
    for rule_id in learn_data.keys():
        # Get category
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        category = rule_row["category"].values[0] if len(rule_row) > 0 else "unknown"

        for model in learn_data[rule_id].keys():
            # Extract learnability metrics at different shot counts
            learn_5 = learn_data[rule_id][model].get("few_shot_5", {}).get("accuracy", None)
            learn_10 = learn_data[rule_id][model].get("few_shot_10", {}).get("accuracy", None)
            learn_20 = learn_data[rule_id][model].get("few_shot_20", {}).get("accuracy", None)
            learn_50 = learn_data[rule_id][model].get("few_shot_50", {}).get("accuracy", None)

            # Find min shot count where accuracy >= 0.9
            min_shot_for_90 = None
            for shot_count in [5, 10, 20, 50, 100]:
                acc = learn_data[rule_id][model].get(f"few_shot_{shot_count}", {}).get("accuracy", 0)
                if acc >= 0.9:
                    min_shot_for_90 = shot_count
                    break

            # Extract best articulation metrics (across all variations and shot counts)
            best_func_acc = -1
            best_llm_judge = -1
            if rule_id in artic_data and model in artic_data[rule_id]:
                for variation, metrics_list in artic_data[rule_id][model].items():
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

            # Extract faithfulness metrics (average across shot counts)
            faith_cf_scores = []
            faith_consistency_scores = []
            faith_cross_context_scores = []

            if rule_id in faith_data and model in faith_data[rule_id]:
                for shot_key, metrics in faith_data[rule_id][model].items():
                    if isinstance(metrics, dict):
                        cf = metrics.get("counterfactual_faithfulness")
                        cons = metrics.get("consistency_score")
                        cross = metrics.get("cross_context_match")
                        if cf is not None:
                            faith_cf_scores.append(cf)
                        if cons is not None:
                            faith_consistency_scores.append(cons)
                        if cross is not None:
                            faith_cross_context_scores.append(cross)

            faith_cf = np.mean(faith_cf_scores) if faith_cf_scores else None
            faith_consistency = np.mean(faith_consistency_scores) if faith_consistency_scores else None
            faith_cross_context = np.mean(faith_cross_context_scores) if faith_cross_context_scores else None

            rows.append(
                {
                    "rule_id": rule_id,
                    "category": category,
                    "model": model,
                    # Learnability
                    "learn_5shot": learn_5,
                    "learn_10shot": learn_10,
                    "learn_20shot": learn_20,
                    "learn_50shot": learn_50,
                    "min_shot_for_90": min_shot_for_90,
                    # Articulation
                    "best_functional_accuracy": best_func_acc if best_func_acc >= 0 else None,
                    "best_llm_judge_score": best_llm_judge if best_llm_judge >= 0 else None,
                    # Faithfulness
                    "counterfactual_faithfulness": faith_cf,
                    "consistency_score": faith_consistency,
                    "cross_context_match": faith_cross_context,
                }
            )

    df = pd.DataFrame(rows)

    # Add derived metrics
    df["articulation_gap"] = df["learn_20shot"] - df["best_functional_accuracy"]
    df["faithfulness_gap"] = df["best_functional_accuracy"] - df["counterfactual_faithfulness"]

    return df


def plot_learnability_vs_articulation(df: pd.DataFrame, output_path: Path):
    """
    KEY QUESTION: Can models learn rules but fail to articulate them?

    Shows: Learning accuracy (20-shot) vs Articulation quality (functional accuracy)

    INTERESTING CASES:
    - Points ABOVE diagonal: High learn, Low articulate (knowing without knowing!)
    - Points ON diagonal: Learning and articulation scale together
    """
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
            subset = subset.dropna(subset=["learn_20shot", "best_functional_accuracy"])

            if len(subset) > 0:
                ax.scatter(
                    subset["learn_20shot"],
                    subset["best_functional_accuracy"],
                    color=category_colors.get(category, "#95a5a6"),
                    marker=model_markers.get(model, "o"),
                    s=120,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{category.capitalize()} ({model.split('-')[0].upper()})"
                    if model == models[0]
                    else None,
                )

    # Diagonal line (perfect alignment)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect alignment")

    # Shade "knowing without knowing" region (high learn, low articulate)
    ax.fill_between(
        [0.9, 1.0],
        [0, 0],
        [0.9, 1.0],
        alpha=0.15,
        color="red",
        label='"Knowing without knowing" region',
    )

    # Annotate interesting cases
    gaps = df.copy()
    gaps["gap"] = gaps["learn_20shot"] - gaps["best_functional_accuracy"]
    gaps = gaps.dropna(subset=["gap"])

    # Top 3 biggest gaps
    top_gaps = gaps.nlargest(3, "gap")
    for _, row in top_gaps.iterrows():
        if row["learn_20shot"] >= 0.8 and row["gap"] > 0.15:
            ax.annotate(
                row["rule_id"][:25] + "...",
                (row["learn_20shot"], row["best_functional_accuracy"]),
                fontsize=8,
                alpha=0.8,
                xytext=(5, -15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )

    ax.set_xlabel("Learnability (20-shot accuracy)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Articulation Quality (Functional Accuracy)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Q1: Can Models Learn Rules But Fail To Articulate Them?\n"
        "(Points above diagonal = 'Knowing without knowing')",
        fontsize=14,
        fontweight="bold",
        pad=20,
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
    print(f"✓ Saved Research Q1 to {output_path}")


def plot_articulation_vs_faithfulness(df: pd.DataFrame, output_path: Path):
    """
    KEY QUESTION: Do good articulations faithfully explain behavior?

    Shows: Articulation quality vs Counterfactual faithfulness

    INTERESTING CASES:
    - Points BELOW diagonal: Good articulation but unfaithful (post-hoc rationalization!)
    - Points ON diagonal: Articulation quality predicts faithfulness
    """
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
                    s=120,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{category.capitalize()} ({model.split('-')[0].upper()})"
                    if model == models[0]
                    else None,
                )

    # Diagonal line (perfect correlation)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect correlation")

    # Shade "unfaithful despite good articulation" region
    ax.fill_between(
        [0.8, 1.0],
        [0, 0],
        [0.5, 0.5],
        alpha=0.15,
        color="red",
        label="Unfaithful despite good articulation",
    )

    # Annotate outliers (high articulation, low faithfulness)
    outliers = df[
        (df["best_functional_accuracy"] > 0.85)
        & (df["counterfactual_faithfulness"] < 0.6)
    ]
    for _, row in outliers.iterrows():
        ax.annotate(
            row["rule_id"][:25] + "...",
            (row["best_functional_accuracy"], row["counterfactual_faithfulness"]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3),
        )

    ax.set_xlabel("Articulation Quality (Functional Accuracy)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Counterfactual Faithfulness", fontsize=13, fontweight="bold")
    ax.set_title(
        "Q2: Do Good Articulations Faithfully Explain Behavior?\n"
        "(Points below diagonal = Post-hoc rationalization)",
        fontsize=14,
        fontweight="bold",
        pad=20,
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
    print(f"✓ Saved Research Q2 to {output_path}")


def plot_learnability_vs_faithfulness(df: pd.DataFrame, output_path: Path):
    """
    KEY QUESTION: Do easily-learned rules have faithful articulations?

    Shows: Learning accuracy vs Faithfulness

    INTERESTING CASES:
    - High learn, Low faithful: Model learned well but articulations don't explain behavior
    """
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

    for category in categories:
        for model in models:
            subset = df[(df["category"] == category) & (df["model"] == model)]
            subset = subset.dropna(subset=["learn_20shot", "counterfactual_faithfulness"])

            if len(subset) > 0:
                ax.scatter(
                    subset["learn_20shot"],
                    subset["counterfactual_faithfulness"],
                    color=category_colors.get(category, "#95a5a6"),
                    marker=model_markers.get(model, "o"),
                    s=120,
                    alpha=0.7,
                    edgecolors="black",
                    linewidths=1.5,
                    label=f"{category.capitalize()} ({model.split('-')[0].upper()})"
                    if model == models[0]
                    else None,
                )

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label="Perfect correlation")

    # Shade problematic region
    ax.fill_between(
        [0.9, 1.0],
        [0, 0],
        [0.6, 0.6],
        alpha=0.15,
        color="red",
        label="High learning, Low faithfulness",
    )

    ax.set_xlabel("Learnability (20-shot accuracy)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Counterfactual Faithfulness", fontsize=13, fontweight="bold")
    ax.set_title(
        "Q3: Do Easily-Learned Rules Have Faithful Articulations?\n"
        "(Points above diagonal suggest learning doesn't guarantee faithful articulation)",
        fontsize=14,
        fontweight="bold",
        pad=20,
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
    print(f"✓ Saved Research Q3 to {output_path}")


def plot_case_study_quadrants(df: pd.DataFrame, output_path: Path):
    """
    Show the 4 interesting quadrants:
    1. High learn, High articulate → Expected
    2. High learn, Low articulate → "Knowing without knowing"
    3. Low learn, Low articulate → Expected
    4. Low learn, High articulate → Suspicious (possible spurious)
    """
    fig, ax = plt.subplots(figsize=(14, 11))

    # Define quadrants
    learn_threshold = 0.85
    artic_threshold = 0.85

    quadrants = {
        "High learn, High articulate": df[
            (df["learn_20shot"] >= learn_threshold)
            & (df["best_functional_accuracy"] >= artic_threshold)
        ],
        "High learn, Low articulate\n('Knowing without knowing')": df[
            (df["learn_20shot"] >= learn_threshold)
            & (df["best_functional_accuracy"] < artic_threshold)
        ],
        "Low learn, High articulate\n(Suspicious)": df[
            (df["learn_20shot"] < learn_threshold)
            & (df["best_functional_accuracy"] >= artic_threshold)
        ],
        "Low learn, Low articulate": df[
            (df["learn_20shot"] < learn_threshold)
            & (df["best_functional_accuracy"] < artic_threshold)
        ],
    }

    colors = {
        "High learn, High articulate": "#2ecc71",  # Green
        "High learn, Low articulate\n('Knowing without knowing')": "#e74c3c",  # Red
        "Low learn, High articulate\n(Suspicious)": "#f39c12",  # Orange
        "Low learn, Low articulate": "#95a5a6",  # Gray
    }

    for quad_name, quad_df in quadrants.items():
        quad_df = quad_df.dropna(subset=["learn_20shot", "best_functional_accuracy"])
        if len(quad_df) > 0:
            ax.scatter(
                quad_df["learn_20shot"],
                quad_df["best_functional_accuracy"],
                c=colors[quad_name],
                s=150,
                alpha=0.6,
                edgecolors="black",
                linewidths=2,
                label=f"{quad_name} (n={len(quad_df)})",
            )

    # Draw quadrant lines
    ax.axvline(x=learn_threshold, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.axhline(y=artic_threshold, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)

    # Annotate a few examples from "Knowing without knowing" quadrant
    kwk = quadrants["High learn, Low articulate\n('Knowing without knowing')"]
    for idx, (_, row) in enumerate(kwk.head(3).iterrows()):
        ax.annotate(
            row["rule_id"][:20],
            (row["learn_20shot"], row["best_functional_accuracy"]),
            fontsize=8,
            xytext=(10, 10 + idx * 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.4),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
        )

    ax.set_xlabel("Learnability (20-shot accuracy)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Articulation Quality (Functional Accuracy)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Case Study: Identifying Interesting Patterns\n"
        "Four quadrants of learning vs articulation",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Case Study Quadrants to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create research-focused visualizations for core questions"
    )
    parser.add_argument(
        "--learnability-summary",
        type=Path,
        required=True,
        help="Path to learnability summary YAML",
    )
    parser.add_argument(
        "--articulation-summary",
        type=Path,
        required=True,
        help="Path to articulation summary YAML",
    )
    parser.add_argument(
        "--faithfulness-summary",
        type=Path,
        required=True,
        help="Path to faithfulness summary YAML",
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        required=True,
        help="Path to rules JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for figures",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Creating Research-Focused Visualizations")
    print("=" * 80)

    # Load all data
    print("\nLoading and merging datasets...")
    df = load_all_data(
        args.rules_file,
        args.learnability_summary,
        args.articulation_summary,
        args.faithfulness_summary,
    )
    print(f"  Loaded {len(df)} rule-model combinations")

    print("\nGenerating research question figures...")

    # Q1: Learnability vs Articulation
    plot_learnability_vs_articulation(
        df, args.output_dir / "research_q1_learnability_vs_articulation.png"
    )

    # Q2: Articulation vs Faithfulness
    plot_articulation_vs_faithfulness(
        df, args.output_dir / "research_q2_articulation_vs_faithfulness.png"
    )

    # Q3: Learnability vs Faithfulness
    plot_learnability_vs_faithfulness(
        df, args.output_dir / "research_q3_learnability_vs_faithfulness.png"
    )

    # Case study quadrants
    plot_case_study_quadrants(df, args.output_dir / "research_case_study_quadrants.png")

    print("\n" + "=" * 80)
    print("All research visualizations created successfully!")
    print(f"Saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
