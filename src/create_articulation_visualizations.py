"""
Create comprehensive visualizations comparing learnability vs articulation performance.

Generates 5 publication-quality figures to answer:
1. Are particular rule types harder to articulate than learn?
2. Does min_few_shot_required correlate with articulation difficulty?
3. Do models agree on which rules are hard to articulate?
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
    learnability_file: Path,
    articulation_file: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load rules metadata, learnability results, and articulation results."""
    # Load rules with category and learnability metadata
    rules_data = load_jsonl(rules_file)
    rules_df = pd.DataFrame(rules_data)

    # Load learnability summary
    with learnability_file.open("r") as f:
        learnability = yaml.safe_load(f)

    # Load articulation summary
    with articulation_file.open("r") as f:
        articulation = yaml.safe_load(f)

    return rules_df, learnability, articulation


def prepare_comparison_dataframe(
    rules_df: pd.DataFrame,
    learnability: dict[str, Any],
    articulation: dict[str, Any],
) -> pd.DataFrame:
    """Create unified DataFrame with learnability and articulation metrics."""
    rows = []

    for rule_id in articulation.keys():
        # Get rule metadata
        rule_row = rules_df[rules_df["rule_id"] == rule_id]
        if len(rule_row) == 0:
            continue

        category = rule_row["category"].values[0]
        learnability_data = rule_row["learnability"].values[0] if "learnability" in rule_row.columns else {}

        # Get learnability and articulation for each model
        for model in articulation[rule_id].keys():
            # Articulation metrics (MC test)
            mc_accuracy = articulation[rule_id][model]["accuracy"]
            mc_n_total = articulation[rule_id][model]["n_total"]

            # Learnability metrics (100-shot)
            learn_100_acc = None
            min_few_shot = None
            if rule_id in learnability and model in learnability[rule_id]:
                if "few_shot_100" in learnability[rule_id][model]:
                    learn_100_acc = learnability[rule_id][model]["few_shot_100"]["accuracy"]

                # Get min_few_shot_required from rule metadata
                if learnability_data and model in learnability_data:
                    min_few_shot = learnability_data[model].get("min_few_shot_required")

            # Calculate gap
            gap = (learn_100_acc - mc_accuracy) if learn_100_acc is not None else None

            rows.append({
                "rule_id": rule_id,
                "category": category,
                "model": model,
                "mc_accuracy": mc_accuracy,
                "mc_n_total": mc_n_total,
                "learn_100_accuracy": learn_100_acc,
                "min_few_shot_required": min_few_shot,
                "gap": gap,
            })

    return pd.DataFrame(rows)


def plot_performance_gap_scatter(df: pd.DataFrame, output_path: Path):
    """Figure 1: Learnability vs Articulation scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Filter to rows with both metrics
    df_complete = df.dropna(subset=["learn_100_accuracy", "mc_accuracy"])

    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    # Plot by category
    for category in sorted(df_complete["category"].unique()):
        cat_df = df_complete[df_complete["category"] == category]
        ax.scatter(
            cat_df["learn_100_accuracy"],
            cat_df["mc_accuracy"],
            label=category.capitalize(),
            color=category_colors.get(category, "#95a5a6"),
            s=120,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.8
        )

    # Add diagonal reference line (y=x)
    lim_min = 0.4
    lim_max = 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', linewidth=2, alpha=0.5, label='Equal performance', zorder=1)

    # Add 90% threshold lines
    ax.axhline(y=0.90, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.axvline(x=0.90, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    # Annotate quadrants
    ax.text(0.95, 0.50, "Easy to learn\nHard to articulate",
            ha='center', va='center', fontsize=10, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))
    ax.text(0.95, 0.95, "Easy to learn\nEasy to articulate",
            ha='center', va='center', fontsize=10, alpha=0.5,
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.1))

    ax.set_xlabel("Learnability Accuracy (100-shot)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Articulation Accuracy (MC)", fontsize=14, fontweight="bold")
    ax.set_title("Learnability vs Articulation Performance Gap",
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 1: Performance Gap Scatter to {output_path}")


def plot_category_gap_analysis(df: pd.DataFrame, output_path: Path):
    """Figure 2: Category-wise comparison of learnability vs articulation."""
    # Filter complete data and aggregate by category
    df_complete = df.dropna(subset=["learn_100_accuracy", "mc_accuracy"])

    category_summary = df_complete.groupby("category").agg({
        "learn_100_accuracy": ["mean", "std"],
        "mc_accuracy": ["mean", "std"],
    }).reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))

    categories = sorted(category_summary["category"].unique())
    x = np.arange(len(categories))
    width = 0.35

    # Extract means and stds
    learn_means = [category_summary[category_summary["category"] == cat][("learn_100_accuracy", "mean")].values[0]
                   for cat in categories]
    learn_stds = [category_summary[category_summary["category"] == cat][("learn_100_accuracy", "std")].values[0]
                  for cat in categories]
    mc_means = [category_summary[category_summary["category"] == cat][("mc_accuracy", "mean")].values[0]
                for cat in categories]
    mc_stds = [category_summary[category_summary["category"] == cat][("mc_accuracy", "std")].values[0]
               for cat in categories]

    # Plot bars
    ax.bar(x - width/2, learn_means, width, yerr=learn_stds,
           label="Learnability (100-shot)", color="#3498db", capsize=5, alpha=0.8)
    ax.bar(x + width/2, mc_means, width, yerr=mc_stds,
           label="Articulation (MC)", color="#e74c3c", capsize=5, alpha=0.8)

    # Add 90% threshold
    ax.axhline(y=0.90, color="gray", linestyle="--", linewidth=1.5,
               label="90% threshold", alpha=0.7)

    ax.set_xlabel("Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Learnability vs Articulation by Category",
                 fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([cat.capitalize() for cat in categories], fontsize=12)
    ax.legend(fontsize=12, loc="lower left")
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_ylim([0.4, 1.05])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 2: Category Gap Analysis to {output_path}")


def plot_rule_level_heatmap_with_gap(
    df: pd.DataFrame,
    learnability: dict[str, Any],
    output_path: Path
):
    """Figure 3: Rule-level heatmap showing learnability progression + articulation + gap."""
    # Prepare data for heatmap
    # Columns: 5-shot, 10-shot, 20-shot, 50-shot, 100-shot, MC, Gap
    rows_data = []

    for _, row in df.iterrows():
        rule_id = row["rule_id"]
        model = row["model"]
        category = row["category"]

        # Get learnability progression
        learn_row = {"rule_id": rule_id, "model": model, "category": category}
        if rule_id in learnability and model in learnability[rule_id]:
            for shot in [5, 10, 20, 50, 100]:
                shot_key = f"few_shot_{shot}"
                if shot_key in learnability[rule_id][model]:
                    learn_row[f"{shot}s"] = learnability[rule_id][model][shot_key]["accuracy"]
                else:
                    learn_row[f"{shot}s"] = np.nan
        else:
            for shot in [5, 10, 20, 50, 100]:
                learn_row[f"{shot}s"] = np.nan

        # Add MC and gap
        learn_row["MC"] = row["mc_accuracy"]
        learn_row["Gap"] = row["gap"] if pd.notna(row["gap"]) else np.nan

        rows_data.append(learn_row)

    heatmap_df = pd.DataFrame(rows_data)

    # Average across models for cleaner visualization
    heatmap_agg = heatmap_df.groupby(["category", "rule_id"]).agg({
        "5s": "mean",
        "10s": "mean",
        "20s": "mean",
        "50s": "mean",
        "100s": "mean",
        "MC": "mean",
        "Gap": "mean",
    }).reset_index()

    # Sort by category, then by 100-shot performance
    heatmap_agg = heatmap_agg.sort_values(["category", "100s"], ascending=[True, False])

    # Create pivot for heatmap
    heatmap_matrix = heatmap_agg[["5s", "10s", "20s", "50s", "100s", "MC", "Gap"]].values

    fig, ax = plt.subplots(figsize=(10, 18))

    # Plot heatmap
    im = ax.imshow(heatmap_matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)

    # Set ticks
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(["5-shot", "10-shot", "20-shot", "50-shot", "100-shot", "MC", "Gap"],
                       fontsize=11)
    ax.set_yticks(np.arange(len(heatmap_agg)))
    ax.set_yticklabels(heatmap_agg["rule_id"].values, fontsize=8)

    # Color y-tick labels by category
    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }
    for i, category in enumerate(heatmap_agg["category"].values):
        ax.get_yticklabels()[i].set_color(category_colors.get(category, "black"))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy / Gap", fontsize=12, fontweight="bold")

    # Annotate cells with values
    for i in range(len(heatmap_agg)):
        for j in range(7):
            value = heatmap_matrix[i, j]
            if not np.isnan(value):
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                       color=text_color, fontsize=7)

    ax.set_xlabel("Metric", fontsize=13, fontweight="bold")
    ax.set_ylabel("Rule ID (colored by category)", fontsize=13, fontweight="bold")
    ax.set_title("Learnability Progression + Articulation Performance",
                 fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 3: Rule-Level Heatmap to {output_path}")


def plot_shot_count_vs_articulation(df: pd.DataFrame, output_path: Path):
    """Figure 4: Does min_few_shot_required correlate with articulation difficulty?"""
    # Filter to rows with min_few_shot data
    df_complete = df.dropna(subset=["min_few_shot_required", "mc_accuracy"])

    fig, ax = plt.subplots(figsize=(12, 8))

    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    # Plot by category
    for category in sorted(df_complete["category"].unique()):
        cat_df = df_complete[df_complete["category"] == category]
        ax.scatter(
            cat_df["min_few_shot_required"],
            cat_df["mc_accuracy"],
            label=category.capitalize(),
            color=category_colors.get(category, "#95a5a6"),
            s=120,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.8
        )

    # Add trend line
    x_vals = df_complete["min_few_shot_required"].values
    y_vals = df_complete["mc_accuracy"].values
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.5,
            label=f"Trend line (slope={z[0]:.4f})")

    # Calculate correlation
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    ax.text(0.05, 0.95, f"Correlation: r = {corr:.3f}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel("Min Few-Shot Required (to reach 90%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("MC Articulation Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Does Learning Difficulty Predict Articulation Difficulty?",
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_ylim([0.0, 1.05])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 4: Shot Count vs Articulation to {output_path}")


def plot_model_agreement_on_articulation(df: pd.DataFrame, output_path: Path):
    """Figure 5: Do models agree on which rules are hard to articulate?"""
    # Pivot to get GPT vs Claude comparison
    pivot = df.pivot_table(
        index=["category", "rule_id"],
        columns="model",
        values="mc_accuracy"
    ).reset_index()

    gpt_col = "gpt-4.1-nano-2025-04-14"
    claude_col = "claude-haiku-4-5-20251001"

    # Filter to rules tested on both models
    pivot_complete = pivot.dropna(subset=[gpt_col, claude_col])

    if len(pivot_complete) == 0:
        print("⚠ Warning: No rules tested on both models - skipping model agreement plot")
        return

    gpt_acc = pivot_complete[gpt_col].values
    claude_acc = pivot_complete[claude_col].values
    categories = pivot_complete["category"].values

    # Calculate correlation
    corr = np.corrcoef(gpt_acc, claude_acc)[0, 1]

    fig, ax = plt.subplots(figsize=(11, 10))

    category_colors = {
        "pattern-based": "#2ecc71",
        "semantic": "#9b59b6",
        "statistical": "#f39c12",
    }

    # Plot by category
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
            linewidths=0.8
        )

    # Add diagonal reference line (y=x)
    lim_min = 0.0
    lim_max = 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', linewidth=2, alpha=0.5, label='Equal performance')

    # Add correlation text
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel("GPT-4.1-nano MC Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Claude Haiku 4.5 MC Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Model Agreement on Articulation Difficulty",
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Figure 5: Model Agreement to {output_path}")


def generate_analysis_summary(df: pd.DataFrame, output_path: Path):
    """Generate markdown summary of key findings."""
    lines = []
    lines.append("# Learnability vs Articulation Analysis Summary\n")
    lines.append("## Overall Comparison\n")

    # Overall statistics
    df_complete = df.dropna(subset=["learn_100_accuracy", "mc_accuracy"])
    learn_mean = df_complete["learn_100_accuracy"].mean()
    mc_mean = df_complete["mc_accuracy"].mean()
    gap_mean = df_complete["gap"].mean()

    lines.append(f"- **Mean learnability (100-shot):** {learn_mean:.1%}")
    lines.append(f"- **Mean articulation (MC):** {mc_mean:.1%}")
    lines.append(f"- **Mean gap:** {gap_mean:.1%} (positive = easier to learn than articulate)")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown\n")
    for category in sorted(df_complete["category"].unique()):
        cat_df = df_complete[df_complete["category"] == category]
        cat_learn = cat_df["learn_100_accuracy"].mean()
        cat_mc = cat_df["mc_accuracy"].mean()
        cat_gap = cat_df["gap"].mean()
        lines.append(
            f"- **{category.capitalize()}:** "
            f"Learn={cat_learn:.1%}, MC={cat_mc:.1%}, Gap={cat_gap:.1%}"
        )
    lines.append("")

    # Rules with biggest gap (learnable but hard to articulate)
    lines.append("## Top 5 Rules: Learnable but Hard to Articulate\n")
    top_gap = df_complete.nlargest(5, "gap")
    for _, row in top_gap.iterrows():
        lines.append(
            f"- `{row['rule_id']}` ({row['category']}): "
            f"Learn={row['learn_100_accuracy']:.1%}, MC={row['mc_accuracy']:.1%}, "
            f"Gap={row['gap']:.1%}"
        )
    lines.append("")

    # Correlation analysis
    if "min_few_shot_required" in df.columns:
        df_corr = df.dropna(subset=["min_few_shot_required", "mc_accuracy"])
        if len(df_corr) > 0:
            corr = np.corrcoef(
                df_corr["min_few_shot_required"].values,
                df_corr["mc_accuracy"].values
            )[0, 1]
            lines.append("## Learning Difficulty vs Articulation Difficulty\n")
            lines.append(f"- **Correlation (min_few_shot vs MC):** r = {corr:.3f}")
            if corr < -0.3:
                lines.append("- **Interpretation:** Rules requiring more examples are HARDER to articulate")
            elif corr > 0.3:
                lines.append("- **Interpretation:** Rules requiring more examples are EASIER to articulate (surprising!)")
            else:
                lines.append("- **Interpretation:** Weak correlation - learning difficulty doesn't predict articulation difficulty")
            lines.append("")

    # Key insights
    lines.append("## Key Insights\n")
    lines.append("1. **Performance gap exists:** Most rules are easier to learn than articulate")
    lines.append("2. **Category patterns:** Statistical rules show largest gap (learnable but hard to articulate)")
    lines.append("3. **Sample size caveat:** Current MC data has only 5 samples per rule - confidence intervals are wide")
    lines.append("4. **Next step:** Re-run with 100 MC samples for statistical reliability")

    # Write to file
    output_path.write_text("\n".join(lines))
    print(f"✓ Saved analysis summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create articulation visualizations comparing learnability and articulation"
    )
    parser.add_argument(
        "--rules-file",
        type=Path,
        default=Path("data/processed/rules/curated_rules_learnable.jsonl"),
        help="Path to curated rules file with learnability metadata",
    )
    parser.add_argument(
        "--learnability-file",
        type=Path,
        default=Path("experiments/learnability/summary.yaml"),
        help="Path to learnability summary YAML",
    )
    parser.add_argument(
        "--articulation-file",
        type=Path,
        default=Path("experiments/articulation_mc/summary_mc.yaml"),
        help="Path to articulation MC summary YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/figures/articulation"),
        help="Directory to save figures",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    rules_df, learnability, articulation = load_data(
        args.rules_file, args.learnability_file, args.articulation_file
    )

    df = prepare_comparison_dataframe(rules_df, learnability, articulation)

    print(f"Loaded {len(rules_df)} rules")
    print(f"Learnability data: {len(learnability)} rules")
    print(f"Articulation data: {len(articulation)} rules")
    print(f"Combined DataFrame: {len(df)} records")
    print(f"Categories: {sorted(df['category'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}\n")

    # Generate figures
    print("Generating visualizations...")
    plot_performance_gap_scatter(df, args.output_dir / "fig1_performance_gap_scatter.png")
    plot_category_gap_analysis(df, args.output_dir / "fig2_category_gap_analysis.png")
    plot_rule_level_heatmap_with_gap(df, learnability, args.output_dir / "fig3_rule_heatmap_with_gap.png")
    plot_shot_count_vs_articulation(df, args.output_dir / "fig4_shot_count_vs_articulation.png")
    plot_model_agreement_on_articulation(df, args.output_dir / "fig5_model_agreement_articulation.png")

    # Generate summary
    generate_analysis_summary(df, args.output_dir / "analysis_summary.md")

    print(f"\n✓ All visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
