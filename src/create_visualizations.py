"""
Create visualizations for LLM rule articulation study results.

Generates figures for:
1. Learnability by model
2. Articulation performance (MC vs free-form)
3. Faithfulness scores (counterfactual vs functional)
4. Model comparison across pipeline
5. Spurious correlation analysis
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from src.model_registry import GPTModels, ClaudeModels, get_display_name

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_yaml_summary(filepath: Path) -> dict[str, Any]:
    """Load YAML summary file."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def plot_learnability_by_model(summary_file: Path, output_dir: Path):
    """Plot learnability results by model."""
    data = load_yaml_summary(summary_file)

    # Extract model performance - group by best_model
    models = {}
    for rule_id, rule_data in data.items():
        if not isinstance(rule_data, dict):
            continue

        best_model = rule_data.get('best_model')
        best_accuracy = rule_data.get('best_accuracy', 0)

        if best_model:
            if best_model not in models:
                models[best_model] = {'accuracies': [], 'rules': []}
            models[best_model]['accuracies'].append(best_accuracy)
            models[best_model]['rules'].append(rule_id)

    # Create figure with just accuracy (no parse rate in summary)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot accuracies
    model_names = sorted(models.keys())
    accuracies = [np.mean(models[m]['accuracies']) for m in model_names]
    rule_counts = [len(models[m]['rules']) for m in model_names]

    # Create colors based on model names
    colors = []
    for m in model_names:
        if 'gpt' in m.lower():
            colors.append('#e74c3c')
        elif 'claude' in m.lower():
            colors.append('#2ecc71')
        else:
            colors.append('#3498db')

    bars = ax.bar(range(len(model_names)), accuracies, color=colors, alpha=0.8)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([get_display_name(m) for m in model_names], rotation=0)
    ax.set_ylabel('Average Accuracy')
    ax.set_title('Learnability: Average Accuracy by Best Model')
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% threshold', alpha=0.5)
    ax.set_ylim(0, 1.05)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, rule_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={count}',
                ha='center', va='bottom', fontsize=10)

    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'learnability_by_model.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'learnability_by_model.png'}")
    plt.close()


def plot_articulation_performance(artic_summary: Path, output_dir: Path):
    """Plot articulation performance."""
    data = load_yaml_summary(artic_summary)

    # Compute aggregated statistics from raw data
    by_model = {}
    by_prompt = {}

    for rule_id, rule_data in data.items():
        if not isinstance(rule_data, dict):
            continue
        for model, model_data in rule_data.items():
            if not isinstance(model_data, dict):
                continue
            if model not in by_model:
                by_model[model] = {'llm_judge': [], 'functional': []}

            for prompt_type, metrics in model_data.items():
                if not isinstance(metrics, dict):
                    continue

                # Aggregate by model
                by_model[model]['llm_judge'].append(metrics.get('llm_judge', 0))
                by_model[model]['functional'].append(metrics.get('functional_accuracy', 0))

                # Aggregate by prompt
                if prompt_type not in by_prompt:
                    by_prompt[prompt_type] = {'llm_judge': [], 'functional': []}
                by_prompt[prompt_type]['llm_judge'].append(metrics.get('llm_judge', 0))
                by_prompt[prompt_type]['functional'].append(metrics.get('functional_accuracy', 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # By model
    models = sorted(by_model.keys())
    model_labels = [get_display_name(m) for m in models]
    llm_scores = [np.mean(by_model[m]['llm_judge']) for m in models]
    func_scores = [np.mean(by_model[m]['functional']) for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, llm_scores, width, label='LLM Judge', color='#3498db')
    ax1.bar(x + width/2, func_scores, width, label='Functional', color='#2ecc71')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Articulation Performance by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=15, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.grid(axis='y', alpha=0.3)

    # By prompt
    prompts = sorted(by_prompt.keys())
    prompt_labels = [p.replace('simple', 'Simple').replace('cot', 'CoT').replace('explicit', 'Explicit')
                     for p in prompts]
    llm_prompt = [np.mean(by_prompt[p]['llm_judge']) for p in prompts]
    func_prompt = [np.mean(by_prompt[p]['functional']) for p in prompts]

    x2 = np.arange(len(prompts))

    ax2.bar(x2 - width/2, llm_prompt, width, label='LLM Judge', color='#3498db')
    ax2.bar(x2 + width/2, func_prompt, width, label='Functional', color='#2ecc71')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Articulation Performance by Prompt')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(prompt_labels)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'articulation_performance.png'}")
    plt.close()


def plot_faithfulness_scores(faith_summary: Path, output_dir: Path):
    """Plot faithfulness test results."""
    data = load_yaml_summary(faith_summary)

    # Extract scores by rule and model
    rules = []
    gpt_counter = []
    gpt_func = []
    claude_counter = []
    claude_func = []

    for rule_id, rule_data in data.items():
        rules.append(rule_id.split('_')[0][:10])  # Shorten names

        if GPTModels.GPT_4_1_NANO in rule_data:
            gpt_data = rule_data[GPTModels.GPT_4_1_NANO]
            gpt_counter.append(gpt_data.get('counterfactual_faithfulness', 0))
            gpt_func.append(gpt_data.get('functional_accuracy', 0))
        else:
            gpt_counter.append(0)
            gpt_func.append(0)

        if ClaudeModels.CLAUDE_HAIKU_4_5 in rule_data:
            claude_data = rule_data[ClaudeModels.CLAUDE_HAIKU_4_5]
            claude_counter.append(claude_data.get('counterfactual_faithfulness', 0))
            claude_func.append(claude_data.get('functional_accuracy', 0))
        else:
            claude_counter.append(0)
            claude_func.append(0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(len(rules))
    width = 0.35

    # Counterfactual faithfulness
    ax1.bar(x - width/2, gpt_counter, width, label='GPT-4.1-nano', color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, claude_counter, width, label='Claude-Haiku-4.5', color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('Counterfactual Faithfulness')
    ax1.set_title('Faithfulness: Counterfactual Prediction Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rules, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

    # Functional accuracy
    ax2.bar(x - width/2, gpt_func, width, label='GPT-4.1-nano', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, claude_func, width, label='Claude-Haiku-4.5', color='#2ecc71', alpha=0.8)
    ax2.set_ylabel('Functional Accuracy')
    ax2.set_title('Faithfulness: Functional Classification Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(rules, rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')

    plt.tight_layout()
    plt.savefig(output_dir / 'faithfulness_scores.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'faithfulness_scores.png'}")
    plt.close()


def plot_pipeline_comparison(output_dir: Path):
    """Plot model performance across entire pipeline."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Data (approximate averages from results)
    stages = ['Learnability\n(Accuracy)', 'Articulation\n(LLM Judge)', 'Articulation\n(Functional)',
              'Faithfulness\n(Counterfactual)', 'Faithfulness\n(Functional)']

    gpt_scores = [0.88, 0.585, 0.892, 0.39, 0.78]  # GPT-4.1-nano
    claude_scores = [0.42, 0.727, 0.974, 0.53, 0.99]  # Claude-Haiku-4.5

    x = np.arange(len(stages))
    width = 0.35

    ax.bar(x - width/2, gpt_scores, width, label='GPT-4.1-nano', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, claude_scores, width, label='Claude-Haiku-4.5', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Model Performance Across Research Pipeline', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.3, label='90% threshold')

    # Add grid
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pipeline_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pipeline_comparison.png'}")
    plt.close()


def plot_spurious_correlation_analysis(faith_summary: Path, output_dir: Path):
    """Plot spurious correlation failures."""
    data = load_yaml_summary(faith_summary)

    # Identify spurious failures (low counterfactual, high LLM judge from articulation)
    failures = []

    for rule_id, rule_data in data.items():
        for model, model_data in rule_data.items():
            counter = model_data.get('counterfactual_faithfulness', 0)
            func = model_data.get('functional_accuracy', 0)

            if counter < 0.5:  # Low counterfactual = spurious
                failures.append({
                    'rule': rule_id.split('_')[0][:15],
                    'model': 'GPT' if 'gpt' in model else 'Claude',
                    'counterfactual': counter,
                    'functional': func,
                    'gap': func - counter
                })

    # Sort by gap
    failures.sort(key=lambda x: x['gap'], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    rules_labels = [f"{f['rule']}\n({f['model']})" for f in failures]
    counter_scores = [f['counterfactual'] for f in failures]
    func_scores = [f['functional'] for f in failures]

    x = np.arange(len(failures))
    width = 0.4

    ax.bar(x - width/2, counter_scores, width, label='Counterfactual', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, func_scores, width, label='Functional', color='#3498db', alpha=0.8)

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Spurious Correlation Failures\n(Low Counterfactual Despite High Functional)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rules_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Highlight gap
    for i, f in enumerate(failures):
        if f['gap'] > 0.3:
            ax.annotate(f"Gap: {f['gap']:.0%}",
                       xy=(i, (f['counterfactual'] + f['functional'])/2),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'spurious_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'spurious_correlations.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create visualizations for research results")
    parser.add_argument(
        "--learnability-summary",
        type=Path,
        default=Path("experiments/learnability/learnable_rules.yaml"),
        help="Learnability summary YAML"
    )
    parser.add_argument(
        "--articulation-summary",
        type=Path,
        default=Path("experiments/articulation_freeform/summary.yaml"),
        help="Articulation summary YAML"
    )
    parser.add_argument(
        "--faithfulness-summary",
        type=Path,
        default=Path("experiments/faithfulness/summary_faithfulness.yaml"),
        help="Faithfulness summary YAML"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures"),
        help="Output directory for figures"
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Creating Visualizations for LLM Rule Articulation Study")
    print("=" * 80)
    print()

    # Generate plots
    print("1. Learnability by model...")
    plot_learnability_by_model(args.learnability_summary, args.output_dir)

    print("2. Articulation performance...")
    plot_articulation_performance(args.articulation_summary, args.output_dir)

    print("3. Faithfulness scores...")
    plot_faithfulness_scores(args.faithfulness_summary, args.output_dir)

    print("4. Pipeline comparison...")
    plot_pipeline_comparison(args.output_dir)

    print("5. Spurious correlation analysis...")
    plot_spurious_correlation_analysis(args.faithfulness_summary, args.output_dir)

    print()
    print("=" * 80)
    print(f"All visualizations saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
