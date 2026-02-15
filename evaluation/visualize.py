"""
Visualization utilities for model comparison and analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metric_name: str,
    k_values: List[int],
    save_path: str
):
    """
    Plot comparison of a metric across models and K values.

    Args:
        comparison_df: DataFrame with model comparison results
        metric_name: Base metric name (e.g., "precision", "ndcg")
        k_values: List of K values
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = comparison_df['model_name'].values
    x = np.arange(len(models))
    width = 0.25

    for i, k in enumerate(k_values):
        col_name = f'{metric_name}@{k}'
        if col_name in comparison_df.columns:
            values = comparison_df[col_name].values
            ax.bar(x + i * width, values, width, label=f'@{k}')

    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved {metric_name} comparison plot: {save_path}")


def plot_all_metrics_heatmap(
    comparison_df: pd.DataFrame,
    save_path: str
):
    """
    Create heatmap of all metrics across models.

    Args:
        comparison_df: DataFrame with model comparison results
        save_path: Path to save plot
    """
    # Select only metric columns (exclude metadata)
    metric_cols = [col for col in comparison_df.columns
                   if col not in ['model_name', 'n_users_evaluated']
                   and not col.endswith('_std')]

    # Create data matrix
    data = comparison_df[metric_cols].values
    models = comparison_df['model_name'].values

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        data.T,
        annot=True,
        fmt='.3f',
        cmap='YlGnBu',
        xticklabels=models,
        yticklabels=metric_cols,
        cbar_kws={'label': 'Score'},
        ax=ax
    )

    ax.set_title('Model Comparison Heatmap')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved metrics heatmap: {save_path}")


def plot_ranking_metrics_comparison(
    comparison_df: pd.DataFrame,
    k: int,
    save_path: str
):
    """
    Plot radar chart comparing ranking metrics at specific K.

    Args:
        comparison_df: DataFrame with model comparison results
        k: K value to visualize
        save_path: Path to save plot
    """
    metrics = ['precision', 'recall', 'ndcg', 'hit_rate', 'map']
    metric_cols = [f'{m}@{k}' for m in metrics]

    # Filter to available metrics
    metric_cols = [col for col in metric_cols if col in comparison_df.columns]

    if len(metric_cols) < 3:
        logger.warning(f"Not enough metrics for radar chart at K={k}")
        return

    models = comparison_df['model_name'].values
    n_models = len(models)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Angle for each metric
    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Plot each model
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        values = comparison_df.loc[comparison_df['model_name'] == model, metric_cols].values[0].tolist()
        values += values[:1]  # Close the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Customize plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([col.replace('@', ' @') for col in metric_cols])
    ax.set_ylim(0, 1)
    ax.set_title(f'Ranking Metrics Comparison @{k}', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved radar chart: {save_path}")


def plot_catalog_metrics(
    comparison_df: pd.DataFrame,
    save_path: str
):
    """
    Plot catalog-level metrics (coverage, diversity, novelty).

    Args:
        comparison_df: DataFrame with model comparison results
        save_path: Path to save plot
    """
    catalog_metrics = ['coverage', 'diversity', 'novelty']
    available_metrics = [m for m in catalog_metrics if m in comparison_df.columns]

    if not available_metrics:
        logger.warning("No catalog metrics available to plot")
        return

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))
    if len(available_metrics) == 1:
        axes = [axes]

    models = comparison_df['model_name'].values
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for i, metric in enumerate(available_metrics):
        values = comparison_df[metric].values
        axes[i].bar(models, values, color=colors)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=15)
        axes[i].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved catalog metrics plot: {save_path}")


def plot_precision_recall_curve(
    comparison_df: pd.DataFrame,
    k_values: List[int],
    save_path: str
):
    """
    Plot Precision-Recall curve across K values.

    Args:
        comparison_df: DataFrame with model comparison results
        k_values: List of K values
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = comparison_df['model_name'].values
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        model_data = comparison_df[comparison_df['model_name'] == model]

        precisions = []
        recalls = []
        for k in k_values:
            p_col = f'precision@{k}'
            r_col = f'recall@{k}'
            if p_col in model_data.columns and r_col in model_data.columns:
                precisions.append(model_data[p_col].values[0])
                recalls.append(model_data[r_col].values[0])

        if precisions and recalls:
            ax.plot(recalls, precisions, 'o-', label=model, color=colors[i], linewidth=2, markersize=8)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Across K Values')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved precision-recall curve: {save_path}")


def create_all_visualizations(
    comparison_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    output_dir: str = "results/visualizations"
):
    """
    Create all visualization plots.

    Args:
        comparison_df: DataFrame with model comparison results
        k_values: List of K values
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating visualizations...")

    # 1. Precision comparison
    plot_metric_comparison(
        comparison_df, 'precision', k_values,
        str(output_path / 'precision_comparison.png')
    )

    # 2. NDCG comparison
    plot_metric_comparison(
        comparison_df, 'ndcg', k_values,
        str(output_path / 'ndcg_comparison.png')
    )

    # 3. Hit Rate comparison
    plot_metric_comparison(
        comparison_df, 'hit_rate', k_values,
        str(output_path / 'hit_rate_comparison.png')
    )

    # 4. All metrics heatmap
    plot_all_metrics_heatmap(
        comparison_df,
        str(output_path / 'metrics_heatmap.png')
    )

    # 5. Radar chart for K=10
    plot_ranking_metrics_comparison(
        comparison_df,
        10,
        str(output_path / 'radar_chart_k10.png')
    )

    # 6. Catalog metrics
    plot_catalog_metrics(
        comparison_df,
        str(output_path / 'catalog_metrics.png')
    )

    # 7. Precision-Recall curve
    plot_precision_recall_curve(
        comparison_df, k_values,
        str(output_path / 'precision_recall_curve.png')
    )

    logger.info(f"All visualizations saved to: {output_dir}")
