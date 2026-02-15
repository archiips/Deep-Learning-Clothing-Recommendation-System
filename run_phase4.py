"""
Master script for Phase 4: Model Development
Trains all models, evaluates, and generates comparison visualizations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Model training
from models.popularity import train_popularity_baseline, PopularityBaseline
from training.train_mf import train_matrix_factorization
from training.train_ncf import train_neural_cf

# Evaluation
from evaluation.evaluator import Evaluator
from evaluation.visualize import create_all_visualizations

# Utilities
from utils.data_loader import DataManager
from utils.logger import setup_logger

logger = setup_logger(__name__)


def print_banner(text: str):
    """Print a decorative banner."""
    logger.info("=" * 80)
    logger.info(f"  {text}")
    logger.info("=" * 80)


def train_all_models(dataset_dir: str = "dataset"):
    """
    Train all three models.

    Args:
        dataset_dir: Directory with dataset files

    Returns:
        Dictionary of trained models
    """
    models = {}

    # 1. Popularity Baseline
    print_banner("TRAINING POPULARITY BASELINE")
    start_time = time.time()
    try:
        pop_model = train_popularity_baseline(
            dataset_dir=dataset_dir,
            checkpoint_dir="checkpoints/popularity"
        )
        models['Popularity'] = pop_model
        pop_time = time.time() - start_time
        logger.info(f"Popularity Baseline trained in {pop_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to train Popularity Baseline: {e}")
        raise

    # 2. Matrix Factorization
    print_banner("TRAINING MATRIX FACTORIZATION")
    start_time = time.time()
    try:
        mf_model = train_matrix_factorization(
            config_path="configs/mf_config.yaml",
            dataset_dir=dataset_dir
        )
        models['MatrixFactorization'] = mf_model
        mf_time = time.time() - start_time
        logger.info(f"Matrix Factorization trained in {mf_time:.2f}s ({mf_time/60:.1f} min)")
    except Exception as e:
        logger.error(f"Failed to train Matrix Factorization: {e}")
        raise

    # 3. Neural Collaborative Filtering
    print_banner("TRAINING NEURAL COLLABORATIVE FILTERING")
    start_time = time.time()
    try:
        ncf_model = train_neural_cf(
            config_path="configs/ncf_config.yaml",
            dataset_dir=dataset_dir
        )
        models['NeuralCF'] = ncf_model
        ncf_time = time.time() - start_time
        logger.info(f"Neural CF trained in {ncf_time:.2f}s ({ncf_time/60:.1f} min)")
    except Exception as e:
        logger.error(f"Failed to train Neural CF: {e}")
        raise

    return models


def evaluate_all_models(models: dict, dataset_dir: str = "dataset"):
    """
    Evaluate all models and create comparison.

    Args:
        models: Dictionary of trained models
        dataset_dir: Directory with dataset files

    Returns:
        Comparison DataFrame
    """
    print_banner("EVALUATING ALL MODELS")

    # Load data for evaluation
    data_manager = DataManager(dataset_dir)
    data = data_manager.load_all()

    # Create evaluator
    evaluator = Evaluator(
        train_matrix=data['train_matrix'],
        test_matrix=data['test_matrix'],
        item_features=data['item_features'],
        k_values=[5, 10, 20]
    )

    # Evaluate each model
    models_to_evaluate = [
        (models['Popularity'], 'Popularity'),
        (models['MatrixFactorization'], 'MatrixFactorization'),
        (models['NeuralCF'], 'NeuralCF')
    ]

    comparison_df = evaluator.compare_models(
        models=models_to_evaluate,
        save_path="results/metrics/model_comparison.csv"
    )

    return comparison_df


def print_results(comparison_df: pd.DataFrame):
    """
    Print formatted comparison results.

    Args:
        comparison_df: Comparison DataFrame
    """
    print_banner("MODEL COMPARISON RESULTS")

    # Format for display
    display_cols = [
        'model_name',
        'precision@5', 'precision@10', 'precision@20',
        'ndcg@5', 'ndcg@10', 'ndcg@20',
        'hit_rate@10',
        'coverage', 'diversity', 'novelty'
    ]

    # Filter to available columns
    display_cols = [col for col in display_cols if col in comparison_df.columns]

    display_df = comparison_df[display_cols].copy()

    # Round numeric columns
    for col in display_df.columns:
        if col != 'model_name':
            display_df[col] = display_df[col].round(4)

    # Print table
    print("\n" + display_df.to_string(index=False) + "\n")

    # Print key metrics
    logger.info("\nKey Metrics (Precision@10):")
    for _, row in display_df.iterrows():
        model_name = row['model_name']
        p10 = row.get('precision@10', 0)
        ndcg10 = row.get('ndcg@10', 0)
        logger.info(f"  {model_name:25s} P@10={p10:.4f}  NDCG@10={ndcg10:.4f}")


def generate_visualizations(comparison_df: pd.DataFrame):
    """
    Generate all visualization plots.

    Args:
        comparison_df: Comparison DataFrame
    """
    print_banner("GENERATING VISUALIZATIONS")

    try:
        create_all_visualizations(
            comparison_df=comparison_df,
            k_values=[5, 10, 20],
            output_dir="results/visualizations"
        )
        logger.info("All visualizations generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")
        raise


def main():
    """
    Main execution pipeline for Phase 4.
    """
    print_banner("PHASE 4: MODEL DEVELOPMENT")

    start_time = time.time()

    # Step 1: Train all models
    models = train_all_models()

    # Step 2: Evaluate all models
    comparison_df = evaluate_all_models(models)

    # Step 3: Print results
    print_results(comparison_df)

    # Step 4: Generate visualizations
    generate_visualizations(comparison_df)

    # Summary
    total_time = time.time() - start_time
    print_banner(f"PHASE 4 COMPLETE (Total time: {total_time/60:.1f} minutes)")

    logger.info("All outputs saved:")
    logger.info("  - Models: checkpoints/popularity/, checkpoints/mf/, checkpoints/ncf/")
    logger.info("  - Metrics: results/metrics/model_comparison.csv")
    logger.info("  - Visualizations: results/visualizations/*.png")
    logger.info("  - Logs: logs/*.log")


if __name__ == "__main__":
    main()
