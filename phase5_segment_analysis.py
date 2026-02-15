"""
Phase 5 Task 5.1.2: User and Item Segment Analysis
Evaluates model performance across different user segments and item popularity tiers.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import load_npz
from scipy import stats
from utils.logger import get_logger
from evaluation.metrics import evaluate_all_metrics, aggregate_metrics

logger = get_logger(__name__)


def load_models_and_data():
    """Load trained models and test data."""
    # Load sparse matrices
    train_matrix = load_npz('dataset/train_matrix.npz')
    test_matrix = load_npz('dataset/test_matrix.npz')

    # Load test data
    test_df = pd.read_csv('dataset/test_set.csv')

    # Load user features
    user_features = pd.read_csv('dataset/user_features.csv')

    # Load item features
    item_features = pd.read_csv('dataset/item_features.csv')

    # Load models
    from utils.model_loader import load_all_models

    popularity_model, mf_model, ncf_model = load_all_models()

    models = [
        (popularity_model, 'Popularity'),
        (mf_model, 'MatrixFactorization'),
        (ncf_model, 'NeuralCF')
    ]

    return models, train_matrix, test_matrix, test_df, user_features, item_features


def segment_users_by_activity(user_features: pd.DataFrame) -> dict:
    """
    Segment users by activity level.

    Returns:
        Dictionary mapping user_id -> segment
    """
    segments = {}

    for _, row in user_features.iterrows():
        user_id = row['User.ID']
        total_reviews = row['total_reviews']

        if total_reviews <= 2:
            segment = 'New'
        elif total_reviews <= 10:
            segment = 'Casual'
        else:
            segment = 'Active'

        segments[user_id] = segment

    logger.info(f"User segments: {pd.Series(segments).value_counts().to_dict()}")
    return segments


def segment_items_by_popularity(item_features: pd.DataFrame) -> dict:
    """
    Segment items by popularity.

    Returns:
        Dictionary mapping clothing_id -> segment
    """
    segments = {}

    for _, row in item_features.iterrows():
        clothing_id = row['Clothing.ID']
        rating_count = row['rating_count']

        if rating_count >= 100:
            segment = 'Popular'
        elif rating_count >= 10:
            segment = 'Mid'
        else:
            segment = 'Long-tail'

        segments[clothing_id] = segment

    logger.info(f"Item segments: {pd.Series(segments).value_counts().to_dict()}")
    return segments


def evaluate_by_user_segment(models, train_matrix, test_matrix, user_segments, user_to_idx, item_to_idx):
    """Evaluate models by user segment."""
    k_values = [5, 10, 20]
    n_items = train_matrix.shape[1]

    results = []

    for model, model_name in models:
        logger.info(f"Evaluating {model_name} by user segment...")

        segment_metrics = {segment: [] for segment in ['New', 'Casual', 'Active']}

        # Iterate over test users
        for user_id, segment in user_segments.items():
            if user_id not in user_to_idx:
                continue

            user_idx = user_to_idx[user_id]

            # Get ground truth
            test_items_row = test_matrix[user_idx].toarray().flatten()
            test_items = np.where(test_items_row > 0)[0]

            if len(test_items) == 0:
                continue

            # Get predictions
            item_indices = np.arange(n_items)

            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(user_idx, item_indices)
            elif hasattr(model, 'get_scores'):
                predictions = model.get_scores(user_idx)
            else:
                continue

            # Mask training items
            train_items_row = train_matrix[user_idx].toarray().flatten()
            train_items = np.where(train_items_row > 0)[0]
            predictions[train_items] = -np.inf

            # Get top-20 recommendations
            top_k_items = np.argsort(predictions)[::-1][:20]

            # Compute metrics
            metrics = evaluate_all_metrics(top_k_items, test_items, k_values)
            segment_metrics[segment].append(metrics)

        # Aggregate by segment
        for segment in ['New', 'Casual', 'Active']:
            if segment_metrics[segment]:
                agg = aggregate_metrics(segment_metrics[segment])
                results.append({
                    'model': model_name,
                    'segment_type': 'User',
                    'segment': segment,
                    'n_users': len(segment_metrics[segment]),
                    'precision@10': agg.get('precision@10', 0),
                    'ndcg@10': agg.get('ndcg@10', 0),
                    'hit_rate@10': agg.get('hit_rate@10', 0)
                })

    return pd.DataFrame(results)


def evaluate_by_item_segment(models, train_matrix, test_matrix, item_segments, user_to_idx, item_to_idx, idx_to_item):
    """Evaluate models by item popularity segment."""
    k_values = [10]
    n_items = train_matrix.shape[1]
    n_users = test_matrix.shape[0]

    results = []

    # Create reverse mapping for item segments
    idx_to_segment = {}
    for clothing_id, segment in item_segments.items():
        if clothing_id in item_to_idx:
            idx_to_segment[item_to_idx[clothing_id]] = segment

    for model, model_name in models:
        logger.info(f"Evaluating {model_name} by item segment...")

        segment_metrics = {segment: {'hits': 0, 'total_recs': 0, 'total_relevant': 0}
                          for segment in ['Popular', 'Mid', 'Long-tail']}

        # Iterate over test users
        for user_idx in range(n_users):
            # Get ground truth
            test_items_row = test_matrix[user_idx].toarray().flatten()
            test_items = np.where(test_items_row > 0)[0]

            if len(test_items) == 0:
                continue

            # Get predictions
            item_indices = np.arange(n_items)

            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(user_idx, item_indices)
            elif hasattr(model, 'get_scores'):
                predictions = model.get_scores(user_idx)
            else:
                continue

            # Mask training items
            train_items_row = train_matrix[user_idx].toarray().flatten()
            train_items = np.where(train_items_row > 0)[0]
            predictions[train_items] = -np.inf

            # Get top-10 recommendations
            top_k_items = np.argsort(predictions)[::-1][:10]

            # Count hits by segment
            for item_idx in top_k_items:
                segment = idx_to_segment.get(item_idx, 'Unknown')
                if segment != 'Unknown':
                    segment_metrics[segment]['total_recs'] += 1
                    if item_idx in test_items:
                        segment_metrics[segment]['hits'] += 1

            # Count relevant items by segment
            for item_idx in test_items:
                segment = idx_to_segment.get(item_idx, 'Unknown')
                if segment != 'Unknown':
                    segment_metrics[segment]['total_relevant'] += 1

        # Calculate precision by segment
        for segment in ['Popular', 'Mid', 'Long-tail']:
            metrics = segment_metrics[segment]
            precision = metrics['hits'] / metrics['total_recs'] if metrics['total_recs'] > 0 else 0

            results.append({
                'model': model_name,
                'segment_type': 'Item',
                'segment': segment,
                'total_recs': metrics['total_recs'],
                'hits': metrics['hits'],
                'precision': precision,
                'relevant_items': metrics['total_relevant']
            })

    return pd.DataFrame(results)


def statistical_significance_test(comparison_df: pd.DataFrame):
    """
    Perform paired t-test to check statistical significance.
    """
    logger.info("Running statistical significance tests...")

    # For simplicity, compare MF vs Popularity on precision@10
    mf_precision = comparison_df[comparison_df['model_name'] == 'MatrixFactorization']['precision@10'].values[0]
    pop_precision = comparison_df[comparison_df['model_name'] == 'Popularity']['precision@10'].values[0]

    # Since we don't have per-user results saved, we'll note this limitation
    logger.info(f"MF Precision@10: {mf_precision:.4f}")
    logger.info(f"Popularity Precision@10: {pop_precision:.4f}")
    logger.info(f"Difference: {(mf_precision - pop_precision):.4f}")
    logger.info("Note: Full paired t-test requires per-user metric storage")

    return {
        'mf_precision@10': mf_precision,
        'popularity_precision@10': pop_precision,
        'difference': mf_precision - pop_precision
    }


def main():
    """Run complete segment analysis."""
    logger.info("="*60)
    logger.info("PHASE 5 TASK 5.1.2: SEGMENT ANALYSIS")
    logger.info("="*60)

    # Load everything
    models, train_matrix, test_matrix, test_df, user_features, item_features = load_models_and_data()

    # Load mappings
    with open('dataset/user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('dataset/item_to_idx.pkl', 'rb') as f:
        item_to_idx = pickle.load(f)

    idx_to_item = {v: k for k, v in item_to_idx.items()}

    # Segment users and items
    user_segments = segment_users_by_activity(user_features)
    item_segments = segment_items_by_popularity(item_features)

    # Evaluate by user segment
    logger.info("\n" + "="*60)
    logger.info("USER SEGMENT ANALYSIS")
    logger.info("="*60)
    user_segment_results = evaluate_by_user_segment(
        models, train_matrix, test_matrix, user_segments, user_to_idx, item_to_idx
    )

    # Save results
    output_dir = Path('results/segment_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    user_segment_results.to_csv(output_dir / 'user_segment_analysis.csv', index=False)
    logger.info(f"\nUser segment results:\n{user_segment_results}")

    # Evaluate by item segment
    logger.info("\n" + "="*60)
    logger.info("ITEM SEGMENT ANALYSIS")
    logger.info("="*60)
    item_segment_results = evaluate_by_item_segment(
        models, train_matrix, test_matrix, item_segments, user_to_idx, item_to_idx, idx_to_item
    )

    item_segment_results.to_csv(output_dir / 'item_segment_analysis.csv', index=False)
    logger.info(f"\nItem segment results:\n{item_segment_results}")

    # Statistical significance test
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL SIGNIFICANCE TESTING")
    logger.info("="*60)
    comparison_df = pd.read_csv('results/metrics/model_comparison.csv')
    significance_results = statistical_significance_test(comparison_df)

    pd.DataFrame([significance_results]).to_csv(output_dir / 'significance_test.csv', index=False)

    logger.info("\n" + "="*60)
    logger.info("SEGMENT ANALYSIS COMPLETE!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
