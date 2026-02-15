"""
Evaluation orchestrator for all recommendation models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.sparse import csr_matrix
from tqdm import tqdm

from evaluation.metrics import (
    evaluate_all_metrics,
    aggregate_metrics,
    coverage,
    diversity,
    novelty
)
from utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for recommendation models.
    """

    def __init__(
        self,
        train_matrix: csr_matrix,
        test_matrix: csr_matrix,
        item_features: pd.DataFrame,
        k_values: List[int] = [5, 10, 20]
    ):
        """
        Initialize evaluator.

        Args:
            train_matrix: Training interaction matrix (users x items)
            test_matrix: Test interaction matrix (users x items)
            item_features: Item features DataFrame
            k_values: List of K values for ranking metrics
        """
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.item_features = item_features
        self.k_values = k_values
        self.n_users, self.n_items = train_matrix.shape

        # Precompute item popularity for novelty metric
        self.item_popularity = self._compute_item_popularity()

        # Precompute item categories for diversity metric
        self.item_categories = self._get_item_categories()

        logger.info(f"Initialized Evaluator: {self.n_users} users, {self.n_items} items, K={k_values}")

    def _compute_item_popularity(self) -> Dict[int, float]:
        """
        Compute item popularity scores (normalized interaction counts).

        Returns:
            Dictionary mapping item_idx -> popularity [0, 1]
        """
        # Count interactions per item
        item_counts = np.array(self.train_matrix.sum(axis=0)).flatten()
        total_interactions = item_counts.sum()

        # Normalize to [0, 1]
        popularity = {}
        for item_idx in range(self.n_items):
            popularity[item_idx] = item_counts[item_idx] / total_interactions if total_interactions > 0 else 0.0

        return popularity

    def _get_item_categories(self) -> Dict[int, int]:
        """
        Get item categories from features.

        Returns:
            Dictionary mapping item_idx -> category_idx
        """
        categories = {}
        if 'Department.Name' in self.item_features.columns:
            # Use department as category
            for idx, row in self.item_features.iterrows():
                categories[idx] = row.get('dept_encoded', 0)
        else:
            # All items in same category (fallback)
            for idx in range(self.n_items):
                categories[idx] = 0

        return categories

    def _get_user_train_history(self, user_idx: int) -> np.ndarray:
        """
        Get items a user interacted with in training set.

        Args:
            user_idx: User index

        Returns:
            Array of item indices
        """
        user_row = self.train_matrix[user_idx].toarray().flatten()
        return np.where(user_row > 0)[0]

    def _get_user_test_items(self, user_idx: int) -> np.ndarray:
        """
        Get ground truth items for a user from test set.

        Args:
            user_idx: User index

        Returns:
            Array of item indices
        """
        user_row = self.test_matrix[user_idx].toarray().flatten()
        return np.where(user_row > 0)[0]

    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        top_k: int = 20,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a recommendation model on test set.

        Args:
            model: Model with predict_batch() or recommend() method
            model_name: Name of the model (for logging)
            top_k: Number of recommendations to generate per user
            verbose: Show progress bar

        Returns:
            Dictionary of aggregated metrics
        """
        logger.info(f"Evaluating {model_name}...")

        user_metrics = []
        all_recommendations = []

        # Iterate over test users
        users_to_evaluate = range(self.n_users)
        if verbose:
            users_to_evaluate = tqdm(users_to_evaluate, desc=f"Evaluating {model_name}")

        for user_idx in users_to_evaluate:
            # Get ground truth
            test_items = self._get_user_test_items(user_idx)
            if len(test_items) == 0:
                continue  # Skip users with no test interactions

            # Get training history (to exclude from recommendations)
            train_items = self._get_user_train_history(user_idx)

            # Generate predictions for all items
            predictions = self._predict_for_user(model, user_idx)

            # Mask out training items (don't recommend what user already has)
            predictions[train_items] = -np.inf

            # Get top-K recommendations
            top_k_items = np.argsort(predictions)[::-1][:top_k]
            all_recommendations.append(top_k_items)

            # Compute metrics for this user
            metrics = evaluate_all_metrics(top_k_items, test_items, self.k_values)
            user_metrics.append(metrics)

        # Aggregate metrics across all users
        aggregated = aggregate_metrics(user_metrics)

        # Add catalog-level metrics
        aggregated['coverage'] = coverage(all_recommendations, self.n_items)
        aggregated['diversity'] = diversity(all_recommendations, self.item_categories)
        aggregated['novelty'] = novelty(all_recommendations, self.item_popularity)

        # Add metadata
        aggregated['n_users_evaluated'] = len(user_metrics)
        aggregated['model_name'] = model_name

        logger.info(f"{model_name} evaluation complete: {len(user_metrics)} users")
        logger.info(f"  Precision@10: {aggregated.get('precision@10', 0):.4f}")
        logger.info(f"  NDCG@10: {aggregated.get('ndcg@10', 0):.4f}")
        logger.info(f"  Coverage: {aggregated['coverage']:.4f}")

        return aggregated

    def _predict_for_user(self, model: Any, user_idx: int) -> np.ndarray:
        """
        Get predictions for all items for a single user.

        Args:
            model: Model instance
            user_idx: User index

        Returns:
            Array of prediction scores (length = n_items)
        """
        # Create array of all item indices
        item_indices = np.arange(self.n_items)

        # Check if model has predict_batch method
        if hasattr(model, 'predict_batch'):
            predictions = model.predict_batch(user_idx, item_indices)
        elif hasattr(model, 'predict'):
            # Fallback to predict method
            predictions = np.array([model.predict(user_idx, item_idx) for item_idx in item_indices])
        elif hasattr(model, 'recommend'):
            # For popularity baseline - get scores directly
            predictions = model.get_scores(user_idx)
        else:
            raise ValueError(f"Model does not have predict_batch, predict, or recommend method")

        return predictions

    def compare_models(
        self,
        models: List[Tuple[Any, str]],
        save_path: str = "results/metrics/model_comparison.csv"
    ) -> pd.DataFrame:
        """
        Evaluate and compare multiple models.

        Args:
            models: List of (model, model_name) tuples
            save_path: Path to save comparison CSV

        Returns:
            DataFrame with comparison results
        """
        all_results = []

        for model, model_name in models:
            metrics = self.evaluate_model(model, model_name)
            all_results.append(metrics)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)

        # Reorder columns for readability
        metric_order = ['model_name', 'n_users_evaluated']
        for k in self.k_values:
            metric_order.extend([
                f'precision@{k}',
                f'recall@{k}',
                f'ndcg@{k}',
                f'hit_rate@{k}',
                f'map@{k}'
            ])
        metric_order.extend(['coverage', 'diversity', 'novelty'])

        # Only keep columns that exist
        metric_order = [col for col in metric_order if col in comparison_df.columns]
        comparison_df = comparison_df[metric_order]

        # Save to CSV
        comparison_df.to_csv(save_path, index=False)
        logger.info(f"Model comparison saved to: {save_path}")

        return comparison_df
