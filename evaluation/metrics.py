"""
Evaluation metrics for recommendation systems.
Implements ranking metrics: Precision@K, Recall@K, NDCG@K, Hit Rate, MAP, Coverage, Diversity, Novelty.
"""
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(recommendations: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate Precision@K.

    Args:
        recommendations: Array of recommended item indices (sorted by score)
        ground_truth: Array of ground truth item indices
        k: Number of top recommendations to consider

    Returns:
        Precision@K value [0, 1]

    Example:
        >>> recommendations = np.array([1, 3, 5, 7, 9])
        >>> ground_truth = np.array([1, 5, 10])
        >>> precision_at_k(recommendations, ground_truth, k=5)
        0.4  # 2 out of 5 recommendations are relevant
    """
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = recommendations[:k]
    ground_truth_set = set(ground_truth)
    relevant_in_top_k = len(set(top_k) & ground_truth_set)

    return relevant_in_top_k / k


def recall_at_k(recommendations: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        recommendations: Array of recommended item indices (sorted by score)
        ground_truth: Array of ground truth item indices
        k: Number of top recommendations to consider

    Returns:
        Recall@K value [0, 1]

    Example:
        >>> recommendations = np.array([1, 3, 5, 7, 9])
        >>> ground_truth = np.array([1, 5, 10])
        >>> recall_at_k(recommendations, ground_truth, k=5)
        0.667  # 2 out of 3 ground truth items are in top-5
    """
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = recommendations[:k]
    ground_truth_set = set(ground_truth)
    relevant_in_top_k = len(set(top_k) & ground_truth_set)

    return relevant_in_top_k / len(ground_truth)


def hit_rate_at_k(recommendations: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate Hit Rate@K (binary: 1 if any ground truth item is in top-K, else 0).

    Args:
        recommendations: Array of recommended item indices (sorted by score)
        ground_truth: Array of ground truth item indices
        k: Number of top recommendations to consider

    Returns:
        1.0 if hit, 0.0 otherwise

    Example:
        >>> recommendations = np.array([1, 3, 5, 7, 9])
        >>> ground_truth = np.array([1, 5, 10])
        >>> hit_rate_at_k(recommendations, ground_truth, k=5)
        1.0  # At least one ground truth item is in top-5
    """
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = recommendations[:k]
    ground_truth_set = set(ground_truth)

    return 1.0 if len(set(top_k) & ground_truth_set) > 0 else 0.0


def ndcg_at_k(recommendations: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K.

    Args:
        recommendations: Array of recommended item indices (sorted by score)
        ground_truth: Array of ground truth item indices
        k: Number of top recommendations to consider

    Returns:
        NDCG@K value [0, 1]

    Example:
        >>> recommendations = np.array([1, 3, 5, 7, 9])
        >>> ground_truth = np.array([1, 5, 10])
        >>> ndcg_at_k(recommendations, ground_truth, k=5)
        0.756  # DCG normalized by ideal DCG
    """
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = recommendations[:k]
    ground_truth_set = set(ground_truth)

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth_set:
            # Relevance is 1 for binary feedback, position is i+1
            dcg += 1.0 / np.log2(i + 2)  # +2 because positions start at 1, log2(1) is 0

    # Calculate Ideal DCG
    idcg = 0.0
    for i in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def map_at_k(recommendations: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate Mean Average Precision@K.

    Args:
        recommendations: Array of recommended item indices (sorted by score)
        ground_truth: Array of ground truth item indices
        k: Number of top recommendations to consider

    Returns:
        MAP@K value [0, 1]

    Example:
        >>> recommendations = np.array([1, 3, 5, 7, 9])
        >>> ground_truth = np.array([1, 5, 10])
        >>> map_at_k(recommendations, ground_truth, k=5)
        0.633  # Average of precision values at relevant positions
    """
    if len(recommendations) == 0 or len(ground_truth) == 0:
        return 0.0

    top_k = recommendations[:k]
    ground_truth_set = set(ground_truth)

    precision_sum = 0.0
    num_hits = 0

    for i, item in enumerate(top_k):
        if item in ground_truth_set:
            num_hits += 1
            # Precision at this position
            precision_at_i = num_hits / (i + 1)
            precision_sum += precision_at_i

    if num_hits == 0:
        return 0.0

    return precision_sum / min(len(ground_truth), k)


def coverage(all_recommendations: List[np.ndarray], n_items: int) -> float:
    """
    Calculate catalog coverage (percentage of items recommended to at least one user).

    Args:
        all_recommendations: List of recommendation arrays (one per user)
        n_items: Total number of items in catalog

    Returns:
        Coverage percentage [0, 1]

    Example:
        >>> recs = [np.array([1, 2, 3]), np.array([2, 4, 5]), np.array([1, 5, 6])]
        >>> coverage(recs, n_items=10)
        0.6  # 6 unique items recommended out of 10
    """
    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)

    return len(unique_items) / n_items


def diversity(all_recommendations: List[np.ndarray], item_categories: Dict[int, int]) -> float:
    """
    Calculate category diversity (average number of unique categories per user).

    Args:
        all_recommendations: List of recommendation arrays (one per user)
        item_categories: Dictionary mapping item_idx -> category_idx

    Returns:
        Average number of unique categories [0, max_categories]

    Example:
        >>> recs = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> categories = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
        >>> diversity(recs, categories)
        2.0  # Average: (2 + 2) / 2
    """
    if not all_recommendations:
        return 0.0

    diversity_scores = []
    for recs in all_recommendations:
        categories_in_recs = set()
        for item in recs:
            if item in item_categories:
                categories_in_recs.add(item_categories[item])
        diversity_scores.append(len(categories_in_recs))

    return np.mean(diversity_scores)


def novelty(all_recommendations: List[np.ndarray], item_popularity: Dict[int, float]) -> float:
    """
    Calculate novelty (average -log(popularity) of recommended items).
    Higher novelty = recommending less popular (more novel) items.

    Args:
        all_recommendations: List of recommendation arrays (one per user)
        item_popularity: Dictionary mapping item_idx -> popularity score [0, 1]

    Returns:
        Average novelty score (higher is more novel)

    Example:
        >>> recs = [np.array([1, 2, 3]), np.array([4, 5])]
        >>> popularity = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.2, 5: 0.05}
        >>> novelty(recs, popularity)
        # Higher value indicates more novel (less popular) items
    """
    if not all_recommendations:
        return 0.0

    novelty_scores = []
    for recs in all_recommendations:
        for item in recs:
            if item in item_popularity and item_popularity[item] > 0:
                # -log(popularity): high popularity -> low novelty
                novelty_scores.append(-np.log2(item_popularity[item] + 1e-10))

    if not novelty_scores:
        return 0.0

    return np.mean(novelty_scores)


def evaluate_all_metrics(
    recommendations: np.ndarray,
    ground_truth: np.ndarray,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate all ranking metrics for a single user.

    Args:
        recommendations: Array of recommended item indices (sorted by score)
        ground_truth: Array of ground truth item indices
        k_values: List of K values to evaluate

    Returns:
        Dictionary of metric_name -> value

    Example:
        >>> recs = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
        >>> truth = np.array([1, 5, 10])
        >>> metrics = evaluate_all_metrics(recs, truth, k_values=[5, 10])
        >>> print(metrics['precision@5'], metrics['ndcg@10'])
    """
    metrics = {}

    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(recommendations, ground_truth, k)
        metrics[f'recall@{k}'] = recall_at_k(recommendations, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(recommendations, ground_truth, k)
        metrics[f'hit_rate@{k}'] = hit_rate_at_k(recommendations, ground_truth, k)
        metrics[f'map@{k}'] = map_at_k(recommendations, ground_truth, k)

    return metrics


def aggregate_metrics(user_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across all users (take mean).

    Args:
        user_metrics: List of metric dictionaries (one per user)

    Returns:
        Dictionary of aggregated metrics

    Example:
        >>> user1 = {'precision@10': 0.2, 'ndcg@10': 0.3}
        >>> user2 = {'precision@10': 0.4, 'ndcg@10': 0.5}
        >>> aggregate_metrics([user1, user2])
        {'precision@10': 0.3, 'ndcg@10': 0.4}
    """
    if not user_metrics:
        return {}

    aggregated = defaultdict(list)

    for metrics in user_metrics:
        for metric_name, value in metrics.items():
            aggregated[metric_name].append(value)

    # Calculate mean for each metric
    result = {}
    for metric_name, values in aggregated.items():
        result[metric_name] = np.mean(values)
        result[f'{metric_name}_std'] = np.std(values)

    return result
