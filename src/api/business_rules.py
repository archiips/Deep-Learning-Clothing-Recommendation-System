"""
Business rules for filtering and enhancing recommendations.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter


def apply_department_filter(
    item_indices: np.ndarray,
    item_features_df: pd.DataFrame,
    department: str
) -> np.ndarray:
    """
    Filter items by department.

    Args:
        item_indices: Array of item indices
        item_features_df: DataFrame with item features
        department: Department name to filter by

    Returns:
        Filtered array of item indices
    """
    if department is None:
        return item_indices

    # Get items in the specified department
    dept_items = item_features_df[item_features_df['department'] == department].index.values

    # Filter to only include items in this department
    filtered_items = [idx for idx in item_indices if idx in dept_items]

    return np.array(filtered_items, dtype=int)


def enforce_diversity(
    item_indices: np.ndarray,
    item_features_df: pd.DataFrame,
    max_per_department: int = 3,
    max_per_class: int = 2
) -> np.ndarray:
    """
    Enforce diversity by limiting items per department and class.

    Args:
        item_indices: Array of item indices (ordered by score)
        item_features_df: DataFrame with item features
        max_per_department: Maximum items per department
        max_per_class: Maximum items per class

    Returns:
        Filtered array of item indices with diversity enforced
    """
    if len(item_indices) == 0:
        return item_indices

    diversified = []
    dept_counts = Counter()
    class_counts = Counter()

    for item_idx in item_indices:
        if item_idx >= len(item_features_df):
            continue

        item_row = item_features_df.iloc[item_idx]
        dept = item_row.get('department', 'Unknown')
        cls = item_row.get('class_name', 'Unknown')

        # Check department limit
        if dept_counts[dept] >= max_per_department:
            continue

        # Check class limit
        if class_counts[cls] >= max_per_class:
            continue

        # Add item and update counts
        diversified.append(item_idx)
        dept_counts[dept] += 1
        class_counts[cls] += 1

    return np.array(diversified, dtype=int)


def filter_low_quality_items(
    item_indices: np.ndarray,
    item_features_df: pd.DataFrame,
    min_avg_rating: float = 3.0,
    min_num_reviews: int = 3
) -> np.ndarray:
    """
    Filter out low-quality items based on ratings and review count.

    Args:
        item_indices: Array of item indices
        item_features_df: DataFrame with item features
        min_avg_rating: Minimum average rating threshold
        min_num_reviews: Minimum number of reviews threshold

    Returns:
        Filtered array of item indices
    """
    filtered = []

    for item_idx in item_indices:
        if item_idx >= len(item_features_df):
            continue

        item_row = item_features_df.iloc[item_idx]
        avg_rating = item_row.get('avg_rating', 0.0)
        # Use rating_count instead of num_reviews
        num_reviews = item_row.get('rating_count', item_row.get('num_reviews', 0))

        # Apply quality filters
        if avg_rating >= min_avg_rating and num_reviews >= min_num_reviews:
            filtered.append(item_idx)

    return np.array(filtered, dtype=int)


def apply_business_rules(
    item_indices: np.ndarray,
    item_features_df: pd.DataFrame,
    department_filter: Optional[str] = None,
    enforce_quality: bool = True,
    enforce_item_diversity: bool = True
) -> np.ndarray:
    """
    Apply all business rules to recommendation list.

    Args:
        item_indices: Array of item indices (ordered by predicted score)
        item_features_df: DataFrame with item features
        department_filter: Department to filter by (optional)
        enforce_quality: Whether to filter low-quality items
        enforce_item_diversity: Whether to enforce diversity constraints

    Returns:
        Filtered and processed array of item indices
    """
    # Start with all items
    filtered_items = item_indices.copy()

    # Apply department filter if specified
    if department_filter:
        filtered_items = apply_department_filter(
            filtered_items,
            item_features_df,
            department_filter
        )

    # Filter low-quality items
    if enforce_quality:
        filtered_items = filter_low_quality_items(
            filtered_items,
            item_features_df,
            min_avg_rating=3.0,
            min_num_reviews=3
        )

    # Enforce diversity
    if enforce_item_diversity:
        filtered_items = enforce_diversity(
            filtered_items,
            item_features_df,
            max_per_department=4,
            max_per_class=2
        )

    return filtered_items


def get_item_metadata(
    item_idx: int,
    item_features_df: pd.DataFrame,
    idx_to_item: Dict[int, int]
) -> Dict:
    """
    Get metadata for a single item.

    Args:
        item_idx: Item index
        item_features_df: DataFrame with item features
        idx_to_item: Mapping from index to clothing ID

    Returns:
        Dictionary with item metadata
    """
    if item_idx >= len(item_features_df):
        return {
            'clothing_id': idx_to_item.get(item_idx, -1),
            'department': 'Unknown',
            'class_name': 'Unknown',
            'avg_rating': 0.0,
            'num_reviews': 0
        }

    item_row = item_features_df.iloc[item_idx]

    return {
        'clothing_id': idx_to_item.get(item_idx, item_row.get('Clothing.ID', -1)),
        'department': item_row.get('department', 'Unknown'),
        'class_name': item_row.get('class_name', 'Unknown'),
        'avg_rating': float(item_row.get('avg_rating', 0.0)),
        # Use rating_count instead of num_reviews
        'num_reviews': int(item_row.get('rating_count', item_row.get('num_reviews', 0)))
    }
