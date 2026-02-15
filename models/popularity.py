"""
Popularity-based baseline recommendation model.
Recommends items based on global and category-specific popularity.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from utils.logger import get_logger
from utils.checkpoint import save_pickle, load_pickle

logger = get_logger(__name__)


class PopularityBaseline:
    """
    Popularity-based recommendation baseline.

    Recommends items based on: avg_rating * log(num_reviews + 1)
    Supports category-aware recommendations (by department).
    """

    def __init__(self, use_categories: bool = True):
        """
        Initialize popularity baseline.

        Args:
            use_categories: Whether to use category-specific popularity
        """
        self.use_categories = use_categories
        self.global_popularity = None
        self.category_popularity = None
        self.item_categories = None
        self.user_favorite_categories = None
        self.n_items = None

        logger.info(f"Initialized PopularityBaseline (use_categories={use_categories})")

    def fit(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame,
        user_features: pd.DataFrame = None
    ):
        """
        Fit the popularity model.

        Args:
            train_df: Training interactions (user_idx, item_idx, Rating)
            item_features: Item features (with Clothing.ID, department, etc.)
            user_features: User features (optional, for category preferences)
        """
        logger.info("Fitting PopularityBaseline...")

        # Add item_idx if not present (use DataFrame index)
        if 'item_idx' not in item_features.columns:
            item_features = item_features.copy()
            item_features['item_idx'] = item_features.index

        # Get number of items
        self.n_items = item_features['item_idx'].max() + 1

        # Compute global popularity scores
        self.global_popularity = self._compute_global_popularity(train_df, item_features)

        if self.use_categories and 'department' in item_features.columns:
            # Compute category-specific popularity
            self.category_popularity = self._compute_category_popularity(train_df, item_features)

            # Get item categories
            self.item_categories = item_features.set_index('item_idx')['department'].to_dict()

            # Get user favorite categories (most interacted category)
            if user_features is not None and 'favorite_dept' in user_features.columns:
                self.user_favorite_categories = user_features.set_index('user_idx')['favorite_dept'].to_dict()
            else:
                self.user_favorite_categories = self._compute_user_favorite_categories(train_df, item_features)

        logger.info(f"Fitted PopularityBaseline: {len(self.global_popularity)} items")

    def _compute_global_popularity(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame
    ) -> Dict[int, float]:
        """
        Compute global popularity: avg_rating * log(num_reviews + 1)

        Args:
            train_df: Training interactions
            item_features: Item features

        Returns:
            Dictionary mapping item_idx -> popularity score
        """
        # Aggregate statistics per item
        item_stats = train_df.groupby('item_idx').agg({
            'Rating': ['mean', 'count']
        }).reset_index()
        item_stats.columns = ['item_idx', 'avg_rating', 'num_reviews']

        # Compute popularity score
        popularity = {}
        for _, row in item_stats.iterrows():
            item_idx = row['item_idx']
            avg_rating = row['avg_rating']
            num_reviews = row['num_reviews']

            # Score: avg_rating * log(num_reviews + 1)
            score = avg_rating * np.log(num_reviews + 1)
            popularity[item_idx] = score

        # Fill missing items with 0
        for item_idx in range(self.n_items):
            if item_idx not in popularity:
                popularity[item_idx] = 0.0

        return popularity

    def _compute_category_popularity(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute popularity within each category.

        Args:
            train_df: Training interactions
            item_features: Item features

        Returns:
            Dictionary mapping category -> {item_idx -> popularity_score}
        """
        # Merge to get categories
        train_with_cat = train_df.merge(
            item_features[['item_idx', 'department']],
            on='item_idx',
            how='left'
        )

        category_popularity = defaultdict(dict)

        for category in train_with_cat['department'].unique():
            if pd.isna(category):
                continue

            cat_df = train_with_cat[train_with_cat['department'] == category]

            # Aggregate per item
            item_stats = cat_df.groupby('item_idx').agg({
                'Rating': ['mean', 'count']
            }).reset_index()
            item_stats.columns = ['item_idx', 'avg_rating', 'num_reviews']

            for _, row in item_stats.iterrows():
                item_idx = row['item_idx']
                avg_rating = row['avg_rating']
                num_reviews = row['num_reviews']
                score = avg_rating * np.log(num_reviews + 1)
                category_popularity[category][item_idx] = score

        return dict(category_popularity)

    def _compute_user_favorite_categories(
        self,
        train_df: pd.DataFrame,
        item_features: pd.DataFrame
    ) -> Dict[int, str]:
        """
        Compute each user's favorite category (most interactions).

        Args:
            train_df: Training interactions
            item_features: Item features

        Returns:
            Dictionary mapping user_idx -> favorite_category
        """
        # Merge to get categories
        train_with_cat = train_df.merge(
            item_features[['item_idx', 'department']],
            on='item_idx',
            how='left'
        )

        # Count interactions per user-category
        user_cat_counts = train_with_cat.groupby(['user_idx', 'department']).size().reset_index(name='count')

        # Get favorite category per user
        favorite_categories = {}
        for user_idx in train_with_cat['user_idx'].unique():
            user_cats = user_cat_counts[user_cat_counts['user_idx'] == user_idx]
            if len(user_cats) > 0:
                favorite_cat = user_cats.loc[user_cats['count'].idxmax(), 'department']
                favorite_categories[user_idx] = favorite_cat

        return favorite_categories

    def get_scores(self, user_idx: int) -> np.ndarray:
        """
        Get popularity scores for all items for a user.

        Args:
            user_idx: User index

        Returns:
            Array of scores (length = n_items)
        """
        scores = np.zeros(self.n_items)

        if self.use_categories and self.user_favorite_categories is not None:
            # Use category-specific popularity if available
            favorite_cat = self.user_favorite_categories.get(user_idx)

            if favorite_cat and favorite_cat in self.category_popularity:
                # Fill scores from category-specific popularity
                for item_idx, score in self.category_popularity[favorite_cat].items():
                    scores[int(item_idx)] = score

                # Fill remaining items with global popularity (scaled down)
                for item_idx in range(self.n_items):
                    if scores[item_idx] == 0 and item_idx in self.global_popularity:
                        scores[item_idx] = self.global_popularity[int(item_idx)] * 0.5

                return scores

        # Fallback to global popularity
        for item_idx, score in self.global_popularity.items():
            scores[int(item_idx)] = score

        return scores

    def recommend(self, user_idx: int, k: int = 10, exclude_items: List[int] = None) -> np.ndarray:
        """
        Generate top-K recommendations for a user.

        Args:
            user_idx: User index
            k: Number of recommendations
            exclude_items: Items to exclude (e.g., already consumed)

        Returns:
            Array of top-K item indices
        """
        scores = self.get_scores(user_idx)

        # Exclude items
        if exclude_items is not None:
            scores[exclude_items] = -np.inf

        # Get top-K
        top_k_items = np.argsort(scores)[::-1][:k]
        return top_k_items

    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Predict scores for specific items (for evaluation).

        Args:
            user_idx: User index
            item_indices: Array of item indices

        Returns:
            Array of prediction scores
        """
        all_scores = self.get_scores(user_idx)
        return all_scores[item_indices]

    def save(self, file_path: str):
        """
        Save model to file.

        Args:
            file_path: Path to save model
        """
        model_data = {
            'use_categories': self.use_categories,
            'global_popularity': self.global_popularity,
            'category_popularity': self.category_popularity,
            'item_categories': self.item_categories,
            'user_favorite_categories': self.user_favorite_categories,
            'n_items': self.n_items
        }
        save_pickle(model_data, file_path)
        logger.info(f"Saved PopularityBaseline to: {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """
        Load model from file.

        Args:
            file_path: Path to model file

        Returns:
            PopularityBaseline instance
        """
        model_data = load_pickle(file_path)

        model = cls(use_categories=model_data['use_categories'])
        model.global_popularity = model_data['global_popularity']
        model.category_popularity = model_data['category_popularity']
        model.item_categories = model_data['item_categories']
        model.user_favorite_categories = model_data['user_favorite_categories']
        model.n_items = model_data['n_items']

        logger.info(f"Loaded PopularityBaseline from: {file_path}")
        return model


def train_popularity_baseline(
    dataset_dir: str = "dataset",
    checkpoint_dir: str = "checkpoints/popularity"
) -> PopularityBaseline:
    """
    Train and save popularity baseline model.

    Args:
        dataset_dir: Directory with dataset files
        checkpoint_dir: Directory to save model

    Returns:
        Trained PopularityBaseline model
    """
    from pathlib import Path
    import pandas as pd
    import pickle

    logger.info("Training Popularity Baseline...")

    # Load data
    train_df = pd.read_csv(Path(dataset_dir) / "train_set.csv")
    item_features = pd.read_csv(Path(dataset_dir) / "item_features.csv")
    user_features = pd.read_csv(Path(dataset_dir) / "user_features.csv")

    # Load mappings
    with open(Path(dataset_dir) / "user_to_idx.pkl", 'rb') as f:
        user_to_idx = pickle.load(f)
    with open(Path(dataset_dir) / "item_to_idx.pkl", 'rb') as f:
        item_to_idx = pickle.load(f)

    # Add user_idx and item_idx columns
    train_df['user_idx'] = train_df['User.ID'].map(user_to_idx)
    train_df['item_idx'] = train_df['Clothing.ID'].map(item_to_idx)

    # Add item_idx to item_features (using index as item_idx)
    item_features['item_idx'] = item_features.index

    # Add user_idx to user_features (using index as user_idx)
    user_features['user_idx'] = user_features.index

    # Train model
    model = PopularityBaseline(use_categories=True)
    model.fit(train_df, item_features, user_features)

    # Save model
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(checkpoint_dir) / "baseline.pkl")

    logger.info("Popularity Baseline training complete")
    return model
