"""
Centralized data loading utilities for Phase 4 models.
Loads all preprocessed data from Phase 3.
"""
import pandas as pd
import numpy as np
import pickle
import torch
from scipy.sparse import csr_matrix, load_npz
from pathlib import Path
from typing import Dict, Tuple, List, Any
from torch.utils.data import DataLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for recommendation system (imported from Phase 3).
    """

    def __init__(self, user_indices, item_indices, ratings=None):
        self.user_indices = torch.LongTensor(user_indices)
        self.item_indices = torch.LongTensor(item_indices)
        if ratings is not None:
            self.ratings = torch.FloatTensor(ratings)
        else:
            self.ratings = None

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]
        else:
            return self.user_indices[idx], self.item_indices[idx]


class DataManager:
    """
    Centralized data manager for all Phase 4 models.
    """

    def __init__(self, dataset_dir: str = "dataset"):
        """
        Initialize data manager.

        Args:
            dataset_dir: Directory containing Phase 3 outputs
        """
        self.dataset_dir = Path(dataset_dir)
        self.config = None
        self.encoders = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None
        self.n_users = None
        self.n_items = None

        logger.info(f"Initializing DataManager with dataset_dir: {self.dataset_dir}")

    def load_config(self) -> Dict[str, Any]:
        """
        Load PyTorch configuration from Phase 3.

        Returns:
            Configuration dictionary
        """
        config_path = self.dataset_dir / "pytorch_config.pkl"
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        self.n_users = self.config['n_users']
        self.n_items = self.config['n_items']

        logger.info(f"Loaded config: {self.n_users} users, {self.n_items} items")
        return self.config

    def load_encoders(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Load user/item encodings and categorical encoders.

        Returns:
            Tuple of (user_to_idx, item_to_idx, idx_to_user, idx_to_item)
        """
        # Load user/item mappings
        with open(self.dataset_dir / "user_to_idx.pkl", 'rb') as f:
            self.user_to_idx = pickle.load(f)

        with open(self.dataset_dir / "item_to_idx.pkl", 'rb') as f:
            self.item_to_idx = pickle.load(f)

        # Create reverse mappings
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        # Load categorical encoders
        with open(self.dataset_dir / "encoders.pkl", 'rb') as f:
            self.encoders = pickle.load(f)

        logger.info(f"Loaded encoders: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items")
        return self.user_to_idx, self.item_to_idx, self.idx_to_user, self.idx_to_item

    def load_matrices(self) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        """
        Load sparse CSR matrices.

        Returns:
            Tuple of (train_matrix, test_matrix, full_matrix)
        """
        train_matrix = load_npz(self.dataset_dir / "train_matrix.npz")
        test_matrix = load_npz(self.dataset_dir / "test_matrix.npz")
        full_matrix = load_npz(self.dataset_dir / "user_item_matrix.npz")

        logger.info(f"Loaded matrices: train={train_matrix.shape}, test={test_matrix.shape}")
        return train_matrix, test_matrix, full_matrix

    def load_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load user and item features.

        Returns:
            Tuple of (user_features, item_features)
        """
        user_features = pd.read_csv(self.dataset_dir / "user_features.csv")
        item_features = pd.read_csv(self.dataset_dir / "item_features.csv")

        logger.info(f"Loaded features: {len(user_features)} users, {len(item_features)} items")
        return user_features, item_features

    def load_train_test_sets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test sets and add user_idx/item_idx columns.

        Returns:
            Tuple of (train_df, test_df)
        """
        train_df = pd.read_csv(self.dataset_dir / "train_set.csv")
        test_df = pd.read_csv(self.dataset_dir / "test_set.csv")

        # Load mappings if not already loaded
        if self.user_to_idx is None or self.item_to_idx is None:
            self.load_encoders()

        # Add user_idx and item_idx columns
        train_df['user_idx'] = train_df['User.ID'].map(self.user_to_idx)
        train_df['item_idx'] = train_df['Clothing.ID'].map(self.item_to_idx)
        test_df['user_idx'] = test_df['User.ID'].map(self.user_to_idx)
        test_df['item_idx'] = test_df['Clothing.ID'].map(self.item_to_idx)

        logger.info(f"Loaded splits: {len(train_df)} train, {len(test_df)} test interactions")
        return train_df, test_df

    def load_negative_samples(self) -> pd.DataFrame:
        """
        Load negative samples for implicit feedback training.

        Returns:
            DataFrame with negative samples
        """
        neg_df = pd.read_csv(self.dataset_dir / "train_samples_with_negatives.csv")
        logger.info(f"Loaded negative samples: {len(neg_df)} total samples")
        return neg_df

    def create_dataloaders(
        self,
        batch_size: int = 256,
        use_negatives: bool = False,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and testing.

        Args:
            batch_size: Batch size
            use_negatives: Whether to use negative samples (for implicit feedback)
            shuffle: Whether to shuffle training data
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, test_loader)
        """
        if use_negatives:
            # Load negative samples for implicit feedback
            train_df = self.load_negative_samples()
            logger.info("Using negative samples for implicit feedback training")
        else:
            # Load regular training set for explicit feedback
            train_df, _ = self.load_train_test_sets()
            logger.info("Using explicit feedback training")

        _, test_df = self.load_train_test_sets()

        # Create datasets
        train_dataset = RecommendationDataset(
            user_indices=train_df['user_idx'].values,
            item_indices=train_df['item_idx'].values,
            ratings=train_df['Rating'].values if 'Rating' in train_df.columns else train_df['label'].values
        )

        test_dataset = RecommendationDataset(
            user_indices=test_df['user_idx'].values,
            item_indices=test_df['item_idx'].values,
            ratings=test_df['Rating'].values
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        logger.info(f"Created DataLoaders: batch_size={batch_size}, train_batches={len(train_loader)}, test_batches={len(test_loader)}")
        return train_loader, test_loader

    def get_user_history(self, user_idx: int, matrix: csr_matrix) -> np.ndarray:
        """
        Get items a user has interacted with.

        Args:
            user_idx: User index
            matrix: Sparse interaction matrix

        Returns:
            Array of item indices
        """
        user_row = matrix[user_idx].toarray().flatten()
        item_indices = np.where(user_row > 0)[0]
        return item_indices

    def get_all_user_histories(self, matrix: csr_matrix) -> Dict[int, np.ndarray]:
        """
        Get interaction histories for all users (cached for efficiency).

        Args:
            matrix: Sparse interaction matrix

        Returns:
            Dictionary mapping user_idx -> item_indices
        """
        user_histories = {}
        for user_idx in range(matrix.shape[0]):
            user_histories[user_idx] = self.get_user_history(user_idx, matrix)
        logger.info(f"Cached histories for {len(user_histories)} users")
        return user_histories

    def load_all(self) -> Dict[str, Any]:
        """
        Load all data at once (convenience method).

        Returns:
            Dictionary with all loaded data
        """
        config = self.load_config()
        user_to_idx, item_to_idx, idx_to_user, idx_to_item = self.load_encoders()
        train_matrix, test_matrix, full_matrix = self.load_matrices()
        user_features, item_features = self.load_features()
        train_df, test_df = self.load_train_test_sets()

        data = {
            'config': config,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_item': idx_to_item,
            'train_matrix': train_matrix,
            'test_matrix': test_matrix,
            'full_matrix': full_matrix,
            'user_features': user_features,
            'item_features': item_features,
            'train_df': train_df,
            'test_df': test_df
        }

        logger.info("Loaded all data successfully")
        return data
