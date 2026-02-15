"""
Matrix Factorization model for collaborative filtering.
Uses user and item embeddings with biases to predict ratings.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class MatrixFactorization(nn.Module):
    """
    Matrix Factorization model with biases.

    Architecture:
    - User embedding: (n_users, embedding_dim)
    - Item embedding: (n_items, embedding_dim)
    - User bias: (n_users, 1)
    - Item bias: (n_items, 1)
    - Global bias: scalar

    Prediction = dot(user_emb, item_emb) + user_bias + item_bias + global_bias
    Scaled to [1, 5] using sigmoid * 4 + 1
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        use_bias: bool = True
    ):
        """
        Initialize Matrix Factorization model.

        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of latent factors
            use_bias: Whether to use bias terms
        """
        super(MatrixFactorization, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.use_bias = use_bias

        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Bias terms
        if use_bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

        # Initialize weights
        self._init_weights()

        logger.info(f"Initialized MatrixFactorization: {n_users} users, {n_items} items, dim={embedding_dim}")

    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        if self.use_bias:
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            user_indices: Tensor of user indices (batch_size,)
            item_indices: Tensor of item indices (batch_size,)

        Returns:
            Predicted ratings (batch_size,) in range [1, 5]
        """
        # Get embeddings
        user_emb = self.user_embedding(user_indices)  # (batch_size, embedding_dim)
        item_emb = self.item_embedding(item_indices)  # (batch_size, embedding_dim)

        # Dot product
        dot_product = (user_emb * item_emb).sum(dim=1)  # (batch_size,)

        # Add biases
        if self.use_bias:
            user_b = self.user_bias(user_indices).squeeze()  # (batch_size,)
            item_b = self.item_bias(item_indices).squeeze()  # (batch_size,)
            prediction = dot_product + user_b + item_b + self.global_bias
        else:
            prediction = dot_product

        # Scale to [1, 5] using sigmoid * 4 + 1
        prediction = torch.sigmoid(prediction) * 4.0 + 1.0

        return prediction

    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for a single user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Predicted rating
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            item_tensor = torch.LongTensor([item_idx])
            prediction = self.forward(user_tensor, item_tensor)
            return prediction.item()

    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Predict ratings for a user and multiple items.

        Args:
            user_idx: User index
            item_indices: Array of item indices

        Returns:
            Array of predicted ratings
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * len(item_indices))
            item_tensor = torch.LongTensor(item_indices)
            predictions = self.forward(user_tensor, item_tensor)
            return predictions.cpu().numpy()

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """
        Get user embedding vector.

        Args:
            user_idx: User index

        Returns:
            Embedding vector
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            embedding = self.user_embedding(user_tensor)
            return embedding.cpu().numpy().flatten()

    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """
        Get item embedding vector.

        Args:
            item_idx: Item index

        Returns:
            Embedding vector
        """
        self.eval()
        with torch.no_grad():
            item_tensor = torch.LongTensor([item_idx])
            embedding = self.item_embedding(item_tensor)
            return embedding.cpu().numpy().flatten()

    def get_all_user_embeddings(self) -> np.ndarray:
        """
        Get all user embeddings.

        Returns:
            Matrix of shape (n_users, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.user_embedding.weight.cpu().numpy()

    def get_all_item_embeddings(self) -> np.ndarray:
        """
        Get all item embeddings.

        Returns:
            Matrix of shape (n_items, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.item_embedding.weight.cpu().numpy()
