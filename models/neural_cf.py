"""
Neural Collaborative Filtering (NCF) model.
Dual-path architecture combining GMF and MLP for recommendation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List
from utils.logger import get_logger

logger = get_logger(__name__)


class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering model with dual-path architecture.

    Architecture:
    1. GMF path: Element-wise product of user and item embeddings
    2. MLP path: Concatenate embeddings -> MLP layers
    3. Fusion: Concatenate GMF + MLP outputs -> Final prediction

    Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_embedding_dim: int = 64,
        mlp_embedding_dim: int = 32,
        mlp_hidden_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        use_pretrain: bool = False
    ):
        """
        Initialize Neural CF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            gmf_embedding_dim: Embedding dimension for GMF path
            mlp_embedding_dim: Embedding dimension for MLP path
            mlp_hidden_layers: List of hidden layer sizes for MLP
            dropout: Dropout rate
            use_pretrain: Whether to use pretrained embeddings
        """
        super(NeuralCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.gmf_embedding_dim = gmf_embedding_dim
        self.mlp_embedding_dim = mlp_embedding_dim
        self.dropout = dropout

        # GMF path embeddings
        self.gmf_user_embedding = nn.Embedding(n_users, gmf_embedding_dim)
        self.gmf_item_embedding = nn.Embedding(n_items, gmf_embedding_dim)

        # MLP path embeddings
        self.mlp_user_embedding = nn.Embedding(n_users, mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(n_items, mlp_embedding_dim)

        # MLP layers
        mlp_layers = []
        input_dim = mlp_embedding_dim * 2  # Concatenate user and item

        for hidden_dim in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Fusion layer
        fusion_input_dim = gmf_embedding_dim + mlp_hidden_layers[-1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1),
            nn.Sigmoid()  # Output in [0, 1] for implicit feedback
        )

        # Initialize weights
        if not use_pretrain:
            self._init_weights()

        logger.info(f"Initialized NeuralCF: {n_users} users, {n_items} items")
        logger.info(f"  GMF dim={gmf_embedding_dim}, MLP dim={mlp_embedding_dim}, hidden={mlp_hidden_layers}")

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        # GMF embeddings
        nn.init.xavier_uniform_(self.gmf_user_embedding.weight)
        nn.init.xavier_uniform_(self.gmf_item_embedding.weight)

        # MLP embeddings
        nn.init.xavier_uniform_(self.mlp_user_embedding.weight)
        nn.init.xavier_uniform_(self.mlp_item_embedding.weight)

        # MLP and fusion layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dual-path architecture.

        Args:
            user_indices: Tensor of user indices (batch_size,)
            item_indices: Tensor of item indices (batch_size,)

        Returns:
            Predicted scores (batch_size,) in range [0, 1]
        """
        # GMF path
        gmf_user_emb = self.gmf_user_embedding(user_indices)  # (batch_size, gmf_dim)
        gmf_item_emb = self.gmf_item_embedding(item_indices)  # (batch_size, gmf_dim)
        gmf_output = gmf_user_emb * gmf_item_emb  # Element-wise product

        # MLP path
        mlp_user_emb = self.mlp_user_embedding(user_indices)  # (batch_size, mlp_dim)
        mlp_item_emb = self.mlp_item_embedding(item_indices)  # (batch_size, mlp_dim)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)  # (batch_size, 2*mlp_dim)
        mlp_output = self.mlp(mlp_input)  # (batch_size, mlp_hidden_layers[-1])

        # Fusion
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        prediction = self.fusion(fusion_input).squeeze()  # (batch_size,)

        return prediction

    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict score for a single user-item pair.

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Predicted score [0, 1]
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            item_tensor = torch.LongTensor([item_idx])
            prediction = self.forward(user_tensor, item_tensor)
            return prediction.item()

    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Predict scores for a user and multiple items.

        Args:
            user_idx: User index
            item_indices: Array of item indices

        Returns:
            Array of predicted scores [0, 1]
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * len(item_indices))
            item_tensor = torch.LongTensor(item_indices)
            predictions = self.forward(user_tensor, item_tensor)
            return predictions.cpu().numpy()

    def get_gmf_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get GMF user embedding vector."""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            embedding = self.gmf_user_embedding(user_tensor)
            return embedding.cpu().numpy().flatten()

    def get_gmf_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get GMF item embedding vector."""
        self.eval()
        with torch.no_grad():
            item_tensor = torch.LongTensor([item_idx])
            embedding = self.gmf_item_embedding(item_tensor)
            return embedding.cpu().numpy().flatten()

    def get_mlp_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get MLP user embedding vector."""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            embedding = self.mlp_user_embedding(user_tensor)
            return embedding.cpu().numpy().flatten()

    def get_mlp_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get MLP item embedding vector."""
        self.eval()
        with torch.no_grad():
            item_tensor = torch.LongTensor([item_idx])
            embedding = self.mlp_item_embedding(item_tensor)
            return embedding.cpu().numpy().flatten()

    def get_all_gmf_embeddings(self) -> tuple:
        """
        Get all GMF embeddings.

        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        self.eval()
        with torch.no_grad():
            user_emb = self.gmf_user_embedding.weight.cpu().numpy()
            item_emb = self.gmf_item_embedding.weight.cpu().numpy()
            return user_emb, item_emb
