"""
Utility functions to load trained models from checkpoints.
"""
import torch
import pickle
from pathlib import Path
from typing import Tuple

from models.popularity import PopularityBaseline
from models.matrix_factorization import MatrixFactorization
from models.neural_cf import NeuralCF
from utils.checkpoint import load_pickle
from utils.logger import get_logger

logger = get_logger(__name__)


def load_mf_model(checkpoint_path: str = "checkpoints/mf/mf_best.pt") -> MatrixFactorization:
    """
    Load Matrix Factorization model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded MatrixFactorization model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config
    config = checkpoint.get('config', {})
    n_users = config.get('n_users')
    n_items = config.get('n_items')

    if n_users is None or n_items is None:
        # Try to infer from state_dict
        state_dict = checkpoint['model_state_dict']
        n_users = state_dict['user_embedding.weight'].shape[0]
        n_items = state_dict['item_embedding.weight'].shape[0]

    # Create model
    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config.get('embedding_dim', 64)
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded MatrixFactorization model from: {checkpoint_path}")
    logger.info(f"  n_users={n_users}, n_items={n_items}, embedding_dim={model.embedding_dim}")

    return model


def load_ncf_model(checkpoint_path: str = "checkpoints/ncf/ncf_best.pt") -> NeuralCF:
    """
    Load Neural CF model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded NeuralCF model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get config
    config = checkpoint.get('config', {})
    n_users = config.get('n_users')
    n_items = config.get('n_items')

    if n_users is None or n_items is None:
        # Try to infer from state_dict
        state_dict = checkpoint['model_state_dict']
        n_users = state_dict['gmf_user_embedding.weight'].shape[0]
        n_items = state_dict['gmf_item_embedding.weight'].shape[0]

    # Create model
    model = NeuralCF(
        n_users=n_users,
        n_items=n_items,
        gmf_embedding_dim=config.get('gmf_embedding_dim', 64),
        mlp_embedding_dim=config.get('mlp_embedding_dim', 32),
        mlp_hidden_layers=config.get('mlp_hidden_layers', [128, 64, 32]),
        dropout=config.get('dropout', 0.2)
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded NeuralCF model from: {checkpoint_path}")
    logger.info(f"  n_users={n_users}, n_items={n_items}")

    return model


def load_popularity_model(checkpoint_path: str = "checkpoints/popularity/baseline.pkl") -> PopularityBaseline:
    """
    Load Popularity Baseline model from pickle file.

    Args:
        checkpoint_path: Path to pickle file

    Returns:
        Loaded PopularityBaseline model
    """
    model = PopularityBaseline.load(checkpoint_path)
    logger.info(f"Loaded PopularityBaseline model from: {checkpoint_path}")
    return model


def load_all_models() -> Tuple[PopularityBaseline, MatrixFactorization, NeuralCF]:
    """
    Load all three trained models.

    Returns:
        Tuple of (popularity_model, mf_model, ncf_model)
    """
    logger.info("Loading all models...")

    pop_model = load_popularity_model()
    mf_model = load_mf_model()
    ncf_model = load_ncf_model()

    logger.info("All models loaded successfully!")

    return pop_model, mf_model, ncf_model
