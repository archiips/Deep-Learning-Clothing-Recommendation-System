"""
Training script for Matrix Factorization model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from models.matrix_factorization import MatrixFactorization
from utils.data_loader import DataManager
from utils.checkpoint import ModelCheckpoint
from utils.logger import setup_logger

logger = setup_logger(__name__)


class MFTrainer:
    """
    Trainer for Matrix Factorization model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Set random seed
        torch.manual_seed(config.get('random_seed', 42))
        np.random.seed(config.get('random_seed', 42))

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Model will be initialized in train()
        self.model = None
        self.optimizer = None
        self.criterion = None

    def train(
        self,
        train_loader,
        val_loader,
        n_users: int,
        n_items: int
    ) -> MatrixFactorization:
        """
        Train the Matrix Factorization model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_users: Number of users
            n_items: Number of items

        Returns:
            Trained model
        """
        logger.info("Starting Matrix Factorization training...")

        # Initialize model
        model_config = self.config['model']
        self.model = MatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=model_config['embedding_dim'],
            use_bias=model_config['use_bias']
        ).to(self.device)

        # Initialize optimizer
        optimizer_config = self.config['optimizer']
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps']
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Checkpoint manager
        checkpoint_config = self.config['checkpointing']
        checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_config['checkpoint_dir'],
            model_name="mf",
            metric_name=checkpoint_config['metric_to_track'],
            mode=checkpoint_config['mode']
        )

        # Training loop
        max_epochs = self.config['training']['max_epochs']
        patience = self.config['training']['early_stopping_patience']
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_loss = self._validate_epoch(val_loader)

            logger.info(f"Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Check if best model
            is_best = checkpoint_manager.should_save(val_loss)

            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss
            }

            checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics,
                config=self.config,
                is_best=is_best
            )

            # Early stopping
            if is_best:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break

        # Load best model
        best_checkpoint = checkpoint_manager.load_best()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']}")

        logger.info("Matrix Factorization training complete")
        return self.model

    def _train_epoch(self, train_loader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for user_indices, item_indices, ratings in tqdm(train_loader, desc="Training", leave=False):
            # Move to device
            user_indices = user_indices.to(self.device)
            item_indices = item_indices.to(self.device)
            ratings = ratings.to(self.device)

            # Forward pass
            predictions = self.model(user_indices, item_indices)
            loss = self.criterion(predictions, ratings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(self, val_loader) -> float:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for user_indices, item_indices, ratings in val_loader:
                # Move to device
                user_indices = user_indices.to(self.device)
                item_indices = item_indices.to(self.device)
                ratings = ratings.to(self.device)

                # Forward pass
                predictions = self.model(user_indices, item_indices)
                loss = self.criterion(predictions, ratings)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches


def train_matrix_factorization(
    config_path: str = "configs/mf_config.yaml",
    dataset_dir: str = "dataset"
) -> MatrixFactorization:
    """
    Train Matrix Factorization model from config file.

    Args:
        config_path: Path to configuration file
        dataset_dir: Directory with dataset files

    Returns:
        Trained model
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")

    # Load data
    data_manager = DataManager(dataset_dir)
    data_manager.load_config()

    # Create data loaders (explicit feedback)
    train_loader, test_loader = data_manager.create_dataloaders(
        batch_size=config['training']['batch_size'],
        use_negatives=False,  # Explicit ratings
        shuffle=True
    )

    # Split validation from training (simple split)
    # For simplicity, we'll use test_loader as val_loader
    val_loader = test_loader

    # Train model
    trainer = MFTrainer(config)
    model = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_users=data_manager.n_users,
        n_items=data_manager.n_items
    )

    return model


if __name__ == "__main__":
    model = train_matrix_factorization()
    logger.info("Training complete!")
