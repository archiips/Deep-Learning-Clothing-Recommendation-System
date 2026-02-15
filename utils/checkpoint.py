"""
Model checkpointing utilities for saving and loading trained models.
"""
import torch
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelCheckpoint:
    """
    Handles model checkpointing with support for tracking best models.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        metric_name: str = "ndcg@10",
        mode: str = "max"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            model_name: Name of the model (for file naming)
            metric_name: Metric to track for best model (e.g., "ndcg@10", "loss")
            mode: "max" for metrics to maximize, "min" for metrics to minimize
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.metric_name = metric_name
        self.mode = mode
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0

    def should_save(self, current_metric: float) -> bool:
        """
        Check if current metric is better than best metric.

        Args:
            current_metric: Current metric value

        Returns:
            True if current metric is better
        """
        if self.mode == 'max':
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric

        if is_better:
            self.best_metric = current_metric
            return True
        return False

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer (can be None)
            epoch: Current epoch
            metrics: Dictionary of metric values
            config: Model/training configuration
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'best_metric': self.best_metric
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if config is not None:
            checkpoint['config'] = config

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path} ({self.metric_name}={self.best_metric:.4f})")
            self.best_epoch = epoch
            return str(best_path)

        return str(checkpoint_path)

    def load(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint

    def load_best(self) -> Optional[Dict[str, Any]]:
        """
        Load best checkpoint.

        Returns:
            Best checkpoint dictionary or None if not found
        """
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if best_path.exists():
            return self.load(str(best_path))
        logger.warning(f"Best checkpoint not found: {best_path}")
        return None


def save_pickle(obj: Any, file_path: str):
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        file_path: Path to save file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle: {file_path}")


def load_pickle(file_path: str) -> Any:
    """
    Load object from pickle file.

    Args:
        file_path: Path to pickle file

    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle: {file_path}")
    return obj
