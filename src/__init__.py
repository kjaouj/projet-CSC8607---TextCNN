# src package
"""
CSC8607 Deep Learning Project - AG_NEWS Text Classification with TextCNN

This package contains modules for:
- data_loading: Loading and preprocessing AG_NEWS dataset
- model: TextCNN model architecture
- train: Training loop with TensorBoard logging
- evaluate: Model evaluation with classification metrics
- lr_finder: Learning rate finder utility
- grid_search: Hyperparameter grid search
- preprocessing: Text preprocessing utilities
- augmentation: Data augmentation (placeholder for NLP)
- utils: General utilities (seed setting, device handling, etc.)
"""

from .data_loading import get_dataloaders
from .model import build_model
from .train import train_loop
from .evaluate import evaluate_model
from .utils import set_seed, get_device, count_parameters, save_config_snapshot

__all__ = [
    'get_dataloaders',
    'build_model',
    'train_loop',
    'evaluate_model',
    'set_seed',
    'get_device',
    'count_parameters',
    'save_config_snapshot'
]
