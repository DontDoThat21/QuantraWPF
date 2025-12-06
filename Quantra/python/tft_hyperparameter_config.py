#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFT Hyperparameter Configuration and Search Spaces

Defines optimal search spaces for Temporal Fusion Transformer hyperparameters
based on research and best practices from the original TFT paper.
"""

import optuna
from typing import Dict, Any, Optional


class TFTHyperparameterConfig:
    """Configuration for TFT hyperparameter optimization."""

    @staticmethod
    def get_default_search_space() -> Dict[str, Any]:
        """
        Get default search space for TFT hyperparameters.

        Based on the original TFT paper recommendations:
        - Hidden dimensions: 64-256 for most applications
        - Attention heads: 1-8, must divide hidden_dim
        - LSTM layers: 1-3 (deeper may cause vanishing gradients)
        - Dropout: 0.05-0.3 (higher for smaller datasets)
        - Learning rate: 1e-4 to 1e-2 (Adam optimizer)
        """
        return {
            # Model architecture
            'hidden_dim': {
                'type': 'categorical',
                'values': [64, 96, 128, 160, 192, 256],
                'description': 'Hidden dimension for all TFT layers'
            },
            'num_heads': {
                'type': 'categorical',
                'values': [1, 2, 4, 8],
                'description': 'Number of attention heads (must divide hidden_dim)'
            },
            'num_lstm_layers': {
                'type': 'int',
                'low': 1,
                'high': 3,
                'description': 'Number of LSTM encoder layers'
            },
            'num_attention_layers': {
                'type': 'int',
                'low': 1,
                'high': 4,
                'description': 'Number of self-attention layers'
            },

            # Regularization
            'dropout': {
                'type': 'float',
                'low': 0.05,
                'high': 0.3,
                'description': 'Dropout rate for regularization'
            },

            # Training hyperparameters
            'learning_rate': {
                'type': 'loguniform',
                'low': 1e-4,
                'high': 1e-2,
                'description': 'Learning rate for Adam optimizer'
            },
            'batch_size': {
                'type': 'categorical',
                'values': [16, 32, 64, 128],
                'description': 'Training batch size'
            },
            'epochs': {
                'type': 'categorical',
                'values': [30, 50, 75, 100],
                'description': 'Number of training epochs'
            }
        }

    @staticmethod
    def get_quick_search_space() -> Dict[str, Any]:
        """
        Get reduced search space for quick optimization (fewer trials needed).
        """
        return {
            'hidden_dim': {
                'type': 'categorical',
                'values': [96, 128, 192],
                'description': 'Hidden dimension for all TFT layers'
            },
            'num_heads': {
                'type': 'categorical',
                'values': [2, 4],
                'description': 'Number of attention heads'
            },
            'num_lstm_layers': {
                'type': 'int',
                'low': 1,
                'high': 2,
                'description': 'Number of LSTM encoder layers'
            },
            'num_attention_layers': {
                'type': 'int',
                'low': 1,
                'high': 3,
                'description': 'Number of self-attention layers'
            },
            'dropout': {
                'type': 'float',
                'low': 0.1,
                'high': 0.2,
                'description': 'Dropout rate'
            },
            'learning_rate': {
                'type': 'loguniform',
                'low': 5e-4,
                'high': 5e-3,
                'description': 'Learning rate'
            },
            'batch_size': {
                'type': 'categorical',
                'values': [32, 64],
                'description': 'Training batch size'
            },
            'epochs': {
                'type': 'categorical',
                'values': [30, 50],
                'description': 'Number of training epochs'
            }
        }

    @staticmethod
    def get_production_defaults() -> Dict[str, Any]:
        """
        Get production-ready default hyperparameters based on empirical research.
        These are safe defaults that work well across most stock prediction tasks.
        """
        return {
            'hidden_dim': 160,
            'num_heads': 4,
            'num_lstm_layers': 2,
            'num_attention_layers': 2,
            'dropout': 0.15,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50
        }

    @staticmethod
    def suggest_params_from_optuna_trial(trial: optuna.Trial,
                                          search_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Suggest hyperparameters from an Optuna trial.

        Args:
            trial: Optuna trial object
            search_space: Optional custom search space (uses default if None)

        Returns:
            Dictionary of suggested hyperparameters
        """
        if search_space is None:
            search_space = TFTHyperparameterConfig.get_default_search_space()

        params = {}

        for param_name, param_config in search_space.items():
            param_type = param_config['type']

            if param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['values'])
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
            elif param_type == 'loguniform':
                params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)

        # Validate that num_heads divides hidden_dim
        if params['hidden_dim'] % params['num_heads'] != 0:
            # Adjust hidden_dim to nearest divisible value
            remainder = params['hidden_dim'] % params['num_heads']
            params['hidden_dim'] = params['hidden_dim'] - remainder + (params['num_heads'] if remainder > params['num_heads']//2 else 0)

        return params

    @staticmethod
    def validate_params(params: Dict[str, Any]) -> bool:
        """
        Validate hyperparameter configuration.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check hidden_dim is divisible by num_heads
        if params.get('hidden_dim', 128) % params.get('num_heads', 4) != 0:
            raise ValueError(
                f"hidden_dim ({params['hidden_dim']}) must be divisible by "
                f"num_heads ({params['num_heads']})"
            )

        # Check dropout is in valid range
        if not 0.0 <= params.get('dropout', 0.1) <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {params['dropout']}")

        # Check learning rate is positive
        if params.get('learning_rate', 0.001) <= 0:
            raise ValueError(f"learning_rate must be positive, got {params['learning_rate']}")

        # Check epochs and batch_size are positive
        if params.get('epochs', 50) <= 0:
            raise ValueError(f"epochs must be positive, got {params['epochs']}")
        if params.get('batch_size', 32) <= 0:
            raise ValueError(f"batch_size must be positive, got {params['batch_size']}")

        return True


# Recommended configurations for different scenarios
TFT_CONFIGS = {
    'fast_training': {
        'description': 'Quick training for development/testing',
        'params': {
            'hidden_dim': 96,
            'num_heads': 4,
            'num_lstm_layers': 1,
            'num_attention_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.002,
            'batch_size': 64,
            'epochs': 30
        }
    },
    'balanced': {
        'description': 'Balanced accuracy and training time (recommended default)',
        'params': {
            'hidden_dim': 160,
            'num_heads': 4,
            'num_lstm_layers': 2,
            'num_attention_layers': 2,
            'dropout': 0.15,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50
        }
    },
    'high_accuracy': {
        'description': 'Maximum accuracy, longer training time',
        'params': {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_lstm_layers': 3,
            'num_attention_layers': 3,
            'dropout': 0.2,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 100
        }
    },
    'small_dataset': {
        'description': 'For datasets with < 1000 samples (higher regularization)',
        'params': {
            'hidden_dim': 96,
            'num_heads': 2,
            'num_lstm_layers': 1,
            'num_attention_layers': 2,
            'dropout': 0.25,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 75
        }
    }
}


def get_config(config_name: str = 'balanced') -> Dict[str, Any]:
    """
    Get a predefined TFT configuration.

    Args:
        config_name: Name of the configuration ('fast_training', 'balanced',
                     'high_accuracy', 'small_dataset')

    Returns:
        Dictionary with hyperparameters
    """
    if config_name not in TFT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(TFT_CONFIGS.keys())}")

    return TFT_CONFIGS[config_name]['params'].copy()


if __name__ == "__main__":
    # Print available configurations
    print("=" * 70)
    print("TFT Hyperparameter Configurations")
    print("=" * 70)

    for config_name, config_info in TFT_CONFIGS.items():
        print(f"\n{config_name.upper()}:")
        print(f"  Description: {config_info['description']}")
        print("  Parameters:")
        for param, value in config_info['params'].items():
            print(f"    {param:20s} = {value}")

    print("\n" + "=" * 70)
    print("Search Spaces")
    print("=" * 70)

    print("\nDefault Search Space:")
    for param, config in TFTHyperparameterConfig.get_default_search_space().items():
        print(f"  {param:20s}: {config['description']}")
