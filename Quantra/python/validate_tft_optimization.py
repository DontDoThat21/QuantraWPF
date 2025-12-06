#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation Script for TFT Hyperparameter Optimization

Tests the performance of optimized TFT hyperparameters vs. defaults.
Provides comprehensive comparison including:
- Prediction accuracy (MSE, MAE, R2)
- Training time
- Model complexity
- Visualization of results
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from tft_integration import TFTStockPredictor, create_static_features
    from tft_hyperparameter_config import get_config, TFT_CONFIGS
    from optimize_tft import load_optimization_results
    TFT_AVAILABLE = True
except ImportError as e:
    TFT_AVAILABLE = False
    print(f"TFT not available: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('validate_tft')


def generate_synthetic_stock_data(n_samples: int = 1000,
                                   seq_len: int = 60,
                                   n_features: int = 20,
                                   static_dim: int = 10,
                                   noise_level: float = 0.1) -> Dict[str, np.ndarray]:
    """
    Generate synthetic stock-like data for testing.

    Args:
        n_samples: Number of samples
        seq_len: Sequence length
        n_features: Number of temporal features
        static_dim: Number of static features
        noise_level: Noise level for targets

    Returns:
        Dictionary with X_past, X_static, y
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {seq_len} seq_len, {n_features} features")

    # Generate temporal features with some correlation structure
    X_past = np.zeros((n_samples, seq_len, n_features))
    for i in range(n_samples):
        # Random walk for price-like behavior
        base = np.cumsum(np.random.randn(seq_len)) * 0.1
        for j in range(n_features):
            feature = base + np.random.randn(seq_len) * 0.5
            X_past[i, :, j] = feature

    # Generate static features
    X_static = np.random.randn(n_samples, static_dim).astype(np.float32)

    # Generate targets as percentage changes with some predictability
    # Target depends on recent trends (last 5 time steps average)
    recent_trend = X_past[:, -5:, 0].mean(axis=1)  # Use first feature
    base_targets = recent_trend * 0.3  # Some correlation with trend

    # Create multi-horizon targets
    y = np.zeros((n_samples, 4))
    for h in range(4):
        # Further horizons have more uncertainty
        horizon_noise = noise_level * (1 + h * 0.2)
        y[:, h] = base_targets + np.random.randn(n_samples) * horizon_noise

    return {
        'X_past': X_past.astype(np.float32),
        'X_static': X_static.astype(np.float32),
        'y': y.astype(np.float32)
    }


def evaluate_model(model: TFTStockPredictor,
                   X_past: np.ndarray,
                   X_static: np.ndarray,
                   y_true: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: Trained TFT model
        X_past: Temporal features
        X_static: Static features
        y_true: True targets

    Returns:
        Dictionary of metrics
    """
    predictions = model.predict(X_past, X_static)
    y_pred = predictions['median_predictions']

    # Ensure shapes match
    if y_true.ndim == 1:
        y_true = np.column_stack([y_true] * y_pred.shape[1])

    # Calculate metrics
    mse = np.mean((y_pred - y_true) ** 2)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)

    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }


def compare_configurations(data: Dict[str, np.ndarray],
                           config_names: list = None) -> pd.DataFrame:
    """
    Compare different TFT configurations.

    Args:
        data: Dictionary with training/validation data
        config_names: List of config names to compare (uses all if None)

    Returns:
        DataFrame with comparison results
    """
    if config_names is None:
        config_names = list(TFT_CONFIGS.keys())

    # Split data
    n_train = int(0.8 * len(data['X_past']))
    X_train = data['X_past'][:n_train]
    X_static_train = data['X_static'][:n_train]
    y_train = data['y'][:n_train]

    X_val = data['X_past'][n_train:]
    X_static_val = data['X_static'][n_train:]
    y_val = data['y'][n_train:]

    results = []

    for config_name in config_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing configuration: {config_name}")
        logger.info(f"{'='*60}")

        # Get configuration
        config = get_config(config_name)

        # Create model
        model = TFTStockPredictor(
            input_dim=X_train.shape[2],
            static_dim=X_static_train.shape[1],
            hidden_dim=config['hidden_dim'],
            num_heads=config['num_heads'],
            num_lstm_layers=config['num_lstm_layers'],
            dropout=config['dropout'],
            num_attention_layers=config['num_attention_layers']
        )

        # Train and measure time
        start_time = time.time()
        history = model.fit(
            X_train, X_static_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate'],
            verbose=False
        )
        training_time = time.time() - start_time

        # Evaluate
        train_metrics = evaluate_model(model, X_train, X_static_train, y_train)
        val_metrics = evaluate_model(model, X_val, X_static_val, y_val)

        # Count parameters
        n_params = sum(p.numel() for p in model.model.parameters())

        results.append({
            'config': config_name,
            'train_mse': train_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'val_mse': val_metrics['mse'],
            'val_mae': val_metrics['mae'],
            'val_r2': val_metrics['r2'],
            'training_time_sec': training_time,
            'n_parameters': n_params,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        })

        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Validation MSE: {val_metrics['mse']:.6f}")
        logger.info(f"Validation R2: {val_metrics['r2']:.4f}")
        logger.info(f"Parameters: {n_params:,}")

    return pd.DataFrame(results)


def visualize_comparison(results_df: pd.DataFrame, output_path: str = None):
    """
    Create visualization of configuration comparison.

    Args:
        results_df: DataFrame with comparison results
        output_path: Path to save figure (shows if None)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TFT Hyperparameter Configuration Comparison', fontsize=16, fontweight='bold')

    configs = results_df['config'].tolist()
    x_pos = np.arange(len(configs))

    # Plot 1: Validation MSE
    ax = axes[0, 0]
    bars = ax.bar(x_pos, results_df['val_mse'], alpha=0.7, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Validation MSE (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = results_df['val_mse'].idxmin()
    bars[best_idx].set_color('green')

    # Plot 2: Validation R2
    ax = axes[0, 1]
    bars = ax.bar(x_pos, results_df['val_r2'], alpha=0.7, color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Validation R²')
    ax.set_title('Validation R² (Higher is Better)')
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = results_df['val_r2'].idxmax()
    bars[best_idx].set_color('green')

    # Plot 3: Training Time
    ax = axes[1, 0]
    bars = ax.bar(x_pos, results_df['training_time_sec'], alpha=0.7, color='purple')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)

    # Highlight fastest
    best_idx = results_df['training_time_sec'].idxmin()
    bars[best_idx].set_color('green')

    # Plot 4: Model Size vs Performance
    ax = axes[1, 1]
    scatter = ax.scatter(results_df['n_parameters'], results_df['val_r2'],
                        s=200, alpha=0.6, c=results_df['val_mse'],
                        cmap='RdYlGn_r')
    for i, config in enumerate(configs):
        ax.annotate(config, (results_df.loc[i, 'n_parameters'], results_df.loc[i, 'val_r2']),
                   fontsize=8, ha='center')
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Validation R²')
    ax.set_title('Model Complexity vs Performance')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Val MSE')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate TFT hyperparameter optimization")
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--configs', nargs='+', help='Configs to compare (all if not specified)')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--plot', type=str, help='Output path for plot')

    args = parser.parse_args()

    print("=" * 70)
    print("TFT Hyperparameter Optimization Validation")
    print("=" * 70)

    # Generate synthetic data
    data = generate_synthetic_stock_data(n_samples=args.samples)

    print(f"\nGenerated {args.samples} samples with:")
    print(f"  - Temporal features: {data['X_past'].shape}")
    print(f"  - Static features: {data['X_static'].shape}")
    print(f"  - Targets: {data['y'].shape}")

    # Compare configurations
    print("\nComparing configurations...")
    results = compare_configurations(data, args.configs)

    # Display results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))

    # Find best configuration
    best_by_mse = results.loc[results['val_mse'].idxmin()]
    best_by_r2 = results.loc[results['val_r2'].idxmax()]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)
    print(f"\nBest by Validation MSE: {best_by_mse['config']}")
    print(f"  MSE: {best_by_mse['val_mse']:.6f}")
    print(f"  R2: {best_by_mse['val_r2']:.4f}")
    print(f"  Training time: {best_by_mse['training_time_sec']:.2f}s")

    print(f"\nBest by Validation R²: {best_by_r2['config']}")
    print(f"  MSE: {best_by_r2['val_mse']:.6f}")
    print(f"  R2: {best_by_r2['val_r2']:.4f}")
    print(f"  Training time: {best_by_r2['training_time_sec']:.2f}s")

    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

    # Create visualization
    if args.plot:
        visualize_comparison(results, args.plot)
    else:
        print("\nGenerating visualization...")
        visualize_comparison(results)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
