#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TFT Hyperparameter Optimization Script

Uses Optuna to find optimal hyperparameters for the Temporal Fusion Transformer
model. Includes support for:
- Bayesian optimization with TPE sampler
- Cross-validation for robust evaluation
- Automatic parameter validation
- Result persistence and visualization
"""

import os
import sys
import numpy as np
import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import TFT components
try:
    from tft_integration import TFTStockPredictor, create_static_features
    from tft_hyperparameter_config import TFTHyperparameterConfig, get_config
    TFT_AVAILABLE = True
except ImportError as e:
    TFT_AVAILABLE = False
    print(f"TFT not available: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optimize_tft')

# Paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OPTIMIZATION_DIR = os.path.join(BASE_DIR, 'optimization_results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)


class TFTObjective:
    """Objective function for TFT hyperparameter optimization."""

    def __init__(self,
                 X_past: np.ndarray,
                 X_static: np.ndarray,
                 y: np.ndarray,
                 search_space: Optional[Dict[str, Any]] = None,
                 cv_splits: int = 3,
                 forecast_horizons: list = None):
        """
        Initialize TFT optimization objective.

        Args:
            X_past: Temporal features (n_samples, seq_len, n_features)
            X_static: Static features (n_samples, static_dim)
            y: Targets (n_samples, n_horizons) or (n_samples,)
            search_space: Hyperparameter search space (uses default if None)
            cv_splits: Number of cross-validation splits
            forecast_horizons: Forecast horizons (e.g., [5, 10, 20, 30])
        """
        self.X_past = X_past
        self.X_static = X_static
        self.y = y
        self.search_space = search_space or TFTHyperparameterConfig.get_default_search_space()
        self.cv_splits = cv_splits
        self.forecast_horizons = forecast_horizons or [5, 10, 20, 30]

        # Derive dimensions
        self.input_dim = X_past.shape[2] if len(X_past.shape) == 3 else X_past.shape[1]
        self.static_dim = X_static.shape[1]

        logger.info(f"Initialized TFT objective with {len(X_past)} samples, "
                    f"{self.input_dim} features, {self.static_dim} static features")

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Evaluate a hyperparameter configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Validation loss (to minimize)
        """
        # Suggest hyperparameters
        params = TFTHyperparameterConfig.suggest_params_from_optuna_trial(trial, self.search_space)

        try:
            # Validate parameters
            TFTHyperparameterConfig.validate_params(params)
        except ValueError as e:
            logger.warning(f"Invalid parameters: {e}")
            return float('inf')

        # Perform time-series cross-validation
        cv_scores = []
        n_samples = len(self.X_past)
        split_size = n_samples // (self.cv_splits + 1)

        for fold in range(self.cv_splits):
            # Time-series split: train on earlier data, validate on later data
            train_end = split_size * (fold + 1)
            val_start = train_end
            val_end = min(val_start + split_size, n_samples)

            if val_end - val_start < 10:  # Need at least 10 samples for validation
                continue

            X_train = self.X_past[:train_end]
            X_static_train = self.X_static[:train_end]
            y_train = self.y[:train_end]

            X_val = self.X_past[val_start:val_end]
            X_static_val = self.X_static[val_start:val_end]
            y_val = self.y[val_start:val_end]

            # Create and train model
            try:
                model = TFTStockPredictor(
                    input_dim=self.input_dim,
                    static_dim=self.static_dim,
                    hidden_dim=params['hidden_dim'],
                    forecast_horizons=self.forecast_horizons,
                    num_heads=params['num_heads'],
                    num_lstm_layers=params['num_lstm_layers'],
                    dropout=params['dropout'],
                    num_attention_layers=params['num_attention_layers']
                )

                # Train with suggested hyperparameters
                history = model.fit(
                    X_train, X_static_train, y_train,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    lr=params['learning_rate'],
                    verbose=False
                )

                # Evaluate on validation set
                predictions = model.predict(X_val, X_static_val)
                median_preds = predictions['median_predictions']

                # Ensure y_val has correct shape
                if y_val.ndim == 1:
                    y_val_expanded = np.column_stack([y_val] * len(self.forecast_horizons))
                else:
                    y_val_expanded = y_val

                # Calculate MSE
                mse = np.mean((median_preds - y_val_expanded) ** 2)
                cv_scores.append(mse)

                # Report intermediate value for pruning
                trial.report(mse, fold)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

            except Exception as e:
                logger.warning(f"Error in fold {fold}: {e}")
                cv_scores.append(float('inf'))

        # Return average validation loss
        if not cv_scores:
            return float('inf')

        avg_score = np.mean(cv_scores)
        logger.info(f"Trial {trial.number}: avg_mse={avg_score:.6f}, params={params}")

        return avg_score


def optimize_tft_hyperparameters(
        X_past: np.ndarray,
        X_static: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        search_space: str = 'default',
        cv_splits: int = 3,
        forecast_horizons: list = None,
        study_name: Optional[str] = None,
        timeout: Optional[int] = None
) -> Tuple[Dict[str, Any], optuna.Study]:
    """
    Optimize TFT hyperparameters using Optuna.

    Args:
        X_past: Temporal features (n_samples, seq_len, n_features)
        X_static: Static features (n_samples, static_dim)
        y: Targets (n_samples, n_horizons) or (n_samples,)
        n_trials: Number of optimization trials
        search_space: 'default' or 'quick'
        cv_splits: Number of cross-validation splits
        forecast_horizons: Forecast horizons (e.g., [5, 10, 20, 30])
        study_name: Name for the Optuna study (auto-generated if None)
        timeout: Timeout in seconds (None for no timeout)

    Returns:
        Tuple of (best_params, optuna_study)
    """
    if not TFT_AVAILABLE:
        raise RuntimeError("TFT components not available")

    # Get search space
    if search_space == 'quick':
        search_space_dict = TFTHyperparameterConfig.get_quick_search_space()
    else:
        search_space_dict = TFTHyperparameterConfig.get_default_search_space()

    # Create objective
    objective = TFTObjective(
        X_past=X_past,
        X_static=X_static,
        y=y,
        search_space=search_space_dict,
        cv_splits=cv_splits,
        forecast_horizons=forecast_horizons
    )

    # Create study
    if study_name is None:
        study_name = f"tft_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )

    logger.info(f"Starting optimization with {n_trials} trials...")
    logger.info(f"Search space: {search_space}")
    logger.info(f"CV splits: {cv_splits}")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Optimization complete!")
    logger.info(f"Best validation MSE: {best_value:.6f}")
    logger.info(f"Best parameters: {best_params}")

    return best_params, study


def save_optimization_results(
        best_params: Dict[str, Any],
        study: optuna.Study,
        output_path: Optional[str] = None
) -> str:
    """
    Save optimization results to disk.

    Args:
        best_params: Best hyperparameters found
        study: Optuna study object
        output_path: Output file path (auto-generated if None)

    Returns:
        Path to saved file
    """
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(OPTIMIZATION_DIR, f'tft_optimization_{timestamp}.pkl')

    # Save results
    results = {
        'best_params': best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'study_name': study.study_name,
        'timestamp': datetime.now().isoformat(),
        'study': study
    }

    joblib.dump(results, output_path)
    logger.info(f"Saved optimization results to {output_path}")

    # Also save JSON summary
    json_path = output_path.replace('.pkl', '.json')
    import json
    json_results = {
        'best_params': best_params,
        'best_value': float(study.best_value),
        'n_trials': len(study.trials),
        'study_name': study.study_name,
        'timestamp': datetime.now().isoformat()
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Saved JSON summary to {json_path}")

    return output_path


def load_optimization_results(filepath: str) -> Dict[str, Any]:
    """Load optimization results from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    results = joblib.load(filepath)
    logger.info(f"Loaded optimization results from {filepath}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize TFT hyperparameters")
    parser.add_argument('--data', type=str, help='Path to training data (NPZ file)')
    parser.add_argument('--trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--search-space', type=str, default='default',
                        choices=['default', 'quick'], help='Search space to use')
    parser.add_argument('--cv-splits', type=int, default=3, help='CV splits')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--timeout', type=int, help='Timeout in seconds')

    args = parser.parse_args()

    if args.data:
        # Load data from NPZ file
        logger.info(f"Loading data from {args.data}")
        data = np.load(args.data)
        X_past = data['X_past']
        X_static = data['X_static']
        y = data['y']
    else:
        # Generate synthetic data for testing
        logger.info("No data provided, generating synthetic data for testing")
        n_samples = 500
        seq_len = 60
        n_features = 20
        static_dim = 10

        X_past = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        X_static = np.random.randn(n_samples, static_dim).astype(np.float32)
        y = np.random.randn(n_samples, 4).astype(np.float32) * 0.05  # 4 horizons

    # Run optimization
    best_params, study = optimize_tft_hyperparameters(
        X_past=X_past,
        X_static=X_static,
        y=y,
        n_trials=args.trials,
        search_space=args.search_space,
        cv_splits=args.cv_splits,
        timeout=args.timeout
    )

    # Save results
    output_path = save_optimization_results(best_params, study, args.output)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best parameters saved to: {output_path}")
    print(f"\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param:20s} = {value}")
    print(f"\nBest validation MSE: {study.best_value:.6f}")
    print("=" * 70)
