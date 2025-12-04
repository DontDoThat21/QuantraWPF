#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter Optimization Module

This module implements automated hyperparameter optimization for machine learning models,
providing Grid Search, Randomized Search, and Bayesian Optimization via Optuna.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
from typing import Dict, List, Union, Callable, Optional, Tuple, Any

# sklearn imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit

# Try to import optuna for Bayesian optimization
try:
    import optuna
    import optuna.visualization as vis
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hyperparameter_optimization')

# Set up constant paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIMIZATION_DIR = os.path.join(BASE_DIR, 'optimization_results')
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)


class OptimizationResult:
    """Class to store and manage hyperparameter optimization results."""
    
    def __init__(self, 
                 best_params: Dict,
                 best_score: float,
                 all_params: List[Dict] = None,
                 all_scores: List[float] = None,
                 model_type: str = "sklearn",
                 optimization_method: str = "grid_search",
                 metric_name: str = "score",
                 higher_is_better: bool = True,
                 cv_results: Dict = None,
                 timestamp: str = None):
        """
        Initialize the OptimizationResult class.
        
        Args:
            best_params (Dict): Best hyperparameters found
            best_score (float): Best score achieved
            all_params (List[Dict], optional): List of all parameters tested
            all_scores (List[float], optional): List of all scores achieved
            model_type (str): Type of model (sklearn, pytorch, tensorflow)
            optimization_method (str): Method used for optimization
            metric_name (str): Name of the metric used
            higher_is_better (bool): Whether higher metric values are better
            cv_results (Dict, optional): Full cross-validation results
            timestamp (str, optional): Timestamp of optimization run
        """
        self.best_params = best_params
        self.best_score = best_score
        self.all_params = all_params or []
        self.all_scores = all_scores or []
        self.model_type = model_type
        self.optimization_method = optimization_method
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.cv_results = cv_results or {}
        self.timestamp = timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    def to_dict(self) -> Dict:
        """Convert the optimization result to a dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": float(self.best_score),
            "model_type": self.model_type,
            "optimization_method": self.optimization_method,
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "timestamp": self.timestamp,
            "num_trials": len(self.all_scores) if self.all_scores else 0
        }
    
    def save(self, filepath: str = None):
        """Save optimization results to disk."""
        if filepath is None:
            model_name = self.model_type.lower().replace(" ", "_")
            method_name = self.optimization_method.lower().replace(" ", "_")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(OPTIMIZATION_DIR, f"{model_name}_{method_name}_{timestamp}.pkl")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the object
        joblib.dump(self, filepath)
        
        # Also save a JSON summary for easier inspection
        json_path = filepath.replace(".pkl", ".json")
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        logger.info(f"Saved optimization results to {filepath} and {json_path}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str):
        """Load optimization results from disk."""
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        try:
            return joblib.load(filepath)
        except Exception as e:
            logger.error(f"Error loading optimization results: {str(e)}")
            return None
    
    def plot_optimization_history(self, figsize: Tuple[int, int] = (10, 6), save_path: str = None):
        """Plot the optimization history."""
        if not self.all_params or not self.all_scores:
            logger.warning("Cannot plot optimization history: missing trials data")
            return None
            
        plt.figure(figsize=figsize)
        plt.plot(range(len(self.all_scores)), self.all_scores, 'b-')
        plt.scatter(range(len(self.all_scores)), self.all_scores, c='b')
        
        best_idx = np.argmax(self.all_scores) if self.higher_is_better else np.argmin(self.all_scores)
        plt.scatter([best_idx], [self.all_scores[best_idx]], c='r', s=100, marker='*', label='Best')
        
        plt.xlabel('Trial')
        plt.ylabel(self.metric_name)
        plt.title(f'Optimization History - {self.model_type} - {self.optimization_method}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization history plot to {save_path}")
            
        return plt.gcf()
    
    def get_summary(self) -> str:
        """Get a text summary of the optimization results."""
        summary = []
        summary.append(f"Hyperparameter Optimization Summary")
        summary.append(f"==================================")
        summary.append(f"Model Type: {self.model_type}")
        summary.append(f"Optimization Method: {self.optimization_method}")
        summary.append(f"Metric: {self.metric_name} ({'higher is better' if self.higher_is_better else 'lower is better'})")
        summary.append(f"Best Score: {self.best_score:.6f}")
        summary.append(f"Number of Trials: {len(self.all_scores) if self.all_scores else 0}")
        summary.append(f"Timestamp: {self.timestamp}")
        summary.append("")
        summary.append("Best Parameters:")
        for param_name, param_value in self.best_params.items():
            summary.append(f"  {param_name}: {param_value}")
            
        return "\n".join(summary)


def optimize_sklearn_model(
    model_class: Any, 
    X_train: np.ndarray, 
    y_train: np.ndarray,
    param_grid: Dict,
    cv: int = 5,
    method: str = "grid_search",
    scoring: str = "neg_mean_squared_error",
    n_iter: int = 20,
    verbose: int = 1,
    n_jobs: int = -1,
    random_state: int = 42,
    return_train_score: bool = True
) -> OptimizationResult:
    """
    Optimize hyperparameters for scikit-learn models using Grid Search or Randomized Search.
    
    Args:
        model_class: Scikit-learn model class
        X_train: Training features
        y_train: Training target
        param_grid: Parameter grid to search
        cv: Number of cross-validation folds or CV splitter object
        method: Optimization method, one of ["grid_search", "random_search"]
        scoring: Scoring metric (scikit-learn scoring string)
        n_iter: Number of iterations for randomized search
        verbose: Verbosity level
        n_jobs: Number of parallel jobs
        random_state: Random state for reproducibility
        return_train_score: Whether to return train scores
    
    Returns:
        OptimizationResult: Object containing optimization results
    """
    # Create the base model
    base_model = model_class()
    
    # Choose appropriate search method
    if method.lower() == "random_search":
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=return_train_score
        )
    else:  # Default to grid search
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=return_train_score
        )
    
    # Fit the search
    logger.info(f"Starting {method} for {model_class.__name__}")
    search.fit(X_train, y_train)
    logger.info(f"Completed {method} for {model_class.__name__}")
    
    # Get the results
    best_params = search.best_params_
    best_score = search.best_score_
    
    # Determine if higher score is better based on the scoring metric
    higher_is_better = True
    if scoring.startswith('neg_'):
        higher_is_better = False
        best_score = -best_score  # Convert back to positive scale
    
    # Get all parameters and scores
    all_params = []
    all_scores = []
    
    for i in range(len(search.cv_results_['params'])):
        all_params.append(search.cv_results_['params'][i])
        score = search.cv_results_['mean_test_score'][i]
        if not higher_is_better:
            score = -score
        all_scores.append(score)
    
    # Create and return the result object
    result = OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_params=all_params,
        all_scores=all_scores,
        model_type=model_class.__name__,
        optimization_method=method,
        metric_name=scoring.replace('neg_', '') if scoring.startswith('neg_') else scoring,
        higher_is_better=higher_is_better,
        cv_results=search.cv_results_
    )
    
    return result


def optimize_sklearn_model_optuna(
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_ranges: Dict[str, Callable],
    cv: int = 5,
    scoring: Callable = None,
    n_trials: int = 100,
    timeout: Optional[int] = None,
    verbose: bool = True,
    random_state: int = 42,
    higher_is_better: bool = False
) -> OptimizationResult:
    """
    Optimize hyperparameters for scikit-learn models using Bayesian Optimization with Optuna.
    
    Args:
        model_class: Scikit-learn model class
        X_train: Training features
        y_train: Training target
        param_ranges: Dictionary mapping parameter names to callable functions that
                     generate trial parameters from an optuna trial
                     e.g., {'n_estimators': lambda trial: trial.suggest_int('n_estimators', 10, 100)}
        cv: Number of cross-validation folds or CV splitter object
        scoring: Callable scoring function (X, y, y_pred) -> score
        n_trials: Number of optimization trials
        timeout: Timeout in seconds for the optimization
        verbose: Whether to print progress
        random_state: Random state for reproducibility
        higher_is_better: Whether higher scores are better
    
    Returns:
        OptimizationResult: Object containing optimization results
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
    
    # Define default scoring function if none provided
    if scoring is None:
        # Default to MSE for regression, accuracy for classification
        # Determine if regression or classification based on y_train values
        if len(np.unique(y_train)) < 10 or all(isinstance(y, (int, bool)) for y in y_train):
            # Likely classification
            scoring = accuracy_score
            higher_is_better = True
        else:
            # Likely regression
            scoring = mean_squared_error
            higher_is_better = False
    
    # Set up cross-validation
    if isinstance(cv, int):
        # If dealing with time series data, use TimeSeriesSplit
        if isinstance(X_train, pd.DataFrame) and isinstance(X_train.index, pd.DatetimeIndex):
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv
    
    # Store all parameter configurations and scores
    all_params = []
    all_scores = []
    
    # Define the objective function for optimization
    def objective(trial):
        # Create parameter dictionary for this trial
        params = {}
        for param_name, param_func in param_ranges.items():
            params[param_name] = param_func(trial)
        
        # Create and configure the model
        model = model_class(**params)
        
        # Perform cross-validation
        cv_scores = []
        for train_idx, val_idx in cv_splitter.split(X_train):
            # Split data
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            model.fit(X_cv_train, y_cv_train)
            
            # Make predictions
            y_pred = model.predict(X_cv_val)
            
            # Compute score
            score = scoring(y_cv_val, y_pred)
            cv_scores.append(score)
        
        # Compute mean cross-validation score
        mean_score = np.mean(cv_scores)
        
        # Store the parameters and score
        all_params.append(params)
        all_scores.append(mean_score)
        
        # Return the score (Optuna minimizes the objective by default)
        return -mean_score if higher_is_better else mean_score
    
    # Create a study for optimization
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    
    # Start optimization
    logger.info(f"Starting Optuna optimization for {model_class.__name__} with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    logger.info(f"Completed Optuna optimization for {model_class.__name__}")
    
    # Get the best parameters
    best_params = study.best_params
    best_score = -study.best_value if higher_is_better else study.best_value
    
    # Create and return the result object
    result = OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_params=all_params,
        all_scores=all_scores,
        model_type=model_class.__name__,
        optimization_method="bayesian_optimization",
        metric_name=scoring.__name__,
        higher_is_better=higher_is_better
    )
    
    return result


def optimize_pytorch_model(
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_ranges: Dict[str, Callable],
    validation_data: Tuple[np.ndarray, np.ndarray] = None,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 5,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    device: str = None,
    verbose: bool = True,
    random_state: int = 42
) -> OptimizationResult:
    """
    Optimize hyperparameters for PyTorch models using Optuna.
    
    Args:
        model_class: PyTorch model class
        X_train: Training features
        y_train: Training target
        param_ranges: Dictionary mapping parameter names to callables that
                     generate trial parameters from an optuna trial
                     e.g., {'hidden_dim': lambda trial: trial.suggest_int('hidden_dim', 32, 256)}
        validation_data: Optional tuple of (X_val, y_val) for validation
        batch_size: Mini-batch size (can also be included in param_ranges)
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs with no improvement)
        n_trials: Number of optimization trials
        timeout: Timeout in seconds for the optimization
        device: PyTorch device ('cuda' or 'cpu')
        verbose: Whether to print progress
        random_state: Random state for reproducibility
    
    Returns:
        OptimizationResult: Object containing optimization results
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PyTorch model optimization. Install with: pip install torch")
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"Using device: {device}")
    
    # Create validation set if not provided
    if validation_data is None:
        # Split data
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
    else:
        X_train_split, y_train_split = X_train, y_train
        X_val, y_val = validation_data
    
    # Convert to PyTorch tensors
    if not isinstance(X_train_split, torch.Tensor):
        X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    else:
        X_train_tensor, y_train_tensor = X_train_split, y_train_split
        X_val_tensor, y_val_tensor = X_val, y_val
    
    # Store all parameter configurations and scores
    all_params = []
    all_scores = []
    
    # Define the objective function for optimization
    def objective(trial):
        # Get hyperparameters for this trial
        params = {}
        for param_name, param_func in param_ranges.items():
            params[param_name] = param_func(trial)

        # Check if model_class is a wrapper (like PyTorchStockPredictor) or a raw nn.Module
        # Check class attributes instead of instantiating to avoid parameter mismatches
        is_wrapper = (
            hasattr(model_class, 'fit') and
            hasattr(model_class, 'predict') and
            not issubclass(model_class, nn.Module)
        )

        if is_wrapper:
            # Use wrapper's fit/predict interface (e.g., PyTorchStockPredictor)
            wrapper = model_class(**params)

            # Convert tensors to numpy for wrapper's fit method
            if isinstance(X_train_tensor, torch.Tensor):
                X_train_np = X_train_tensor.cpu().numpy()
                y_train_np = y_train_tensor.cpu().numpy()
                X_val_np = X_val_tensor.cpu().numpy()
                y_val_np = y_val_tensor.cpu().numpy()
            else:
                X_train_np = X_train_tensor
                y_train_np = y_train_tensor
                X_val_np = X_val_tensor
                y_val_np = y_val_tensor

            # Train using wrapper's fit method
            lr = params.get('lr', 0.001)
            local_batch_size = params.get('batch_size', batch_size)
            wrapper.fit(X_train_np, y_train_np, epochs=max_epochs, batch_size=local_batch_size, lr=lr, verbose=False)

            # Validate using wrapper's predict method
            val_predictions = wrapper.predict(X_val_np)

            # Calculate validation loss
            val_loss = np.mean((val_predictions - y_val_np) ** 2)
            best_val_loss = float(val_loss)

        else:
            # Raw nn.Module - use manual training loop
            model = model_class(**params).to(device)

            # Define loss function and optimizer
            criterion = nn.MSELoss()

            # Get optimizer from trial if specified, otherwise use Adam
            if 'optimizer_type' in param_ranges:
                optimizer_type = param_ranges['optimizer_type'](trial)
                if optimizer_type == 'adam':
                    optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
                elif optimizer_type == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=params.get('lr', 0.01),
                                          momentum=params.get('momentum', 0.9))
                elif optimizer_type == 'rmsprop':
                    optimizer = optim.RMSprop(model.parameters(), lr=params.get('lr', 0.001))
                else:
                    optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 0.001))
            else:
                optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 0.001))

            # Create DataLoader
            local_batch_size = params.get('batch_size', batch_size)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=local_batch_size, shuffle=True)

            # Training loop
            best_val_loss = float('inf')
            epochs_no_improve = 0

            for epoch in range(max_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    X_val_tensor_local = X_val_tensor.to(device)
                    y_val_tensor_local = y_val_tensor.to(device)
                    val_outputs = model(X_val_tensor_local)
                    val_loss = criterion(val_outputs, y_val_tensor_local).item()

                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    # Early stopping check
                    if epochs_no_improve >= patience:
                        if verbose:
                            logger.info(f"Early stopping after {epoch+1} epochs")
                        break

        # Store the parameters and score
        all_params.append(params)
        all_scores.append(-best_val_loss)  # Convert to maximize (negative loss)

        return best_val_loss
    
    # Create a study for optimization
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    
    # Start optimization
    logger.info(f"Starting Optuna optimization for PyTorch model with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    logger.info(f"Completed Optuna optimization for PyTorch model")
    
    # Get the best parameters
    best_params = study.best_params
    best_score = -study.best_value  # Convert to negative for consistency (higher is better)
    
    # Create and return the result object
    result = OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_params=all_params,
        all_scores=all_scores,
        model_type=model_class.__name__,
        optimization_method="pytorch_optuna",
        metric_name="negative_validation_loss",
        higher_is_better=True
    )
    
    return result


def optimize_tensorflow_model(
    model_builder: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_ranges: Dict[str, Callable],
    validation_data: Tuple[np.ndarray, np.ndarray] = None,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 5,
    n_trials: int = 20,
    timeout: Optional[int] = None,
    verbose: bool = True,
    random_state: int = 42
) -> OptimizationResult:
    """
    Optimize hyperparameters for TensorFlow/Keras models using Optuna.
    
    Args:
        model_builder: Function that takes parameters and returns a compiled TF model
        X_train: Training features
        y_train: Training target
        param_ranges: Dictionary mapping parameter names to callables that
                     generate trial parameters from an optuna trial
                     e.g., {'units': lambda trial: trial.suggest_int('units', 32, 256)}
        validation_data: Optional tuple of (X_val, y_val) for validation
        batch_size: Mini-batch size (can also be included in param_ranges)
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs with no improvement)
        n_trials: Number of optimization trials
        timeout: Timeout in seconds for the optimization
        verbose: Whether to print progress
        random_state: Random state for reproducibility
    
    Returns:
        OptimizationResult: Object containing optimization results
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for TF model optimization. Install with: pip install tensorflow")
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
    
    # Set random seed for TensorFlow
    tf.random.set_seed(random_state)
    
    # Create validation set if not provided
    if validation_data is None:
        # Split data
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state
        )
        validation_data = (X_val, y_val)
    else:
        X_train_split, y_train_split = X_train, y_train
    
    # Store all parameter configurations and scores
    all_params = []
    all_scores = []
    
    # Define the objective function for optimization
    def objective(trial):
        # Get hyperparameters for this trial
        params = {}
        for param_name, param_func in param_ranges.items():
            params[param_name] = param_func(trial)
        
        # Build model using the provided builder function
        model = model_builder(**params)
        
        # Define callbacks for early stopping
        callbacks = []
        if patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)
        
        # Train the model
        local_batch_size = params.get('batch_size', batch_size)
        history = model.fit(
            X_train_split, y_train_split,
            epochs=max_epochs,
            batch_size=local_batch_size,
            validation_data=validation_data,
            verbose=0 if not verbose else 1,
            callbacks=callbacks
        )
        
        # Get the best validation loss
        val_losses = history.history['val_loss']
        best_val_loss = min(val_losses)
        
        # Store the parameters and score
        all_params.append(params)
        all_scores.append(-best_val_loss)  # Convert to maximize (negative loss)
        
        return best_val_loss
    
    # Create a study for optimization
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    
    # Start optimization
    logger.info(f"Starting Optuna optimization for TensorFlow model with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    logger.info(f"Completed Optuna optimization for TensorFlow model")
    
    # Get the best parameters
    best_params = study.best_params
    best_score = -study.best_value  # Convert to negative for consistency (higher is better)
    
    # Create and return the result object
    result = OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_params=all_params,
        all_scores=all_scores,
        model_type="TensorFlow",
        optimization_method="tensorflow_optuna",
        metric_name="negative_validation_loss",
        higher_is_better=True
    )
    
    return result


def visualize_optimization_results(results: OptimizationResult, 
                                  plots: List[str] = None,
                                  save_dir: str = None) -> Dict:
    """
    Visualize optimization results with plots.
    
    Args:
        results: OptimizationResult object
        plots: List of plot types to generate ['history', 'importance', 'contour', 'slice']
        save_dir: Directory to save plots
    
    Returns:
        Dict: Dictionary of generated plots
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for visualization. Install with: pip install optuna")
    
    # Default plots
    if plots is None:
        plots = ['history']
    
    # Create output directory if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # Initialize dictionary of plots
    generated_plots = {}
    
    # Create optimization history plot
    if 'history' in plots:
        history_plot = results.plot_optimization_history()
        generated_plots['history'] = history_plot
        
        if save_dir:
            save_path = os.path.join(save_dir, f"optimization_history_{results.model_type}.png")
            history_plot.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # If we have Optuna study results from results.all_params and results.all_scores,
    # convert them to a study for more advanced visualizations
    if results.all_params and results.all_scores:
        try:
            # Create a new study
            study = optuna.create_study(direction="maximize" if results.higher_is_better else "minimize")
            
            # Add trials to the study
            for params, score in zip(results.all_params, results.all_scores):
                trial = optuna.trial.create_trial(
                    params=params,
                    value=score if results.higher_is_better else -score,
                    state=optuna.trial.TrialState.COMPLETE
                )
                study.add_trial(trial)
            
            # Generate parameter importance plot
            if 'importance' in plots:
                try:
                    importance_plot = vis.plot_param_importances(study)
                    generated_plots['importance'] = importance_plot
                    
                    if save_dir:
                        save_path = os.path.join(save_dir, f"param_importance_{results.model_type}.png")
                        importance_plot.write_image(save_path)
                except Exception as e:
                    logger.warning(f"Failed to generate parameter importance plot: {str(e)}")
            
            # Generate contour plot
            if 'contour' in plots and len(results.all_params[0]) >= 2:
                try:
                    contour_plot = vis.plot_contour(study)
                    generated_plots['contour'] = contour_plot
                    
                    if save_dir:
                        save_path = os.path.join(save_dir, f"contour_plot_{results.model_type}.png")
                        contour_plot.write_image(save_path)
                except Exception as e:
                    logger.warning(f"Failed to generate contour plot: {str(e)}")
            
            # Generate slice plot
            if 'slice' in plots:
                try:
                    slice_plot = vis.plot_slice(study)
                    generated_plots['slice'] = slice_plot
                    
                    if save_dir:
                        save_path = os.path.join(save_dir, f"slice_plot_{results.model_type}.png")
                        slice_plot.write_image(save_path)
                except Exception as e:
                    logger.warning(f"Failed to generate slice plot: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Failed to generate Optuna visualizations: {str(e)}")
    
    return generated_plots


def create_param_grid_example(model_type='random_forest', task_type='regression'):
    """
    Create example parameter grids/ranges for common model types.
    
    Args:
        model_type: Type of model ('random_forest', 'xgboost', 'neural_network', etc.)
        task_type: Type of task ('regression' or 'classification')
        
    Returns:
        tuple: (param_grid for grid/random search, param_ranges for Optuna)
    """
    # Different parameter grids based on model type
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        
        param_ranges = {
            'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
            'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': lambda trial: trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }
        
    elif model_type == 'gradient_boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        param_ranges = {
            'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0)
        }
        
    elif model_type == 'xgboost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        param_ranges = {
            'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': lambda trial: trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': lambda trial: trial.suggest_int('min_child_weight', 1, 10),
            'gamma': lambda trial: trial.suggest_float('gamma', 0, 0.5),
            'subsample': lambda trial: trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': lambda trial: trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
    elif model_type == 'svm':
        if task_type == 'classification':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
            
            param_ranges = {
                'C': lambda trial: trial.suggest_float('C', 0.1, 100, log=True),
                'kernel': lambda trial: trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                'gamma': lambda trial: trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('gamma_type', ['auto', 'value']) == 'auto' else trial.suggest_float('gamma', 0.001, 10, log=True)
            }
        else:  # regression
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1],
                'epsilon': [0.01, 0.1, 0.2]
            }
            
            param_ranges = {
                'C': lambda trial: trial.suggest_float('C', 0.1, 100, log=True),
                'kernel': lambda trial: trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': lambda trial: trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('gamma_type', ['auto', 'value']) == 'auto' else trial.suggest_float('gamma', 0.001, 10, log=True),
                'epsilon': lambda trial: trial.suggest_float('epsilon', 0.01, 0.5)
            }
            
    elif model_type == 'pytorch_lstm':
        # These would be used for PyTorch model
        param_grid = {}  # Not applicable for PyTorch with standard grid search
        
        param_ranges = {
            'input_dim': lambda trial: trial.suggest_categorical('input_dim', [X_train.shape[1]]),  # Fixed by data
            'hidden_dim': lambda trial: trial.suggest_int('hidden_dim', 32, 256, step=32),
            'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
            'dropout': lambda trial: trial.suggest_float('dropout', 0.0, 0.5),
            'lr': lambda trial: trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size': lambda trial: trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }
        
    elif model_type == 'tensorflow_lstm':
        # These would be used for TensorFlow model
        param_grid = {}  # Not applicable for TensorFlow with standard grid search
        
        param_ranges = {
            'input_dim': lambda trial: trial.suggest_categorical('input_dim', [X_train.shape[1]]),  # Fixed by data
            'units': lambda trial: trial.suggest_int('units', 32, 256, step=32),
            'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
            'dropout': lambda trial: trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': lambda trial: trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'activation': lambda trial: trial.suggest_categorical('activation', ['relu', 'tanh'])
        }
    
    else:  # Default simple grid
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.1, 0.01, 0.001]
        }
        
        param_ranges = {
            'param1': lambda trial: trial.suggest_int('param1', 1, 10),
            'param2': lambda trial: trial.suggest_float('param2', 0.001, 0.1, log=True)
        }
    
    return param_grid, param_ranges


def recommend_optimization_method(n_params, data_size, compute_budget='medium', task_complexity='medium'):
    """
    Recommend an optimization method based on the problem characteristics.
    
    Args:
        n_params: Number of hyperparameters to optimize
        data_size: Size of the dataset (small, medium, large)
        compute_budget: Available compute resources (low, medium, high)
        task_complexity: Complexity of the task (low, medium, high)
        
    Returns:
        dict: Dictionary with recommended optimization approach
    """
    # Convert inputs to standardized format
    if isinstance(data_size, int):
        if data_size < 10000:
            data_size = 'small'
        elif data_size < 100000:
            data_size = 'medium'
        else:
            data_size = 'large'
    
    # Decision logic
    if n_params <= 3:
        if compute_budget == 'low':
            method = 'grid_search'
            n_trials = min(10, 5**n_params)  # Limit grid points for very low budget
        else:  # medium or high
            method = 'grid_search'
            n_trials = min(100, 5**n_params)  # Full grid up to a reasonable size
            
    elif n_params <= 5:
        if compute_budget == 'low':
            method = 'random_search'
            n_trials = 15
        elif compute_budget == 'medium':
            if data_size == 'small':
                method = 'grid_search'
                n_trials = min(100, 4**n_params)
            else:
                method = 'random_search'
                n_trials = 30
        else:  # high compute
            method = 'bayesian_optimization'
            n_trials = 50
            
    else:  # many parameters
        if compute_budget == 'low':
            method = 'random_search'
            n_trials = 20
        elif compute_budget == 'medium':
            method = 'bayesian_optimization'
            n_trials = 30
        else:  # high compute
            method = 'bayesian_optimization'
            n_trials = 100
    
    # Adjust for task complexity
    if task_complexity == 'high':
        n_trials = int(n_trials * 1.5)
    elif task_complexity == 'low':
        n_trials = int(n_trials * 0.7)
    
    # Calculate estimated time based on data size
    if data_size == 'small':
        time_factor = 1
    elif data_size == 'medium':
        time_factor = 5
    else:  # large
        time_factor = 20
    
    estimated_time = n_trials * time_factor  # In minutes
    
    # Return recommendation
    return {
        'recommended_method': method,
        'recommended_trials': n_trials,
        'estimated_time_minutes': estimated_time,
        'cross_validation': 5 if data_size != 'large' else 3,
    }


def main():
    """Main function for command-line usage."""
    if len(sys.argv) != 3:
        print("Usage: python hyperparameter_optimization.py input.json output.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Load input JSON
        with open(input_file, 'r') as f:
            config = json.load(f)
        
        # Extract configuration
        model_type = config.get('model_type', 'random_forest')
        data = config.get('data', {})
        X = np.array(data.get('features', []))
        y = np.array(data.get('target', []))
        
        # Check if we have data
        if len(X) == 0 or len(y) == 0:
            result = {
                "error": "No data provided in input file",
                "status": "error"
            }
        else:
            # Get optimization configuration
            optimization_config = config.get('optimization', {})
            method = optimization_config.get('method', 'auto')
            max_trials = optimization_config.get('max_trials', 20)
            
            # If method is auto, recommend an optimization method
            if method == 'auto':
                recommendation = recommend_optimization_method(
                    n_params=len(config.get('param_grid', {}).keys()),
                    data_size=len(X),
                    compute_budget=optimization_config.get('compute_budget', 'medium'),
                    task_complexity=optimization_config.get('task_complexity', 'medium')
                )
                method = recommendation['recommended_method']
                max_trials = recommendation.get('recommended_trials', max_trials)
                
            # Get or create parameter grid
            if 'param_grid' in config:
                param_grid = config['param_grid']
                param_ranges = {}  # Convert param_grid to param_ranges for Optuna
                for param, values in param_grid.items():
                    if isinstance(values, list):
                        param_ranges[param] = lambda trial, p=param, vals=values: trial.suggest_categorical(p, vals)
            else:
                # Use default parameter grid based on model type
                param_grid, param_ranges = create_param_grid_example(model_type, optimization_config.get('task_type', 'regression'))
                
            # Run optimization based on model type
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                
                # Determine if regression or classification
                task_type = optimization_config.get('task_type', 'auto')
                if task_type == 'auto':
                    # Infer from data
                    n_unique = len(np.unique(y))
                    task_type = 'classification' if n_unique <= 10 else 'regression'
                
                # Select the appropriate model class
                model_class = RandomForestRegressor if task_type == 'regression' else RandomForestClassifier
                
                # Determine the scoring metric
                if task_type == 'regression':
                    scoring = optimization_config.get('scoring', 'neg_mean_squared_error')
                else:
                    scoring = optimization_config.get('scoring', 'f1')
                
                # Run the appropriate optimization method
                if method == 'bayesian_optimization':
                    result_obj = optimize_sklearn_model_optuna(
                        model_class=model_class,
                        X_train=X,
                        y_train=y,
                        param_ranges=param_ranges,
                        n_trials=max_trials,
                        cv=optimization_config.get('cv', 5),
                        verbose=True,
                        random_state=optimization_config.get('random_state', 42)
                    )
                else:
                    result_obj = optimize_sklearn_model(
                        model_class=model_class,
                        X_train=X,
                        y_train=y,
                        param_grid=param_grid,
                        method='random_search' if method == 'random_search' else 'grid_search',
                        n_iter=max_trials,
                        scoring=scoring,
                        cv=optimization_config.get('cv', 5),
                        verbose=1,
                        random_state=optimization_config.get('random_state', 42)
                    )
                
                # Save results
                result_path = result_obj.save()
                
                # Create response
                result = {
                    "best_params": result_obj.best_params,
                    "best_score": result_obj.best_score,
                    "model_type": result_obj.model_type,
                    "optimization_method": result_obj.optimization_method,
                    "metric_name": result_obj.metric_name,
                    "result_path": result_path,
                    "status": "success"
                }
                
            else:
                # For other model types - implement specific handling
                result = {
                    "error": f"Model type '{model_type}' not yet implemented for CLI usage",
                    "status": "error"
                }
        
        # Write output JSON
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Return success
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        
        # Create error response
        error_result = {
            "error": str(e),
            "status": "error"
        }
        
        # Write error output
        with open(output_file, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        sys.exit(1)


if __name__ == "__main__":
    main()