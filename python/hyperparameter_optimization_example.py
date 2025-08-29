#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter Optimization Example Script

This script demonstrates how to use the hyperparameter_optimization module
with the different models in the Quantra application.
"""

import numpy as np
import pandas as pd
import os
import sys
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hyperparameter_optimization_example')

# Import the hyperparameter optimization module
from hyperparameter_optimization import (
    optimize_sklearn_model,
    optimize_sklearn_model_optuna,
    optimize_pytorch_model,
    optimize_tensorflow_model,
    visualize_optimization_results,
    create_param_grid_example,
    OptimizationResult
)

# Try to import model-specific modules
try:
    from stock_predictor import (
        PyTorchStockPredictor, 
        TensorFlowStockPredictor,
        create_features,
        prepare_data_for_ml
    )
    STOCK_PREDICTOR_AVAILABLE = True
except ImportError:
    STOCK_PREDICTOR_AVAILABLE = False
    logger.warning("stock_predictor module not available, some examples may not work")

# Check for available frameworks
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def generate_sample_data(n_samples=1000, n_features=10, random_state=42):
    """Generate sample data for examples."""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target (regression)
    true_weights = np.random.randn(n_features)
    y = X.dot(true_weights) + 0.1 * np.random.randn(n_samples)
    
    return X, y


def example_sklearn_random_forest():
    """Example optimizing a scikit-learn Random Forest model."""
    logger.info("Running Random Forest optimization example")
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=500, n_features=10)
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Import the Random Forest model
    from sklearn.ensemble import RandomForestRegressor
    
    # Run grid search optimization
    logger.info("Running Grid Search optimization")
    grid_result = optimize_sklearn_model(
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        method="grid_search",
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1
    )
    
    # Save and print results
    grid_result.save()
    logger.info(f"Grid Search Best Parameters: {grid_result.best_params}")
    logger.info(f"Grid Search Best Score: {grid_result.best_score}")
    
    # Define parameter ranges for Bayesian optimization
    param_ranges = {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 300),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
    }
    
    # Run Bayesian optimization
    logger.info("Running Bayesian Optimization with Optuna")
    optuna_result = optimize_sklearn_model_optuna(
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        param_ranges=param_ranges,
        n_trials=30,
        cv=3,
        verbose=True
    )
    
    # Save and print results
    optuna_result.save()
    logger.info(f"Optuna Best Parameters: {optuna_result.best_params}")
    logger.info(f"Optuna Best Score: {optuna_result.best_score}")
    
    # Compare the optimization methods
    logger.info("Comparing optimization methods:")
    logger.info(f"Grid Search: MSE = {-grid_result.best_score}")
    logger.info(f"Bayesian Optimization: MSE = {-optuna_result.best_score}")
    
    # Visualize optimization results
    visualize_optimization_results(optuna_result, plots=['history'], save_dir='.')
    
    # Create a model with the best parameters and evaluate on test set
    best_model = RandomForestRegressor(**optuna_result.best_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test set evaluation:")
    logger.info(f"MSE: {test_mse}")
    logger.info(f"R²: {test_r2}")
    
    return optuna_result


def example_pytorch_lstm():
    """Example optimizing a PyTorch LSTM model."""
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch is not available. Skipping PyTorch LSTM example.")
        return None
    
    if not STOCK_PREDICTOR_AVAILABLE:
        logger.warning("stock_predictor module not available. Skipping PyTorch LSTM example.")
        return None
    
    logger.info("Running PyTorch LSTM optimization example")
    
    # Generate time series data
    n_samples = 500
    seq_length = 10
    n_features = 5
    
    # Create sequential data with time dependency
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    
    # Create a target with temporal dependency
    for i in range(seq_length, n_samples):
        y[i] = 0.1 * np.sum(X[i-seq_length:i, 0]) + 0.1 * np.random.randn()
    
    # Reshape data for LSTM: [samples, sequence_length, features]
    X_lstm = np.zeros((n_samples - seq_length, seq_length, n_features))
    y_lstm = np.zeros(n_samples - seq_length)
    
    for i in range(len(X_lstm)):
        X_lstm[i] = X[i:i+seq_length]
        y_lstm[i] = y[i+seq_length]
    
    # Split data
    split_idx = int(0.8 * len(X_lstm))
    X_train, X_val = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_val = y_lstm[:split_idx], y_lstm[split_idx:]
    
    # Define a simplified PyTorch LSTM model for optimization
    import torch
    import torch.nn as nn
    
    class SimpleLSTM(nn.Module):
        def __init__(self, input_dim=n_features, hidden_dim=64, num_layers=1, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # Take only the last time step
            return output.squeeze(-1)
    
    # Define parameter ranges
    param_ranges = {
        'hidden_dim': lambda trial: trial.suggest_int('hidden_dim', 16, 128, step=16),
        'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
        'dropout': lambda trial: trial.suggest_float('dropout', 0.0, 0.5),
    }
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Run the optimization
    logger.info("Running PyTorch LSTM optimization")
    result = optimize_pytorch_model(
        model_class=SimpleLSTM,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        param_ranges=param_ranges,
        validation_data=(X_val_tensor, y_val_tensor),
        batch_size=32,
        max_epochs=20,
        patience=3,
        n_trials=20,
        verbose=True
    )
    
    # Save and print results
    result.save()
    logger.info(f"Best Parameters: {result.best_params}")
    logger.info(f"Best Score: {result.best_score}")
    
    # Visualize results
    visualize_optimization_results(result, plots=['history'], save_dir='.')
    
    return result


def example_tensorflow_lstm():
    """Example optimizing a TensorFlow LSTM model."""
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow is not available. Skipping TensorFlow LSTM example.")
        return None
    
    if not STOCK_PREDICTOR_AVAILABLE:
        logger.warning("stock_predictor module not available. Skipping TensorFlow LSTM example.")
        return None
    
    logger.info("Running TensorFlow LSTM optimization example")
    
    # Generate time series data (same as in PyTorch example)
    n_samples = 500
    seq_length = 10
    n_features = 5
    
    # Create sequential data
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    
    # Create a target with temporal dependency
    for i in range(seq_length, n_samples):
        y[i] = 0.1 * np.sum(X[i-seq_length:i, 0]) + 0.1 * np.random.randn()
    
    # Reshape data for LSTM: [samples, sequence_length, features]
    X_lstm = np.zeros((n_samples - seq_length, seq_length, n_features))
    y_lstm = np.zeros(n_samples - seq_length)
    
    for i in range(len(X_lstm)):
        X_lstm[i] = X[i:i+seq_length]
        y_lstm[i] = y[i+seq_length]
    
    # Split data
    split_idx = int(0.8 * len(X_lstm))
    X_train, X_val = X_lstm[:split_idx], X_lstm[split_idx:]
    y_train, y_val = y_lstm[:split_idx], y_lstm[split_idx:]
    
    # Define a function to create TensorFlow LSTM model
    def create_tf_lstm_model(units=64, num_layers=1, dropout=0.0, learning_rate=0.001):
        import tensorflow as tf
        from tensorflow import keras
        
        # Create model
        model = keras.Sequential()
        
        # Add LSTM layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            if i == 0:
                model.add(keras.layers.LSTM(
                    units=units,
                    input_shape=(seq_length, n_features),
                    return_sequences=return_sequences
                ))
            else:
                model.add(keras.layers.LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))
            
            if dropout > 0 and i < num_layers - 1:
                model.add(keras.layers.Dropout(dropout))
        
        # Output layer
        model.add(keras.layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    # Define parameter ranges
    param_ranges = {
        'units': lambda trial: trial.suggest_int('units', 16, 128, step=16),
        'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
        'dropout': lambda trial: trial.suggest_float('dropout', 0.0, 0.5),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    }
    
    # Run the optimization
    logger.info("Running TensorFlow LSTM optimization")
    result = optimize_tensorflow_model(
        model_builder=create_tf_lstm_model,
        X_train=X_train,
        y_train=y_train,
        param_ranges=param_ranges,
        validation_data=(X_val, y_val),
        batch_size=32,
        max_epochs=20,
        patience=3,
        n_trials=20,
        verbose=True
    )
    
    # Save and print results
    result.save()
    logger.info(f"Best Parameters: {result.best_params}")
    logger.info(f"Best Score: {result.best_score}")
    
    # Visualize results
    visualize_optimization_results(result, plots=['history'], save_dir='.')
    
    return result


def example_stock_predictor_integration():
    """Example showcasing integration with the stock_predictor.py module."""
    if not STOCK_PREDICTOR_AVAILABLE:
        logger.warning("stock_predictor module not available. Skipping integration example.")
        return None
    
    logger.info("Running stock_predictor integration example")
    
    # Generate synthetic stock data
    n_samples = 300
    dates = [datetime.now() - timedelta(days=i) for i in range(n_samples, 0, -1)]
    
    np.random.seed(42)
    prices = [100]  # Start with $100
    for i in range(1, n_samples):
        # Random walk with drift
        change = np.random.normal(0.0005, 0.01)  # Small upward drift
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data
    data = {
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'volume': [int(abs(np.random.normal(1000000, 200000))) for _ in prices]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create features
    features_df = create_features(df)
    
    # Prepare data for ML
    X, y = prepare_data_for_ml(df)
    
    # Split the data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Define parameters for RandomForest optimization
    param_ranges = {
        'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 200),
        'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 5)
    }
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Run Bayesian optimization
    logger.info("Optimizing RandomForest for stock prediction")
    result = optimize_sklearn_model_optuna(
        model_class=RandomForestRegressor,
        X_train=X_train,
        y_train=y_train,
        param_ranges=param_ranges,
        n_trials=20,
        cv=3,
        verbose=True
    )
    
    # Save and print results
    result.save()
    logger.info(f"Best Parameters: {result.best_params}")
    logger.info(f"Best Score: {result.best_score}")
    
    # Create the best model
    best_model = RandomForestRegressor(**result.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Evaluate on test set
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test MSE: {mse}")
    logger.info(f"Test R²: {r2}")
    
    return result


def main():
    """Run selected examples based on command-line arguments."""
    # Default to running all available examples
    examples_to_run = ["rf", "pytorch", "tensorflow", "integration"]
    
    if len(sys.argv) > 1:
        examples_to_run = sys.argv[1].split(",")
    
    results = {}
    
    if "rf" in examples_to_run:
        results["random_forest"] = example_sklearn_random_forest()
    
    if "pytorch" in examples_to_run:
        results["pytorch"] = example_pytorch_lstm()
        
    if "tensorflow" in examples_to_run:
        results["tensorflow"] = example_tensorflow_lstm()
        
    if "integration" in examples_to_run:
        results["integration"] = example_stock_predictor_integration()
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Optimization Examples Summary:")
    for name, result in results.items():
        if result:
            logger.info(f"{name}: Best score = {result.best_score}")
        else:
            logger.info(f"{name}: Not run or failed")
    
    logger.info("=" * 50)


if __name__ == "__main__":
    main()