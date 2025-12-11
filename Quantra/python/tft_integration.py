#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration layer between TFT and existing stock_predictor.py infrastructure.
Provides TFTStockPredictor wrapper compatible with existing prediction interface.

CRITICAL FIXES:
1. Properly accepts real historical sequences instead of synthetic repeated values
2. Validates that targets are properly formatted for multi-horizon forecasting
3. Prevents data leakage by fitting scalers ONLY on training data

IMPORTANT: Data Preprocessing and Look-Ahead Bias Prevention
=============================================================
To prevent look-ahead bias (data leakage), the preprocessing pipeline follows these steps:

1. Split data into train/validation sets BEFORE any scaling (80/20 split, sequential for time series)
2. Fit all scalers (temporal, static, target) ONLY on training data
3. Transform both train and validation data using the fitted scalers
4. During inference, use the same fitted scalers (never refit on test data)

INCORRECT (Data Leakage):
    # Wrong: Fit scaler on all data (train + validation together)
    X_scaled = scaler.fit_transform(X_all)
    X_train, X_val = split(X_scaled)  # Validation statistics leaked into training!

CORRECT (No Data Leakage):
    # Right: Split first, then fit only on training data
    X_train, X_val = split(X_all)
    scaler.fit(X_train)  # Fit only on training data
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # Use same scaler for validation

This ensures that:
- Training data never sees statistics from validation data
- Validation/test performance accurately reflects real-world generalization
- No information from the "future" leaks into the past

IMPORTANT: Multi-Horizon Target Requirements
============================================
For multi-horizon forecasting, targets MUST be a 2D array of shape (n_samples, num_horizons)
where each column corresponds to the actual future value at that specific forecast horizon.

INCORRECT (DO NOT DO THIS):
    # Wrong: Repeating the same target for all horizons
    y_wrong = np.column_stack([y_single] * len(forecast_horizons))
    # This trains the model to predict the same value for 5 days and 30 days ahead!

CORRECT (DO THIS):
    # Right: Compute actual targets for each horizon from sequential price data
    prices = np.array([100, 102, 105, 103, 108, 110, 115, 112, 120, ...])
    forecast_horizons = [5, 10, 20, 30]

    # Use the helper function to compute proper targets
    y_correct = compute_multi_horizon_targets(prices, forecast_horizons)
    # For sample at t=0 (price=100):
    #   - y_correct[0, 0] = (prices[5] - 100) / 100   # 5-day return
    #   - y_correct[0, 1] = (prices[10] - 100) / 100  # 10-day return
    #   - y_correct[0, 2] = (prices[20] - 100) / 100  # 20-day return
    #   - y_correct[0, 3] = (prices[30] - 100) / 100  # 30-day return

The fit() method will raise a clear error if targets are not properly formatted.
"""

import torch
import numpy as np
import os
import sys
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

logger = logging.getLogger('tft_integration')

# Constants
DEFAULT_STATIC_DIM = 10  # Default number of static features (sector, market cap, etc.)
DEFAULT_LOOKBACK = 60  # Default lookback sequence length

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
TFT_MODEL_PATH = os.path.join(MODEL_DIR, 'tft_model.pt')
TFT_SCALER_PATH = os.path.join(MODEL_DIR, 'tft_scaler.pkl')

# Add BASE_DIR to sys.path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import TFT components
try:
    from temporal_fusion_transformer import (
        TemporalFusionTransformer, 
        train_tft_model, 
        QuantileLoss
    )
    TFT_AVAILABLE = True
except ImportError as e:
    TFT_AVAILABLE = False
    logging.warning(f"TFT not available: {e}")


class TFTStockPredictor:
    """
    Wrapper for TFT model compatible with existing predictor interface.
    Provides multi-horizon forecasting with uncertainty quantification.
    
    CRITICAL FIX: Now properly handles real historical sequences in predict_single()
    """
    
    def __init__(self, input_dim=50, static_dim=DEFAULT_STATIC_DIM, hidden_dim=128, 
                 forecast_horizons=None, num_heads=4, num_lstm_layers=2,
                 dropout=0.1, num_attention_layers=2):
        """
        Initialize TFT Stock Predictor.
        
        Args:
            input_dim: Number of temporal input features
            static_dim: Number of static features (sector, market cap, etc.)
            hidden_dim: Hidden dimension for all layers
            forecast_horizons: List of forecast horizons (e.g., [5, 10, 20, 30] days)
            num_heads: Number of attention heads
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            num_attention_layers: Number of self-attention layers
        """
        if forecast_horizons is None:
            forecast_horizons = [5, 10, 20, 30]
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizons = forecast_horizons
        self.num_heads = num_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.num_attention_layers = num_attention_layers
        
        self.model = TemporalFusionTransformer(
            input_dim=input_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            forecast_horizons=forecast_horizons,
            num_attention_layers=num_attention_layers
        ).to(self.device)
        
        self.scaler = None
        self.static_scaler = None
        self.target_scaler = None  # NEW: Scaler for target values (percentage changes)
        self.feature_names = []
        self.is_trained = False
        
        logger.info(f"Initialized TFTStockPredictor on {self.device}")
        
    def fit(self, X_past: np.ndarray, X_static: np.ndarray, y: np.ndarray,
            future_features: Optional[np.ndarray] = None,
            epochs: int = 50, batch_size: int = 32, lr: float = 0.001,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the TFT model.

        Args:
            X_past: (n_samples, seq_len, features) - Historical temporal features
            X_static: (n_samples, static_features) - Static features
            y: (n_samples, num_horizons) - Target values for each forecast horizon
               CRITICAL: y MUST be (n_samples, num_horizons) where each column corresponds
               to the actual future value at that horizon. Use compute_multi_horizon_targets()
               to create proper targets from sequential price data.
               DO NOT pass single-value targets - each horizon needs its own target!
            future_features: (n_samples, forecast_horizon, future_features) - Known future features (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate
            verbose: Whether to print training progress

        Returns:
            Training history with train_loss and val_loss

        Raises:
            ValueError: If y shape doesn't match the number of forecast horizons
        """
        # CRITICAL: Split data BEFORE scaling to prevent look-ahead bias
        # We must fit scalers ONLY on training data, not on validation data
        n_samples = X_past.shape[0]
        train_size = int(0.8 * n_samples)

        # Split indices
        train_indices = np.arange(train_size)
        val_indices = np.arange(train_size, n_samples)

        logger.info(f"Splitting {n_samples} samples: {train_size} train, {n_samples - train_size} validation")

        # Initialize scalers
        self.scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # CRITICAL FIX: Scale temporal features correctly to prevent data leakage
        # Fit scaler ONLY on training data
        seq_len, n_features = X_past.shape[1], X_past.shape[2]

        # Get training data and reshape for fitting
        X_past_train = X_past[train_indices]
        X_past_train_reshaped = X_past_train.reshape(-1, n_features)

        # Fit scaler on training data ONLY
        self.scaler.fit(X_past_train_reshaped)
        logger.info(f"Fitted temporal scaler on {X_past_train_reshaped.shape[0]} training time steps")

        # Transform all data using the fitted scaler
        X_past_reshaped = X_past.reshape(-1, n_features)
        X_past_scaled = self.scaler.transform(X_past_reshaped)
        X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)

        # Scale static features - fit only on training data
        X_static_train = X_static[train_indices]
        self.static_scaler.fit(X_static_train)
        X_static_scaled = self.static_scaler.transform(X_static)
        logger.info(f"Fitted static scaler on {X_static_train.shape[0]} training samples")

        # Handle future features if provided
        has_future = future_features is not None and future_features.size > 0
        if has_future:
            # Initialize future scaler
            self.future_scaler = StandardScaler()

            # Fit on training data ONLY
            future_seq_len, n_future_features = future_features.shape[1], future_features.shape[2]
            future_train = future_features[train_indices]
            future_train_reshaped = future_train.reshape(-1, n_future_features)

            self.future_scaler.fit(future_train_reshaped)
            logger.info(f"Fitted future scaler on {future_train_reshaped.shape[0]} training time steps")

            # Transform all data
            future_reshaped = future_features.reshape(-1, n_future_features)
            future_scaled = self.future_scaler.transform(future_reshaped)
            future_scaled = future_scaled.reshape(n_samples, future_seq_len, n_future_features)
            logger.info(f"Scaled future features: {future_scaled.shape}")
        else:
            future_scaled = None
            self.future_scaler = None
            logger.info("No future features provided for training")
        
        # CRITICAL: Validate target shape - targets MUST be (n_samples, num_horizons)
        # Each column must correspond to the actual future value at that horizon
        if y.ndim == 1:
            raise ValueError(
                f"Target array y must be 2D with shape (n_samples, {len(self.forecast_horizons)}), "
                f"but got 1D array of shape {y.shape}. Each forecast horizon needs its own target!\n"
                f"Use compute_multi_horizon_targets() to create proper multi-horizon targets from "
                f"sequential price data. DO NOT repeat the same target for all horizons - that's incorrect!"
            )

        if y.shape[1] != len(self.forecast_horizons):
            raise ValueError(
                f"Target array y has {y.shape[1]} columns but model has {len(self.forecast_horizons)} "
                f"forecast horizons {self.forecast_horizons}. Each horizon needs exactly one target column!\n"
                f"Expected shape: ({y.shape[0]}, {len(self.forecast_horizons)}), got {y.shape}"
            )

        logger.info(f"Validated targets: shape {y.shape} matches {len(self.forecast_horizons)} forecast horizons")
        logger.info(f"Target statistics by horizon (all data):")
        for i, horizon in enumerate(self.forecast_horizons):
            logger.info(f"  Horizon {horizon}: mean={y[:, i].mean():.4f}, std={y[:, i].std():.4f}, "
                       f"min={y[:, i].min():.4f}, max={y[:, i].max():.4f}")

        # CRITICAL FIX: Fit target scaler ONLY on training data to prevent look-ahead bias
        y_train = y[train_indices]
        self.target_scaler.fit(y_train)
        logger.info(f"Fitted target scaler on {y_train.shape[0]} training samples")
        logger.info(f"Target scaler statistics - mean: {self.target_scaler.mean_}, std: {np.sqrt(self.target_scaler.var_)}")

        # Transform all targets using the fitted scaler
        y_scaled = self.target_scaler.transform(y)
        
        # Create tensors
        past_tensor = torch.FloatTensor(X_past_scaled)
        static_tensor = torch.FloatTensor(X_static_scaled)
        # Use scaled targets: y_scaled contains percentage changes (e.g., 0.05 for 5%)
        # transformed by StandardScaler for improved training stability and convergence
        targets_tensor = torch.FloatTensor(y_scaled)

        # CRITICAL FIX: Use the same train/val split we used for fitting scalers
        # For time series, we use sequential split (not random) to maintain temporal order
        if has_future:
            future_tensor = torch.FloatTensor(future_scaled)
            # Split into train and val using our pre-computed indices
            train_dataset = TensorDataset(
                past_tensor[train_indices],
                static_tensor[train_indices],
                future_tensor[train_indices],
                targets_tensor[train_indices]
            )
            val_dataset = TensorDataset(
                past_tensor[val_indices],
                static_tensor[val_indices],
                future_tensor[val_indices],
                targets_tensor[val_indices]
            )
        else:
            # Split into train and val using our pre-computed indices
            train_dataset = TensorDataset(
                past_tensor[train_indices],
                static_tensor[train_indices],
                targets_tensor[train_indices]
            )
            val_dataset = TensorDataset(
                past_tensor[val_indices],
                static_tensor[val_indices],
                targets_tensor[val_indices]
            )

        logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # CRITICAL FIX: Use generator function wrapper without converting to list
        # This prevents loading entire dataset into memory and allows streaming
        class DictBatchWrapper:
            """Wraps DataLoader to yield dict batches instead of tuples."""
            def __init__(self, loader, include_future=False):
                self.loader = loader
                self.include_future = include_future

            def __iter__(self):
                for batch_data in self.loader:
                    if self.include_future:
                        past, static, future, targets = batch_data
                        yield {
                            'past_features': past,
                            'static_features': static,
                            'future_features': future,
                            'targets': targets
                        }
                    else:
                        past, static, targets = batch_data
                        yield {
                            'past_features': past,
                            'static_features': static,
                            'targets': targets
                        }

        # Wrap loaders to yield dicts (streaming, no memory overhead)
        train_loader_dict = DictBatchWrapper(train_loader, include_future=has_future)
        val_loader_dict = DictBatchWrapper(val_loader, include_future=has_future)

        logger.info(f"Using streaming data loaders (memory efficient)")

        # Train model
        history = train_tft_model(
            self.model,
            train_loader_dict,
            val_loader_dict,
            epochs=epochs,
            lr=lr,
            device=str(self.device)
        )
        
        self.is_trained = True
        
        if verbose:
            logger.info(f"Training complete. Final train loss: {history['train_loss'][-1]:.6f}")
            logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        
        return history
    
    def predict(self, X_past: np.ndarray, X_static: np.ndarray, batch_size: int = 512) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.
        Uses batched inference to avoid GPU memory issues with large datasets.
        
        Args:
            X_past: (n_samples, seq_len, features) - Historical temporal features
            X_static: (n_samples, static_features) - Static features
            batch_size: Batch size for inference (default: 512 for memory efficiency)
            
        Returns:
            dict with keys:
                - 'median_predictions': (n_samples, num_horizons)
                - 'lower_bound': (n_samples, num_horizons) - 10th percentile
                - 'upper_bound': (n_samples, num_horizons) - 90th percentile
                - 'q25': 25th percentile
                - 'q75': 75th percentile
                - 'feature_importance': (n_samples, input_dim)
                - 'attention_weights': List of attention weight arrays
        """
        # CRITICAL: Set model to evaluation mode and disable gradient computation
        # This prevents the "cudnn RNN backward can only be called in training mode" error
        self.model.eval()
        
        n_samples, seq_len, n_features = X_past.shape
        
        # Scale features
        X_past_reshaped = X_past.reshape(-1, n_features)
        X_past_scaled = self.scaler.transform(X_past_reshaped)
        X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
        X_static_scaled = self.static_scaler.transform(X_static)
        
        # Initialize result containers
        all_predictions = {horizon: [] for horizon in self.forecast_horizons}
        all_feature_importance = []
        all_attention_weights = []
        
        # Process in batches to avoid GPU OOM
        num_batches = (n_samples + batch_size - 1) // batch_size
        logger.info(f"Processing {n_samples} samples in {num_batches} batches of size {batch_size}")
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_past = X_past_scaled[i:end_idx]
            batch_static = X_static_scaled[i:end_idx]
            
            # CRITICAL: Use inference_mode for complete gradient disabling
            with torch.inference_mode():
                # Convert batch to tensors
                past_tensor = torch.FloatTensor(batch_past).to(self.device)
                static_tensor = torch.FloatTensor(batch_static).to(self.device)
                
                # Get predictions for this batch
                batch_outputs = self.model(past_tensor, static_tensor)
                
                # Store predictions for each horizon
                for horizon in self.forecast_horizons:
                    horizon_key = f"horizon_{horizon}"
                    quantiles = batch_outputs['predictions'][horizon_key].cpu().numpy()
                    all_predictions[horizon].append(quantiles)
                
                # Store feature importance and attention weights
                all_feature_importance.append(batch_outputs['variable_importance'].cpu().numpy())
                all_attention_weights.append([w.cpu().numpy() for w in batch_outputs['attention_weights']])
            
            # Clear GPU cache periodically
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        median_predictions = []
        lower_bounds = []
        upper_bounds = []
        q25_list = []
        q75_list = []
        
        for horizon in self.forecast_horizons:
            # Concatenate all batch predictions for this horizon
            quantiles_all = np.concatenate(all_predictions[horizon], axis=0)
            
            median_predictions.append(quantiles_all[:, 2])  # 50th percentile
            lower_bounds.append(quantiles_all[:, 0])        # 10th percentile
            upper_bounds.append(quantiles_all[:, 4])        # 90th percentile
            q25_list.append(quantiles_all[:, 1])            # 25th percentile
            q75_list.append(quantiles_all[:, 3])            # 75th percentile
        
        # Stack predictions and inverse transform all at once
        median_stacked = np.column_stack(median_predictions)
        lower_stacked = np.column_stack(lower_bounds)
        upper_stacked = np.column_stack(upper_bounds)
        q25_stacked = np.column_stack(q25_list)
        q75_stacked = np.column_stack(q75_list)
        
        # NEW: Inverse transform to get actual percentage changes
        median_unscaled = self.target_scaler.inverse_transform(median_stacked)
        lower_unscaled = self.target_scaler.inverse_transform(lower_stacked)
        upper_unscaled = self.target_scaler.inverse_transform(upper_stacked)
        q25_unscaled = self.target_scaler.inverse_transform(q25_stacked)
        q75_unscaled = self.target_scaler.inverse_transform(q75_stacked)
        
        # Concatenate feature importance across all batches
        feature_importance = np.concatenate(all_feature_importance, axis=0)
        
        # For attention weights, take the first batch as representative
        # (averaging across all batches would be memory-intensive and less meaningful)
        attention_weights = all_attention_weights[0] if all_attention_weights else []
        
        return {
            'median_predictions': median_unscaled,
            'lower_bound': lower_unscaled,
            'upper_bound': upper_unscaled,
            'q25': q25_unscaled,
            'q75': q75_unscaled,
            'feature_importance': feature_importance,
            'attention_weights': attention_weights
        }
    
    def predict_single(self,
                      historical_sequence: Optional[List[Dict[str, float]]] = None,
                      calendar_features: Optional[List[Dict[str, int]]] = None,
                      static_dict: Optional[Dict[str, Any]] = None,
                      features_dict: Optional[Dict[str, float]] = None,
                      lookback: int = 60) -> Dict[str, Any]:
        """
        Predict for a single symbol using REAL historical data.

        CRITICAL: Proper Usage Requires Real Historical Sequences
        ==========================================================
        TFT is a TEMPORAL model that learns from time-series patterns. It requires:
        - Actual historical sequences with temporal variation
        - NOT synthetic repeated values

        CORRECT Usage (Recommended):
            historical_sequence = [
                {'date': '2024-01-01', 'open': 100, 'close': 102, 'volume': 1000000, ...},
                {'date': '2024-01-02', 'open': 102, 'close': 98, 'volume': 950000, ...},
                ...  # 60 days of real data
            ]
            predictor.predict_single(historical_sequence=historical_sequence)

        INCORRECT Usage (Deprecated - Will Give Poor Results):
            features_dict = {'close': 100, 'volume': 1000000}  # Single snapshot
            predictor.predict_single(features_dict=features_dict)
            # This repeats the same values 60 times, defeating temporal modeling!

        Args:
            historical_sequence: REQUIRED - List of dicts with OHLCV data for last 60 days
                                Format: [{'date': '2024-01-01', 'close': 98.0, 'volume': 1M, ...}, ...]
                                Must contain at least 'open', 'high', 'low', 'close', 'volume' keys
            calendar_features: Optional list of calendar feature dicts for historical + future period
                              Format: [{'dayofweek': 0, 'month': 1, 'is_friday': 0, ...}, ...]
            static_dict: Optional dictionary of static features (sector, market_cap, etc.)
            features_dict: DEPRECATED - Single feature dict (creates unrealistic repeated sequence)
                          Only provided for backward compatibility - DO NOT USE for new code
            lookback: Number of lookback periods (default 60)

        Returns:
            Prediction result with multi-horizon forecasts and uncertainty

        Raises:
            ValueError: If historical_sequence is None/empty when features_dict is not provided
        """
        try:
            # Handle backward compatibility: if features_dict is provided but not historical_sequence
            if historical_sequence is None and features_dict is not None:
                logger.warning(
                    "DEPRECATED: Using legacy features_dict interface with synthetic repeated values. "
                    "This creates unrealistic temporal sequences (same value repeated 60 times) "
                    "which defeats the purpose of temporal modeling. "
                    "Please use 'historical_sequence' parameter with real historical data for proper predictions."
                )
                # Create synthetic sequence from single feature dict (LEGACY - NOT RECOMMENDED)
                # WARNING: This repeats the same values over time, which is unrealistic
                feature_names = list(features_dict.keys())
                feature_values = np.array([features_dict[k] for k in feature_names])
                n_features = len(feature_values)
                X_past = np.tile(feature_values.reshape(1, 1, n_features), (1, lookback, 1)).astype(np.float32)
                current_price = features_dict.get('current_price',
                               features_dict.get('close',
                               features_dict.get('price', 0.0)))
            else:
                # Use REAL historical sequence (preferred method)
                if historical_sequence is None or len(historical_sequence) == 0:
                    raise ValueError("historical_sequence is required and cannot be empty")
                
                # 1. Convert historical_sequence to numpy array
                feature_names = ['open', 'high', 'low', 'close', 'volume']
                historical_array = np.array([
                    [entry.get(fname, 0.0) for fname in feature_names]
                    for entry in historical_sequence
                ])  # Shape: (n_days, 5)
                
                # 2. Add technical indicators using stock_predictor.py create_features()
                df = pd.DataFrame(historical_array, columns=feature_names)
                if 'date' in historical_sequence[0]:
                    df['date'] = pd.to_datetime([entry['date'] for entry in historical_sequence])
                
                # Import from stock_predictor
                try:
                    from stock_predictor import create_features
                    df = create_features(df, feature_type='balanced', use_feature_engineering=True)
                except ImportError:
                    logger.warning("stock_predictor not found. Using basic features.")
                    # Fallback: basic feature calculation
                    df['returns'] = df['close'].pct_change().fillna(0)
                    df['volatility'] = df['returns'].rolling(20, min_periods=1).std().fillna(0)
                    df['sma_5'] = df['close'].rolling(5, min_periods=1).mean().fillna(df['close'])
                    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean().fillna(df['close'])
                    df['momentum'] = (df['close'] - df['close'].shift(5)).fillna(0)
                    df['rsi'] = 50.0  # Placeholder
                
                # 3. Prepare temporal features (past 60 days)
                # CRITICAL FIX: Use the EXACT same feature selection logic as training
                # During training, prepare_data_for_ml uses: features_df.drop([...], axis=1, errors='ignore')
                # We must replicate this EXACTLY to ensure consistent feature dimensions
                
                # First log what columns we have BEFORE dropping
                logger.info(f"Columns BEFORE drop (total={len(df.columns)}): {list(df.columns)}")
                
                df = df.drop(['date', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
                feature_cols = list(df.columns)
                
                # Log what columns we have AFTER dropping
                logger.info(f"Columns AFTER drop (total={len(feature_cols)}): {feature_cols}")
                
                # CRITICAL: Check if we need to match the scaler's expected number of features
                if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
                    expected_features = self.scaler.n_features_in_
                    logger.info(f"Scaler expects {expected_features} features, we have {len(feature_cols)} features")
                    
                    if len(feature_cols) != expected_features:
                        logger.error(f"FEATURE MISMATCH: Model expects {expected_features} features but got {len(feature_cols)}")
                        logger.error(f"Available features: {feature_cols}")
                        
                        # Try to fix by adding missing features or using saved feature names
                        if hasattr(self, 'feature_names') and self.feature_names:
                            logger.info(f"Attempting to align features using saved feature_names: {self.feature_names}")
                            # Create a DataFrame with all expected features, filling missing ones with 0
                            aligned_df = pd.DataFrame(0.0, index=df.index, columns=self.feature_names)
                            # Fill in the features we do have
                            for col in feature_cols:
                                if col in aligned_df.columns:
                                    aligned_df[col] = df[col]
                                else:
                                    logger.warning(f"Feature '{col}' in prediction data but not in model's expected features")
                            df = aligned_df
                            feature_cols = list(df.columns)
                            logger.info(f"Aligned features (total={len(feature_cols)}): {feature_cols}")
                
                if not feature_cols:
                    logger.warning("No feature columns found after dropping OHLCV. Using fallback basic features.")
                    # This shouldn't happen if create_features worked properly
                    feature_cols = ['returns', 'volatility', 'sma_5', 'sma_20', 'momentum', 
                                   'roc', 'atr', 'bb_width', 'rsi']
                    feature_cols = [col for col in feature_cols if col in df.columns]
                
                # Final logging
                logger.info(f"Using {len(feature_cols)} features for prediction: {feature_cols}")
                temporal_features = df[feature_cols].values  # Shape: (n_days, n_features)
                logger.info(f"Temporal features shape: {temporal_features.shape}")
                X_past = prepare_temporal_features(temporal_features, lookback=lookback)
                logger.info(f"X_past shape after prepare_temporal_features: {X_past.shape}")
                
                # Get current price from last historical entry
                current_price = historical_sequence[-1].get('close', 0.0)
            
            # 4. Create static features
            X_static = create_static_features(static_dict, static_dim=self.static_dim)
            X_static = X_static.reshape(1, -1).astype(np.float32)
            
            # 5. Make prediction with TFT model
            outputs = self.predict(X_past, X_static)
            
            # 6. Process and return results
            # NOTE: median_predictions are now properly inverse-transformed percentage changes
            # (e.g., 0.05 for 5% increase, -0.02 for 2% decrease)
            median_predictions = outputs['median_predictions'][0]
            lower_bounds = outputs['lower_bound'][0]
            upper_bounds = outputs['upper_bound'][0]
            
            # Build multi-horizon response
            horizons_data = {}

            # DIAGNOSTIC: Log current price and predictions
            logger.info(f"Building horizons - Current price: {current_price}")
            logger.info(f"Median predictions: {median_predictions}")
            logger.info(f"Lower bounds: {lower_bounds}")
            logger.info(f"Upper bounds: {upper_bounds}")

            for i, horizon in enumerate(self.forecast_horizons):
                # median_change is a percentage change (e.g., 0.05 = 5%)
                median_change = median_predictions[i]
                target_price = current_price * (1 + median_change) if current_price > 0 else 0.0
                lower_price = current_price * (1 + lower_bounds[i]) if current_price > 0 else 0.0
                upper_price = current_price * (1 + upper_bounds[i]) if current_price > 0 else 0.0

                # Calculate confidence from interval width (narrower = more confident)
                interval_width = upper_bounds[i] - lower_bounds[i]
                horizon_confidence = max(0.5, 1.0 - min(1.0, interval_width))

                # DIAGNOSTIC: Log calculated values for this horizon
                logger.info(f"Horizon {horizon}d: median_change={median_change:.4f}, target_price={target_price:.2f}, " +
                          f"lower={lower_price:.2f}, upper={upper_price:.2f}, confidence={horizon_confidence:.2f}")

                # C# HorizonPredictionData expects: MedianPrice, TargetPrice, LowerBound, UpperBound, Confidence
                horizons_data[f'{horizon}d'] = {
                    'MedianPrice': float(target_price),  # Use target_price as median
                    'TargetPrice': float(target_price),  # Same as median for consistency
                    'LowerBound': float(lower_price),    # Convert percentage to price
                    'UpperBound': float(upper_price),    # Convert percentage to price
                    'Confidence': float(horizon_confidence),
                    'Q25': float(current_price * (1 + (median_change + lower_bounds[i]) / 2) if current_price > 0 else 0.0),
                    'Q75': float(current_price * (1 + (median_change + upper_bounds[i]) / 2) if current_price > 0 else 0.0)
                }
            
            # Determine action from shortest horizon
            first_prediction = median_predictions[0]
            if abs(first_prediction) < 0.01:
                action = "HOLD"
            elif first_prediction > 0:
                action = "BUY"
            else:
                action = "SELL"
            
            # Calculate confidence and target price for primary prediction
            interval_width = upper_bounds[0] - lower_bounds[0]
            confidence = max(0.5, 1.0 - min(1.0, interval_width))
            target_price = current_price * (1 + median_predictions[0]) if current_price > 0 else 0.0
            
            # Build FeatureWeights dictionary (C# expects Dictionary<string, double>)
            feature_weights = {}
            if len(outputs['feature_importance']) > 0 and outputs['feature_importance'].shape[0] > 0 and outputs['feature_importance'].shape[1] > 0:
                # Get feature importance for first sample
                importance_values = outputs['feature_importance'][0]

                # Use feature names if available, otherwise use generic names
                if hasattr(self, 'feature_names') and self.feature_names and len(self.feature_names) == len(importance_values):
                    feature_names_list = self.feature_names
                else:
                    feature_names_list = [f'feature_{i}' for i in range(len(importance_values))]

                # Create dictionary mapping feature name to importance value
                for fname, importance_val in zip(feature_names_list, importance_values):
                    feature_weights[fname] = float(importance_val)

                logger.info(f"Generated FeatureWeights with {len(feature_weights)} features")
            else:
                logger.warning("Feature importance array is empty or invalid")

            # Build TemporalAttention dictionary (C# expects Dictionary<int, double>)
            temporal_attention = {}
            if 'attention_weights' in outputs and len(outputs['attention_weights']) > 0:
                # attention_weights is a list of arrays, take first layer's weights
                attention_array = outputs['attention_weights'][0]
                if attention_array.ndim >= 2:
                    # Take first sample's attention weights (shape: [batch, seq_len] or [batch, heads, seq_len])
                    if attention_array.ndim == 3:
                        # Average across heads
                        attention_weights_sample = np.mean(attention_array[0], axis=0)
                    else:
                        attention_weights_sample = attention_array[0]

                    # Map to time steps (negative indices for lookback: -1, -2, ..., -60)
                    seq_len = len(attention_weights_sample)
                    for i, attn_val in enumerate(attention_weights_sample):
                        time_step = -(seq_len - i)  # -60, -59, ..., -2, -1
                        temporal_attention[time_step] = float(attn_val)

                    logger.info(f"Generated TemporalAttention with {len(temporal_attention)} time steps")
                else:
                    logger.warning(f"Attention array has unexpected shape: {attention_array.shape}")
            else:
                logger.warning("No attention weights available in model outputs")

            result = {
                'symbol': historical_sequence[-1].get('symbol', 'UNKNOWN') if historical_sequence else 'UNKNOWN',
                'action': action,
                'confidence': float(confidence),
                'currentPrice': float(current_price),
                'targetPrice': float(target_price),
                'medianPrediction': float(median_predictions[0]),
                'lowerBound': float(current_price * (1 + lower_bounds[0]) if current_price > 0 else 0.0),
                'upperBound': float(current_price * (1 + upper_bounds[0]) if current_price > 0 else 0.0),
                'horizons': horizons_data,
                'featureWeights': feature_weights,  # Dictionary<string, double>
                'temporalAttention': temporal_attention,  # Dictionary<int, double>
                'modelType': 'tft',
                'uncertainty': float(interval_width),
                'potentialReturn': float((target_price - current_price) / current_price if current_price > 0 else 0.0)
            }

            # DIAGNOSTIC: Log the final result structure
            logger.info(f"Returning prediction result:")
            logger.info(f"  Action: {result['action']}, Confidence: {result['confidence']:.2%}")
            logger.info(f"  Current: ${result['currentPrice']:.2f}, Target: ${result['targetPrice']:.2f}")
            logger.info(f"  Horizons count: {len(result['horizons'])}")
            for h_key, h_val in result['horizons'].items():
                logger.info(f"    {h_key}: Target=${h_val.get('TargetPrice', 0):.2f}, " +
                          f"Lower=${h_val.get('LowerBound', 0):.2f}, " +
                          f"Upper=${h_val.get('UpperBound', 0):.2f}, " +
                          f"Confidence={h_val.get('Confidence', 0):.2f}")

            return result
            
        except Exception as e:
            logger.error(f"Error in predict_single: {e}", exc_info=True)
            fallback_price = 0.0
            if historical_sequence and len(historical_sequence) > 0:
                fallback_price = historical_sequence[-1].get('close', 0.0)
            elif features_dict:
                fallback_price = features_dict.get('current_price', 0.0)
                
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'targetPrice': fallback_price,
                'currentPrice': fallback_price,
                'error': str(e),
                'modelType': 'tft'
            }
    
    def save(self, model_path: str = TFT_MODEL_PATH, 
             scaler_path: str = TFT_SCALER_PATH) -> bool:
        """Save TFT model and scalers."""
        try:
            # CRITICAL: Ensure directory exists with proper error handling
            model_dir = os.path.dirname(model_path)
            if not model_dir:
                model_dir = os.path.dirname(os.path.abspath(__file__))  # Use script directory
                model_path = os.path.join(model_dir, 'models', 'tft_model.pt')
                scaler_path = os.path.join(model_dir, 'models', 'tft_scaler.pkl')
                model_dir = os.path.join(model_dir, 'models')
            
            # Create directory with explicit error handling
            try:
                os.makedirs(model_dir, exist_ok=True)
                logger.info(f"Created/verified models directory: {model_dir}")
            except Exception as dir_error:
                logger.error(f"Failed to create models directory {model_dir}: {dir_error}")
                # Try alternative location in temp directory
                import tempfile
                model_dir = os.path.join(tempfile.gettempdir(), 'quantra_models')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, 'tft_model.pt')
                scaler_path = os.path.join(model_dir, 'tft_scaler.pkl')
                logger.warning(f"Using alternative save location: {model_dir}")
            
            # Save model with detailed error reporting
            logger.info(f"Saving TFT model to {model_path}...")
            
            # CRITICAL FIX: Save calendar_dim to track if model was trained with future features
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'static_dim': self.static_dim,
                'hidden_dim': self.hidden_dim,
                'forecast_horizons': self.forecast_horizons,
                'feature_names': self.feature_names,
                'num_heads': self.num_heads,
                'num_lstm_layers': self.num_lstm_layers,
                'dropout': self.dropout,
                'num_attention_layers': self.num_attention_layers,
                'is_trained': self.is_trained,
                'calendar_dim': self.model.calendar_dim  # Save calendar dimension used during training
            }
            
            torch.save(checkpoint, model_path)
            logger.info(f"Model state saved successfully")
            
            # Save scalers with detailed error reporting
            logger.info(f"Saving scalers to {scaler_path}...")
            scalers_dict = {
                'scaler': self.scaler,
                'static_scaler': self.static_scaler,
                'target_scaler': self.target_scaler,
                'future_scaler': self.future_scaler if hasattr(self, 'future_scaler') else None
            }
            joblib.dump(scalers_dict, scaler_path)
            logger.info(f"Scalers saved successfully")
            
            # Verify files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found after save: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found after save: {scaler_path}")
            
            logger.info(f"TFT model and scalers saved successfully to {model_dir}")
            logger.info(f"  Model: {os.path.basename(model_path)} ({os.path.getsize(model_path)} bytes)")
            logger.info(f"  Scalers: {os.path.basename(scaler_path)} ({os.path.getsize(scaler_path)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR saving TFT model: {e}", exc_info=True)
            logger.error(f"Attempted save path: {model_path}")
            logger.error(f"Working directory: {os.getcwd()}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load(self, model_path: str = TFT_MODEL_PATH,
             scaler_path: str = TFT_SCALER_PATH) -> bool:
        """Load TFT model and scalers."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.warning(f"TFT model files not found at {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Reconstruct model with saved parameters
            self.input_dim = checkpoint['input_dim']
            self.static_dim = checkpoint['static_dim']
            self.hidden_dim = checkpoint['hidden_dim']
            self.forecast_horizons = checkpoint['forecast_horizons']
            self.feature_names = checkpoint.get('feature_names', [])
            self.num_heads = checkpoint.get('num_heads', 4)
            self.num_lstm_layers = checkpoint.get('num_lstm_layers', 2)
            self.dropout = checkpoint.get('dropout', 0.1)
            self.num_attention_layers = checkpoint.get('num_attention_layers', 2)
            self.is_trained = checkpoint.get('is_trained', True)
            calendar_dim = checkpoint.get('calendar_dim', None)  # Get saved calendar dimension
            
            # Rebuild model
            self.model = TemporalFusionTransformer(
                input_dim=self.input_dim,
                static_dim=self.static_dim,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_lstm_layers=self.num_lstm_layers,
                dropout=self.dropout,
                forecast_horizons=self.forecast_horizons,
                num_attention_layers=self.num_attention_layers
            ).to(self.device)
            
            # CRITICAL FIX: Initialize future_embedding if model was trained with calendar features
            if calendar_dim is not None:
                import torch.nn as nn
                self.model.future_embedding = nn.Linear(calendar_dim, self.hidden_dim).to(self.device)
                self.model.calendar_dim = calendar_dim
                logger.info(f"Initialized future_embedding for loading: {calendar_dim} -> {self.hidden_dim}")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load scalers
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['scaler']
            self.static_scaler = scalers['static_scaler']
            # Load target scaler with fallback to identity transformation for old models
            if 'target_scaler' in scalers:
                self.target_scaler = scalers['target_scaler']
            else:
                # For backward compatibility: use identity scaler (no transformation)
                # This means old models will still have issues, but won't crash
                logger.warning("Loading old model without target_scaler. Predictions may be inaccurate. Please retrain the model.")
                identity_scaler = StandardScaler()
                identity_scaler.mean_ = np.array([0.0])
                identity_scaler.scale_ = np.array([1.0])
                identity_scaler.n_features_in_ = 1
                self.target_scaler = identity_scaler
            
            # Load future scaler if exists
            if 'future_scaler' in scalers and scalers['future_scaler'] is not None:
                self.future_scaler = scalers['future_scaler']
            else:
                self.future_scaler = None
            
            logger.info(f"TFT model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TFT model: {e}")
            return False


def compute_multi_horizon_targets(prices: np.ndarray,
                                  forecast_horizons: List[int],
                                  return_type: str = 'percentage') -> np.ndarray:
    """
    Compute proper multi-horizon targets from sequential price data.

    For each sample at time t, computes the target return/change for each forecast horizon.

    Args:
        prices: Array of shape (n_samples + max_horizon,) with sequential prices
        forecast_horizons: List of forecast horizons (e.g., [5, 10, 20, 30])
        return_type: 'percentage' for percentage returns or 'absolute' for price differences

    Returns:
        Targets of shape (n_samples, num_horizons) where each column is the
        actual future return at that horizon

    Example:
        If prices = [100, 102, 105, 103, 108, 110, 115, ...] and horizons = [2, 5]
        For sample at t=0 (price=100):
            - horizon_2 target: (prices[2] - prices[0]) / prices[0] = (105-100)/100 = 0.05
            - horizon_5 target: (prices[5] - prices[0]) / prices[0] = (110-100)/100 = 0.10
    """
    max_horizon = max(forecast_horizons)
    n_samples = len(prices) - max_horizon

    if n_samples <= 0:
        raise ValueError(f"Not enough data: need at least {max_horizon} extra samples "
                        f"beyond training data to compute targets for max horizon {max_horizon}")

    targets = np.zeros((n_samples, len(forecast_horizons)), dtype=np.float32)

    for i in range(n_samples):
        current_price = prices[i]
        if current_price == 0:
            logger.warning(f"Zero price at index {i}, using 0 target")
            continue

        for j, horizon in enumerate(forecast_horizons):
            future_price = prices[i + horizon]

            if return_type == 'percentage':
                # Percentage return: (future - current) / current
                targets[i, j] = (future_price - current_price) / current_price
            elif return_type == 'absolute':
                # Absolute difference
                targets[i, j] = future_price - current_price
            else:
                raise ValueError(f"Unknown return_type: {return_type}")

    return targets


def create_static_features(symbol_info: Optional[Dict[str, Any]] = None,
                          static_dim: int = DEFAULT_STATIC_DIM) -> np.ndarray:
    """
    Create static feature vector from symbol information.
    
    Args:
        symbol_info: Dictionary with symbol metadata
        static_dim: Dimension of static feature vector
        
    Returns:
        Static feature array of shape (static_dim,)
    """
    static_features = np.zeros(static_dim, dtype=np.float32)
    
    if symbol_info is None:
        return static_features
    
    # Encode sector
    sector_map = {
        'technology': 1.0, 'healthcare': 2.0, 'financial': 3.0,
        'consumer': 4.0, 'industrial': 5.0, 'energy': 6.0,
        'materials': 7.0, 'utilities': 8.0, 'real_estate': 9.0,
        'communication': 10.0
    }
    
    sector = symbol_info.get('sector', '').lower()
    static_features[0] = sector_map.get(sector, 0.0) / 10.0  # Normalize
    
    # Market cap category
    market_cap_map = {
        'mega': 5.0, 'large': 4.0, 'mid': 3.0, 'small': 2.0, 'micro': 1.0
    }
    
    market_cap = symbol_info.get('market_cap_category', '').lower()
    static_features[1] = market_cap_map.get(market_cap, 2.5) / 5.0  # Normalize
    
    # Additional features
    if 'beta' in symbol_info:
        static_features[2] = min(3.0, max(-1.0, symbol_info['beta'])) / 3.0
    
    if 'avg_volume' in symbol_info:
        static_features[3] = np.log1p(symbol_info['avg_volume']) / 20.0
    
    if 'pe_ratio' in symbol_info:
        pe = symbol_info['pe_ratio']
        static_features[4] = np.clip(pe / 100.0, -1.0, 1.0) if pe is not None else 0.0
    
    return static_features


def prepare_temporal_features(historical_data: np.ndarray,
                             lookback: int = DEFAULT_LOOKBACK) -> np.ndarray:
    """
    Prepare temporal features from historical data.
    
    CRITICAL FIX: Uses zero padding instead of repeated values to avoid bias.
    
    Args:
        historical_data: Array of shape (n_days, n_features) with historical data
        lookback: Number of days to use for lookback window
        
    Returns:
        Temporal features of shape (1, lookback, n_features)
    """
    if len(historical_data) < lookback:
        # Pad with zeros to avoid introducing bias from repeated values
        padding_size = lookback - len(historical_data)
        padding = np.zeros((padding_size, historical_data.shape[1]), dtype=historical_data.dtype)
        historical_data = np.vstack([padding, historical_data])
        logger.warning(f"Padded {padding_size} zero values. Provide at least {lookback} days of data for best accuracy.")
    
    # Take the last lookback days
    temporal_features = historical_data[-lookback:].astype(np.float32)
    
    # Add batch dimension
    return temporal_features.reshape(1, lookback, -1)
