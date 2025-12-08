#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration layer between TFT and existing stock_predictor.py infrastructure.
Provides TFTStockPredictor wrapper compatible with existing prediction interface.

CRITICAL FIX: This module now properly accepts real historical sequences instead
of synthetic repeated values for TFT predictions.
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
            y: (n_samples, num_horizons) or (n_samples,) - Target values
            future_features: (n_samples, forecast_horizon, future_features) - Known future features (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            lr: Learning rate
            verbose: Whether to print training progress
            
        Returns:
            Training history with train_loss and val_loss
        """
        # Initialize scalers
        self.scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # NEW: Scaler for target values
        
        # Scale temporal features
        n_samples, seq_len, n_features = X_past.shape
        X_past_reshaped = X_past.reshape(-1, n_features)
        X_past_scaled = self.scaler.fit_transform(X_past_reshaped)
        X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
        
        # Scale static features
        X_static_scaled = self.static_scaler.fit_transform(X_static)
        
        # Handle future features if provided
        has_future = future_features is not None and future_features.size > 0
        if has_future:
            # Initialize future scaler
            self.future_scaler = StandardScaler()
            
            # Scale future features
            n_samples_future, future_seq_len, n_future_features = future_features.shape
            future_reshaped = future_features.reshape(-1, n_future_features)
            future_scaled = self.future_scaler.fit_transform(future_reshaped)
            future_scaled = future_scaled.reshape(n_samples_future, future_seq_len, n_future_features)
            logger.info(f"Scaled future features: {future_scaled.shape}")
        else:
            future_scaled = None
            self.future_scaler = None  # Ensure consistency
            logger.info("No future features provided for training")
        
        # Ensure targets have correct shape
        if y.ndim == 1:
            y = np.column_stack([y] * len(self.forecast_horizons))
        
        # NEW: Fit and transform targets (percentage changes)
        # Targets are percentage changes (e.g., 0.05 for 5%), need to be scaled
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Create tensors
        past_tensor = torch.FloatTensor(X_past_scaled)
        static_tensor = torch.FloatTensor(X_static_scaled)
        # Use scaled targets: y_scaled contains percentage changes (e.g., 0.05 for 5%)
        # transformed by StandardScaler for improved training stability and convergence
        targets_tensor = torch.FloatTensor(y_scaled)
        
        # Create DataLoader with or without future features
        if has_future:
            future_tensor = torch.FloatTensor(future_scaled)
            dataset = TensorDataset(past_tensor, static_tensor, future_tensor, targets_tensor)
        else:
            dataset = TensorDataset(past_tensor, static_tensor, targets_tensor)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Generator function to yield dict batches
        def dict_batch_generator(loader, include_future=False):
            for batch_data in loader:
                if include_future:
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
        
        train_loader_dict = list(dict_batch_generator(train_loader, include_future=has_future))
        val_loader_dict = list(dict_batch_generator(val_loader, include_future=has_future))
        
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
    
    def predict(self, X_past: np.ndarray, X_static: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.
        
        Args:
            X_past: (n_samples, seq_len, features) - Historical temporal features
            X_static: (n_samples, static_features) - Static features
            
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
        
        # CRITICAL: Use inference_mode for complete gradient disabling
        with torch.inference_mode():
            # Scale features
            n_samples, seq_len, n_features = X_past.shape
            X_past_reshaped = X_past.reshape(-1, n_features)
            X_past_scaled = self.scaler.transform(X_past_reshaped)
            X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
            
            X_static_scaled = self.static_scaler.transform(X_static)
            
            # Convert to tensors
            past_tensor = torch.FloatTensor(X_past_scaled).to(self.device)
            static_tensor = torch.FloatTensor(X_static_scaled).to(self.device)
            
            outputs = self.model(past_tensor, static_tensor)
        
        # Extract predictions for each horizon
        median_predictions = []
        lower_bounds = []
        upper_bounds = []
        q25_list = []
        q75_list = []
        
        for horizon in self.forecast_horizons:
            horizon_key = f"horizon_{horizon}"
            quantiles = outputs['predictions'][horizon_key].cpu().numpy()
            
            # NEW: Inverse transform scaled predictions back to percentage changes
            # The model outputs scaled values, we need to convert back to actual percentage changes
            median_predictions.append(quantiles[:, 2])  # 50th percentile
            lower_bounds.append(quantiles[:, 0])        # 10th percentile
            upper_bounds.append(quantiles[:, 4])        # 90th percentile
            q25_list.append(quantiles[:, 1])            # 25th percentile
            q75_list.append(quantiles[:, 3])            # 75th percentile
        
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
        
        return {
            'median_predictions': median_unscaled,
            'lower_bound': lower_unscaled,
            'upper_bound': upper_unscaled,
            'q25': q25_unscaled,
            'q75': q75_unscaled,
            'feature_importance': outputs['variable_importance'].cpu().numpy(),
            'attention_weights': [w.cpu().numpy() for w in outputs['attention_weights']]
        }
    
    def predict_single(self, 
                      historical_sequence: Optional[List[Dict[str, float]]] = None,
                      calendar_features: Optional[List[Dict[str, int]]] = None,
                      static_dict: Optional[Dict[str, Any]] = None,
                      features_dict: Optional[Dict[str, float]] = None,
                      lookback: int = 60) -> Dict[str, Any]:
        """
        Predict for a single symbol using REAL historical data.
        
        CRITICAL FIX: This method now accepts actual historical sequences
        instead of creating synthetic repeated values.
        
        Args:
            historical_sequence: List of dicts with OHLCV data for last 60 days
                                [{'date': '2024-01-01', 'close': 98.0, 'volume': 1M, ...}, ...]
            calendar_features: List of calendar feature dicts for historical + future period
                              [{'dayofweek': 0, 'month': 1, 'is_friday': 0, ...}, ...]
            static_dict: Dictionary of static features (sector, market_cap, etc.)
            features_dict: Legacy parameter for backward compatibility (single feature dict)
            lookback: Number of lookback periods (default 60)
            
        Returns:
            Prediction result with multi-horizon forecasts and uncertainty
        """
        try:
            # Handle backward compatibility: if features_dict is provided but not historical_sequence
            if historical_sequence is None and features_dict is not None:
                logger.warning("Using legacy features_dict interface. For best results, use historical_sequence.")
                # Create synthetic sequence from single feature dict (legacy behavior)
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
                FEATURE_COLUMNS = ['returns', 'volatility', 'sma_5', 'sma_20', 'momentum', 
                                   'roc', 'atr', 'bb_width', 'rsi']  # Add more as available
                
                # Filter to only available columns
                available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
                if not available_features:
                    logger.warning("No feature columns found. Using OHLCV directly.")
                    available_features = feature_names
                
                temporal_features = df[available_features].values  # Shape: (n_days, n_features)
                X_past = prepare_temporal_features(temporal_features, lookback=lookback)
                
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
            for i, horizon in enumerate(self.forecast_horizons):
                # median_change is a percentage change (e.g., 0.05 = 5%)
                median_change = median_predictions[i]
                target_price = current_price * (1 + median_change) if current_price > 0 else 0.0
                
                horizons_data[f'{horizon}d'] = {
                    'median_price': float(target_price),
                    'lower_bound': float(current_price * (1 + lower_bounds[i]) if current_price > 0 else 0.0),
                    'upper_bound': float(current_price * (1 + upper_bounds[i]) if current_price > 0 else 0.0),
                    'confidence': float(1.0 - (upper_bounds[i] - lower_bounds[i]))
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
            
            return {
                'symbol': historical_sequence[-1].get('symbol', 'UNKNOWN') if historical_sequence else 'UNKNOWN',
                'action': action,
                'confidence': float(confidence),
                'currentPrice': float(current_price),
                'targetPrice': float(target_price),
                'medianPrediction': float(median_predictions[0]),
                'lowerBound': float(current_price * (1 + lower_bounds[0]) if current_price > 0 else 0.0),
                'upperBound': float(current_price * (1 + upper_bounds[0]) if current_price > 0 else 0.0),
                'horizons': horizons_data,
                'modelType': 'tft',
                'uncertainty': float(interval_width),
                'featureImportance': outputs['feature_importance'].tolist()
            }
            
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
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            torch.save({
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
                'is_trained': self.is_trained
            }, model_path)
            
            scalers_dict = {
                'scaler': self.scaler,
                'static_scaler': self.static_scaler,
                'target_scaler': self.target_scaler,
                'future_scaler': self.future_scaler if hasattr(self, 'future_scaler') else None
            }
            joblib.dump(scalers_dict, scaler_path)
            
            logger.info(f"TFT model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving TFT model: {e}")
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
