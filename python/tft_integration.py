#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integration layer between TFT and existing stock_predictor.py infrastructure.
Provides TFTStockPredictor wrapper compatible with existing prediction interface.
"""

import torch
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import joblib

# Import TFT components
from temporal_fusion_transformer import (
    TemporalFusionTransformer, 
    train_tft_model, 
    QuantileLoss
)

logger = logging.getLogger('tft_integration')

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
TFT_MODEL_PATH = os.path.join(MODEL_DIR, 'tft_model.pt')
TFT_SCALER_PATH = os.path.join(MODEL_DIR, 'tft_scaler.pkl')


class TFTStockPredictor:
    """
    Wrapper for TFT model compatible with existing predictor interface.
    Provides multi-horizon forecasting with uncertainty quantification.
    """
    
    def __init__(self,
                 input_dim: int = 50,
                 static_dim: int = 10,
                 hidden_dim: int = 128,
                 forecast_horizons: Optional[List[int]] = None,
                 num_heads: int = 4,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.1,
                 num_attention_layers: int = 2):
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
        self.feature_names = []
        self.is_trained = False
        
        logger.info(f"Initialized TFTStockPredictor on {self.device}")
        
    def fit(self, X_past: np.ndarray, X_static: np.ndarray, y: np.ndarray,
            epochs: int = 50, batch_size: int = 32, lr: float = 0.001,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the TFT model.
        
        Args:
            X_past: (n_samples, seq_len, features) - Historical temporal features
            X_static: (n_samples, static_features) - Static features (sector, market cap, etc.)
            y: (n_samples, num_horizons) or (n_samples,) - Target values for each horizon
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
        
        # Scale temporal features
        n_samples, seq_len, n_features = X_past.shape
        X_past_reshaped = X_past.reshape(-1, n_features)
        X_past_scaled = self.scaler.fit_transform(X_past_reshaped)
        X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
        
        # Scale static features
        X_static_scaled = self.static_scaler.fit_transform(X_static)
        
        # Ensure targets have correct shape
        if y.ndim == 1:
            # Repeat single target for all horizons
            y = np.column_stack([y] * len(self.forecast_horizons))
        
        # Create tensors
        past_tensor = torch.FloatTensor(X_past_scaled)
        static_tensor = torch.FloatTensor(X_static_scaled)
        targets_tensor = torch.FloatTensor(y)
        
        # Create DataLoader
        dataset = TensorDataset(past_tensor, static_tensor, targets_tensor)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Generator function to yield dict batches without loading all into memory
        def dict_batch_generator(loader):
            """Yield batches as dicts to avoid loading all data into memory at once."""
            for past, static, targets in loader:
                yield {
                    'past_features': past,
                    'static_features': static,
                    'targets': targets
                }
        
        # For training, we need to iterate multiple times (one per epoch)
        # The train_tft_model expects an iterable per epoch, so we create fresh generators
        # For simplicity with the current train_tft_model interface, we use list for small datasets
        # For large datasets, modify train_tft_model to accept DataLoader directly
        train_loader_dict = list(dict_batch_generator(train_loader))
        val_loader_dict = list(dict_batch_generator(val_loader))
        
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
        self.model.eval()
        
        # Scale features
        n_samples, seq_len, n_features = X_past.shape
        X_past_reshaped = X_past.reshape(-1, n_features)
        X_past_scaled = self.scaler.transform(X_past_reshaped)
        X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
        
        X_static_scaled = self.static_scaler.transform(X_static)
        
        # Convert to tensors
        past_tensor = torch.FloatTensor(X_past_scaled).to(self.device)
        static_tensor = torch.FloatTensor(X_static_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(past_tensor, static_tensor)
        
        # Extract predictions for each horizon
        median_predictions = []
        lower_bounds = []
        upper_bounds = []
        q25_list = []
        q75_list = []
        
        for horizon in self.forecast_horizons:
            horizon_key = f"horizon_{horizon}"
            quantiles = outputs['predictions'][horizon_key].cpu().numpy()  # (n_samples, 5)
            
            median_predictions.append(quantiles[:, 2])  # 50th percentile (median)
            lower_bounds.append(quantiles[:, 0])        # 10th percentile
            upper_bounds.append(quantiles[:, 4])        # 90th percentile
            q25_list.append(quantiles[:, 1])            # 25th percentile
            q75_list.append(quantiles[:, 3])            # 75th percentile
        
        return {
            'median_predictions': np.column_stack(median_predictions),
            'lower_bound': np.column_stack(lower_bounds),
            'upper_bound': np.column_stack(upper_bounds),
            'q25': np.column_stack(q25_list),
            'q75': np.column_stack(q75_list),
            'feature_importance': outputs['variable_importance'].cpu().numpy(),
            'attention_weights': [w.cpu().numpy() for w in outputs['attention_weights']]
        }
    
    def predict_single(self, features_dict: Dict[str, float], 
                      static_dict: Optional[Dict[str, Any]] = None,
                      lookback: int = 60) -> Dict[str, Any]:
        """
        Predict for a single symbol (interface compatible with stock_predictor.py).
        
        Args:
            features_dict: Dictionary of temporal features (most recent values)
            static_dict: Dictionary of static features (sector, market_cap_category, etc.)
            lookback: Number of lookback periods (default 60)
            
        Returns:
            Prediction result compatible with existing PredictionModel
        """
        # Extract feature values and create temporal sequence
        feature_names = list(features_dict.keys())
        feature_values = np.array([features_dict[k] for k in feature_names])
        
        # Create a synthetic temporal sequence (repeat values for lookback)
        # In production, you would pass actual historical data
        n_features = len(feature_values)
        # Reshape to (1, 1, n_features) then tile to (1, lookback, n_features)
        X_past = np.tile(feature_values.reshape(1, 1, n_features), (1, lookback, 1)).astype(np.float32)
        
        # Create static features
        if static_dict is None:
            static_dict = {}
        
        # Default static features
        static_features = np.zeros((1, self.static_dim), dtype=np.float32)
        
        # Map common static features
        static_mapping = {
            'sector': 0,
            'market_cap_category': 1,
            'volatility_regime': 2,
            'volume_regime': 3,
            'trend_regime': 4
        }
        
        for key, idx in static_mapping.items():
            if key in static_dict and idx < self.static_dim:
                static_features[0, idx] = float(static_dict[key])
        
        # Make prediction
        try:
            outputs = self.predict(X_past, static_features)
            
            # Extract median prediction for first horizon
            median_pred = outputs['median_predictions'][0, 0]  # First sample, first horizon
            lower = outputs['lower_bound'][0, 0]
            upper = outputs['upper_bound'][0, 0]
            
            # Get current price from features
            current_price = features_dict.get('current_price', 
                           features_dict.get('close', 
                           features_dict.get('price', 0.0)))
            
            # Calculate target price
            # Prediction is typically percentage change
            target_price = current_price * (1 + median_pred) if current_price > 0 else median_pred
            
            # Calculate confidence from prediction interval width
            interval_width = upper - lower
            confidence = max(0.5, 1.0 - min(1.0, interval_width))
            
            # Determine action
            if abs(median_pred) < 0.01:  # Less than 1% change
                action = "HOLD"
            elif median_pred > 0:
                action = "BUY"
            else:
                action = "SELL"
            
            # Build multi-horizon predictions
            horizons_data = {}
            for i, horizon in enumerate(self.forecast_horizons):
                horizons_data[f'{horizon}d'] = {
                    'median': float(outputs['median_predictions'][0, i]),
                    'lower': float(outputs['lower_bound'][0, i]),
                    'upper': float(outputs['upper_bound'][0, i]),
                    'target_price': float(current_price * (1 + outputs['median_predictions'][0, i])) if current_price > 0 else float(outputs['median_predictions'][0, i])
                }
            
            return {
                'action': action,
                'confidence': float(confidence),
                'targetPrice': float(target_price),
                'currentPrice': float(current_price),
                'medianPrediction': float(median_pred),
                'lowerBound': float(lower),
                'upperBound': float(upper),
                'horizons': horizons_data,
                'featureImportance': outputs['feature_importance'].tolist(),
                'modelType': 'tft',
                'uncertainty': float(interval_width)
            }
            
        except Exception as e:
            logger.error(f"Error in predict_single: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'targetPrice': features_dict.get('current_price', 0.0),
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
            
            joblib.dump({
                'scaler': self.scaler,
                'static_scaler': self.static_scaler
            }, scaler_path)
            
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
            
            logger.info(f"TFT model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TFT model: {e}")
            return False
    
    def feature_importance(self, X_past: np.ndarray, X_static: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using gradient-based sensitivity analysis.
        
        Args:
            X_past: Temporal features
            X_static: Static features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        self.model.eval()
        
        # Scale features
        n_samples, seq_len, n_features = X_past.shape
        X_past_reshaped = X_past.reshape(-1, n_features)
        X_past_scaled = self.scaler.transform(X_past_reshaped)
        X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
        
        X_static_scaled = self.static_scaler.transform(X_static)
        
        # Convert to tensors with gradient tracking
        past_tensor = torch.FloatTensor(X_past_scaled).to(self.device)
        past_tensor.requires_grad = True
        static_tensor = torch.FloatTensor(X_static_scaled).to(self.device)
        
        # Forward pass
        outputs = self.model(past_tensor, static_tensor)
        
        # Get median prediction from first horizon
        horizon_key = f"horizon_{self.forecast_horizons[0]}"
        median_pred = outputs['predictions'][horizon_key][:, 2]  # Median (50th percentile)
        
        # Compute gradients
        median_pred.sum().backward()
        
        # Get gradient magnitudes
        gradients = past_tensor.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        
        # Normalize
        importance = gradients / (gradients.sum() + 1e-10)
        
        # Map to feature names
        if len(self.feature_names) == len(importance):
            return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}


def create_static_features(symbol_info: Optional[Dict[str, Any]] = None,
                          static_dim: int = 10) -> np.ndarray:
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
    
    # Encode sector (one-hot or numerical)
    sector_map = {
        'technology': 1.0,
        'healthcare': 2.0,
        'financial': 3.0,
        'consumer': 4.0,
        'industrial': 5.0,
        'energy': 6.0,
        'materials': 7.0,
        'utilities': 8.0,
        'real_estate': 9.0,
        'communication': 10.0
    }
    
    sector = symbol_info.get('sector', '').lower()
    static_features[0] = sector_map.get(sector, 0.0) / 10.0  # Normalize
    
    # Market cap category
    market_cap_map = {
        'mega': 5.0,
        'large': 4.0,
        'mid': 3.0,
        'small': 2.0,
        'micro': 1.0
    }
    
    market_cap = symbol_info.get('market_cap_category', '').lower()
    static_features[1] = market_cap_map.get(market_cap, 2.5) / 5.0  # Normalize
    
    # Other static features can be added here
    # Volatility regime, beta, average volume, etc.
    
    if 'beta' in symbol_info:
        static_features[2] = min(3.0, max(-1.0, symbol_info['beta'])) / 3.0
    
    if 'avg_volume' in symbol_info:
        static_features[3] = np.log1p(symbol_info['avg_volume']) / 20.0  # Log-scale normalize
    
    if 'pe_ratio' in symbol_info:
        pe = symbol_info['pe_ratio']
        static_features[4] = np.clip(pe / 100.0, -1.0, 1.0) if pe is not None else 0.0
    
    return static_features


def prepare_temporal_features(historical_data: np.ndarray,
                             lookback: int = 60) -> np.ndarray:
    """
    Prepare temporal features from historical data.
    
    Args:
        historical_data: Array of shape (n_days, n_features) with historical data
        lookback: Number of days to use for lookback window
        
    Returns:
        Temporal features of shape (1, lookback, n_features)
        
    Note:
        If historical_data has fewer than lookback days, zero padding is used
        to avoid introducing bias. In production, ensure sufficient historical
        data is available for accurate predictions.
    """
    if len(historical_data) < lookback:
        # Pad with zeros to avoid introducing bias from repeated values
        # In production, consider requiring sufficient historical data
        padding_size = lookback - len(historical_data)
        padding = np.zeros((padding_size, historical_data.shape[1]), dtype=historical_data.dtype)
        historical_data = np.vstack([padding, historical_data])
        logger.warning(f"Insufficient historical data ({len(historical_data) - padding_size} days). "
                      f"Padded with {padding_size} zero values. Consider providing at least {lookback} days of data.")
    
    # Take the last lookback days
    temporal_features = historical_data[-lookback:].astype(np.float32)
    
    # Add batch dimension
    return temporal_features.reshape(1, lookback, -1)
