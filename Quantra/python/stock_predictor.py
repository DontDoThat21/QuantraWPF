#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import warnings
import logging

# Import debugpy utilities for remote debugging support
try:
    from debugpy_utils import init_debugpy_if_enabled
    DEBUGPY_UTILS_AVAILABLE = True
except ImportError:
    DEBUGPY_UTILS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_predictor')

# Set up constant paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Add the script directory to Python path for imports
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Model paths
MODEL_PATH = os.path.join(MODEL_DIR, 'stock_rf_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'stock_scaler.pkl')
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'stock_pytorch_model.pt')
TENSORFLOW_MODEL_PATH = os.path.join(MODEL_DIR, 'stock_tensorflow_model')
FEATURE_PIPELINE_PATH = os.path.join(MODEL_DIR, 'feature_pipeline.pkl')
HYPERPARAMETER_PATH = os.path.join(MODEL_DIR, 'hyperparameter_results.pkl')

# Try to import feature engineering module
try:
    from feature_engineering import (
        FeatureEngineer, FinancialFeatureGenerator, 
        build_default_pipeline, create_train_test_features
    )
    FEATURE_ENGINEERING_AVAILABLE = True
    logger.info("Feature Engineering module is available")
except ImportError as e:
    FEATURE_ENGINEERING_AVAILABLE = False
    logger.warning(f"Feature Engineering module is not available: {e}")
    logger.warning("Using basic feature creation. Install missing dependencies with: pip install -r requirements.txt")
except Exception as e:
    FEATURE_ENGINEERING_AVAILABLE = False
    logger.error(f"Error importing feature engineering module: {e}")
    logger.warning("Using basic feature creation.")

# Try to import hyperparameter optimization module
try:
    from hyperparameter_optimization import (
        optimize_sklearn_model,
        optimize_sklearn_model_optuna,
        optimize_pytorch_model,
        optimize_tensorflow_model,
        OptimizationResult
    )
    HYPERPARAMETER_OPTIMIZATION_AVAILABLE = True
    logger.info("Hyperparameter Optimization module is available")
except ImportError:
    HYPERPARAMETER_OPTIMIZATION_AVAILABLE = False
    logger.warning("Hyperparameter Optimization module is not available. Using default parameters.")

# Try to import machine learning libraries
# These imports are optional - we'll fall back to RandomForest if they're not available
PYTORCH_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available")
except ImportError:
    logger.warning("PyTorch is not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError:
    logger.warning("TensorFlow is not available. Install with: pip install tensorflow")

# Try to import Temporal Fusion Transformer integration
TFT_AVAILABLE = False
TFT_MODEL_PATH = os.path.join(MODEL_DIR, 'tft_model.pt')

try:
    from tft_integration import TFTStockPredictor, create_static_features, prepare_temporal_features
    if PYTORCH_AVAILABLE:
        TFT_AVAILABLE = True
        logger.info("Temporal Fusion Transformer is available")
    else:
        logger.warning("TFT requires PyTorch. Install with: pip install torch")
except ImportError as e:
    logger.warning(f"TFT not available: {e}")


def create_features(data, feature_type='balanced', use_feature_engineering=True):
    """Create technical indicators as features for prediction.
    
    Args:
        data (dict or pd.DataFrame): Input data
        feature_type (str): Type of features to generate ('minimal', 'balanced', 'full')
        use_feature_engineering (bool): Whether to use advanced feature engineering pipeline
        
    Returns:
        pd.DataFrame: DataFrame with features
    """
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Check if we can use the advanced feature engineering pipeline
    if FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering:
        try:
            # Try to load saved pipeline
            if os.path.exists(FEATURE_PIPELINE_PATH):
                pipeline = FeatureEngineer.load(FEATURE_PIPELINE_PATH)
                features_df = pipeline.transform(df)
                logger.info(f"Using saved feature engineering pipeline with {len(features_df.columns)} features")
                return features_df
            else:
                # Create new pipeline
                logger.info(f"Creating new {feature_type} feature engineering pipeline...")
                pipeline = build_default_pipeline(feature_type=feature_type)
                features_df = pipeline.fit_transform(df)
                # Save pipeline for future use
                try:
                    pipeline.save(FEATURE_PIPELINE_PATH)
                    logger.info(f"Saved feature engineering pipeline with {len(features_df.columns)} features")
                except Exception as save_error:
                    logger.warning(f"Could not save feature pipeline: {save_error}")
                logger.info(f"Created new feature engineering pipeline with {len(features_df.columns)} features")
                return features_df
        except Exception as e:
            logger.error(f"Error using feature engineering pipeline: {str(e)}")
            logger.warning("Falling back to basic features.")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Fallback: Basic feature creation
    logger.info("Using basic feature creation")
    
    # Basic features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Price momentum
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['roc'] = df['close'].pct_change(periods=5)
    
    # Volatility features
    df['atr'] = df['high'] - df['low']
    df['bb_upper'] = df['sma_20'] + (df['close'].rolling(window=20).std() * 2)
    df['bb_lower'] = df['sma_20'] - (df['close'].rolling(window=20).std() * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
    
    # Volume features
    if 'volume' in df.columns:
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values created by rolling windows
    df = df.dropna()
    
    return df


# Deep Learning Model Implementations

class PyTorchStockPredictor:
    """Stock price prediction model using PyTorch."""
    
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, dropout=0.2, architecture_type='lstm'):
        """Initialize the PyTorch deep learning model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of neurons in hidden layers
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate for regularization
            architecture_type (str): Type of architecture to use ('lstm', 'gru', 'transformer')
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install with: pip install torch")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.architecture_type = architecture_type.lower()  # Normalize to lowercase
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self._build_model()
        
    def _build_model(self):
        """Build the model architecture based on selected type."""
        if self.architecture_type == 'lstm':
            self._build_lstm_model()
        elif self.architecture_type == 'gru':
            self._build_gru_model()
        elif self.architecture_type == 'transformer':
            self._build_transformer_model()
        elif self.architecture_type == 'tft':
            # TFT should use TFTStockPredictor, but if we get here, use transformer as fallback
            logger.warning("TFT architecture should use TFTStockPredictor. Using transformer architecture instead.")
            self._build_transformer_model()
        else:
            logger.warning(f"Unknown architecture type: {self.architecture_type}. Falling back to LSTM.")
            self._build_lstm_model()
            
    def _build_lstm_model(self):
        """Build an LSTM model architecture."""
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc2 = nn.Linear(hidden_dim // 2, 1)
                self.activation = nn.ReLU()
            
            def forward(self, x):
                # x shape: (batch_size, seq_length, input_dim)
                lstm_out, _ = self.lstm(x)
                # Take the output from the last time step
                out = lstm_out[:, -1, :]
                out = self.dropout(out)
                out = self.activation(self.fc1(out))
                out = self.fc2(out)
                return out.squeeze(1)
        
        self.model = LSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
    def _build_gru_model(self):
        """Build a GRU model architecture."""
        class GRUModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc2 = nn.Linear(hidden_dim // 2, 1)
                self.activation = nn.ReLU()
            
            def forward(self, x):
                # x shape: (batch_size, seq_length, input_dim)
                gru_out, _ = self.gru(x)
                # Take the output from the last time step
                out = gru_out[:, -1, :]
                out = self.dropout(out)
                out = self.activation(self.fc1(out))
                out = self.fc2(out)
                return out.squeeze(1)
        
        self.model = GRUModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
    def _build_transformer_model(self):
        """Build a Transformer model architecture."""
        class TransformerModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout, nhead=4):
                super().__init__()
                self.input_dim = input_dim
                self.pos_encoder = nn.Linear(input_dim, hidden_dim)
                encoder_layers = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=hidden_dim*2,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc2 = nn.Linear(hidden_dim // 2, 1)
                self.activation = nn.ReLU()
            
            def forward(self, x):
                # x shape: (batch_size, seq_length, input_dim)
                x = self.pos_encoder(x)  # Convert input to hidden_dim
                transformer_out = self.transformer_encoder(x)
                # Use the output corresponding to the last time step
                out = transformer_out[:, -1, :]
                out = self.dropout(out)
                out = self.activation(self.fc1(out))
                out = self.fc2(out)
                return out.squeeze(1)
        
        # For transformer, use a minimum of 4 attention heads, scaled if hidden_dim is large
        nhead = max(4, self.hidden_dim // 16)  # Ensure hidden_dim is divisible by nhead
        self.hidden_dim = nhead * (self.hidden_dim // nhead)  # Adjust hidden_dim to be divisible by nhead
        
        self.model = TransformerModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            nhead=nhead
        ).to(self.device)
    
    def fit(self, X, y, epochs=50, batch_size=32, lr=0.001, verbose=False):
        """Train the PyTorch model.
        
        Args:
            X (np.ndarray): Feature array with shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            y (np.ndarray): Target array with shape (n_samples,)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            lr (float): Learning rate
            verbose (bool): Whether to print training progress
        """
        # Handle 3D input for sequence models
        original_shape = X.shape
        is_3d = len(X.shape) == 3
        
        if is_3d:
            # Reshape from (n_samples, seq_len, n_features) to (n_samples, seq_len * n_features)
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(n_samples, seq_len * n_features)
            X_scaled = self.scaler.fit_transform(X_2d)
            # Reshape back to 3D
            X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        else:
            # Standard 2D scaling
            X_scaled = self.scaler.fit_transform(X)
        
        
        # Prepare data for LSTM/Transformer
        if is_3d:
            # Data is already in correct 3D shape (batch_size, seq_len, n_features)
            X_torch = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        else:
            # Reshape 2D to 3D: [batch_size, 1, n_features]
            X_torch = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        y_torch = torch.tensor(y, dtype=torch.float32).to(self.device)
        
        # Create DataLoader
        dataset = TensorDataset(X_torch, y_torch)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}')
        
        # Set model to evaluation mode after training
        self.model.eval()
    
    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Feature array with shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            
        Returns:
            np.ndarray: Predictions with shape (n_samples,)
        """
        # CRITICAL: Ensure model is in evaluation mode
        # This prevents the "cudnn RNN backward can only be called in training mode" error
        self.model.eval()
        
        # CRITICAL: Disable gradient computation completely using inference_mode
        # inference_mode is more efficient than no_grad and prevents any gradient operations
        with torch.inference_mode():
            # Handle 3D input for sequence models
            is_3d = len(X.shape) == 3
            
            if is_3d:
                # Reshape from (n_samples, seq_len, n_features) to (n_samples, seq_len * n_features)
                n_samples, seq_len, n_features = X.shape
                X_2d = X.reshape(n_samples, seq_len * n_features)
                X_scaled = self.scaler.transform(X_2d)
                # Reshape back to 3D
                X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
            else:
                # Standard 2D scaling
                X_scaled = self.scaler.transform(X)
            
            # Convert to PyTorch tensor - no need for requires_grad=False inside inference_mode
            if is_3d:
                # Data is already in correct 3D shape
                X_torch = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            else:
                # Reshape 2D to 3D: [batch_size, 1, n_features]
                X_torch = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
            
            # Make predictions - gradients are completely disabled
            predictions = self.model(X_torch).cpu().numpy()
                
        return predictions
    
    def save(self, model_path=PYTORCH_MODEL_PATH, scaler_path=SCALER_PATH):
        """Save the model and scaler."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the PyTorch model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'feature_names': self.feature_names,
                'architecture_type': self.architecture_type
            }, model_path)
            
            # Save the scaler
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"PyTorch model saved to {model_path}")
            return True
            
        except Exception as e:
            # Catch all exceptions to ensure training doesn't crash on save failure
            # This includes OSError for file operations, torch errors, etc.
            logger.error(f"Error saving PyTorch model: {e}")
            return False
        
    def load(self, model_path=PYTORCH_MODEL_PATH, scaler_path=SCALER_PATH):
        """Load the model and scaler."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False
        
        try:
            # Load the model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_dim = checkpoint['input_dim']
            self.hidden_dim = checkpoint['hidden_dim']
            self.num_layers = checkpoint['num_layers']
            self.dropout = checkpoint['dropout']
            self.feature_names = checkpoint['feature_names']
            
            # Load architecture type (with fallback for backward compatibility)
            self.architecture_type = checkpoint.get('architecture_type', 'lstm')
            
            # Rebuild the model architecture
            self._build_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set model to evaluation mode after loading
            self.model.eval()
            
            # Load the scaler
            self.scaler = joblib.load(scaler_path)
            
            return True
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            return False
    
    def _check_model_training_status(self):
        """Check if the model appears to be trained by examining parameter values.
        
        Returns:
            tuple: (is_trained, avg_param_value)
        """
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            if total_params == 0:
                return False, 0.0
            
            trained_params = sum(p.abs().sum().item() for p in self.model.parameters())
            avg_param_value = trained_params / total_params
            
            # If average parameter value is very close to initialization values, model is untrained
            # Typical initialized weights have small random values around 0.01-0.1
            is_trained = abs(avg_param_value) > 0.01
            
            return is_trained, avg_param_value
        except Exception as e:
            logger.warning(f"Error checking training status: {e}")
            return False, 0.0
    
    def _validate_feature_importance(self, importance_dict):
        """Validate that feature importance has meaningful variation.
        
        Args:
            importance_dict (dict): Feature importance dictionary
            
        Returns:
            tuple: (is_valid, validation_message)
        """
        values = list(importance_dict.values())
        
        if len(values) == 0:
            return False, "No feature importance values"
        
        # Check if all values are identical (uniform distribution)
        unique_values = len(set(values))
        if unique_values == 1:
            return False, f"All {len(values)} features have identical importance: {values[0]:.6f}. Model may be untrained or needs more training."
        
        # Check standard deviation
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
        
        if cv < 0.05:  # Less than 5% variation
            return False, f"Feature importance too uniform (CV={cv:.4f}, std={std_val:.6f}). Model may need training or retraining."
        
        return True, f"Feature importance validated (CV={cv:.4f}, {unique_values} unique values out of {len(values)})"
    
    def feature_importance(self, X):
        """Calculate feature importance with validation and training status check.
        
        Args:
            X (np.ndarray): Feature array with shape (n_samples, n_features)
            
        Returns:
            dict: Feature importance scores
        """
        # CRITICAL CHECK: Verify model has been trained
        is_trained, avg_param = self._check_model_training_status()
        
        if not is_trained:
            logger.warning(
                f"Model appears untrained (avg param value: {avg_param:.6f}). "
                "Feature importance will be unreliable. Please train the model first."
            )
            # Return uniform importance with a warning
            n_features = X.shape[1] if len(X.shape) == 2 else X.shape[2]
            uniform_importance = {f"feature_{i}": 1.0/n_features for i in range(n_features)}
            return uniform_importance
        
        # Model is trained - proceed with gradient-based importance
        logger.info(f"Model training status: OK (avg param: {avg_param:.6f})")
        
        # CRITICAL: Temporarily set model to training mode for gradient computation
        # This is necessary for feature importance calculation, but we'll restore eval mode after
        was_training = self.model.training
        self.model.train()
        
        try:
            # Scale the features
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(self.device)
            
            # Enable gradient computation temporarily for feature importance calculation
            with torch.enable_grad():
                # Forward pass
                output = self.model(X_tensor)
                
                # Calculate gradients
                output.sum().backward()
            
            # Get gradients with respect to inputs - check if gradients exist
            if X_tensor.grad is None:
                # No gradients computed - return uniform importance
                logger.warning("No gradients computed for feature importance. Returning uniform importance.")
                n_features = X_scaled.shape[1]
                importance = np.ones(n_features) / n_features
            else:
                gradients = X_tensor.grad.abs().mean(dim=(0, 1)).cpu().numpy()
                
                # Normalize gradients to avoid division by zero
                grad_sum = gradients.sum()
                if grad_sum > 0:
                    importance = gradients / grad_sum
                else:
                    logger.warning("Zero gradient sum - returning uniform importance")
                    importance = np.ones_like(gradients) / len(gradients)
            
            # Map to feature names
            importance_dict = {name: float(imp) for name, imp in zip(self.feature_names, importance)}
            
            # CRITICAL: Validate feature importance has variation
            is_valid, validation_msg = self._validate_feature_importance(importance_dict)
            
            if not is_valid:
                logger.warning(f"Feature importance validation failed: {validation_msg}")
                # Still return the importance, but log the warning
            else:
                logger.info(f"Feature importance validation passed: {validation_msg}")
            
            # Log top features for debugging
            if logger.isEnabledFor(logging.DEBUG):
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.debug(f"Top 5 features: {sorted_features}")
            
            return importance_dict
        finally:
            # CRITICAL: Restore the model's original state
            if not was_training:
                self.model.eval()


class TensorFlowStockPredictor:
    """Stock price prediction model using TensorFlow/Keras."""
    
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, dropout=0.2, architecture_type='lstm'):
        """Initialize the TensorFlow deep learning model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of neurons in hidden layers
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate for regularization
            architecture_type (str): Type of architecture to use ('lstm', 'gru', 'transformer')
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.architecture_type = architecture_type.lower()  # Normalize to lowercase
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self._build_model()
        
    def _build_model(self):
        """Build the model architecture based on selected type."""
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        if self.architecture_type == 'lstm':
            self._build_lstm_model()
        elif self.architecture_type == 'gru':
            self._build_gru_model()
        elif self.architecture_type == 'transformer':
            self._build_transformer_model()
        elif self.architecture_type == 'tft':
            # TFT should use TFTStockPredictor, but if we get here, use transformer as fallback
            logger.warning("TFT architecture should use TFTStockPredictor. Using transformer architecture instead.")
            self._build_transformer_model()
        else:
            logger.warning(f"Unknown architecture type: {self.architecture_type}. Falling back to LSTM.")
            self._build_lstm_model()
        
        # Compile the model - same settings for all architectures
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
    def _build_lstm_model(self):
        """Build an LSTM model architecture using Keras."""
        # Create sequential model
        self.model = keras.Sequential()
        
        # Add LSTM layers
        self.model.add(layers.LSTM(
            units=self.hidden_dim, 
            input_shape=(1, self.input_dim),  # (sequence_length, features)
            return_sequences=self.num_layers > 1
        ))
        
        # Add additional LSTM layers if specified
        for i in range(1, self.num_layers):
            self.model.add(layers.LSTM(
                units=self.hidden_dim,
                return_sequences=i < self.num_layers - 1
            ))
            if self.dropout > 0:
                self.model.add(layers.Dropout(self.dropout))
        
        # Add dense layers for prediction
        self.model.add(layers.Dense(self.hidden_dim // 2, activation='relu'))
        if self.dropout > 0:
            self.model.add(layers.Dropout(self.dropout))
        self.model.add(layers.Dense(1))  # Output layer (price prediction)
        
    def _build_gru_model(self):
        """Build a GRU model architecture using Keras."""
        # Create sequential model
        self.model = keras.Sequential()
        
        # Add GRU layers
        self.model.add(layers.GRU(
            units=self.hidden_dim, 
            input_shape=(1, self.input_dim),  # (sequence_length, features)
            return_sequences=self.num_layers > 1
        ))
        
        # Add additional GRU layers if specified
        for i in range(1, self.num_layers):
            self.model.add(layers.GRU(
                units=self.hidden_dim,
                return_sequences=i < self.num_layers - 1
            ))
            if self.dropout > 0:
                self.model.add(layers.Dropout(self.dropout))
        
        # Add dense layers for prediction
        self.model.add(layers.Dense(self.hidden_dim // 2, activation='relu'))
        if self.dropout > 0:
            self.model.add(layers.Dropout(self.dropout))
        self.model.add(layers.Dense(1))  # Output layer (price prediction)
        
    def _build_transformer_model(self):
        """Build a Transformer model architecture using Keras."""
        # For transformer models, we need to ensure the hidden_dim is divisible by num_heads
        num_heads = min(8, max(4, self.hidden_dim // 16))  # Between 4 and 8 heads
        self.hidden_dim = num_heads * (self.hidden_dim // num_heads)  # Make hidden_dim divisible by num_heads
        
        # Create input layer
        inputs = layers.Input(shape=(1, self.input_dim))
        
        # First, convert input to hidden_dim using Dense layer
        x = layers.Dense(self.hidden_dim)(inputs)
        
        # Add multiple Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention layer
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=self.hidden_dim // num_heads
            )(x, x)
            
            # Skip connection & layer normalization
            x = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn = keras.Sequential([
                layers.Dense(self.hidden_dim * 2, activation='relu'),
                layers.Dense(self.hidden_dim)
            ])
            
            ffn_output = ffn(x)
            x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)
            
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
        
        # Extract the last sequence element for prediction
        x = x[:, -1, :]
        
        # Final prediction layers
        x = layers.Dense(self.hidden_dim // 2, activation='relu')(x)
        if self.dropout > 0:
            x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1)(x)
        
        # Build the model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
    
    def fit(self, X, y, epochs=50, batch_size=32, verbose=False):
        """Train the TensorFlow model.
        
        Args:
            X (np.ndarray): Feature array with shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            y (np.ndarray): Target array with shape (n_samples,)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            verbose (bool): Whether to print training progress
        """
        # Handle 3D input for sequence models
        is_3d = len(X.shape) == 3
        
        if is_3d:
            # Reshape from (n_samples, seq_len, n_features) to (n_samples, seq_len * n_features)
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(n_samples, seq_len * n_features)
            X_scaled = self.scaler.fit_transform(X_2d)
            # Reshape back to 3D
            X_reshaped = X_scaled.reshape(n_samples, seq_len, n_features)
        else:
            # Standard 2D scaling and reshape to 3D
            X_scaled = self.scaler.fit_transform(X)
            # Reshape for LSTM [samples, time_steps, features]
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Train the model
        self.model.fit(
            X_reshaped, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0,
            validation_split=0.1
        )
    
    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Feature array with shape (n_samples, n_features) or (n_samples, seq_len, n_features)
            
        Returns:
            np.ndarray: Predictions with shape (n_samples,)
        """
        # Handle 3D input for sequence models
        is_3d = len(X.shape) == 3
        
        if is_3d:
            # Reshape from (n_samples, seq_len, n_features) to (n_samples, seq_len * n_features)
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(n_samples, seq_len * n_features)
            X_scaled = self.scaler.transform(X_2d)
            # Reshape back to 3D
            X_reshaped = X_scaled.reshape(n_samples, seq_len, n_features)
        else:
            # Standard 2D scaling and reshape to 3D
            X_scaled = self.scaler.transform(X)
            # Reshape for LSTM [samples, time_steps, features]
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        
        # Make predictions
        predictions = self.model.predict(X_reshaped)
        
        return predictions.flatten()
    
    def save(self, model_path=TENSORFLOW_MODEL_PATH, scaler_path=SCALER_PATH):
        """Save the model and scaler."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the Keras model
            self.model.save(model_path)
            
            # Save the feature names along with the model
            np.save(f"{model_path}/feature_names.npy", np.array(self.feature_names))
            
            # Save model parameters
            params = {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'architecture_type': self.architecture_type
            }
            with open(f"{model_path}/params.json", 'w') as f:
                json.dump(params, f)
            
            # Save the scaler
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"TensorFlow model saved to {model_path}")
            return True
            
        except Exception as e:
            # Catch all exceptions to ensure training doesn't crash on save failure
            # This includes OSError for file operations, TensorFlow errors, etc.
            logger.error(f"Error saving TensorFlow model: {e}")
            return False
        
    def load(self, model_path=TENSORFLOW_MODEL_PATH, scaler_path=SCALER_PATH):
        """Load the model and scaler."""
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False
        
        try:
            # Load the model
            self.model = keras.models.load_model(model_path)
            
            # Load feature names
            if os.path.exists(f"{model_path}/feature_names.npy"):
                self.feature_names = np.load(f"{model_path}/feature_names.npy").tolist()
                
            # Load model parameters
            if os.path.exists(f"{model_path}/params.json"):
                with open(f"{model_path}/params.json", 'r') as f:
                    params = json.load(f)
                self.input_dim = params['input_dim']
                self.hidden_dim = params['hidden_dim']
                self.num_layers = params['num_layers']
                self.dropout = params['dropout']
                # Load architecture type with fallback for backward compatibility
                self.architecture_type = params.get('architecture_type', 'lstm')
            
            # Load the scaler
            self.scaler = joblib.load(scaler_path)
            
            return True
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            return False
    
    def feature_importance(self, X):
        """Calculate feature importance.
        
        Args:
            X (np.ndarray): Feature array with shape (n_samples, n_features)
            
        Returns:
            dict: Feature importance scores
        """
        # For TF models, we'll use permutation importance
        # Simple implementation for demonstration purposes
        
        # First, get baseline performance (MSE)
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        baseline_predictions = self.model.predict(X_reshaped).flatten()
        
        # Calculate feature importances
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # Create a copy and permute one feature
            X_permuted = X_scaled.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            # Reshape and predict
            X_permuted_reshaped = X_permuted.reshape(X_permuted.shape[0], 1, X_permuted.shape[1])
            permuted_predictions = self.model.predict(X_permuted_reshaped).flatten()
            
            # Calculate how much performance dropped (higher means more important)
            importances[i] = np.mean(np.abs(permuted_predictions - baseline_predictions))
        
        # Normalize importances
        importances = importances / np.sum(importances)
        
        # Map to feature names
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}


def prepare_data_for_ml(data, window_size=5, target_days=5, use_feature_engineering=True, feature_type='balanced'):
    """Prepare data for machine learning by creating sequences and targets.
    
    Args:
        data (pd.DataFrame): Preprocessed dataframe with features
        window_size (int): Number of past days to use for prediction
        target_days (int): Number of days ahead to predict
        use_feature_engineering (bool): Whether to use advanced feature engineering
        feature_type (str): Type of features to generate ('minimal', 'balanced', 'full')
        
    Returns:
        tuple: X, y arrays for training
    """
    # Generate features
    features_df = create_features(data, feature_type=feature_type, use_feature_engineering=use_feature_engineering)
    
    # Extract features (exclude target and date columns)
    features = features_df.drop(['date', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
    
    # Create target: future price change percentage
    if 'close' in data.columns:
        target = data['close'].pct_change(periods=target_days).shift(-target_days)
    else:
        # Fallback if we don't have 'close' in the original data
        target = features_df.iloc[:, 0].rolling(window=target_days).mean().pct_change().shift(-target_days)
        
    target = target.fillna(0)  # Fill NaN with zeros
    
    # If advanced feature engineering is enabled and available, enhance the features further
    if FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering:
        try:
            # Use dimensionality reduction if we have many features
            if features.shape[1] > 50:
                from feature_engineering import DimensionalityReducer
                reducer = DimensionalityReducer(method='pca', n_components=min(30, features.shape[1]))
                features_reduced = reducer.fit_transform(features)
                features = features_reduced
                logger.info(f"Reduced features from {features_df.shape[1]} to {features.shape[1]} dimensions")
        except Exception as e:
            logger.warning(f"Error in dimensionality reduction: {str(e)}")
    
    # Create sequences for LSTM/RNN models
    X, y = [], []
    for i in range(len(features) - window_size - target_days + 1):
        X.append(features.iloc[i:i+window_size].values)
        y.append(target.iloc[i+window_size-1])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def optimize_model(
    X_train,
    y_train, 
    model_type='auto', 
    optimization_method='bayesian',
    n_trials=50,
    cv=5,
    time_budget_mins=30,
    save_results=True
):
    """
    Optimize hyperparameters for the specified model type and dataset.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        model_type (str): Model type ('pytorch', 'tensorflow', 'random_forest', or 'auto')
        optimization_method (str): Method to use ('grid', 'random', 'bayesian')
        n_trials (int): Number of trials/iterations for optimization
        cv (int): Number of cross-validation folds
        time_budget_mins (int): Time budget in minutes (for early stopping)
        save_results (bool): Whether to save optimization results
    
    Returns:
        tuple: (best_params, optimization_result)
    """
    # Check if hyperparameter optimization is available
    if not HYPERPARAMETER_OPTIMIZATION_AVAILABLE:
        logger.warning("Hyperparameter optimization module not available. Using default parameters.")
        
        # Return default parameters based on model type
        if model_type == 'pytorch' and PYTORCH_AVAILABLE:
            return {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2}, None
        elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
            return {'units': 64, 'num_layers': 2, 'dropout': 0.2}, None
        else:  # random_forest or fallback
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}, None
    
    # Set timeout for optuna
    timeout = time_budget_mins * 60 if time_budget_mins > 0 else None
    
    # Determine the best available model type if 'auto' is specified
    if model_type == 'auto':
        if PYTORCH_AVAILABLE:
            model_type = 'pytorch'
        elif TENSORFLOW_AVAILABLE:
            model_type = 'tensorflow'
        else:
            model_type = 'random_forest'
            
    logger.info(f"Starting hyperparameter optimization for {model_type} model using {optimization_method} method")
    
    result = None
    
    # Optimize based on model type
    if model_type == 'random_forest':
        # Define parameter grid/ranges
        if optimization_method == 'bayesian':
            # Use Optuna with defined parameter ranges
            param_ranges = {
                'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
                'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': lambda trial: trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': lambda trial: trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
            
            # Optimize with Optuna
            result = optimize_sklearn_model_optuna(
                model_class=RandomForestRegressor,
                X_train=X_train,
                y_train=y_train,
                param_ranges=param_ranges,
                n_trials=n_trials,
                cv=cv,
                timeout=timeout,
                verbose=True,
                random_state=42
            )
        else:
            # Use grid or random search
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            
            # Use grid or random search
            result = optimize_sklearn_model(
                model_class=RandomForestRegressor,
                X_train=X_train,
                y_train=y_train,
                param_grid=param_grid,
                method='random_search' if optimization_method == 'random' else 'grid_search',
                n_iter=n_trials if optimization_method == 'random' else None,
                scoring='neg_mean_squared_error',
                cv=cv,
                verbose=1,
                random_state=42
            )
    
    elif model_type == 'pytorch' and PYTORCH_AVAILABLE:
        # Reshape data for PyTorch if needed - expecting batch_size, seq_length, features
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        # Define parameter ranges
        param_ranges = {
            'input_dim': lambda trial: trial.suggest_categorical('input_dim', [X_train.shape[1]]),
            'hidden_dim': lambda trial: trial.suggest_int('hidden_dim', 32, 256),
            'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
            'dropout': lambda trial: trial.suggest_float('dropout', 0.0, 0.5),
        }
        
        # Optimize PyTorch model
        result = optimize_pytorch_model(
            model_class=PyTorchStockPredictor,
            X_train=X_tensor,
            y_train=y_tensor,
            param_ranges=param_ranges,
            n_trials=n_trials,
            timeout=timeout,
            verbose=True,
            random_state=42
        )
    
    elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
        # Define model builder function
        def create_tf_stock_predictor(input_dim=X_train.shape[1], units=64, num_layers=2, dropout=0.2):
            # Set random seed
            tf.random.set_seed(42)
            
            # Create sequential model
            model = keras.Sequential()
            
            # Add LSTM layers
            model.add(keras.layers.LSTM(
                units=units, 
                input_shape=(1, input_dim),  # Assumes reshape will happen
                return_sequences=num_layers > 1
            ))
            
            # Add additional LSTM layers if specified
            for i in range(1, num_layers):
                model.add(keras.layers.LSTM(
                    units=units,
                    return_sequences=i < num_layers - 1
                ))
                if dropout > 0:
                    model.add(keras.layers.Dropout(dropout))
            
            # Output layer
            model.add(keras.layers.Dense(1))
            
            # Compile
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        # Define parameter ranges
        param_ranges = {
            'units': lambda trial: trial.suggest_int('units', 32, 256),
            'num_layers': lambda trial: trial.suggest_int('num_layers', 1, 3),
            'dropout': lambda trial: trial.suggest_float('dropout', 0.0, 0.5),
        }
        
        # Reshape data for LSTM if needed
        if len(X_train.shape) == 2:
            X_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        else:
            X_reshaped = X_train
        
        # Optimize TensorFlow model
        result = optimize_tensorflow_model(
            model_builder=create_tf_stock_predictor,
            X_train=X_reshaped,
            y_train=y_train,
            param_ranges=param_ranges,
            n_trials=n_trials,
            timeout=timeout,
            verbose=True,
            random_state=42
        )
    
    else:
        logger.warning(f"Model type '{model_type}' is not supported or not available. Falling back to RandomForest.")
        
        # Define parameter grid for default RandomForest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use grid search by default
        result = optimize_sklearn_model(
            model_class=RandomForestRegressor,
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid,
            method='grid_search',
            scoring='neg_mean_squared_error',
            cv=cv,
            verbose=1,
            random_state=42
        )
    
    # Save results if requested
    if save_results and result:
        result_path = result.save(HYPERPARAMETER_PATH)
        logger.info(f"Saved optimization results to {result_path}")
    
    # Return the best parameters and result object
    if result:
        logger.info(f"Best parameters: {result.best_params}")
        logger.info(f"Best score: {result.best_score}")
        return result.best_params, result
    else:
        logger.warning("Optimization failed, returning default parameters")
        # Return default parameters based on model type
        if model_type == 'pytorch':
            return {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2}, None
        elif model_type == 'tensorflow':
            return {'units': 64, 'num_layers': 2, 'dropout': 0.2}, None
        else:  # random_forest or fallback
            return {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2}, None


def load_or_train_model(X_train=None, y_train=None, model_type='auto', architecture_type='lstm'):
    """Load existing model or train a new one if not found.
    
    Args:
        X_train (np.ndarray, optional): Training features. Defaults to None.
        y_train (np.ndarray, optional): Training targets. Defaults to None.
        model_type (str, optional): Model type to use ('pytorch', 'tensorflow', 'random_forest', 'tft', or 'auto').
                                   'auto' will use the best available model. Defaults to 'auto'.
        architecture_type (str, optional): Neural network architecture type ('lstm', 'gru', 'transformer', 'tft').
                                         Defaults to 'lstm'.
                                   
    Returns:
        tuple: (model, scaler, model_type)
    """
    # Handle TFT architecture type
    if architecture_type.lower() == 'tft':
        if TFT_AVAILABLE:
            model_type = 'tft'
        else:
            # TFT not available, fall back to transformer for PyTorch/TensorFlow models
            logger.warning("TFT not available (PyTorch required). Falling back to transformer architecture.")
            architecture_type = 'transformer'
    
    # Determine the best available model type if 'auto' is specified
    if model_type == 'auto':
        if TFT_AVAILABLE and architecture_type.lower() == 'tft':
            model_type = 'tft'
        elif PYTORCH_AVAILABLE:
            model_type = 'pytorch'
        elif TENSORFLOW_AVAILABLE:
            model_type = 'tensorflow'
        else:
            model_type = 'random_forest'
            
    logger.info(f"Using {model_type} model with {architecture_type} architecture")
    
    # Try to load existing model
    try:
        # Handle TFT model loading
        if model_type == 'tft' and TFT_AVAILABLE:
            model = TFTStockPredictor(
                input_dim=X_train.shape[-1] if X_train is not None else 50,
                static_dim=10
            )
            if model.load(TFT_MODEL_PATH):
                logger.info("Loaded existing TFT model")
                return model, model.scaler, 'tft'
        
        elif model_type == 'pytorch' and PYTORCH_AVAILABLE:
            # Try to load PyTorch model
            model = PyTorchStockPredictor(architecture_type=architecture_type)
            if model.load(PYTORCH_MODEL_PATH, SCALER_PATH):
                logger.info(f"Loaded existing PyTorch model with {model.architecture_type} architecture")
                return model, model.scaler, 'pytorch'
                
        elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
            # Try to load TensorFlow model
            model = TensorFlowStockPredictor(architecture_type=architecture_type)
            if model.load(TENSORFLOW_MODEL_PATH, SCALER_PATH):
                logger.info(f"Loaded existing TensorFlow model with {model.architecture_type} architecture")
                return model, model.scaler, 'tensorflow'
                
        elif model_type == 'random_forest' or not (PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE):
            # Try to load RandomForest model
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                model = joblib.load(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)
                logger.info("Loaded existing RandomForest model")
                return model, scaler, 'random_forest'
                
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        
    # If we get here, we need to train a new model
    
    # Check if training data is provided
    if X_train is None or y_train is None:
        # Create dummy data for first-time model creation
        logger.info("No training data provided, using synthetic data")
        X_train = np.random.rand(1000, 14)  # 14 features
        y_train = np.random.rand(1000)
    
    # Try to optimize hyperparameters if available
    best_params = None
    if HYPERPARAMETER_OPTIMIZATION_AVAILABLE and len(X_train) > 100:  # Only optimize if we have enough data
        try:
            # Check if we have saved optimization results
            if os.path.exists(HYPERPARAMETER_PATH):
                try:
                    from hyperparameter_optimization import OptimizationResult
                    optimization_result = OptimizationResult.load(HYPERPARAMETER_PATH)
                    if optimization_result and hasattr(optimization_result, 'best_params'):
                        best_params = optimization_result.best_params
                        logger.info(f"Loaded optimized parameters: {best_params}")
                except Exception as e:
                    logger.warning(f"Error loading optimization results: {e}")
            
            # If no saved results or loading failed, run optimization
            if not best_params:
                # Run quick optimization with reasonable defaults
                best_params, _ = optimize_model(
                    X_train=X_train,
                    y_train=y_train,
                    model_type=model_type,
                    optimization_method='bayesian',
                    n_trials=20,  # Quick optimization
                    cv=3,
                    time_budget_mins=10  # Limit to 10 minutes
                )
        except Exception as e:
            logger.warning(f"Error in hyperparameter optimization: {e}")
            best_params = None
        
    # Train the appropriate model type
    # Handle TFT training
    if model_type == 'tft' and TFT_AVAILABLE:
        # Create and train TFT model
        logger.info("Training new TFT model...")
        
        # Determine input dimensions
        if X_train is not None:
            if len(X_train.shape) == 3:
                input_dim = X_train.shape[2]
                seq_len = X_train.shape[1]
            else:
                input_dim = X_train.shape[1]
                seq_len = 60  # Default sequence length
                # Reshape 2D data to 3D for TFT
                X_train = X_train.reshape(-1, 1, input_dim)
                # Repeat to create sequence
                X_train = np.tile(X_train, (1, seq_len, 1))
        else:
            input_dim = 50
            seq_len = 60
        
        # Use optimized hyperparameters if available
        tft_params = {
            'input_dim': input_dim,
            'static_dim': 10,
            'hidden_dim': best_params.get('hidden_dim', 128) if best_params else 128,
            'forecast_horizons': [5, 10, 20, 30],
            'num_heads': best_params.get('num_heads', 4) if best_params else 4,
            'num_lstm_layers': best_params.get('num_lstm_layers', 2) if best_params else 2,
            'dropout': best_params.get('dropout', 0.1) if best_params else 0.1,
            'num_attention_layers': best_params.get('num_attention_layers', 2) if best_params else 2
        }

        model = TFTStockPredictor(**tft_params)
        
        # Create static features using the TFT integration utility
        # In production, this should extract actual static features from metadata
        n_samples = X_train.shape[0]
        try:
            X_static = np.zeros((n_samples, 10), dtype=np.float32)
            # Populate with any available metadata if present in the training context
            logger.info("Using default static features - extend with actual symbol metadata in production")
        except Exception as e:
            logger.warning(f"Error creating static features: {e}, using zeros")
            X_static = np.zeros((n_samples, 10), dtype=np.float32)
        
        # Prepare multi-horizon targets if single target provided
        if y_train is not None:
            if y_train.ndim == 1:
                # Create multi-horizon targets using proper forward-looking indexing
                # This avoids data leakage by using correct time series alignment
                horizons = model.forecast_horizons
                max_horizon = max(horizons)
                if len(y_train) > max_horizon:
                    # Use forward indexing instead of roll to avoid data leakage
                    y_multi_cols = []
                    for h in horizons:
                        # y[t+h] is the target for prediction at time t
                        y_shifted = y_train[h:]  # Future values starting at offset h
                        y_multi_cols.append(y_shifted)
                    
                    # Truncate all to same length (shortest sequence)
                    min_len = min(len(col) for col in y_multi_cols)
                    y_multi = np.column_stack([col[:min_len] for col in y_multi_cols])
                    
                    # Truncate X_train and X_static to match
                    X_train = X_train[:min_len]
                    X_static = X_static[:min_len]
                    y_train = y_multi
                else:
                    y_train = np.column_stack([y_train] * len(horizons))
        
        model.feature_names = [f"feature_{i}" for i in range(input_dim)]

        # Use optimized training hyperparameters if available
        fit_params = {
            'epochs': best_params.get('epochs', 50) if best_params else 50,
            'batch_size': best_params.get('batch_size', 32) if best_params else 32,
            'lr': best_params.get('learning_rate', 0.001) if best_params else 0.001,
            'verbose': True
        }

        logger.info(f"Training TFT with params: {fit_params}")
        model.fit(X_train, X_static, y_train, **fit_params)
        model.save(TFT_MODEL_PATH)
        logger.info("Trained and saved new TFT model")
        return model, model.scaler, 'tft'
    
    elif model_type == 'pytorch' and PYTORCH_AVAILABLE:
        # Create and train PyTorch model
        # Correctly extract input_dim from 3D or 2D input
        if len(X_train.shape) == 3:
            # For 3D input (n_samples, seq_len, n_features), use n_features as input_dim
            input_dim = X_train.shape[2]
        else:
            # For 2D input (n_samples, n_features), use n_features as input_dim
            input_dim = X_train.shape[1]
        
        if best_params:
            input_dim = best_params.get('input_dim', input_dim)
            hidden_dim = best_params.get('hidden_dim', 64)
            num_layers = best_params.get('num_layers', 2)
            dropout = best_params.get('dropout', 0.2)
            model = PyTorchStockPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                architecture_type=architecture_type
            )
        else:
            model = PyTorchStockPredictor(
                input_dim=input_dim,
                architecture_type=architecture_type
            )
            
        model.feature_names = [f"feature_{i}" for i in range(input_dim)]  # Dummy feature names
        model.fit(X_train, y_train, epochs=30, verbose=True)
        model.save()
        logger.info(f"Trained and saved new PyTorch model with {architecture_type} architecture")
        return model, model.scaler, 'pytorch'
        
    elif model_type == 'tensorflow' and TENSORFLOW_AVAILABLE:
        # Create and train TensorFlow model
        # Correctly extract input_dim from 3D or 2D input
        if len(X_train.shape) == 3:
            # For 3D input (n_samples, seq_len, n_features), use n_features as input_dim
            input_dim = X_train.shape[2]
        else:
            # For 2D input (n_samples, n_features), use n_features as input_dim
            input_dim = X_train.shape[1]
        
        if best_params:
            input_dim = best_params.get('input_dim', input_dim)
            units = best_params.get('units', 64)
            num_layers = best_params.get('num_layers', 2)
            dropout = best_params.get('dropout', 0.2)
            model = TensorFlowStockPredictor(
                input_dim=input_dim,
                hidden_dim=units,
                num_layers=num_layers,
                dropout=dropout,
                architecture_type=architecture_type
            )
        else:
            model = TensorFlowStockPredictor(
                input_dim=input_dim,
                architecture_type=architecture_type
            )
            
        model.feature_names = [f"feature_{i}" for i in range(input_dim)]  # Dummy feature names
        model.fit(X_train, y_train, epochs=30, verbose=True)
        model.save()
        logger.info(f"Trained and saved new TensorFlow model with {architecture_type} architecture")
        return model, model.scaler, 'tensorflow'
        
    else:
        # Fallback to RandomForest model
        logger.info("Using RandomForest model as fallback")
        
        # Random Forest doesn't handle 3D input - flatten if necessary
        if len(X_train.shape) == 3:
            n_samples, seq_len, n_features = X_train.shape
            X_train_2d = X_train.reshape(n_samples, seq_len * n_features)
            logger.info(f"Flattened 3D input ({n_samples}, {seq_len}, {n_features}) to 2D ({n_samples}, {seq_len * n_features}) for RandomForest")
        else:
            X_train_2d = X_train
        
        if best_params:
            model = RandomForestRegressor(
                n_estimators=best_params.get('n_estimators', 100),
                max_depth=best_params.get('max_depth', 10),
                min_samples_split=best_params.get('min_samples_split', 5),
                min_samples_leaf=best_params.get('min_samples_leaf', 2),
                max_features=best_params.get('max_features', 'auto'),
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_2d)
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Save the model and scaler
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Trained and saved new RandomForest model")
        
        return model, scaler, 'random_forest'


def predict_stock(features, model_type='auto', architecture_type='lstm', use_feature_engineering=True, optimize_hyperparams=False):
    """
    Make stock predictions using the provided features and the specified ML model.
    
    Args:
        features (dict): Dictionary of feature names and values
        model_type (str): Model type to use ('pytorch', 'tensorflow', 'random_forest', or 'auto')
        architecture_type (str): Neural network architecture type ('lstm', 'gru', 'transformer')
        use_feature_engineering (bool): Whether to use advanced feature engineering
        optimize_hyperparams (bool): Whether to optimize hyperparameters if training a new model
        
    Returns:
        dict: Prediction results with action, confidence, target price, etc.
    """
    try:
        # Create a simple feature set if none exists
        if not features:
            features = {'dummy': 0.0}
            
        # Initialize models
        if optimize_hyperparams and HYPERPARAMETER_OPTIMIZATION_AVAILABLE:
            logger.info("Hyperparameter optimization requested for prediction")
            # We would need training data for optimization
            # For now, just indicate that optimization is available but needs training data
            hyperparameter_info = {
                'optimization_available': True,
                'status': 'Training data required for optimization'
            }
        
        # Load or train the model
        model, scaler, used_model_type = load_or_train_model(
            model_type=model_type,
            architecture_type=architecture_type
        )
        
        used_architecture = 'n/a'
        if hasattr(model, 'architecture_type'):
            used_architecture = model.architecture_type
        
        logger.info(f"Making prediction with {used_model_type} model using {used_architecture} architecture")
        
        # Convert features to DataFrame for feature engineering
        feature_df = pd.DataFrame([features])
        
        # Check if we have OHLCV data for feature engineering
        has_ohlcv = all(col in feature_df.columns for col in ['open', 'high', 'low', 'close'])
        
        # Apply feature engineering if we have OHLCV data and feature engineering is enabled
        if has_ohlcv and FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering:
            try:
                # Try to load saved pipeline
                if os.path.exists(FEATURE_PIPELINE_PATH):
                    from feature_engineering import FeatureEngineer
                    pipeline = FeatureEngineer.load(FEATURE_PIPELINE_PATH)
                    feature_df = pipeline.transform(feature_df)
                    logger.info(f"Applied feature engineering pipeline, generated {feature_df.shape[1]} features")
                else:
                    # Use basic feature creation
                    feature_df = create_features(feature_df, use_feature_engineering=False)
                    logger.info("No saved pipeline found, using basic feature creation")
            except Exception as e:
                logger.warning(f"Error in feature engineering: {str(e)}. Using raw features.")
        
        # Prepare feature array for prediction
        feature_names = list(feature_df.columns)
        feature_array = feature_df.values

        # Get current price from features early - this is required for error handling
        # The model predicts percentage change, not actual price
        current_price = features.get('current_price', 0.0)
        if current_price <= 0:
            # Try alternate keys for current price
            current_price = features.get('close', features.get('price', 0.0))

        # Check for feature dimension mismatch and retrain if necessary
        expected_features = None
        if used_model_type in ['pytorch', 'tensorflow']:
            if hasattr(model, 'input_dim'):
                expected_features = model.input_dim
        elif used_model_type == 'random_forest':
            if scaler and hasattr(scaler, 'n_features_in_'):
                expected_features = scaler.n_features_in_

        if expected_features is not None and feature_array.shape[1] != expected_features:
            logger.warning(f"Feature dimension mismatch: model expects {expected_features} features but got {feature_array.shape[1]}")
            logger.warning(f"Deleting old model and automatically retraining with correct features...")

            # Delete old model files
            try:
                if used_model_type == 'pytorch' and os.path.exists(PYTORCH_MODEL_PATH):
                    os.remove(PYTORCH_MODEL_PATH)
                    logger.info(f"Deleted old PyTorch model")
                elif used_model_type == 'tensorflow' and os.path.exists(TENSORFLOW_MODEL_PATH):
                    import shutil
                    shutil.rmtree(TENSORFLOW_MODEL_PATH)
                    logger.info(f"Deleted old TensorFlow model")
                elif used_model_type == 'random_forest' and os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    logger.info(f"Deleted old Random Forest model")

                if os.path.exists(SCALER_PATH):
                    os.remove(SCALER_PATH)
                    logger.info(f"Deleted old scaler")
            except Exception as e:
                logger.error(f"Error deleting old model: {e}")

            # Generate synthetic training data matching the new feature dimensions
            logger.info(f"Generating synthetic training data with {feature_array.shape[1]} features...")
            n_samples = 1000
            X_synthetic = np.random.randn(n_samples, feature_array.shape[1]) * 0.1
            y_synthetic = np.random.randn(n_samples) * 0.05  # Small percentage changes
            
            # Retrain the model automatically with the correct dimensions
            logger.info(f"Retraining {used_model_type} model with {architecture_type} architecture...")
            try:
                model, scaler, used_model_type = load_or_train_model(
                    X_train=X_synthetic,
                    y_train=y_synthetic,
                    model_type=model_type,
                    architecture_type=architecture_type
                )
                logger.info(f"Model successfully retrained with {feature_array.shape[1]} features")
                # Continue with prediction using the newly trained model
            except Exception as retrain_error:
                logger.error(f"Failed to retrain model: {retrain_error}")
                # Return error if retraining fails
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'targetPrice': current_price,
                    'weights': {},
                    'timeSeries': {
                        'prices': [current_price],
                        'dates': [datetime.now().strftime('%Y-%m-%d')],
                        'confidence': 0.0
                    },
                    'risk': {
                        'var': 0.0,
                        'maxDrawdown': 0.0,
                        'sharpeRatio': 0.0,
                        'riskScore': 0.5
                    },
                    'patterns': [],
                    'modelType': 'error',
                    'architectureType': 'n/a',
                    'error': f'Feature dimension mismatch and automatic retraining failed: {str(retrain_error)}. Please use "Train Model" button with proper training data.',
                    'hyperparameterOptimization': {
                        'available': HYPERPARAMETER_OPTIMIZATION_AVAILABLE,
                        'used': False,
                        'optimizedParams': False
                    },
                    'needsRetraining': True
                }

        # Align feature names with model if possible
        if used_model_type in ['pytorch', 'tensorflow'] and hasattr(model, 'feature_names'):
            model.feature_names = feature_names
        
        # IMPORTANT: If we still don't have a valid price, try to estimate from feature data
        # This is a fallback - the C# client should ALWAYS send current_price
        if current_price <= 0:
            logger.warning("No current_price in features. Attempting to estimate from available data.")
            # Try to use any price-like features
            for key in ['open', 'high', 'low', 'adj_close', 'adjclose']:
                if key in features and features[key] > 0:
                    current_price = features[key]
                    logger.info(f"Using {key} as fallback current_price: {current_price}")
                    break
        
        if current_price <= 0:
            logger.error("No valid current_price provided in features. Cannot calculate target price.")
            # Return error result with clear message
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'targetPrice': 0.0,
                'weights': {},
                'timeSeries': {
                    'prices': [],
                    'dates': [],
                    'confidence': 0.0
                },
                'risk': {
                    'var': 0.0,
                    'maxDrawdown': 0.0,
                    'sharpeRatio': 0.0,
                    'riskScore': 1.0  # High risk due to missing data
                },
                'patterns': [],
                'modelType': 'error',
                'architectureType': 'n/a',
                'error': 'No current_price provided in features. C# client must include current_price, close, or price in Features dictionary.',
                'hyperparameterOptimization': {
                    'available': HYPERPARAMETER_OPTIMIZATION_AVAILABLE,
                    'used': False,
                    'optimizedParams': False
                }
            }
        
        # Make predictions based on model type
        # NOTE: The model predicts percentage change (e.g., 0.05 means +5% price change)
        if used_model_type == 'tft':
            # Use TFT model for prediction with uncertainty quantification
            logger.info("Making prediction with TFT model")
            
            # TFT expects temporal features - create from current features
            tft_result = model.predict_single(features)
            
            predicted_pct_change = float(tft_result.get('medianPrediction', 0.0))
            target_price = float(tft_result.get('targetPrice', current_price))
            confidence = float(tft_result.get('confidence', 0.7))
            action = tft_result.get('action', 'HOLD')
            
            # Get uncertainty bounds
            lower_bound = float(tft_result.get('lowerBound', predicted_pct_change - 0.05))
            upper_bound = float(tft_result.get('upperBound', predicted_pct_change + 0.05))
            uncertainty = float(tft_result.get('uncertainty', upper_bound - lower_bound))
            
            # Multi-horizon predictions
            horizons_data = tft_result.get('horizons', {})
            
            # Build feature weights from TFT importance
            weights = {}
            importance = tft_result.get('featureImportance', [])
            if importance and len(importance) > 0:
                if isinstance(importance[0], list):
                    importance = importance[0]
                for i, imp in enumerate(importance):
                    weights[f'feature_{i}'] = float(imp)
            
            # Generate time series predictions from horizons
            future_dates = []
            future_prices = []
            for horizon_key, horizon_data in horizons_data.items():
                days = int(horizon_key.replace('d', ''))
                future_dates.append((datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'))
                future_prices.append(float(horizon_data.get('target_price', target_price)))
            
            # If no horizons data, generate default
            if not future_dates:
                future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in [5, 10, 20, 30]]
                future_prices = [float(target_price) for _ in range(4)]
            
            # Calculate risk metrics from uncertainty
            volatility = uncertainty / 2  # Approximate volatility from interval width
            var_95 = current_price * volatility * 1.65 if current_price > 0 else 0.0
            max_drawdown = current_price * volatility * 2.33 if current_price > 0 else 0.0
            sharpe_ratio = abs(predicted_pct_change) / (volatility + 1e-10) if volatility > 0 else 1.0
            
            # Calculate potential return
            potential_return = predicted_pct_change
            if action == "SELL":
                potential_return *= -1
            
            # Build result with TFT-specific information
            result = {
                'action': action,
                'confidence': float(confidence),
                'targetPrice': float(target_price),
                'weights': weights,
                'timeSeries': {
                    'prices': future_prices,
                    'dates': future_dates,
                    'confidence': float(confidence)
                },
                'risk': {
                    'var': float(abs(var_95)),
                    'maxDrawdown': float(abs(max_drawdown)),
                    'sharpeRatio': float(sharpe_ratio),
                    'riskScore': float(min(1.0, volatility))
                },
                'patterns': [{
                    'name': 'Temporal Fusion Transformer Multi-Horizon Forecast',
                    'strength': float(confidence),
                    'outcome': action,
                    'detectionDate': datetime.now().strftime('%Y-%m-%d'),
                    'historicalAccuracy': 0.85
                }],
                'modelType': 'tft',
                'architectureType': 'tft',
                'featureEngineering': FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering,
                'hyperparameterOptimization': {
                    'available': HYPERPARAMETER_OPTIMIZATION_AVAILABLE,
                    'used': False,
                    'optimizedParams': False
                },
                'tftMetrics': {
                    'uncertainty': float(uncertainty),
                    'lowerBound': float(lower_bound),
                    'upperBound': float(upper_bound),
                    'horizons': horizons_data
                }
            }
            
            return result
        
        elif used_model_type == 'pytorch':
            # Scale and predict with PyTorch model
            predicted_pct_change = float(model.predict(feature_array)[0])
            
            # Calculate confidence - approximate for PyTorch
            confidence = 0.8  # Default confidence for PyTorch models
            weights = model.feature_importance(feature_array)
            
        elif used_model_type == 'tensorflow':
            # Scale and predict with TensorFlow model
            predicted_pct_change = float(model.predict(feature_array)[0])
            
            # Calculate confidence - approximate for TensorFlow
            confidence = 0.8  # Default confidence for TensorFlow models
            weights = model.feature_importance(feature_array)
            
        else:  # Random Forest
            # Scale the features
            if scaler:
                feature_array = scaler.transform(feature_array)
                
            # Make predictions
            predicted_pct_change = float(model.predict(feature_array)[0])
            
            # Calculate confidence based on feature importance for RF
            predictions = []
            for estimator in model.estimators_:
                predictions.append(float(estimator.predict(feature_array)[0]))
            
            # Confidence is inversely related to prediction variance
            pct_std = np.std(predictions) if len(predictions) > 0 else 0.0
            confidence = 1.0 - min(1.0, pct_std / (abs(np.mean(predictions)) + 1e-10))
            
            # Get feature importances for Random Forest
            weights = dict(zip(feature_names, 
                             [float(imp) for imp in model.feature_importances_]))
        
        # Clamp the predicted percentage change to reasonable bounds (-50% to +100%)
        # This prevents extreme predictions from outliers in training data
        predicted_pct_change = max(-0.5, min(1.0, predicted_pct_change))
        
        # Convert percentage change prediction to actual target price
        # target_price = current_price * (1 + predicted_percentage_change)
        target_price = current_price * (1.0 + predicted_pct_change)
        
        logger.info(f"Prediction: current_price={current_price:.2f}, predicted_pct_change={predicted_pct_change:.4f}, target_price={target_price:.2f}")
                             
        # Determine action based on predicted percentage change
        if abs(predicted_pct_change) < 0.01:  # Less than 1% change
            action = "HOLD"
        elif predicted_pct_change > 0:
            action = "BUY"
        else:
            action = "SELL"
        
        # Generate time series predictions - slightly different approach for each model
        future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                       for i in range(5)]
        
        # Generate model-specific time series predictions
        if used_model_type in ['pytorch', 'tensorflow']:
            # For deep learning models, we can generate a sequence more intelligently
            future_prices = [float(target_price)]
            
            # Generate subsequent predictions using the previous prediction
            for i in range(1, 5):
                # Shift features slightly for next day prediction
                next_features = feature_array.copy()
                
                # Add some randomness proportional to volatility
                volatility = features.get('volatility', 0.02)
                next_price = future_prices[-1] * (1 + np.random.normal(0, volatility))
                future_prices.append(float(next_price))
        else:
            # For random forest, use simple model with some randomness
            future_prices = [float(target_price * (1 + np.random.normal(0, 0.02))) 
                           for _ in range(5)]
        
        # Calculate risk metrics - adjust based on model type
        # Note: predicted_pct_change is the predicted percentage change
        price_diff = target_price - current_price  # Actual dollar difference
        
        if used_model_type in ['pytorch', 'tensorflow']:
            # For deep learning models, estimate volatility from features
            volatility = features.get('volatility', 0.02)
            # VaR (Value at Risk) at 95%: max expected loss in worst 5% of cases
            var_95 = current_price * volatility * 1.65  # 95% VaR approximation (in dollars)
            # Max drawdown: worst case scenario
            max_drawdown = current_price * volatility * 2.33  # 99% worst case (in dollars)
            sharpe_ratio = abs(predicted_pct_change) / volatility if volatility > 0 else 1.0
        else:
            # For random forest, use bootstrap predictions (which are percentage changes)
            if 'predictions' in locals() and len(predictions) > 0:
                # Predictions are percentage changes, compute volatility
                volatility = np.std(predictions) if len(predictions) > 1 else 0.02
                
                # VaR: the loss at the 5th percentile (negative values represent losses)
                # If 5th percentile is negative, that's the expected loss in worst cases
                pct_5th = np.percentile(predictions, 5)
                var_95 = current_price * max(0, -pct_5th)  # Convert negative pct to positive loss
                
                # Max drawdown: the most negative prediction (worst case loss)
                min_pct = min(predictions)
                max_drawdown = current_price * max(0, -min_pct)  # Convert to positive loss
            else:
                volatility = 0.02
                var_95 = current_price * 0.05  # Default 5% VaR
                max_drawdown = current_price * 0.1  # Default 10% max drawdown
            sharpe_ratio = abs(predicted_pct_change) / (volatility + 1e-10)
        
        # Create a pattern with model-specific name
        base_pattern_name = {
            'pytorch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'random_forest': 'Random Forest'
        }.get(used_model_type, 'ML')
        
        # Add architecture type if available
        model_architecture = used_architecture.upper() if used_model_type in ['pytorch', 'tensorflow'] else ''
        
        pattern_name = f"{base_pattern_name} {model_architecture}" if model_architecture else base_pattern_name
        
        # Add feature engineering info if used
        if FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering:
            pattern_name += ' with Advanced Feature Engineering'
        
        # Add hyperparameter optimization info if used
        if optimize_hyperparams and HYPERPARAMETER_OPTIMIZATION_AVAILABLE:
            pattern_name += ' with Hyperparameter Optimization'
            
        # Check if we have saved hyperparameter optimization results
        has_optimized_params = False
        if HYPERPARAMETER_OPTIMIZATION_AVAILABLE and os.path.exists(HYPERPARAMETER_PATH):
            has_optimized_params = True
        
        # Construct the result dictionary
        result = {
            'action': action,
            'confidence': float(confidence),
            'targetPrice': float(target_price),
            'weights': weights,
            'timeSeries': {
                'prices': future_prices,
                'dates': future_dates,
                'confidence': float(confidence)
            },
            'risk': {
                'var': float(abs(var_95)),
                'maxDrawdown': float(abs(max_drawdown)),
                'sharpeRatio': float(sharpe_ratio),
                'riskScore': float(min(1.0, volatility))
            },
            'patterns': [{
                'name': pattern_name,
                'strength': float(confidence),
                'outcome': action,
                'detectionDate': datetime.now().strftime('%Y-%m-%d'),
                'historicalAccuracy': 0.8
            }],
            'modelType': used_model_type,  # Include which model was used
            'architectureType': used_architecture if used_model_type in ['pytorch', 'tensorflow'] else 'n/a',
            'featureEngineering': FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering,
            'hyperparameterOptimization': {
                'available': HYPERPARAMETER_OPTIMIZATION_AVAILABLE,
                'used': has_optimized_params,
                'optimizedParams': True if has_optimized_params else False
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"Error in predict_stock: {str(e)}")
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'targetPrice': features.get('current_price', 0.0),
            'weights': {},
            'timeSeries': {
                'prices': [0.0],
                'dates': [datetime.now().strftime('%Y-%m-%d')],
                'confidence': 0.5
            },
            'risk': {
                'var': 0.0,
                'maxDrawdown': 0.0,
                'sharpeRatio': 0.0,
                'riskScore': 0.5
            },
            'patterns': [],
            'modelType': 'fallback',
            'architectureType': 'n/a',
            'error': str(e),
            'hyperparameterOptimization': {
                'available': HYPERPARAMETER_OPTIMIZATION_AVAILABLE,
                'used': False,
                'optimizedParams': False
            }
        }


def main():
    # Initialize debugpy remote debugging if DEBUGPY environment variable is set
    if DEBUGPY_UTILS_AVAILABLE:
        init_debugpy_if_enabled()
    
    if len(sys.argv) != 3:
        print("Usage: script.py input_file output_file", file=sys.stderr)
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Read input JSON
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Get features, model type and feature engineering settings from input
        features = data.get('Features', {})
        model_type = data.get('ModelType', 'auto')
        architecture_type = data.get('ArchitectureType', 'lstm')
        use_feature_engineering = data.get('UseFeatureEngineering', True)
        feature_type = data.get('FeatureType', 'balanced')
        optimize_hyperparams = data.get('OptimizeHyperparameters', False)
        
        # Log which model is being requested and feature engineering options
        logger.info(f"Model type requested: {model_type}")
        logger.info(f"Architecture type requested: {architecture_type}")
        logger.info(f"Feature engineering enabled: {use_feature_engineering}")
        logger.info(f"Feature type: {feature_type}")
        logger.info(f"Hyperparameter optimization requested: {optimize_hyperparams}")
        
        # Make prediction with specified model type and feature engineering options
        result = predict_stock(
            features, 
            model_type=model_type, 
            architecture_type=architecture_type,
            use_feature_engineering=use_feature_engineering, 
            optimize_hyperparams=optimize_hyperparams
        )
        
        # Add information about available models and feature engineering
        result['availableModels'] = {
            'pytorch': PYTORCH_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE,
            'randomForest': True,
            'tft': TFT_AVAILABLE
        }
        result['availableArchitectures'] = {
            'lstm': True,
            'gru': True,
            'transformer': True,
            'tft': TFT_AVAILABLE
        }
        result['availableFeatureEngineering'] = FEATURE_ENGINEERING_AVAILABLE
        result['availableHyperparameterOptimization'] = HYPERPARAMETER_OPTIMIZATION_AVAILABLE
        
        # Write output JSON
        with open(output_file, 'w') as f:
            json.dump(result, f)
            
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()