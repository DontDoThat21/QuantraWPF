#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train stock prediction models using historical data from the database.
This script can be called from C# to train models on all cached symbols.
"""

import sys
import json
import pyodbc
import gzip
import base64
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from stock_predictor import (
    load_or_train_model,
    prepare_data_for_ml,
    create_features,
    PYTORCH_AVAILABLE,
    TENSORFLOW_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('train_from_database')


def decompress_gzip_data(compressed_str):
    """Decompress GZIP compressed data from database"""
    if compressed_str.startswith('GZIP:'):
        # Remove GZIP: prefix
        base64_data = compressed_str[5:]
        # Decode from base64
        compressed_bytes = base64.b64decode(base64_data)
        # Decompress
        decompressed = gzip.decompress(compressed_bytes)
        # Decode to string
        return decompressed.decode('utf-8')
    return compressed_str


def fetch_all_historical_data(connection_string, limit=None):
    """
    Fetch all historical data from the database.
    
    Args:
        connection_string: Database connection string
        limit: Optional limit on number of symbols to fetch
        
    Returns:
        List of (symbol, historical_data) tuples
    """
    logger.info("Connecting to database...")
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    # Get all unique symbols with their most recent data
    query = """
    SELECT s.Symbol, s.Data, s.CacheTime
    FROM (
        SELECT Symbol, Data, CacheTime,
               ROW_NUMBER() OVER (PARTITION BY Symbol ORDER BY CacheTime DESC) as rn
        FROM StockDataCache
        WHERE TimeRange IN ('1mo', '3mo', '6mo', '1y')
    ) s
    WHERE s.rn = 1
    """
    
    if limit:
        query += f" ORDER BY s.Symbol OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
    
    logger.info(f"Fetching historical data from database{f' (limit: {limit})' if limit else ''}...")
    cursor.execute(query)
    
    results = []
    for row in cursor.fetchall():
        symbol, compressed_data, cache_time = row
        
        try:
            # Decompress the data
            json_data = decompress_gzip_data(compressed_data)
            historical_prices = json.loads(json_data)
            
            if len(historical_prices) > 0:
                results.append((symbol, historical_prices))
                logger.debug(f"Loaded {len(historical_prices)} records for {symbol}")
        except Exception as e:
            logger.warning(f"Error loading data for {symbol}: {str(e)}")
    
    conn.close()
    logger.info(f"Successfully loaded data for {len(results)} symbols")
    return results


def add_known_future_features(df):
    """
    Add calendar and known future features required by TFT.
    These features are KNOWN in advance and critical for TFT performance.
    
    Args:
        df: DataFrame with 'date' column
        
    Returns:
        DataFrame with added future-known features
    """
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calendar features (always known in the future)
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
    
    # Market-specific features
    df['is_month_start'] = (df['day'] <= 5).astype(int)  # First week trading effect
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)  # Friday effect
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)  # Monday effect
    
    # Simple holiday approximation (NYSE major holidays)
    # For production, use pandas_market_calendars library
    df['is_potential_holiday_week'] = 0
    # Week containing New Year's, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas
    holiday_months = [1, 5, 7, 9, 11, 12]
    df.loc[df['month'].isin(holiday_months), 'is_potential_holiday_week'] = 1
    
    logger.info(f"Added {13} known-future calendar features for TFT")
    
    return df


def prepare_training_data_from_historicals(symbols_data, min_samples=50, use_feature_engineering=True, feature_type='balanced'):
    """
    Prepare training data from multiple symbols with TFT-compatible features.
    Includes known-future covariates (calendar features) required by Temporal Fusion Transformer.
        
    Args:
        symbols_data: List of (symbol, historical_data) tuples
        min_samples (int): Minimum number of data points required per symbol (default: 50)
        use_feature_engineering (bool): Whether to use advanced feature engineering (default: True)
        feature_type (str): Type of features to generate ('minimal', 'balanced', 'full') (default: 'balanced')
            
    Returns:
        Tuple of (X_train, y_train, feature_names, X_future_known, future_feature_names, symbol_metrics)
    """
    logger.info(f"Preparing TFT training data with feature_type={feature_type}, use_feature_engineering={use_feature_engineering}...")
    
    all_X = []
    all_y = []
    all_X_future = []  # Known future features
    feature_names = None
    future_feature_names = None
    symbols_used = []
    symbol_metrics = []  # Track per-symbol metrics for database
    
    for symbol, historical_data in symbols_data:
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Rename columns to expected format
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date to ensure temporal order
            df = df.sort_values('date').reset_index(drop=True)
            
            # Skip symbols with insufficient data
            if len(df) < min_samples:
                logger.debug(f"Skipping {symbol}: insufficient data ({len(df)} < {min_samples})")
                symbol_metrics.append({
                    'symbol': symbol,
                    'data_points': len(df),
                    'training_samples': 0,
                    'test_samples': 0,
                    'included': False,
                    'exclusion_reason': f'Insufficient data: {len(df)} < {min_samples}',
                    'data_start_date': df['date'].min().strftime('%Y-%m-%d') if len(df) > 0 else None,
                    'data_end_date': df['date'].max().strftime('%Y-%m-%d') if len(df) > 0 else None
                })
                continue
            
            # Add known-future features (CRITICAL for TFT)
            df_with_calendar = add_known_future_features(df.copy())
            
            # Extract known-future features (these will be separate input to TFT)
            future_cols = ['dayofweek', 'day', 'month', 'quarter', 'year', 
                          'is_month_end', 'is_quarter_end', 'is_year_end',
                          'is_month_start', 'is_friday', 'is_monday', 
                          'is_potential_holiday_week']
            
            # Prepare ML data for this symbol
            X, y = prepare_data_for_ml(
                df,  # Original OHLCV data
                window_size=60,  # Increased to 60 days for TFT (recommended minimum)
                target_days=5,   # Predict 5 days ahead
                use_feature_engineering=use_feature_engineering,  # Use config value
                feature_type=feature_type  # Use config value
            )
            
            # Extract calendar features for the same time windows
            # TFT needs both past AND future calendar features
            X_future_list = []
            for i in range(len(df_with_calendar) - 60 - 5 + 1):
                # Get calendar features for the same window + future horizon
                future_window = df_with_calendar.iloc[i:i+60+5][future_cols].values
                X_future_list.append(future_window)
            
            X_future = np.array(X_future_list)
            
            if len(X) > 0 and len(X_future) > 0:
                # Ensure lengths match
                min_len = min(len(X), len(X_future))
                all_X.append(X[:min_len])
                all_y.append(y[:min_len])
                all_X_future.append(X_future[:min_len])
                symbols_used.append(symbol)
                
                # Track included symbol metrics
                symbol_metrics.append({
                    'symbol': symbol,
                    'data_points': len(df),
                    'training_samples': len(X[:min_len]),
                    'test_samples': 0,  # Will be calculated after train/test split
                    'included': True,
                    'exclusion_reason': None,
                    'data_start_date': df['date'].min().strftime('%Y-%m-%d'),
                    'data_end_date': df['date'].max().strftime('%Y-%m-%d')
                })
                
                # Store feature names from first symbol
                if feature_names is None:
                    # Create feature names based on the number of features
                    feature_names = [f"feature_{i}" for i in range(X.shape[-1])]
                    future_feature_names = future_cols
                
                logger.debug(f"Prepared {len(X[:min_len])} samples from {symbol} with calendar features")
        
        except Exception as e:
            logger.warning(f"Error preparing data for {symbol}: {str(e)}")
            symbol_metrics.append({
                'symbol': symbol,
                'data_points': 0,
                'training_samples': 0,
                'test_samples': 0,
                'included': False,
                'exclusion_reason': f'Error: {str(e)}',
                'data_start_date': None,
                'data_end_date': None
            })
    
    if not all_X:
        raise ValueError("No valid training data could be prepared from any symbol")
    
    # Concatenate all data
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    X_future_train = np.concatenate(all_X_future, axis=0)
    
    logger.info(f"Prepared TFT training dataset:")
    logger.info(f"  - Symbols used: {len(symbols_used)}")
    logger.info(f"  - Total samples: {len(X_train)}")
    logger.info(f"  - Past features per sample: {X_train.shape}")
    logger.info(f"  - Future-known features per sample: {X_future_train.shape}")
    logger.info(f"  - Symbols: {', '.join(symbols_used[:10])}{'...' if len(symbols_used) > 10 else ''}")
    logger.info(f"  - Symbol metrics tracked: {len(symbol_metrics)} total, {sum(1 for m in symbol_metrics if m['included'])} included")
    
    return X_train, y_train, feature_names, X_future_train, future_feature_names, symbol_metrics


def train_model_from_database(
    connection_string,
    model_type='auto',
    architecture_type='lstm',
    max_symbols=None,
    test_split=0.2,
    hyperparameters=None
):
    """
    Train a model using all historical data from the database.
    Enhanced for TFT with known-future covariates and configurable hyperparameters.

    Args:
        connection_string: Database connection string
        model_type: Model type ('pytorch', 'tensorflow', 'random_forest', 'tft', 'auto')
        architecture_type: Neural network architecture ('lstm', 'gru', 'transformer', 'tft')
        max_symbols: Optional limit on number of symbols to use
        test_split: Fraction of data to use for testing
        hyperparameters: Optional dictionary of hyperparameters (epochs, batch_size, learning_rate, etc.)

    Returns:
        Dictionary with training results
    """
    start_time = datetime.now()

    # Extract hyperparameters or use defaults
    if hyperparameters is None:
        hyperparameters = {}

    epochs = hyperparameters.get('epochs', 50)
    batch_size = hyperparameters.get('batch_size', 32)
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    dropout = hyperparameters.get('dropout', 0.1)
    hidden_dim = hyperparameters.get('hidden_dim', 128)
    num_layers = hyperparameters.get('num_layers', 2)
    num_heads = hyperparameters.get('num_heads', 4)
    num_attention_layers = hyperparameters.get('num_attention_layers', 2)
    
    # Extract feature engineering parameters (FIX: these were missing)
    feature_type = hyperparameters.get('feature_type', 'balanced')
    use_feature_engineering = hyperparameters.get('use_feature_engineering', True)
    
    # Map 'comprehensive' to 'full' for Python compatibility
    if feature_type == 'comprehensive':
        feature_type = 'full'

    logger.info(f"Using hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"  hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}")
    logger.info(f"  feature_type={feature_type}, use_feature_engineering={use_feature_engineering}")

    # Fetch all historical data
    symbols_data = fetch_all_historical_data(connection_string, limit=max_symbols)
    
    if not symbols_data:
        raise ValueError("No historical data found in database")
    
    # Prepare training data with TFT-compatible features
    X, y, feature_names, X_future, future_feature_names, symbol_metrics = prepare_training_data_from_historicals(
        symbols_data,
        use_feature_engineering=use_feature_engineering,
        feature_type=feature_type
    )
    
    # CRITICAL FIX: Normalize targets (percentage changes) for better training stability
    # Targets are percentage changes (e.g., 0.05 for 5%), need to be scaled
    # Use StandardScaler to normalize distribution
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    # Use RobustScaler instead of StandardScaler for better handling of outliers
    target_scaler = RobustScaler()
    
    # Fit scaler on training data only (to prevent data leakage)
    # But we need to split first
    split_idx = int(len(X) * (1 - test_split))
    
    # Split data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_future_train, X_future_test = X_future[:split_idx], X_future[split_idx:]
    
    # Check the shape of y_train and reshape if needed
    logger.info(f"Target shape before scaling: {y_train.shape}")
    
    # RobustScaler expects 2D array - reshape if 1D
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        logger.info(f"Reshaped targets to 2D: {y_train.shape}")
    
    # Fit target scaler on training targets only
    target_scaler.fit(y_train)
    
    # Transform targets
    y_train_scaled = target_scaler.transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # Flatten back to 1D if needed for model training (depends on model)
    if y_train_scaled.shape[1] == 1:
        y_train_scaled = y_train_scaled.ravel()
        y_test_scaled = y_test_scaled.ravel()
        logger.info(f"Flattened scaled targets back to 1D: {y_train_scaled.shape}")
    
    logger.info(f"Target statistics before scaling:")
    logger.info(f"  Train mean: {np.mean(y_train, axis=0)}")
    logger.info(f"  Train std: {np.std(y_train, axis=0)}")
    logger.info(f"Target statistics after scaling:")
    logger.info(f"  Train mean: {np.mean(y_train_scaled, axis=0)}")
    logger.info(f"  Train std: {np.std(y_train_scaled, axis=0)}")
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Known-future features: {future_feature_names}")
    
    # Check if TFT is requested
    is_tft = model_type == 'tft' or architecture_type == 'tft'
    
    if is_tft:
        logger.info("Training Temporal Fusion Transformer with known-future covariates...")
        # TFT requires special handling - import here to avoid circular dependency
        try:
            from stock_predictor import TFT_AVAILABLE
            if not TFT_AVAILABLE:
                logger.warning("TFT not available, falling back to LSTM")
                model_type = 'pytorch'
                architecture_type = 'lstm'
                is_tft = False
        except ImportError:
            logger.warning("Cannot import TFT, falling back to LSTM")
            model_type = 'pytorch'
            architecture_type = 'lstm'
            is_tft = False
    
    # Train the model
    logger.info(f"Training {model_type} model with {architecture_type} architecture...")
    
    # Delete old models that may have incompatible feature dimensions
    # This prevents feature mismatch errors when switching between TFT and non-TFT modes
    import os
    import shutil
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    if is_tft:
        # For TFT, we're training with different feature dimensions - delete old models
        pytorch_model = os.path.join(model_dir, 'stock_pytorch_model.pt')
        if os.path.exists(pytorch_model):
            os.remove(pytorch_model)
            logger.info("Deleted old PyTorch model to retrain with TFT-compatible features")
    
    if is_tft:
        # For TFT, we need to pass future features
        logger.info("TFT training with future-known covariates...")
        
        # Import TFT components
        from tft_integration import TFTStockPredictor, create_static_features, DEFAULT_STATIC_DIM, DEFAULT_LOOKBACK
        
        # Determine input dimensions
        if len(X_train.shape) == 3:
            input_dim = X_train.shape[2]
            seq_len = X_train.shape[1]
        else:
            input_dim = X_train.shape[1]
            seq_len = hyperparameters.get('lookback_period', DEFAULT_LOOKBACK)
            # Reshape 2D data to 3D for TFT
            X_train = X_train.reshape(-1, 1, input_dim)
            X_train = np.tile(X_train, (1, seq_len, 1))
            X_test = X_test.reshape(-1, 1, input_dim)
            X_test = np.tile(X_test, (1, seq_len, 1))
        
        # Create TFT model properly with hyperparameters
        static_dim = DEFAULT_STATIC_DIM
        model = TFTStockPredictor(
            input_dim=input_dim,
            static_dim=static_dim,
            hidden_dim=hidden_dim,
            forecast_horizons=[5, 10, 20, 30],
            num_heads=num_heads,
            num_lstm_layers=num_layers,
            dropout=dropout,
            num_attention_layers=num_attention_layers
        )
        
        # Create static features (use zeros for now, can be enhanced with real metadata)
        n_samples_train = X_train.shape[0]
        n_samples_test = X_test.shape[0]
        static_features_train = np.zeros((n_samples_train, static_dim), dtype=np.float32)
        static_features_test = np.zeros((n_samples_test, static_dim), dtype=np.float32)
        
        logger.info("Static features created - using defaults (enhance with real metadata in production)")
        
        # Prepare targets for multi-horizon
        if y_train_scaled.ndim == 1:
            y_train_scaled = np.column_stack([y_train_scaled] * len(model.forecast_horizons))
        if y_test_scaled.ndim == 1:
            y_test_scaled = np.column_stack([y_test_scaled] * len(model.forecast_horizons))
        
        # CRITICAL: Pass X_future_train to fit()
        logger.info(f"Training TFT with future features: {X_future_train.shape}")
        model.fit(
            X_past=X_train,
            X_static=static_features_train,
            y=y_train_scaled,
            future_features=X_future_train,  # ← THIS IS NOW PASSED!
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate
        )
        
        # Set feature names and model type
        model.feature_names = feature_names
        used_model_type = 'tft'
        scaler = model.scaler
    else:
        # Standard training for LSTM/GRU/Transformer/RandomForest
        # CRITICAL FIX: Use scaled targets for training
        model, scaler, used_model_type = load_or_train_model(
            X_train=X_train,
            y_train=y_train_scaled,  # Use scaled targets
            model_type=model_type,
            architecture_type=architecture_type
        )
        
        # Store target scaler in model for inverse transform during prediction
        if hasattr(model, '__dict__'):
            model.target_scaler = target_scaler
    
    # Set feature names and save the model
    if hasattr(model, 'feature_names'):
        model.feature_names = feature_names
    
    # CRITICAL: Always save the model after training, not just when feature_names exist
    if hasattr(model, 'save'):
        logger.info(f"Saving trained {used_model_type} model...")
        try:
            save_success = model.save()
            if save_success:
                logger.info(f"✓ Model saved successfully")
            else:
                logger.error(f"✗ Model save returned False - check error logs above for details")
                logger.error(f"This means the model was trained but NOT saved to disk")
                logger.error(f"The model will need to be retrained on next use")
        except Exception as save_error:
            logger.error(f"✗ Exception during model save: {save_error}", exc_info=True)
            logger.error(f"Model training succeeded but save failed")
    else:
        logger.warning(f"Model of type {used_model_type} does not have a save method")
    
    logger.info(f"Model training complete. Model type: {used_model_type}")
    
    # Evaluate on test set
    if hasattr(model, 'predict'):
        logger.info("Evaluating model on test set...")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Model type: {used_model_type}")
        
        try:
            if used_model_type == 'random_forest':
                # For Random Forest, we need to flatten the 3D data to 2D
                # Shape: (samples, window_size, features) -> (samples, window_size * features)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                logger.info(f"Flattened test data shape for RF: {X_test_flat.shape}")
                logger.info(f"Scaler expects: {scaler.n_features_in_} features")
                y_pred_scaled = model.predict(scaler.transform(X_test_flat))
            elif used_model_type == 'tft':
                # For TFT, we need to pass both temporal and static features
                logger.info(f"Making TFT predictions with X_test: {X_test.shape}, static_test: {static_features_test.shape}")
                predictions_dict = model.predict(X_test, static_features_test)
                # Extract median predictions (shape: n_samples, num_horizons)
                y_pred_scaled = predictions_dict['median_predictions']
                # For evaluation, use first horizon predictions
                if y_pred_scaled.shape[1] > 1:
                    y_pred_scaled = y_pred_scaled[:, 0]  # Use first horizon (5-day)
                logger.info(f"TFT predictions shape: {y_pred_scaled.shape}")
            else:
                # For PyTorch/TensorFlow (LSTM, GRU, Transformer), the predict method will handle the 3D data
                y_pred_scaled = model.predict(X_test)
            
            # CRITICAL FIX: Inverse transform predictions back to original scale
            # Reshape to 2D if needed for inverse transform
            if len(y_pred_scaled.shape) == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = target_scaler.inverse_transform(y_pred_scaled)
            # Flatten back to original shape
            if y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()
            
            # Also reshape y_test if needed for metrics calculation
            if len(y_test.shape) == 2 and y_test.shape[1] == 1:
                y_test = y_test.ravel()
        except ValueError as e:
            if "features" in str(e).lower():
                logger.error(f"Feature dimension mismatch during evaluation: {e}")
                logger.error(f"This usually means the model was trained with different feature dimensions.")
                logger.error(f"Skipping model evaluation and returning training results only.")
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                return {
                    'success': True,
                    'model_type': used_model_type,
                    'architecture_type': architecture_type if used_model_type in ['pytorch', 'tensorflow'] else 'n/a',
                    'symbols_count': len(symbols_data),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'training_time_seconds': training_time,
                    'has_future_covariates': True,
                    'future_covariate_count': len(future_feature_names),
                    'message': 'Model trained successfully but evaluation skipped due to feature mismatch. Model will retrain on next use.',
                    'performance': {
                        'mse': 0.0,
                        'mae': 0.0,
                        'rmse': 0.0,
                        'r2_score': 0.0
                    }
                }
            else:
                raise
        
        # Calculate metrics on original scale (not scaled targets)
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(mse)
        
        # Calculate R2 score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        logger.info(f"Evaluation metrics (on original percentage change scale):")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R2: {r2:.6f}")
        
        logger.info(f"Test Set Performance:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  R2 Score: {r2:.6f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': True,
            'model_type': used_model_type,
            'architecture_type': architecture_type if used_model_type in ['pytorch', 'tensorflow'] else 'n/a',
            'symbols_count': len(symbols_data),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'training_time_seconds': training_time,
            'has_future_covariates': True,
            'future_covariate_count': len(future_feature_names),
            'symbol_results': symbol_metrics,  # CRITICAL: Add this for database logging
            'performance': {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2)
            }
        }
    
    return {
        'success': True,
        'model_type': used_model_type,
        'symbol_results': symbol_metrics,  # CRITICAL: Add this for database logging
        'message': 'Model trained successfully but evaluation not available'
    }


def main():
    """Main entry point for the script"""
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description='Train stock prediction models from database')
    parser.add_argument('connection_string', help='Database connection string')
    parser.add_argument('output_file', help='Output JSON file for results')
    parser.add_argument('model_type', nargs='?', default='auto', help='Model type (auto, pytorch, tensorflow, random_forest)')
    parser.add_argument('architecture_type', nargs='?', default='lstm', help='Architecture type (lstm, gru, transformer, tft)')
    parser.add_argument('max_symbols', nargs='?', type=int, help='Maximum number of symbols to train on')
    parser.add_argument('--config', type=str, help='Path to training configuration JSON file')

    args = parser.parse_args()

    # Load configuration from file if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded training configuration from: {args.config}")
            logger.info(f"Configuration: {config.get('configurationName', 'Custom')}")
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {e}")
            config = None

    # Extract training parameters from config or use command line arguments
    if config:
        model_type = config.get('modelType', args.model_type)
        architecture_type = config.get('architectureType', args.architecture_type)
        max_symbols = config.get('maxSymbols', args.max_symbols)

        # Extract hyperparameters from config
        epochs = config.get('epochs', 50)
        batch_size = config.get('batchSize', 32)
        learning_rate = config.get('learningRate', 0.001)
        dropout = config.get('dropout', 0.1)
        hidden_dim = config.get('hiddenDim', 128)
        num_layers = config.get('numLayers', 2)

        # TFT-specific
        num_heads = config.get('numHeads', 4)
        num_attention_layers = config.get('numAttentionLayers', 2)

        # Training optimization
        use_early_stopping = config.get('useEarlyStopping', True)
        early_stopping_patience = config.get('earlyStoppingPatience', 10)
        use_lr_scheduler = config.get('useLearningRateScheduler', True)

        # Random Forest
        number_of_trees = config.get('numberOfTrees', 100)
        max_depth = config.get('maxDepth', 10)
        
        # Feature Engineering
        feature_type = config.get('featureType', 'balanced')
        use_feature_engineering = config.get('useFeatureEngineering', True)
        lookback_period = config.get('lookbackPeriod', 60)

        logger.info(f"Training configuration:")
        logger.info(f"  Model: {model_type}, Architecture: {architecture_type}")
        logger.info(f"  Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
        logger.info(f"  Hidden Dim: {hidden_dim}, Layers: {num_layers}, Dropout: {dropout}")
        if architecture_type == 'tft':
            logger.info(f"  TFT - Heads: {num_heads}, Attention Layers: {num_attention_layers}")
    else:
        # Use command line arguments
        model_type = args.model_type
        architecture_type = args.architecture_type
        max_symbols = args.max_symbols

        # Use default hyperparameters
        epochs = 50
        batch_size = 32
        learning_rate = 0.001
        dropout = 0.1
        hidden_dim = 128
        num_layers = 2
        num_heads = 4
        num_attention_layers = 2
        use_early_stopping = True
        early_stopping_patience = 10
        use_lr_scheduler = True
        number_of_trees = 100
        max_depth = 10
        feature_type = 'balanced'
        use_feature_engineering = True
        lookback_period = 60

    try:
        logger.info("=" * 60)
        logger.info("Starting batch model training from database")
        logger.info("=" * 60)

        # Store hyperparameters in a dict for passing to training function
        hyperparameters = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_attention_layers': num_attention_layers,
            'use_early_stopping': use_early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'use_lr_scheduler': use_lr_scheduler,
            'number_of_trees': number_of_trees,
            'max_depth': max_depth,
            'feature_type': feature_type,
            'use_feature_engineering': use_feature_engineering,
            'lookback_period': lookback_period
        }

        # Train the model with hyperparameters
        results = train_model_from_database(
            connection_string=args.connection_string,
            model_type=model_type,
            architecture_type=architecture_type,
            max_symbols=max_symbols,
            hyperparameters=hyperparameters
        )

        # Write results to output file
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Results saved to: {args.output_file}")
        logger.info("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)

        # Write error to output file
        error_result = {
            'success': False,
            'error': str(e)
        }

        with open(args.output_file, 'w') as f:
            json.dump(error_result, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
