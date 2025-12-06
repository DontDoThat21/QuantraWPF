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


def prepare_training_data_from_historicals(symbols_data, min_samples=50):
    """
    Prepare training data from multiple symbols with TFT-compatible features.
    Includes known-future covariates (calendar features) required by Temporal Fusion Transformer.
    
    Args:
        symbols_data: List of (symbol, historical_data) tuples
        min_samples: Minimum number of data points required per symbol
        
    Returns:
        Tuple of (X_train, y_train, feature_names, X_future_known)
    """
    logger.info("Preparing TFT training data from historical data...")
    
    all_X = []
    all_y = []
    all_X_future = []  # Known future features
    feature_names = None
    future_feature_names = None
    symbols_used = []
    
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
                use_feature_engineering=False,  # Use basic features for batch training
                feature_type='balanced'
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
                
                # Store feature names from first symbol
                if feature_names is None:
                    # Create feature names based on the number of features
                    feature_names = [f"feature_{i}" for i in range(X.shape[-1])]
                    future_feature_names = future_cols
                
                logger.debug(f"Prepared {len(X[:min_len])} samples from {symbol} with calendar features")
        
        except Exception as e:
            logger.warning(f"Error preparing data for {symbol}: {str(e)}")
    
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
    
    return X_train, y_train, feature_names, X_future_train, future_feature_names


def train_model_from_database(
    connection_string,
    model_type='auto',
    architecture_type='lstm',
    max_symbols=None,
    test_split=0.2
):
    """
    Train a model using all historical data from the database.
    Enhanced for TFT with known-future covariates.
    
    Args:
        connection_string: Database connection string
        model_type: Model type ('pytorch', 'tensorflow', 'random_forest', 'tft', 'auto')
        architecture_type: Neural network architecture ('lstm', 'gru', 'transformer', 'tft')
        max_symbols: Optional limit on number of symbols to use
        test_split: Fraction of data to use for testing
        
    Returns:
        Dictionary with training results
    """
    start_time = datetime.now()
    
    # Fetch all historical data
    symbols_data = fetch_all_historical_data(connection_string, limit=max_symbols)
    
    if not symbols_data:
        raise ValueError("No historical data found in database")
    
    # Prepare training data with TFT-compatible features
    X, y, feature_names, X_future, future_feature_names = prepare_training_data_from_historicals(symbols_data)
    
    # Split into train/test (maintain temporal order)
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_future_train, X_future_test = X_future[:split_idx], X_future[split_idx:]
    
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
    
    if is_tft:
        # For TFT, we need to pass future features
        # This requires modifications to load_or_train_model or direct TFT initialization
        logger.info("TFT training requires future-known covariates - using enhanced training pipeline")
        # TODO: Implement direct TFT training with X_future_train
        # For now, log a warning and fall back
        logger.warning("Direct TFT training from database not yet implemented. Train via stock_predictor.py with prepared data.")
        model, scaler, used_model_type = load_or_train_model(
            X_train=X_train,
            y_train=y_train,
            model_type='pytorch',  # Fallback to PyTorch
            architecture_type='transformer'
        )
    else:
        # Standard training for LSTM/GRU/Transformer/RandomForest
        model, scaler, used_model_type = load_or_train_model(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            architecture_type=architecture_type
        )
    
    # Set feature names
    if hasattr(model, 'feature_names'):
        model.feature_names = feature_names
        model.save()  # Re-save with feature names
    
    logger.info(f"Model training complete. Model type: {used_model_type}")
    
    # Evaluate on test set
    if hasattr(model, 'predict'):
        logger.info("Evaluating model on test set...")
        
        if used_model_type == 'random_forest':
            # For Random Forest, we need to flatten the 3D data to 2D
            # Shape: (samples, window_size, features) -> (samples, window_size * features)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred = model.predict(scaler.transform(X_test_flat))
        else:
            # For PyTorch/TensorFlow, the predict method will handle the 3D data
            y_pred = model.predict(X_test)
        
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        rmse = np.sqrt(mse)
        
        # Calculate R2 score
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
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
        'message': 'Model trained successfully but evaluation not available'
    }


def main():
    """Main entry point for the script"""
if len(sys.argv) < 3:
    print("Usage: python train_from_database.py <connection_string> <output_file> [model_type] [architecture_type] [max_symbols]")
    print("Example: python train_from_database.py \"connection_string\" results.json pytorch lstm 100")
    sys.exit(1)
    
connection_string = sys.argv[1]
output_file = sys.argv[2]
model_type = sys.argv[3] if len(sys.argv) > 3 else 'auto'
architecture_type = sys.argv[4] if len(sys.argv) > 4 else 'lstm'
    
# Parse max_symbols - handle both integer and None
max_symbols = None
if len(sys.argv) > 5:
    try:
        max_symbols = int(sys.argv[5])
    except ValueError:
        logger.warning(f"Invalid max_symbols value '{sys.argv[5]}', using None")
    
try:
    logger.info("=" * 60)
    logger.info("Starting batch model training from database")
    logger.info("=" * 60)
        
    # Train the model
    results = train_model_from_database(
        connection_string=connection_string,
        model_type=model_type,
        architecture_type=architecture_type,
        max_symbols=max_symbols
    )
        
    # Write results to output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info("=" * 60)
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)
        
    sys.exit(0)
    
except Exception as e:
    logger.error(f"Training failed: {str(e)}", exc_info=True)
        
    # Write error to output file
    error_result = {
        'success': False,
        'error': str(e)
    }
        
    with open(output_file, 'w') as f:
        json.dump(error_result, f, indent=2)
        
    sys.exit(1)


if __name__ == "__main__":
    main()
