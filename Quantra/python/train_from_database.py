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


def prepare_training_data_from_historicals(symbols_data, min_samples=50):
    """
    Prepare training data from multiple symbols.
    
    Args:
        symbols_data: List of (symbol, historical_data) tuples
        min_samples: Minimum number of data points required per symbol
        
    Returns:
        Tuple of (X_train, y_train, feature_names)
    """
    logger.info("Preparing training data from historical data...")
    
    all_X = []
    all_y = []
    feature_names = None
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
            df['date'] = pd.to_datetime(df['Date'])
            
            # Skip symbols with insufficient data
            if len(df) < min_samples:
                logger.debug(f"Skipping {symbol}: insufficient data ({len(df)} < {min_samples})")
                continue
            
            # Prepare ML data for this symbol
            X, y = prepare_data_for_ml(
                df,
                window_size=20,  # Use 20 days of history
                target_days=5,   # Predict 5 days ahead
                use_feature_engineering=False,  # Use basic features for batch training
                feature_type='balanced'
            )
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                symbols_used.append(symbol)
                
                # Store feature names from first symbol
                if feature_names is None:
                    # Create feature names based on the number of features
                    feature_names = [f"feature_{i}" for i in range(X.shape[-1])]
                
                logger.debug(f"Prepared {len(X)} samples from {symbol}")
        
        except Exception as e:
            logger.warning(f"Error preparing data for {symbol}: {str(e)}")
    
    if not all_X:
        raise ValueError("No valid training data could be prepared from any symbol")
    
    # Concatenate all data
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)
    
    logger.info(f"Prepared training dataset:")
    logger.info(f"  - Symbols used: {len(symbols_used)}")
    logger.info(f"  - Total samples: {len(X_train)}")
    logger.info(f"  - Features per sample: {X_train.shape[-1]}")
    logger.info(f"  - Symbols: {', '.join(symbols_used[:10])}{'...' if len(symbols_used) > 10 else ''}")
    
    return X_train, y_train, feature_names


def train_model_from_database(
    connection_string,
    model_type='auto',
    architecture_type='lstm',
    max_symbols=None,
    test_split=0.2
):
    """
    Train a model using all historical data from the database.
    
    Args:
        connection_string: Database connection string
        model_type: Model type ('pytorch', 'tensorflow', 'random_forest', 'auto')
        architecture_type: Neural network architecture ('lstm', 'gru', 'transformer')
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
    
    # Prepare training data
    X, y, feature_names = prepare_training_data_from_historicals(symbols_data)
    
    # Split into train/test
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Train the model
    logger.info(f"Training {model_type} model with {architecture_type} architecture...")
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
            y_pred = model.predict(scaler.transform(X_test))
        else:
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
