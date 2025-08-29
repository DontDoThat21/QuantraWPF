#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Anomaly Detection Module

This module implements various machine learning techniques to identify unusual market behaviors
and sudden price deviations to enable proactive risk management and timely decision-making.

Techniques implemented:
1. Isolation Forests for outlier detection
2. Local Outlier Factor for density-based anomaly detection
3. One-Class SVM for novelty detection
4. Autoencoder-based anomaly detection (if TensorFlow is available)
5. Ensemble methods to combine multiple anomaly detection techniques
"""

import sys
import json
import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('anomaly_detection')

# Import scikit-learn components for anomaly detection
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Try to import feature engineering module
try:
    from feature_engineering import (
        FeatureEngineer, FinancialFeatureGenerator,
        build_default_pipeline
    )
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    logger.warning("Feature Engineering module not available. Using basic feature generation.")

# Try to import TensorFlow for advanced anomaly detection
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, losses
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available for advanced anomaly detection")
except ImportError:
    logger.warning("TensorFlow is not available. Install with: pip install tensorflow")

# Model paths for persistence
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH_IF = os.path.join(MODELS_DIR, 'anomaly_isolation_forest.pkl')
MODEL_PATH_LOF = os.path.join(MODELS_DIR, 'anomaly_lof.pkl')
MODEL_PATH_OCSVM = os.path.join(MODELS_DIR, 'anomaly_ocsvm.pkl')
MODEL_PATH_AUTOENCODER = os.path.join(MODELS_DIR, 'anomaly_autoencoder')
SCALER_PATH = os.path.join(MODELS_DIR, 'anomaly_scaler.pkl')
FEATURE_PIPELINE_PATH = os.path.join(MODELS_DIR, 'anomaly_feature_pipeline.pkl')

# Anomaly score thresholds - calibrated based on typical financial time series
# These can be tuned based on specific market sensitivity requirements
DEFAULT_THRESHOLDS = {
    'isolation_forest': -0.2,  # Range is usually (-0.5, 0.5) with -0.5 being most anomalous
    'local_outlier_factor': 1.5,  # Values > 1 are inliers, < 1 are outliers
    'one_class_svm': 0.0,  # 0 is the boundary, negative values are outliers
    'autoencoder': 0.01,  # Reconstruction error threshold (depends on data scale)
    'ensemble': 0.7  # Threshold for combined anomaly score (0-1 range)
}

# Anomaly types
ANOMALY_PRICE = "PRICE_ANOMALY"
ANOMALY_VOLUME = "VOLUME_ANOMALY"
ANOMALY_VOLATILITY = "VOLATILITY_ANOMALY"
ANOMALY_PATTERN = "PATTERN_ANOMALY"
ANOMALY_CORRELATION = "CORRELATION_ANOMALY"

def create_anomaly_features(data, use_feature_engineering=True):
    """
    Create features specifically designed for anomaly detection in financial time series.
    
    Args:
        data (dict or DataFrame): Market data with keys like 'open', 'high', 'low', 'close', 'volume'
        use_feature_engineering (bool): Whether to use advanced feature engineering pipeline
    
    Returns:
        pd.DataFrame: DataFrame with features for anomaly detection
    """
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
        
    # Check if advanced feature engineering is available and requested
    if FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering:
        try:
            # Try to load saved pipeline
            if os.path.exists(FEATURE_PIPELINE_PATH):
                pipeline = FeatureEngineer.load(FEATURE_PIPELINE_PATH)
                features_df = pipeline.transform(df)
                return features_df
            else:
                # Create a new pipeline specifically for anomaly detection
                steps = [
                    # Generate features focusing on volatility and momentum
                    ('generate', {
                        'include_basic': True,
                        'include_trend': True,
                        'include_volatility': True,
                        'include_volume': 'volume' in df.columns,
                        'include_momentum': True,
                        'rolling_windows': [5, 10, 20, 50]
                    }),
                    # Select features with low variance threshold
                    ('select', {
                        'method': 'variance',
                        'params': {'threshold': 0.0001}
                    }),
                    # Robust scaling for financial data
                    ('scale', {'method': 'robust'})
                ]
                
                # Create and apply the pipeline
                pipeline = FeatureEngineer(steps=steps)
                features_df = pipeline.fit_transform(df)
                
                # Save the pipeline for future use
                pipeline.save(FEATURE_PIPELINE_PATH)
                
                return features_df
        except Exception as e:
            logger.warning(f"Error in advanced feature engineering: {str(e)}")
            logger.warning("Falling back to basic feature generation")
    
    # Fallback: Use basic feature generation
    
    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Price-based anomaly detection features
    df['price_z_score'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
    df['price_percent_change'] = df['close'].pct_change().abs()
    df['price_deviation'] = abs(df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).mean()
    
    # Volatility features for volatility spike detection
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volatility_z_score'] = (df['volatility'] - df['volatility'].rolling(window=50).mean()) / df['volatility'].rolling(window=50).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
    
    # Moving average features
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ma_ratio'] = df['sma_10'] / df['sma_50']
    
    # Volume-based anomaly features if volume is available
    if 'volume' in df.columns:
        df['volume_z_score'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['volume_price_corr'] = df['volume'].rolling(window=10).corr(df['returns'].abs())
    
    # Gap detection features
    df['overnight_gap'] = (df['open'] / df['close'].shift(1) - 1)
    df['gap_z_score'] = (df['overnight_gap'] - df['overnight_gap'].rolling(window=20).mean()) / df['overnight_gap'].rolling(window=20).std()
    
    # Pattern breakdown features
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['daily_range_z_score'] = (df['daily_range'] - df['daily_range'].rolling(window=20).mean()) / df['daily_range'].rolling(window=20).std()
    
    # Fill NaNs with 0s for the first few rows due to rolling windows
    df = df.fillna(0)
    
    return df

def load_or_create_scaler(X_train=None):
    """
    Load existing scaler or create a new one if not found.
    
    Args:
        X_train (np.array): Training data to fit scaler if needed
        
    Returns:
        sklearn.preprocessing.RobustScaler: Fitted scaler
    """
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            return scaler
    except Exception as e:
        logger.warning(f"Error loading scaler: {str(e)}")
    
    if X_train is None:
        # Create dummy data for first-time scaler creation
        X_train = np.random.rand(1000, 5)  # 5 features
    
    # Create and fit the scaler - RobustScaler is better for financial data
    scaler = RobustScaler()
    scaler.fit(X_train)
    
    # Save the scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    
    return scaler

def load_or_train_isolation_forest(X_train=None):
    """
    Load existing Isolation Forest model or train a new one if not found.
    
    Args:
        X_train (np.array): Training data to fit model if needed
        
    Returns:
        sklearn.ensemble.IsolationForest: Fitted model
    """
    try:
        if os.path.exists(MODEL_PATH_IF):
            model = joblib.load(MODEL_PATH_IF)
            return model
    except Exception as e:
        logger.warning(f"Error loading Isolation Forest model: {str(e)}")
    
    if X_train is None:
        # Create dummy data for first-time model creation
        X_train = np.random.rand(1000, 5)  # 5 features
    
    # Create and train the Isolation Forest model
    # Isolation Forest works well for financial anomalies with proper parameters
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,  # Assuming 5% of data points are anomalies
        random_state=42
    )
    
    # Train the model
    model.fit(X_train)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH_IF), exist_ok=True)
    joblib.dump(model, MODEL_PATH_IF)
    
    return model

def load_or_train_local_outlier_factor(X_train=None):
    """
    Load or fit a Local Outlier Factor model for anomaly detection.
    Note: LOF doesn't support predict on new data in sklearn, so we save the training data.
    
    Args:
        X_train (np.array): Training data
        
    Returns:
        sklearn.neighbors.LocalOutlierFactor: Fitted model
    """
    # Local Outlier Factor is used for training data reference
    # We can't persist it for predictions on new data in the same way as other models
    
    # Instead, we'll create a new model each time but save the training data
    # and remember the fit_predict results for reference
    
    model = LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        contamination=0.05,  # 5% anomaly ratio
        n_jobs=-1
    )
    
    # We'll just create the model without fitting it yet
    # It will be fit in the detection function
    
    return model

def load_or_train_one_class_svm(X_train=None):
    """
    Load existing One-Class SVM model or train a new one if not found.
    
    Args:
        X_train (np.array): Training data to fit model if needed
        
    Returns:
        sklearn.svm.OneClassSVM: Fitted model
    """
    try:
        if os.path.exists(MODEL_PATH_OCSVM):
            model = joblib.load(MODEL_PATH_OCSVM)
            return model
    except Exception as e:
        logger.warning(f"Error loading One-Class SVM model: {str(e)}")
    
    if X_train is None:
        # Create dummy data for first-time model creation
        X_train = np.random.rand(1000, 5)  # 5 features
    
    # Create and train the One-Class SVM model
    # This can be computationally intensive for large datasets
    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.05  # Equivalent to contamination parameter
    )
    
    # Train the model
    model.fit(X_train)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH_OCSVM), exist_ok=True)
    joblib.dump(model, MODEL_PATH_OCSVM)
    
    return model

def create_autoencoder(input_dim):
    """
    Create a TensorFlow Autoencoder for anomaly detection if TensorFlow is available.
    
    Args:
        input_dim (int): Input dimension (number of features)
        
    Returns:
        tf.keras.Model or None: Autoencoder model or None if TensorFlow is not available
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    # Define encoder
    encoder = keras.Sequential([
        layers.Dense(int(input_dim * 0.75), activation="relu", input_shape=(input_dim,)),
        layers.Dense(int(input_dim * 0.5), activation="relu"),
        layers.Dense(int(input_dim * 0.25), activation="relu")
    ])
    
    # Define decoder
    decoder = keras.Sequential([
        layers.Dense(int(input_dim * 0.5), activation="relu", input_shape=(int(input_dim * 0.25),)),
        layers.Dense(int(input_dim * 0.75), activation="relu"),
        layers.Dense(input_dim)
    ])
    
    # Define autoencoder
    autoencoder = keras.Sequential([encoder, decoder])
    
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def load_or_train_autoencoder(X_train=None, input_dim=None):
    """
    Load existing Autoencoder model or train a new one if not found.
    
    Args:
        X_train (np.array): Training data to fit model if needed
        input_dim (int): Input dimension for creating new model
        
    Returns:
        tf.keras.Model or None: Fitted autoencoder or None if TensorFlow is not available
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        if os.path.exists(MODEL_PATH_AUTOENCODER):
            model = keras.models.load_model(MODEL_PATH_AUTOENCODER)
            return model
    except Exception as e:
        logger.warning(f"Error loading Autoencoder model: {str(e)}")
    
    if X_train is None:
        if input_dim is None:
            input_dim = 5  # Default size
        # Create dummy data for first-time model creation
        X_train = np.random.rand(1000, input_dim)
    else:
        input_dim = X_train.shape[1]
    
    # Create the autoencoder
    model = create_autoencoder(input_dim)
    
    # Train the model
    model.fit(
        X_train, X_train,
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=0
    )
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH_AUTOENCODER), exist_ok=True)
    model.save(MODEL_PATH_AUTOENCODER)
    
    return model

def detect_anomalies_isolation_forest(X, model=None, threshold=None):
    """
    Detect anomalies using Isolation Forest algorithm.
    
    Args:
        X (np.array): Data to analyze
        model (IsolationForest): Pre-trained model (optional)
        threshold (float): Anomaly score threshold (optional)
        
    Returns:
        tuple: (anomaly_scores, anomaly_indices, anomaly_flags)
    """
    if model is None:
        model = load_or_train_isolation_forest(X)
    
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS['isolation_forest']
    
    # Get anomaly scores
    anomaly_scores = model.decision_function(X)
    
    # Isolation Forest: lower scores (negative) are more anomalous
    anomaly_flags = anomaly_scores < threshold
    anomaly_indices = np.where(anomaly_flags)[0]
    
    return anomaly_scores, anomaly_indices, anomaly_flags

def detect_anomalies_local_outlier_factor(X, model=None, threshold=None):
    """
    Detect anomalies using Local Outlier Factor algorithm.
    
    Args:
        X (np.array): Data to analyze
        model (LocalOutlierFactor): Model (optional)
        threshold (float): Anomaly score threshold (optional)
        
    Returns:
        tuple: (anomaly_scores, anomaly_indices, anomaly_flags)
    """
    if model is None:
        model = load_or_train_local_outlier_factor()
    
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS['local_outlier_factor']
    
    # LOF predict returns 1 for inliers, -1 for outliers
    # But we want to convert this into an anomaly score where higher means more anomalous
    predictions = model.fit_predict(X)
    anomaly_flags = predictions == -1
    
    # Get negative outlier factor (higher = more anomalous)
    neg_outlier_factors = -model.negative_outlier_factor_
    
    # For LOF: higher negative_outlier_factor_ means more anomalous
    anomaly_indices = np.where(neg_outlier_factors > threshold)[0]
    
    return neg_outlier_factors, anomaly_indices, anomaly_flags

def detect_anomalies_one_class_svm(X, model=None, threshold=None):
    """
    Detect anomalies using One-Class SVM algorithm.
    
    Args:
        X (np.array): Data to analyze
        model (OneClassSVM): Pre-trained model (optional)
        threshold (float): Anomaly score threshold (optional)
        
    Returns:
        tuple: (anomaly_scores, anomaly_indices, anomaly_flags)
    """
    if model is None:
        model = load_or_train_one_class_svm(X)
    
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS['one_class_svm']
    
    # Get decision scores (lower = more anomalous)
    anomaly_scores = model.decision_function(X)
    
    # One-Class SVM: negative values are outliers
    anomaly_flags = anomaly_scores < threshold
    anomaly_indices = np.where(anomaly_flags)[0]
    
    return anomaly_scores, anomaly_indices, anomaly_flags

def detect_anomalies_autoencoder(X, model=None, threshold=None):
    """
    Detect anomalies using Autoencoder reconstruction error.
    
    Args:
        X (np.array): Data to analyze
        model (tf.keras.Model): Pre-trained autoencoder (optional)
        threshold (float): Anomaly score threshold for reconstruction error (optional)
        
    Returns:
        tuple: (anomaly_scores, anomaly_indices, anomaly_flags) or (None, None, None) if TensorFlow not available
    """
    if not TENSORFLOW_AVAILABLE:
        return None, None, None
    
    if model is None:
        model = load_or_train_autoencoder(X, X.shape[1])
        
    if model is None:  # If still None (TF not available)
        return None, None, None
    
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS['autoencoder']
    
    # Predict reconstructions
    reconstructions = model.predict(X)
    
    # Compute reconstruction error (MSE)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    
    # Higher reconstruction error = more anomalous
    anomaly_flags = mse > threshold
    anomaly_indices = np.where(anomaly_flags)[0]
    
    return mse, anomaly_indices, anomaly_flags

def classify_anomaly_type(features_df, anomaly_indices):
    """
    Classify detected anomalies into specific types based on feature values.
    
    Args:
        features_df (pd.DataFrame): DataFrame with market features
        anomaly_indices (list): Indices of detected anomalies
        
    Returns:
        dict: Dictionary mapping anomaly indices to their types
    """
    anomaly_types = {}
    
    for idx in anomaly_indices:
        # Skip if index is out of range
        if idx >= len(features_df) or idx < 0:
            continue
            
        # Extract relevant features for this data point
        row = features_df.iloc[idx]
        anomaly_reason = []
        
        # Check for price anomalies
        if 'price_z_score' in row and abs(row['price_z_score']) > 2.5:
            anomaly_reason.append(ANOMALY_PRICE)
        elif 'price_percent_change' in row and row['price_percent_change'] > 0.03:  # 3% change
            anomaly_reason.append(ANOMALY_PRICE)
            
        # Check for volume anomalies
        if 'volume_z_score' in row and abs(row['volume_z_score']) > 3:
            anomaly_reason.append(ANOMALY_VOLUME)
        elif 'volume_ratio' in row and row['volume_ratio'] > 2:  # Volume doubled
            anomaly_reason.append(ANOMALY_VOLUME)
            
        # Check for volatility anomalies
        if 'volatility_z_score' in row and abs(row['volatility_z_score']) > 2.5:
            anomaly_reason.append(ANOMALY_VOLATILITY)
        elif 'volatility_ratio' in row and row['volatility_ratio'] > 1.5:
            anomaly_reason.append(ANOMALY_VOLATILITY)
            
        # Check for pattern anomalies
        if 'daily_range_z_score' in row and abs(row['daily_range_z_score']) > 2.5:
            anomaly_reason.append(ANOMALY_PATTERN)
        elif 'gap_z_score' in row and abs(row['gap_z_score']) > 3:
            anomaly_reason.append(ANOMALY_PATTERN)
            
        # Check for correlation anomalies
        if 'volume_price_corr' in row and abs(row['volume_price_corr']) > 0.8:
            anomaly_reason.append(ANOMALY_CORRELATION)
            
        # Use default if no specific reason identified
        if not anomaly_reason:
            anomaly_reason = [ANOMALY_PRICE]
            
        anomaly_types[int(idx)] = anomaly_reason
        
    return anomaly_types

def calculate_anomaly_severity(anomaly_scores, normalized=True):
    """
    Calculate anomaly severity based on anomaly scores.
    
    Args:
        anomaly_scores (np.array): Array of anomaly scores
        normalized (bool): Whether to normalize to 0-1 range
        
    Returns:
        np.array: Severity scores
    """
    # Ensure scores is a numpy array
    scores = np.asarray(anomaly_scores)
    
    if normalized:
        # Scale to 0-1 range where 1 is most severe
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Check if min equals max to avoid division by zero
        if min_score == max_score:
            return np.ones_like(scores)
            
        # Map to 0-1 range, invert if needed so 1 is most anomalous
        if min_score < 0 and max_score > 0:
            # For algorithms like Isolation Forest where negative is anomalous
            return (scores - max_score) / (min_score - max_score)
        else:
            # For algorithms where higher score is more anomalous
            return (scores - min_score) / (max_score - min_score)
    else:
        return scores

def ensemble_anomaly_detection(results, threshold=None):
    """
    Combine multiple anomaly detection results using an ensemble approach.
    
    Args:
        results (dict): Dictionary with results from different algorithms
        threshold (float): Ensemble anomaly score threshold
        
    Returns:
        dict: Combined anomaly detection result
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLDS['ensemble']
    
    # Extract data length from first result
    data_length = 0
    for model_name, model_result in results.items():
        if model_result is not None and model_result['scores'] is not None:
            data_length = len(model_result['scores'])
            break
    
    if data_length == 0:
        return {
            'anomalies': [],
            'anomaly_indices': [],
            'anomaly_types': {},
            'combined_scores': []
        }
    
    # Initialize combined scores array
    combined_scores = np.zeros(data_length)
    vote_counts = np.zeros(data_length)
    
    # Combine normalized scores from each algorithm
    for model_name, model_result in results.items():
        if model_result is None or model_result['scores'] is None:
            continue
            
        # Get normalized scores where 1 = most anomalous
        normalized_scores = calculate_anomaly_severity(model_result['scores'], normalized=True)
        
        # Add scores to combined score
        combined_scores += normalized_scores
        
        # Count algorithms that flagged each point as anomaly
        for idx in model_result['indices']:
            if 0 <= idx < data_length:
                vote_counts[idx] += 1
    
    # Normalize the combined score
    algorithm_count = sum(1 for r in results.values() if r is not None and r['scores'] is not None)
    if algorithm_count > 0:
        combined_scores /= algorithm_count
    
    # Boost score based on agreement among algorithms
    if algorithm_count > 1:
        agreement_factor = vote_counts / algorithm_count
        combined_scores = 0.6 * combined_scores + 0.4 * agreement_factor
    
    # Find anomaly indices based on threshold
    anomaly_indices = np.where(combined_scores > threshold)[0]
    
    return {
        'anomalies': anomaly_indices.tolist(),
        'anomaly_indices': anomaly_indices.tolist(),
        'combined_scores': combined_scores.tolist(),
    }

def generate_anomaly_insights(features_df, anomaly_indices, anomaly_types):
    """
    Generate insights and recommendations for each detected anomaly.
    
    Args:
        features_df (pd.DataFrame): DataFrame with market features
        anomaly_indices (list): Indices of detected anomalies
        anomaly_types (dict): Dictionary mapping anomaly indices to their types
        
    Returns:
        dict: Dictionary of insights for each anomaly index
    """
    insights = {}
    
    for idx in anomaly_indices:
        # Skip if index is out of range
        if idx >= len(features_df) or idx < 0 or idx not in anomaly_types:
            continue
            
        row = features_df.iloc[idx]
        anomaly_type_list = anomaly_types[idx]
        
        insight = {
            "anomaly_types": anomaly_type_list,
            "description": [],
            "potential_causes": [],
            "suggested_actions": []
        }
        
        # Generate insights based on anomaly types
        for anomaly_type in anomaly_type_list:
            if anomaly_type == ANOMALY_PRICE:
                # Price anomaly insights
                is_positive = row.get('returns', 0) > 0
                change = abs(row.get('price_percent_change', 0)) * 100  # Convert to percentage
                
                insight["description"].append(
                    f"Unusual {'upward' if is_positive else 'downward'} price movement of approximately {change:.2f}%"
                )
                
                insight["potential_causes"].extend([
                    "News or event impact",
                    "Large institutional order",
                    "Market overreaction",
                    "Technical breakdown/breakout"
                ])
                
                insight["suggested_actions"].extend([
                    "Review recent news for the asset",
                    "Check for earnings announcements or major events",
                    "Consider adjusting position sizing",
                    "Implement hedging strategies"
                ])
                
            elif anomaly_type == ANOMALY_VOLUME:
                # Volume anomaly insights
                volume_increase = row.get('volume_ratio', 1)
                
                insight["description"].append(
                    f"Unusual trading volume {volume_increase:.2f}x above normal levels"
                )
                
                insight["potential_causes"].extend([
                    "Institutional accumulation/distribution",
                    "News-driven trading activity",
                    "Market maker activity",
                    "Potential trend reversal signal"
                ])
                
                insight["suggested_actions"].extend([
                    "Monitor price action following volume spike",
                    "Check for block trades or unusual options activity",
                    "Look for confirmation from other technical indicators",
                    "Consider increasing position monitoring frequency"
                ])
                
            elif anomaly_type == ANOMALY_VOLATILITY:
                # Volatility anomaly insights
                vol_z = row.get('volatility_z_score', 0)
                
                insight["description"].append(
                    f"Abnormal market volatility detected ({vol_z:.2f} standard deviations from mean)"
                )
                
                insight["potential_causes"].extend([
                    "Market uncertainty",
                    "Approaching major event/announcement",
                    "Liquidity issues",
                    "Market regime change"
                ])
                
                insight["suggested_actions"].extend([
                    "Reduce position sizes",
                    "Widen stop loss orders",
                    "Consider options strategies for protection",
                    "Prepare for potential trend changes"
                ])
                
            elif anomaly_type == ANOMALY_PATTERN:
                # Pattern anomaly insights
                insight["description"].append(
                    "Unusual market pattern detected that breaks from historical behavior"
                )
                
                insight["potential_causes"].extend([
                    "Failed technical pattern",
                    "Change in market structure",
                    "Unusual order flow",
                    "Algorithmic trading influence"
                ])
                
                insight["suggested_actions"].extend([
                    "Reassess technical analysis assumptions",
                    "Look for confirmation across multiple timeframes",
                    "Consider alternate trading strategies",
                    "Wait for clear pattern re-establishment"
                ])
                
            elif anomaly_type == ANOMALY_CORRELATION:
                # Correlation anomaly insights
                insight["description"].append(
                    "Unusual correlation between price and volume detected"
                )
                
                insight["potential_causes"].extend([
                    "Smart money divergence",
                    "Institutional repositioning",
                    "Market inefficiency",
                    "Potential reversal signal"
                ])
                
                insight["suggested_actions"].extend([
                    "Check related assets for similar patterns",
                    "Monitor market breadth indicators",
                    "Consider mean-reversion strategies",
                    "Evaluate sector rotation possibilities"
                ])
        
        # Remove duplicates
        insight["description"] = list(dict.fromkeys(insight["description"]))
        insight["potential_causes"] = list(dict.fromkeys(insight["potential_causes"]))
        insight["suggested_actions"] = list(dict.fromkeys(insight["suggested_actions"]))
        
        insights[idx] = insight
    
    return insights

def detect_market_anomalies(market_data, use_feature_engineering=True, sensitivity=1.0):
    """
    Main function to detect market anomalies based on provided market data.
    
    Args:
        market_data (dict): Dictionary containing OHLCV price data
                          with keys 'open', 'high', 'low', 'close', 'volume'
        use_feature_engineering (bool): Whether to use advanced feature engineering
        sensitivity (float): Sensitivity multiplier for anomaly thresholds (higher = more sensitive)
        
    Returns:
        dict: Market anomaly detection results
    """
    try:
        # Check if we have sufficient data
        if not market_data or 'close' not in market_data or len(market_data['close']) < 50:
            return {
                'anomalies_detected': False,
                'error': 'Insufficient data for anomaly detection',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Create features for anomaly detection
        features_df = create_anomaly_features(market_data, use_feature_engineering)
        
        # Select key features for anomaly detection
        key_features = [
            'price_z_score', 'price_percent_change', 'price_deviation',
            'volatility', 'volatility_z_score', 'volatility_ratio',
            'ma_ratio'
        ]
        
        # Add volume features if available
        if 'volume' in market_data:
            volume_features = ['volume_z_score', 'volume_ratio']
            key_features.extend([f for f in volume_features if f in features_df.columns])
        
        # Use only available features
        key_features = [f for f in key_features if f in features_df.columns]
        
        # Extract the relevant features
        X = features_df[key_features].values
        
        # Scale the features
        scaler = load_or_create_scaler(X)
        X_scaled = scaler.transform(X)
        
        # Adjust thresholds based on sensitivity
        adjusted_thresholds = {
            k: v * (2.0 - sensitivity) for k, v in DEFAULT_THRESHOLDS.items()
        }
        
        # Apply each anomaly detection algorithm
        results = {}
        
        # 1. Isolation Forest
        try:
            model_if = load_or_train_isolation_forest(X_scaled)
            scores_if, indices_if, flags_if = detect_anomalies_isolation_forest(
                X_scaled, model_if, threshold=adjusted_thresholds['isolation_forest']
            )
            results['isolation_forest'] = {
                'scores': scores_if,
                'indices': indices_if,
                'flags': flags_if
            }
        except Exception as e:
            logger.warning(f"Error in Isolation Forest detection: {str(e)}")
            results['isolation_forest'] = None
        
        # 2. Local Outlier Factor
        try:
            model_lof = load_or_train_local_outlier_factor()
            scores_lof, indices_lof, flags_lof = detect_anomalies_local_outlier_factor(
                X_scaled, model_lof, threshold=adjusted_thresholds['local_outlier_factor']
            )
            results['local_outlier_factor'] = {
                'scores': scores_lof,
                'indices': indices_lof,
                'flags': flags_lof
            }
        except Exception as e:
            logger.warning(f"Error in Local Outlier Factor detection: {str(e)}")
            results['local_outlier_factor'] = None
        
        # 3. One-Class SVM
        try:
            model_ocsvm = load_or_train_one_class_svm(X_scaled)
            scores_ocsvm, indices_ocsvm, flags_ocsvm = detect_anomalies_one_class_svm(
                X_scaled, model_ocsvm, threshold=adjusted_thresholds['one_class_svm']
            )
            results['one_class_svm'] = {
                'scores': scores_ocsvm,
                'indices': indices_ocsvm,
                'flags': flags_ocsvm
            }
        except Exception as e:
            logger.warning(f"Error in One-Class SVM detection: {str(e)}")
            results['one_class_svm'] = None
        
        # 4. Autoencoder (if TensorFlow is available)
        if TENSORFLOW_AVAILABLE:
            try:
                model_ae = load_or_train_autoencoder(X_scaled, X_scaled.shape[1])
                scores_ae, indices_ae, flags_ae = detect_anomalies_autoencoder(
                    X_scaled, model_ae, threshold=adjusted_thresholds['autoencoder']
                )
                results['autoencoder'] = {
                    'scores': scores_ae,
                    'indices': indices_ae,
                    'flags': flags_ae
                }
            except Exception as e:
                logger.warning(f"Error in Autoencoder detection: {str(e)}")
                results['autoencoder'] = None
        
        # Combine results using ensemble approach
        ensemble_result = ensemble_anomaly_detection(results, threshold=adjusted_thresholds['ensemble'])
        
        # Get anomaly indices from ensemble
        anomaly_indices = ensemble_result['anomaly_indices']
        
        # Classify anomaly types
        anomaly_types = classify_anomaly_type(features_df, anomaly_indices)
        
        # Generate insights for detected anomalies
        insights = generate_anomaly_insights(features_df, anomaly_indices, anomaly_types)
        
        # Format the output
        output = {
            'anomalies_detected': len(anomaly_indices) > 0,
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_types': anomaly_types,
            'anomaly_insights': insights,
            'detection_methods': list(results.keys()),
            'feature_engineering_used': FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Include most recent anomalies information for easy access
        recent_anomalies = []
        for idx in anomaly_indices[-5:] if anomaly_indices else []:
            if idx in anomaly_types:
                recent_anomalies.append({
                    'index': int(idx),
                    'date': features_df.index[idx].strftime('%Y-%m-%d') if hasattr(features_df.index[idx], 'strftime') else str(features_df.index[idx]),
                    'types': anomaly_types[idx],
                    'insights': insights.get(idx, {})
                })
        
        output['recent_anomalies'] = recent_anomalies
        
        return output
        
    except Exception as e:
        logger.error(f"Error in detect_market_anomalies: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'anomalies_detected': False,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    """Main function to handle command-line operation"""
    if len(sys.argv) != 3:
        print("Usage: python anomaly_detection.py input_file output_file", file=sys.stderr)
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        # Read input JSON containing market data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Check if data contains market data and options
        market_data = data
        use_feature_engineering = True
        sensitivity = 1.0
        
        # If data is wrapped in an object with additional options
        if isinstance(data, dict) and 'data' in data:
            market_data = data['data']
            use_feature_engineering = data.get('use_feature_engineering', True)
            sensitivity = data.get('sensitivity', 1.0)
        
        # Detect market anomalies
        result = detect_market_anomalies(market_data, use_feature_engineering, sensitivity)
        
        # Write output JSON
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()