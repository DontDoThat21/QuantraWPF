#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Market Regime Detection Module

This module implements various machine learning techniques to identify market regimes
(trending or ranging) to enhance trading strategy adaptability.

Techniques implemented:
1. Hidden Markov Models (HMM) for regime state transitions
2. Clustering techniques for unsupervised market state detection
3. Supervised classification for predicting market regimes
4. Ensemble methods to combine multiple model predictions
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from hmmlearn import hmm

# Try to import feature engineering module
try:
    from feature_engineering import (
        FeatureEngineer, FinancialFeatureGenerator,
        build_default_pipeline
    )
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

# Model paths for persistence
MODEL_PATH_HMM = 'python/regime_hmm_model.pkl'
MODEL_PATH_CLUSTER = 'python/regime_cluster_model.pkl'
MODEL_PATH_CLASSIFIER = 'python/regime_classifier_model.pkl'
SCALER_PATH = 'python/regime_scaler.pkl'
FEATURE_PIPELINE_PATH = 'python/regime_feature_pipeline.pkl'

# Regime definitions
REGIME_TRENDING_UP = "TRENDING_UP"
REGIME_TRENDING_DOWN = "TRENDING_DOWN"
REGIME_RANGING = "RANGING"

def create_regime_features(data, use_feature_engineering=True):
    """
    Create technical indicators and features specifically for regime detection.
    
    Args:
        data (dict): Dictionary containing price data with keys like 'open', 'high', 'low', 'close', 'volume'
                     Each value should be a list of historical data.
        use_feature_engineering (bool): Whether to use the advanced feature engineering pipeline
    
    Returns:
        pd.DataFrame: DataFrame with features for regime detection
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
                # Create a new pipeline specifically for regime detection
                steps = [
                    # Generate features focusing on trend and volatility
                    ('generate', {
                        'include_basic': True,
                        'include_trend': True,
                        'include_volatility': True,
                        'include_volume': 'volume' in df.columns,
                        'include_momentum': True,
                        'rolling_windows': [5, 10, 20, 50, 200]
                    }),
                    # Select features with low variance threshold
                    ('select', {
                        'method': 'variance',
                        'params': {'threshold': 0.0001}
                    }),
                    # Add custom interaction terms for regime detection
                    ('interact', {
                        'interaction_type': 'financial'
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
            print(f"Error in advanced feature engineering: {str(e)}", file=sys.stderr)
            print("Falling back to basic feature generation", file=sys.stderr)
    
    # Fallback: Use basic feature generation
    
    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility features
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['volatility_change'] = df['volatility'].pct_change(5)
    
    # Trend strength indicators
    df['adx'] = calculate_adx(df)
    df['trend_strength'] = df['adx'] / 100.0  # Normalized ADX
    
    # Moving averages and their relationships
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ma_ratio_20_50'] = df['sma_20'] / df['sma_50']
    df['ma_ratio_50_200'] = df['sma_50'] / df['sma_200']
    
    # Price momentum features
    df['roc_5'] = df['close'].pct_change(periods=5)
    df['roc_20'] = df['close'].pct_change(periods=20)
    df['roc_ratio'] = df['roc_5'] / (df['roc_20'] + 1e-10)
    
    # Bollinger Bands features for ranging detection
    df['bb_width'] = calculate_bbands_width(df, window=20, num_std=2)
    df['bb_width_change'] = df['bb_width'].pct_change(5)
    df['close_to_bb_center'] = abs(df['close'] - df['sma_20']) / (df['bb_width'] * df['sma_20'] + 1e-10)
    
    # Volume-based features
    if 'volume' in df.columns:
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20']
        df['volume_trend'] = df['volume'].rolling(window=10).apply(lambda x: 1 if x[-1] > x[0] else -1, raw=True)
    
    # Drop NaN values created by rolling windows
    df = df.dropna()
    
    return df

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX) for trend strength"""
    # Calculate True Range (TR)
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df.drop(['tr1', 'tr2', 'tr3'], axis=1, inplace=True)
    
    # Calculate Directional Movement
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Calculate smoothed averages
    df['tr14'] = df['tr'].rolling(window=period).mean()
    df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['tr14'])
    df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['tr14'])
    
    # Calculate Directional Index
    df['dx'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'] + 1e-10)
    
    # Calculate ADX
    adx = df['dx'].rolling(window=period).mean()
    
    return adx

def calculate_bbands_width(df, window=20, num_std=2):
    """Calculate Bollinger Bands width normalized by price"""
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = df['sma_20'] + (rolling_std * num_std)
    lower_band = df['sma_20'] - (rolling_std * num_std)
    return (upper_band - lower_band) / df['sma_20']

def load_or_train_hmm_model(X_train=None):
    """Load existing HMM model or train a new one if not found."""
    try:
        if os.path.exists(MODEL_PATH_HMM):
            model = joblib.load(MODEL_PATH_HMM)
            return model
    except:
        pass
    
    if X_train is None:
        # Create dummy data for first-time model creation
        X_train = np.random.rand(1000, 5)  # 5 features

    # Create and train the HMM model
    model = hmm.GaussianHMM(
        n_components=3,  # 3 states: trending up, trending down, ranging
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH_HMM), exist_ok=True)
    joblib.dump(model, MODEL_PATH_HMM)
    
    return model

def load_or_train_cluster_model(X_train=None):
    """Load existing cluster model or train a new one if not found."""
    try:
        if os.path.exists(MODEL_PATH_CLUSTER):
            model = joblib.load(MODEL_PATH_CLUSTER)
            return model
    except:
        pass
    
    if X_train is None:
        # Create dummy data for first-time model creation
        X_train = np.random.rand(1000, 5)  # 5 features

    # Create and train the cluster model
    model = KMeans(
        n_clusters=3,  # 3 clusters: trending up, trending down, ranging
        random_state=42
    )
    
    # Train the model
    model.fit(X_train)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH_CLUSTER), exist_ok=True)
    joblib.dump(model, MODEL_PATH_CLUSTER)
    
    return model

def load_or_train_classifier_model(X_train=None, y_train=None):
    """Load existing classifier model or train a new one if not found."""
    try:
        if os.path.exists(MODEL_PATH_CLASSIFIER):
            model = joblib.load(MODEL_PATH_CLASSIFIER)
            return model
    except:
        pass
    
    if X_train is None or y_train is None:
        # Create dummy data for first-time model creation
        X_train = np.random.rand(1000, 5)  # 5 features
        y_train = np.random.randint(0, 3, size=1000)  # 3 classes
    
    # Create and train the classifier model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH_CLASSIFIER), exist_ok=True)
    joblib.dump(model, MODEL_PATH_CLASSIFIER)
    
    return model

def load_or_create_scaler(X_train=None):
    """Load existing scaler or create a new one if not found."""
    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            return scaler
    except:
        pass
    
    if X_train is None:
        # Create dummy data for first-time scaler creation
        X_train = np.random.rand(1000, 5)  # 5 features
    
    # Create and fit the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save the scaler
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    
    return scaler

def detect_regime_hmm(features_df, key_features=None):
    """
    Detect market regimes using Hidden Markov Models
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        key_features (list): List of feature names to use for regime detection
        
    Returns:
        dict: Dictionary containing regime detection results
    """
    if key_features is None:
        key_features = ['returns', 'volatility', 'adx', 'ma_ratio_20_50', 'bb_width']
    
    # Extract the relevant features
    X = features_df[key_features].values
    
    # Scale the features
    scaler = load_or_create_scaler(X)
    X_scaled = scaler.transform(X)
    
    # Load or train the HMM model
    model = load_or_train_hmm_model(X_scaled)
    
    # Predict the hidden states
    hidden_states = model.predict(X_scaled)
    
    # Get the most recent state
    recent_state = hidden_states[-1]
    
    # Calculate state probabilities
    state_probs = model.predict_proba(X_scaled[-1:])
    
    # Match states to regimes based on state characteristics
    # Calculate average returns for each state
    state_returns = {}
    for state in range(model.n_components):
        state_returns[state] = features_df['returns'].iloc[hidden_states == state].mean()
    
    # Trending up: state with highest positive returns
    trending_up_state = max(state_returns.items(), key=lambda x: x[1])[0]
    # Trending down: state with lowest negative returns
    trending_down_state = min(state_returns.items(), key=lambda x: x[1])[0]
    # Ranging: the remaining state
    ranging_state = list(set(range(model.n_components)) - {trending_up_state, trending_down_state})[0]
    
    # Map the recent state to a regime
    if recent_state == trending_up_state:
        regime = REGIME_TRENDING_UP
        confidence = float(state_probs[0][trending_up_state])
    elif recent_state == trending_down_state:
        regime = REGIME_TRENDING_DOWN
        confidence = float(state_probs[0][trending_down_state])
    else:  # ranging_state
        regime = REGIME_RANGING
        confidence = float(state_probs[0][ranging_state])
    
    return {
        'regime': regime,
        'confidence': confidence,
        'model_type': 'hmm'
    }

def detect_regime_clustering(features_df, key_features=None):
    """
    Detect market regimes using clustering techniques
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        key_features (list): List of feature names to use for regime detection
        
    Returns:
        dict: Dictionary containing regime detection results
    """
    if key_features is None:
        key_features = ['returns', 'volatility', 'adx', 'ma_ratio_20_50', 'bb_width']
    
    # Extract the relevant features
    X = features_df[key_features].values
    
    # Scale the features
    scaler = load_or_create_scaler(X)
    X_scaled = scaler.transform(X)
    
    # Load or train the cluster model
    model = load_or_train_cluster_model(X_scaled)
    
    # Predict the clusters
    clusters = model.predict(X_scaled)
    
    # Get the most recent cluster
    recent_cluster = clusters[-1]
    
    # Match clusters to regimes based on characteristics
    # Calculate average returns and volatility for each cluster
    cluster_returns = {}
    cluster_volatility = {}
    for cluster in range(model.n_clusters):
        cluster_returns[cluster] = features_df['returns'].iloc[clusters == cluster].mean()
        cluster_volatility[cluster] = features_df['volatility'].iloc[clusters == cluster].mean()
    
    # Trending up: cluster with highest returns
    trending_up_cluster = max(cluster_returns.items(), key=lambda x: x[1])[0]
    # Trending down: cluster with lowest returns
    trending_down_cluster = min(cluster_returns.items(), key=lambda x: x[1])[0]
    # Ranging: cluster with moderate returns but high volatility
    ranging_candidates = list(set(range(model.n_clusters)) - {trending_up_cluster, trending_down_cluster})
    if ranging_candidates:
        ranging_cluster = ranging_candidates[0]
    else:
        # Fallback if we have fewer clusters than expected
        ranging_cluster = max(cluster_volatility.items(), key=lambda x: x[1])[0]
    
    # Map the recent cluster to a regime
    if recent_cluster == trending_up_cluster:
        regime = REGIME_TRENDING_UP
    elif recent_cluster == trending_down_cluster:
        regime = REGIME_TRENDING_DOWN
    else:  # ranging_cluster
        regime = REGIME_RANGING
    
    # Calculate distance to cluster center for confidence
    center_distance = np.linalg.norm(X_scaled[-1] - model.cluster_centers_[recent_cluster])
    confidence = 1.0 / (1.0 + center_distance)  # Convert distance to confidence score (0-1)
    
    return {
        'regime': regime,
        'confidence': float(confidence),
        'model_type': 'clustering'
    }

def detect_regime_classification(features_df, key_features=None):
    """
    Detect market regimes using supervised classification
    
    Args:
        features_df (pd.DataFrame): DataFrame with features
        key_features (list): List of feature names to use for regime detection
        
    Returns:
        dict: Dictionary containing regime detection results
    """
    if key_features is None:
        key_features = ['returns', 'volatility', 'adx', 'trend_strength', 
                        'ma_ratio_20_50', 'ma_ratio_50_200', 'bb_width', 'roc_5', 'roc_20']
    
    # Extract the relevant features
    X = features_df[key_features].values
    
    # Scale the features
    scaler = load_or_create_scaler(X)
    X_scaled = scaler.transform(X)
    
    # Generate labels for initial training - this would ideally come from historical labels
    # Here we create synthetic labels based on rules for demonstration
    y_train = np.zeros(len(features_df))
    
    # Trending up: positive returns and low BB width
    trending_up_mask = (features_df['returns'] > 0) & (features_df['adx'] > 25) & (features_df['ma_ratio_20_50'] > 1)
    y_train[trending_up_mask] = 0  # Class 0: Trending Up
    
    # Trending down: negative returns and low BB width
    trending_down_mask = (features_df['returns'] < 0) & (features_df['adx'] > 25) & (features_df['ma_ratio_20_50'] < 1)
    y_train[trending_down_mask] = 1  # Class 1: Trending Down
    
    # Ranging: low ADX and high BB width
    ranging_mask = (features_df['adx'] < 25) | (features_df['bb_width'] > features_df['bb_width'].median())
    y_train[ranging_mask] = 2  # Class 2: Ranging
    
    # Load or train the classifier model - only use training data before the most recent point
    X_train, y_train = X_scaled[:-1], y_train[:-1]
    
    # Only train if we have enough data
    if len(X_train) > 10:
        model = load_or_train_classifier_model(X_train, y_train)
        
        # Predict the regime for the most recent data point
        regime_class = model.predict(X_scaled[-1:])
        
        # Get prediction probabilities for confidence
        proba = model.predict_proba(X_scaled[-1:])
        confidence = float(proba[0][regime_class[0]])
        
        # Map class to regime
        if regime_class[0] == 0:
            regime = REGIME_TRENDING_UP
        elif regime_class[0] == 1:
            regime = REGIME_TRENDING_DOWN
        else:  # class 2
            regime = REGIME_RANGING
    else:
        # Not enough data for supervised learning, fallback to a simple rule-based approach
        if features_df['returns'].iloc[-1] > 0 and features_df['adx'].iloc[-1] > 25:
            regime = REGIME_TRENDING_UP
            confidence = 0.6
        elif features_df['returns'].iloc[-1] < 0 and features_df['adx'].iloc[-1] > 25:
            regime = REGIME_TRENDING_DOWN
            confidence = 0.6
        else:
            regime = REGIME_RANGING
            confidence = 0.6
    
    return {
        'regime': regime,
        'confidence': confidence,
        'model_type': 'classification'
    }

def ensemble_regime_detection(hmm_result, cluster_result, classification_result):
    """
    Combine results from multiple regime detection models
    
    Args:
        hmm_result (dict): Result from HMM model
        cluster_result (dict): Result from clustering model
        classification_result (dict): Result from classification model
        
    Returns:
        dict: Combined regime detection result
    """
    # Create a score for each regime
    scores = {
        REGIME_TRENDING_UP: 0,
        REGIME_TRENDING_DOWN: 0,
        REGIME_RANGING: 0
    }
    
    # Weight each model's contribution by its confidence
    for result in [hmm_result, cluster_result, classification_result]:
        scores[result['regime']] += result['confidence']
    
    # Find the regime with the highest weighted score
    final_regime = max(scores.items(), key=lambda x: x[1])[0]
    
    # Calculate overall confidence based on agreement between models
    if hmm_result['regime'] == cluster_result['regime'] == classification_result['regime']:
        # Perfect agreement
        overall_confidence = 0.9
    elif hmm_result['regime'] == cluster_result['regime'] or hmm_result['regime'] == classification_result['regime'] or cluster_result['regime'] == classification_result['regime']:
        # Partial agreement (2 out of 3)
        overall_confidence = 0.7
    else:
        # No agreement
        overall_confidence = 0.5
    
    # Adjust confidence by weighted score
    max_score = scores[final_regime]
    total_score = sum(scores.values())
    score_ratio = max_score / (total_score + 1e-10)
    overall_confidence *= score_ratio
    
    # Generate suggested trading approaches based on regime
    trading_approaches = []
    if final_regime == REGIME_TRENDING_UP:
        trading_approaches = [
            "Use trend-following strategies",
            "Consider longer holding periods",
            "Look for pullbacks as buying opportunities",
            "Use trailing stops to protect profits",
            "Focus on momentum indicators"
        ]
    elif final_regime == REGIME_TRENDING_DOWN:
        trading_approaches = [
            "Consider short positions or inverse ETFs",
            "Use trend-following strategies with downside focus",
            "Tighter stop losses",
            "Reduce position sizes",
            "Look for relief rallies as shorting opportunities"
        ]
    else:  # RANGING
        trading_approaches = [
            "Use mean-reversion strategies",
            "Trade within identified range boundaries",
            "Implement oscillator-based strategies",
            "Shorter holding periods",
            "Avoid trend-following strategies"
        ]
    
    # Create detailed result
    return {
        'regime': final_regime,
        'confidence': float(overall_confidence),
        'modelResults': {
            'hmm': {
                'regime': hmm_result['regime'],
                'confidence': float(hmm_result['confidence'])
            },
            'clustering': {
                'regime': cluster_result['regime'],
                'confidence': float(cluster_result['confidence'])
            },
            'classification': {
                'regime': classification_result['regime'],
                'confidence': float(classification_result['confidence'])
            }
        },
        'tradingApproaches': trading_approaches,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def detect_market_regime(market_data, use_feature_engineering=True):
    """
    Main function to detect market regime based on provided market data.
    
    Args:
        market_data (dict): Dictionary containing OHLCV price data
                           with keys 'open', 'high', 'low', 'close', 'volume'
                           and values as lists or arrays
        use_feature_engineering (bool): Whether to use the advanced feature engineering
    
    Returns:
        dict: Market regime detection results
    """
    try:
        # Check if we have the necessary data
        if not market_data or 'close' not in market_data or len(market_data['close']) < 200:
            return {
                'regime': REGIME_RANGING,
                'confidence': 0.5,
                'error': 'Insufficient data for regime detection',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Create features for regime detection
        features_df = create_regime_features(market_data, use_feature_engineering)
        
        # Apply each model for regime detection
        hmm_result = detect_regime_hmm(features_df)
        cluster_result = detect_regime_clustering(features_df)
        classification_result = detect_regime_classification(features_df)
        
        # Combine results using ensemble approach
        final_result = ensemble_regime_detection(hmm_result, cluster_result, classification_result)
        
        # Add feature engineering information
        final_result['feature_engineering_used'] = (
            FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering
        )
        final_result['feature_count'] = len(features_df.columns)
        
        return final_result
        
    except Exception as e:
        print(f"Error in detect_market_regime: {str(e)}", file=sys.stderr)
        return {
            'regime': REGIME_RANGING,
            'confidence': 0.5,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def main():
    if len(sys.argv) != 3:
        print("Usage: python market_regime_detection.py input_file output_file", file=sys.stderr)
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
        
        # If data is wrapped in an object with additional options
        if isinstance(data, dict) and 'data' in data:
            market_data = data['data']
            use_feature_engineering = data.get('use_feature_engineering', True)
        
        # Detect market regime with specified options
        result = detect_market_regime(market_data, use_feature_engineering)
        
        # Add feature engineering availability info
        if 'feature_engineering_used' not in result:
            result['feature_engineering_used'] = FEATURE_ENGINEERING_AVAILABLE and use_feature_engineering
            
        # Write output JSON
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def create_rl_regime_detector():
    """
    Create a regime detector object compatible with the RL module.
    
    Returns:
        RegimeDetector: Object with predict_regime method for RL integration
    """
    class RegimeDetector:
        def predict_regime(self, data_dict):
            """
            Predict market regime for RL environment.
            
            Args:
                data_dict: Dictionary with keys 'close', 'volume', 'high', 'low'
                          Each value should be a list or array of recent values
            
            Returns:
                str: Predicted regime (TRENDING_UP, TRENDING_DOWN, RANGING)
            """
            try:
                # Convert data_dict to the format expected by detect_market_regime
                market_data = {
                    'open': data_dict.get('close', [100]),  # Use close as open if not available
                    'high': data_dict.get('high', data_dict.get('close', [100])),
                    'low': data_dict.get('low', data_dict.get('close', [100])),
                    'close': data_dict.get('close', [100]),
                    'volume': data_dict.get('volume', [1000000])
                }
                
                # Ensure we have enough data points
                min_length = max(len(v) for v in market_data.values()) if market_data.values() else 1
                for key in market_data:
                    if len(market_data[key]) < min_length:
                        # Pad with the last available value
                        last_val = market_data[key][-1] if market_data[key] else 100
                        market_data[key] = market_data[key] + [last_val] * (min_length - len(market_data[key]))
                
                # Detect regime using our existing function
                result = detect_market_regime(market_data)
                return result.get('regime', REGIME_RANGING)
                
            except Exception as e:
                # Fallback to RANGING if detection fails
                return REGIME_RANGING
    
    return RegimeDetector()

def load_regime_model(model_path):
    """
    Load a pre-trained regime detection model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        RegimeDetector: Loaded regime detector object
    """
    try:
        # For now, just return a new detector
        # In practice, this would load saved model weights
        return create_rl_regime_detector()
    except Exception:
        return create_rl_regime_detector()

if __name__ == "__main__":
    main()