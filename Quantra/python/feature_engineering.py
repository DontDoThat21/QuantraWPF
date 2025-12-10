#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Feature Engineering Pipeline

This module provides tools for automating feature engineering tasks for financial machine learning models,
including feature selection, dimensionality reduction, interaction term generation, and feature scaling.

Classes:
    FeatureSelector: Class for automated feature selection using various methods
    FeatureTransformer: Class for automated feature transformation and generation
    FeatureEngineer: Main class that orchestrates the feature engineering pipeline
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import logging
import warnings

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel, RFE,
    f_regression, mutual_info_regression, f_classif, mutual_info_classif,
    VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature_engineering')


class FinancialFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate financial features from OHLCV data.
    
    This transformer creates common technical indicators used in financial analysis,
    serving as the first step in the feature engineering pipeline.
    """
    
    def __init__(self, 
                 include_basic=True,
                 include_trend=True, 
                 include_volatility=True, 
                 include_volume=True,
                 include_momentum=True,
                 rolling_windows=None):
        """
        Initialize the financial feature generator.
        
        Args:
            include_basic (bool): Include basic price and return features
            include_trend (bool): Include trend indicators (moving averages, etc.)
            include_volatility (bool): Include volatility indicators
            include_volume (bool): Include volume-based features
            include_momentum (bool): Include momentum indicators
            rolling_windows (list): List of window sizes for rolling calculations
                                   If None, uses default [5, 10, 20, 50, 200]
        """
        self.include_basic = include_basic
        self.include_trend = include_trend
        self.include_volatility = include_volatility
        self.include_volume = include_volume
        self.include_momentum = include_momentum
        self.rolling_windows = rolling_windows or [5, 10, 20, 50, 200]
        self.feature_names_ = []
        
    def fit(self, X, y=None):
        """
        Fit the transformer (no fitting required, just returns self).
        
        Args:
            X (pd.DataFrame): Input data with OHLCV columns
            y: Ignored
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Transform the input data by generating financial features.
        
        Args:
            X (pd.DataFrame): Input data with OHLCV columns
            
        Returns:
            pd.DataFrame: DataFrame with generated features
        """
        # Make a copy of the input data
        df = X.copy()
        
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            required_columns.append('volume')
        
        # Generate features
        features = []
        
        # Basic features
        if self.include_basic:
            features.extend(self._generate_basic_features(df))
            
        # Trend features
        if self.include_trend:
            features.extend(self._generate_trend_features(df))
            
        # Volatility features
        if self.include_volatility:
            features.extend(self._generate_volatility_features(df))
            
        # Volume features
        if self.include_volume and 'volume' in df.columns:
            features.extend(self._generate_volume_features(df))
            
        # Momentum features
        if self.include_momentum:
            features.extend(self._generate_momentum_features(df))
        
        # Combine all features
        result = pd.concat(features, axis=1)
        
        # Store feature names
        self.feature_names_ = list(result.columns)
        
        # Drop NaN values created by rolling windows
        result = result.dropna()
        
        return result
    
    def _generate_basic_features(self, df):
        """Generate basic price and return features."""
        result = pd.DataFrame(index=df.index)
        
        # Returns
        result['returns'] = df['close'].pct_change()
        result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price difference
        result['close_to_open'] = df['close'] / df['open'] - 1
        result['high_to_low'] = df['high'] / df['low'] - 1
        
        # Range
        result['daily_range'] = (df['high'] - df['low']) / df['close']
        
        return [result]
    
    def _generate_trend_features(self, df):
        """Generate trend indicators."""
        results = []
        
        # Moving averages
        ma_df = pd.DataFrame(index=df.index)
        for window in self.rolling_windows:
            ma_df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            ma_df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            
        results.append(ma_df)
        
        # Moving average ratios
        ma_ratio_df = pd.DataFrame(index=df.index)
        for i, fast_window in enumerate(self.rolling_windows[:-1]):
            for slow_window in self.rolling_windows[i+1:]:
                ma_ratio_df[f'ma_ratio_{fast_window}_{slow_window}'] = (
                    df['close'].rolling(window=fast_window).mean() / 
                    df['close'].rolling(window=slow_window).mean()
                )
                
        results.append(ma_ratio_df)
        
        # MACD
        macd_df = pd.DataFrame(index=df.index)
        macd_df['macd'] = (
            df['close'].ewm(span=12, adjust=False).mean() - 
            df['close'].ewm(span=26, adjust=False).mean()
        )
        macd_df['macd_signal'] = macd_df['macd'].ewm(span=9, adjust=False).mean()
        macd_df['macd_hist'] = macd_df['macd'] - macd_df['macd_signal']
        
        results.append(macd_df)
        
        return results
    
    def _generate_volatility_features(self, df):
        """Generate volatility indicators."""
        results = []
        
        # Historical volatility
        vol_df = pd.DataFrame(index=df.index)
        for window in self.rolling_windows:
            vol_df[f'volatility_{window}'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        results.append(vol_df)
        
        # Bollinger Bands
        bb_df = pd.DataFrame(index=df.index)
        for window in [20]:  # Standard Bollinger Bands use 20-day window
            # Calculate middle band (20-day SMA)
            middle_band = df['close'].rolling(window=window).mean()
            # Calculate standard deviation
            std = df['close'].rolling(window=window).std()
            # Upper band (SMA + 2*STD)
            bb_df[f'bb_upper_{window}'] = middle_band + (std * 2)
            # Lower band (SMA - 2*STD)
            bb_df[f'bb_lower_{window}'] = middle_band - (std * 2)
            # Bandwidth
            bb_df[f'bb_width_{window}'] = (bb_df[f'bb_upper_{window}'] - bb_df[f'bb_lower_{window}']) / middle_band
            # %B (current price position within bands)
            bb_df[f'bb_pct_{window}'] = (df['close'] - bb_df[f'bb_lower_{window}']) / (bb_df[f'bb_upper_{window}'] - bb_df[f'bb_lower_{window}'])
        
        results.append(bb_df)
        
        # Average True Range (ATR)
        atr_df = pd.DataFrame(index=df.index)
        
        # Calculate True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR for different windows
        for window in [14, 20]:
            atr_df[f'atr_{window}'] = true_range.rolling(window=window).mean()
            # Normalized ATR
            atr_df[f'natr_{window}'] = atr_df[f'atr_{window}'] / df['close']
        
        results.append(atr_df)
        
        return results
    
    def _generate_volume_features(self, df):
        """Generate volume-based indicators."""
        if 'volume' not in df.columns:
            return []
            
        results = []
        
        # Volume moving averages
        vol_ma_df = pd.DataFrame(index=df.index)
        for window in self.rolling_windows:
            vol_ma_df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            vol_ma_df[f'volume_ema_{window}'] = df['volume'].ewm(span=window, adjust=False).mean()
            
        results.append(vol_ma_df)
        
        # Volume ratios
        vol_ratio_df = pd.DataFrame(index=df.index)
        vol_ratio_df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
        for window in self.rolling_windows:
            vol_ratio_df[f'volume_ratio_{window}'] = df['volume'] / df['volume'].rolling(window=window).mean()
            
        results.append(vol_ratio_df)
        
        # On-Balance Volume (OBV)
        obv_df = pd.DataFrame(index=df.index)
        obv = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        obv_df['obv'] = obv
        
        # OBV moving average
        obv_df['obv_ma'] = obv.rolling(window=20).mean()
        
        results.append(obv_df)
        
        # Price-Volume trend
        pv_df = pd.DataFrame(index=df.index)
        pv_df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        results.append(pv_df)
        
        return results
    
    def _generate_momentum_features(self, df):
        """Generate momentum indicators."""
        results = []
        
        # Rate of Change
        roc_df = pd.DataFrame(index=df.index)
        for period in [5, 10, 20, 60]:
            roc_df[f'roc_{period}'] = df['close'].pct_change(periods=period)
            
        results.append(roc_df)
        
        # Relative Strength Index (RSI)
        rsi_df = pd.DataFrame(index=df.index)
        
        # Calculate gains and losses
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate RSI for different periods
        for period in [6, 14, 20]:
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            rsi_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
        results.append(rsi_df)
        
        # Stochastic Oscillator
        stoch_df = pd.DataFrame(index=df.index)
        for period in [14]:  # Standard period
            # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            stoch_df[f'stoch_k_{period}'] = (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
            # %D = 3-day SMA of %K
            stoch_df[f'stoch_d_{period}'] = stoch_df[f'stoch_k_{period}'].rolling(window=3).mean()
            
        results.append(stoch_df)
        
        return results


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Automated feature selection using various methods.
    
    This transformer performs feature selection using statistical methods,
    model-based methods, or correlation analysis.
    """
    
    def __init__(self, method='variance', params=None):
        """
        Initialize the feature selector.
        
        Args:
            method (str): Feature selection method to use
                - 'variance': Remove low-variance features
                - 'correlation': Remove highly correlated features
                - 'kbest': Select K best features
                - 'percentile': Select features by percentile
                - 'model': Use a model to select features
                - 'rfe': Recursive Feature Elimination
            params (dict): Parameters for the selected method
                - For 'variance': {'threshold': float}
                - For 'correlation': {'threshold': float}
                - For 'kbest': {'k': int, 'score_func': callable}
                - For 'percentile': {'percentile': int, 'score_func': callable}
                - For 'model': {'estimator': object, 'threshold': float}
                - For 'rfe': {'estimator': object, 'n_features_to_select': int}
        """
        self.method = method
        self.params = params or {}
        self.selected_features_ = None
        self.support_ = None
        self.selector_ = None
        
    def fit(self, X, y=None):
        """
        Fit the feature selector to the data.
        
        Args:
            X (pd.DataFrame): Input features
            y (array-like, optional): Target variable (required for some methods)
            
        Returns:
            self: The fitted feature selector
        """
        # Convert X to pandas DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Store original feature names
        self.feature_names_ = list(X.columns)
        
        # Apply the selected feature selection method
        if self.method == 'variance':
            self._fit_variance_threshold(X)
        elif self.method == 'correlation':
            self._fit_correlation_threshold(X)
        elif self.method == 'kbest':
            self._fit_kbest(X, y)
        elif self.method == 'percentile':
            self._fit_percentile(X, y)
        elif self.method == 'model':
            self._fit_model_based(X, y)
        elif self.method == 'rfe':
            self._fit_rfe(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        
        # Store selected feature names
        self.selected_features_ = [f for f, s in zip(self.feature_names_, self.support_) if s]
        
        return self
    
    def transform(self, X):
        """
        Transform the input data by selecting only the chosen features.
        
        Args:
            X (pd.DataFrame or array-like): Input features
            
        Returns:
            pd.DataFrame: Transformed data with selected features only
        """
        # Convert X to pandas DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # For correlation method which doesn't use a selector
        if self.method == 'correlation':
            return X[self.selected_features_]
        
        # For other methods that use a scikit-learn selector
        return pd.DataFrame(
            self.selector_.transform(X),
            index=X.index,
            columns=self.selected_features_
        )
    
    def _fit_variance_threshold(self, X):
        """Fit a variance threshold selector."""
        threshold = self.params.get('threshold', 0.01)
        self.selector_ = VarianceThreshold(threshold=threshold)
        self.selector_.fit(X)
        self.support_ = self.selector_.get_support()
        
    def _fit_correlation_threshold(self, X):
        """Remove highly correlated features."""
        threshold = self.params.get('threshold', 0.9)
        corr_matrix = X.corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Create support mask
        self.support_ = [col not in to_drop for col in X.columns]
        
        # No selector_ for correlation method, we'll handle transform manually
        
    def _fit_kbest(self, X, y):
        """Fit a SelectKBest selector."""
        if y is None:
            raise ValueError("Target variable y is required for k-best feature selection")
            
        k = self.params.get('k', min(10, X.shape[1]))
        
        # Default score function based on whether y is continuous or categorical
        if self.params.get('score_func') is not None:
            score_func = self.params.get('score_func')
        else:
            if self._is_classification_target(y):
                score_func = f_classif
            else:
                score_func = f_regression
                
        self.selector_ = SelectKBest(score_func=score_func, k=k)
        self.selector_.fit(X, y)
        self.support_ = self.selector_.get_support()
        
    def _fit_percentile(self, X, y):
        """Fit a SelectPercentile selector."""
        if y is None:
            raise ValueError("Target variable y is required for percentile feature selection")
            
        percentile = self.params.get('percentile', 10)
        
        # Default score function based on whether y is continuous or categorical
        if self.params.get('score_func') is not None:
            score_func = self.params.get('score_func')
        else:
            if self._is_classification_target(y):
                score_func = f_classif
            else:
                score_func = f_regression
                
        self.selector_ = SelectPercentile(score_func=score_func, percentile=percentile)
        self.selector_.fit(X, y)
        self.support_ = self.selector_.get_support()
        
    def _fit_model_based(self, X, y):
        """Fit a model-based feature selector."""
        if y is None:
            raise ValueError("Target variable y is required for model-based feature selection")
            
        # Get estimator or create a default one
        estimator = self.params.get('estimator')
        if estimator is None:
            if self._is_classification_target(y):
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
        threshold = self.params.get('threshold', 'mean')
        
        self.selector_ = SelectFromModel(estimator=estimator, threshold=threshold)
        self.selector_.fit(X, y)
        self.support_ = self.selector_.get_support()
        
    def _fit_rfe(self, X, y):
        """Fit a Recursive Feature Elimination selector."""
        if y is None:
            raise ValueError("Target variable y is required for RFE feature selection")
            
        # Get estimator or create a default one
        estimator = self.params.get('estimator')
        if estimator is None:
            if self._is_classification_target(y):
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
                
        n_features = self.params.get('n_features_to_select', X.shape[1] // 2)
        
        self.selector_ = RFE(estimator=estimator, n_features_to_select=n_features)
        self.selector_.fit(X, y)
        self.support_ = self.selector_.get_support()
        
    def _is_classification_target(self, y):
        """Determine if the target is for classification or regression."""
        # Simple heuristic: if y has few unique values, it's probably classification
        unique_values = np.unique(y)
        
        # If y has <= 10 unique values and they're all integers, assume classification
        if len(unique_values) <= 10 and np.all(np.equal(np.mod(unique_values, 1), 0)):
            return True
        return False

    def get_feature_importances(self):
        """
        Get feature importances or scores from the selector.
        
        Returns:
            dict: Dictionary mapping feature names to importance scores
        """
        if self.method == 'correlation':
            return {f: 1.0 for f in self.selected_features_}
        
        if self.method in ['kbest', 'percentile']:
            scores = self.selector_.scores_
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
            return dict(zip(self.feature_names_, normalized_scores))
        
        if self.method == 'model' and hasattr(self.selector_.estimator_, 'feature_importances_'):
            return dict(zip(self.feature_names_, self.selector_.estimator_.feature_importances_))
        
        if self.method == 'rfe':
            # For RFE, use ranking_ (lower is better)
            rankings = self.selector_.ranking_
            max_rank = np.max(rankings)
            normalized_scores = 1 - (rankings - 1) / max_rank
            return dict(zip(self.feature_names_, normalized_scores))
        
        # Default: just return 1.0 for selected features, 0.0 for others
        return {f: (1.0 if s else 0.0) for f, s in zip(self.feature_names_, self.support_)}


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Reduce dimensionality of feature space using PCA, t-SNE or other methods.
    """
    
    def __init__(self, method='pca', n_components=None, random_state=42, **kwargs):
        """
        Initialize the dimensionality reducer.
        
        Args:
            method (str): Dimensionality reduction method ('pca' or 'tsne')
            n_components (int): Number of components to retain
            random_state (int): Random state for reproducibility
            **kwargs: Additional parameters for the reduction method
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self.reducer_ = None
        
    def fit(self, X, y=None):
        """
        Fit the dimensionality reducer to the data.
        
        Args:
            X (array-like): Input features
            y: Ignored
            
        Returns:
            self: The fitted reducer
        """
        # Determine n_components if not specified
        if self.n_components is None:
            self.n_components = min(X.shape[1], 10)  # Default to min of 10 or number of features
        
        # Create and fit the reducer
        if self.method == 'pca':
            self.reducer_ = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )
        elif self.method == 'tsne':
            self.reducer_ = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")
        
        self.reducer_.fit(X)
        
        # Generate feature names for transformed dimensions
        self.feature_names_ = [f'dim_{i+1}' for i in range(self.n_components)]
        
        return self
    
    def transform(self, X):
        """
        Transform the input data to the reduced dimension space.
        
        Args:
            X (array-like): Input features
            
        Returns:
            pd.DataFrame: Transformed data with reduced dimensions
        """
        # For t-SNE, fit_transform is the only option
        if self.method == 'tsne':
            X_reduced = self.reducer_.fit_transform(X)
        else:
            X_reduced = self.reducer_.transform(X)
            
        # Return as DataFrame with feature names
        return pd.DataFrame(
            X_reduced,
            index=X.index if hasattr(X, 'index') else None,
            columns=self.feature_names_
        )
    
    def get_explained_variance(self):
        """
        Get explained variance for PCA.
        
        Returns:
            dict: Dictionary with explained variance information (PCA only)
        """
        if self.method != 'pca':
            return {}
            
        return {
            'explained_variance_ratio': list(self.reducer_.explained_variance_ratio_),
            'cumulative_variance_ratio': list(np.cumsum(self.reducer_.explained_variance_ratio_)),
            'singular_values': list(self.reducer_.singular_values_)
        }


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generate interaction features including polynomial features and custom financial interactions.
    """
    
    def __init__(self, 
                 interaction_type='polynomial',
                 degree=2,
                 interaction_only=False,
                 include_bias=False,
                 custom_interactions=None):
        """
        Initialize the interaction feature generator.
        
        Args:
            interaction_type (str): Type of interactions to generate
                - 'polynomial': Polynomial features
                - 'custom': Custom defined interactions
                - 'financial': Common financial interactions
            degree (int): Degree of polynomial features (for 'polynomial' type)
            interaction_only (bool): If True, only include interaction terms, not pure polynomials
            include_bias (bool): Include a bias column (all 1s)
            custom_interactions (list): List of tuples defining custom interactions
                                        Each tuple contains (feature1, feature2, function)
        """
        self.interaction_type = interaction_type
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.custom_interactions = custom_interactions
        self.feature_names_ = []
        
    def fit(self, X, y=None):
        """
        Fit the interaction feature generator.
        
        Args:
            X (pd.DataFrame): Input features
            y: Ignored
            
        Returns:
            self: The fitted interaction generator
        """
        # Store original feature names
        self.input_features_ = list(X.columns)
        
        # If using polynomial features, initialize the transformer
        if self.interaction_type == 'polynomial':
            self.poly_ = PolynomialFeatures(
                degree=self.degree,
                interaction_only=self.interaction_only,
                include_bias=self.include_bias
            )
            self.poly_.fit(X)
            self.feature_names_ = self.poly_.get_feature_names_out(self.input_features_)
            
        elif self.interaction_type == 'financial':
            # Define standard financial interactions here
            self.financial_interactions_ = [
                # Volatility and returns
                ('returns', 'volatility', lambda x, y: x * y, 'returns_volatility'),
                # Price / Moving average ratios
                ('close', 'sma_20', lambda x, y: x / y, 'price_to_ma'),
                # Volume and price changes
                ('returns', 'volume_ratio', lambda x, y: x * y, 'vol_price_momentum')
            ]
            
            # Check if required features exist
            self.valid_interactions_ = []
            for feat1, feat2, func, name in self.financial_interactions_:
                if feat1 in X.columns and feat2 in X.columns:
                    self.valid_interactions_.append((feat1, feat2, func, name))
                    
            # Update feature names
            self.feature_names_ = [i[3] for i in self.valid_interactions_]
            
        else:  # custom
            if self.custom_interactions is None:
                self.custom_interactions = []
                
            # Check if required features exist
            self.valid_interactions_ = []
            for interaction in self.custom_interactions:
                feat1, feat2, func, name = interaction
                if feat1 in X.columns and feat2 in X.columns:
                    self.valid_interactions_.append((feat1, feat2, func, name))
                    
            # Update feature names
            self.feature_names_ = [i[3] for i in self.valid_interactions_]
            
        return self
    
    def transform(self, X):
        """
        Transform the input data by generating interaction features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: DataFrame with generated interaction features
        """
        if self.interaction_type == 'polynomial':
            # Use scikit-learn's PolynomialFeatures
            X_poly = self.poly_.transform(X)
            return pd.DataFrame(
                X_poly,
                index=X.index,
                columns=self.feature_names_
            )
        else:
            # Apply custom or financial interactions
            result = pd.DataFrame(index=X.index)
            
            for feat1, feat2, func, name in self.valid_interactions_:
                result[name] = func(X[feat1], X[feat2])
                
            return result


class FeatureEngineer:
    """
    Main class for orchestrating the automated feature engineering pipeline.
    
    This class combines feature generation, interaction terms, selection, 
    and dimensionality reduction into a customizable pipeline.
    """
    
    def __init__(self, steps=None, verbose=False, random_state=42):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            steps (list): List of pipeline steps as tuples (name, params)
            verbose (bool): Whether to print detailed information
            random_state (int): Random state for reproducibility
        """
        self.verbose = verbose
        self.random_state = random_state
        self.pipeline_steps = []
        self.pipeline_ = None
        self.feature_names_ = []
        
        # Initialize with default steps if none provided
        if steps is None:
            self.steps = [
                ('generate', {'include_basic': True, 'include_trend': True, 'include_volatility': True}),
                ('select', {'method': 'variance', 'params': {'threshold': 0.001}}),
                ('scale', {'method': 'standard'})
            ]
        else:
            self.steps = steps
            
        # Build the pipeline
        self._build_pipeline()
        
    def _build_pipeline(self):
        """Build the scikit-learn pipeline from the specified steps."""
        self.pipeline_steps = []
        
        for name, params in self.steps:
            # Extract the base step name (handle names with suffixes like 'select_kbest')
            base_name = name.split('_')[0]
            
            if base_name == 'generate':
                # Feature generation step
                step = (name, FinancialFeatureGenerator(**params))
            elif base_name == 'select':
                # Feature selection step (handles 'select', 'select_kbest', 'select_model', etc.)
                step = (name, FeatureSelector(**params))
            elif base_name == 'reduce':
                # Dimensionality reduction step
                step = (name, DimensionalityReducer(**params))
            elif base_name == 'interact':
                # Interaction feature generator
                step = (name, InteractionFeatureGenerator(**params))
            elif base_name == 'scale':
                # Feature scaling
                if params.get('method') == 'standard':
                    step = (name, StandardScaler())
                elif params.get('method') == 'minmax':
                    feature_range = params.get('feature_range', (0, 1))
                    step = (name, MinMaxScaler(feature_range=feature_range))
                elif params.get('method') == 'robust':
                    step = (name, RobustScaler())
                else:
                    raise ValueError(f"Unknown scaling method: {params.get('method')}")
            else:
                raise ValueError(f"Unknown pipeline step: {name}")
                
            self.pipeline_steps.append(step)
            
        # Create the pipeline
        self.pipeline_ = Pipeline(self.pipeline_steps)
        
    def fit_transform(self, X, y=None):
        """
        Fit the pipeline and transform the input data.
        
        Args:
            X (pd.DataFrame): Input data with OHLCV columns
            y (array-like, optional): Target variable for supervised selection methods
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.verbose:
            logger.info(f"Input data shape: {X.shape}")
            
        # Fit and transform the data
        X_transformed = self.pipeline_.fit_transform(X, y)
        
        # Get feature names from the last step that has feature_names_ attribute
        for step_name, step_transformer in reversed(self.pipeline_.named_steps.items()):
            if hasattr(step_transformer, 'feature_names_'):
                self.feature_names_ = step_transformer.feature_names_
                break
        
        # Convert to DataFrame if it's not already
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=self.feature_names_
            )
        
        if self.verbose:
            logger.info(f"Transformed data shape: {X_transformed.shape}")
            logger.info(f"Features: {self.feature_names_}")
            
        return X_transformed
    
    def transform(self, X):
        """
        Transform new data using the fitted pipeline.
        
        Args:
            X (pd.DataFrame): Input data with OHLCV columns
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Transform the data
        X_transformed = self.pipeline_.transform(X)
        
        # Convert to DataFrame if it's not already
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(
                X_transformed,
                index=X.index,
                columns=self.feature_names_
            )
            
        return X_transformed
    
    def save(self, filepath):
        """
        Save the fitted pipeline to disk.
        
        Args:
            filepath (str): Path to save the pipeline
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pipeline_data = {
            'pipeline': self.pipeline_,
            'feature_names': self.feature_names_,
            'steps': self.steps
        }
        joblib.dump(pipeline_data, filepath)
        
    @classmethod
    def load(cls, filepath):
        """
        Load a saved pipeline from disk.
        
        Args:
            filepath (str): Path to the saved pipeline
            
        Returns:
            FeatureEngineer: Loaded feature engineering pipeline
        """
        pipeline_data = joblib.load(filepath)
        
        # Create a new instance
        instance = cls(steps=pipeline_data['steps'])
        
        # Replace the pipeline with the loaded one
        instance.pipeline_ = pipeline_data['pipeline']
        instance.feature_names_ = pipeline_data['feature_names']
        
        return instance
    
    def get_feature_importance(self, X, y):
        """
        Get feature importance using a random forest model.
        
        Args:
            X (pd.DataFrame): Feature data
            y (array-like): Target variable
            
        Returns:
            dict: Feature importance scores
        """
        # Check if y is for classification or regression
        unique_values = np.unique(y)
        if len(unique_values) <= 10 and np.all(np.equal(np.mod(unique_values, 1), 0)):
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            
        # Fit the model
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        
        # Return as dictionary
        return dict(zip(X.columns, importance))
    
    def get_pipeline_step(self, name):
        """
        Get a specific step from the pipeline.
        
        Args:
            name (str): Name of the step
            
        Returns:
            object: The transformer object for the step
        """
        return self.pipeline_.named_steps.get(name)
    
    def evaluate_features(self, X, y, cv=5):
        """
        Evaluate feature importance and model performance.
        
        Args:
            X (pd.DataFrame): Feature data
            y (array-like): Target variable
            cv (int): Cross-validation folds
            
        Returns:
            dict: Evaluation results
        """
        # Check if y is for classification or regression
        unique_values = np.unique(y)
        if len(unique_values) <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
        # Fit model to get feature importance
        model.fit(X, y)
        importance = model.feature_importances_
        
        # Return evaluation results
        return {
            'cv_scores': list(cv_scores),
            'mean_score': float(cv_scores.mean()),
            'std_score': float(cv_scores.std()),
            'feature_importance': dict(zip(X.columns, importance))
        }
    
    def visualize_feature_importance(self, importance_dict, top_n=20, figsize=(10, 8), save_path=None):
        """
        Visualize feature importance as a bar chart.
        
        Args:
            importance_dict (dict): Feature importance dictionary
            top_n (int): Number of top features to show
            figsize (tuple): Figure size (width, height)
            save_path (str): Path to save the figure (optional)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Sort importance values
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N features
        top_features = sorted_importance[:top_n]
        features = [x[0] for x in top_features]
        importances = [x[1] for x in top_features]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(features)), importances, align='center')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Display the highest importance at the top
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if a path is provided
        if save_path:
            plt.savefig(save_path)
            
        return fig


def build_default_pipeline(target_type='regression', feature_type='full'):
    """
    Build a default feature engineering pipeline based on target type.
    
    Args:
        target_type (str): Type of target variable ('regression' or 'classification')
        feature_type (str): Type of features to generate ('minimal', 'balanced', 'full')
        
    Returns:
        FeatureEngineer: Configured feature engineering pipeline
    """
    # Configure generator based on feature type
    if feature_type == 'minimal':
        generator_params = {
            'include_basic': True,
            'include_trend': True,
            'include_volatility': False,
            'include_volume': True,
            'include_momentum': False,
            'rolling_windows': [5, 20]
        }
    elif feature_type == 'balanced':
        generator_params = {
            'include_basic': True,
            'include_trend': True,
            'include_volatility': True,
            'include_volume': True,
            'include_momentum': True,
            'rolling_windows': [5, 10, 20]
        }
    else:  # full
        generator_params = {
            'include_basic': True,
            'include_trend': True,
            'include_volatility': True,
            'include_volume': True,
            'include_momentum': True,
            'rolling_windows': [5, 10, 20, 50, 200]
        }
        
    # Create steps list
    steps = [
        ('generate', generator_params),
        ('select', {'method': 'variance', 'params': {'threshold': 0.0001}}),
        ('scale', {'method': 'robust'})  # Robust scaling works well for financial data
    ]
    
    # Add selection step based on target type
    if target_type == 'classification':
        # For classification, add a model-based selection using RandomForest
        steps.insert(2, ('select_model', {
            'method': 'model',
            'params': {
                'estimator': RandomForestClassifier(n_estimators=100, random_state=42),
                'threshold': 'mean'
            }
        }))
    else:
        # For regression, add a k-best selection
        steps.insert(2, ('select_kbest', {
            'method': 'kbest',
            'params': {
                'k': 20,
                'score_func': f_regression
            }
        }))
        
    # Create the pipeline
    return FeatureEngineer(steps=steps, verbose=True)


def create_train_test_features(data, target_col='close', target_shift=5, test_size=0.2, pipeline=None):
    """
    Create train and test datasets with engineered features.
    
    Args:
        data (pd.DataFrame): Raw OHLCV data
        target_col (str): Column to use as target
        target_shift (int): Number of periods to shift target for prediction
        test_size (float): Proportion of data to use for testing
        pipeline (FeatureEngineer): Optional pre-configured pipeline
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, pipeline)
    """
    # Create target variable (future price change)
    target = data[target_col].pct_change(periods=target_shift).shift(-target_shift)
    
    # Split into train and test
    train_size = int(len(data) * (1 - test_size))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    train_target = target.iloc[:train_size]
    test_target = target.iloc[train_size:]
    
    # Create and fit the pipeline if not provided
    if pipeline is None:
        pipeline = build_default_pipeline()
        
    # Apply the pipeline to train and test data
    X_train = pipeline.fit_transform(train_data, train_target)
    X_test = pipeline.transform(test_data)
    
    # Drop NaN values
    train_mask = ~np.isnan(train_target)
    test_mask = ~np.isnan(test_target)
    
    X_train = X_train[train_mask]
    y_train = train_target[train_mask]
    
    X_test = X_test[test_mask]
    y_test = test_target[test_mask]
    
    return X_train, X_test, y_train, y_test, pipeline


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Automated Financial Feature Engineering")
    parser.add_argument("input_file", help="Input CSV file with OHLCV data")
    parser.add_argument("--output_file", help="Output file for transformed features", default="features.csv")
    parser.add_argument("--target", help="Target column name", default="close")
    parser.add_argument("--shift", help="Target shift periods", type=int, default=5)
    parser.add_argument("--feature_type", help="Feature set complexity", choices=["minimal", "balanced", "full"], default="balanced")
    parser.add_argument("--visualize", help="Visualize feature importance", action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load data
        data = pd.read_csv(args.input_file)
        
        # Create date index if not already present
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
        # Create pipeline
        pipeline = build_default_pipeline(feature_type=args.feature_type)
        
        # Create target (future returns)
        target = data[args.target].pct_change(periods=args.shift).shift(-args.shift)
        
        # Generate features
        features = pipeline.fit_transform(data, target)
        
        # Save features to CSV
        features.to_csv(args.output_file)
        print(f"Transformed features saved to {args.output_file}")
        print(f"Generated {features.shape[1]} features from {len(data)} data points")
        
        # Visualize feature importance if requested
        if args.visualize:
            # Remove NaN values from target
            valid_mask = ~np.isnan(target)
            X = features[valid_mask]
            y = target[valid_mask]
            
            # Evaluate features
            evaluation = pipeline.evaluate_features(X, y)
            
            # Print evaluation results
            print(f"CV Score: {evaluation['mean_score']:.4f} ± {evaluation['std_score']:.4f}")
            
            # Visualize importance
            pipeline.visualize_feature_importance(
                evaluation['feature_importance'],
                top_n=20,
                save_path="feature_importance.png"
            )
            print("Feature importance visualization saved to feature_importance.png")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()