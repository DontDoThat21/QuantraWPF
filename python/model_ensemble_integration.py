#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Ensemble Integration Module

This module provides integration functions to connect the ensemble learning framework
with other prediction modules in the system (stock_predictor, market_regime_detection,
anomaly_detection).
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Union, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_ensemble_integration')

try:
    from python.ensemble_learning import (
        ModelWrapper, EnsembleModel, create_ensemble_from_models, train_model_ensemble
    )
    ENSEMBLE_LEARNING_AVAILABLE = True
except ImportError:
    try:
        from ensemble_learning import (
            ModelWrapper, EnsembleModel, create_ensemble_from_models, train_model_ensemble
        )
        ENSEMBLE_LEARNING_AVAILABLE = True
    except ImportError:
        ENSEMBLE_LEARNING_AVAILABLE = False
        logger.warning("Ensemble learning module not available. Functionality will be limited.")

# Set up paths for model persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_DIR = os.path.join(BASE_DIR, 'models', 'ensembles')
os.makedirs(ENSEMBLE_DIR, exist_ok=True)


class StockPredictionEnsemble:
    """
    Ensemble wrapper for stock prediction models.
    
    Combines multiple stock prediction models (Random Forests, GBMs, Neural Networks)
    into an ensemble for more robust predictions.
    """
    def __init__(self, 
                 ensemble_name: str = 'stock_prediction_ensemble',
                 ensemble_method: str = 'weighted_average',
                 use_all_available_models: bool = True):
        """
        Initialize the stock prediction ensemble.
        
        Args:
            ensemble_name: Name for the ensemble (used for saving/loading)
            ensemble_method: Method for combining predictions ('weighted_average', 'stacking', etc.)
            use_all_available_models: Whether to include all available model types
        """
        if not ENSEMBLE_LEARNING_AVAILABLE:
            raise ImportError("Ensemble learning module not available.")
        
        self.ensemble_name = ensemble_name
        self.ensemble_method = ensemble_method
        self.use_all_available_models = use_all_available_models
        self.ensemble = None
        self.models = []
        self.model_configs = []
        self.features = None
        
        # Path for saving/loading
        self.model_path = os.path.join(ENSEMBLE_DIR, f"{ensemble_name}")
    
    def add_sklearn_model(self, model, name: str = None, weight: float = 1.0):
        """Add a scikit-learn based model to the ensemble."""
        wrapper = ModelWrapper(model, model_type='sklearn', name=name, weight=weight)
        self.models.append(wrapper)
    
    def add_pytorch_model(self, model, name: str = None, weight: float = 1.0):
        """Add a PyTorch based model to the ensemble."""
        wrapper = ModelWrapper(model, model_type='pytorch', name=name, weight=weight)
        self.models.append(wrapper)
    
    def add_tensorflow_model(self, model, name: str = None, weight: float = 1.0):
        """Add a TensorFlow based model to the ensemble."""
        wrapper = ModelWrapper(model, model_type='tensorflow', name=name, weight=weight)
        self.models.append(wrapper)
    
    def add_model_config(self, model_config: Dict[str, Any]):
        """
        Add a model configuration to be trained as part of the ensemble.
        
        Args:
            model_config: Dictionary with model configuration
                - model_class: Class or import string
                - model_type: 'sklearn', 'pytorch', 'tensorflow', 'custom'
                - other parameters specific to the model
        """
        self.model_configs.append(model_config)
    
    def build_ensemble(self):
        """Build the ensemble from added models."""
        if not self.models and not self.model_configs:
            raise ValueError("No models or model configurations added to the ensemble")
            
        # Create ensemble from existing models if any
        if self.models:
            self.ensemble = EnsembleModel(
                models=self.models,
                ensemble_method=self.ensemble_method,
                task_type='regression'
            )
        else:
            # Ensemble will be built during training
            pass
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None):
        """
        Train the ensemble on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            test_size: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            dict: Training results with metrics
        """
        # If we have model configs, train them
        if self.model_configs:
            # Train ensemble
            self.ensemble, training_results = train_model_ensemble(
                X, y,
                models_to_train=self.model_configs,
                test_size=test_size,
                ensemble_method=self.ensemble_method,
                task_type='regression',
                dynamic_weighting=True,
                random_state=random_state
            )
            return training_results
        
        # Otherwise use already added models
        elif self.models:
            # Build ensemble if not already built
            if self.ensemble is None:
                self.build_ensemble()
                
            # Evaluate on data
            return self.ensemble.evaluate(X, y)
        
        else:
            raise ValueError("No models available for training")
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make stock predictions using the ensemble.
        
        Args:
            X: Features
            
        Returns:
            dict: Prediction results
        """
        if self.ensemble is None:
            raise ValueError("Ensemble not built. Call build_ensemble() or train() first.")
        
        # Make prediction
        prediction = self.ensemble.predict(X)
        
        # Get feature importance
        feature_importance = self.ensemble.get_feature_importance(X)
        
        # Get confidence (estimated from model weights or predictions)
        confidence = 0.8  # Default confidence
        
        # Infer action based on prediction
        current_price = X[0, 0] if X.shape[0] > 0 and X.shape[1] > 0 else 0.0
        target_price = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
        
        price_diff = target_price - current_price
        if abs(price_diff) / max(current_price, 1e-10) < 0.01:  # Less than 1% change
            action = "HOLD"
        elif price_diff > 0:
            action = "BUY"
        else:
            action = "SELL"
        
        # Format the result
        result = {
            'action': action,
            'confidence': float(confidence),
            'targetPrice': float(target_price),
            'weights': {str(k): float(v) for k, v in feature_importance.items()},
            'ensemble_method': self.ensemble_method,
            'model_count': len(self.models) if self.models else len(self.ensemble.models),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
    
    def save(self):
        """Save the ensemble to disk."""
        if self.ensemble is None:
            raise ValueError("No ensemble to save")
            
        # Save ensemble
        self.ensemble.save(self.model_path)
        
        # Save additional metadata
        metadata = {
            'ensemble_name': self.ensemble_name,
            'ensemble_method': self.ensemble_method,
            'use_all_available_models': self.use_all_available_models,
            'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(f"{self.model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load(self):
        """Load the ensemble from disk."""
        # Check if ensemble exists
        if not os.path.exists(f"{self.model_path}_metadata.json"):
            raise FileNotFoundError(f"Ensemble {self.ensemble_name} not found at {self.model_path}")
        
        # Load metadata
        with open(f"{self.model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
            
        self.ensemble_name = metadata['ensemble_name']
        self.ensemble_method = metadata['ensemble_method']
        self.use_all_available_models = metadata.get('use_all_available_models', True)
        
        # Load ensemble
        self.ensemble = EnsembleModel.load(self.model_path)
        
        # Update models list from ensemble
        self.models = self.ensemble.models


def integrate_with_stock_predictor():
    """
    Integrate ensemble learning with stock_predictor module.
    
    This function enhances the stock_predictor module by adding ensemble capabilities.
    """
    if not ENSEMBLE_LEARNING_AVAILABLE:
        logger.warning("Ensemble learning module not available, cannot integrate")
        return False
        
    try:
        import stock_predictor
        
        # Check if ensemble learning is already integrated
        if hasattr(stock_predictor, 'predict_with_ensemble'):
            logger.info("Ensemble learning already integrated with stock_predictor")
            return True
            
        # Add ensemble prediction function to the module
        def predict_with_ensemble(features, ensemble_method='weighted_average', use_all_models=True):
            """
            Make stock predictions using an ensemble of models.
            
            Args:
                features (dict): Dictionary of feature names and values
                ensemble_method (str): Method for combining predictions
                use_all_models (bool): Whether to use all available models
                
            Returns:
                dict: Prediction results
            """
            # Convert features to numpy array
            feature_array = np.array([list(features.values())])
            
            # Try to load existing ensemble
            ensemble = StockPredictionEnsemble(
                ensemble_name='default_stock_ensemble',
                ensemble_method=ensemble_method,
                use_all_available_models=use_all_models
            )
            
            try:
                # Try to load existing ensemble
                ensemble.load()
                logger.info("Loaded existing stock prediction ensemble")
            except:
                # Create a new ensemble
                logger.info("Creating new stock prediction ensemble")
                
                # Add model configurations (will be trained when needed)
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import ElasticNet
                
                ensemble.add_model_config({
                    'model_class': RandomForestRegressor,
                    'n_estimators': 100, 
                    'max_depth': 10, 
                    'random_state': 42
                })
                
                ensemble.add_model_config({
                    'model_class': GradientBoostingRegressor,
                    'n_estimators': 100, 
                    'learning_rate': 0.1, 
                    'random_state': 42
                })
                
                ensemble.add_model_config({
                    'model_class': ElasticNet,
                    'alpha': 0.01, 
                    'random_state': 42
                })
                
                # If we have PyTorch/TensorFlow, try to add those models too
                if use_all_models:
                    try:
                        if hasattr(stock_predictor, 'PYTORCH_AVAILABLE') and stock_predictor.PYTORCH_AVAILABLE:
                            pytorchModel = stock_predictor.PyTorchStockPredictor()
                            ensemble.add_pytorch_model(pytorchModel.model, name='PyTorch')
                    except:
                        logger.warning("Could not add PyTorch model to ensemble")
                        
                    try:
                        if hasattr(stock_predictor, 'TENSORFLOW_AVAILABLE') and stock_predictor.TENSORFLOW_AVAILABLE:
                            tensorflowModel = stock_predictor.TensorFlowStockPredictor()
                            ensemble.add_tensorflow_model(tensorflowModel.model, name='TensorFlow')
                    except:
                        logger.warning("Could not add TensorFlow model to ensemble")
                
                # Build ensemble
                ensemble.build_ensemble()
            
            # Make prediction
            prediction_result = ensemble.predict(feature_array)
            
            # Enhance result with additional info
            result = stock_predictor.predict_stock(features)
            result['ensemble'] = prediction_result
            
            # If we have enough confidence in the ensemble, use its action
            if prediction_result['confidence'] > 0.7:
                result['action'] = prediction_result['action']
                result['confidence'] = (result['confidence'] + prediction_result['confidence']) / 2
            
            return result
        
        # Add function to the module
        stock_predictor.predict_with_ensemble = predict_with_ensemble
        
        logger.info("Successfully integrated ensemble learning with stock_predictor")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate ensemble learning with stock_predictor: {str(e)}")
        return False


def integrate_with_market_regime_detection():
    """
    Integrate ensemble learning with market_regime_detection module.
    
    This function enhances the market_regime_detection module with advanced ensemble capabilities.
    """
    if not ENSEMBLE_LEARNING_AVAILABLE:
        logger.warning("Ensemble learning module not available, cannot integrate")
        return False
        
    try:
        import market_regime_detection
        
        # Check if ensemble learning is already integrated
        if hasattr(market_regime_detection, 'advanced_ensemble_regime_detection'):
            logger.info("Ensemble learning already integrated with market_regime_detection")
            return True
            
        # Add enhanced ensemble function
        def advanced_ensemble_regime_detection(results, dynamic_weighting=True):
            """
            Enhanced ensemble method for regime detection that uses advanced ensemble learning techniques.
            
            Args:
                results (list): List of regime detection results from different models
                dynamic_weighting (bool): Whether to use dynamic weighting based on past performance
                
            Returns:
                dict: Combined regime detection result
            """
            # Extract regimes and confidence values
            regimes = [result['regime'] for result in results if 'regime' in result]
            confidences = [result.get('confidence', 0.5) for result in results if 'regime' in result]
            
            if not regimes:
                logger.warning("No valid regime results provided")
                return {
                    'regime': market_regime_detection.REGIME_RANGING,
                    'confidence': 0.5,
                    'error': 'No valid results for ensemble',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
            # Map regimes to numeric values for modeling
            regime_mapping = {
                market_regime_detection.REGIME_TRENDING_UP: 0,
                market_regime_detection.REGIME_TRENDING_DOWN: 1,
                market_regime_detection.REGIME_RANGING: 2
            }
            
            reverse_mapping = {v: k for k, v in regime_mapping.items()}
            
            numeric_regimes = [regime_mapping.get(regime, 2) for regime in regimes]
            
            # Create a simple ensemble for classification
            ensemble = EnsembleModel(
                ensemble_method='weighted_average' if dynamic_weighting else 'simple_average',
                voting='soft',
                task_type='classification',
                dynamic_weighting=dynamic_weighting
            )
            
            # Create mock classifiers that predict specific classes with given confidence
            for i, (regime, confidence) in enumerate(zip(numeric_regimes, confidences)):
                # Create a simple model that always predicts the specified regime
                class FixedRegimeClassifier:
                    def __init__(self, regime, confidence):
                        self.regime = regime
                        self.confidence = confidence
                    
                    def predict(self, X):
                        return np.full(X.shape[0], self.regime)
                    
                    def predict_proba(self, X):
                        proba = np.zeros((X.shape[0], 3))
                        proba[:, self.regime] = self.confidence
                        # Distribute remaining probability equally
                        remaining = (1 - self.confidence) / 2
                        for i in range(3):
                            if i != self.regime:
                                proba[:, i] = remaining
                        return proba
                
                classifier = FixedRegimeClassifier(regime, confidence)
                ensemble.add_model(classifier, model_type='custom', name=f"RegimeModel_{i}", weight=confidence)
            
            # Make prediction with ensemble
            mock_data = np.ones((1, 1))  # Dummy data, not used by the classifiers
            final_regime_numeric = int(ensemble.predict(mock_data)[0])
            final_regime = reverse_mapping.get(final_regime_numeric, market_regime_detection.REGIME_RANGING)
            
            # Try to get probability for the predicted regime
            try:
                probas = ensemble.predict_proba(mock_data)
                confidence = float(probas[0, final_regime_numeric])
            except:
                # Fallback - use average of confidences
                confidence = sum(confidences) / len(confidences)
            
            # Create detailed result similar to original implementation
            model_results = {}
            for i, result in enumerate(results):
                if 'model_type' in result:
                    model_type = result['model_type']
                else:
                    model_type = f"model_{i}"
                    
                model_results[model_type] = {
                    'regime': result.get('regime', market_regime_detection.REGIME_RANGING),
                    'confidence': float(result.get('confidence', 0.5))
                }
            
            # Generate trading approaches based on regime (same as original)
            trading_approaches = []
            if final_regime == market_regime_detection.REGIME_TRENDING_UP:
                trading_approaches = [
                    "Use trend-following strategies",
                    "Consider longer holding periods",
                    "Look for pullbacks as buying opportunities",
                    "Use trailing stops to protect profits",
                    "Focus on momentum indicators"
                ]
            elif final_regime == market_regime_detection.REGIME_TRENDING_DOWN:
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
                'confidence': float(confidence),
                'modelResults': model_results,
                'tradingApproaches': trading_approaches,
                'ensemble_method': 'advanced_ensemble' if dynamic_weighting else 'simple_ensemble',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Add function to the module
        market_regime_detection.advanced_ensemble_regime_detection = advanced_ensemble_regime_detection
        
        logger.info("Successfully integrated ensemble learning with market_regime_detection")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate ensemble learning with market_regime_detection: {str(e)}")
        return False


def integrate_with_anomaly_detection():
    """
    Integrate ensemble learning with anomaly_detection module.
    
    This function enhances the anomaly_detection module with advanced ensemble capabilities.
    """
    if not ENSEMBLE_LEARNING_AVAILABLE:
        logger.warning("Ensemble learning module not available, cannot integrate")
        return False
        
    try:
        import anomaly_detection
        
        # Check if ensemble learning is already integrated
        if hasattr(anomaly_detection, 'advanced_ensemble_anomaly_detection'):
            logger.info("Ensemble learning already integrated with anomaly_detection")
            return True
            
        # Add enhanced ensemble function
        def advanced_ensemble_anomaly_detection(results, threshold=None, adaptive_weighting=True):
            """
            Enhanced ensemble method for anomaly detection using advanced ensemble learning techniques.
            
            Args:
                results (dict): Dictionary with results from different algorithms
                threshold (float): Ensemble anomaly score threshold
                adaptive_weighting (bool): Whether to use adaptive weighting based on algorithm performance
                
            Returns:
                dict: Combined anomaly detection result
            """
            # Use original function as fallback if necessary
            if not results:
                logger.warning("No anomaly detection results provided")
                return anomaly_detection.ensemble_anomaly_detection(results, threshold)
            
            if threshold is None:
                threshold = anomaly_detection.DEFAULT_THRESHOLDS['ensemble']
                
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
            
            # Convert the results to a format suitable for ensemble learning
            model_scores = []
            model_names = []
            model_weights = []
            
            for model_name, model_result in results.items():
                if model_result is not None and model_result['scores'] is not None:
                    # Get normalized scores where 1 = most anomalous
                    normalized_scores = anomaly_detection.calculate_anomaly_severity(
                        model_result['scores'], normalized=True
                    )
                    
                    model_scores.append(normalized_scores)
                    model_names.append(model_name)
                    
                    # Calculate weight based on number of anomalies detected
                    if adaptive_weighting:
                        # More selective algorithms get higher weight
                        selectivity = 1.0 - (len(model_result['indices']) / data_length)
                        model_weights.append(0.5 + 0.5 * selectivity)
                    else:
                        model_weights.append(1.0)
            
            if not model_scores:
                return {
                    'anomalies': [],
                    'anomaly_indices': [],
                    'anomaly_types': {},
                    'combined_scores': []
                }
            
            # Create a regression ensemble to combine scores
            ensemble = EnsembleModel(
                ensemble_method='weighted_average',
                task_type='regression',
                dynamic_weighting=adaptive_weighting
            )
            
            # Create simple models that return the precalculated scores
            class ScorePredictor:
                def __init__(self, scores):
                    self.scores = scores
                
                def predict(self, X=None):
                    return self.scores
            
            # Add each model's scores
            for scores, name, weight in zip(model_scores, model_names, model_weights):
                predictor = ScorePredictor(scores)
                ensemble.add_model(predictor, model_type='custom', name=name, weight=weight)
            
            # Generate combined scores using ensemble
            dummy_data = np.ones((data_length, 1))  # Dummy data, not used by predictors
            combined_scores = ensemble.predict(dummy_data)
            
            # Find anomaly indices based on threshold
            anomaly_indices = np.where(combined_scores > threshold)[0]
            
            # Enhanced metadata about the ensemble
            ensemble_info = {
                'model_weights': dict(zip(model_names, model_weights)),
                'threshold': threshold,
                'adaptive_weighting': adaptive_weighting,
                'ensemble_method': 'advanced_weighted_ensemble'
            }
            
            return {
                'anomalies': anomaly_indices.tolist(),
                'anomaly_indices': anomaly_indices.tolist(),
                'combined_scores': combined_scores.tolist(),
                'ensemble_info': ensemble_info
            }
        
        # Add function to the module
        anomaly_detection.advanced_ensemble_anomaly_detection = advanced_ensemble_anomaly_detection
        
        logger.info("Successfully integrated ensemble learning with anomaly_detection")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate ensemble learning with anomaly_detection: {str(e)}")
        return False


def integrate_all_modules():
    """Integrate ensemble learning with all compatible modules."""
    results = {}
    
    # Integrate with stock_predictor
    results['stock_predictor'] = integrate_with_stock_predictor()
    
    # Integrate with market_regime_detection
    results['market_regime_detection'] = integrate_with_market_regime_detection()
    
    # Integrate with anomaly_detection
    results['anomaly_detection'] = integrate_with_anomaly_detection()
    
    return results


if __name__ == "__main__":
    # When run directly, integrate with all modules
    results = integrate_all_modules()
    
    # Print integration results
    print("Ensemble Integration Results:")
    for module, success in results.items():
        print(f"  {module}: {'Success' if success else 'Failed'}")