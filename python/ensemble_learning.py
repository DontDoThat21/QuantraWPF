#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Learning Module

This module provides tools for combining multiple machine learning models 
to enhance prediction robustness and accuracy through ensemble techniques.

Techniques implemented:
1. Simple averaging/voting
2. Weighted averaging based on model confidence or performance
3. Stacking (meta-learning)
4. Dynamic weighting based on historical performance
5. Model diversity analysis
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
import logging
from datetime import datetime
from typing import List, Dict, Any, Union, Tuple, Callable, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils import resample

# Import model handlers
try:
    from .model_handlers import ModelHandlerFactory
except ImportError:
    from model_handlers import ModelHandlerFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ensemble_learning')

# Set up paths for model persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ensemble')
os.makedirs(MODEL_DIR, exist_ok=True)

# Try to import external model frameworks
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available for ensemble learning")
except ImportError:
    logger.warning("TensorFlow is not available. Some ensemble capabilities may be limited.")

try:
    import torch
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available for ensemble learning")
except ImportError:
    logger.warning("PyTorch is not available. Some ensemble capabilities may be limited.")


class ModelWrapper:
    """
    Wrapper to standardize interface for different types of models (sklearn, PyTorch, TensorFlow).
    This allows ensemble methods to work with any model type transparently.
    """
    def __init__(self, model: Any, model_type: str, name: str = None, weight: float = 1.0):
        """
        Initialize model wrapper with a model and metadata.
        
        Args:
            model: The model object (sklearn, TensorFlow, PyTorch, etc.)
            model_type: Type of model ('sklearn', 'tensorflow', 'pytorch', 'custom')
            name: Optional name for the model
            weight: Initial weight for ensemble contribution (default: 1.0)
        """
        self.model = model
        self.model_type = model_type.lower()
        self.name = name if name else f"{model_type}_model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.weight = weight
        self.performance_history = []
        
        # Validate model type
        valid_types = ModelHandlerFactory.get_supported_types()
        if self.model_type not in valid_types:
            logger.warning(f"Unknown model type: {model_type}. Must be one of {valid_types}")
            self.model_type = 'custom'
        
        # Get the appropriate handler
        self.handler = ModelHandlerFactory.get_handler(self.model_type)
        
        # Check if model is compatible with its claimed type
        self._validate_model_compatibility()
    
    def _validate_model_compatibility(self):
        """Validate that the model matches its claimed type"""
        if not self.handler.validate_model_compatibility(self.model):
            logger.warning(f"Model claimed to be {self.model_type} but compatibility check failed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction using the wrapped model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Model predictions
        """
        return self.handler.predict(self.model, X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions for classification models.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Probability predictions or None if not applicable
        """
        try:
            return self.handler.predict_proba(self.model, X)
        except Exception as e:
            logger.warning(f"Error getting probabilities from {self.name}: {str(e)}")
            return None
    
    def get_feature_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get feature importances if available.
        
        Args:
            X: Input features (for models that need data to compute importance)
            
        Returns:
            dict: Feature names/indices and importance scores
        """
        try:
            return self.handler.get_feature_importance(self.model, X)
        except Exception as e:
            logger.warning(f"Error getting feature importance from {self.name}: {str(e)}")
            return {}
    
    def update_weight(self, new_weight: float):
        """Update the model's weight in the ensemble"""
        self.weight = new_weight
    
    def record_performance(self, metric_value: float, metric_name: str = 'error'):
        """Record a performance metric for this model"""
        self.performance_history.append({
            'metric': metric_name,
            'value': float(metric_value),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # Keep only the last 100 performance records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_recent_performance(self, metric_name: str = None, n: int = 10) -> float:
        """
        Get average recent performance for specified metric.
        
        Args:
            metric_name: Name of metric to filter by (or None for any metric)
            n: Number of recent records to average
            
        Returns:
            float: Average performance value
        """
        if not self.performance_history:
            return 0.0
            
        # Filter by metric name if specified
        relevant_history = [
            record for record in self.performance_history 
            if metric_name is None or record['metric'] == metric_name
        ]
        
        if not relevant_history:
            return 0.0
            
        # Get the n most recent records
        recent_records = relevant_history[-n:]
        return sum(record['value'] for record in recent_records) / len(recent_records)
    
    def save(self, path: str):
        """
        Save the model wrapper.
        
        Args:
            path: Path to save the model wrapper
        """
        # Save metadata
        metadata = {
            'name': self.name,
            'model_type': self.model_type,
            'weight': self.weight,
            'performance_history': self.performance_history,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model itself based on its type
        try:
            self.handler.save_model(self.model, path)
        except Exception as e:
            logger.warning(f"Error saving model {self.name}: {str(e)}")
            raise
        
        # Save metadata
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'ModelWrapper':
        """
        Load a model wrapper from disk.
        
        Args:
            path: Path where the model wrapper was saved
            
        Returns:
            ModelWrapper: Loaded model wrapper
        """
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load the model based on its type
        model_type = metadata['model_type']
        handler = ModelHandlerFactory.get_handler(model_type)
        
        try:
            model = handler.load_model(path)
        except Exception as e:
            logger.error(f"Error loading model of type {model_type}: {str(e)}")
            raise
        
        # Create wrapper
        wrapper = cls(model=model, model_type=model_type, name=metadata['name'], weight=metadata['weight'])
        wrapper.performance_history = metadata['performance_history']
        
        return wrapper


class EnsembleModel(BaseEstimator):
    """
    Ensemble model class that combines multiple models for improved predictions.
    Implements common sklearn interfaces for compatibility.
    """
    def __init__(self, 
                 models: List[Union[ModelWrapper, Any]] = None,
                 model_types: List[str] = None,
                 weights: List[float] = None,
                 ensemble_method: str = 'weighted_average',
                 voting: str = 'soft',
                 dynamic_weighting: bool = False,
                 task_type: str = 'regression'):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of ModelWrappers or raw models
            model_types: List of model types if raw models are provided
            weights: Initial weights for models
            ensemble_method: Method for combining predictions ('simple_average', 'weighted_average', 'stacking')
            voting: For classification, 'hard' or 'soft' voting
            dynamic_weighting: Whether to adjust weights based on model performance
            task_type: Type of prediction task ('regression' or 'classification')
        """
        self.models = []  # List of ModelWrappers
        self.ensemble_method = ensemble_method.lower()
        self.voting = voting.lower()
        self.dynamic_weighting = dynamic_weighting
        self.task_type = task_type.lower()
        self.meta_model = None  # For stacking
        self.classes_ = None  # For classification
        
        # Setup the model wrappers
        if models:
            for i, model in enumerate(models):
                if isinstance(model, ModelWrapper):
                    # Already a wrapper
                    self.models.append(model)
                else:
                    # Get model type
                    if model_types and i < len(model_types):
                        model_type = model_types[i]
                    else:
                        # Try to guess model type
                        if hasattr(model, 'predict') and hasattr(model, 'fit'):
                            model_type = 'sklearn'
                        elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
                            model_type = 'tensorflow'
                        elif PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                            model_type = 'pytorch'
                        else:
                            model_type = 'custom'
                    
                    # Get weight if provided
                    weight = weights[i] if weights and i < len(weights) else 1.0
                    
                    # Create wrapper
                    self.models.append(ModelWrapper(model, model_type, weight=weight))
        
        # Validate ensemble method
        valid_methods = ['simple_average', 'weighted_average', 'stacking', 'majority_vote', 
                        'blending', 'dynamic_selection', 'cv_stacking']
        if self.ensemble_method not in valid_methods:
            logger.warning(f"Unknown ensemble method: {ensemble_method}. Must be one of {valid_methods}")
            self.ensemble_method = 'weighted_average'
        
        # Validate task type
        if self.task_type not in ['regression', 'classification']:
            logger.warning(f"Unknown task type: {task_type}. Must be 'regression' or 'classification'")
            self.task_type = 'regression'
    
    def add_model(self, model: Any, model_type: str = None, weight: float = 1.0, name: str = None):
        """
        Add a new model to the ensemble.
        
        Args:
            model: Model object or ModelWrapper
            model_type: Type of model if not a ModelWrapper
            weight: Initial weight for the model
            name: Name for the model
        """
        if isinstance(model, ModelWrapper):
            self.models.append(model)
        else:
            # Determine model type if not provided
            if model_type is None:
                if hasattr(model, 'predict') and hasattr(model, 'fit'):
                    model_type = 'sklearn'
                elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
                    model_type = 'tensorflow'
                elif PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                    model_type = 'pytorch'
                else:
                    model_type = 'custom'
            
            # Create wrapper
            wrapper = ModelWrapper(model, model_type, name=name, weight=weight)
            self.models.append(wrapper)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        Fit the stacking meta-model if using stacking ensemble method.
        
        Args:
            X: Training data features
            y: Training data targets
            sample_weight: Optional sample weights
            
        Returns:
            self: The fitted estimator
        """
        if self.ensemble_method == 'stacking':
            # For stacking, we need to fit a meta-model on the predictions of base models
            # We assume base models are already trained
            
            # Generate predictions from each model
            meta_features = self._generate_meta_features(X)
            
            # Create and fit meta-model
            if self.task_type == 'regression':
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)
            else:  # classification
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
                self.classes_ = np.unique(y)
            
            # Fit meta-model
            self.meta_model.fit(meta_features, y, sample_weight=sample_weight)
        
        # For classification, store classes
        if self.task_type == 'classification':
            self.classes_ = np.unique(y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Classification with hard voting (majority vote)
        if self.task_type == 'classification' and self.voting == 'hard':
            # Get predictions from each model
            predictions = []
            for model_wrapper in self.models:
                try:
                    pred = model_wrapper.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error getting predictions from {model_wrapper.name}: {str(e)}")
            
            # Convert predictions to array
            predictions = np.array(predictions)
            
            # Majority vote - mode of predictions for each sample
            from scipy import stats
            return stats.mode(predictions, axis=0, keepdims=False)[0]
        
        # For regression or soft voting classification
        if self.ensemble_method == 'stacking' and self.meta_model is not None:
            # For stacking, use the meta-model
            meta_features = self._generate_meta_features(X)
            return self.meta_model.predict(meta_features)
        else:
            # Simple average or weighted average
            return self._combine_predictions(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self.ensemble_method == 'stacking' and self.meta_model is not None:
            # Use meta-model for stacking
            meta_features = self._generate_meta_features(X)
            return self.meta_model.predict_proba(meta_features)
        
        # Combine probability predictions from all models
        proba_predictions = []
        weights = []
        
        for model_wrapper in self.models:
            try:
                proba = model_wrapper.predict_proba(X)
                if proba is not None:
                    proba_predictions.append(proba)
                    weights.append(model_wrapper.weight)
            except Exception as e:
                logger.warning(f"Error getting probability predictions from {model_wrapper.name}: {str(e)}")
        
        if not proba_predictions:
            raise ValueError("No models provided probability predictions")
        
        # Ensure all predictions have the same shape
        # We might need to handle the case where different models predict different number of classes
        n_classes = max(p.shape[1] for p in proba_predictions)
        
        # Ensure all predictions have the same classes
        aligned_predictions = []
        for proba in proba_predictions:
            if proba.shape[1] < n_classes:
                # Pad with zeros for missing classes
                padded = np.zeros((proba.shape[0], n_classes))
                padded[:, :proba.shape[1]] = proba
                aligned_predictions.append(padded)
            else:
                aligned_predictions.append(proba)
        
        # Convert to array
        proba_predictions = np.array(aligned_predictions)
        weights = np.array(weights).reshape(-1, 1, 1)
        
        # Weighted average of probabilities
        if self.ensemble_method == 'weighted_average':
            weighted_sum = np.sum(proba_predictions * weights, axis=0)
            return weighted_sum / np.sum(weights)
        else:
            # Simple average
            return np.mean(proba_predictions, axis=0)
    
    def _combine_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Combine predictions from all models.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Combined predictions
        """
        predictions = []
        weights = []
        
        for model_wrapper in self.models:
            try:
                pred = model_wrapper.predict(X)
                predictions.append(pred)
                weights.append(model_wrapper.weight)
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_wrapper.name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No models provided predictions")
        
        # Convert to array
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # For classification with soft voting, we might have probability predictions
        if self.task_type == 'classification' and self.voting == 'soft':
            # Try to get probability predictions first
            try:
                return self.predict_proba(X).argmax(axis=1)
            except:
                # Fall back to weighted average of hard predictions
                pass
        
        # Weighted average
        if self.ensemble_method == 'weighted_average':
            # Reshape weights for broadcasting
            weights = weights.reshape(-1, 1)
            weighted_sum = np.sum(predictions * weights, axis=0)
            return weighted_sum / np.sum(weights)
        else:
            # Simple average
            return np.mean(predictions, axis=0)
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate features for meta-model from base model predictions.
        
        Args:
            X: Original features
            
        Returns:
            np.ndarray: Meta-features (predictions from base models)
        """
        meta_features = []
        
        for model_wrapper in self.models:
            try:
                # For regression, use raw predictions
                if self.task_type == 'regression':
                    preds = model_wrapper.predict(X)
                    meta_features.append(preds)
                else:
                    # For classification, try to get probabilities first
                    probs = model_wrapper.predict_proba(X)
                    if probs is not None:
                        # Flatten probabilities for each class
                        meta_features.extend([probs[:, i] for i in range(probs.shape[1])])
                    else:
                        # Fall back to hard predictions
                        preds = model_wrapper.predict(X)
                        meta_features.append(preds)
            except Exception as e:
                logger.warning(f"Error generating meta-features from {model_wrapper.name}: {str(e)}")
        
        # Stack meta features horizontally
        if meta_features:
            return np.column_stack(meta_features)
        else:
            raise ValueError("No valid meta-features generated from base models")
    
    def update_weights(self, X: np.ndarray, y: np.ndarray):
        """
        Update model weights based on performance.
        
        Args:
            X: Validation features
            y: Validation targets
        """
        if not self.dynamic_weighting:
            return
        
        # Evaluate each model
        for model_wrapper in self.models:
            try:
                # Get predictions
                preds = model_wrapper.predict(X)
                
                # Calculate error metric based on task type
                if self.task_type == 'regression':
                    mse = mean_squared_error(y, preds)
                    error = np.sqrt(mse)  # RMSE
                    model_wrapper.record_performance(error, 'rmse')
                    # Higher error should lead to lower weight (inversely proportional)
                    model_wrapper.update_weight(1.0 / (error + 1e-10))
                else:  # classification
                    accuracy = accuracy_score(y, preds)
                    model_wrapper.record_performance(accuracy, 'accuracy')
                    # Higher accuracy should lead to higher weight
                    model_wrapper.update_weight(accuracy + 1e-10)
            except Exception as e:
                logger.warning(f"Error updating weight for {model_wrapper.name}: {str(e)}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble and individual models.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        results = {'ensemble': {}, 'models': {}}
        
        # Evaluate ensemble
        ensemble_preds = self.predict(X)
        
        if self.task_type == 'regression':
            # Calculate RMSE manually to avoid version compatibility issues
            mse = mean_squared_error(y, ensemble_preds)
            rmse = float(np.sqrt(mse))
            results['ensemble']['rmse'] = rmse
            results['ensemble']['mae'] = float(mean_absolute_error(y, ensemble_preds))
            results['ensemble']['r2'] = float(r2_score(y, ensemble_preds))
        else:  # classification
            results['ensemble']['accuracy'] = float(accuracy_score(y, ensemble_preds))
            
            # Add other metrics for classification
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score
                results['ensemble']['precision'] = float(precision_score(y, ensemble_preds, average='weighted'))
                results['ensemble']['recall'] = float(recall_score(y, ensemble_preds, average='weighted'))
                results['ensemble']['f1'] = float(f1_score(y, ensemble_preds, average='weighted'))
            except Exception as e:
                logger.warning(f"Error computing additional classification metrics: {str(e)}")
        
        # Evaluate individual models
        for model_wrapper in self.models:
            model_name = model_wrapper.name
            results['models'][model_name] = {}
            
            try:
                preds = model_wrapper.predict(X)
                
                if self.task_type == 'regression':
                    mse = float(mean_squared_error(y, preds))
                    rmse = float(np.sqrt(mse))
                    mae = float(mean_absolute_error(y, preds))
                    r2 = float(r2_score(y, preds))
                    
                    results['models'][model_name]['rmse'] = rmse
                    results['models'][model_name]['mae'] = mae
                    results['models'][model_name]['r2'] = r2
                    
                    # Record performance for dynamic weighting
                    model_wrapper.record_performance(rmse, 'rmse')
                
                else:  # classification
                    acc = float(accuracy_score(y, preds))
                    results['models'][model_name]['accuracy'] = acc
                    
                    # Add other metrics for classification
                    try:
                        from sklearn.metrics import precision_score, recall_score, f1_score
                        results['models'][model_name]['precision'] = float(precision_score(y, preds, average='weighted'))
                        results['models'][model_name]['recall'] = float(recall_score(y, preds, average='weighted'))
                        results['models'][model_name]['f1'] = float(f1_score(y, preds, average='weighted'))
                    except:
                        pass
                    
                    # Record performance for dynamic weighting
                    model_wrapper.record_performance(acc, 'accuracy')
            
            except Exception as e:
                logger.warning(f"Error evaluating {model_name}: {str(e)}")
                results['models'][model_name]['error'] = str(e)
        
        return results
    
    def get_feature_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get combined feature importance from all models.
        
        Args:
            X: Input features (for models that need data to compute importance)
            
        Returns:
            dict: Feature indices/names and importance scores
        """
        combined_importance = {}
        total_weight = 0.0
        
        for model_wrapper in self.models:
            try:
                # Get model's feature importance
                importance = model_wrapper.get_feature_importance(X)
                if importance:
                    # Weight the importance by the model's weight
                    weight = model_wrapper.weight
                    total_weight += weight
                    
                    # Add to combined importance
                    for feature_id, score in importance.items():
                        if feature_id not in combined_importance:
                            combined_importance[feature_id] = 0.0
                        combined_importance[feature_id] += score * weight
            except Exception as e:
                logger.warning(f"Error getting feature importance from {model_wrapper.name}: {str(e)}")
        
        # Normalize by total weight
        if total_weight > 0:
            for feature_id in combined_importance:
                combined_importance[feature_id] /= total_weight
        
        return combined_importance
    
    def save(self, path: str):
        """
        Save the ensemble model.
        
        Args:
            path: Path to save the ensemble model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save metadata
        metadata = {
            'ensemble_method': self.ensemble_method,
            'voting': self.voting,
            'dynamic_weighting': self.dynamic_weighting,
            'task_type': self.task_type,
            'model_count': len(self.models),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metadata
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save each model
        for i, model_wrapper in enumerate(self.models):
            model_path = f"{path}_model_{i}"
            model_wrapper.save(model_path)
        
        # Save meta-model if using stacking
        if self.ensemble_method == 'stacking' and self.meta_model is not None:
            meta_model_path = f"{path}_meta_model.pkl"
            joblib.dump(self.meta_model, meta_model_path)
        
        # Save classes for classification
        if self.task_type == 'classification' and self.classes_ is not None:
            np.save(f"{path}_classes.npy", self.classes_)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """
        Load an ensemble model from disk.
        
        Args:
            path: Path where the ensemble model was saved
            
        Returns:
            EnsembleModel: Loaded ensemble model
        """
        # Load metadata
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create empty ensemble
        ensemble = cls(
            ensemble_method=metadata['ensemble_method'],
            voting=metadata['voting'],
            dynamic_weighting=metadata['dynamic_weighting'],
            task_type=metadata['task_type']
        )
        
        # Load each model
        for i in range(metadata['model_count']):
            try:
                model_path = f"{path}_model_{i}"
                model_wrapper = ModelWrapper.load(model_path)
                ensemble.models.append(model_wrapper)
            except Exception as e:
                logger.warning(f"Error loading model {i}: {str(e)}")
        
        # Load meta-model if using stacking
        if metadata['ensemble_method'] == 'stacking':
            meta_model_path = f"{path}_meta_model.pkl"
            if os.path.exists(meta_model_path):
                ensemble.meta_model = joblib.load(meta_model_path)
        
        # Load classes for classification
        if metadata['task_type'] == 'classification':
            classes_path = f"{path}_classes.npy"
            if os.path.exists(classes_path):
                ensemble.classes_ = np.load(classes_path)
        
        return ensemble


def create_ensemble_from_models(
    models: List[Any], 
    model_types: List[str] = None,
    weights: List[float] = None,
    ensemble_method: str = 'weighted_average',
    voting: str = 'soft',
    task_type: str = 'regression'
) -> EnsembleModel:
    """
    Create an ensemble from a list of models.
    
    Args:
        models: List of trained models
        model_types: List of model types
        weights: List of model weights
        ensemble_method: Method for combining predictions
        voting: For classification, 'hard' or 'soft' voting
        task_type: Type of prediction task ('regression' or 'classification')
        
    Returns:
        EnsembleModel: Initialized ensemble model
    """
    # Create ensemble
    ensemble = EnsembleModel(
        models=models,
        model_types=model_types,
        weights=weights,
        ensemble_method=ensemble_method,
        voting=voting,
        task_type=task_type
    )
    
    return ensemble


def train_model_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    models_to_train: List[Union[BaseEstimator, Dict[str, Any]]] = None,
    test_size: float = 0.2,
    ensemble_method: str = 'weighted_average',
    voting: str = 'soft',
    task_type: str = 'regression',
    use_cv: bool = False,
    n_splits: int = 5,
    dynamic_weighting: bool = True,
    random_state: int = None
) -> Tuple[EnsembleModel, Dict[str, Any]]:
    """
    Train an ensemble of models on the given dataset.
    
    Args:
        X: Training features
        y: Training targets
        models_to_train: List of models to train or model configurations
        test_size: Fraction of data to use for validation
        ensemble_method: Method for combining predictions
        voting: For classification, 'hard' or 'soft' voting
        task_type: Type of prediction task ('regression' or 'classification')
        use_cv: Whether to use cross-validation
        n_splits: Number of CV splits
        dynamic_weighting: Whether to adjust weights based on performance
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (ensemble_model, training_results)
    """
    # Default models if none provided
    if models_to_train is None:
        if task_type == 'regression':
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import ElasticNet
            
            models_to_train = [
                RandomForestRegressor(n_estimators=100, random_state=random_state),
                GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                ElasticNet(random_state=random_state)
            ]
        else:  # classification
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            
            models_to_train = [
                RandomForestClassifier(n_estimators=100, random_state=random_state),
                GradientBoostingClassifier(n_estimators=100, random_state=random_state),
                LogisticRegression(max_iter=1000, random_state=random_state)
            ]
    
    # Process model definitions
    processed_models = []
    for model_def in models_to_train:
        if isinstance(model_def, dict):
            # Create model from configuration
            model_config = model_def.copy()
            model_class = model_config.pop('model_class', None)
            model_type = model_config.pop('model_type', 'sklearn')
            
            if model_class:
                if isinstance(model_class, str):
                    # Import model class from string
                    parts = model_class.split('.')
                    module_name = '.'.join(parts[:-1])
                    class_name = parts[-1]
                    
                    try:
                        module = __import__(module_name, fromlist=[class_name])
                        model_class = getattr(module, class_name)
                    except:
                        logger.error(f"Could not import model class {model_class}")
                        continue
                
                # Create model instance
                try:
                    if random_state is not None and 'random_state' not in model_config:
                        model_config['random_state'] = random_state
                    model = model_class(**model_config)
                    processed_models.append((model, model_type))
                except Exception as e:
                    logger.error(f"Error creating model: {str(e)}")
            else:
                logger.warning(f"No model_class specified in configuration: {model_def}")
        else:
            # Use model directly
            processed_models.append((model_def, 'sklearn'))
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train each model
    trained_models = []
    trained_model_types = []
    model_metrics = {}
    
    for model, model_type in processed_models:
        try:
            # Train the model
            logger.info(f"Training model: {model.__class__.__name__}")
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            try:
                val_preds = model.predict(X_val)
                
                if task_type == 'regression':
                    val_rmse = mean_squared_error(y_val, val_preds, squared=False)
                    val_mae = mean_absolute_error(y_val, val_preds)
                    val_r2 = r2_score(y_val, val_preds)
                    
                    metrics = {
                        'rmse': float(val_rmse),
                        'mae': float(val_mae),
                        'r2': float(val_r2)
                    }
                    
                    logger.info(f"  Validation RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
                else:  # classification
                    val_acc = accuracy_score(y_val, val_preds)
                    
                    metrics = {
                        'accuracy': float(val_acc)
                    }
                    
                    # Add other metrics for classification
                    try:
                        from sklearn.metrics import precision_score, recall_score, f1_score
                        metrics['precision'] = float(precision_score(y_val, val_preds, average='weighted'))
                        metrics['recall'] = float(recall_score(y_val, val_preds, average='weighted'))
                        metrics['f1'] = float(f1_score(y_val, val_preds, average='weighted'))
                    except:
                        pass
                    
                    logger.info(f"  Validation Accuracy: {val_acc:.4f}")
                
                model_name = f"{model.__class__.__name__}_{len(trained_models)}"
                model_metrics[model_name] = metrics
            
            except Exception as e:
                logger.warning(f"Error evaluating model: {str(e)}")
            
            # Add to trained models
            trained_models.append(model)
            trained_model_types.append(model_type)
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
    
    if not trained_models:
        raise ValueError("No models could be successfully trained")
    
    # Calculate weights based on performance if using dynamic weighting
    weights = None
    if dynamic_weighting:
        weights = []
        for i, model in enumerate(trained_models):
            model_name = f"{model.__class__.__name__}_{i}"
            if model_name in model_metrics:
                # Use inverse RMSE for regression
                if task_type == 'regression' and 'rmse' in model_metrics[model_name]:
                    rmse = model_metrics[model_name]['rmse']
                    weight = 1.0 / (rmse + 1e-10)
                # Use accuracy for classification
                elif task_type == 'classification' and 'accuracy' in model_metrics[model_name]:
                    weight = model_metrics[model_name]['accuracy']
                else:
                    weight = 1.0
            else:
                weight = 1.0
            
            weights.append(weight)
    
    # Create and train ensemble
    ensemble = EnsembleModel(
        models=trained_models,
        model_types=trained_model_types,
        weights=weights,
        ensemble_method=ensemble_method,
        voting=voting,
        dynamic_weighting=dynamic_weighting,
        task_type=task_type
    )
    
    # Train stacking meta-model if using stacking
    if ensemble_method == 'stacking':
        ensemble.fit(X_val, y_val)
    
    # Evaluate ensemble
    ensemble_metrics = ensemble.evaluate(X_val, y_val)
    
    # Prepare training results
    training_results = {
        'ensemble_metrics': ensemble_metrics['ensemble'],
        'model_metrics': model_metrics,
        'ensemble_method': ensemble_method,
        'task_type': task_type,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return ensemble, training_results


class HomogeneousEnsemble(BaseEstimator):
    """
    Homogeneous ensemble that uses multiple instances of the same base model
    trained on different subsets of data or features.
    
    Supports:
    - Bagging (Bootstrap Aggregating)
    - Random Subspace Method (feature bagging)
    - Pasting (sampling without replacement)
    - Extra Trees (extremely randomized trees)
    """
    
    def __init__(self, 
                 base_estimator: BaseEstimator,
                 n_estimators: int = 10,
                 ensemble_method: str = 'bagging',
                 max_samples: Union[int, float] = 1.0,
                 max_features: Union[int, float] = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = False,
                 random_state: int = None,
                 n_jobs: int = None):
        """
        Initialize homogeneous ensemble.
        
        Args:
            base_estimator: Base model to replicate
            n_estimators: Number of estimators in the ensemble
            ensemble_method: Method ('bagging', 'random_subspace', 'pasting', 'extra_trees')
            max_samples: Number/fraction of samples to draw
            max_features: Number/fraction of features to draw
            bootstrap: Whether to bootstrap samples
            bootstrap_features: Whether to bootstrap features
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.ensemble_method = ensemble_method.lower()
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Internal state
        self.estimators_ = []
        self.feature_indices_ = []
        self.sample_indices_ = []
        self.classes_ = None
        
        # Validate ensemble method
        valid_methods = ['bagging', 'random_subspace', 'pasting', 'extra_trees']
        if self.ensemble_method not in valid_methods:
            logger.warning(f"Unknown homogeneous ensemble method: {ensemble_method}. Using 'bagging'")
            self.ensemble_method = 'bagging'
    
    def _validate_parameters(self, X: np.ndarray):
        """Validate and adjust parameters based on data"""
        n_samples, n_features = X.shape
        
        # Validate max_samples
        if isinstance(self.max_samples, float):
            if not 0.0 < self.max_samples <= 1.0:
                raise ValueError("max_samples must be in (0, 1] when float")
            self.max_samples_ = int(self.max_samples * n_samples)
        else:
            if self.max_samples <= 0 or self.max_samples > n_samples:
                raise ValueError(f"max_samples must be in (0, {n_samples}] when int")
            self.max_samples_ = self.max_samples
        
        # Validate max_features
        if isinstance(self.max_features, float):
            if not 0.0 < self.max_features <= 1.0:
                raise ValueError("max_features must be in (0, 1] when float")
            self.max_features_ = int(self.max_features * n_features)
        else:
            if self.max_features <= 0 or self.max_features > n_features:
                raise ValueError(f"max_features must be in (0, {n_features}] when int")
            self.max_features_ = self.max_features
        
        # Ensure we have at least 1 feature and sample
        self.max_samples_ = max(1, self.max_samples_)
        self.max_features_ = max(1, self.max_features_)
    
    def _generate_indices(self, X: np.ndarray, random_state: np.random.RandomState):
        """Generate sample and feature indices for one estimator"""
        n_samples, n_features = X.shape
        
        # Generate sample indices
        if self.ensemble_method in ['bagging', 'extra_trees']:
            # Bootstrap sampling (with replacement)
            sample_indices = random_state.choice(
                n_samples, size=self.max_samples_, replace=self.bootstrap
            )
        elif self.ensemble_method == 'pasting':
            # Sampling without replacement
            sample_indices = random_state.choice(
                n_samples, size=self.max_samples_, replace=False
            )
        elif self.ensemble_method == 'random_subspace':
            # Use all samples for random subspace
            sample_indices = np.arange(n_samples)
        else:
            sample_indices = np.arange(n_samples)
        
        # Generate feature indices
        if self.ensemble_method == 'random_subspace' or self.bootstrap_features:
            # Random feature selection
            feature_indices = random_state.choice(
                n_features, size=self.max_features_, replace=False
            )
        elif self.ensemble_method == 'extra_trees':
            # For extra trees, use random feature subset
            feature_indices = random_state.choice(
                n_features, size=self.max_features_, replace=False
            )
        else:
            # Use all features
            feature_indices = np.arange(n_features)
        
        return sample_indices, feature_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the homogeneous ensemble.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self: The fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Validate parameters based on data
        self._validate_parameters(X)
        
        # Set up random state
        random_state = np.random.RandomState(self.random_state)
        
        # Clear previous state
        self.estimators_ = []
        self.feature_indices_ = []
        self.sample_indices_ = []
        
        # Store classes for classification
        if hasattr(self.base_estimator, 'predict_proba'):
            self.classes_ = np.unique(y)
        
        # Train each estimator
        for i in range(self.n_estimators):
            # Generate indices for this estimator
            sample_indices, feature_indices = self._generate_indices(X, random_state)
            
            # Create subset of data
            X_subset = X[np.ix_(sample_indices, feature_indices)]
            y_subset = y[sample_indices]
            
            # Clone and fit estimator
            estimator = clone(self.base_estimator)
            
            # Set random state for estimator if supported
            if hasattr(estimator, 'random_state'):
                estimator.random_state = random_state.randint(0, 2**31)
            
            # For extra trees, set additional randomness parameters
            if self.ensemble_method == 'extra_trees':
                if hasattr(estimator, 'max_features'):
                    estimator.max_features = min(self.max_features_, estimator.max_features or X.shape[1])
                if hasattr(estimator, 'bootstrap'):
                    estimator.bootstrap = False  # Extra trees typically don't use bootstrap
                if hasattr(estimator, 'criterion') and hasattr(estimator, 'splitter'):
                    # Enable extra randomness for tree-based models
                    estimator.splitter = 'random'
            
            # Fit the estimator
            estimator.fit(X_subset, y_subset)
            
            # Store estimator and indices
            self.estimators_.append(estimator)
            self.feature_indices_.append(feature_indices)
            self.sample_indices_.append(sample_indices)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        X = np.asarray(X)
        
        # Collect predictions from all estimators
        predictions = []
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_indices_):
            # Use only the features this estimator was trained on
            X_subset = X[:, feature_indices]
            pred = estimator.predict(X_subset)
            predictions.append(pred)
        
        # Combine predictions
        predictions = np.array(predictions)
        
        # Average for regression, majority vote for classification
        if self.classes_ is not None:
            # Classification: majority vote
            from scipy import stats
            return stats.mode(predictions, axis=0, keepdims=False)[0]
        else:
            # Regression: average
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Class probabilities
        """
        if self.classes_ is None:
            raise ValueError("predict_proba is only available for classification tasks")
        
        X = np.asarray(X)
        
        # Collect probability predictions from all estimators
        probas = []
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_indices_):
            if hasattr(estimator, 'predict_proba'):
                X_subset = X[:, feature_indices]
                proba = estimator.predict_proba(X_subset)
                
                # Ensure all estimators predict the same number of classes
                if proba.shape[1] < len(self.classes_):
                    # Pad with zeros for missing classes
                    padded_proba = np.zeros((proba.shape[0], len(self.classes_)))
                    padded_proba[:, :proba.shape[1]] = proba
                    proba = padded_proba
                
                probas.append(proba)
        
        if not probas:
            raise ValueError("No estimators support predict_proba")
        
        # Average probabilities
        return np.mean(probas, axis=0)


class HeterogeneousEnsemble(EnsembleModel):
    """
    Enhanced heterogeneous ensemble with advanced combining strategies.
    
    Extends the basic EnsembleModel with additional methods:
    - Blending (using holdout validation set)
    - Dynamic ensemble selection
    - Advanced cross-validation stacking
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize enhanced heterogeneous ensemble."""
        # Extract additional parameters before passing to parent
        self.blending_holdout = kwargs.pop('blending_holdout', 0.2)
        self.dynamic_selection = kwargs.pop('dynamic_selection', False)
        self.cv_folds = kwargs.pop('cv_folds', 5)
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        Fit the enhanced ensemble with support for new methods.
        
        Args:
            X: Training features
            y: Training targets
            sample_weight: Sample weights
            
        Returns:
            self: The fitted estimator
        """
        if self.ensemble_method == 'blending':
            return self._fit_blending(X, y, sample_weight)
        elif self.ensemble_method == 'cv_stacking':
            return self._fit_cv_stacking(X, y, sample_weight)
        else:
            # Use parent implementation for other methods
            return super().fit(X, y, sample_weight)
    
    def _fit_blending(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Fit ensemble using blending (holdout validation set)"""
        # Split data into training and holdout sets
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=self.blending_holdout, random_state=42
        )
        
        # Generate predictions on holdout set from base models
        meta_features = []
        for model_wrapper in self.models:
            try:
                if self.task_type == 'regression':
                    preds = model_wrapper.predict(X_holdout)
                    meta_features.append(preds)
                else:
                    # For classification, use probabilities if available
                    probs = model_wrapper.predict_proba(X_holdout)
                    if probs is not None:
                        meta_features.extend([probs[:, i] for i in range(probs.shape[1])])
                    else:
                        preds = model_wrapper.predict(X_holdout)
                        meta_features.append(preds)
            except Exception as e:
                logger.warning(f"Error generating meta-features from {model_wrapper.name}: {str(e)}")
        
        if meta_features:
            meta_features = np.column_stack(meta_features)
            
            # Create and fit meta-model
            if self.task_type == 'regression':
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)
            else:
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
                self.classes_ = np.unique(y)
            
            self.meta_model.fit(meta_features, y_holdout, sample_weight=sample_weight)
        
        return self
    
    def _fit_cv_stacking(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Fit ensemble using cross-validation stacking"""
        # Generate out-of-fold predictions
        meta_features = []
        
        for model_wrapper in self.models:
            try:
                if self.task_type == 'regression':
                    oof_preds = cross_val_predict(
                        model_wrapper.model, X, y, cv=self.cv_folds, method='predict'
                    )
                    meta_features.append(oof_preds)
                else:
                    # For classification
                    if hasattr(model_wrapper.model, 'predict_proba'):
                        oof_probs = cross_val_predict(
                            model_wrapper.model, X, y, cv=self.cv_folds, method='predict_proba'
                        )
                        meta_features.extend([oof_probs[:, i] for i in range(oof_probs.shape[1])])
                    else:
                        oof_preds = cross_val_predict(
                            model_wrapper.model, X, y, cv=self.cv_folds, method='predict'
                        )
                        meta_features.append(oof_preds)
            except Exception as e:
                logger.warning(f"Error generating CV meta-features from {model_wrapper.name}: {str(e)}")
        
        if meta_features:
            meta_features = np.column_stack(meta_features)
            
            # Create and fit meta-model
            if self.task_type == 'regression':
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(alpha=1.0)
            else:
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
                self.classes_ = np.unique(y)
            
            self.meta_model.fit(meta_features, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with enhanced ensemble methods.
        
        Args:
            X: Features
            
        Returns:
            np.ndarray: Predictions
        """
        if self.ensemble_method in ['blending', 'cv_stacking'] and self.meta_model is not None:
            # Generate meta-features for new data
            meta_features = self._generate_meta_features(X)
            return self.meta_model.predict(meta_features)
        elif self.ensemble_method == 'dynamic_selection':
            return self._predict_dynamic_selection(X)
        else:
            # Use parent implementation
            return super().predict(X)
    
    def _predict_dynamic_selection(self, X: np.ndarray) -> np.ndarray:
        """Dynamic ensemble selection based on local competence"""
        # Simplified dynamic selection: weight models based on recent performance
        predictions = []
        weights = []
        
        for model_wrapper in self.models:
            try:
                pred = model_wrapper.predict(X)
                predictions.append(pred)
                
                # Use recent performance as weight
                recent_perf = model_wrapper.get_recent_performance()
                if self.task_type == 'regression':
                    # For regression, lower error is better
                    weight = 1.0 / (recent_perf + 1e-10) if recent_perf > 0 else 1.0
                else:
                    # For classification, higher accuracy is better
                    weight = recent_perf if recent_perf > 0 else 0.5
                
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Error in dynamic selection from {model_wrapper.name}: {str(e)}")
        
        if predictions:
            predictions = np.array(predictions)
            weights = np.array(weights)
            
            # Normalize weights
            weights = weights / (np.sum(weights) + 1e-10)
            
            # Weighted average
            weights = weights.reshape(-1, 1)
            return np.sum(predictions * weights, axis=0)
        else:
            raise ValueError("No valid predictions available")


class EnsembleFactory:
    """
    Factory class for creating different types of ensembles.
    """
    
    @staticmethod
    def create_homogeneous_ensemble(
        base_estimator: BaseEstimator,
        ensemble_type: str = 'bagging',
        n_estimators: int = 10,
        **kwargs
    ) -> HomogeneousEnsemble:
        """
        Create a homogeneous ensemble.
        
        Args:
            base_estimator: Base model to replicate
            ensemble_type: Type of ensemble ('bagging', 'random_subspace', 'pasting', 'extra_trees')
            n_estimators: Number of estimators
            **kwargs: Additional parameters
            
        Returns:
            HomogeneousEnsemble: Configured ensemble
        """
        return HomogeneousEnsemble(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            ensemble_method=ensemble_type,
            **kwargs
        )
    
    @staticmethod
    def create_heterogeneous_ensemble(
        models: List[Any],
        ensemble_type: str = 'weighted_average',
        model_types: List[str] = None,
        **kwargs
    ) -> HeterogeneousEnsemble:
        """
        Create a heterogeneous ensemble.
        
        Args:
            models: List of different models
            ensemble_type: Type of ensemble ('weighted_average', 'stacking', 'blending', etc.)
            model_types: List of model types
            **kwargs: Additional parameters
            
        Returns:
            HeterogeneousEnsemble: Configured ensemble
        """
        return HeterogeneousEnsemble(
            models=models,
            model_types=model_types,
            ensemble_method=ensemble_type,
            **kwargs
        )
    
    @staticmethod
    def create_bagging_ensemble(
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_samples: Union[int, float] = 1.0,
        bootstrap: bool = True,
        **kwargs
    ) -> HomogeneousEnsemble:
        """Create a bagging ensemble."""
        return EnsembleFactory.create_homogeneous_ensemble(
            base_estimator=base_estimator,
            ensemble_type='bagging',
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            **kwargs
        )
    
    @staticmethod
    def create_random_subspace_ensemble(
        base_estimator: BaseEstimator,
        n_estimators: int = 10,
        max_features: Union[int, float] = 0.5,
        **kwargs
    ) -> HomogeneousEnsemble:
        """Create a random subspace ensemble."""
        return EnsembleFactory.create_homogeneous_ensemble(
            base_estimator=base_estimator,
            ensemble_type='random_subspace',
            n_estimators=n_estimators,
            max_features=max_features,
            **kwargs
        )
    
    @staticmethod
    def create_stacking_ensemble(
        models: List[Any],
        meta_learner: BaseEstimator = None,
        cv_folds: int = 5,
        **kwargs
    ) -> HeterogeneousEnsemble:
        """Create a stacking ensemble with cross-validation."""
        if meta_learner is None:
            from sklearn.linear_model import Ridge
            meta_learner = Ridge()
        
        return EnsembleFactory.create_heterogeneous_ensemble(
            models=models,
            ensemble_type='cv_stacking',
            cv_folds=cv_folds,
            **kwargs
        )
    
    @staticmethod
    def create_blending_ensemble(
        models: List[Any],
        holdout_size: float = 0.2,
        **kwargs
    ) -> HeterogeneousEnsemble:
        """Create a blending ensemble."""
        return EnsembleFactory.create_heterogeneous_ensemble(
            models=models,
            ensemble_type='blending',
            blending_holdout=holdout_size,
            **kwargs
        )


def main():
    """Example usage of the ensemble learning module."""
    from sklearn.datasets import make_regression, make_classification
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    # Create regression dataset
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_reg_scaled = scaler.fit_transform(X_reg)
    X_train, X_test, y_train, y_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)
    
    print("=== Homogeneous Ensemble Examples ===")
    
    # Example 1: Bagging Ensemble
    print("\n1. Bagging Ensemble with Linear Regression")
    bagging_ensemble = EnsembleFactory.create_bagging_ensemble(
        base_estimator=LinearRegression(),
        n_estimators=10,
        max_samples=0.8,
        random_state=42
    )
    bagging_ensemble.fit(X_train, y_train)
    bagging_preds = bagging_ensemble.predict(X_test)
    bagging_rmse = np.sqrt(mean_squared_error(y_test, bagging_preds))
    print(f"Bagging Ensemble RMSE: {bagging_rmse:.4f}")
    
    # Example 2: Random Subspace Ensemble
    print("\n2. Random Subspace Ensemble with RandomForest")
    rs_ensemble = EnsembleFactory.create_random_subspace_ensemble(
        base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
        n_estimators=5,
        max_features=0.6,
        random_state=42
    )
    rs_ensemble.fit(X_train, y_train)
    rs_preds = rs_ensemble.predict(X_test)
    rs_rmse = np.sqrt(mean_squared_error(y_test, rs_preds))
    print(f"Random Subspace Ensemble RMSE: {rs_rmse:.4f}")
    
    print("\n=== Heterogeneous Ensemble Examples ===")
    
    # Train individual models for heterogeneous ensemble
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    
    en = ElasticNet(alpha=0.1, random_state=42)
    en.fit(X_train, y_train)
    
    models = [rf, gb, en]
    
    # Example 3: Stacking Ensemble
    print("\n3. Cross-Validation Stacking Ensemble")
    stacking_ensemble = EnsembleFactory.create_stacking_ensemble(
        models=models,
        cv_folds=5,
        task_type='regression'
    )
    stacking_ensemble.fit(X_train, y_train)
    stacking_preds = stacking_ensemble.predict(X_test)
    stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_preds))
    print(f"Stacking Ensemble RMSE: {stacking_rmse:.4f}")
    
    # Example 4: Blending Ensemble
    print("\n4. Blending Ensemble")
    blending_ensemble = EnsembleFactory.create_blending_ensemble(
        models=models,
        holdout_size=0.2,
        task_type='regression'
    )
    blending_ensemble.fit(X_train, y_train)
    blending_preds = blending_ensemble.predict(X_test)
    blending_rmse = np.sqrt(mean_squared_error(y_test, blending_preds))
    print(f"Blending Ensemble RMSE: {blending_rmse:.4f}")
    
    # Compare individual models
    print("\n=== Individual Model Performance ===")
    rf_preds = rf.predict(X_test)
    gb_preds = gb.predict(X_test)
    en_preds = en.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
    en_rmse = np.sqrt(mean_squared_error(y_test, en_preds))
    
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
    print(f"ElasticNet RMSE: {en_rmse:.4f}")
    
    print("\n=== Classification Example ===")
    
    # Create classification dataset
    X_cls, y_cls = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                      n_redundant=2, n_classes=3, random_state=42)
    X_cls_scaled = scaler.fit_transform(X_cls)
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X_cls_scaled, y_cls, test_size=0.2, random_state=42
    )
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Homogeneous ensemble for classification
    print("\n5. Bagging Ensemble for Classification")
    cls_bagging = EnsembleFactory.create_bagging_ensemble(
        base_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_estimators=10,
        max_samples=0.8,
        random_state=42
    )
    cls_bagging.fit(X_cls_train, y_cls_train)
    cls_preds = cls_bagging.predict(X_cls_test)
    cls_acc = accuracy_score(y_cls_test, cls_preds)
    print(f"Bagging Classification Accuracy: {cls_acc:.4f}")
    
    print("\n=== Testing automatic ensemble creation ===")
    auto_ensemble, training_results = train_model_ensemble(
        X_reg_scaled, y_reg,
        task_type='regression',
        ensemble_method='weighted_average',
        dynamic_weighting=True
    )
    
    auto_preds = auto_ensemble.predict(X_test)
    auto_rmse = np.sqrt(mean_squared_error(y_test, auto_preds))
    print(f"Automatic Ensemble RMSE: {auto_rmse:.4f}")
    print("Training results:", json.dumps(training_results, indent=2))


if __name__ == "__main__":
    main()