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
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

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
        valid_types = ['sklearn', 'tensorflow', 'pytorch', 'custom']
        if self.model_type not in valid_types:
            logger.warning(f"Unknown model type: {model_type}. Must be one of {valid_types}")
            self.model_type = 'custom'
        
        # Check if model is compatible with its claimed type
        self._validate_model_compatibility()
    
    def _validate_model_compatibility(self):
        """Validate that the model matches its claimed type"""
        if self.model_type == 'sklearn' and not hasattr(self.model, 'predict'):
            logger.warning("Model claimed to be sklearn but lacks predict method")
        
        elif self.model_type == 'tensorflow' and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow model specified but TensorFlow is not available")
            
        elif self.model_type == 'pytorch' and not PYTORCH_AVAILABLE:
            logger.warning("PyTorch model specified but PyTorch is not available")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction using the wrapped model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Model predictions
        """
        if self.model_type == 'sklearn':
            return self.model.predict(X)
            
        elif self.model_type == 'tensorflow':
            # Handle TensorFlow models - assumes model has predict method
            return self.model.predict(X).flatten()
            
        elif self.model_type == 'pytorch':
            # Handle PyTorch models
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            else:
                # Convert to torch tensor if needed
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32)
                
                # Set model to eval mode
                self.model.eval()
                with torch.no_grad():
                    return self.model(X).cpu().numpy()
                
        elif self.model_type == 'custom':
            # Assume custom model implements predict method
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            else:
                # Try direct call
                return self.model(X)
        
        raise ValueError(f"Prediction not implemented for model type: {self.model_type}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions for classification models.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Probability predictions or None if not applicable
        """
        try:
            if self.model_type == 'sklearn' and hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
                
            elif self.model_type == 'tensorflow':
                # Most TF classification models output probabilities directly
                probs = self.model.predict(X)
                # Ensure we have proper probabilities format
                if probs.shape[-1] == 1:  # Binary case
                    return np.hstack([1-probs, probs])
                return probs
                
            elif self.model_type == 'pytorch':
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                else:
                    # Convert to torch tensor if needed
                    if not isinstance(X, torch.Tensor):
                        X = torch.tensor(X, dtype=torch.float32)
                    
                    # Set model to eval mode
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.model(X)
                        # Apply softmax if needed (output logits)
                        if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                        else:
                            # Binary classification
                            probs = torch.sigmoid(outputs)
                            probs = torch.cat([1-probs, probs], dim=1)
                        return probs.cpu().numpy()
                    
            elif self.model_type == 'custom':
                # Try predict_proba if available
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)
                
            # Fallback - return None to indicate probabilities not available
            return None
            
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
            if self.model_type == 'sklearn':
                if hasattr(self.model, 'feature_importances_'):
                    return dict(enumerate(self.model.feature_importances_))
                elif hasattr(self.model, 'coef_'):
                    return dict(enumerate(np.abs(self.model.coef_).flatten()))
                
            elif self.model_type == 'tensorflow' or self.model_type == 'pytorch':
                # For deep learning models, assume the model has a method for this
                if hasattr(self.model, 'feature_importance') and X is not None:
                    return self.model.feature_importance(X)
                
            elif self.model_type == 'custom':
                if hasattr(self.model, 'feature_importances_'):
                    return dict(enumerate(self.model.feature_importances_))
                elif hasattr(self.model, 'get_feature_importance'):
                    return self.model.get_feature_importance()
                    
            # No feature importance available
            return {}
            
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
        if self.model_type == 'sklearn':
            joblib.dump(self.model, f"{path}_model.pkl")
        elif self.model_type == 'tensorflow':
            self.model.save(f"{path}_model")
        elif self.model_type == 'pytorch':
            if hasattr(self.model, 'save'):
                self.model.save(f"{path}_model.pt")
            else:
                torch.save(self.model.state_dict(), f"{path}_model.pt")
        elif self.model_type == 'custom':
            # Try joblib as default
            try:
                joblib.dump(self.model, f"{path}_model.pkl")
            except:
                logger.warning(f"Could not save model of type {type(self.model)}")
        
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
        
        if model_type == 'sklearn':
            model = joblib.load(f"{path}_model.pkl")
        elif model_type == 'tensorflow':
            if TENSORFLOW_AVAILABLE:
                model = tf.keras.models.load_model(f"{path}_model")
            else:
                raise ImportError("TensorFlow is required to load this model")
        elif model_type == 'pytorch':
            if PYTORCH_AVAILABLE:
                # Need model class to load state dict - use basic wrapper
                class BasicTorchModel(torch.nn.Module):
                    def forward(self, x):
                        # Will be overridden when state dict is loaded
                        return x
                
                model = BasicTorchModel()
                model.load_state_dict(torch.load(f"{path}_model.pt"))
            else:
                raise ImportError("PyTorch is required to load this model")
        elif model_type == 'custom':
            try:
                model = joblib.load(f"{path}_model.pkl")
            except:
                raise ValueError(f"Could not load custom model from {path}")
        
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
        valid_methods = ['simple_average', 'weighted_average', 'stacking', 'majority_vote']
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


def main():
    """Example usage of the ensemble learning module."""
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Load dataset
    boston = load_boston()
    X, y = boston.data, boston.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("Training individual models...")
    
    # Train individual models
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    
    en = ElasticNet(alpha=0.1, random_state=42)
    en.fit(X_train, y_train)
    
    # Create model wrappers
    rf_wrapper = ModelWrapper(rf, model_type='sklearn', name='RandomForest')
    gb_wrapper = ModelWrapper(gb, model_type='sklearn', name='GradientBoosting')
    en_wrapper = ModelWrapper(en, model_type='sklearn', name='ElasticNet')
    
    # Create ensemble
    print("Creating ensemble model...")
    ensemble = EnsembleModel(
        models=[rf_wrapper, gb_wrapper, en_wrapper],
        ensemble_method='weighted_average',
        task_type='regression'
    )
    
    # Evaluate models
    print("Evaluating models...")
    rf_preds = rf.predict(X_test)
    gb_preds = gb.predict(X_test)
    en_preds = en.predict(X_test)
    ensemble_preds = ensemble.predict(X_test)
    
    rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
    gb_rmse = mean_squared_error(y_test, gb_preds, squared=False)
    en_rmse = mean_squared_error(y_test, en_preds, squared=False)
    ensemble_rmse = mean_squared_error(y_test, ensemble_preds, squared=False)
    
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
    print(f"ElasticNet RMSE: {en_rmse:.4f}")
    print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
    
    # Test automatic ensemble creation
    print("\nTesting automatic ensemble creation...")
    auto_ensemble, training_results = train_model_ensemble(
        X_scaled, y,
        task_type='regression',
        ensemble_method='weighted_average',
        dynamic_weighting=True
    )
    
    auto_preds = auto_ensemble.predict(X_test)
    auto_rmse = mean_squared_error(y_test, auto_preds, squared=False)
    print(f"Automatic Ensemble RMSE: {auto_rmse:.4f}")
    print("Training results:", json.dumps(training_results, indent=2))


if __name__ == "__main__":
    main()