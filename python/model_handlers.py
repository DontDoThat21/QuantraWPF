#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Type Handlers Module

This module provides abstracted handlers for different model types (sklearn, TensorFlow, PyTorch, custom)
to eliminate repetitive code patterns and improve maintainability in the ensemble learning framework.
"""

import numpy as np
import logging
import os
import joblib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger('model_handlers')

# Check for framework availability
TENSORFLOW_AVAILABLE = False
PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    pass


class BaseModelHandler(ABC):
    """Abstract base class for model type handlers."""
    
    @abstractmethod
    def validate_model_compatibility(self, model: Any) -> bool:
        """Validate that the model is compatible with this handler."""
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using the model."""
        pass
    
    @abstractmethod
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for classification models."""
        pass
    
    @abstractmethod
    def get_feature_importance(self, model: Any, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get feature importances if available."""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, path: str) -> None:
        """Save the model to the specified path."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> Any:
        """Load the model from the specified path."""
        pass


class SklearnModelHandler(BaseModelHandler):
    """Handler for scikit-learn models."""
    
    def validate_model_compatibility(self, model: Any) -> bool:
        """Validate that the model has sklearn-compatible interface."""
        return hasattr(model, 'predict')
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using sklearn model."""
        return model.predict(X)
    
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for sklearn classification models."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None
    
    def get_feature_importance(self, model: Any, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get feature importances from sklearn model."""
        if hasattr(model, 'feature_importances_'):
            return dict(enumerate(model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(enumerate(np.abs(model.coef_).flatten()))
        return {}
    
    def save_model(self, model: Any, path: str) -> None:
        """Save sklearn model using joblib."""
        joblib.dump(model, f"{path}_model.pkl")
    
    def load_model(self, path: str) -> Any:
        """Load sklearn model using joblib."""
        return joblib.load(f"{path}_model.pkl")


class TensorflowModelHandler(BaseModelHandler):
    """Handler for TensorFlow/Keras models."""
    
    def validate_model_compatibility(self, model: Any) -> bool:
        """Validate that TensorFlow is available and model is compatible."""
        return TENSORFLOW_AVAILABLE and hasattr(model, 'predict')
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using TensorFlow model."""
        return model.predict(X).flatten()
    
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for TensorFlow classification models."""
        probs = model.predict(X)
        # Ensure we have proper probabilities format
        if probs.shape[-1] == 1:  # Binary case
            return np.hstack([1-probs, probs])
        return probs
    
    def get_feature_importance(self, model: Any, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get feature importances from TensorFlow model."""
        # For deep learning models, assume the model has a method for this
        if hasattr(model, 'feature_importance') and X is not None:
            return model.feature_importance(X)
        return {}
    
    def save_model(self, model: Any, path: str) -> None:
        """Save TensorFlow model."""
        model.save(f"{path}_model")
    
    def load_model(self, path: str) -> Any:
        """Load TensorFlow model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to load this model")
        import tensorflow as tf
        return tf.keras.models.load_model(f"{path}_model")


class PytorchModelHandler(BaseModelHandler):
    """Handler for PyTorch models."""
    
    def validate_model_compatibility(self, model: Any) -> bool:
        """Validate that PyTorch is available and model is compatible."""
        return PYTORCH_AVAILABLE and hasattr(model, '__call__')
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using PyTorch model."""
        if hasattr(model, 'predict'):
            return model.predict(X)
        else:
            # Convert to torch tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            
            # Set model to eval mode
            model.eval()
            with torch.no_grad():
                return model(X).cpu().numpy()
    
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for PyTorch classification models."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # Convert to torch tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            
            # Set model to eval mode
            model.eval()
            with torch.no_grad():
                outputs = model(X)
                # Apply softmax if needed (output logits)
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    # Binary classification
                    probs = torch.sigmoid(outputs)
                    probs = torch.cat([1-probs, probs], dim=1)
                return probs.cpu().numpy()
    
    def get_feature_importance(self, model: Any, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get feature importances from PyTorch model."""
        # For deep learning models, assume the model has a method for this
        if hasattr(model, 'feature_importance') and X is not None:
            return model.feature_importance(X)
        return {}
    
    def save_model(self, model: Any, path: str) -> None:
        """Save PyTorch model."""
        if hasattr(model, 'save'):
            model.save(f"{path}_model.pt")
        else:
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required to save this model")
            import torch
            torch.save(model.state_dict(), f"{path}_model.pt")
    
    def load_model(self, path: str) -> Any:
        """Load PyTorch model."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required to load this model")
        import torch
        
        # Need model class to load state dict - use basic wrapper
        class BasicTorchModel(torch.nn.Module):
            def forward(self, x):
                # Will be overridden when state dict is loaded
                return x
        
        model = BasicTorchModel()
        model.load_state_dict(torch.load(f"{path}_model.pt"))
        return model


class CustomModelHandler(BaseModelHandler):
    """Handler for custom/unknown model types."""
    
    def validate_model_compatibility(self, model: Any) -> bool:
        """Custom models are always considered compatible as fallback."""
        return True
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions using custom model."""
        if hasattr(model, 'predict'):
            return model.predict(X)
        else:
            # Try direct call
            return model(X)
    
    def predict_proba(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for custom classification models."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None
    
    def get_feature_importance(self, model: Any, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get feature importances from custom model."""
        if hasattr(model, 'feature_importances_'):
            return dict(enumerate(model.feature_importances_))
        elif hasattr(model, 'get_feature_importance'):
            return model.get_feature_importance()
        return {}
    
    def save_model(self, model: Any, path: str) -> None:
        """Save custom model using joblib."""
        try:
            joblib.dump(model, f"{path}_model.pkl")
        except Exception as e:
            logger.warning(f"Could not save model of type {type(model)}: {str(e)}")
            raise
    
    def load_model(self, path: str) -> Any:
        """Load custom model using joblib."""
        try:
            return joblib.load(f"{path}_model.pkl")
        except Exception as e:
            raise ValueError(f"Could not load custom model from {path}: {str(e)}")


class ModelHandlerFactory:
    """Factory class to create appropriate model handlers."""
    
    _handlers = {
        'sklearn': SklearnModelHandler(),
        'tensorflow': TensorflowModelHandler(),
        'pytorch': PytorchModelHandler(),
        'custom': CustomModelHandler()
    }
    
    @classmethod
    def get_handler(cls, model_type: str) -> BaseModelHandler:
        """Get the appropriate handler for the given model type."""
        model_type = model_type.lower()
        if model_type not in cls._handlers:
            logger.warning(f"Unknown model type: {model_type}. Using custom handler.")
            return cls._handlers['custom']
        return cls._handlers[model_type]
    
    @classmethod
    def auto_detect_handler(cls, model: Any) -> BaseModelHandler:
        """Auto-detect the appropriate handler for a model."""
        # Try TensorFlow first
        if TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
            return cls._handlers['tensorflow']
        
        # Try PyTorch
        if PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            return cls._handlers['pytorch']
        
        # Try sklearn (has predict and fit methods)
        if hasattr(model, 'predict') and hasattr(model, 'fit'):
            return cls._handlers['sklearn']
        
        # Default to custom
        return cls._handlers['custom']
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported model types."""
        return list(cls._handlers.keys())