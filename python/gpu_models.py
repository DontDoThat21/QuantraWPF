"""
GPU Model Wrappers for Quantra Trading Platform

This module provides wrapper classes for machine learning models that utilize GPU acceleration.
It includes an abstract base class for consistent interface across frameworks,
and specific implementations for PyTorch and TensorFlow.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np

# Import local modules
from gpu_utils import GPUManager, get_default_gpu_manager

# Set up logging
logger = logging.getLogger(__name__)


class GPUModelBase(ABC):
    """
    Abstract base class for GPU-accelerated models.
    
    This provides a unified interface for various ML frameworks (PyTorch, TensorFlow)
    with consistent methods for training, prediction, and data movement between CPU/GPU.
    """
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize the base GPU model.
        
        Args:
            gpu_manager: GPUManager instance for device handling
        """
        self.gpu_manager = gpu_manager or get_default_gpu_manager()
        self.model = None
        self.is_trained = False
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.framework_name = "undefined"
    
    @abstractmethod
    def to_gpu(self) -> None:
        """Move the model to GPU."""
        pass
    
    @abstractmethod
    def to_cpu(self) -> None:
        """Move the model to CPU."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """
        Train the model on the provided data.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional training parameters
            
        Returns:
            Training history or model itself
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions using the model.
        
        Args:
            X: Input features
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions
        """
        pass
    
    def move_data_to_gpu(self, data: np.ndarray) -> Any:
        """
        Move data to the GPU in the appropriate format for the current framework.
        
        Args:
            data: NumPy array to move to GPU
            
        Returns:
            Data in framework-specific GPU format
        """
        raise NotImplementedError("Implemented in subclasses")
    
    def move_data_to_cpu(self, data: Any) -> np.ndarray:
        """
        Move data from GPU to CPU as a NumPy array.
        
        Args:
            data: Framework-specific GPU tensor/array
            
        Returns:
            Data as NumPy array on CPU
        """
        raise NotImplementedError("Implemented in subclasses")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the model.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "training_time_seconds": self.training_time,
            "prediction_time_seconds": self.prediction_time,
            "is_trained": self.is_trained,
            "framework": self.framework_name
        }
    
    def __repr__(self) -> str:
        """String representation of the GPU model."""
        metrics = self.get_performance_metrics()
        return (f"{self.__class__.__name__}("
                f"framework={metrics['framework']}, "
                f"trained={metrics['is_trained']}, "
                f"training_time={metrics['training_time_seconds']:.2f}s)")


class PyTorchGPUModel(GPUModelBase):
    """
    GPU-accelerated PyTorch model implementation.
    
    This wrapper handles device management, data transfer, and provides
    a sklearn-compatible interface for PyTorch models.
    """
    
    def __init__(self, model_builder: Callable, gpu_manager: Optional[GPUManager] = None, **model_kwargs):
        """
        Initialize PyTorch GPU model.
        
        Args:
            model_builder: Function that builds and returns a PyTorch model
            gpu_manager: GPUManager instance for device handling
            **model_kwargs: Additional arguments to pass to model_builder
        """
        super().__init__(gpu_manager)
        self.framework_name = "pytorch"
        
        try:
            import torch
            self.torch = torch
            self.device = self.gpu_manager.get_pytorch_device()
            
            # Build the model
            self.model = model_builder(**model_kwargs)
            
            # Move model to GPU if available
            if self.torch.cuda.is_available():
                self.to_gpu()
                logger.info(f"PyTorch model created on {self.device}")
            else:
                logger.warning("PyTorch GPU not available, using CPU")
                self.device = self.torch.device("cpu")
        
        except ImportError:
            logger.error("PyTorch is not installed. Cannot create PyTorch GPU model.")
            self.device = None
    
    def to_gpu(self) -> None:
        """Move the model to GPU."""
        if self.model is not None and self.device.type == "cuda":
            self.model.to(self.device)
    
    def to_cpu(self) -> None:
        """Move the model to CPU."""
        if self.model is not None:
            cpu_device = self.torch.device("cpu")
            self.model.to(cpu_device)
    
    def move_data_to_gpu(self, data: np.ndarray) -> 'torch.Tensor':
        """
        Move NumPy array to GPU as PyTorch tensor.
        
        Args:
            data: NumPy array
        
        Returns:
            PyTorch tensor on GPU
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        tensor = self.torch.tensor(data, device=self.device)
        if data.dtype == np.float64 or data.dtype == np.float32:
            tensor = tensor.float()
        
        return tensor
    
    def move_data_to_cpu(self, tensor: 'torch.Tensor') -> np.ndarray:
        """
        Move PyTorch tensor from GPU to CPU as NumPy array.
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            NumPy array
        """
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100,
            batch_size: int = 32, 
            validation_split: float = 0.1,
            optimizer_cls=None,
            loss_fn=None,
            learning_rate: float = 0.001,
            **kwargs) -> Dict[str, List[float]]:
        """
        Train the PyTorch model.
        
        Args:
            X: Input features
            y: Target values
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            optimizer_cls: PyTorch optimizer class
            loss_fn: PyTorch loss function
            learning_rate: Learning rate for optimizer
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        
        # Set default optimizer and loss function if not provided
        if optimizer_cls is None:
            optimizer_cls = self.torch.optim.Adam
        
        if loss_fn is None:
            if len(y.shape) > 1 and y.shape[1] > 1:
                # Multi-class classification
                loss_fn = self.torch.nn.CrossEntropyLoss()
            else:
                # Regression or binary classification
                loss_fn = self.torch.nn.MSELoss()
        
        # Create optimizer
        optimizer = optimizer_cls(self.model.parameters(), lr=learning_rate)
        
        # Convert data to PyTorch tensors on the appropriate device
        X_tensor = self.move_data_to_gpu(X)
        y_tensor = self.move_data_to_gpu(y)
        
        # Split data for validation if requested
        if validation_split > 0:
            val_size = int(len(X) * validation_split)
            X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
            y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None
        
        # Training history
        history = {
            "loss": [],
            "val_loss": [] if validation_split > 0 else None
        }
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = self.torch.randperm(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Batch training
            total_loss = 0
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(X_train))
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            history["loss"].append(avg_loss)
            
            # Validation
            if validation_split > 0 and X_val is not None:
                self.model.eval()
                with self.torch.no_grad():
                    y_val_pred = self.model(X_val)
                    val_loss = loss_fn(y_val_pred, y_val).item()
                    history["val_loss"].append(val_loss)
                self.model.train()
                
                if epoch % 10 == 0 or epoch == epochs - 1:
                    logger.info(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}")
            elif epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        # Set model to evaluation mode
        self.model.eval()
        self.is_trained = True
        
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Generate predictions using the PyTorch model.
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions as NumPy array
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model is not initialized or not trained")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert data to PyTorch tensor
        X_tensor = self.move_data_to_gpu(X)
        
        # Predict in batches to avoid memory issues
        start_time = time.time()
        predictions = []
        
        with self.torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                X_batch = X_tensor[i:i + batch_size]
                batch_preds = self.model(X_batch)
                predictions.append(batch_preds)
        
        # Concatenate batch predictions
        full_preds = self.torch.cat(predictions, dim=0)
        
        # Convert to NumPy
        result = self.move_data_to_cpu(full_preds)
        
        self.prediction_time = time.time() - start_time
        return result


class TensorFlowGPUModel(GPUModelBase):
    """
    GPU-accelerated TensorFlow/Keras model implementation.
    
    This wrapper handles device management, data transfer, and provides
    a consistent interface for TensorFlow models.
    """
    
    def __init__(self, model_builder: Callable, gpu_manager: Optional[GPUManager] = None, **model_kwargs):
        """
        Initialize TensorFlow GPU model.
        
        Args:
            model_builder: Function that builds and returns a TensorFlow/Keras model
            gpu_manager: GPUManager instance for device handling
            **model_kwargs: Additional arguments to pass to model_builder
        """
        super().__init__(gpu_manager)
        self.framework_name = "tensorflow"
        
        try:
            import tensorflow as tf
            self.tf = tf
            
            # Configure TensorFlow to use GPU
            self.gpu_initialized = self.gpu_manager.initialize_tensorflow()
            
            # Build the model
            self.model = model_builder(**model_kwargs)
            
            if self.gpu_initialized:
                logger.info(f"TensorFlow model created with GPU acceleration")
            else:
                logger.warning("TensorFlow GPU not available, using CPU")
        
        except ImportError:
            logger.error("TensorFlow is not installed. Cannot create TensorFlow GPU model.")
            self.model = None
    
    def to_gpu(self) -> None:
        """Move operations to GPU (handled automatically by TensorFlow)."""
        # TensorFlow handles device placement automatically
        pass
    
    def to_cpu(self) -> None:
        """Move operations to CPU."""
        # Force TensorFlow to use CPU
        try:
            import tensorflow as tf
            with tf.device('/CPU:0'):
                if hasattr(self.model, 'get_weights'):
                    weights = self.model.get_weights()
                    self.model.set_weights(weights)
        except (ImportError, AttributeError):
            pass
    
    def move_data_to_gpu(self, data: np.ndarray) -> 'tf.Tensor':
        """
        Prepare data for TensorFlow GPU operations.
        
        Args:
            data: NumPy array
            
        Returns:
            TensorFlow tensor
        """
        # TensorFlow automatically handles device placement
        return self.tf.convert_to_tensor(data)
    
    def move_data_to_cpu(self, tensor: 'tf.Tensor') -> np.ndarray:
        """
        Move TensorFlow tensor to CPU as NumPy array.
        
        Args:
            tensor: TensorFlow tensor
            
        Returns:
            NumPy array
        """
        return tensor.numpy()
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.1,
            compile_kwargs: Dict = None,
            **kwargs) -> Any:
        """
        Train the TensorFlow model.
        
        Args:
            X: Input features
            y: Target values
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            compile_kwargs: Arguments for model.compile()
            **kwargs: Additional training parameters passed to model.fit()
            
        Returns:
            Keras training history
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        
        # Set default compile arguments if not provided
        if compile_kwargs is None:
            compile_kwargs = {
                'optimizer': 'adam',
                'loss': 'mse' if len(y.shape) <= 1 or y.shape[1] == 1 else 'categorical_crossentropy',
            }
            
            # Add metrics if not provided
            if 'metrics' not in compile_kwargs:
                if compile_kwargs['loss'] == 'categorical_crossentropy':
                    compile_kwargs['metrics'] = ['accuracy']
                else:
                    compile_kwargs['metrics'] = ['mae']
        
        # Compile the model
        self.model.compile(**compile_kwargs)
        
        # Train the model
        start_time = time.time()
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            **kwargs
        )
        
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Generate predictions using the TensorFlow model.
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Model predictions as NumPy array
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model is not initialized or not trained")
        
        start_time = time.time()
        predictions = self.model.predict(X, batch_size=batch_size, **kwargs)
        self.prediction_time = time.time() - start_time
        
        return predictions


# Factory function to create a GPU model based on the specified framework
def create_gpu_model(framework: str,
                    model_builder: Callable,
                    gpu_manager: Optional[GPUManager] = None,
                    **model_kwargs) -> GPUModelBase:
    """
    Create a GPU-accelerated model using the specified framework.
    
    Args:
        framework: Framework to use ('pytorch' or 'tensorflow')
        model_builder: Function that builds and returns a model
        gpu_manager: GPUManager instance for device handling
        **model_kwargs: Additional arguments to pass to model_builder
        
    Returns:
        GPU model instance
    """
    if framework.lower() == 'pytorch':
        return PyTorchGPUModel(model_builder, gpu_manager, **model_kwargs)
    elif framework.lower() in ('tensorflow', 'tf', 'keras'):
        return TensorFlowGPUModel(model_builder, gpu_manager, **model_kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}. "
                         f"Supported frameworks: 'pytorch', 'tensorflow'")


# Example model builders for testing
def create_pytorch_mlp(input_dim: int = 10, 
                      hidden_dims: List[int] = [64, 32],
                      output_dim: int = 1):
    """
    Create a simple PyTorch MLP.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        
    Returns:
        PyTorch model
    """
    try:
        import torch.nn as nn
        
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(prev_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, output_dim))
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        return MLP()
    
    except ImportError:
        logger.error("PyTorch is not installed. Cannot create PyTorch model.")
        return None


def create_tensorflow_mlp(input_dim: int = 10,
                         hidden_dims: List[int] = [64, 32],
                         output_dim: int = 1):
    """
    Create a simple TensorFlow MLP.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        
    Returns:
        TensorFlow/Keras model
    """
    try:
        import tensorflow as tf
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        for hidden_dim in hidden_dims:
            model.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))
        
        model.add(tf.keras.layers.Dense(output_dim))
        
        return model
    
    except ImportError:
        logger.error("TensorFlow is not installed. Cannot create TensorFlow model.")
        return None


if __name__ == "__main__":
    # Set up logging for script execution
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check GPU availability
    from gpu_utils import is_gpu_available, get_gpu_info
    
    logger.info(f"GPU Available: {is_gpu_available()}")
    gpu_info = get_gpu_info()
    
    # Create GPU manager
    gpu_manager = get_default_gpu_manager()
    
    # Generate some dummy data
    X = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.rand(1000, 1).astype(np.float32)
    
    # Test PyTorch implementation if available
    if gpu_info['framework_support']['pytorch']:
        logger.info("Testing PyTorch GPU model...")
        
        # Create PyTorch model
        torch_model = create_gpu_model(
            'pytorch',
            create_pytorch_mlp,
            gpu_manager,
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1
        )
        
        # Train the model
        torch_model.fit(X, y, epochs=5, batch_size=32)
        
        # Make predictions
        y_pred = torch_model.predict(X)
        
        logger.info(f"PyTorch model performance: {torch_model.get_performance_metrics()}")
    
    # Test TensorFlow implementation if available
    if gpu_info['framework_support']['tensorflow']:
        logger.info("Testing TensorFlow GPU model...")
        
        # Create TensorFlow model
        tf_model = create_gpu_model(
            'tensorflow',
            create_tensorflow_mlp,
            gpu_manager,
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=1
        )
        
        # Train the model
        tf_model.fit(X, y, epochs=5, batch_size=32)
        
        # Make predictions
        y_pred = tf_model.predict(X)
        
        logger.info(f"TensorFlow model performance: {tf_model.get_performance_metrics()}")