"""
GPU Utilities Module for Quantra Trading Platform

This module provides utilities for GPU detection, selection, and management
across different machine learning frameworks (PyTorch, TensorFlow, CuPy).
It handles device configuration, memory management, and provides fallback
mechanisms when GPU acceleration is not available.
"""

import logging
import os
import sys
import platform
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MEMORY_GROWTH = True
DEFAULT_MEMORY_LIMIT = 0.8  # Use 80% of GPU memory by default


def is_gpu_available() -> bool:
    """
    Check if any GPU is available on the system.
    
    Returns:
        bool: True if at least one GPU is available, False otherwise
    """
    # Try PyTorch first
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("GPU available via PyTorch CUDA")
            return True
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Try TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info("GPU available via TensorFlow")
            return True
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Try CuPy
    try:
        import cupy as cp
        try:
            # Try to create a small array on the GPU
            x = cp.array([1, 2, 3])
            del x
            logger.info("GPU available via CuPy")
            return True
        except cp.cuda.runtime.CUDARuntimeError:
            pass
    except (ImportError, ModuleNotFoundError):
        pass
    
    logger.warning("No GPU detected on the system")
    return False


def get_gpu_info() -> Dict:
    """
    Get information about available GPUs.
    
    Returns:
        Dict: Information about available GPUs
    """
    gpu_info = {
        "available": is_gpu_available(),
        "count": 0,
        "devices": [],
        "framework_support": {
            "pytorch": False,
            "tensorflow": False,
            "cupy": False
        }
    }
    
    # PyTorch GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["framework_support"]["pytorch"] = True
            gpu_info["count"] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "framework": "pytorch"
                }
                gpu_info["devices"].append(device_info)
    except (ImportError, ModuleNotFoundError):
        pass
    
    # TensorFlow GPU info
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info["framework_support"]["tensorflow"] = True
            # Update count if TensorFlow sees more GPUs than PyTorch
            if len(gpus) > gpu_info["count"]:
                gpu_info["count"] = len(gpus)
            
            # Only add devices that weren't already added by PyTorch
            if not gpu_info["devices"]:
                for i, gpu in enumerate(gpus):
                    device_info = {
                        "index": i,
                        "name": gpu.name,
                        "framework": "tensorflow"
                    }
                    gpu_info["devices"].append(device_info)
    except (ImportError, ModuleNotFoundError):
        pass
    
    # CuPy GPU info
    try:
        import cupy as cp
        try:
            gpu_info["framework_support"]["cupy"] = True
            # CuPy info is only added if not already captured by PyTorch or TF
            if not gpu_info["devices"]:
                n_gpus = cp.cuda.runtime.getDeviceCount()
                gpu_info["count"] = n_gpus
                
                for i in range(n_gpus):
                    cp.cuda.runtime.setDevice(i)
                    attributes = cp.cuda.runtime.getDeviceProperties(i)
                    device_info = {
                        "index": i,
                        "name": attributes["name"].decode('utf-8'),
                        "memory_total": attributes["totalGlobalMem"],
                        "framework": "cupy"
                    }
                    gpu_info["devices"].append(device_info)
        except cp.cuda.runtime.CUDARuntimeError:
            pass
    except (ImportError, ModuleNotFoundError):
        pass
    
    return gpu_info


def setup_pytorch_gpu(device_id: int = 0) -> 'torch.device':
    """
    Configure PyTorch to use the specified GPU.
    
    Args:
        device_id (int): ID of the GPU to use
    
    Returns:
        torch.device: PyTorch device object
    """
    try:
        import torch
        
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(device_id)
            logger.info(f"PyTorch using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            return device
        else:
            logger.warning("PyTorch GPU setup failed, falling back to CPU")
            return torch.device("cpu")
    except (ImportError, ModuleNotFoundError):
        logger.warning("PyTorch not installed, GPU setup skipped")
        return None


def setup_tensorflow_gpu(device_id: int = 0, 
                         memory_growth: bool = DEFAULT_MEMORY_GROWTH,
                         memory_limit: float = DEFAULT_MEMORY_LIMIT) -> bool:
    """
    Configure TensorFlow to use the specified GPU with memory settings.
    
    Args:
        device_id (int): ID of the GPU to use
        memory_growth (bool): Whether to enable memory growth
        memory_limit (float): Fraction of GPU memory to allocate (0.0 to 1.0)
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        import tensorflow as tf
        
        # Get list of available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            logger.warning("TensorFlow: No GPUs available")
            return False
        
        if device_id >= len(gpus):
            logger.warning(f"TensorFlow: Requested GPU {device_id} not available, "
                          f"only {len(gpus)} GPUs found")
            return False
        
        # Configure memory growth
        if memory_growth:
            try:
                tf.config.experimental.set_memory_growth(gpus[device_id], True)
                logger.info(f"TensorFlow: Memory growth enabled for GPU {device_id}")
            except RuntimeError as e:
                logger.error(f"TensorFlow: Failed to set memory growth: {e}")
                return False
        
        # Configure memory limit
        if 0.0 < memory_limit <= 1.0:
            try:
                memory_bytes = int(memory_limit * 1024 * 1024 * 1024)  # Convert to bytes
                tf.config.set_logical_device_configuration(
                    gpus[device_id],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_bytes)]
                )
                logger.info(f"TensorFlow: Memory limit set to {memory_limit*100}% for GPU {device_id}")
            except RuntimeError as e:
                logger.error(f"TensorFlow: Failed to set memory limit: {e}")
        
        # Set visible devices to only the selected GPU
        tf.config.set_visible_devices([gpus[device_id]], 'GPU')
        logger.info(f"TensorFlow using GPU {device_id}")
        return True
    
    except (ImportError, ModuleNotFoundError):
        logger.warning("TensorFlow not installed, GPU setup skipped")
        return False


def setup_cupy_gpu(device_id: int = 0) -> bool:
    """
    Configure CuPy to use the specified GPU.
    
    Args:
        device_id (int): ID of the GPU to use
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        import cupy as cp
        
        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
            
            if n_gpus == 0:
                logger.warning("CuPy: No GPUs available")
                return False
            
            if device_id >= n_gpus:
                logger.warning(f"CuPy: Requested GPU {device_id} not available, "
                              f"only {n_gpus} GPUs found")
                return False
            
            cp.cuda.runtime.setDevice(device_id)
            logger.info(f"CuPy using GPU {device_id}")
            return True
        
        except cp.cuda.runtime.CUDARuntimeError as e:
            logger.error(f"CuPy GPU setup error: {e}")
            return False
    
    except (ImportError, ModuleNotFoundError):
        logger.warning("CuPy not installed, GPU setup skipped")
        return False


class GPUManager:
    """
    Manager class for GPU device selection and configuration across frameworks.
    Provides a unified interface for GPU setup and management.
    """
    
    def __init__(self, device_id: int = 0, 
                 memory_growth: bool = DEFAULT_MEMORY_GROWTH,
                 memory_limit: float = DEFAULT_MEMORY_LIMIT):
        """
        Initialize the GPU manager.
        
        Args:
            device_id (int): ID of the GPU to use
            memory_growth (bool): Whether to enable memory growth for TensorFlow
            memory_limit (float): Fraction of GPU memory to allocate (0.0 to 1.0)
        """
        self.device_id = device_id
        self.memory_growth = memory_growth
        self.memory_limit = memory_limit
        self.gpu_available = is_gpu_available()
        self.pytorch_device = None
        self.tf_initialized = False
        self.cupy_initialized = False
        
        # Auto-initialize on creation
        if self.gpu_available:
            self.initialize_all()
    
    def initialize_all(self) -> None:
        """Initialize all available GPU frameworks."""
        self.initialize_pytorch()
        self.initialize_tensorflow()
        self.initialize_cupy()
    
    def initialize_pytorch(self) -> 'torch.device':
        """Initialize PyTorch GPU."""
        self.pytorch_device = setup_pytorch_gpu(self.device_id)
        return self.pytorch_device
    
    def initialize_tensorflow(self) -> bool:
        """Initialize TensorFlow GPU."""
        self.tf_initialized = setup_tensorflow_gpu(
            self.device_id, 
            self.memory_growth, 
            self.memory_limit
        )
        return self.tf_initialized
    
    def initialize_cupy(self) -> bool:
        """Initialize CuPy GPU."""
        self.cupy_initialized = setup_cupy_gpu(self.device_id)
        return self.cupy_initialized
    
    def get_pytorch_device(self) -> 'torch.device':
        """Get the PyTorch device (initializing if necessary)."""
        if self.pytorch_device is None:
            self.initialize_pytorch()
        return self.pytorch_device
    
    def get_framework_device_info(self) -> Dict:
        """Get status information about GPU framework initialization."""
        return {
            "gpu_available": self.gpu_available,
            "device_id": self.device_id,
            "pytorch": self.pytorch_device is not None,
            "tensorflow": self.tf_initialized,
            "cupy": self.cupy_initialized
        }
    
    def __repr__(self) -> str:
        """String representation of the GPU manager."""
        info = self.get_framework_device_info()
        return f"GPUManager(available={info['gpu_available']}, " \
               f"device_id={info['device_id']}, " \
               f"pytorch={info['pytorch']}, " \
               f"tensorflow={info['tensorflow']}, " \
               f"cupy={info['cupy']})"


# Convenience function to create a GPU manager with default settings
def get_default_gpu_manager() -> GPUManager:
    """Create a GPU manager with default settings."""
    return GPUManager()


if __name__ == "__main__":
    # Set up logging for script execution
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Print system information
    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version.split()[0]}")
    
    # Check if GPU is available
    logger.info(f"GPU Available: {is_gpu_available()}")
    
    # Get detailed GPU info
    gpu_info = get_gpu_info()
    logger.info(f"GPU Count: {gpu_info['count']}")
    logger.info(f"Framework Support: {gpu_info['framework_support']}")
    
    # Print details of each GPU
    for i, device in enumerate(gpu_info['devices']):
        logger.info(f"GPU {i}: {device['name']} (via {device['framework']})")
    
    # Create GPU manager
    manager = get_default_gpu_manager()
    logger.info(f"GPU Manager: {manager}")