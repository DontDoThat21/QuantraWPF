"""
GPU Performance Monitoring System for Quantra Trading Platform

This module provides utilities for monitoring GPU performance metrics including:
- GPU utilization
- Memory usage
- Temperature
- Training/inference throughput
- Benchmarking tools
"""

import logging
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd

# Import local modules
from gpu_utils import GPUManager, get_default_gpu_manager, get_gpu_info

# Set up logging
logger = logging.getLogger(__name__)


class GPUMetricCollector:
    """
    Collects real-time metrics from the GPU.
    
    Monitors:
    - GPU utilization
    - Memory usage
    - Temperature (if available)
    - Power consumption (if available)
    """
    
    def __init__(self, device_id: int = 0, sample_interval: float = 1.0):
        """
        Initialize the GPU metric collector.
        
        Args:
            device_id: ID of the GPU to monitor
            sample_interval: Time in seconds between metric samples
        """
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.metrics = []
        self.is_collecting = False
        self.collection_thread = None
        self.gpu_info = get_gpu_info()
        
        # Check if NVIDIA Management Library (pynvml) is available
        self.nvml_available = False
        try:
            import pynvml
            self.pynvml = pynvml
            self.nvml_available = True
            logger.info("NVIDIA Management Library (pynvml) is available for GPU monitoring")
        except ImportError:
            logger.warning("pynvml not installed. Limited GPU metrics will be available.")
            self.pynvml = None
    
    def _collect_nvidia_metrics(self) -> Dict:
        """Collect metrics using NVIDIA Management Library."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id
        }
        
        try:
            # Initialize NVML
            self.pynvml.nvmlInit()
            
            # Get device handle
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            
            # Get device name
            metrics['name'] = self.pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get utilization
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics['gpu_utilization'] = utilization.gpu
            metrics['memory_utilization'] = utilization.memory
            
            # Get memory info
            memory_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics['memory_total'] = memory_info.total / (1024 * 1024)  # Convert to MB
            metrics['memory_used'] = memory_info.used / (1024 * 1024)
            metrics['memory_free'] = memory_info.free / (1024 * 1024)
            metrics['memory_percentage'] = (memory_info.used / memory_info.total) * 100
            
            # Get temperature
            metrics['temperature'] = self.pynvml.nvmlDeviceGetTemperature(
                handle, self.pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power usage
            try:
                power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                metrics['power_usage'] = power_usage
            except self.pynvml.NVMLError:
                # Power usage not supported on this GPU
                metrics['power_usage'] = None
            
            # Get clock speeds
            try:
                metrics['gpu_clock'] = self.pynvml.nvmlDeviceGetClockInfo(
                    handle, self.pynvml.NVML_CLOCK_GRAPHICS)
                metrics['mem_clock'] = self.pynvml.nvmlDeviceGetClockInfo(
                    handle, self.pynvml.NVML_CLOCK_MEM)
            except self.pynvml.NVMLError:
                metrics['gpu_clock'] = None
                metrics['mem_clock'] = None
            
            # Clean up
            self.pynvml.nvmlShutdown()
            
        except Exception as e:
            logger.error(f"Error collecting NVIDIA metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _collect_pytorch_metrics(self) -> Dict:
        """Collect metrics using PyTorch."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                # Get device name
                metrics['name'] = torch.cuda.get_device_name(self.device_id)
                
                # Get memory usage
                allocated = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)
                
                metrics['memory_allocated_mb'] = allocated
                metrics['memory_reserved_mb'] = reserved
                metrics['memory_total'] = torch.cuda.get_device_properties(self.device_id).total_memory / (1024 * 1024)
                metrics['memory_percentage'] = (allocated / metrics['memory_total']) * 100
            else:
                metrics['error'] = "CUDA not available"
        
        except ImportError:
            metrics['error'] = "PyTorch not installed"
            logger.warning("PyTorch not installed. Cannot collect PyTorch GPU metrics.")
        except Exception as e:
            metrics['error'] = str(e)
            logger.error(f"Error collecting PyTorch metrics: {e}")
        
        return metrics
    
    def collect_metrics_once(self) -> Dict:
        """
        Collect GPU metrics once.
        
        Returns:
            Dictionary of current GPU metrics
        """
        if self.nvml_available:
            return self._collect_nvidia_metrics()
        else:
            return self._collect_pytorch_metrics()
    
    def _metrics_collection_loop(self):
        """Background thread for continuous metrics collection."""
        while self.is_collecting:
            metrics = self.collect_metrics_once()
            self.metrics.append(metrics)
            time.sleep(self.sample_interval)
    
    def start_collection(self):
        """Start collecting metrics in a background thread."""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self._metrics_collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info(f"Started GPU metric collection for device {self.device_id}")
    
    def stop_collection(self):
        """Stop collecting metrics."""
        self.is_collecting = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
            logger.info("Stopped GPU metric collection")
    
    def get_metrics(self) -> List[Dict]:
        """
        Get collected metrics.
        
        Returns:
            List of metric dictionaries
        """
        return self.metrics
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get metrics as a pandas DataFrame.
        
        Returns:
            DataFrame of metrics
        """
        return pd.DataFrame(self.metrics)
    
    def get_latest_metrics(self) -> Dict:
        """
        Get the most recent metrics.
        
        Returns:
            Dictionary of latest metrics or empty dict if none
        """
        if self.metrics:
            return self.metrics[-1]
        return {}
    
    def clear_metrics(self):
        """Clear the collected metrics."""
        self.metrics = []
    
    def save_metrics(self, filename: str):
        """
        Save metrics to a JSON file.
        
        Args:
            filename: Name of file to save metrics to
        """
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved {len(self.metrics)} metric records to {filename}")
    
    def save_metrics_csv(self, filename: str):
        """
        Save metrics to a CSV file.
        
        Args:
            filename: Name of file to save metrics to
        """
        if not self.metrics:
            logger.warning("No metrics to save")
            return
        
        df = self.get_metrics_dataframe()
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} metric records to {filename}")


class GPUPerformanceMonitor:
    """
    Monitor and analyze GPU performance for ML operations.
    
    Provides utilities for:
    - Timing operations
    - Benchmarking model performance
    - Comparing CPU vs GPU performance
    - Finding optimal batch sizes
    """
    
    def __init__(self, gpu_manager: Optional[GPUManager] = None):
        """
        Initialize the GPU performance monitor.
        
        Args:
            gpu_manager: GPUManager instance for device handling
        """
        self.gpu_manager = gpu_manager or get_default_gpu_manager()
        self.benchmarks = {}
        self.metric_collector = None
        
        # Try to initialize the metric collector
        try:
            self.metric_collector = GPUMetricCollector()
        except Exception as e:
            logger.warning(f"Failed to initialize GPU metric collector: {e}")
    
    def time_function(self, func: Callable, *args, 
                    collect_gpu_metrics: bool = False,
                    name: str = None,
                    **kwargs) -> Tuple[Any, float, Optional[Dict]]:
        """
        Time the execution of a function and collect GPU metrics.
        
        Args:
            func: Function to time
            *args: Arguments to pass to the function
            collect_gpu_metrics: Whether to collect GPU metrics during execution
            name: Name for this timing operation
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (function result, execution time in seconds, metrics if collected)
        """
        if collect_gpu_metrics and self.metric_collector:
            self.metric_collector.clear_metrics()
            self.metric_collector.start_collection()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        metrics = None
        if collect_gpu_metrics and self.metric_collector:
            self.metric_collector.stop_collection()
            metrics = self.metric_collector.get_metrics()
        
        # Save benchmark if name is provided
        if name:
            self.benchmarks[name] = {
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
        
        return result, execution_time, metrics
    
    def compare_cpu_gpu_performance(self, 
                                  cpu_func: Callable, 
                                  gpu_func: Callable,
                                  *args,
                                  name: str = None,
                                  iterations: int = 3,
                                  **kwargs) -> Dict:
        """
        Compare performance between CPU and GPU implementations.
        
        Args:
            cpu_func: CPU implementation of function
            gpu_func: GPU implementation of function
            *args: Arguments to pass to both functions
            name: Name for this benchmark
            iterations: Number of iterations to run for each implementation
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            Dictionary with performance comparison results
        """
        cpu_times = []
        gpu_times = []
        
        logger.info(f"Running performance comparison: CPU vs GPU ({iterations} iterations each)")
        
        # Run CPU implementation multiple times
        logger.info("Running CPU implementation...")
        for i in range(iterations):
            _, execution_time, _ = self.time_function(cpu_func, *args, **kwargs)
            cpu_times.append(execution_time)
            logger.info(f"  CPU iteration {i+1}/{iterations}: {execution_time:.4f} seconds")
        
        # Run GPU implementation multiple times
        logger.info("Running GPU implementation...")
        for i in range(iterations):
            _, execution_time, _ = self.time_function(gpu_func, *args, **kwargs)
            gpu_times.append(execution_time)
            logger.info(f"  GPU iteration {i+1}/{iterations}: {execution_time:.4f} seconds")
        
        # Calculate statistics
        cpu_avg = np.mean(cpu_times)
        gpu_avg = np.mean(gpu_times)
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float('inf')
        
        results = {
            'cpu_times': cpu_times,
            'gpu_times': gpu_times,
            'cpu_avg_seconds': cpu_avg,
            'gpu_avg_seconds': gpu_avg,
            'cpu_std_seconds': np.std(cpu_times),
            'gpu_std_seconds': np.std(gpu_times),
            'speedup': speedup,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance comparison results:")
        logger.info(f"  CPU average: {cpu_avg:.4f} seconds")
        logger.info(f"  GPU average: {gpu_avg:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Save benchmark if name is provided
        if name:
            self.benchmarks[name] = results
        
        return results
    
    def find_optimal_batch_size(self,
                              func: Callable,
                              batch_sizes: List[int],
                              *args,
                              name: str = None,
                              iterations: int = 3,
                              **kwargs) -> Dict:
        """
        Find the optimal batch size for GPU operations.
        
        Args:
            func: Function that takes a batch_size parameter
            batch_sizes: List of batch sizes to test
            *args: Additional arguments to pass to the function
            name: Name for this benchmark
            iterations: Number of iterations for each batch size
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            Dictionary with results and optimal batch size
        """
        results = {}
        times = []
        
        logger.info(f"Finding optimal batch size (testing {len(batch_sizes)} sizes, {iterations} iterations each)")
        
        for batch_size in batch_sizes:
            batch_times = []
            
            logger.info(f"Testing batch size {batch_size}...")
            for i in range(iterations):
                func_args = args + (batch_size,) if not any(isinstance(a, dict) and 'batch_size' in a for a in args) else args
                try:
                    _, execution_time, _ = self.time_function(func, *func_args, **kwargs)
                    batch_times.append(execution_time)
                    logger.info(f"  Iteration {i+1}/{iterations}: {execution_time:.4f} seconds")
                except Exception as e:
                    logger.error(f"  Error with batch size {batch_size}: {e}")
                    batch_times.append(float('inf'))
            
            avg_time = np.mean(batch_times) if not all(t == float('inf') for t in batch_times) else float('inf')
            results[batch_size] = {
                'times': batch_times,
                'avg_time': avg_time,
                'std_time': np.std(batch_times) if not all(t == float('inf') for t in batch_times) else 0
            }
            times.append(avg_time)
            
            logger.info(f"  Average time: {avg_time:.4f} seconds")
        
        # Find the batch size with the minimum average time
        valid_times = [(bs, results[bs]['avg_time']) for bs in batch_sizes
                       if results[bs]['avg_time'] != float('inf')]
        
        if valid_times:
            optimal_batch_size = min(valid_times, key=lambda x: x[1])[0]
        else:
            optimal_batch_size = batch_sizes[0]
            logger.warning("No valid batch size found, using the first one")
        
        summary = {
            'batch_sizes': batch_sizes,
            'average_times': times,
            'optimal_batch_size': optimal_batch_size,
            'optimal_time': results[optimal_batch_size]['avg_time'],
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Optimal batch size: {optimal_batch_size} "
                  f"({summary['optimal_time']:.4f} seconds)")
        
        # Save benchmark if name is provided
        if name:
            self.benchmarks[name] = summary
        
        return summary
    
    def analyze_memory_efficiency(self, 
                                model_func: Callable, 
                                input_sizes: List[Tuple[int, ...]],
                                name: str = None) -> Dict:
        """
        Analyze memory efficiency for different input sizes.
        
        Args:
            model_func: Function that creates and runs a model
            input_sizes: List of input shapes to test
            name: Name for this benchmark
            
        Returns:
            Dictionary with memory usage results
        """
        if not self.metric_collector or not self.metric_collector.nvml_available:
            logger.warning("NVIDIA Management Library not available, memory analysis will be limited")
        
        results = {}
        
        logger.info(f"Analyzing memory efficiency for {len(input_sizes)} different input sizes")
        
        for i, size in enumerate(input_sizes):
            logger.info(f"Testing input size {size}...")
            
            # Clear metrics
            if self.metric_collector:
                self.metric_collector.clear_metrics()
                self.metric_collector.start_collection()
            
            try:
                # Execute the model function
                start_time = time.time()
                model_func(*size)
                execution_time = time.time() - start_time
                
                # Collect metrics
                if self.metric_collector:
                    self.metric_collector.stop_collection()
                    metrics = self.metric_collector.get_metrics()
                    
                    # Extract relevant metrics
                    max_memory = max([m.get('memory_used', 0) for m in metrics]) if metrics else 0
                    max_utilization = max([m.get('gpu_utilization', 0) for m in metrics]) if metrics else 0
                else:
                    metrics = []
                    max_memory = 0
                    max_utilization = 0
                
                results[str(size)] = {
                    'input_size': size,
                    'execution_time': execution_time,
                    'max_memory_mb': max_memory,
                    'max_utilization': max_utilization,
                    'metrics': metrics
                }
                
                logger.info(f"  Time: {execution_time:.4f}s, Max Memory: {max_memory:.1f}MB, "
                          f"Max Utilization: {max_utilization}%")
                
            except Exception as e:
                logger.error(f"  Error with input size {size}: {e}")
                results[str(size)] = {
                    'input_size': size,
                    'error': str(e)
                }
        
        summary = {
            'input_sizes': input_sizes,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save benchmark if name is provided
        if name:
            self.benchmarks[name] = summary
        
        return summary
    
    def save_benchmarks(self, filename: str):
        """
        Save all benchmarks to a JSON file.
        
        Args:
            filename: Name of file to save benchmarks to
        """
        # Convert any non-serializable objects
        def json_serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return str(obj)
        
        # Use custom serializer
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.integer, np.floating)):
                    return json_serialize(obj)
                return super().default(obj)
        
        with open(filename, 'w') as f:
            json.dump(self.benchmarks, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved {len(self.benchmarks)} benchmarks to {filename}")


def compare_frameworks(data: np.ndarray, task_type: str = 'inference'):
    """
    Compare performance between different frameworks (PyTorch, TensorFlow).
    
    Args:
        data: Input data for the comparison
        task_type: Type of task ('inference' or 'training')
        
    Returns:
        Dictionary with performance comparison results
    """
    try:
        import torch
        import tensorflow as tf
        
        # GPU info
        gpu_info = get_gpu_info()
        monitor = GPUPerformanceMonitor()
        
        # Create simple models
        input_shape = data.shape[1] if len(data.shape) > 1 else 1
        output_shape = 1  # For simplicity
        
        # PyTorch implementation
        def pytorch_func():
            class SimpleModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(input_shape, 64)
                    self.fc2 = torch.nn.Linear(64, 32)
                    self.fc3 = torch.nn.Linear(32, output_shape)
                    
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    return self.fc3(x)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleModel().to(device)
            
            # Convert data to PyTorch tensor
            tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
            
            if task_type == 'inference':
                # Just do inference
                with torch.no_grad():
                    result = model(tensor_data)
            else:
                # Simple training loop
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_fn = torch.nn.MSELoss()
                
                for i in range(10):
                    optimizer.zero_grad()
                    output = model(tensor_data)
                    loss = loss_fn(output, torch.zeros_like(output))
                    loss.backward()
                    optimizer.step()
            
            return "PyTorch executed successfully"
        
        # TensorFlow implementation
        def tensorflow_func():
            # Ensure GPU is available and configured
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(output_shape)
            ])
            
            # Convert data to TensorFlow tensor
            tensor_data = tf.convert_to_tensor(data, dtype=tf.float32)
            
            if task_type == 'inference':
                # Just do inference
                result = model(tensor_data)
            else:
                # Simple training loop
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                loss_fn = tf.keras.losses.MeanSquaredError()
                
                for i in range(10):
                    with tf.GradientTape() as tape:
                        output = model(tensor_data)
                        loss = loss_fn(tf.zeros_like(output), output)
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return "TensorFlow executed successfully"
        
        # Run benchmark
        results = monitor.compare_cpu_gpu_performance(
            lambda: time.sleep(0.1),  # Placeholder for CPU function
            pytorch_func,
            name=f"pytorch_{task_type}",
            iterations=3
        )
        
        tf_results = monitor.compare_cpu_gpu_performance(
            lambda: time.sleep(0.1),  # Placeholder for CPU function
            tensorflow_func,
            name=f"tensorflow_{task_type}",
            iterations=3
        )
        
        # Compare frameworks
        framework_comparison = {
            'pytorch': {
                'avg_time': results['gpu_avg_seconds'],
                'std_time': results['gpu_std_seconds']
            },
            'tensorflow': {
                'avg_time': tf_results['gpu_avg_seconds'],
                'std_time': tf_results['gpu_std_seconds']
            },
            'faster_framework': 'pytorch' if results['gpu_avg_seconds'] < tf_results['gpu_avg_seconds'] else 'tensorflow',
            'speedup_ratio': tf_results['gpu_avg_seconds'] / results['gpu_avg_seconds'] 
                            if results['gpu_avg_seconds'] < tf_results['gpu_avg_seconds'] else
                            results['gpu_avg_seconds'] / tf_results['gpu_avg_seconds']
        }
        
        return framework_comparison
        
    except ImportError as e:
        logger.error(f"Framework comparison failed: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    # Set up logging for script execution
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Check GPU availability
    from gpu_utils import is_gpu_available, get_gpu_info
    
    logger.info(f"GPU Available: {is_gpu_available()}")
    gpu_info = get_gpu_info()
    
    # Create GPU metric collector if available
    try:
        collector = GPUMetricCollector()
        logger.info("Collecting GPU metrics...")
        metrics = collector.collect_metrics_once()
        
        # Print some key metrics
        logger.info(f"GPU: {metrics.get('name', 'Unknown')}")
        logger.info(f"Memory Used: {metrics.get('memory_used', 'N/A'):.1f} MB")
        logger.info(f"Memory Total: {metrics.get('memory_total', 'N/A'):.1f} MB")
        logger.info(f"Utilization: {metrics.get('gpu_utilization', 'N/A')}%")
        logger.info(f"Temperature: {metrics.get('temperature', 'N/A')}°C")
        
        # Start continuous monitoring
        logger.info("Starting continuous GPU monitoring (5 seconds)...")
        collector.start_collection()
        time.sleep(5)  # Monitor for 5 seconds
        collector.stop_collection()
        
        # Get collected metrics
        all_metrics = collector.get_metrics()
        logger.info(f"Collected {len(all_metrics)} metric samples")
        
        # Save metrics to file if any were collected
        if all_metrics:
            collector.save_metrics('gpu_metrics.json')
    
    except Exception as e:
        logger.error(f"Error in GPU metric collection: {e}")
    
    # Create performance monitor
    monitor = GPUPerformanceMonitor()
    
    # Simple function to time
    def simple_array_multiplication(size):
        if is_gpu_available():
            try:
                import cupy as cp
                # Create large arrays on GPU
                a = cp.random.rand(size, size).astype(cp.float32)
                b = cp.random.rand(size, size).astype(cp.float32)
                # Perform matrix multiplication
                result = cp.dot(a, b)
                # Ensure the operation is complete
                cp.cuda.Stream.null.synchronize()
                return result
            except ImportError:
                logger.warning("CuPy not available, falling back to NumPy")
        
        # CPU fallback
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        return np.dot(a, b)
    
    # Time a function
    logger.info("Timing matrix multiplication function...")
    result, execution_time, _ = monitor.time_function(
        simple_array_multiplication, 1000, collect_gpu_metrics=True, name="matrix_mult_1000")
    logger.info(f"Matrix multiplication took {execution_time:.4f} seconds")
    
    # Find optimal batch size (if GPU available)
    if is_gpu_available():
        logger.info("Finding optimal batch size...")
        batch_sizes = [32, 64, 128, 256, 512, 1024]
        
        def batch_process(batch_size):
            # Simple function that processes data in batches
            data_size = 10000
            
            try:
                import torch
                device = torch.device('cuda')
                
                # Create a large tensor
                data = torch.rand(data_size, 100, device=device)
                
                # Process in batches
                for i in range(0, data_size, batch_size):
                    batch = data[i:i+batch_size]
                    result = torch.nn.functional.normalize(batch)
                
                return "Processing completed"
            
            except ImportError:
                logger.warning("PyTorch not available")
                # CPU fallback
                data = np.random.rand(data_size, 100).astype(np.float32)
                for i in range(0, data_size, batch_size):
                    batch = data[i:i+batch_size]
                    # Simple processing
                    result = batch / np.linalg.norm(batch, axis=1, keepdims=True)
                
                return "Processing completed (CPU)"
        
        # Find optimal batch size
        optimal = monitor.find_optimal_batch_size(
            batch_process, batch_sizes, name="batch_optimization")
        
        # Save all benchmarks
        monitor.save_benchmarks('gpu_benchmarks.json')
        logger.info("Benchmarks saved to gpu_benchmarks.json")