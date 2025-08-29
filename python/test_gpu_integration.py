"""
GPU Acceleration Integration Test for Quantra

This script demonstrates the use of GPU acceleration utilities and
provides examples of integrating them with existing Quantra functionality.
"""

import logging
import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import GPU utilities
try:
    from gpu_utils import GPUManager, get_default_gpu_manager, is_gpu_available, get_gpu_info
    from gpu_models import PyTorchGPUModel, TensorFlowGPUModel, create_gpu_model
    from gpu_data_pipeline import GPUDataPipeline, calculate_technical_indicators
    from gpu_monitor import GPUPerformanceMonitor, GPUMetricCollector
    GPU_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"GPU modules import error: {e}")
    GPU_MODULES_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper function to create test dataset
def create_stock_data(n_samples=1000):
    """Create synthetic stock market data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n_samples)
    
    # Generate price data
    close = 100 * (1 + np.cumsum(np.random.normal(0, 0.02, n_samples)))
    
    # Add some random noise and seasonal patterns
    t = np.arange(n_samples)
    seasonal = 10 * np.sin(2 * np.pi * t / 252)  # Yearly cycle (252 trading days)
    trend = 0.1 * t  # Small upward trend
    
    close = close + seasonal + trend
    
    # Generate other price columns
    high = close * (1 + np.random.uniform(0, 0.03, n_samples))
    low = close * (1 - np.random.uniform(0, 0.03, n_samples))
    open_price = low + np.random.uniform(0, 1, n_samples) * (high - low)
    
    # Generate volume
    volume = np.random.normal(1000000, 200000, n_samples)
    volume = np.abs(volume)
    
    # Add relationship between volume and price volatility
    volatility = np.abs(np.diff(np.append([0], close)))
    volume += volatility * 100000
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    df.set_index('date', inplace=True)
    return df

# Helper function to visualize model performance
def plot_predictions(y_true, y_pred, title='Model Predictions'):
    """Plot true vs predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:100], label='True Values', color='blue')
    plt.plot(y_pred[:100], label='Predictions', color='red')
    plt.legend()
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.savefig('gpu_model_predictions.png')
    plt.close()

# Function to create PyTorch model for price prediction
def create_pytorch_lstm_model(input_dim=5, hidden_dim=64, output_dim=1, num_layers=2):
    """Create a PyTorch LSTM model for time series prediction."""
    try:
        import torch
        import torch.nn as nn
        
        class LSTMPredictor(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super(LSTMPredictor, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                # LSTM layer
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                
                # Output layer
                self.fc = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                
                # Forward propagate LSTM
                out, _ = self.lstm(x, (h0, c0))
                
                # Decode the hidden state of the last time step
                out = self.fc(out[:, -1, :])
                return out
        
        return LSTMPredictor(input_dim, hidden_dim, output_dim, num_layers)
    
    except ImportError:
        logger.error("PyTorch is not installed. Cannot create PyTorch model.")
        return None

# Function to create TensorFlow model for price prediction
def create_tensorflow_lstm_model(input_dim=5, hidden_dim=64, output_dim=1):
    """Create a TensorFlow LSTM model for time series prediction."""
    try:
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(hidden_dim, input_shape=(None, input_dim), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(hidden_dim//2),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_dim)
        ])
        
        return model
    
    except ImportError:
        logger.error("TensorFlow is not installed. Cannot create TensorFlow model.")
        return None

# Function to prepare data for time series prediction
def prepare_time_series_data(df, sequence_length=10, target_col='close', feature_cols=None):
    """
    Prepare time series data for sequence modeling.
    
    Args:
        df: DataFrame with time series data
        sequence_length: Number of time steps in each sequence
        target_col: Column name to use as prediction target
        feature_cols: List of column names to use as features
    
    Returns:
        X: Input sequences (n_samples, sequence_length, n_features)
        y: Target values (n_samples, 1)
    """
    # Default feature columns if none provided
    if feature_cols is None:
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Extract features and target
    data = df[feature_cols].values
    target = df[target_col].values
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length])
    
    return np.array(X), np.array(y).reshape(-1, 1)

# Function to test GPU availability and performance
def test_gpu_availability():
    """Test GPU availability and performance."""
    logger.info("Testing GPU availability...")
    
    if not GPU_MODULES_AVAILABLE:
        logger.warning("GPU modules not available. Skipping GPU availability test.")
        return
    
    # Check for GPU
    gpu_available = is_gpu_available()
    logger.info(f"GPU available: {gpu_available}")
    
    if gpu_available:
        gpu_info = get_gpu_info()
        logger.info(f"GPU count: {gpu_info['count']}")
        logger.info(f"Framework support: {gpu_info['framework_support']}")
        
        # Print information about each GPU
        for i, device in enumerate(gpu_info['devices']):
            logger.info(f"  GPU {i}: {device['name']} (via {device['framework']})")
    else:
        logger.warning("No GPU detected. Running in CPU-only mode.")

# Function to test PyTorch GPU model
def test_pytorch_gpu_model():
    """Test GPU acceleration with PyTorch."""
    logger.info("Testing PyTorch GPU model...")
    
    if not GPU_MODULES_AVAILABLE:
        logger.warning("GPU modules not available. Skipping PyTorch GPU test.")
        return
    
    # Create sample data
    logger.info("Creating sample data...")
    df = create_stock_data(2000)
    df = calculate_technical_indicators(df)
    
    # Add target column (next day's close price)
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    
    # Create feature and target columns
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma_5', 'ma_20']
    feature_cols = [col for col in feature_cols if col in df.columns]  # Filter to available columns
    
    # Prepare sequences
    X, y = prepare_time_series_data(df, sequence_length=10, 
                                   target_col='next_close', 
                                   feature_cols=feature_cols)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create GPU model
    try:
        # Get GPU manager
        gpu_manager = get_default_gpu_manager()
        
        # Create PyTorch GPU model
        gpu_model = create_gpu_model(
            'pytorch',
            lambda **kwargs: create_pytorch_lstm_model(
                input_dim=len(feature_cols),
                hidden_dim=64,
                output_dim=1,
                num_layers=2
            ),
            gpu_manager=gpu_manager
        )
        
        # Train the model
        logger.info("Training PyTorch GPU model...")
        history = gpu_model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.1,
            learning_rate=0.001
        )
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = gpu_model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        logger.info(f"Mean Squared Error: {mse:.6f}")
        
        # Plot predictions
        plot_predictions(y_test, y_pred, title='PyTorch GPU Model Predictions')
        
        # Log performance metrics
        metrics = gpu_model.get_performance_metrics()
        logger.info(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
        logger.info(f"Prediction time: {metrics['prediction_time_seconds']:.2f} seconds")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in PyTorch GPU test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Function to test TensorFlow GPU model
def test_tensorflow_gpu_model():
    """Test GPU acceleration with TensorFlow."""
    logger.info("Testing TensorFlow GPU model...")
    
    if not GPU_MODULES_AVAILABLE:
        logger.warning("GPU modules not available. Skipping TensorFlow GPU test.")
        return
    
    # Create sample data
    logger.info("Creating sample data...")
    df = create_stock_data(2000)
    df = calculate_technical_indicators(df)
    
    # Add target column (next day's close price)
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    
    # Create feature and target columns
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'ma_5', 'ma_20']
    feature_cols = [col for col in feature_cols if col in df.columns]  # Filter to available columns
    
    # Prepare sequences
    X, y = prepare_time_series_data(df, sequence_length=10, 
                                   target_col='next_close', 
                                   feature_cols=feature_cols)
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create GPU model
    try:
        # Get GPU manager
        gpu_manager = get_default_gpu_manager()
        
        # Create TensorFlow GPU model
        gpu_model = create_gpu_model(
            'tensorflow',
            lambda **kwargs: create_tensorflow_lstm_model(
                input_dim=len(feature_cols),
                hidden_dim=64,
                output_dim=1
            ),
            gpu_manager=gpu_manager
        )
        
        # Compile and train the model
        logger.info("Training TensorFlow GPU model...")
        history = gpu_model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.1,
            compile_kwargs={
                'optimizer': 'adam',
                'loss': 'mse',
                'metrics': ['mae']
            }
        )
        
        # Make predictions
        logger.info("Making predictions...")
        y_pred = gpu_model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        logger.info(f"Mean Squared Error: {mse:.6f}")
        
        # Plot predictions
        plot_predictions(y_test, y_pred, title='TensorFlow GPU Model Predictions')
        
        # Log performance metrics
        metrics = gpu_model.get_performance_metrics()
        logger.info(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
        logger.info(f"Prediction time: {metrics['prediction_time_seconds']:.2f} seconds")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in TensorFlow GPU test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Function to test GPU data pipeline
def test_gpu_data_pipeline():
    """Test GPU data pipeline for technical indicator calculation."""
    logger.info("Testing GPU data pipeline...")
    
    if not GPU_MODULES_AVAILABLE:
        logger.warning("GPU modules not available. Skipping GPU data pipeline test.")
        return
    
    # Create sample data
    logger.info("Creating sample data...")
    df = create_stock_data(5000)
    
    # Initialize pipeline
    pipeline = GPUDataPipeline()
    
    # Check if GPU acceleration is available
    info = pipeline.get_performance_metrics()
    logger.info(f"GPU acceleration available: {info['is_gpu_available']}")
    logger.info(f"CuPy available: {info['cupy_available']}")
    logger.info(f"cuDF available: {info['cudf_available']}")
    
    # Add feature engineering steps
    logger.info("Adding feature engineering steps...")
    pipeline.add_feature_engineering_step("technical_indicators", 
                                         lambda df: calculate_technical_indicators(df))
    
    # Test CPU vs GPU performance
    start_time = time.time()
    df_cpu = calculate_technical_indicators(df)
    cpu_time = time.time() - start_time
    logger.info(f"CPU technical indicators calculation: {cpu_time:.4f} seconds")
    
    # Test GPU version
    start_time = time.time()
    df_gpu = pipeline.apply_feature_engineering(df)
    gpu_time = time.time() - start_time
    logger.info(f"GPU technical indicators calculation: {gpu_time:.4f} seconds")
    
    # Convert back to pandas if needed for comparison
    if hasattr(df_gpu, 'to_pandas'):
        df_gpu = df_gpu.to_pandas()
    
    # Compare results (should be similar but not identical due to floating point differences)
    logger.info(f"CPU result shape: {df_cpu.shape}")
    logger.info(f"GPU result shape: {df_gpu.shape}")
    
    # Check performance gain
    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        logger.info(f"GPU speedup: {speedup:.2f}x faster")
    else:
        slowdown = gpu_time / cpu_time
        logger.info(f"GPU slowdown: {slowdown:.2f}x slower (overhead may exceed benefits for small datasets)")
    
    return True

# Function to test GPU monitoring
def test_gpu_monitoring():
    """Test GPU monitoring capabilities."""
    logger.info("Testing GPU monitoring...")
    
    if not GPU_MODULES_AVAILABLE:
        logger.warning("GPU modules not available. Skipping GPU monitoring test.")
        return
    
    # Check if GPU is available
    if not is_gpu_available():
        logger.warning("No GPU detected. Skipping GPU monitoring.")
        return
    
    try:
        # Create metric collector
        collector = GPUMetricCollector()
        
        # Get current metrics
        metrics = collector.collect_metrics_once()
        
        logger.info("Current GPU metrics:")
        for key, value in metrics.items():
            if key not in ['timestamp', 'device_id', 'error']:
                logger.info(f"  {key}: {value}")
        
        # Start continuous monitoring
        logger.info("Starting continuous monitoring for 5 seconds...")
        collector.start_collection()
        
        # Run a GPU-intensive operation to generate metrics
        def gpu_load_test():
            # Try with PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    # Create large tensors and perform operations
                    a = torch.rand(2000, 2000, device='cuda')
                    b = torch.rand(2000, 2000, device='cuda')
                    for _ in range(10):
                        c = torch.matmul(a, b)
                        del c
                    torch.cuda.empty_cache()
            except ImportError:
                # Try with TensorFlow
                try:
                    import tensorflow as tf
                    if tf.config.list_physical_devices('GPU'):
                        with tf.device('/GPU:0'):
                            a = tf.random.normal([2000, 2000])
                            b = tf.random.normal([2000, 2000])
                            for _ in range(10):
                                c = tf.matmul(a, b)
                                del c
                except ImportError:
                    # Try with CuPy
                    try:
                        import cupy as cp
                        a = cp.random.rand(2000, 2000)
                        b = cp.random.rand(2000, 2000)
                        for _ in range(10):
                            c = cp.dot(a, b)
                            del c
                        cp.get_default_memory_pool().free_all_blocks()
                    except ImportError:
                        logger.warning("No GPU computation libraries available for load testing")
        
        # Run load test
        gpu_load_test()
        
        # Wait for more metrics
        time.sleep(5)
        
        # Stop collection
        collector.stop_collection()
        
        # Get all collected metrics
        all_metrics = collector.get_metrics()
        logger.info(f"Collected {len(all_metrics)} metric samples")
        
        # Save metrics to file
        collector.save_metrics("gpu_monitoring_test.json")
        logger.info("Saved metrics to gpu_monitoring_test.json")
        
        # Create performance monitor
        monitor = GPUPerformanceMonitor()
        
        # Test batch size optimization
        logger.info("Testing batch size optimization...")
        
        def batch_operation(batch_size):
            """Simple function to test different batch sizes."""
            # Try with PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    data_size = 10000
                    dim = 50
                    
                    # Create data
                    data = torch.rand(data_size, dim, device='cuda')
                    
                    # Process in batches
                    for i in range(0, data_size, batch_size):
                        batch = data[i:i+batch_size]
                        result = torch.nn.functional.normalize(batch)
                    
                    return "PyTorch batch processing"
            except ImportError:
                # Try with NumPy/CuPy
                try:
                    import cupy as cp
                    data_size = 10000
                    dim = 50
                    
                    # Create data
                    data = cp.random.rand(data_size, dim)
                    
                    # Process in batches
                    for i in range(0, data_size, batch_size):
                        batch = data[i:i+batch_size]
                        result = batch / cp.linalg.norm(batch, axis=1, keepdims=True)
                    
                    return "CuPy batch processing"
                except ImportError:
                    # Fallback to NumPy
                    data_size = 10000
                    dim = 50
                    
                    # Create data
                    data = np.random.rand(data_size, dim)
                    
                    # Process in batches
                    for i in range(0, data_size, batch_size):
                        batch = data[i:i+batch_size]
                        result = batch / np.linalg.norm(batch, axis=1, keepdims=True)
                    
                    return "NumPy batch processing"
        
        # Find optimal batch size
        batch_sizes = [32, 64, 128, 256, 512, 1024]
        batch_results = monitor.find_optimal_batch_size(
            batch_operation, batch_sizes, name="batch_size_test", iterations=2)
        
        logger.info(f"Optimal batch size: {batch_results['optimal_batch_size']}")
        
        # Save benchmarks
        monitor.save_benchmarks("gpu_benchmarks_test.json")
        logger.info("Saved benchmarks to gpu_benchmarks_test.json")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in GPU monitoring test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Main function
def main():
    """Run GPU acceleration tests."""
    logger.info("Starting GPU acceleration integration test...")
    
    # Check Python version and platform
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Test GPU availability
    test_gpu_availability()
    
    # Test data pipeline
    try:
        test_gpu_data_pipeline()
    except Exception as e:
        logger.error(f"Error in GPU data pipeline test: {e}")
    
    # Test PyTorch GPU model
    try:
        test_pytorch_gpu_model()
    except Exception as e:
        logger.error(f"Error in PyTorch GPU test: {e}")
    
    # Test TensorFlow GPU model
    try:
        test_tensorflow_gpu_model()
    except Exception as e:
        logger.error(f"Error in TensorFlow GPU test: {e}")
    
    # Test GPU monitoring
    try:
        test_gpu_monitoring()
    except Exception as e:
        logger.error(f"Error in GPU monitoring test: {e}")
    
    logger.info("GPU acceleration integration test completed.")


if __name__ == "__main__":
    main()