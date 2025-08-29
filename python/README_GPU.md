# Quantra GPU Acceleration Module

This module provides GPU acceleration for machine learning and data processing tasks in the Quantra trading platform. It enables significant performance improvements for training models, executing predictions, and processing large datasets.

## Components

The GPU acceleration module consists of several components:

1. **GPU Utils** (`gpu_utils.py`): Core utilities for GPU detection, selection, and management across frameworks.
2. **GPU Models** (`gpu_models.py`): Model wrappers for GPU-accelerated training and inference.
3. **GPU Data Pipeline** (`gpu_data_pipeline.py`): Optimized data processing pipelines for GPU acceleration.
4. **GPU Monitor** (`gpu_monitor.py`): Performance monitoring and metrics collection for GPU operations.
5. **Migration Guide** (`migration_guide.py`): Utilities and examples for migrating CPU code to GPU.

## Requirements

- CUDA Toolkit 11.0+
- One of the following GPU frameworks:
  - PyTorch 1.13+
  - TensorFlow 2.9+
  - CuPy 10.0+
- RAPIDS libraries (optional, for additional acceleration)
  - cuDF 22.10+
  - cuML 22.10+

## Installation

1. Run the setup script to install dependencies and configure your GPU environment:

```bash
./setup_gpu.sh
```

2. Verify that your GPU is detected and working:

```bash
python -c "from gpu_utils import is_gpu_available, get_gpu_info; print(f'GPU available: {is_gpu_available()}'); print(get_gpu_info())"
```

## Usage

### GPU Manager

```python
from gpu_utils import get_default_gpu_manager

# Get GPU manager
gpu_manager = get_default_gpu_manager()

# Check GPU availability
if gpu_manager.gpu_available:
    print("GPU is available for acceleration")
else:
    print("Running in CPU-only mode")
```

### GPU-Accelerated Models

```python
from gpu_models import create_gpu_model, create_pytorch_mlp, create_tensorflow_mlp

# Create PyTorch model with GPU acceleration
pytorch_model = create_gpu_model(
    'pytorch',
    create_pytorch_mlp,
    input_dim=10,
    hidden_dims=[64, 32],
    output_dim=1
)

# Train the model
history = pytorch_model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = pytorch_model.predict(X_test)
```

### GPU Data Pipeline

```python
from gpu_data_pipeline import GPUDataPipeline, calculate_technical_indicators

# Create data pipeline
pipeline = GPUDataPipeline()

# Add feature engineering steps
pipeline.add_feature_engineering_step("technical_indicators", 
                                     lambda df: calculate_technical_indicators(df))

# Apply feature engineering
enhanced_data = pipeline.apply_feature_engineering(stock_data)

# Scale features
X_scaled = pipeline.fit_transform(X, scaler_type='standard')
```

### GPU Performance Monitoring

```python
from gpu_monitor import GPUPerformanceMonitor, GPUMetricCollector

# Create metric collector
collector = GPUMetricCollector()

# Start monitoring
collector.start_collection()

# Run your GPU operations
# ...

# Stop monitoring
collector.stop_collection()

# Get metrics
metrics = collector.get_metrics()
collector.save_metrics("gpu_metrics.json")

# Find optimal batch size
monitor = GPUPerformanceMonitor()
batch_sizes = [32, 64, 128, 256, 512]
results = monitor.find_optimal_batch_size(your_batch_function, batch_sizes)
print(f"Optimal batch size: {results['optimal_batch_size']}")
```

### Migrating CPU Code to GPU

```python
from migration_guide import CodeConverter

# Create code converter
converter = CodeConverter()

# Convert NumPy code to CuPy
gpu_code = converter.suggest_numpy_to_cupy(cpu_code)

# Convert pandas code to cuDF
gpu_code = converter.suggest_pandas_to_cudf(cpu_code)
```

## Integration Testing

Run the integration test script to verify that the GPU acceleration is working properly:

```bash
python test_gpu_integration.py
```

## Configuration

You can customize the GPU settings by editing the `gpu_config.yaml` file:

```yaml
# General GPU settings
gpu:
  enabled: true
  device_id: 0
  memory_growth: true
  memory_limit: 0.8
  
# Framework-specific settings
frameworks:
  pytorch:
    enabled: true
    cuda_visible_devices: "0"
  
  tensorflow:
    enabled: true
    xla_compilation: true
```

## Performance Expectations

When properly configured, you can expect the following performance improvements:

- **Training**: 10-100x speedup for deep learning models
- **Inference**: 5-50x speedup for batch predictions
- **Data Processing**: 3-20x speedup for numerical operations
- **Memory Efficiency**: 2-5x larger model/dataset capacity

The actual speedup will depend on your specific hardware, the size of your data, and the complexity of your models.

## Troubleshooting

- **GPU Not Detected**: Ensure CUDA is properly installed and that your GPU drivers are up-to-date
- **Out of Memory Errors**: Reduce batch size, use mixed precision, or implement gradient checkpointing
- **Slow Performance**: Check for data transfer bottlenecks between CPU and GPU
- **Framework Errors**: Ensure you have compatible versions of CUDA and the ML frameworks