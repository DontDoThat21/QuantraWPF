# Real-Time ML Inference Pipeline

This document explains how to use the real-time ML inference pipeline for low-latency stock predictions in the Quantra trading application.

## Overview

The real-time inference pipeline provides:

- **Low-latency predictions**: Sub-100ms inference times for live trading
- **Streaming data processing**: Handles continuous market data feeds
- **Model caching**: Fast model access with intelligent caching
- **Performance monitoring**: Real-time metrics and health checks
- **Error handling**: Graceful degradation and fallback mechanisms
- **Scalable architecture**: Multi-threaded processing with configurable concurrency

## Architecture

```
Market Data → Feature Engineering → Model Inference → Prediction Results
     ↓              ↓                    ↓                ↓
Live Feeds    Technical Indicators   Cached Models    Trading Signals
```

## Components

### Python Components

1. **`real_time_inference.py`** - Main inference pipeline
2. **`interactive_inference_service.py`** - Interactive service for C# integration
3. **`test_real_time_inference.py`** - Unit tests
4. **`test_real_time_integration.py`** - Integration tests

### C# Components

1. **`RealTimeInferenceService.cs`** - C# service for real-time predictions
2. **Enhanced `PredictionModel.cs`** - Extended model with real-time properties

## Usage

### Python Usage

```python
from real_time_inference import create_inference_pipeline

# Create and configure pipeline
config = {
    'model_types': ['ensemble', 'random_forest'],
    'max_queue_size': 1000,
    'prediction_timeout': 0.1,  # 100ms timeout
    'enable_monitoring': True
}

pipeline = create_inference_pipeline(config)
pipeline.start()

# Get real-time prediction
market_data = {
    'close': 150.50,
    'open': 149.75,
    'high': 151.25,
    'low': 149.25,
    'volume': 2500000,
    'rsi': 65.0,
    'macd': 1.2
}

# Synchronous prediction (blocks until complete)
result = pipeline.predict_sync(market_data, model_type='auto', timeout=1.0)

print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Inference Time: {result['inference_time_ms']:.2f}ms")

# Asynchronous prediction (non-blocking)
def on_prediction(prediction):
    print(f"Async result: {prediction['action']}")

request_id = pipeline.predict_async(market_data, callback=on_prediction)

# Get performance metrics
metrics = pipeline.get_performance_metrics()
print(f"Average inference time: {metrics['inference']['avg_inference_time_ms']:.2f}ms")
print(f"Requests per minute: {metrics['inference']['predictions_per_minute']}")

pipeline.stop()
```

### C# Usage

```csharp
using Quantra.DAL.Services.Interfaces;

// Create inference service
var inferenceService = new RealTimeInferenceService(maxConcurrentRequests: 10);

// Initialize the service
bool initialized = await inferenceService.InitializeAsync();
if (!initialized)
{
    throw new Exception("Failed to initialize inference service");
}

// Prepare market data
var marketData = new Dictionary<string, double>
{
    ["close"] = 150.50,
    ["open"] = 149.75,
    ["high"] = 151.25,
    ["low"] = 149.25,
    ["volume"] = 2500000,
    ["rsi"] = 65.0,
    ["macd"] = 1.2
};

// Get prediction
var prediction = await inferenceService.GetPredictionAsync(
    marketData, 
    modelType: "auto"
);

Console.WriteLine($"Action: {prediction.Action}");
Console.WriteLine($"Confidence: {prediction.Confidence:F2}");
Console.WriteLine($"Inference Time: {prediction.FormattedInferenceTime}");

// Get performance metrics
var metrics = inferenceService.GetPerformanceMetrics();
Console.WriteLine($"Success Rate: {(1 - metrics.ErrorRate):P2}");
Console.WriteLine($"Average Latency: {metrics.AverageInferenceTimeMs:F2}ms");

// Health check
bool isHealthy = await inferenceService.HealthCheckAsync();

// Cleanup
inferenceService.Dispose();
```

## Configuration

### Pipeline Configuration

```python
config = {
    # Model types to support
    'model_types': ['auto', 'ensemble', 'random_forest'],
    
    # Maximum number of queued requests
    'max_queue_size': 1000,
    
    # Maximum time to wait for prediction (seconds)
    'prediction_timeout': 0.1,
    
    # Enable performance monitoring
    'enable_monitoring': True
}
```

### Service Configuration

The C# service can be configured through constructor parameters:

- `maxConcurrentRequests`: Maximum number of concurrent prediction requests
- Custom timeout values through method parameters
- Model type selection per request

## Performance Characteristics

### Latency Targets

- **P50 Latency**: < 50ms for typical market data
- **P95 Latency**: < 100ms for typical market data
- **P99 Latency**: < 200ms for typical market data

### Throughput

- **Synchronous**: 100-500 predictions/second depending on model complexity
- **Asynchronous**: Up to 1000+ predictions/second with proper queuing

### Resource Usage

- **Memory**: ~100-500MB depending on cached models
- **CPU**: 1-4 cores recommended for optimal performance
- **Network**: Minimal (local Python process communication)

## Monitoring and Metrics

### Available Metrics

- **Total Requests**: Cumulative number of prediction requests
- **Success Rate**: Percentage of successful predictions
- **Error Rate**: Percentage of failed predictions
- **Average Inference Time**: Mean time for predictions
- **P95 Inference Time**: 95th percentile inference time
- **Requests Per Minute**: Current request rate
- **Queue Size**: Number of pending requests
- **Cache Hit Rate**: Model cache effectiveness

### Health Checks

The service provides health check endpoints that verify:

- Pipeline initialization status
- Model availability
- Python process health
- Recent prediction success rates

## Error Handling

### Graceful Degradation

When errors occur, the system provides:

1. **Fallback Predictions**: Safe "HOLD" signals with 50% confidence
2. **Error Information**: Detailed error messages for debugging
3. **Service Continuity**: Pipeline continues running despite individual failures
4. **Automatic Recovery**: Automatic restart of failed components

### Common Error Scenarios

1. **Invalid Market Data**: Automatic data sanitization and fallback values
2. **Model Loading Failures**: Fallback to mock models or default predictions
3. **Timeout Errors**: Configurable timeouts with clear error reporting
4. **Resource Exhaustion**: Queue management and request throttling

## Best Practices

### For Trading Applications

1. **Set Appropriate Timeouts**: Use 100-200ms timeouts for live trading
2. **Monitor Performance**: Track latency and error rates continuously
3. **Use Async for High Frequency**: Prefer async predictions for high-frequency trading
4. **Implement Circuit Breakers**: Stop trading if error rate exceeds thresholds
5. **Cache Results**: Implement application-level caching for repeated predictions

### For Development

1. **Use Mock Mode**: Test with mock models during development
2. **Monitor Resource Usage**: Track memory and CPU usage during load testing
3. **Test Error Scenarios**: Verify graceful handling of invalid data
4. **Benchmark Performance**: Measure latency under expected load

## Integration Examples

### With Trading Strategies

```csharp
public class RealTimeStrategy : StrategyProfile
{
    private readonly RealTimeInferenceService _inference;
    
    public RealTimeStrategy(RealTimeInferenceService inference)
    {
        _inference = inference;
    }
    
    public override async Task<string> GenerateSignalAsync(List<HistoricalPrice> prices)
    {
        var latestPrice = prices.Last();
        var marketData = PrepareMarketData(latestPrice, prices);
        
        var prediction = await _inference.GetPredictionAsync(marketData);
        
        // Only trade on high-confidence predictions
        if (prediction.Confidence > 0.75 && !prediction.HasError)
        {
            return prediction.Action;
        }
        
        return "HOLD";
    }
}
```

### With Risk Management

```csharp
public class RiskAwareInference
{
    private readonly RealTimeInferenceService _inference;
    
    public async Task<bool> ShouldExecuteTrade(Dictionary<string, double> marketData)
    {
        var prediction = await _inference.GetPredictionAsync(marketData);
        
        // Risk checks
        if (prediction.RiskScore > 0.8) return false;
        if (prediction.ValueAtRisk > MaxAcceptableVaR) return false;
        if (prediction.Confidence < MinConfidenceThreshold) return false;
        
        return true;
    }
}
```

## Troubleshooting

### Common Issues

1. **High Latency**: Check model complexity, increase cache size, optimize features
2. **Memory Usage**: Reduce cache size, limit concurrent requests
3. **Prediction Errors**: Verify market data format, check model availability
4. **Service Startup**: Ensure Python dependencies are installed, check file paths

### Debug Mode

Enable debug logging in Python:

```python
import logging
logging.getLogger('real_time_inference').setLevel(logging.DEBUG)
```

### Performance Profiling

Use the built-in metrics to identify bottlenecks:

```python
metrics = pipeline.get_performance_metrics()
print(f"Queue size: {metrics['queue_size']}")
print(f"Cache stats: {metrics['cache']}")
print(f"Average latency: {metrics['inference']['avg_inference_time_ms']}")
```

## Future Enhancements

- **GPU Acceleration**: CUDA support for faster model inference
- **Model Hot-Swapping**: Update models without stopping the pipeline
- **Distributed Processing**: Scale across multiple machines
- **Real-time Model Training**: Continuous learning from market data
- **Advanced Caching**: Intelligent prediction result caching
- **Monitoring Integration**: Prometheus/Grafana dashboard support