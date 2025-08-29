# Quantra Cross-Cutting Concerns Framework

This directory contains the cross-cutting concerns framework for the Quantra application, providing centralized and consistent management of operational aspects including logging, error handling, security, and performance monitoring.

## Core Components

### 1. Logging System

The logging system provides structured, hierarchical logging with multiple output targets:

- **Structured Logging**: All logs include contextual information for better filtering and analysis
- **Multiple Sinks**: Console, file-based, and database logging with rotation and retention policies
- **Performance Logging**: Automatic tracking of method execution times and resource utilization
- **Async Logging**: Non-blocking logging pipeline for high-performance operations
- **Contextual Enrichment**: Automatic source context, trading session IDs, and correlation IDs

```csharp
// Basic usage examples
Log.ForType<MyClass>().Information("Processing order {OrderId}", orderId);

// With additional context
var logger = Log.ForContext("TradingSession", sessionId)
               .ForContext("Strategy", strategyName);
logger.Debug("Strategy calculation completed with result {Result}", result);

// Performance tracking
using (logger.BeginTimedOperation("ImportantOperation"))
{
    // Operation being timed
    DoSomethingImportant();
} // Automatically logs duration on disposal

// Exception logging with context
try 
{
    // Operation that might fail
}
catch (Exception ex)
{
    logger.Error(ex, "Failed to execute operation");
}
```

### 2. Error Handling Framework

The error handling framework provides resilience patterns for robust operation:

- **Retry Policies**: Automatic retry with configurable backoff and jitter
- **Circuit Breakers**: Prevent cascading failures through circuit breaker pattern
- **Error Categorization**: Distinguish between transient, user, and system errors
- **Graceful Degradation**: Maintain core functionality when non-critical services fail

```csharp
// Basic retry
ResilienceHelper.Retry(() => CallExternalService());

// Retry with options
var options = new RetryOptions { 
    MaxRetries = 3, 
    DelayMs = 100, 
    BackoffFactor = 2.0 
};
ResilienceHelper.Retry(() => CallExternalService(), options);

// Circuit breaker
ResilienceHelper.WithCircuitBreaker("ServiceName", () => CallExternalService());

// Combined patterns for external APIs
ResilienceHelper.ExternalApiCall("AlphaVantage", () => FetchMarketData());
```

### 3. Monitoring System

The monitoring system tracks application performance and health:

- **Performance Metrics**: Track execution times, success rates, and resource usage
- **Health Checks**: Component and system-level health monitoring
- **Resource Utilization**: CPU, memory, disk, and network resource tracking
- **Alert Integration**: Automatic alerting for performance degradation

```csharp
// Record custom metrics
Performance.RecordMetric("OrderProcessingTime", processingTimeMs);

// Time operations
Performance.Time("ImportantOperation", () => DoSomethingImportant());

// Check component health
var dbHealth = await Performance.CheckComponentHealthAsync("Database");

// Get system resource utilization
var resources = Performance.GetResourceUtilization();
```

### 4. Security System

The security system manages sensitive data and access controls:

- **Sensitive Data Redaction**: Automatic redaction of credentials and personal information
- **Connection String Security**: Secure storage and handling of connection strings
- **Logging Security**: Prevent sensitive data from appearing in logs

```csharp
// Redact sensitive information
var secureText = Security.RedactSensitiveData(textWithCredentials);

// Secure connection strings for logging
var safeConnString = Security.SecureConnectionString(connectionString);

// Check if property is sensitive
if (Security.IsSensitiveProperty(propertyName))
{
    // Handle specially
}
```

## Integration

To use the cross-cutting concerns framework in your code:

1. Initialize the framework at application startup:

```csharp
CrossCuttingRegistry.Initialize();
```

2. Use the static helpers in your code:
   - `Log` for logging
   - `ResilienceHelper` for error handling and resilience
   - `Performance` for monitoring
   - `Security` for security operations

## Configuration

The framework is configured through the standard `appsettings.json` configuration:

```json
{
  "Logging": {
    "MinimumLevel": "Information",
    "UseConsoleLogging": true,
    "UseFileLogging": true,
    "UseDatabaseLogging": true,
    "LogFilePath": "logs/quantra-.log",
    "UseJsonFormatting": true
  },
  "ErrorHandling": {
    "DefaultMaxRetries": 3,
    "DefaultRetryDelayMs": 100,
    "DefaultBackoffFactor": 2.0,
    "UseJitter": true
  },
  "Monitoring": {
    "EnablePerformanceMetrics": true,
    "MetricRetentionDays": 7,
    "ResourceMonitoringIntervalSeconds": 60
  }
}
```