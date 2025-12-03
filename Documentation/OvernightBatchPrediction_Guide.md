# Overnight Batch Prediction System

## Overview

The Overnight Batch Prediction System is designed to efficiently generate ML predictions for large datasets (12,000+ symbols) during off-hours. This system processes historical data from your database and generates predictions in batches, with progress tracking, error recovery, and resource optimization.

## Features

### 1. **Batch Processing**
- Processes symbols in configurable batches (default: 100 symbols per batch)
- Concurrent processing with throttling (default: 10 concurrent predictions)
- Automatic retry logic with exponential backoff

### 2. **Resource Management**
- Throttled concurrent execution prevents system overload
- Configurable delays between batches
- Memory-efficient batch processing

### 3. **Progress Tracking**
- Real-time progress updates (processed, successful, failed)
- Estimated time remaining calculation
- Current symbol being processed
- Success rate metrics

### 4. **Smart Filtering**
- Skip symbols with recent predictions (configurable threshold)
- Filter by specific symbols or sectors
- Limit maximum number of symbols to process

### 5. **Error Recovery**
- Automatic retry on transient failures (default: 3 attempts)
- Individual symbol failures don't stop the batch
- Detailed error logging for troubleshooting

### 6. **Scheduled Execution**
- Automated overnight runs (default: 2:00 AM)
- Background service integration
- Configurable schedule

## Architecture

### Components

1. **BatchPredictionService** (`Quantra.DAL.Services.BatchPredictionService`)
   - Core service for batch prediction generation
   - Handles symbol retrieval, processing, and result storage
   - Implements throttling and retry logic

2. **ScheduledPredictionService** (`Quantra.DAL.Services.ScheduledPredictionService`)
   - Background service for scheduled execution
   - Calculates next run time and manages execution
   - Integrates with .NET Generic Host

3. **BatchPredictionControl** (`Quantra.Controls.BatchPredictionControl`)
   - WPF user control for manual batch execution
   - Real-time progress visualization
   - Configuration UI

## Usage

### Method 1: Scheduled Overnight Execution (Recommended)

The system automatically runs every night at 2:00 AM. This is configured in the `ScheduledPredictionService`.

**Configuration:**
```csharp
// In ScheduledPredictionService.cs
private readonly TimeSpan _scheduledTime = new TimeSpan(2, 0, 0); // 2:00 AM
private readonly bool _enableScheduledPredictions = true;
```

**To enable/disable scheduled predictions:**
1. Set `_enableScheduledPredictions` to `true` or `false`
2. Restart the application

**Service Registration** (add to your DI container):
```csharp
// In App.xaml.cs or Startup.cs
services.AddHostedService<ScheduledPredictionService>();
services.AddSingleton<BatchPredictionService>();
```

### Method 2: Manual Execution via UI

1. **Open the Prediction Analysis view**
2. **Navigate to the "Batch Prediction" tab**
3. **Configure options:**
   - **Timeframe**: Select data timeframe (1 day, 1 week, 1 month)
   - **Model Type**: Choose ML model (Auto, PyTorch, TensorFlow, Random Forest)
   - **Max Symbols**: Leave blank for all symbols, or specify a limit
   - **Sector Filter**: Filter by specific sector or process all
   - **Skip Recent**: Skip symbols with predictions from last 24 hours (recommended)

4. **Click "Start Batch Prediction"**

5. **Monitor progress:**
   - Progress bar shows overall completion
   - Real-time metrics display:
     - Total symbols
     - Processed count
     - Successful predictions
     - Failed predictions
     - Success rate
     - Current symbol
     - Elapsed time
     - Estimated time remaining
     - Processing speed (predictions/sec)

6. **Cancel if needed:**
   - Click "Cancel" to stop processing
   - Already-processed predictions are saved
   - Can resume later (with "Skip Recent" enabled)

### Method 3: Programmatic Execution

```csharp
// Inject BatchPredictionService
private readonly BatchPredictionService _batchPredictionService;

// Configure options
var options = new BatchPredictionOptions
{
    SkipRecentPredictions = true,
    PredictionAgeThresholdHours = 24,
    Timeframe = "1day",
    ModelType = "auto",
    MaxSymbolsToProcess = null, // null = process all
    SectorFilter = null, // null = all sectors
    SymbolsToProcess = null // null = all symbols with historical data
};

// Create progress reporter
var progress = new Progress<BatchPredictionProgress>(p =>
{
    Console.WriteLine($"Progress: {p.ProgressPercentage:F1}% - {p.SuccessfulPredictions}/{p.TotalSymbols}");
});

// Execute
var result = await _batchPredictionService.GenerateOvernightPredictionsAsync(
    options, progress, cancellationToken);

// Check result
if (result.Success)
{
    Console.WriteLine($"Complete: {result.SuccessfulPredictions} predictions in {result.Duration}");
}
```

## Configuration

### Batch Processing Settings

Edit `BatchPredictionService.cs` to adjust performance parameters:

```csharp
// Maximum concurrent predictions
private const int MaxConcurrentPredictions = 10; // Increase for faster processing (more CPU/memory)

// Batch size
private const int BatchSize = 100; // Symbols per batch

// Retry attempts
private const int RetryAttempts = 3; // Number of retries on failure

// Delay between batches
private const int DelayBetweenBatchesMs = 1000; // Milliseconds between batches
```

**Performance Tuning:**
- **Increase MaxConcurrentPredictions** for faster processing (requires more system resources)
- **Increase BatchSize** to reduce overhead (requires more memory)
- **Reduce DelayBetweenBatchesMs** for faster processing (may increase API rate limit issues)

### Scheduled Execution Settings

Edit `ScheduledPredictionService.cs` to adjust schedule:

```csharp
// Schedule time (24-hour format)
private readonly TimeSpan _scheduledTime = new TimeSpan(2, 0, 0); // 2:00 AM

// Enable/disable scheduled execution
private readonly bool _enableScheduledPredictions = true;
```

### Prediction Options

```csharp
var options = new BatchPredictionOptions
{
    // Skip symbols with predictions newer than this threshold
    SkipRecentPredictions = true,
    PredictionAgeThresholdHours = 24, // Hours
    
    // Timeframe for technical indicators
    Timeframe = "1day", // "1day", "1week", "1month"
    
    // ML model type
    ModelType = "auto", // "auto", "pytorch", "tensorflow", "random_forest"
    
    // Optional: Limit number of symbols
    MaxSymbolsToProcess = null, // null = no limit
    
    // Optional: Filter by sector
    SectorFilter = null, // e.g., "Technology", "Healthcare"
    
    // Optional: Specific symbols to process
    SymbolsToProcess = null // e.g., new List<string> { "AAPL", "MSFT" }
};
```

## Performance Expectations

### Expected Processing Times (12,000 symbols)

**Configuration:** Default settings (10 concurrent, 100 batch size)

| Model Type | Average Time per Symbol | Total Time | Notes |
|------------|-------------------------|------------|-------|
| Random Forest | 0.5-1.0 seconds | 2-3 hours | Fastest, good accuracy |
| PyTorch/TensorFlow | 1.0-2.0 seconds | 3-6 hours | Best accuracy, slower |
| Auto | 0.5-1.5 seconds | 2-5 hours | Adaptive based on availability |

**Factors affecting speed:**
- System CPU and memory
- Network latency for API calls
- Database query performance
- Number of concurrent predictions
- Batch size configuration

### Optimization Strategies

1. **For Speed:**
   - Increase `MaxConcurrentPredictions` to 15-20
   - Use Random Forest model
   - Enable "Skip Recent" with 24-hour threshold
   - Increase `BatchSize` to 200

2. **For Accuracy:**
   - Use PyTorch or TensorFlow models
   - Disable "Skip Recent" for fresh predictions daily
   - Process during low-usage hours

3. **For Resource Conservation:**
   - Decrease `MaxConcurrentPredictions` to 5-8
   - Increase `DelayBetweenBatchesMs` to 2000-3000
   - Process in smaller batches (50-75 symbols)

## Database Schema

### StockPredictions Table
Stores generated predictions:
```sql
CREATE TABLE StockPredictions (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Symbol TEXT NOT NULL,
    PredictedAction TEXT NOT NULL, -- BUY, SELL, HOLD
    Confidence REAL NOT NULL, -- 0.0 to 1.0
    CurrentPrice REAL NOT NULL,
    TargetPrice REAL NOT NULL,
    PotentialReturn REAL NOT NULL,
    CreatedDate TEXT NOT NULL,
    TradingRule TEXT,
    ModelType TEXT,
    InferenceTimeMs REAL
);
```

### PredictionCache Table
Caches predictions for fast retrieval:
```sql
CREATE TABLE PredictionCache (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Symbol TEXT NOT NULL,
    ModelVersion TEXT NOT NULL,
    InputDataHash TEXT NOT NULL,
    PredictedAction TEXT,
    Confidence REAL,
    PredictedPrice REAL,
    PredictionTimestamp TEXT,
    CreatedAt TEXT NOT NULL,
    AccessCount INTEGER DEFAULT 0,
    LastAccessedAt TEXT
);
```

## Monitoring and Troubleshooting

### Log Files

The system logs to the application logging service. Key log entries:

```
Info: Starting overnight batch prediction generation
Info: Found 12543 symbols to process
Info: Processing 12543 symbols in 126 batches
Info: Batch progress: 1000/12543 (7.9%) - 987 successful, 13 failed - ETA: 02:45:32
Info: Batch prediction complete: 12345/12543 successful in 03:15:42
```

### Common Issues

**1. High failure rate**
- Check model training status
- Verify historical data availability
- Check API rate limits
- Review error logs for specific symbols

**2. Slow processing**
- Increase `MaxConcurrentPredictions`
- Check system resource usage (CPU, memory)
- Verify database performance
- Check network latency

**3. Memory issues**
- Decrease `BatchSize`
- Decrease `MaxConcurrentPredictions`
- Verify sufficient available RAM

**4. Scheduled execution not running**
- Verify `_enableScheduledPredictions = true`
- Check service registration in DI container
- Review application logs for errors

### Health Checks

Add to your monitoring dashboard:

```csharp
// Check if batch prediction service is healthy
var lastBatchResult = await GetLastBatchResultAsync();
if (lastBatchResult != null)
{
    Console.WriteLine($"Last batch: {lastBatchResult.CompletedAt}");
    Console.WriteLine($"Success rate: {lastBatchResult.SuccessRate:F1}%");
    Console.WriteLine($"Symbols processed: {lastBatchResult.SuccessfulPredictions}");
}
```

## Best Practices

1. **Run during off-hours**: Schedule for early morning (2-4 AM) when system load is low
2. **Enable "Skip Recent"**: Saves processing time by only updating stale predictions
3. **Monitor first run**: Watch the first execution to ensure proper configuration
4. **Start small**: Test with `MaxSymbolsToProcess = 100` before processing all symbols
5. **Review logs daily**: Check for patterns in failures or performance issues
6. **Adjust concurrent limit**: Find optimal balance between speed and stability for your system
7. **Regular cleanup**: Periodically clean up old predictions (>30 days) to maintain database performance

## Integration with Prediction Analysis View

Generated predictions are automatically available in the Prediction Analysis view:

1. **Cached symbol search** shows all symbols with predictions
2. **"Load Cached Predictions"** button retrieves latest predictions
3. **Automated mode** can use cached predictions for trading decisions
4. **Cache metadata** shows when each prediction was generated

## Future Enhancements

Potential improvements for future versions:

1. **Priority Queue**: Process high-priority symbols first (e.g., watchlist, high volume)
2. **Distributed Processing**: Support multiple machines for faster processing
3. **Cloud Integration**: Use cloud computing for scale-out processing
4. **Adaptive Scheduling**: Adjust schedule based on market conditions
5. **Model Versioning**: Track which model version generated each prediction
6. **A/B Testing**: Compare different model configurations
7. **Real-time Updates**: Incrementally update predictions during market hours
8. **Notification System**: Send alerts on completion or errors

## Support

For issues or questions:
1. Check application logs
2. Review error messages in the UI
3. Verify configuration settings
4. Test with small symbol count first
5. Check database connectivity and historical data availability

## Summary

The Overnight Batch Prediction System provides an efficient, scalable solution for generating predictions for large datasets. With smart filtering, progress tracking, and error recovery, it ensures reliable prediction generation even for 12,000+ symbols. The flexible configuration allows optimization for speed, accuracy, or resource conservation based on your specific needs.
