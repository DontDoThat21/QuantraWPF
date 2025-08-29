# Performance Profiling with Timing Logs

This feature adds detailed timing logs before and after each indicator load and UI update to help identify performance bottlenecks in the application.

## What is Logged

The system now logs timing information for:

### Indicator Loading Operations
- Individual indicator calculations (RSI, ADX, ATR, MACD, Bollinger Bands, etc.)
- Batch indicator loading operations  
- GetIndicatorsForPrediction operations per symbol/timeframe

### UI Update Operations
- Setting loading states
- Updating indicator displays
- Error handling and notifications
- Clearing loading states

### Visualization Operations
- PopulateIndicatorData for chart visualizations
- Individual technical indicator calculations for charts

## Log Format

Timing logs appear in the console/debug output with the format:
```
PERFORMANCE: {OperationName} completed in {ExecutionTimeMs:F2}ms (Success: {Success})
```

Example output:
```
PERFORMANCE: GetRSI_AAPL_1day completed in 125.34ms (Success: True)
PERFORMANCE: UIUpdate_SetIndicators_AAPL completed in 3.67ms (Success: True)
PERFORMANCE: GetIndicatorsForPrediction_AAPL_1day completed in 1250.89ms (Success: True)
PERFORMANCE: CalculateBollingerBands completed in 45.12ms (Success: True)
```

## Configuration

Console logging is enabled by default via `appsettings.json`:
```json
{
  "Logging": {
    "UseConsoleLogging": true,
    "MinimumLevel": "Information"
  }
}
```

To see detailed timing logs, ensure the following log sources are set to Information level or higher:
- `Quantra.CrossCutting.Monitoring`
- `Quantra.Services` 
- `Quantra.Controls`

## Performance Analysis

Use the timing logs to identify:

1. **Slow Indicators**: Look for indicator calculations taking >100ms
2. **UI Bottlenecks**: Look for UI update operations taking >10ms
3. **Batch Loading Issues**: Look for total GetIndicatorsForPrediction times >2000ms
4. **Visualization Performance**: Look for PopulateIndicatorData times >500ms

## Implementation Details

The timing functionality uses the existing `MonitoringManager` class which:
- Records execution times using `Stopwatch` for high precision
- Logs to console/debug via the `LoggingManager` 
- Maintains performance metrics and statistics
- Supports both synchronous and asynchronous operations

No additional dependencies were added - this leverages the existing logging and monitoring infrastructure.