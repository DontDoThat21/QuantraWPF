# Integration Guide: Connecting Candlestick Chart to Analysis Workflow

## Quick Start

The candlestick chart is now ready to be integrated with your prediction analysis workflow. Here's how to connect it:

## Integration Points

### 1. After Symbol Analysis

Add this call at the end of `AnalyzeIndividualSymbol(string symbol)` method in `PredictionAnalysis.EventHandlers.cs`:

```csharp
private async Task AnalyzeIndividualSymbol(string symbol)
{
    try
    {
        // ... existing analysis code ...
        
        // YOUR EXISTING PREDICTION LOGIC HERE
        
        // ADD THIS LINE AT THE END:
        // Update the candlestick chart with the analyzed symbol
        await UpdateChartWithPredictionsAsync(symbol);
        
        _loggingService?.Log("Info", $"Completed analysis and chart update for {symbol}");
    }
    catch (Exception ex)
    {
        _loggingService?.LogErrorWithContext(ex, $"Failed to analyze {symbol}");
    }
}
```

### 2. In Refresh Button Handler

Add this call in `RefreshButton_Click` method in `PredictionAnalysis.xaml.cs`:

```csharp
private async void RefreshButton_Click(object sender, RoutedEventArgs e)
{
    try
    {
        // ... existing refresh logic ...
        
        // ADD THIS LINE:
        await RefreshChartForCurrentSymbol();
        
        _loggingService?.Log("Info", "Refresh completed");
    }
    catch (Exception ex)
    {
        _loggingService?.LogErrorWithContext(ex, "Failed to refresh");
    }
}
```

### 3. After Manual Symbol Entry

In `ManualSymbolTextBox_KeyDown` method in `PredictionAnalysis.EventHandlers.cs`:

```csharp
private async void ManualSymbolTextBox_KeyDown(object sender, KeyEventArgs e)
{
    if (e.Key == Key.Enter)
    {
        var symbol = ManualSymbolTextBox.Text?.Trim()?.ToUpper();
        if (!string.IsNullOrEmpty(symbol))
        {
            // YOUR EXISTING ANALYSIS LOGIC
            await AnalyzeIndividualSymbol(symbol);
            
            // ADD THIS LINE:
            // Chart will be updated inside AnalyzeIndividualSymbol
        }
    }
}
```

### 4. After Prediction Generation

In `RunPredictionAlgorithms(string symbol)` method in `PredictionAnalysis.Automation.cs`:

```csharp
public async Task<Quantra.Models.PredictionModel> RunPredictionAlgorithms(string symbol)
{
    try
    {
        // ... existing prediction logic ...
        
        var prediction = /* your prediction result */;
        
        // ADD THIS LINE:
        await OnPredictionGenerated(prediction);
        
        return prediction;
    }
    catch (Exception ex)
    {
        _loggingService?.LogErrorWithContext(ex, "Prediction failed");
        return null;
    }
}
```

## Complete Example

Here's a complete example of an integrated analysis method:

```csharp
private async Task AnalyzeIndividualSymbol(string symbol)
{
    try
    {
        _loggingService?.Log("Info", $"Starting analysis for {symbol}");
        
        // 1. Fetch indicators
        var indicators = await _indicatorService.GetIndicatorsForPrediction(symbol, "1d");
        
        // 2. Run prediction algorithms
        var prediction = await RunPredictionAlgorithms(symbol);
        
        // 3. Save prediction to database
        if (prediction != null)
        {
            await SavePredictionWithModelInfoAsync(prediction);
        }
        
        // 4. Update UI with prediction results
        if (prediction != null && Predictions != null)
        {
            Dispatcher.Invoke(() => 
            {
                Predictions.Add(prediction);
            });
        }
        
        // 5. UPDATE CANDLESTICK CHART - NEW!
        await UpdateChartWithPredictionsAsync(symbol);
        
        _loggingService?.Log("Info", $"Completed analysis and chart update for {symbol}");
    }
    catch (Exception ex)
    {
        _loggingService?.LogErrorWithContext(ex, $"Failed to analyze {symbol}");
        // Clear chart on error
        ClearCandlestickChart();
    }
}
```

## Chart Update Flow

```
User Action (Enter Symbol/Click Analyze)
    ?
AnalyzeIndividualSymbol(symbol)
    ?
Fetch Indicators + Run Predictions
    ?
UpdateChartWithPredictionsAsync(symbol)
    ?
?? Fetch Historical OHLCV Data (HistoricalDataService)
?  ?? 3 months of daily data
?? Get TFT Predictions (TFTPredictionService)
?  ?? Multi-horizon forecasts (5, 10, 20, 30 days)
?? UpdateCandlestickChart(symbol, data, predictions)
   ?? Create historical candlesticks
   ?? Add prediction lines and confidence bands
   ?? Add volume bars
   ?? Update UI bindings
```

## Error Handling Pattern

Always wrap chart updates with try-catch to prevent errors from breaking the analysis flow:

```csharp
try
{
    await UpdateChartWithPredictionsAsync(symbol);
}
catch (Exception chartEx)
{
    // Log error but don't fail the entire analysis
    _loggingService?.LogErrorWithContext(chartEx, 
        $"Chart update failed for {symbol}, continuing with analysis");
}
```

## Testing the Integration

### Step 1: Minimal Integration
Add chart update to one place first (e.g., `AnalyzeIndividualSymbol`):

```csharp
await UpdateChartWithPredictionsAsync(symbol);
```

### Step 2: Test Basic Functionality
1. Enter a symbol (e.g., "AAPL")
2. Click Analyze
3. Verify chart appears with historical data
4. Check console logs for success/error messages

### Step 3: Test with Predictions
1. Ensure TFT model is trained and available
2. Analyze a symbol
3. Verify prediction bands appear on chart
4. Check that horizons (5d, 10d, etc.) are displayed

### Step 4: Test Error Cases
1. Try invalid symbol (e.g., "INVALID")
2. Verify chart clears gracefully
3. Check logs for appropriate error messages

## Troubleshooting

### Chart Not Appearing After Analysis

**Check 1**: Verify method is being called
```csharp
_loggingService?.Log("Info", "About to update chart");
await UpdateChartWithPredictionsAsync(symbol);
_loggingService?.Log("Info", "Chart update completed");
```

**Check 2**: Verify historical data is fetched
```csharp
// In UpdateChartWithPredictionsAsync
if (historicalData == null || historicalData.Count == 0)
{
    _loggingService?.Log("Warning", $"No data for {symbol}");
    // Check why data fetch failed
}
```

**Check 3**: Verify `IsChartVisible` property
```csharp
// After chart update
_loggingService?.Log("Info", $"IsChartVisible: {IsChartVisible}");
```

### Predictions Not Showing

**Check 1**: Verify TFT service is available
```csharp
if (_realTimeInferenceService == null)
{
    _loggingService?.Log("Warning", "TFT service not available");
}
```

**Check 2**: Check TFT result structure
```csharp
if (tftResult != null)
{
    _loggingService?.Log("Info", 
        $"TFT horizons: {tftResult.Horizons?.Count ?? 0}");
}
```

### Performance Issues

**Symptom**: UI freezes during chart update

**Solution**: Ensure async/await pattern is used correctly
```csharp
// WRONG - blocks UI thread
var data = _historicalDataService.GetHistoricalPrices(symbol, "3month", "1d").Result;

// CORRECT - doesn't block
var data = await _historicalDataService.GetHistoricalPrices(symbol, "3month", "1d");
```

## Advanced Customization

### Custom Time Ranges

Modify the timeframe in `UpdateChartWithPredictionsAsync`:

```csharp
// Short-term view (1 month)
var historicalData = await _historicalDataService?.GetHistoricalPrices(
    symbol, "1month", "1d");

// Long-term view (1 year)
var historicalData = await _historicalDataService?.GetHistoricalPrices(
    symbol, "1year", "1wk");
```

### Conditional Chart Display

Only show chart for high-confidence predictions:

```csharp
if (prediction != null && prediction.Confidence >= 0.75)
{
    await UpdateChartWithPredictionsAsync(symbol);
}
else
{
    ClearCandlestickChart();
}
```

### Multiple Symbol Comparison

Loop through symbols and update chart:

```csharp
foreach (var symbol in selectedSymbols)
{
    await AnalyzeIndividualSymbol(symbol);
    // Chart will update for each symbol
    await Task.Delay(500); // Optional delay between updates
}
```

## Next Steps

1. **Add to EventHandlers.cs**: Insert `await UpdateChartWithPredictionsAsync(symbol);` in `AnalyzeIndividualSymbol`
2. **Add to xaml.cs**: Insert `await RefreshChartForCurrentSymbol();` in `RefreshButton_Click`
3. **Test**: Run application and analyze a symbol
4. **Verify**: Check that candlestick chart displays with predictions
5. **Optimize**: Adjust timeframes, colors, and features as needed

## Support

For issues or questions:
1. Check console logs for error messages
2. Verify all dependencies are properly injected
3. Ensure TFT model is trained and available
4. Review CANDLESTICK_CHART_DOCUMENTATION.md for detailed API reference
