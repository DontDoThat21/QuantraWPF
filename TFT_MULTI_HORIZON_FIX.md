# TFT Multi-Horizon JSON Deserialization Fix

## Problem Summary

**Exception**: `System.Text.Json.JsonException: The JSON value could not be converted to System.Collections.Generic.List`1[Quantra.DAL.Services.TFTHorizonData]`

**Root Cause**: The Python TFT integration script was generating the `horizons` field as a **dictionary** with keys like `'5d'`, `'10d'`, etc., but the C# model expected it to be a **list of objects** with snake_case property names (`days_ahead`, `predicted_change`, `lower_bound`, `upper_bound`).

### Original Python Output (Incorrect)
```python
'horizons': {
    '5d': {
        'median_price': 280.5,
        'lower_bound': 275.0,
        'upper_bound': 285.0,
        'confidence': 0.85
    },
    '10d': { ... },
    ...
}
```

### Expected C# Model
```csharp
public class TFTHorizonData
{
    [JsonPropertyName("days_ahead")]
    public int DaysAhead { get; set; }

    [JsonPropertyName("predicted_change")]
    public double PredictedChange { get; set; }

    [JsonPropertyName("lower_bound")]
    public double LowerBound { get; set; }

    [JsonPropertyName("upper_bound")]
    public double UpperBound { get; set; }
}
```

## Solution Applied

### 1. Fixed Python Output Structure (`Quantra\python\tft_integration.py`)

Changed the `predict_single` method to generate horizons as a **list** with correct property names:

```python
# Build multi-horizon response as LIST (not dict) for C# deserialization
horizons_data = []
for i, horizon in enumerate(self.forecast_horizons):
    # median_change is a percentage change (e.g., 0.05 = 5%)
    median_change = median_predictions[i]
    
    horizons_data.append({
        'days_ahead': int(horizon),
        'predicted_change': float(median_change),
        'lower_bound': float(lower_bounds[i]),
        'upper_bound': float(upper_bounds[i])
    })
```

**Key Changes**:
- Changed from `horizons_data = {}` (dict) to `horizons_data = []` (list)
- Changed from `horizons_data[f'{horizon}d'] = {...}` to `horizons_data.append({...})`
- Used correct property names: `days_ahead`, `predicted_change`, `lower_bound`, `upper_bound`
- Values are now percentage changes (e.g., 0.05 = 5%) instead of absolute prices

### 2. Enhanced C# Error Handling (`Quantra.DAL\Services\TFTPredictionService.cs`)

Added better error handling and logging for JSON deserialization failures:

```csharp
TFTPythonResponse pythonResult;
try
{
    pythonResult = JsonSerializer.Deserialize<TFTPythonResponse>(jsonResult, readOptions);
}
catch (JsonException jsonEx)
{
    _loggingService?.Log("Error", $"Failed to deserialize TFT response. First 500 chars: {jsonResult.Substring(0, Math.Min(500, jsonResult.Length))}");
    throw new Exception($"Failed to parse TFT prediction result: {jsonEx.Message}", jsonEx);
}
```

### 3. Added Debug Logging

Enhanced the `ConvertPythonResponseToResult` method with detailed logging:

```csharp
_loggingService?.Log("Debug", $"Processing {pythonResult.Horizons.Count} horizon predictions");

foreach (var horizon in pythonResult.Horizons)
{
    // ... processing ...
    _loggingService?.Log("Debug", $"Horizon {horizon.DaysAhead}d: Change={horizon.PredictedChange:P2}, Price=${predictedPrice:F2}");
}
```

## New JSON Output Format

The Python script now generates JSON in this format:

```json
{
  "symbol": "AAPL",
  "action": "BUY",
  "confidence": 0.75,
  "currentPrice": 278.78,
  "targetPrice": 283.50,
  "medianPrediction": 0.0169,
  "lowerBound": 267.80,
  "upperBound": 295.60,
  "horizons": [
    {
      "days_ahead": 5,
      "predicted_change": 0.0169,
      "lower_bound": -0.0394,
      "upper_bound": 0.0603
    },
    {
      "days_ahead": 10,
      "predicted_change": 0.0225,
      "lower_bound": -0.0512,
      "upper_bound": 0.0785
    },
    {
      "days_ahead": 20,
      "predicted_change": 0.0340,
      "lower_bound": -0.0680,
      "upper_bound": 0.1156
    },
    {
      "days_ahead": 30,
      "predicted_change": 0.0480,
      "lower_bound": -0.0850,
      "upper_bound": 0.1520
    }
  ],
  "modelType": "tft",
  "uncertainty": 0.0997,
  "featureImportance": [0.12, 0.08, ...]
}
```

## Benefits

1. **Proper Deserialization**: C# can now correctly deserialize the horizons field into `List<TFTHorizonData>`
2. **Correct Data Types**: 
   - `days_ahead` is an integer (5, 10, 20, 30)
   - `predicted_change` is the percentage change (-0.05 to 0.10 = -5% to 10%)
   - Bounds are also percentage changes relative to current price
3. **Better Error Messages**: When deserialization fails, developers now see the actual JSON content
4. **Debug Visibility**: Logging shows exactly how many horizons are processed and their values

## Testing

After this fix, when running TFT predictions:

1. The Python script will generate properly formatted JSON with horizons as a list
2. C# will successfully deserialize the response without JsonException
3. Multiple horizon predictions (5d, 10d, 20d, 30d) will be available in the result
4. Debug logs will show the processing of each horizon prediction

## Files Modified

1. **Quantra\python\tft_integration.py** - Fixed horizons data structure in `predict_single` method
2. **Quantra.DAL\Services\TFTPredictionService.cs** - Enhanced error handling and logging

## Related Issues

This fix resolves the JSON deserialization error that was preventing TFT multi-horizon predictions from being processed by the C# application.
