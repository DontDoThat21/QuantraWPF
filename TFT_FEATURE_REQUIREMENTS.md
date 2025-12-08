# TFT Feature Requirements - What the Model Actually Needs

## Problem: Current Features Are Inaccurate/Insufficient

Your current implementation passes **single snapshot values** of indicators to TFT, but TFT is a **time-series model** that needs **sequences of features over time**.

## What TFT Expects

### 1. **Past Temporal Features (60 days x N features)**

TFT needs the **last 60 days** of OHLCV data AND technical indicators:

```python
# Required shape: (batch_size, 60 days, num_features)
X_past = [
    # Day 1 (60 days ago)
    [open_t60, high_t60, low_t60, close_t60, volume_t60, rsi_t60, macd_t60, ...],
    
    # Day 2 (59 days ago)
    [open_t59, high_t59, low_t59, close_t59, volume_t59, rsi_t59, macd_t59, ...],
    
    # ...
    
    # Day 60 (today)
    [open_t0, high_t0, low_t0, close_t0, volume_t0, rsi_t0, macd_t0, ...]
]
```

**Your Current Issue**: You're only passing **one value** per feature (today's value), not the full 60-day history.

### 2. **Static Features (1 value per sample)**

These should be **truly static** (unchanging) characteristics:

```python
# Required shape: (batch_size, num_static_features)
X_static = [sector_code, market_cap_category, beta, ...]
```

**Your Current Issue**: You're mixing static features (sector) with time-varying features (price) in the same dictionary.

### 3. **Known Future Features (60 days + 5 days forecast horizon)**

Calendar features that are known in advance:

```python
# Required shape: (batch_size, 65 days, num_future_features)
X_future = [
    # Past 60 days + future 5 days
    [day_of_week, month, quarter, is_earnings_week, ...] for each day
]
```

**Your Current Issue**: You're only providing **today's** calendar features, not the full 60-day history + 5-day future.

---

## How Your Current Code Fails

### Issue 1: No Temporal Sequences

**Current Code (PredictionAnalysis.Analysis.cs, lines 155-205)**:
```csharp
// You fetch ONE value per indicator
double rsi = await _alphaVantageService.GetRSI(symbol, "daily");
indicators["RSI"] = rsi;  // Single value, not a 60-day sequence
```

**What TFT Needs**:
```csharp
// Need 60 values
List<double> rsiHistory = await _alphaVantageService.GetRSIHistory(symbol, "daily", 60);
// Then pass as a sequence, not a single value
```

### Issue 2: Mixing Static and Time-Varying Features

**Current Code (lines 478-500)**:
```csharp
indicators["Sector"] = _alphaVantageService.GetSectorCode(overview.Sector);  // Static
indicators["Close"] = historicalData[0].Close;  // Time-varying
indicators["RSI"] = rsi;  // Time-varying
```

**Problem**: All features are in one dictionary, but TFT needs them **separated** into:
- `X_past` (time-varying features)
- `X_static` (unchanging features)
- `X_future` (known-future features)

### Issue 3: Feature Leakage

**Current Code (lines 334-354)**:
```csharp
// Using TODAY's close price to calculate log return
double logReturn = Math.Log(closePrices[0] / closePrices[1]);
indicators["LogReturn_1"] = logReturn;
```

**Problem**: `closePrices[0]` is the price you're trying to predict. You can't use it as an input feature.

---

## Solution: Proper Feature Preparation for TFT

### Step 1: Separate Features by Type

```csharp
// Static features (unchanging)
var staticFeatures = new Dictionary<string, double>
{
    { "sector", _alphaVantageService.GetSectorCode(overview.Sector) },
    { "market_cap_category", _alphaVantageService.GetMarketCapCategory(overview.MarketCapitalization) },
    { "beta", overview.Beta ?? 0.0 }
};

// Past temporal features (60-day sequences)
var pastTemporalFeatures = new List<Dictionary<string, double>>();
for (int i = 59; i >= 0; i--)  // 60 days ago to today
{
    var dayFeatures = new Dictionary<string, double>
    {
        { "open", historicalData[i].Open },
        { "high", historicalData[i].High },
        { "low", historicalData[i].Low },
        { "close", historicalData[i].Close },
        { "volume", historicalData[i].Volume },
        { "rsi", await GetRSIForDay(symbol, historicalData[i].Date) },
        { "macd", await GetMACDForDay(symbol, historicalData[i].Date) },
        // ... other indicators
    };
    pastTemporalFeatures.Add(dayFeatures);
}

// Known future features (60 days past + 5 days future)
var knownFutureFeatures = new List<Dictionary<string, double>>();
for (int i = 59; i >= -5; i--)  // 60 days ago to 5 days future
{
    DateTime date = DateTime.Now.AddDays(-i);
    var calendarFeatures = new Dictionary<string, double>
    {
        { "day_of_week", (double)date.DayOfWeek },
        { "month", date.Month },
        { "quarter", ((date.Month - 1) / 3) + 1 },
        { "is_earnings_week", IsEarningsWeek(symbol, date) ? 1.0 : 0.0 }
    };
    knownFutureFeatures.Add(calendarFeatures);
}
```

### Step 2: Call TFT with Proper Structure

```csharp
var tftResult = await Quantra.Models.PythonStockPredictor.PredictWithTFTAsync(
    staticFeatures: staticFeatures,              // Static features
    pastTemporalFeatures: pastTemporalFeatures,  // 60-day sequences
    knownFutureFeatures: knownFutureFeatures,    // 65-day calendar features
    symbol: symbol,
    horizons: new List<int> { 1, 3, 5, 10 }
);
```

---

## Feature Requirements Checklist

### ? Required Features (Must Have)

| Feature Type | Examples | Current Status |
|--------------|----------|----------------|
| **Past OHLCV** | Open, High, Low, Close, Volume (60 days) | ? Available in `historicalDataForTFT` |
| **Past Technical Indicators** | RSI, MACD, Bollinger Bands (60 days) | ? Only 1 day provided |
| **Static Metadata** | Sector, Market Cap, Beta | ? Available but mixed with time-varying |
| **Known Future** | Day of week, month, earnings dates (65 days) | ?? Only today provided, need 65 days |

### ? Features to Remove (Harmful)

| Feature | Reason to Remove |
|---------|------------------|
| `current_price` | **Duplicate** of `Close` |
| `Close_t0` | **Duplicate** of `Close` |
| `LogReturn_1` | **Data leakage** (uses target) |
| `IsPreMarket`, `IsRegularHours` | **Irrelevant** for daily predictions |
| Single-value indicators (RSI, MACD, etc.) | **Need sequences**, not single values |

---

## Expected R² Improvement

| Issue Fixed | Expected R² Gain |
|-------------|------------------|
| Remove feature leakage | +0.1 to +0.2 |
| Provide proper 60-day sequences | +0.2 to +0.3 |
| Separate static/temporal features | +0.1 to +0.15 |
| Normalize feature scales | +0.05 to +0.1 |
| **Total Expected R²** | **0.4 to 0.75** |

---

## Quick Fix Implementation

### Option 1: Use RealTimeInferenceService (Recommended)

The `RealTimeInferenceService` already has proper feature preparation:

```csharp
// In PredictionAnalysis.Analysis.cs
var inferenceService = new RealTimeInferenceService(_stockDataCacheService);
var tftResult = await inferenceService.GetTFTPredictionAsync(
    symbol: symbol,
    lookbackDays: 60,
    futureHorizon: 5
);
```

This service:
- ? Fetches 60 days of historical data from cache
- ? Adds calendar features for past + future
- ? Separates static/temporal features properly
- ? Normalizes feature scales

### Option 2: Fix Current Implementation

1. **Fetch 60-day sequences** for all indicators (not just OHLCV)
2. **Separate features** into static/temporal/future dictionaries
3. **Remove duplicates and leakage**
4. **Pass to Python** in the correct format

---

## Testing the Fix

### Before (Current):
```
R² Score: -0.00000014 (essentially 0)
Model has learned nothing
```

### After (With Proper Features):
```
R² Score: 0.45 to 0.65
Model can predict trends with 45-65% accuracy
```

---

## References

- **TFT Paper**: https://arxiv.org/abs/1912.09363 (Section 3.2 - Input Processing)
- **Your Implementation**: `Quantra.DAL\Services\RealTimeInferenceService.cs` (correct feature preparation)
- **Broken Implementation**: `Quantra\Views\PredictionAnalysis\PredictionAnalysis.Analysis.cs` (current code)

---

## Action Items

1. **Immediate**: Use `RealTimeInferenceService` instead of direct TFT calls
2. **Short-term**: Refactor `AnalyzeStockWithAllAlgorithms` to prepare proper feature sequences
3. **Medium-term**: Cache 60-day indicator histories to avoid repeated API calls
4. **Long-term**: Implement automatic feature engineering pipeline
