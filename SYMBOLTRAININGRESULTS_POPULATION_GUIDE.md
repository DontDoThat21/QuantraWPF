# SymbolTrainingResults Table Population Guide

## Overview

The `SymbolTrainingResults` table stores **per-symbol metrics from ML model training sessions**. It is **NOT** populated during prediction analysis runs - it's specifically for tracking which symbols were used in training and their individual performance characteristics.

## When the Table is Populated

The table is populated **only during ML model training** via:
1. User clicks "Train Model" button in PredictionAnalysisControl
2. `ModelTrainingService.TrainModelFromDatabaseAsync()` executes
3. Python script `train_from_database.py` runs
4. Training results are returned with per-symbol metrics
5. `ModelTrainingHistoryService.LogSymbolResultsWithDatesAsync()` saves the results

## Table Schema

```sql
CREATE TABLE SymbolTrainingResults (
    Id INT PRIMARY KEY IDENTITY,
    TrainingHistoryId INT NOT NULL,           -- Links to ModelTrainingHistory
    Symbol NVARCHAR(20) NOT NULL,             -- Stock symbol (e.g., "AAPL")
    DataPointsCount INT NOT NULL,             -- Total historical data points available
    TrainingSamplesCount INT NOT NULL,        -- Number of samples used for training
    TestSamplesCount INT NOT NULL,            -- Number of samples used for testing
    SymbolMAE FLOAT NULL,                     -- Mean Absolute Error for this symbol (future)
    SymbolRMSE FLOAT NULL,                    -- Root Mean Squared Error (future)
    IncludedInTraining BIT NOT NULL,          -- Whether symbol was included in training
    ExclusionReason NVARCHAR(500) NULL,       -- Why symbol was excluded (if applicable)
    DataStartDate DATETIME NULL,              -- Earliest date in the symbol's training data
    DataEndDate DATETIME NULL,                -- Latest date in the symbol's training data
    FOREIGN KEY (TrainingHistoryId) REFERENCES ModelTrainingHistory(Id)
);
```

## How to Populate the Table

### Step 1: Ensure Historical Data is Cached

Before training, make sure symbols have cached historical data in the `StockDataCache` table:

```csharp
// In PredictionAnalysisControl or Stock Explorer
await _stockDataCacheService.CacheHistoricalDataAsync(symbol, "1y");
```

### Step 2: Train the Model

In the **PredictionAnalysisControl** view:

1. **Select Model Type**: Choose from PyTorch, TensorFlow, Random Forest, or Auto
2. **Select Architecture**: For neural networks, choose LSTM, GRU, or Transformer
3. **Optional: Limit Symbols**:
   - Enter a number in "Max Symbols" to limit training data
   - OR click "Select Training Symbols" to pick specific symbols
4. **Click "Train Model"**

The training process will:
- Fetch all cached historical data from the database
- Prepare features for each symbol
- Track which symbols are included/excluded with reasons
- Train the ML model
- Save results to `ModelTrainingHistory` table
- Save per-symbol metrics to `SymbolTrainingResults` table

### Step 3: Verify Results

After training completes, query the table:

```sql
-- View all symbol results for the latest training session
SELECT sr.Symbol, sr.DataPointsCount, sr.TrainingSamplesCount, sr.TestSamplesCount,
       sr.IncludedInTraining, sr.ExclusionReason, sr.DataStartDate, sr.DataEndDate
FROM SymbolTrainingResults sr
INNER JOIN ModelTrainingHistory mth ON sr.TrainingHistoryId = mth.Id
WHERE mth.Id = (SELECT TOP 1 Id FROM ModelTrainingHistory ORDER BY TrainingDate DESC)
ORDER BY sr.Symbol;

-- View symbols that were excluded from training
SELECT Symbol, ExclusionReason, DataPointsCount
FROM SymbolTrainingResults
WHERE IncludedInTraining = 0
ORDER BY Symbol;

-- View training coverage by symbol
SELECT Symbol, 
       DataPointsCount,
       TrainingSamplesCount,
       TestSamplesCount,
       DATEDIFF(day, DataStartDate, DataEndDate) as DaysCovered
FROM SymbolTrainingResults
WHERE IncludedInTraining = 1
ORDER BY DataPointsCount DESC;
```

## Common Exclusion Reasons

Symbols may be excluded from training for several reasons:

1. **Insufficient data points**: Less than 50 historical data points
2. **Data preparation error**: Failed to calculate technical indicators
3. **Missing price data**: Incomplete OHLCV (Open, High, Low, Close, Volume) data
4. **Invalid values**: NaN or infinite values in the data

## What Gets Stored

For each symbol in the training dataset:

- **Symbol**: Stock ticker
- **DataPointsCount**: Total historical price records available
- **TrainingSamplesCount**: Number of training samples generated (after windowing)
- **TestSamplesCount**: Number of test samples generated
- **IncludedInTraining**: True if symbol contributed to the model
- **ExclusionReason**: Explanation if excluded (null if included)
- **DataStartDate**: Earliest date in the training data
- **DataEndDate**: Latest date in the training data

## Future Enhancements

The table schema includes `SymbolMAE` and `SymbolRMSE` columns for storing per-symbol performance metrics. These are currently unused but could be populated by:

1. Modifying the Python training script to evaluate each symbol separately
2. Calculating per-symbol MAE/RMSE during training
3. Returning these metrics in the JSON result
4. Storing them via `LogSymbolResultsWithDatesAsync()`

## Code Changes Made

### 1. Python Script (`train_from_database.py`)
- Added `symbol_metrics` list to track per-symbol data
- Store metrics for included, excluded, and errored symbols
- Return `symbol_results` in JSON output

### 2. C# Model Classes (`ModelTrainingService.cs`)
- Added `SymbolTrainingMetric` class to deserialize Python results
- Added `SymbolResults` property to `ModelTrainingResult`

### 3. Service Layer (`ModelTrainingHistoryService.cs`)
- Added `LogSymbolResultsWithDatesAsync()` method to save full metrics including date ranges
- Existing `LogSymbolResultsAsync()` retained for backward compatibility

### 4. UI Layer (`PredictionAnalysisControl.Analysis.cs`)
- Updated to call `LogSymbolResultsWithDatesAsync()` after training
- Logs count of symbol results saved

## Troubleshooting

**Q: Table is empty after running predictions**  
A: The table is only populated during **model training**, not prediction runs. Predictions save to `StockPredictions` table instead.

**Q: Training completed but no symbol results saved**  
A: Check the application logs for errors. Ensure the Python script returned `symbol_results` in the JSON output.

**Q: All symbols show as excluded**  
A: Verify that symbols have sufficient historical data cached. Each symbol needs at least 50 data points to be included in training.

**Q: How do I train on specific symbols only?**  
A: Use the "Select Training Symbols" button in the PredictionAnalysisControl UI to choose specific symbols before training.

## Related Tables

- **ModelTrainingHistory**: Overall training session metrics (parent table)
- **StockDataCache**: Historical price data used for training
- **StockPredictions**: Prediction results (separate from training)
- **PredictionIndicators**: Technical indicators for predictions

## Example Workflow

```csharp
// Complete workflow to populate SymbolTrainingResults

// 1. Cache historical data for symbols
var symbols = new[] { "AAPL", "GOOGL", "MSFT", "TSLA" };
foreach (var symbol in symbols)
{
    await _stockDataCacheService.CacheHistoricalDataAsync(symbol, "1y");
}

// 2. Train the model (from UI or code)
var result = await _modelTrainingService.TrainModelFromDatabaseAsync(
    modelType: "pytorch",
    architectureType: "lstm",
    maxSymbols: 10
);

// 3. Save training history
int trainingHistoryId = await _modelTrainingHistoryService.LogTrainingSessionAsync(
    result,
    notes: "Training run with selected symbols"
);

// 4. Save per-symbol results
if (result.SymbolResults != null && result.SymbolResults.Count > 0)
{
    await _modelTrainingHistoryService.LogSymbolResultsWithDatesAsync(
        trainingHistoryId,
        result.SymbolResults
    );
}

// 5. Query results
var symbolResults = await _modelTrainingHistoryService.GetSymbolResultsAsync(trainingHistoryId);
foreach (var sr in symbolResults)
{
    Console.WriteLine($"{sr.Symbol}: {sr.TrainingSamplesCount} samples, Included: {sr.IncludedInTraining}");
}
```

## Summary

The `SymbolTrainingResults` table is a **training audit trail** that shows:
- Which symbols contributed to the ML model
- How much data each symbol provided
- Why symbols were excluded
- Date ranges of training data

To populate it, **train a model** (not run predictions) using the UI or API. The table will automatically be populated with per-symbol metrics from the training session.
