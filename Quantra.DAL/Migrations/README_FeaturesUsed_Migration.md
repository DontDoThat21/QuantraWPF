# FeaturesUsed Column Migration

## Overview

This migration adds a `FeaturesUsed` column to the `ModelTrainingHistory` table to store a JSON list of features used during model training.

## Changes Made

### 1. Database Schema

**New Column**: `FeaturesUsed` (NVARCHAR(MAX))
- Stores a JSON array of feature names
- Example: `["close", "volume", "rsi_14", "sma_20", "macd", ...]`
- NULL if feature list not available

**Computed Column**: `FeatureCount` (INT)
- Automatically calculated from JSON array length
- Persisted for query performance
- Returns 0 if FeaturesUsed is NULL or empty

**Index**: `IX_ModelTrainingHistory_FeaturesUsed`
- Indexed on `FeatureCount` for efficient filtering
- WHERE clause excludes NULL values

### 2. C# Entity Updates

**ModelTrainingHistory.cs**
```csharp
public string FeaturesUsed { get; set; }
```

**ModelTrainingResult.cs**
```csharp
[JsonPropertyName("feature_names")]
public List<string> FeatureNames { get; set; }
```

### 3. Service Layer Updates

**ModelTrainingHistoryService.cs**

New Methods:
- `GetFeatureNamesAsync(int trainingHistoryId)` - Get features for specific training
- `GetActiveModelFeatureNamesAsync(string modelType, string architectureType)` - Get features for active model

Updated Method:
- `LogTrainingSessionAsync()` - Now saves FeaturesUsed as JSON

### 4. Python Integration (To Be Implemented)

The Python training script should return feature names in the result JSON:

```python
result = {
    "success": True,
    "model_type": "pytorch",
    "architecture_type": "transformer",
    "feature_names": list(X_train.columns),  # ADD THIS
    # ... other fields
}
```

## Migration Steps

### Step 1: Run SQL Migration

Execute the migration script:
```sql
-- In SQL Server Management Studio or via EF Core Migration
USE QuantraRelational;
GO
EXEC sp_executesql N'<contents of AddFeaturesUsedToModelTrainingHistory.sql>';
```

Or use Entity Framework Core migration:
```bash
dotnet ef migrations add AddFeaturesUsedToModelTrainingHistory
dotnet ef database update
```

### Step 2: Update Python Training Scripts

Modify `train_from_database.py` to include feature names in the output JSON:

**Location to modify**: After model training, before saving results

```python
# After training and evaluation, before writing results
feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else []

result = {
    "success": True,
    "model_type": model_type,
    "architecture_type": architecture_type,
    "symbols_count": len(unique_symbols),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "training_time_seconds": training_time,
    "performance": {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2)
    },
    "feature_names": feature_names,  # NEW: Add this line
    "symbol_results": symbol_results
}
```

### Step 3: Verify Data Flow

1. **Train a model** via PredictionAnalysis view
2. **Check database** for populated FeaturesUsed column:
```sql
SELECT TOP 1 
    Id, 
    ModelType, 
    ArchitectureType,
    FeatureCount,
    FeaturesUsed
FROM ModelTrainingHistory
WHERE FeaturesUsed IS NOT NULL
ORDER BY TrainingDate DESC;
```

3. **Verify JSON format**:
```sql
-- Should show individual feature names
SELECT feature.value AS FeatureName
FROM ModelTrainingHistory
CROSS APPLY OPENJSON(FeaturesUsed) AS feature
WHERE Id = <your_training_id>;
```

## Usage Examples

### C# - Get Feature Names

```csharp
// Get features for specific training session
var features = await modelTrainingHistoryService.GetFeatureNamesAsync(trainingId);
Console.WriteLine($"Model used {features.Count} features:");
foreach (var feature in features)
{
    Console.WriteLine($"  - {feature}");
}

// Get features for active model
var activeFeatures = await modelTrainingHistoryService.GetActiveModelFeatureNamesAsync(
    "pytorch", 
    "transformer"
);
Console.WriteLine($"Active model uses {activeFeatures.Count} features");
```

### SQL - Query Feature Lists

```sql
-- Get training sessions with their feature counts
SELECT 
    Id,
    ModelType,
    ArchitectureType,
    TrainingDate,
    R2Score,
    FeatureCount
FROM ModelTrainingHistory
WHERE FeaturesUsed IS NOT NULL
ORDER BY TrainingDate DESC;

-- Get all features used by a specific training session
SELECT feature.value AS FeatureName
FROM ModelTrainingHistory
CROSS APPLY OPENJSON(FeaturesUsed) AS feature
WHERE Id = 1;

-- Count models by feature count ranges
SELECT 
    CASE 
        WHEN FeatureCount < 20 THEN 'Minimal (< 20)'
        WHEN FeatureCount < 50 THEN 'Balanced (20-50)'
        ELSE 'Comprehensive (50+)'
    END AS FeatureCategory,
    COUNT(*) AS ModelCount,
    AVG(R2Score) AS AvgR2Score
FROM ModelTrainingHistory
WHERE FeaturesUsed IS NOT NULL
GROUP BY 
    CASE 
        WHEN FeatureCount < 20 THEN 'Minimal (< 20)'
        WHEN FeatureCount < 50 THEN 'Balanced (20-50)'
        ELSE 'Comprehensive (50+)'
    END
ORDER BY AvgR2Score DESC;

-- Find common features across all training sessions
SELECT 
    feature.value AS FeatureName,
    COUNT(*) AS UsageCount,
    AVG(mth.R2Score) AS AvgR2Score
FROM ModelTrainingHistory mth
CROSS APPLY OPENJSON(mth.FeaturesUsed) AS feature
WHERE mth.FeaturesUsed IS NOT NULL
GROUP BY feature.value
ORDER BY UsageCount DESC, AvgR2Score DESC;
```

## Benefits

### 1. Feature Analysis
- Compare which features were used across different training runs
- Identify which feature sets produce better R² scores
- Track feature engineering evolution over time

### 2. Reproducibility
- Know exactly which features were used for each model
- Recreate predictions with the same feature set
- Debug model differences by comparing feature lists

### 3. Model Selection
- Filter models by feature count
- Find models trained with specific features
- Compare minimal vs comprehensive feature sets

### 4. Documentation
- Automatic record of feature engineering decisions
- Audit trail for model development
- Easy to explain model inputs to stakeholders

## Testing Checklist

- [ ] Run SQL migration script
- [ ] Verify FeaturesUsed column exists in database
- [ ] Verify FeatureCount computed column works
- [ ] Update Python training scripts to return feature_names
- [ ] Train a model and verify FeaturesUsed is populated
- [ ] Test GetFeatureNamesAsync() method
- [ ] Test GetActiveModelFeatureNamesAsync() method
- [ ] Query feature lists using SQL OPENJSON
- [ ] Verify NULL handling for old training records
- [ ] Test feature count filtering

## Rollback

If needed, rollback the migration:

```sql
-- Remove the index
DROP INDEX IF EXISTS IX_ModelTrainingHistory_FeaturesUsed 
ON ModelTrainingHistory;

-- Remove the computed column
ALTER TABLE ModelTrainingHistory 
DROP COLUMN IF EXISTS FeatureCount;

-- Remove the FeaturesUsed column
ALTER TABLE ModelTrainingHistory 
DROP COLUMN IF EXISTS FeaturesUsed;
```

## Notes

- Old training records will have NULL in FeaturesUsed column (backwards compatible)
- Feature names are stored in the order they appear in the training data
- JSON format allows for future expansion (e.g., adding feature importance scores)
- Computed column provides O(1) access to feature count

## Python Training Script Locations

Files that need to be updated to return `feature_names`:

1. **`Quantra\python\train_from_database.py`** (main training script)
   - Add `feature_names` to result dictionary
   - Extract from `X_train.columns` after feature engineering

2. **`Quantra\python\stock_predictor.py`** (if used for training)
   - Return feature names in model metadata

3. **`Quantra\python\tft_integration.py`** (TFT-specific training)
   - Return feature names including static and temporal features

## Future Enhancements

1. **Feature Importance Integration**
   ```json
   {
       "feature_names": ["close", "volume", "rsi_14"],
       "feature_importance": [0.45, 0.30, 0.25]
   }
   ```

2. **Feature Categories**
   ```json
   {
       "feature_names": ["close", "volume"],
       "feature_categories": {
           "price": ["close", "open", "high", "low"],
           "volume": ["volume", "volume_sma_20"],
           "technical": ["rsi_14", "macd"]
       }
   }
   ```

3. **Feature Engineering Pipeline Metadata**
   ```json
   {
       "feature_names": [...],
       "feature_engineering": {
           "type": "balanced",
           "raw_features": 5,
           "engineered_features": 45
       }
   }
   ```
