# Python Integration Quick Reference - FeaturesUsed Column

## ?? Goal
Add `feature_names` to the training result JSON so C# can store it in the database.

## ?? Location
**File**: `Quantra/python/train_from_database.py`  
**Line**: ~400-450 (where result dictionary is created)

## ?? Code to Add

### Step 1: Extract Feature Names (after model training)

```python
# After model is trained and evaluated
# After: r2 = r2_score(y_test, y_pred)
# Before: result = { ... }

# Extract feature names from the training data
feature_names = []
if hasattr(X_train, 'columns'):
    # Pandas DataFrame - get column names
    feature_names = list(X_train.columns)
elif hasattr(X_train, 'shape'):
    # NumPy array - generate generic names
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

logger.info(f"Model trained with {len(feature_names)} features")
```

### Step 2: Add to Result Dictionary

```python
# Create the result dictionary
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
    "feature_names": feature_names,  # ? ADD THIS LINE
    "symbol_results": symbol_results,
    "message": f"Training completed successfully"
}
```

## ?? Expected Output

### Before Change
```json
{
  "success": true,
  "model_type": "pytorch",
  "architecture_type": "transformer",
  "training_samples": 5000,
  "feature_names": null  ? Missing or not present
}
```

### After Change
```json
{
  "success": true,
  "model_type": "pytorch",
  "architecture_type": "transformer",
  "training_samples": 5000,
  "feature_names": [
    "close",
    "volume",
    "rsi_14",
    "sma_20",
    "macd",
    "bb_upper_20",
    "..."
  ]
}
```

## ?? Feature Engineering Integration

### If Using Feature Engineering Pipeline

```python
# After feature engineering is applied
# The pipeline transforms the data and creates feature names

if use_feature_engineering:
    # Feature engineering pipeline automatically handles feature names
    X_train_engineered, y_train = prepare_data_for_ml(
        df,
        window_size=60,
        target_days=5,
        use_feature_engineering=True,
        feature_type=feature_type
    )
    
    # X_train_engineered should be a DataFrame with column names
    feature_names = list(X_train_engineered.columns)
    logger.info(f"Feature engineering created {len(feature_names)} features")
else:
    # Basic features
    feature_names = ["close", "volume", "rsi_14", "sma_20", ...]
```

## ? Verification Steps

### 1. Check Python Output
After training, look for this log message:
```
Model trained with 147 features
```

### 2. Check JSON Output File
The temporary JSON result file should contain:
```json
{
  "feature_names": ["close", "volume", ...]
}
```

### 3. Check Database
After training completes, query the database:
```sql
SELECT 
    Id,
    ModelType,
    FeatureCount,
    LEFT(FeaturesUsed, 200) AS FeatureSample
FROM ModelTrainingHistory
ORDER BY TrainingDate DESC;
```

Expected output:
```
Id  | ModelType | FeatureCount | FeatureSample
----+-----------+--------------+------------------------------------------
42  | pytorch   | 147          | ["close","volume","rsi_14","sma_20",...]
```

## ?? Troubleshooting

### Issue: feature_names is empty list []

**Cause**: X_train doesn't have column names (NumPy array)

**Solution**:
```python
# Check the type of X_train
logger.info(f"X_train type: {type(X_train)}")
logger.info(f"X_train shape: {X_train.shape if hasattr(X_train, 'shape') else 'unknown'}")

# If NumPy array, convert to DataFrame first
if isinstance(X_train, np.ndarray):
    # Create DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    X_train = pd.DataFrame(X_train, columns=feature_names)
```

### Issue: Python script crashes when adding feature_names

**Cause**: X_train is None or invalid type

**Solution**:
```python
# Safe feature name extraction
feature_names = []
try:
    if X_train is not None:
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        elif hasattr(X_train, 'shape') and len(X_train.shape) > 1:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    logger.info(f"Extracted {len(feature_names)} feature names")
except Exception as e:
    logger.warning(f"Could not extract feature names: {e}")
    feature_names = []
```

### Issue: C# reports "FeatureNames is null"

**Cause**: Python didn't include feature_names in result, or JSON key doesn't match

**Solution**: Verify JSON property name matches C# property:
- Python: `"feature_names"` (snake_case)
- C#: `[JsonPropertyName("feature_names")]` (must match exactly)

## ?? Complete Example

Here's a complete example of the code section to add:

```python
def train_model_from_database(connection_string, output_file, model_type, architecture_type, config=None):
    # ... existing training code ...
    
    # After model evaluation
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Model R² Score: {r2:.4f}")
    
    # ========================================
    # NEW CODE: Extract feature names
    # ========================================
    feature_names = []
    try:
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
            logger.info(f"Extracted {len(feature_names)} feature names from DataFrame")
        elif hasattr(X_train, 'shape') and len(X_train.shape) > 1:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            logger.info(f"Generated {len(feature_names)} generic feature names")
        else:
            logger.warning("Could not determine feature names from X_train")
    except Exception as e:
        logger.error(f"Error extracting feature names: {e}")
    
    # Log first few feature names for debugging
    if feature_names:
        logger.info(f"First 10 features: {', '.join(feature_names[:10])}")
    # ========================================
    
    # Create result dictionary
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
        "feature_names": feature_names,  # ? NEW: Include feature names
        "symbol_results": symbol_results,
        "message": f"Successfully trained {model_type} model"
    }
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Training complete. Results saved to {output_file}")
    return result
```

## ?? Success Criteria

After implementing this change, you should see:

1. ? Python logs show: `"Extracted 147 feature names from DataFrame"`
2. ? JSON output contains: `"feature_names": ["close", "volume", ...]`
3. ? Database FeaturesUsed column populated with JSON array
4. ? Database FeatureCount column shows correct count (e.g., 147)
5. ? C# GetFeatureNamesAsync() returns the feature list

## ?? Related Documentation

- **Implementation Summary**: `FEATURES_USED_IMPLEMENTATION_SUMMARY.md`
- **Migration Guide**: `README_FeaturesUsed_Migration.md`
- **SQL Migration**: `AddFeaturesUsedToModelTrainingHistory.sql`

---

**Quick Start**: Just add `feature_names = list(X_train.columns)` before creating the result dictionary!
