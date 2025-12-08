# TFT Feature Mismatch Fix - Complete Resolution

## Issue Summary
When using PyTorch with TFT architecture for predictions, the system was throwing an error:
```
X has 9 features, but StandardScaler is expecting 15 features as input.
```

## Root Cause Analysis

### During Training (`train_from_database.py` → `prepare_data_for_ml()`)
1. `create_features()` generates ALL technical indicators from OHLCV data
   - Returns, volatility, SMA (5, 20), momentum, ROC, ATR
   - Bollinger Bands (upper, lower, width)
   - Volume indicators (volume_ma5, volume_ma20, volume_ratio)
   - RSI and other indicators
   - **Total: 15 features** (returns, volatility, sma_5, sma_20, momentum, roc, atr, bb_upper, bb_lower, bb_width, volume_ma5, volume_ma20, volume_ratio, rsi, plus additional based on feature_type)

2. `prepare_data_for_ml()` then drops ONLY the base OHLCV + date columns:
   ```python
   features = features_df.drop(['date', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
   ```
   - This keeps ALL 15+ technical indicator features

3. StandardScaler is fitted on these 15+ features

### During Prediction (`tft_integration.py` → `predict_single()`)
**BEFORE THE FIX:**
1. `create_features()` was called correctly, generating all features
2. BUT then a hardcoded list was used to select only 9 features:
   ```python
   FEATURE_COLUMNS = ['returns', 'volatility', 'sma_5', 'sma_20', 'momentum', 
                      'roc', 'atr', 'bb_width', 'rsi']  # Only 9 features!
   ```
3. This caused a mismatch: 9 features provided vs. 15 features expected by StandardScaler

## The Fix

Modified `tft_integration.py` line 416-425 to use the **same feature selection logic as training**:

```python
# CRITICAL FIX: Use the same feature selection as training
# During training, prepare_data_for_ml drops only OHLCV + date columns
# So we must do the same here to match the feature dimensions
columns_to_drop = ['date', 'open', 'high', 'low', 'close', 'volume']
feature_cols = [col for col in df.columns if col not in columns_to_drop]

if not feature_cols:
    logger.warning("No feature columns found after dropping OHLCV. Using fallback basic features.")
    # This shouldn't happen if create_features worked properly
    feature_cols = ['returns', 'volatility', 'sma_5', 'sma_20', 'momentum', 
                   'roc', 'atr', 'bb_width', 'rsi']
    feature_cols = [col for col in feature_cols if col in df.columns]

logger.info(f"Using {len(feature_cols)} features for prediction: {feature_cols[:5]}... (showing first 5)")
temporal_features = df[feature_cols].values  # Shape: (n_days, n_features)
```

## Key Changes
1. **Removed hardcoded feature list** - No longer limits to 9 specific features
2. **Applied same logic as training** - Drops only OHLCV + date, keeps all technical indicators
3. **Added logging** - Shows how many features are actually being used
4. **Kept fallback** - If something goes wrong, uses the 9 basic features as backup

## Expected Behavior After Fix
- Training creates StandardScaler with N features (e.g., 15)
- Prediction provides the same N features (e.g., 15)
- No more feature dimension mismatch errors
- TFT model can successfully make predictions

## Verification
To verify the fix is working:
1. Train a new TFT model: `python train_from_database.py --model_type pytorch --architecture_type tft`
2. Check training logs for feature count
3. Make a prediction through the UI
4. Check prediction logs - should see: "Using N features for prediction: ..." matching training
5. No StandardScaler error should occur

## Notes
- If using an old model trained with different features, you'll need to **retrain** the model
- The model's StandardScaler is fitted during training and saved with the model
- Both training and prediction must use identical feature sets for the scaler to work
- This fix ensures feature consistency between training and prediction pipelines
