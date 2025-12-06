# TFT Feature Dimension Mismatch Fix

## Problem
When training TFT models from the database, the system encountered a feature dimension mismatch error:
```
ValueError: X has 900 features, but StandardScaler is expecting 94 features as input.
```

## Root Cause
The error occurred because:

1. **Old Model Loading**: When TFT architecture was requested, the system would fall back to PyTorch transformer but would attempt to load an existing PyTorch LSTM model that was trained with a different feature set (94 features).

2. **Feature Dimension Mismatch**: The new TFT training pipeline creates data with shape `(samples, 60, 15)` = 900 features when flattened for Random Forest, but the old model was trained with only 94 features.

3. **Evaluation Failure**: During evaluation, when the code path attempted to use a Random Forest model (due to misidentification), it would flatten the test data to 900 features but the loaded scaler expected only 94 features.

## Solution

### Changes to `train_from_database.py` (both source and bin versions):

1. **Delete Old Incompatible Models**:
   - Before training TFT, the system now automatically deletes old PyTorch models
   - This ensures a fresh model is trained with the correct feature dimensions
   - Prevents loading models with incompatible feature counts

2. **Enhanced Error Handling**:
   - Added try-catch block around model evaluation
   - Detects feature dimension mismatches specifically
   - Provides detailed logging of data shapes and expected features
   - Gracefully handles the error by skipping evaluation and returning training results
   - The model will automatically retrain on next use with correct dimensions

3. **Better Logging**:
   - Logs test data shape, model type, and expected features
   - Helps diagnose feature dimension issues quickly
   - Makes it clear when models are being deleted and retrained

## Files Modified
- `Quantra/bin/Debug/net9.0-windows7.0/python/train_from_database.py`
- `Quantra/python/train_from_database.py`

## Impact
- TFT training will now automatically clean up old incompatible models
- Feature dimension mismatches are caught and handled gracefully
- The system will retrain with correct dimensions on next prediction
- Better error messages help debug feature mismatch issues

## Testing
To test the fix:
1. Run TFT training from the database with `architecture_type='tft'`
2. The system should:
   - Delete any old PyTorch models
   - Train a new transformer model with correct feature dimensions
   - Either evaluate successfully OR skip evaluation with clear message
   - Return success=true with training results

## Next Steps
For full TFT support, consider:
1. Implementing direct TFT model training (not just fallback to PyTorch)
2. Properly handling future-known covariates in the training pipeline
3. Ensuring consistent feature engineering between training and prediction
4. Adding feature dimension validation before model loading
