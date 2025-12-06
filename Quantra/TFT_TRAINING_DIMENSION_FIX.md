# TFT Training Dimension Mismatch Fix

## Issue
Training the Temporal Fusion Transformer (TFT) architecture from the database was failing with error:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1920x15 and 60x64)
```

## Root Cause
When training data was prepared in 3D format `(n_samples, seq_len, n_features)`, the `load_or_train_model` function was incorrectly extracting the input dimension:
- **Data shape**: `(507561, 60, 15)` - 507561 samples, 60 timesteps, 15 features per timestep
- **Bug**: Used `X_train.shape[1]` which returned `60` (sequence length)
- **Expected**: Should use `X_train.shape[2]` which returns `15` (actual feature count)

This caused the PyTorch transformer model to initialize with `input_dim=60` when it should be `input_dim=15`, leading to a dimension mismatch during forward pass:
- Input tensor: `(batch_size, seq_len, n_features)` = `(32, 60, 15)` ? flattened to `(1920, 15)`
- Linear layer: Expected `(*, 60)` but got `(1920, 15)`

## Solution
Updated `stock_predictor.py` in the `load_or_train_model` function to correctly extract feature dimensions:

### PyTorch Model (Lines 1320-1343)
```python
# Correctly extract input_dim from 3D or 2D input
if len(X_train.shape) == 3:
    # For 3D input (n_samples, seq_len, n_features), use n_features as input_dim
    input_dim = X_train.shape[2]
else:
    # For 2D input (n_samples, n_features), use n_features as input_dim
    input_dim = X_train.shape[1]
```

### TensorFlow Model (Lines 1345-1368)
Same fix applied for TensorFlow/Keras models.

### RandomForest Model (Lines 1370-1405)
Added logic to flatten 3D input to 2D since RandomForest doesn't support time series data directly:
```python
if len(X_train.shape) == 3:
    n_samples, seq_len, n_features = X_train.shape
    X_train_2d = X_train.reshape(n_samples, seq_len * n_features)
```

## Files Modified
1. `Quantra/python/stock_predictor.py` (source)
2. `Quantra/bin/Debug/net9.0-windows7.0/python/stock_predictor.py` (deployed)

## Testing
After applying the fix, the model should train successfully with:
- Input dimension: 15 (correct number of features)
- Sequence length: 60 (timesteps)
- No dimension mismatch errors

## Next Steps
1. Stop the current debugging session
2. Restart the application
3. Re-run the "Train Model from Database" operation with TFT architecture
4. Verify that training completes successfully

## Related Context
- Data preparation logs showed: "Past features per sample: (507561, 60, 15)"
- This indicates the data pipeline is correctly creating 3D sequences
- The bug was purely in the model initialization phase
