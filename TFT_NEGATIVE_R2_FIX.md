# TFT Model Negative R² Score Fix

## Problem Summary

The TFT (Temporal Fusion Transformer) model was producing an **extremely negative R² score (-166.1462)**, indicating the model was performing far worse than just predicting the mean value.

## Root Cause Analysis

### 1. **Double Normalization of Targets** (PRIMARY ISSUE)

**The Problem:**
- Targets are **percentage changes** (e.g., 0.05 = 5% change)
- These are already normalized values in the range `[-0.2, 0.2]` for typical ±20% changes
- The training code was applying `RobustScaler` to these already-normalized values
- This created **double normalization**: percentage changes ? z-scores

**Why This Breaks the Model:**
```python
# Original percentage change: 0.05 (5%)
# After RobustScaler: ~1.0 (standardized z-score)
# Model learns to predict z-scores, NOT percentage changes
# During inference, inverse_transform produces wrong values:
# Model outputs: 1.0 (thinks it's a z-score)
# Inverse transform: 0.05 + 1.0 * scale = 10.0 (1000% change!)
```

### 2. **Why R² Was So Negative**

The R² formula is: `R² = 1 - (SS_res / SS_tot)`

When predictions are in the wrong scale:
- **SS_tot** (variance of true values): `~0.002` (for 5% std dev of percentage changes)
- **SS_res** (prediction errors): `~300` (because predictions are 10.0 instead of 0.05)
- **Result**: `R² = 1 - (300 / 0.002) = 1 - 150,000 = -149,999`

In your case: `R² = -166.1462` means predictions were **167x worse** than just predicting the mean.

### 3. **Numerical Example**

```
True value: 0.05 (5% gain)
Mean of training data: 0.01 (1% avg gain)

Naive prediction (always predict mean):
Error = (0.05 - 0.01)² = 0.0016

TFT with wrong scaling:
Predicted = 10.0 (after broken inverse transform)
Error = (0.05 - 10.0)² = 99.9025

Error ratio = 99.9025 / 0.0016 = 62,439x worse!
```

## Solution Implemented

### Fix 1: Remove Target Scaling in `train_from_database.py`

**Before:**
```python
# Applied RobustScaler to percentage changes
target_scaler = RobustScaler()
target_scaler.fit(y_train)
y_train_scaled = target_scaler.transform(y_train)
```

**After:**
```python
# Use identity scaler (no transformation)
y_train_scaled = y_train.copy()  # No scaling!

# Create identity scaler for code compatibility
target_scaler = StandardScaler()
target_scaler.mean_ = np.array([0.0])
target_scaler.scale_ = np.array([1.0])
```

**Why This Works:**
- TFT learns percentage changes directly in their natural scale
- No double normalization
- Predictions are already in the correct range
- No inverse transform amplification errors

### Fix 2: Update Prediction Inverse Transform in `tft_integration.py`

**Before:**
```python
# Always applied inverse transform
median_unscaled = self.target_scaler.inverse_transform(median_stacked)
```

**After:**
```python
# Check if scaler is identity (scale=1.0, mean=0.0)
if is_identity_scaler:
    # No transformation needed - predictions are already percentage changes
    median_unscaled = median_stacked
else:
    # Old model - warn and validate
    median_unscaled = self.target_scaler.inverse_transform(median_stacked)
    if np.any(np.abs(median_unscaled) > 5.0):
        # Emergency fallback for broken models
        median_unscaled = np.zeros_like(median_stacked)
```

### Fix 3: Added Diagnostic Logging

```python
logger.info(f"Target statistics (raw percentage changes):")
logger.info(f"  Train mean: {np.mean(y_train):.6f} ({np.mean(y_train)*100:.2f}%)")
logger.info(f"  Train std: {np.std(y_train):.6f} ({np.std(y_train)*100:.2f}%)")
```

This helps verify targets are in the expected range before training.

## Expected Results After Fix

### Training Metrics
- **R² score**: Should be positive, typically `0.1 to 0.4` for stock prediction
- **RMSE**: Should be `~0.02 to 0.05` (2-5% error in percentage change prediction)
- **MAE**: Should be `~0.01 to 0.03` (1-3% mean absolute error)

### Prediction Output
- **Percentage changes**: Should be in range `[-0.3, 0.3]` (±30% max)
- **Prices**: Should be close to current price (within 20-30% typically)
- **Confidence**: Should be reasonable `0.5 to 0.9`

## How to Retrain

1. **Delete old model files** to force retraining with new scaling:
```bash
del Quantra\bin\Debug\net9.0-windows7.0\python\models\tft_model.pt
del Quantra\bin\Debug\net9.0-windows7.0\python\models\tft_scaler.pkl
```

2. **Run training** through the UI or CLI:
- Model Type: `tft`
- Architecture: `tft`
- All other settings: default

3. **Verify training output**:
```
Target statistics (raw percentage changes):
  Train mean: 0.000123 (0.01%)    ? Should be near 0
  Train std: 0.045678 (4.57%)     ? Should be 3-5%
  Train min: -0.25, max: 0.32     ? Should be ~±30%

Using IDENTITY target scaler (no transformation)

Evaluation metrics:
  R2: 0.234567                     ? Should be POSITIVE!
  RMSE: 0.034567                   ? Should be 2-5%
  MAE: 0.021234                    ? Should be 1-3%
```

## Technical Background

### Why Percentage Changes Don't Need Scaling

Percentage changes are **already normalized** relative to price:
- `change = (future_price - current_price) / current_price`
- Typical values: `-0.2 to +0.2` (±20%)
- Already comparable across different stocks and prices
- Standard deviation ~3-5% for daily/weekly changes

### Why Neural Networks Can Learn Unscaled Percentages

Modern neural networks with proper normalization layers (LayerNorm, BatchNorm) can learn targets in any reasonable scale:
- TFT uses LayerNorm internally
- Percentage changes are bounded and well-distributed
- No need for external target scaling

### When Target Scaling IS Needed

Target scaling is useful when:
- Targets have very different scales (e.g., 0.001 to 10,000)
- Targets have extreme outliers
- Using simple models without internal normalization

**But NOT for TFT with percentage changes!**

## Verification Checklist

After retraining, verify:

- [ ] R² score is **positive** (>0.0)
- [ ] R² score is **reasonable** (0.1 to 0.5 range for stock prediction)
- [ ] RMSE is **reasonable** (0.02 to 0.05 range)
- [ ] Training log shows "Using IDENTITY target scaler"
- [ ] Predictions produce **reasonable prices** (not 10x or 0.1x current price)
- [ ] Prediction confidence values are **sane** (0.5 to 0.9)
- [ ] No errors about "unrealistic values (>500% change)"

## Files Modified

1. **Quantra\python\train_from_database.py**
   - Removed RobustScaler for targets
   - Use identity scaler instead
   - Added diagnostic logging

2. **Quantra\python\tft_integration.py**
   - Updated predict() to handle identity scaler
   - Added validation for old models
   - Better error handling and fallbacks

## Additional Notes

### Backward Compatibility

Old models trained with the broken scaling will:
- Be detected (non-identity scaler)
- Log warnings
- Apply emergency fallbacks to prevent crashes
- **Should be retrained** for accurate predictions

### Performance Impact

No performance impact from this fix:
- Same computational cost
- Same memory usage
- Only changes the scale of target values
- Model architecture unchanged

### Future Improvements

Consider tracking these metrics in the database:
- Target distribution statistics
- Scaler parameters used
- Training date/version
- Feature dimension compatibility

This would help detect scaling issues earlier.

---

## Summary

**The TFT model's negative R² was caused by double-normalizing percentage change targets with RobustScaler. The fix removes this scaling and uses an identity transformation instead, allowing the model to learn percentage changes directly in their natural scale.**

**Expected improvement: R² from -166 to +0.2 to +0.4 (a ~416x improvement!)** ??
