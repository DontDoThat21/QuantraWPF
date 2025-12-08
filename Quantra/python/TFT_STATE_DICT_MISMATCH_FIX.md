# TFT State Dict Mismatch Fix

## Problem
When loading a trained TFT model, the following error occurred:

```
Error(s) in loading state_dict for TemporalFusionTransformer:
Unexpected key(s) in state_dict: "future_embedding.weight", "future_embedding.bias"
```

## Root Cause
The `future_embedding` layer was **dynamically created** during the forward pass when future calendar features were provided:

```python
# OLD CODE (INCORRECT)
if not hasattr(self, 'future_embedding'):
    calendar_dim = future_features.size(-1)
    self.future_embedding = nn.Linear(calendar_dim, self.hidden_dim).to(future_features.device)
```

This caused a mismatch:
1. **During training**: Model was trained with future features ? `future_embedding` layer created and saved
2. **During loading**: Model definition didn't include `future_embedding` ? `load_state_dict()` failed

## Solution
Made `future_embedding` a proper part of the model architecture:

### 1. Initialize in `__init__()` (temporal_fusion_transformer.py)
```python
# NEW CODE (CORRECT)
# Add future_embedding as proper model component
self.future_embedding = None  # Will be initialized when needed
self.calendar_dim = None  # Track the calendar dimension used during training
```

### 2. Initialize properly in `forward()` (temporal_fusion_transformer.py)
```python
# NEW CODE (CORRECT)
if self.future_embedding is None:
    # First time seeing future features - create the embedding layer
    self.future_embedding = nn.Linear(calendar_dim, self.hidden_dim).to(future_features.device)
    self.calendar_dim = calendar_dim
    logger.info(f"Initialized future_embedding layer: {calendar_dim} -> {self.hidden_dim}")
```

### 3. Save `calendar_dim` in checkpoint (tft_integration.py)
```python
# NEW CODE (CORRECT)
checkpoint = {
    'model_state_dict': self.model.state_dict(),
    # ... other fields ...
    'calendar_dim': self.model.calendar_dim  # Save calendar dimension
}
```

### 4. Recreate `future_embedding` during load (tft_integration.py)
```python
# NEW CODE (CORRECT)
calendar_dim = checkpoint.get('calendar_dim', None)

# Rebuild model
self.model = TemporalFusionTransformer(...)

# Initialize future_embedding if model was trained with calendar features
if calendar_dim is not None:
    self.model.future_embedding = nn.Linear(calendar_dim, self.hidden_dim).to(self.device)
    self.model.calendar_dim = calendar_dim
    logger.info(f"Initialized future_embedding for loading: {calendar_dim} -> {self.hidden_dim}")

self.model.load_state_dict(checkpoint['model_state_dict'])
```

## Files Modified
1. `Quantra\python\temporal_fusion_transformer.py`:
   - Added `future_embedding` and `calendar_dim` initialization in `__init__()`
   - Updated `forward()` to properly initialize `future_embedding`

2. `Quantra\python\tft_integration.py`:
   - Updated `save()` to include `calendar_dim` in checkpoint
   - Updated `load()` to recreate `future_embedding` before loading state dict

## Testing
After applying this fix:
1. **Retrain the model** using `train_from_database.py` (the old model won't work with the new code)
2. The model should save with `calendar_dim` in the checkpoint
3. Loading should work without state_dict mismatch errors

## Important Note
**You MUST retrain the TFT model after this fix.** Old model files won't be compatible because:
- Old checkpoints don't have `calendar_dim` saved
- The model architecture has been updated

To retrain:
```bash
cd Quantra\python
python train_from_database.py --model_type tft --symbols AAPL MSFT GOOGL --epochs 100
```

## Related Issue
This fix resolves the error:
```
Exception: Failed to load TFT model. Train the model first using train_from_database.py
```
