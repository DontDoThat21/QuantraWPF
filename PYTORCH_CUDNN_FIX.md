# PyTorch cuDNN RNN Backward Error Fix

## Problem
The application was throwing an exception during prediction:
```
System.Exception: Python prediction error: cudnn RNN backward can only be called in training mode
```

## Root Cause
The PyTorch LSTM/GRU models were attempting to compute gradients during inference (prediction), which is not allowed when using cuDNN-accelerated RNN layers. This occurs when:
1. The model is not explicitly set to evaluation mode (`model.eval()`)
2. Tensors are created with `requires_grad=True` during inference
3. The `torch.no_grad()` context manager is not used to disable gradient computation

## Solution
Updated the `PyTorchStockPredictor.predict()` method in `Quantra/python/stock_predictor.py`:

### Changes Made:

1. **Added explicit `requires_grad=False` when creating tensors**:
   ```python
   X_torch = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=False).to(self.device)
   ```

2. **Wrapped prediction code with `torch.no_grad()` context**:
   ```python
   with torch.no_grad():
       predictions = self.model(X_torch).cpu().numpy()
   ```

3. **Updated `feature_importance()` method** to properly handle gradient computation:
   - Added `torch.enable_grad()` context for intentional gradient computation
   - Added safety check for zero gradient sum to prevent division by zero

### Why This Works:
- `requires_grad=False` explicitly tells PyTorch not to track operations for gradient computation
- `torch.no_grad()` disables autograd engine entirely during the forward pass
- `model.eval()` sets layers like Dropout and BatchNorm to inference mode
- Together, these prevent cuDNN from attempting backward pass computations during inference

## Files Modified:
- `Quantra/python/stock_predictor.py`:
  - Line ~605-645: `PyTorchStockPredictor.predict()` method
  - Line ~697-730: `PyTorchStockPredictor.feature_importance()` method

## Testing:
To test the fix:
1. Stop the current debugging session
2. Rebuild the application
3. Run a prediction using PyTorch LSTM/GRU/Transformer models
4. The error should no longer occur

## Additional Notes:
- This fix applies to all PyTorch-based models (LSTM, GRU, Transformer)
- The fix is also applicable to TFT (Temporal Fusion Transformer) models if they use PyTorch
- No changes needed for TensorFlow or Random Forest models as they don't have this issue
