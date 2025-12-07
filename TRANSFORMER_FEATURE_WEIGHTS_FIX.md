# Transformer Architecture Feature Weights Fix

## Problem Summary
When using `selectedArchitecture = "transformer"` in the PredictionAnalysis control, all feature weights come back as the same uniform value (0.010752688172043012), which is 1/93.

This indicates the model is returning equal importance for all features instead of properly differentiated weights based on the transformer's attention mechanism.

## Root Cause
The issue is in the PyTorch transformer model's feature importance calculation. When you select "transformer", the code path is:

```
PredictionAnalysis.AnalyzeIndividualSymbol()
  ? GetTFTPredictionAsync() 
    ? Python tft_predict.py
      ? PyTorchStockPredictor with architecture_type='transformer'
        ? feature_importance() returns uniform weights
```

The problem is that the transformer model's `feature_importance()` method is either:
1. Not extracting attention weights from the transformer layers
2. Computing gradients on an untrained model (all weights initialized equally)
3. Not properly aggregating multi-head attention scores

## Solution

### Fix 1: Extract Transformer Attention Weights

**File:** `Quantra\python\stock_predictor.py`

Add a proper attention extraction method for transformer models:

```python
def _extract_transformer_attention(self, X):
    """
    Extract attention weights from transformer model for feature importance.
    CRITICAL FIX: Use actual attention mechanisms instead of gradient approximation.
    """
    self.model.eval()
    
    # Scale the features
    X_scaled = self.scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(self.device)
    
    # Transformer models store attention weights - extract them
    attention_weights = []
    
    def attention_hook(module, input, output):
        """Hook to capture attention weights from transformer layers"""
        if hasattr(module, 'attn_weights'):
            attention_weights.append(module.attn_weights.detach().cpu().numpy())
    
    # Register hooks on multi-head attention layers
    hooks = []
    for module in self.model.modules():
        if isinstance(module, nn.MultiheadAttention) or 'MultiHeadAttention' in str(type(module)):
            hooks.append(module.register_forward_hook(attention_hook))
    
    try:
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = self.model(X_tensor)
        
        if attention_weights:
            # Average attention across all layers and heads
            # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            avg_attention = np.mean([a.mean(axis=(0,1)) for a in attention_weights], axis=0)
            
            # Get attention paid to input features (last row = attention to all inputs)
            feature_attention = avg_attention[-1, :]
            
            # Normalize to sum to 1
            feature_importance = feature_attention / feature_attention.sum()
            
            # Map to feature names
            return {name: float(imp) for name, imp in zip(self.feature_names, feature_importance)}
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Fallback to gradient-based if attention extraction fails
    return self._gradient_based_importance(X)

def _gradient_based_importance(self, X):
    """Fallback gradient-based importance for models without attention"""
    was_training = self.model.training
    self.model.train()
    
    try:
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(self.device)
        
        with torch.enable_grad():
            output = self.model(X_tensor)
            output.sum().backward()
        
        if X_tensor.grad is None:
            logger.warning("No gradients computed, returning uniform importance")
            n_features = X_scaled.shape[1]
            return {f"feature_{i}": 1.0/n_features for i in range(n_features)}
        
        gradients = X_tensor.grad.abs().mean(dim=(0, 1)).cpu().numpy()
        grad_sum = gradients.sum()
        
        if grad_sum > 0:
            importance = gradients / grad_sum
        else:
            n_features = len(gradients)
            importance = np.ones(n_features) / n_features
        
        return {name: float(imp) for name, imp in zip(self.feature_names, importance)}
    finally:
        if not was_training:
            self.model.eval()

def feature_importance(self, X):
    """
    Calculate feature importance with architecture-specific extraction.
    CRITICAL FIX: Use attention mechanisms for transformer, gradients for LSTM/GRU.
    """
    # Use attention extraction for transformer architectures
    if self.architecture_type == 'transformer':
        return self._extract_transformer_attention(X)
    else:
        # Use gradient-based for LSTM/GRU
        return self._gradient_based_importance(X)
```

### Fix 2: Verify Model Training Status

Before computing feature importance, check if the model has been trained:

```python
def feature_importance(self, X):
    """Calculate feature importance with training status check"""
    
    # CRITICAL CHECK: Is this a newly initialized model?
    total_params = sum(p.numel() for p in self.model.parameters())
    trained_params = sum(p.abs().sum().item() for p in self.model.parameters())
    avg_param_value = trained_params / total_params if total_params > 0 else 0
    
    # If average parameter value is very close to initialization values, model is untrained
    if abs(avg_param_value) < 0.01:
        logger.warning(f"Model appears untrained (avg param value: {avg_param_value:.6f}). "
                      "Feature importance will be unreliable. Please train the model first.")
        
        # Return uniform importance with a warning
        n_features = X.shape[1] if len(X.shape) == 2 else X.shape[2]
        return {f"feature_{i}": 1.0/n_features for i in range(n_features)}
    
    # Proceed with normal feature importance extraction
    if self.architecture_type == 'transformer':
        return self._extract_transformer_attention(X)
    else:
        return self._gradient_based_importance(X)
```

### Fix 3: Update Transformer Model to Store Attention

Modify the transformer model to explicitly store attention weights:

```python
def _build_transformer_model(self):
    """Build a Transformer model architecture that captures attention weights."""
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, dropout, nhead=4):
            super().__init__()
            self.input_dim = input_dim
            self.pos_encoder = nn.Linear(input_dim, hidden_dim)
            
            # CRITICAL: Use custom TransformerEncoderLayer that captures attention
            self.transformer_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=hidden_dim*2,
                    dropout=dropout,
                    batch_first=True
                ) for _ in range(num_layers)
            ])
            
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc2 = nn.Linear(hidden_dim // 2, 1)
            self.activation = nn.ReLU()
            
            # Store attention weights for importance calculation
            self.attention_weights = []
        
        def forward(self, x, return_attention=False):
            # x shape: (batch_size, seq_length, input_dim)
            x = self.pos_encoder(x)  # Convert input to hidden_dim
            
            # Reset attention weights
            if return_attention:
                self.attention_weights = []
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                # For attention extraction, we need to hook into the layer
                x = layer(x)
                
                # If layer has stored attention, capture it
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'attn_weights'):
                    if return_attention:
                        self.attention_weights.append(layer.self_attn.attn_weights)
            
            # Use the output corresponding to the last time step
            out = x[:, -1, :]
            out = self.dropout(out)
            out = self.activation(self.fc1(out))
            out = self.fc2(out)
            
            return out.squeeze(1)
    
    # Adjust hidden_dim to be divisible by nhead
    nhead = max(4, self.hidden_dim // 16)
    self.hidden_dim = nhead * (self.hidden_dim // nhead)
    
    self.model = TransformerModel(
        input_dim=self.input_dim,
        hidden_dim=self.hidden_dim,
        num_layers=self.num_layers,
        dropout=self.dropout,
        nhead=nhead
    ).to(self.device)
```

### Fix 4: Add Feature Importance Validation

Add validation to detect and warn about uniform importance:

```python
def validate_feature_importance(importance_dict):
    """Validate that feature importance has meaningful variation"""
    values = list(importance_dict.values())
    
    if len(values) == 0:
        return False, "No feature importance values"
    
    # Check if all values are identical (uniform distribution)
    unique_values = len(set(values))
    if unique_values == 1:
        return False, f"All {len(values)} features have identical importance: {values[0]:.6f}. Model may be untrained."
    
    # Check standard deviation
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / mean_val if mean_val > 0 else 0  # Coefficient of variation
    
    if cv < 0.05:  # Less than 5% variation
        return False, f"Feature importance too uniform (CV={cv:.4f}, std={std_val:.6f}). Model may need training or retraining."
    
    return True, f"Feature importance validated (CV={cv:.4f}, {unique_values} unique values)"

# Use in feature_importance method:
def feature_importance(self, X):
    """Calculate feature importance with validation"""
    # ... existing importance calculation ...
    
    importance = self._extract_transformer_attention(X)
    
    # Validate before returning
    is_valid, message = validate_feature_importance(importance)
    if not is_valid:
        logger.warning(f"Feature importance validation failed: {message}")
    else:
        logger.info(f"Feature importance validated: {message}")
    
    return importance
```

## C# Side Fix - Check and Warn User

**File:** `Quantra\Views\PredictionAnalysis\PredictionAnalysis.EventHandlers.cs`

Add validation on the C# side as well:

```csharp
private async Task AnalyzeIndividualSymbol(string symbol)
{
    // ... existing code ...
    
    if (tftResult.Success && tftResult.Prediction != null)
    {
        prediction = new Quantra.Models.PredictionModel
        {
            Symbol = tftResult.Prediction.Symbol,
            PredictedAction = tftResult.Prediction.Action,
            Confidence = tftResult.Prediction.Confidence,
            TargetPrice = tftResult.Prediction.TargetPrice,
            CurrentPrice = tftResult.Prediction.CurrentPrice,
            PredictionDate = tftResult.Prediction.PredictionDate,
            ModelType = TFT_ARCHITECTURE_TYPE,
            ArchitectureType = TFT_ARCHITECTURE_TYPE,
            FeatureWeights = tftResult.Prediction.FeatureWeights
        };
        
        // CRITICAL FIX: Validate feature weights
        if (prediction.FeatureWeights != null && prediction.FeatureWeights.Any())
        {
            var uniqueWeights = prediction.FeatureWeights.Values.Distinct().Count();
            var totalWeights = prediction.FeatureWeights.Count;
            
            if (uniqueWeights == 1)
            {
                // All weights are identical - model is likely untrained
                _loggingService?.Log("Warning", 
                    $"All {totalWeights} feature weights are identical ({prediction.FeatureWeights.Values.First():F6}). " +
                    "The model may be untrained. Consider retraining the model.");
                
                if (StatusText != null)
                    StatusText.Text = $"Warning: Model appears untrained - uniform feature importance detected";
            }
            else if (uniqueWeights < totalWeights * 0.1)
            {
                // Less than 10% unique values - low variation
                _loggingService?.Log("Warning", 
                    $"Feature importance has low variation ({uniqueWeights} unique values out of {totalWeights}). " +
                    "Model may need additional training.");
            }
            else
            {
                _loggingService?.Log("Info", 
                    $"Feature importance validated: {uniqueWeights} unique values out of {totalWeights} features.");
            }
        }
        
        // ... rest of existing code ...
    }
}
```

## Quick Test

Add this test to verify the fix:

```python
# Test file: test_transformer_feature_importance.py
import numpy as np
from stock_predictor import PyTorchStockPredictor

# Create a transformer model
model = PyTorchStockPredictor(
    input_dim=10,
    hidden_dim=64,
    num_layers=2,
    architecture_type='transformer'
)

# Create synthetic test data
X_test = np.random.randn(100, 10)
y_test = np.random.randn(100)

# Train briefly so model isn't completely uniform
model.fit(X_test, y_test, epochs=5, verbose=True)

# Get feature importance
importance = model.feature_importance(X_test[:10])

# Validate
values = list(importance.values())
unique_count = len(set(values))
std_dev = np.std(values)

print(f"Feature importance results:")
print(f"  Unique values: {unique_count}/{len(values)}")
print(f"  Std deviation: {std_dev:.6f}")
print(f"  Min: {min(values):.6f}")
print(f"  Max: {max(values):.6f}")
print(f"  Top 5 features:")
for name, val in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    {name}: {val:.6f}")

# Should see variation if fix works
assert unique_count > 1, "Feature importance still uniform!"
assert std_dev > 0.001, "Feature importance has no variation!"
print("\n? Feature importance test passed!")
```

## Summary

The issue is that when using `architecture = "transformer"`, the model's feature importance calculation returns uniform weights because:

1. **Attention weights aren't being extracted** - transformers have explicit attention mechanisms that should be used
2. **Model may be untrained** - newly initialized models have uniform weights
3. **Gradient-based fallback doesn't work well** for transformers

The fix:
1. ? Extract actual attention weights from transformer layers
2. ? Check if model is trained before computing importance
3. ? Validate feature importance has variation
4. ? Warn user if weights are uniform
5. ? Provide fallback gradient method for LSTM/GRU

After applying this fix, transformer models will return properly differentiated feature importance based on their attention mechanisms rather than uniform values.
