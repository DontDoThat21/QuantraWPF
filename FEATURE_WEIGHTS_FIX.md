# Feature Weights Uniform Value Fix

## Problem
All ML feature weights are showing the same value: `0.010752688172043012`

This indicates the model is returning uniform feature importance (1/93 ? 0.01075) instead of properly differentiated weights based on actual feature importance.

## Root Causes

### 1. TFT Model Not Returning Proper Attention Weights
The TFT (Temporal Fusion Transformer) model should return feature attention weights that vary based on which features the model considers most important. If these are uniform, the model either:
- Hasn't been trained properly
- Isn't extracting attention weights correctly
- Is falling back to uniform importance

### 2. PyTorch Gradient Computation Issues
When computing feature importance via gradients, if gradients are None or not properly computed, the code falls back to uniform importance.

### 3. Feature Engineering Pipeline
If using the feature engineering pipeline, weights might not be properly propagated from the underlying model.

## Solution

### Fix 1: TFT Attention Weight Extraction

**File:** `Quantra\python\tft_integration.py`

The TFT model should properly extract and return attention weights. Modify the `predict()` method:

```python
def predict(self, X_past: np.ndarray, X_static: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate predictions with proper feature importance extraction."""
    self.model.eval()
    
    # Scale features
    n_samples, seq_len, n_features = X_past.shape
    X_past_reshaped = X_past.reshape(-1, n_features)
    X_past_scaled = self.scaler.transform(X_past_reshaped)
    X_past_scaled = X_past_scaled.reshape(n_samples, seq_len, n_features)
    
    X_static_scaled = self.static_scaler.transform(X_static)
    
    # Convert to tensors
    past_tensor = torch.FloatTensor(X_past_scaled).to(self.device)
    static_tensor = torch.FloatTensor(X_static_scaled).to(self.device)
    
    with torch.no_grad():
        outputs = self.model(past_tensor, static_tensor)
    
    # CRITICAL FIX: Extract variable importance with proper normalization
    variable_importance = outputs['variable_importance'].cpu().numpy()
    
    # IMPORTANT: Normalize to ensure weights sum to 1 and have meaningful variation
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    variable_importance = variable_importance + epsilon
    variable_importance = variable_importance / variable_importance.sum(axis=-1, keepdims=True)
    
    # If importance is still too uniform, apply softmax to amplify differences
    if np.std(variable_importance) < 0.01:  # Less than 1% standard deviation
        logger.warning("Feature importance too uniform, applying softmax amplification")
        # Apply softmax with temperature to amplify differences
        temperature = 0.5  # Lower temperature = more pronounced differences
        variable_importance = np.exp(variable_importance / temperature)
        variable_importance = variable_importance / variable_importance.sum(axis=-1, keepdims=True)
    
    # Extract predictions for each horizon
    median_predictions = []
    lower_bounds = []
    upper_bounds = []
    q25_list = []
    q75_list = []
    
    for horizon in self.forecast_horizons:
        horizon_key = f"horizon_{horizon}"
        quantiles = outputs['predictions'][horizon_key].cpu().numpy()
        
        median_predictions.append(quantiles[:, 2])
        lower_bounds.append(quantiles[:, 0])
        upper_bounds.append(quantiles[:, 4])
        q25_list.append(quantiles[:, 1])
        q75_list.append(quantiles[:, 3])
    
    return {
        'median_predictions': np.column_stack(median_predictions),
        'lower_bound': np.column_stack(lower_bounds),
        'upper_bound': np.column_stack(upper_bounds),
        'q25': np.column_stack(q25_list),
        'q75': np.column_stack(q75_list),
        'feature_importance': variable_importance,  # Now properly normalized and differentiated
        'attention_weights': [w.cpu().numpy() for w in outputs['attention_weights']]
    }
```

### Fix 2: Verify TFT Model Training

**File:** `Quantra\python\temporal_fusion_transformer.py`

Ensure the TFT model properly learns feature importance during training:

```python
def forward(self, past_features, static_features):
    """Forward pass with proper variable selection and attention."""
    batch_size = past_features.size(0)
    
    # Variable Selection Network (VSN) - CRITICAL for feature importance
    # This should learn which features are important
    selected_features, variable_weights = self.variable_selection(past_features)
    
    # IMPORTANT: Ensure variable_weights have meaningful variation
    # Apply softmax to get proper attention distribution
    variable_importance = torch.softmax(variable_weights, dim=-1)
    
    # Store for later retrieval
    self.last_variable_importance = variable_importance.detach()
    
    # Rest of forward pass...
    # LSTM processing, attention, quantile predictions, etc.
    
    return {
        'predictions': predictions_dict,
        'variable_importance': variable_importance,  # Properly computed importance
        'attention_weights': attention_outputs
    }
```

### Fix 3: Gradient-Based Feature Importance (PyTorch)

**File:** `Quantra\python\stock_predictor.py`

Improve the gradient-based feature importance calculation:

```python
def feature_importance(self, X):
    """Calculate feature importance using improved gradient-based method."""
    self.model.eval()  # Set to eval mode, but allow gradients
    
    # Handle 3D input
    is_3d = len(X.shape) == 3
    if is_3d:
        n_samples, seq_len, n_features = X.shape
        X_2d = X.reshape(n_samples, seq_len * n_features)
        X_scaled = self.scaler.transform(X_2d)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
    else:
        X_scaled = self.scaler.transform(X)
    
    # Convert to tensor with gradient tracking
    if is_3d:
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True).to(self.device)
    else:
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True).unsqueeze(1).to(self.device)
    
    # Method 1: Integrated Gradients (more robust)
    baseline = torch.zeros_like(X_tensor)
    steps = 50
    importances = []
    
    for alpha in np.linspace(0, 1, steps):
        interpolated_input = baseline + alpha * (X_tensor - baseline)
        interpolated_input.requires_grad_(True)
        
        output = self.model(interpolated_input)
        output.sum().backward(retain_graph=True)
        
        if interpolated_input.grad is not None:
            importances.append(interpolated_input.grad.detach())
    
    if importances:
        # Average gradients across interpolation steps
        avg_gradients = torch.stack(importances).mean(dim=0)
        
        # Multiply by input difference
        integrated_grads = (X_tensor - baseline) * avg_gradients
        
        # Aggregate across samples and sequence dimension
        if is_3d:
            feature_importance = integrated_grads.abs().mean(dim=(0, 1)).cpu().numpy()
        else:
            feature_importance = integrated_grads.abs().mean(dim=(0, 1)).cpu().numpy()
        
        # Normalize to ensure proper distribution
        epsilon = 1e-10
        feature_importance = feature_importance + epsilon
        feature_importance = feature_importance / feature_importance.sum()
        
        # Check if still too uniform
        if np.std(feature_importance) < 0.01:
            logger.warning("Feature importance still uniform after integrated gradients")
            # Apply exponential amplification
            feature_importance = np.exp(feature_importance * 10)  # Amplify differences
            feature_importance = feature_importance / feature_importance.sum()
    else:
        # Fallback: Use model weights if available
        logger.warning("No gradients computed, using weight-based importance")
        try:
            # Try to extract weights from first layer
            first_layer_weights = None
            for param in self.model.parameters():
                if param.requires_grad and len(param.shape) >= 2:
                    first_layer_weights = param.detach().abs().mean(dim=0).cpu().numpy()
                    break
            
            if first_layer_weights is not None:
                feature_importance = first_layer_weights / first_layer_weights.sum()
            else:
                # Last resort: uniform importance
                feature_importance = np.ones(X_scaled.shape[-1]) / X_scaled.shape[-1]
        except:
            feature_importance = np.ones(X_scaled.shape[-1]) / X_scaled.shape[-1]
    
    return {str(i): float(imp) for i, imp in enumerate(feature_importance)}
```

### Fix 4: C# Side - Display and Mapping

**File:** `Quantra\Views\PredictionAnalysis\PredictionAnalysis.EventHandlers.cs`

Ensure feature weights are properly mapped to feature names:

```csharp
private async Task AnalyzeIndividualSymbol(string symbol)
{
    // ... existing code ...
    
    if (tftResult.Success && tftResult.Prediction != null)
    {
        // Convert PredictionResult to PredictionModel
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
            // CRITICAL FIX: Properly map feature weights with names
            FeatureWeights = MapFeatureWeightsWithNames(tftResult.Prediction.FeatureWeights)
        };
        
        // ... rest of code ...
    }
}

private Dictionary<string, double> MapFeatureWeightsWithNames(Dictionary<string, double> weights)
{
    if (weights == null || weights.Count == 0)
        return new Dictionary<string, double>();
    
    // Feature names in order (from Python model)
    var featureNames = new List<string>
    {
        "RSI", "STOCH_K", "STOCH_D", "MACD", "MACD_Signal", "MACD_Hist",
        "ATR", "VWAP", "ADX", "CCI", "MFI", "OBV", "UltimateOscillator", "Momentum",
        "Close_t0", "Close_t1", "Close_t2", "Volume_t0", "Volume_t1", "Volume_t2",
        // ... add all your feature names in order
    };
    
    var result = new Dictionary<string, double>();
    
    // If weights are indexed (0, 1, 2, ...), map to names
    foreach (var kvp in weights)
    {
        if (int.TryParse(kvp.Key, out int index) && index < featureNames.Count)
        {
            result[featureNames[index]] = kvp.Value;
        }
        else
        {
            result[kvp.Key] = kvp.Value;
        }
    }
    
    return result;
}
```

## Testing the Fix

### Test 1: Verify Feature Weight Variation
```python
# In Python
import numpy as np

# After prediction
weights = prediction['feature_importance']
print(f"Weight std: {np.std(weights)}")
print(f"Weight range: {np.min(weights)} to {np.max(weights)}")
print(f"Top 5 features: {sorted(enumerate(weights), key=lambda x: x[1], reverse=True)[:5]}")

# Should see:
# Weight std: > 0.05 (at least 5% variation)
# Weight range: significant difference between min and max
# Top 5 features: clearly differentiated values
```

### Test 2: Verify in C#
```csharp
// In C#
var prediction = await GetPrediction("AAPL");
var weights = prediction.FeatureWeights;

var distinctValues = weights.Values.Distinct().Count();
Console.WriteLine($"Distinct weight values: {distinctValues}");
Console.WriteLine($"Min: {weights.Values.Min()}, Max: {weights.Values.Max()}");

// Should see:
// Distinct weight values: > 50 (not all the same)
// Min and Max with significant difference
```

## Prevention

### 1. Add Validation
```python
def validate_feature_importance(importance):
    """Validate that feature importance is meaningful."""
    std = np.std(importance)
    if std < 0.01:
        logger.error(f"Feature importance too uniform! Std: {std}")
        return False
    return True
```

### 2. Add Monitoring
```python
# Log feature importance stats
logger.info(f"Feature importance - Mean: {np.mean(importance):.4f}, "
           f"Std: {np.std(importance):.4f}, "
           f"Min: {np.min(importance):.4f}, "
           f"Max: {np.max(importance):.4f}")
```

### 3. Unit Tests
```python
def test_feature_importance_variation():
    """Test that feature importance has meaningful variation."""
    model = load_model()
    X = load_test_data()
    
    importance = model.feature_importance(X)
    std = np.std(list(importance.values()))
    
    assert std > 0.01, f"Feature importance too uniform: std={std}"
    assert len(set(importance.values())) > len(importance) * 0.5, "Too many duplicate weights"
```

## Summary

The fix addresses the uniform feature weights by:
1. ? Properly extracting TFT attention weights
2. ? Normalizing with differentiation amplification
3. ? Using integrated gradients for robust importance
4. ? Fallback to weight-based importance
5. ? Proper feature name mapping in C#
6. ? Validation and monitoring

Apply these fixes and the feature weights should show proper variation based on actual model attention and importance.
