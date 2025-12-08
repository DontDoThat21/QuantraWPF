# R² Score Improvement Guide for TFT Model

## Current Situation

**Database R² Value**: `-1.36089898683522E-07` (essentially `0.00`)  
**Training Details**: 106 symbols, 407,616 training samples, 101,904 test samples  
**Training Time**: 8683 seconds (~2.4 hours)

## Problem Analysis

An R² score of ~0.00 means the model has **zero predictive power**. It performs no better than simply predicting the average price change for all stocks.

## Root Causes

### 1. **Insufficient Training Epochs**
- **Current**: 50 epochs
- **Problem**: TFT is a complex architecture that needs more iterations to converge
- **Fix**: Use 100-200 epochs with early stopping

### 2. **Learning Rate Too High**
- **Current**: 0.001
- **Problem**: May cause unstable training and prevent convergence
- **Fix**: Reduce to 0.0001-0.0005

### 3. **Feature Engineering Issues**
- **Current**: Using `balanced` feature type
- **Problem**: May not capture enough market dynamics
- **Fix**: Try `comprehensive` (full) features

### 4. **Target Scaling Masking Issues**
- **Current**: RobustScaler applied to targets
- **Problem**: Scaling can hide poor model performance during training
- **Fix**: Monitor unscaled metrics during training

### 5. **Batch Size**
- **Current**: 64 (optimized for TFT)
- **Status**: This is likely OK
  
### 6. **Model Complexity vs Data**
- **Current**: 106 symbols, ~500K samples
- **Problem**: TFT has many parameters; may need more diverse data
- **Fix**: Either increase data or reduce model complexity

## Recommended Solutions

### Solution 1: **Use R² Score Fix Configuration**

The codebase already has a `CreateR2ScoreFix()` configuration designed for this exact problem:

```csharp
var config = TrainingConfiguration.CreateR2ScoreFix();
// This sets:
// - Transformer architecture (proven to work)
// - 100 epochs
// - Batch size: 64
// - Learning Rate: 0.001
// - Hidden Dim: 256
// - Dropout: 0.2
// - 'balanced' features
```

**Steps**:
1. In PredictionAnalysis, click "? Configure"
2. Select "R² Score Fix" from the preset dropdown
3. Click "? OK"
4. Click "?? Train Model"

### Solution 2: **Manually Optimize TFT Hyperparameters**

1. **Lower Learning Rate**: 
   - Change from `0.001` ? `0.0005` or `0.0001`

2. **Increase Epochs**:
   - Change from `50` ? `100` or `150`

3. **Enable Early Stopping**:
   - Ensure `Use Early Stopping` is checked
   - Set patience to 15

4. **Try Full Features**:
   - Change `Feature Type` from `balanced` ? `comprehensive`

### Solution 3: **Switch to Transformer Architecture**

TFT is very complex. If it continues to fail, try the simpler **Transformer** architecture:

1. Click "? Configure"
2. Set "Architecture Type" ? "Transformer"
3. Keep other TFT hyperparameters
4. Train

The Transformer architecture has proven to work well in your system based on previous training runs.

### Solution 4: **Increase Training Data**

1. Remove `Max Symbols` limit (train on ALL cached symbols)
2. Or explicitly select more symbols via "Select Symbols" button
3. More diverse data helps TFT learn generalizable patterns

### Solution 5: **Check Data Quality**

Verify that:
- Symbols have sufficient historical data (60+ days)
- Data is not corrupted or contains NaN values
- Price changes are within reasonable ranges

## Step-by-Step Fix Process

### Quick Fix (5 minutes):

1. Open PredictionAnalysis view
2. Click "? Configure"
3. Select "R² Score Fix" preset
4. Click "? OK"
5. Click "?? Train Model"
6. Wait for training (~30-60 minutes)
7. Check R² score ? should be > 0.3

### Detailed Fix (15 minutes):

1. **Configure TFT Optimally**:
   ```
   Model Type: pytorch
   Architecture: tft
   Epochs: 150
   Batch Size: 64
   Learning Rate: 0.0005
   Dropout: 0.15
   Hidden Dim: 160
   Num Layers: 2
   Num Heads: 4
   Num Attention Layers: 2
   Feature Type: comprehensive
   Use Feature Engineering: ?
   Use Early Stopping: ?
   Early Stopping Patience: 15
   Use LR Scheduler: ?
   ```

2. **Train Model**

3. **Monitor Logs**:
   - Look for decreasing loss values
   - Check if model is converging

4. **Evaluate Results**:
   - R² > 0.3 = Good
   - R² > 0.5 = Very Good
   - R² > 0.7 = Excellent

## Expected Outcomes

| Configuration | Expected R² | Training Time |
|--------------|-------------|---------------|
| **R² Score Fix (Transformer)** | 0.4 - 0.6 | 1-2 hours |
| **TFT Optimized (150 epochs)** | 0.3 - 0.5 | 2-3 hours |
| **TFT + Full Features** | 0.4 - 0.6 | 2-4 hours |

## Monitoring Training Progress

### In Python Logs:
Look for these patterns:

? **Good Signs**:
```
Epoch 10: Train Loss: 0.85, Val Loss: 0.92
Epoch 20: Train Loss: 0.72, Val Loss: 0.79
Epoch 30: Train Loss: 0.65, Val Loss: 0.71
```
*Loss is decreasing - model is learning*

? **Bad Signs**:
```
Epoch 10: Train Loss: 1.23, Val Loss: 1.25
Epoch 20: Train Loss: 1.22, Val Loss: 1.24
Epoch 30: Train Loss: 1.23, Val Loss: 1.26
```
*Loss is flat - model is NOT learning*

### In Database:
- Check `ModelTrainingHistory` table after each run
- Compare R² scores between runs
- Track which hyperparameters work best

## If All Else Fails

### Fallback to Random Forest:
Random Forest is simpler and more robust:

1. Click "? Configure"
2. Set "Model Type" ? "random_forest"
3. Set "Number of Trees" ? 200
4. Set "Max Depth" ? 15
5. Train

Random Forest typically achieves R² of 0.3-0.5 reliably.

## Long-Term Solutions

1. **Hyperparameter Optimization** ? **ALREADY IMPLEMENTED**:
   - Use existing `optimize_tft.py` for automatic hyperparameter tuning
   - Optuna-based Bayesian optimization already available
   - See `Quantra\python\HYPERPARAMETER_OPTIMIZATION_STATUS.md` for details
   - **Usage**: `python optimize_tft.py --trials 50 --search-space default`

2. **Data Augmentation**:
   - Add more technical indicators
   - Include macroeconomic features (VIX, Treasury yields, etc.)
   - Add sector/industry classification features

3. **Ensemble Models**:
   - Combine TFT + Transformer + Random Forest predictions
   - Use voting or weighted averaging

4. **Regular Retraining**:
   - Retrain models weekly with fresh data
   - Market dynamics change; models need updating

## References

- `TrainingConfiguration.cs`: `CreateR2ScoreFix()` method (line ~213)
- `train_from_database.py`: Target scaling implementation (line ~228)
- `TFT_COMPLETION_STATUS.md`: TFT implementation status
- `TRAINING_CONFIGURATION_INTEGRATION_COMPLETE.md`: Configuration system guide

## Support

If R² remains poor after trying these fixes:
1. Check Python logs for errors
2. Verify database contains quality data
3. Consider using simpler models (LSTM, GRU, Transformer)
4. Reach out for help with specific error messages
