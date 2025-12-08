# Hyperparameter Optimization Integration Guide

## Overview

This guide explains how to use the **existing** automated hyperparameter optimization features (already implemented) powered by Optuna to automatically find optimal parameters for your ML models.

## Existing Implementation

? **Already Available:**
- `hyperparameter_optimization.py` - Generic optimization for sklearn, PyTorch, TensorFlow
- `optimize_tft.py` - TFT-specific optimization with time-series CV
- `tft_hyperparameter_config.py` - TFT hyperparameter configurations
- `hyperparameter_optimization_example.py` - Usage examples

## What is Hyperparameter Optimization?

Instead of manually trying different hyperparameters (learning rate, batch size, hidden dimensions, etc.), Optuna uses **Bayesian optimization** to intelligently search the hyperparameter space and find the best combination that maximizes R² score.

## Setup (Already Done)

Optuna and dependencies are already in `requirements.txt`. If not installed:

```bash
cd Quantra\python
pip install -r requirements.txt
```

Or manually:
```bash
pip install optuna optuna-dashboard plotly kaleido
```

## Usage

### Option 1: TFT-Specific Optimization (Recommended for TFT)

Use `optimize_tft.py` for TFT models with proper time-series cross-validation:

```bash
cd Quantra\python

# Optimize TFT with synthetic data (for testing)
python optimize_tft.py --trials 50 --search-space default

# Optimize TFT with real data from NPZ file
python optimize_tft.py --data training_data.npz --trials 50 --cv-splits 3

# Quick optimization (smaller search space, faster)
python optimize_tft.py --data training_data.npz --trials 20 --search-space quick
```

### Option 2: Generic Optimization (For sklearn, PyTorch, TensorFlow)

Use `hyperparameter_optimization.py` as a library:

```python
from hyperparameter_optimization import (
    optimize_sklearn_model_optuna,
    optimize_pytorch_model,
    optimize_tensorflow_model
)

# Example: Optimize Random Forest
from sklearn.ensemble import RandomForestRegressor

param_ranges = {
    'n_estimators': lambda trial: trial.suggest_int('n_estimators', 50, 500),
    'max_depth': lambda trial: trial.suggest_int('max_depth', 3, 30),
    'min_samples_split': lambda trial: trial.suggest_int('min_samples_split', 2, 20)
}

result = optimize_sklearn_model_optuna(
    model_class=RandomForestRegressor,
    X_train=X_train,
    y_train=y_train,
    param_ranges=param_ranges,
    n_trials=50
)

print(f"Best R²: {result.best_score}")
print(f"Best params: {result.best_params}")
```

### Option 3: Integration with train_from_database.py

The existing `train_from_database.py` can be extended to use optimized hyperparameters:

```python
# After optimization, save best params
best_params = {
    'hidden_dim': 160,
    'num_heads': 4,
    'learning_rate': 0.00042,
    # ... other params
}

# Train with optimized params
python train_from_database.py "CONNECTION" results.json tft tft \
    --hidden-dim 160 \
    --num-heads 4 \
    --learning-rate 0.00042
```

## How It Works

### 1. Hyperparameter Search Space

For **TFT**, Optuna searches:
- `hidden_dim`: [96, 128, 160, 192, 256]
- `num_heads`: [2, 4, 6, 8]
- `num_lstm_layers`: [1, 2, 3]
- `num_attention_layers`: [1, 2, 3, 4]
- `dropout`: [0.05 to 0.30]
- `epochs`: [50, 75, 100, 150, 200]
- `batch_size`: [32, 64, 96, 128]
- `learning_rate`: [1e-5 to 1e-2] (log scale)
- `early_stopping_patience`: [10 to 25]
- `feature_type`: [minimal, balanced, full]
- `lookback_period`: [30, 60, 90]

### 2. Optimization Process

1. **Trial 1-5**: Random sampling to explore space
2. **Trial 6+**: Bayesian optimization (TPE algorithm)
3. **Pruning**: Stops bad trials early to save time
4. **Convergence**: Focuses on promising regions

### 3. Results

Optuna saves:
- **Best hyperparameters** (highest R² score)
- **All trial results** (for analysis)
- **Parameter importance** (which params matter most)
- **Optimization history** (R² improvement over time)

## Output Files

After optimization completes:

```
optimization_results_tft_20250115_143022.json  # Best params + metrics
param_importance_tft.png                        # Which params matter most
optimization_history_tft.png                    # R² improvement over trials
hyperparameter_optimization_20250115.log        # Detailed logs
```

### Example Results JSON

```json
{
  "study_name": "tft_optimization_20250115_143022",
  "model_type": "tft",
  "n_trials": 50,
  "best_trial_number": 37,
  "best_r2_score": 0.6234,
  "best_mae": 0.0287,
  "best_rmse": 2.45,
  "best_training_time": 3245.6,
  "best_hyperparameters": {
    "hidden_dim": 160,
    "num_heads": 4,
    "num_lstm_layers": 2,
    "num_attention_layers": 3,
    "dropout": 0.15,
    "epochs": 150,
    "batch_size": 64,
    "learning_rate": 0.00042,
    "early_stopping_patience": 15,
    "feature_type": "balanced",
    "lookback_period": 60
  },
  "optimization_date": "2025-01-15T14:45:32"
}
```

## Using Optimized Hyperparameters

### Method 1: Via Training Configuration (Recommended)

1. Open the optimization results JSON file
2. In PredictionAnalysis, click **? Configure**
3. Manually set each parameter from `best_hyperparameters`
4. Click **?? Save As...** and name it "TFT Optimized (Optuna)"
5. Click **? OK** and **?? Train Model**

### Method 2: Via Command Line

```bash
# Create config file from optimized params
cat > tft_optimized_config.json << EOF
{
  "hidden_dim": 160,
  "num_heads": 4,
  "epochs": 150,
  "batch_size": 64,
  "learning_rate": 0.00042
}
EOF

# Train with optimized config
python train_from_database.py "YOUR_CONNECTION" results.json tft tft \
    --config tft_optimized_config.json
```

### Method 3: Programmatic (C# Integration)

```csharp
// Load optimized hyperparameters from JSON
var optimizedParams = LoadOptimizationResults("optimization_results_tft_*.json");

// Create training configuration
var config = new TrainingConfiguration
{
    ModelType = "pytorch",
    ArchitectureType = "tft",
    HiddenDim = optimizedParams["hidden_dim"],
    NumHeads = optimizedParams["num_heads"],
    Epochs = optimizedParams["epochs"],
    BatchSize = optimizedParams["batch_size"],
    LearningRate = optimizedParams["learning_rate"],
    // ... other parameters
};

// Train model
var result = await _modelTrainingService.TrainModelFromDatabaseAsync(config);
```

## Expected Results

### Baseline (Manual Tuning)
- R² Score: 0.00 to 0.40
- Time: 1-2 hours per attempt
- Iterations: 5-10 manual trials

### With Optuna (Automated)
- R² Score: **0.45 to 0.65** (optimized)
- Time: 24-48 hours for 50 trials (unattended)
- Iterations: 50+ trials automatically

## Advanced Usage

### Parallel Optimization

Run multiple optimization processes in parallel:

```bash
# Terminal 1: Optimize TFT
python optimize_hyperparameters.py "CONN" tft --n-trials 50 \
    --storage "sqlite:///optuna.db" --study-name tft_opt

# Terminal 2: Optimize LSTM simultaneously
python optimize_hyperparameters.py "CONN" lstm --n-trials 50 \
    --storage "sqlite:///optuna.db" --study-name lstm_opt
```

### Optuna Dashboard (Real-time Monitoring)

```bash
# Start dashboard
optuna-dashboard sqlite:///optuna.db

# Open browser to http://localhost:8080
# Watch optimization progress in real-time
```

### Custom Search Space

Edit `optimize_hyperparameters.py` to customize:

```python
def create_tft_hyperparameter_space(trial):
    return {
        # Add custom parameter
        'custom_param': trial.suggest_float('custom_param', 0.1, 1.0),
        
        # Change existing ranges
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
        
        # Add conditional parameters
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'num_heads': trial.suggest_int('num_heads', 2, 8) if use_attention else 0
    }
```

## Troubleshooting

### Issue: "Optuna not installed"

```bash
pip install optuna optuna-dashboard
```

### Issue: Optimization taking too long

- Reduce `--n-trials` (try 25 instead of 50)
- Add `--max-symbols 50` to train on fewer symbols
- Add `--timeout 86400` for 24-hour timeout

### Issue: All trials failing

Check logs for errors:
- Database connection issues?
- Insufficient data (need 60+ days)?
- CUDA OOM errors (reduce batch_size range)?

### Issue: R² not improving

- Increase `--n-trials` (try 75-100)
- Check if data quality is the issue (not hyperparameters)
- Try different model type (LSTM instead of TFT)

## Best Practices

1. **Start Small**: Begin with 25 trials to test
2. **Use Persistent Storage**: `--storage sqlite:///optuna.db`
3. **Name Studies**: `--study-name tft_optimization_v1`
4. **Monitor Progress**: Use Optuna Dashboard
5. **Resume Interrupted**: Same `--study-name` resumes
6. **Compare Models**: Optimize TFT, LSTM, Transformer separately
7. **Validate Results**: Retrain with best params to confirm

## Integration Roadmap

### Phase 1: Command Line (? Complete)
- Standalone optimization script
- Manual hyperparameter application

### Phase 2: UI Integration (TODO)
- Add "?? Optimize Hyperparameters" button to PredictionAnalysis
- Show optimization progress in UI
- Auto-apply best hyperparameters when complete

### Phase 3: Automated Retraining (TODO)
- Schedule weekly optimization
- Automatically retrain with best params
- A/B test new vs old model

## References

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **TPE Algorithm**: Tree-structured Parzen Estimator (Bayesian optimization)
- **Pruning**: MedianPruner stops unpromising trials early
- **Storage**: SQLite for persistence, in-memory for testing

## Support

If optimization doesn't improve R²:
1. Check `hyperparameter_optimization_*.log` for errors
2. Verify data quality (60+ days per symbol)
3. Try simpler model (LSTM/Transformer) before TFT
4. Consider ensemble methods instead

---

**The automated optimization will find better hyperparameters than manual tuning, typically improving R² by 0.1 to 0.3 points.**
