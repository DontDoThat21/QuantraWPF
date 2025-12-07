# Quantra Python Environment Setup

This directory contains the Python scripts and machine learning models used by Quantra for stock prediction and analysis.

## Quick Start

### Windows
```batch
cd Quantra\python
setup_environment.bat
```

### Linux/Mac
```bash
cd Quantra/python
chmod +x setup_environment.sh
./setup_environment.sh
```

## Manual Installation

### Prerequisites
- Python 3.8 or later
- pip (Python package installer)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python check_dependencies.py
```

## Dependency Issues

If you encounter "Using basic feature creation" message, it means the advanced feature engineering module is not being used. This can happen due to missing dependencies.

### Common Issues and Solutions

1. **Missing scipy**
   ```bash
   pip install scipy>=1.7.0
   ```

2. **Incompatible numpy version**
   ```bash
   pip install "numpy>=1.21.0,<2.0.0"
   ```

3. **Missing scikit-learn**
   ```bash
   pip install scikit-learn>=1.0.0
   ```

4. **Module import errors**
   Run the dependency checker to see which modules are missing:
   ```bash
   python check_dependencies.py
   ```

## Project Structure

```
Quantra/python/
??? requirements.txt              # Python dependencies
??? setup_environment.bat         # Windows setup script
??? setup_environment.sh          # Linux/Mac setup script
??? check_dependencies.py         # Dependency verification tool
??? stock_predictor.py           # Main prediction script
??? feature_engineering.py       # Advanced feature engineering
??? hyperparameter_optimization.py  # Model optimization
??? tft_integration.py           # Temporal Fusion Transformer
??? models/                      # Saved models directory
    ??? stock_rf_model.pkl       # Random Forest model
    ??? stock_scaler.pkl         # Feature scaler
    ??? stock_pytorch_model.pt   # PyTorch model
    ??? stock_tensorflow_model/  # TensorFlow model
    ??? tft_model.pt             # TFT model
    ??? feature_pipeline.pkl     # Feature engineering pipeline
```

## Core Dependencies

### Required (Core Functionality)
- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **scipy** - Scientific computing (required by scikit-learn)
- **joblib** - Model serialization
- **matplotlib** - Plotting and visualization

### Optional (Enhanced Features)

#### Machine Learning Frameworks
- **torch** (PyTorch) - Deep learning (LSTM, GRU, Transformer)
- **tensorflow** - Deep learning alternative

#### Optimization
- **optuna** - Hyperparameter optimization
- **plotly** - Interactive visualizations

#### Time Series
- **hmmlearn** - Hidden Markov Models
- **ta-lib** - Technical analysis indicators (optional)

#### Reinforcement Learning
- **gym** - RL environment
- **stable-baselines3** - RL algorithms

#### GPU Acceleration (Optional)
- **cudf** - GPU DataFrames (NVIDIA CUDA required)
- **cuml** - GPU Machine Learning (NVIDIA CUDA required)
- **cupy** - GPU NumPy (NVIDIA CUDA required)

## Feature Engineering

The `feature_engineering.py` module provides advanced feature generation:

### Feature Types
- **minimal** - Basic price and volume features (~20 features)
- **balanced** - Standard technical indicators (~50 features)
- **full** - Comprehensive feature set (~100+ features)

### Generated Features
1. **Basic Features**
   - Returns, log returns
   - Price differences (close-to-open, high-to-low)
   - Daily range

2. **Trend Indicators**
   - Simple Moving Averages (SMA)
   - Exponential Moving Averages (EMA)
   - MACD (Moving Average Convergence Divergence)
   - MA ratios

3. **Volatility Indicators**
   - Historical volatility
   - Bollinger Bands (upper, lower, width, %B)
   - Average True Range (ATR)
   - Normalized ATR

4. **Volume Features**
   - Volume moving averages
   - Volume ratios
   - On-Balance Volume (OBV)
   - Price-Volume trend

5. **Momentum Indicators**
   - Rate of Change (ROC)
   - Relative Strength Index (RSI)
   - Stochastic Oscillator (%K, %D)

## Model Types

### 1. Random Forest (Default)
- Baseline model, no special dependencies
- Fast training and prediction
- Good interpretability

### 2. PyTorch Models
Requires: `torch`
- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Units
- **Transformer** - Self-attention mechanisms

### 3. TensorFlow Models
Requires: `tensorflow`
- **LSTM** - Keras-based LSTM
- **GRU** - Keras-based GRU
- **Transformer** - Multi-head attention

### 4. Temporal Fusion Transformer (TFT)
Requires: `torch`
- State-of-the-art time series forecasting
- Multi-horizon predictions
- Uncertainty quantification
- Feature importance analysis

## Usage from C#

The C# application calls Python scripts through process execution:

```csharp
// Example: Make a prediction
var pythonService = new PythonInteropService();
var result = await pythonService.RunStockPredictorAsync(
    features: new Dictionary<string, double> { 
        { "close", 150.0 }, 
        { "volume", 1000000 }, 
        ... 
    },
    modelType: "pytorch",
    architectureType: "lstm",
    useFeatureEngineering: true
);
```

## Troubleshooting

### "Using basic feature creation" message

This indicates the advanced feature engineering is not being used. Check:

1. Run dependency checker:
   ```bash
   python check_dependencies.py
   ```

2. Look for error messages in the logs:
   ```
   Feature Engineering module is not available: <error>
   ```

3. Install missing dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify scipy is installed:
   ```bash
   python -c "import scipy; print(scipy.__version__)"
   ```

### Import errors

If you see `ImportError` or `ModuleNotFoundError`:

1. Ensure you're using the correct Python environment
2. Reinstall dependencies:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

### GPU support

GPU acceleration is optional but can significantly speed up training:

**NVIDIA CUDA Installation:**
1. Install NVIDIA drivers
2. Install CUDA Toolkit (11.0+)
3. Install cuDNN
4. Install GPU packages:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   pip install tensorflow[and-cuda]
   ```

### Version conflicts

If you encounter version conflicts:

1. Create a fresh virtual environment:
   ```bash
   python -m venv quantra_env
   source quantra_env/bin/activate  # Linux/Mac
   # or
   quantra_env\Scripts\activate  # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development

### Testing the Feature Engineering
```python
from feature_engineering import build_default_pipeline
import pandas as pd

# Create sample data
data = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [105, 106, 107],
    'low': [99, 100, 101],
    'close': [103, 104, 105],
    'volume': [1000, 1100, 1200]
})

# Build and test pipeline
pipeline = build_default_pipeline(feature_type='balanced')
features = pipeline.fit_transform(data)

print(f"Generated {features.shape[1]} features")
print(features.head())
```

### Testing Stock Prediction
```python
from stock_predictor import predict_stock

features = {
    'close': 150.0,
    'open': 149.0,
    'high': 151.0,
    'low': 148.5,
    'volume': 1000000,
    'current_price': 150.0
}

result = predict_stock(
    features, 
    model_type='pytorch',
    architecture_type='lstm',
    use_feature_engineering=True
)

print(f"Action: {result['action']}")
print(f"Confidence: {result['confidence']}")
print(f"Target Price: {result['targetPrice']}")
```

## Support

For issues or questions:
1. Check the dependency checker output
2. Review the Python logs in the application
3. Ensure all required packages are installed
4. Verify Python version compatibility (3.8+)

## License

Part of the Quantra trading platform.
