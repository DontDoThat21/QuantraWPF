# Prediction Analysis Control: Machine Learning Features and Training

## Introduction

The Prediction Analysis Control integrates sophisticated machine learning capabilities for stock price prediction through Python-based models. This document provides comprehensive details on how features are generated and retrieved for machine learning models, the model training process, and the analysis workflow.

## Table of Contents

1. [Overview](#overview)
2. [Feature Generation Pipeline](#feature-generation-pipeline)
3. [Feature Categories](#feature-categories)
4. [Feature Engineering](#feature-engineering)
5. [Model Architectures](#model-architectures)
6. [Training Configuration](#training-configuration)
7. [Train Model Functionality](#train-model-functionality)
8. [Analyze Functionality](#analyze-functionality)
9. [Feature Importance and Interpretability](#feature-importance-and-interpretability)
10. [Usage Examples](#usage-examples)

---

## Overview

The ML integration consists of two main workflows:

1. **Analyze**: Generates features from current market data and uses a trained model to predict stock price movements
2. **Train Model**: Collects historical data from the database, generates features, and trains ML models

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PredictionAnalysis UI                       │
│                                                                 │
│  ┌────────────────┐              ┌──────────────────┐          │
│  │ Analyze Button │              │ Train Model Btn  │          │
│  └────────┬───────┘              └────────┬─────────┘          │
└───────────┼──────────────────────────────┼────────────────────┘
            │                              │
            ▼                              ▼
   ┌────────────────┐           ┌────────────────────┐
   │  Feature Gen   │           │ Model Training     │
   │  (Real-time)   │           │   Service          │
   └────────┬───────┘           └─────────┬──────────┘
            │                              │
            │                              ▼
            │                   ┌──────────────────────┐
            │                   │ Database Repository  │
            │                   │ (Historical Prices)  │
            │                   └──────────┬───────────┘
            │                              │
            ▼                              ▼
   ┌─────────────────────────────────────────────────┐
   │         Python ML Engine                        │
   │  ┌────────────────┐    ┌──────────────────┐    │
   │  │ stock_predictor│    │ feature_engineer │    │
   │  │     .py        │    │     ing.py       │    │
   │  └────────────────┘    └──────────────────┘    │
   │                                                 │
   │  ┌──────────────┐  ┌──────────────────────┐   │
   │  │ RandomForest │  │ PyTorch/TensorFlow   │   │
   │  │    Model     │  │  (LSTM/GRU/Transform)│   │
   │  └──────────────┘  └──────────────────────┘   │
   └─────────────────────────────────────────────────┘
```


---

## Feature Generation Pipeline

### Phase 1: Data Collection (C# Layer)

The feature generation starts in the `AnalyzeStockWithAllAlgorithms` method in `PredictionAnalysis.Analysis.cs`:

```csharp
private async Task<Quantra.Models.PredictionModel> AnalyzeStockWithAllAlgorithms(string symbol)
{
    Dictionary<string, double> indicators = new Dictionary<string, double>();
    
    // Fetch technical indicators from Alpha Vantage API
    // ... indicator collection code
}
```

#### Technical Indicators Collected

The system collects comprehensive technical indicators via the Alpha Vantage API:

| Indicator Category | Indicators | Purpose |
|-------------------|------------|---------|
| **Momentum** | RSI, Momentum Score, ROC (10, 20-day) | Identify overbought/oversold conditions |
| **Trend** | MACD (line, signal, histogram), ADX, SMA (7, 14, 30, 50-day) | Detect trend direction and strength |
| **Volatility** | ATR, Bollinger Bands (upper, lower, width, %B), CCI | Measure market volatility |
| **Volume** | OBV, MFI, Volume (t-0, t-1, t-2) | Analyze volume patterns |
| **Oscillators** | Stochastic (K, D), Ultimate Oscillator, Williams %R | Multi-timeframe momentum |
| **Price Data** | Open, High, Low, Close (t-0, t-1, t-2) | Raw price information |

#### Fundamental and Static Features

Beyond technical indicators, the system enriches features with:

```csharp
// Static company metadata (for TFT and other advanced models)
indicators["Sector"] = await _alphaVantageService.GetSectorCode(symbol);
indicators["MarketCapCategory"] = await _alphaVantageService.GetMarketCapCategory(symbol);
indicators["Exchange"] = await _alphaVantageService.GetExchangeCode(symbol);
indicators["MarketCapBillions"] = marketCap / 1_000_000_000;
indicators["Beta"] = beta;
```

**Static Features**:
- **Sector Code**: Numeric encoding (Technology=0, Healthcare=1, Financial=2, etc.)
- **Market Cap Category**: Small-cap=0, Mid-cap=1, Large-cap=2, Mega-cap=3
- **Exchange Code**: NYSE=0, NASDAQ=1, AMEX=2, Other=3
- **Market Cap (Billions)**: Raw market capitalization value
- **Beta**: Stock volatility relative to market

#### Sentiment Features

Sentiment analysis adds behavioral context:

```csharp
// Sentiment correlation analysis
var sentimentResult = await AnalyzeSentimentPriceCorrelation(symbol);
indicators["SentimentScore"] = sentimentScore;
indicators["SentimentVolume"] = sentimentVolume;
indicators["SentimentMomentum"] = sentimentMomentum;
```

**Sentiment Metrics**:
- **Twitter Sentiment**: Social media sentiment from FinBERT and VADER analysis
- **News Sentiment**: Financial news sentiment scores
- **Earnings Transcript Sentiment**: Sentiment from earnings call transcripts
- **Analyst Ratings**: Consensus analyst ratings and targets
- **Insider Trading Activity**: Recent insider buying/selling patterns


### Phase 2: Feature Engineering (Python Layer)

Once basic features are collected, the Python layer performs advanced feature engineering via `feature_engineering.py`.

#### FinancialFeatureGenerator

The `FinancialFeatureGenerator` class is a scikit-learn compatible transformer that generates financial-specific features:

```python
from feature_engineering import FinancialFeatureGenerator

generator = FinancialFeatureGenerator(
    include_basic=True,
    include_trend=True,
    include_volatility=True,
    include_volume=True,
    include_momentum=True,
    rolling_windows=[5, 10, 20, 50, 200]
)

# Transform OHLCV data into features
features = generator.fit_transform(df)
```

**Configuration Options**:
- `include_basic`: Price returns, log returns, daily ranges
- `include_trend`: Moving averages (SMA, EMA), MACD, MA ratios
- `include_volatility`: Historical volatility, Bollinger Bands, ATR
- `include_volume`: Volume MAs, volume ratios, OBV, price-volume trend
- `include_momentum`: RSI, Stochastic, momentum indicators
- `rolling_windows`: List of window sizes for rolling calculations

---

## Feature Categories

### 1. Basic Features

Generated from raw OHLCV data:

```python
def _generate_basic_features(self, df):
    result = pd.DataFrame(index=df.index)
    
    # Returns
    result['returns'] = df['close'].pct_change()
    result['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Price relationships
    result['close_to_open'] = df['close'] / df['open'] - 1
    result['high_to_low'] = df['high'] / df['low'] - 1
    result['daily_range'] = (df['high'] - df['low']) / df['close']
    
    return [result]
```

**Generated Features**:
- `returns`: Percentage change in closing price
- `log_returns`: Logarithmic returns (better for normal distribution)
- `close_to_open`: Intraday price change
- `high_to_low`: Daily price range relative to low
- `daily_range`: Normalized daily price range

### 2. Trend Features

Identify and quantify market trends:

```python
def _generate_trend_features(self, df):
    # Moving averages for multiple windows
    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    # Moving average ratios (crossover signals)
    df['ma_ratio_5_20'] = df['sma_5'] / df['sma_20']
    df['ma_ratio_50_200'] = df['sma_50'] / df['sma_200']
    
    # MACD
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
```

**Generated Features**:
- **SMA/EMA**: Simple and Exponential Moving Averages for various periods
- **MA Ratios**: Ratio between fast and slow moving averages (crossover detection)
- **MACD**: Moving Average Convergence Divergence with signal line and histogram

### 3. Volatility Features

Measure market volatility and risk:

**Generated Features**:
- **Historical Volatility**: Rolling standard deviation of returns (annualized)
- **Bollinger Bands**: Upper/lower bands, bandwidth, %B position
- **ATR/NATR**: Average True Range and normalized ATR

### 4. Volume Features

Analyze trading volume patterns:

**Generated Features**:
- **Volume SMAs**: Moving averages of volume
- **Volume Ratios**: Current volume relative to average
- **OBV**: On-Balance Volume and its moving average
- **Price-Volume Trend**: Cumulative price-weighted volume

### 5. Momentum Features

Capture price momentum and oscillator signals:

**Generated Features**:
- **RSI**: Relative Strength Index (14-period)
- **Stochastic**: Stochastic %K and %D oscillators
- **ROC**: Rate of Change for multiple periods
- **CCI**: Commodity Channel Index

---

## Feature Engineering

### Feature Selection and Dimensionality Reduction

The system employs automated feature selection to identify the most predictive features:

```python
from feature_engineering import FeatureSelector

selector = FeatureSelector(
    method='variance_threshold',  # Options: variance_threshold, mutual_info, rfe, lasso
    k_best=50,                     # Number of top features to select
    threshold=0.01                 # Variance threshold
)

# Select best features
X_selected = selector.fit_transform(X_train, y_train)
selected_features = selector.get_selected_features()
```

**Feature Selection Methods**:

1. **Variance Threshold**: Remove low-variance features
2. **Mutual Information**: Select features with highest mutual information with target
3. **Recursive Feature Elimination (RFE)**: Iteratively remove least important features
4. **L1 Regularization (Lasso)**: Use L1 penalty to select features

### Feature Scaling

All features are normalized before model training using StandardScaler, MinMaxScaler, or RobustScaler.

### Complete Feature Pipeline

The `build_default_pipeline` creates a complete feature engineering pipeline:

```python
from feature_engineering import build_default_pipeline

# Create pipeline
pipeline = build_default_pipeline(
    feature_type='balanced',        # Options: minimal, balanced, full
    scaling_method='standard',      # Options: standard, minmax, robust
    dimensionality_reduction=None,  # Options: pca, tsne
    n_components=50
)

# Fit and transform
X_train_features = pipeline.fit_transform(X_train, y_train)
X_test_features = pipeline.transform(X_test)

# Save pipeline for later use
pipeline.save('feature_pipeline.pkl')
```

**Pipeline Stages**:
1. Financial feature generation
2. Feature scaling
3. Feature selection
4. Dimensionality reduction (optional)
5. Polynomial feature generation (optional)

---

## Model Architectures

The system supports multiple ML architectures, each with specific strengths:

### 1. Random Forest (Default)

**Architecture**: Ensemble of decision trees  
**File**: `stock_predictor.py`

**Strengths**:
- High interpretability (feature importance)
- Robust to overfitting
- Handles non-linear relationships
- No feature scaling required
- Fast training and inference

**Use Cases**: General-purpose prediction, feature importance analysis, quick baseline model

### 2. LSTM (Long Short-Term Memory)

**Architecture**: Recurrent Neural Network  
**File**: `stock_predictor.py` - `PyTorchStockPredictor` / `TensorFlowStockPredictor`

**Strengths**:
- Captures temporal dependencies
- Handles sequential data naturally
- Memory cells for long-term patterns
- Good for time-series prediction

**Use Cases**: Sequential price prediction, pattern recognition over time, trend continuation modeling

### 3. GRU (Gated Recurrent Unit)

**Architecture**: Simplified recurrent network  
**File**: `stock_predictor.py` - `PyTorchStockPredictor`

**Strengths**:
- Faster training than LSTM
- Fewer parameters
- Similar performance to LSTM
- Better for smaller datasets

**Use Cases**: Similar to LSTM but with faster training, limited computational resources, shorter sequences

### 4. Transformer

**Architecture**: Self-attention mechanism  
**File**: `stock_predictor.py` - `PyTorchStockPredictor` / `TensorFlowStockPredictor`

**Strengths**:
- Parallel processing (faster training)
- Attention mechanism (interpretability)
- Captures global dependencies
- State-of-the-art for sequence modeling

**Use Cases**: Complex pattern recognition, multi-timeframe analysis, when interpretability is important

### 5. Temporal Fusion Transformer (TFT)

**Architecture**: Advanced transformer with variable selection  
**File**: `tft_integration.py`, `temporal_fusion_transformer.py`

**Key Components**:

1. **Variable Selection Networks (VSN)** - Selects relevant input features
2. **Gated Residual Networks (GRN)** - Non-linear feature processing
3. **Interpretable Multi-Head Attention** - Temporal attention weights
4. **Quantile Regression** - Prediction intervals and uncertainty quantification

**Strengths**:
- Multi-horizon forecasting (predict multiple future points)
- Quantile predictions (uncertainty estimates)
- Feature importance via attention weights
- Handles static and temporal features separately

**Use Cases**: Long-term price forecasting, risk assessment, when interpretability is critical, multi-step ahead predictions

---

## Training Configuration

The training process is configured via the `TrainingConfiguration` class:

### Configuration Parameters

```csharp
public class TrainingConfiguration
{
    // Model Architecture
    public string ModelType { get; set; }           // "random_forest", "pytorch", "tensorflow"
    public string ArchitectureType { get; set; }    // "lstm", "gru", "transformer", "tft"
    
    // Training Parameters
    public int Epochs { get; set; } = 50;           // Number of training epochs
    public int BatchSize { get; set; } = 32;        // Batch size for training
    public double LearningRate { get; set; } = 0.001;  // Learning rate
    
    // Model Architecture Parameters
    public int HiddenDim { get; set; } = 64;        // Hidden layer size
    public int NumLayers { get; set; } = 2;         // Number of layers
    public double Dropout { get; set; } = 0.2;      // Dropout rate
    
    // Feature Engineering
    public string FeatureType { get; set; } = "balanced";  // minimal, balanced, full
    public string ScalingMethod { get; set; } = "standard"; // standard, minmax, robust
    
    // Data Selection
    public int? MaxSymbols { get; set; } = null;    // Limit symbols (null = all)
    public List<string> SelectedSymbols { get; set; }  // Specific symbols to train on
}
```

### Predefined Configurations

- **Quick Training**: Epochs: 20, BatchSize: 64, LR: 0.01 (Fast, lower accuracy)
- **Balanced Training**: Epochs: 50, BatchSize: 32, LR: 0.001 (Good balance)
- **Deep Training**: Epochs: 100, BatchSize: 16, LR: 0.0005 (Slow, higher accuracy)

---

## Train Model Functionality

The "Train Model" button initiates the complete model training workflow.

### Training Workflow

```
User Clicks "Train Model"
         │
         ▼
┌─────────────────────────┐
│  Validate Configuration │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Prepare Training Data  │
│  - Query database       │
│  - Load historical data │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Feature Engineering    │
│  - Generate features    │
│  - Scale features       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Model Training         │
│  - Initialize model     │
│  - Train epochs         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Model Evaluation       │
│  - Calculate MAE/RMSE   │
│  - Calculate R² score   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Save Model & Results   │
└──────────────────────────┘
```

### Training Process

The training service communicates with Python via subprocess:

1. **Configuration**: UI selections packaged into TrainingConfiguration object
2. **Database Query**: Historical price data loaded from HistoricalPrices table
3. **Feature Generation**: Python FinancialFeatureGenerator creates 200+ features
4. **Train/Test Split**: 80/20 split maintaining chronological order
5. **Model Training**: Selected architecture trained for configured epochs
6. **Evaluation**: Performance metrics calculated on test set
7. **Persistence**: Model saved to python/models directory
8. **Logging**: Results logged to database for tracking

---

## Analyze Functionality

The "Analyze" button performs real-time prediction using the trained model.

### Analysis Workflow

```
User Selects Symbol & Clicks "Analyze"
         │
         ▼
┌──────────────────────────┐
│  Collect Current Data    │
│  - Current price (API)   │
│  - Technical indicators  │
│  - Sentiment data        │
└──────────┬───────────────┘
            │
            ▼
┌──────────────────────────┐
│  Invoke Python Predictor │
│  - Pass features as JSON │
│  - Receive prediction    │
└──────────┬───────────────┘
            │
            ▼
┌──────────────────────────┐
│  Process & Display       │
│  - Calculate target price│
│  - Update UI             │
└──────────────────────────┘
```

### Analysis Steps

1. **Feature Collection**: Gather 50+ current indicators via Alpha Vantage API
2. **Model Loading**: Load trained model and scaler from disk
3. **Prediction**: Model predicts next-day return percentage
4. **Post-Processing**: Convert return to target price and action (BUY/SELL/HOLD)
5. **UI Update**: Display prediction with confidence level
6. **Database Logging**: Save prediction for tracking

---

## Feature Importance and Interpretability

### Random Forest Feature Importance

Random Forest provides built-in feature importance based on impurity decrease:

**Typical Important Features**:
1. Recent price trends (Close_t0, Close_t1, SMA ratios)
2. Momentum indicators (RSI, MACD)
3. Volatility measures (BB_Width, ATR)
4. Volume patterns (Volume ratios, OBV)

### TFT Attention Weights

TFT provides attention-based interpretability showing:
- **Temporal Attention**: Which historical time steps are most relevant
- **Feature Attention**: Which input features drive the prediction

---

## Usage Examples

### Example 1: Quick Analysis with Default Model

```
1. Select symbol "AAPL" from dropdown
2. Click "Analyze" button
3. System collects 50+ features automatically
4. Prediction displayed: "AAPL - BUY with 78% confidence. Target: $182.15 (+3.52%)"
```

### Example 2: Training Custom LSTM Model

```
1. Click "Train Model" button
2. Select model type: "PyTorch"
3. Select architecture: "LSTM"
4. Configure: Epochs: 100, Batch Size: 32, Learning Rate: 0.001
5. Select 50 training symbols from S&P 500
6. Click "Start Training"
7. Results: MAE: 0.018, RMSE: 0.025, R²: 0.68, Time: 342s
```

### Example 3: TFT Multi-Horizon Forecasting

```
1. Select architecture: "TFT"
2. Configure: Lookback: 60 days, Horizons: 5, 10, 20, 30 days
3. Click "Analyze"
4. View multi-horizon predictions with confidence bands
5. Examine attention heatmap showing which days mattered most
6. Review feature importance bar chart
```

---

## Performance Metrics

### Evaluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actuals
- **RMSE (Root Mean Squared Error)**: Square root of average squared errors (penalizes large errors)
- **R² Score**: Proportion of variance explained by model (0-1, higher is better)

### Typical Performance Ranges

- **Random Forest**: R² = 0.55-0.65, MAE = 0.020-0.030
- **LSTM**: R² = 0.60-0.70, MAE = 0.018-0.025
- **Transformer**: R² = 0.65-0.75, MAE = 0.015-0.022
- **TFT**: R² = 0.70-0.80, MAE = 0.012-0.020

---

## Best Practices

### For Training

1. **Symbol Selection**: Use diverse set of symbols across sectors
2. **Data Quality**: Ensure sufficient historical data (60+ days minimum)
3. **Feature Selection**: Use 'balanced' for general use, 'full' for deep models
4. **Hyperparameters**: Start with defaults, optimize if needed
5. **Validation**: Monitor both training and test metrics to detect overfitting

### For Analysis

1. **Model Freshness**: Retrain models weekly or monthly
2. **Feature Consistency**: Ensure same features used in training and prediction
3. **Confidence Thresholds**: Act only on high-confidence predictions (>70%)
4. **Multiple Models**: Use ensemble of models for robust predictions
5. **Market Conditions**: Consider market regime (trending vs ranging)

---

## Troubleshooting

### Common Issues

**Issue**: Model predictions are always neutral (HOLD)
- **Cause**: Insufficient feature variation or poorly trained model
- **Solution**: Retrain with more symbols, increase epochs, check feature scaling

**Issue**: Very low R² score (<0.3)
- **Cause**: Overfitting, insufficient data, or wrong architecture
- **Solution**: Reduce model complexity, add more training data, try different architecture

**Issue**: Training fails with "NaN in gradients"
- **Cause**: Learning rate too high or feature scaling issues
- **Solution**: Reduce learning rate, ensure proper feature normalization

**Issue**: Prediction takes too long
- **Cause**: Large model or inefficient feature generation
- **Solution**: Use Random Forest for faster inference, optimize feature pipeline

---

## Conclusion

The Prediction Analysis Control's machine learning integration provides a sophisticated yet accessible framework for stock price prediction. Key strengths include:

1. **Comprehensive Feature Generation**: Automatic collection of 50+ technical indicators, sentiment metrics, and static features
2. **Advanced Feature Engineering**: Automated pipeline with feature selection, scaling, and dimensionality reduction
3. **Multiple Model Architectures**: Support for Random Forest, LSTM, GRU, Transformer, and TFT
4. **Flexible Training Configuration**: Predefined and custom configurations with hyperparameter optimization
5. **Real-time Prediction**: Fast inference using trained models for actionable trading signals
6. **Interpretability**: Feature importance and attention weights for understanding predictions
7. **Complete Workflow**: Seamless integration between C# UI and Python ML engine

This architecture enables both novice users (via default configurations) and advanced users (via customization) to leverage state-of-the-art machine learning for algorithmic trading.

---

## Related Documentation

- [1_Overview_and_Architecture.md](1_Overview_and_Architecture.md) - System architecture
- [2_Technical_Components_and_Data_Flow.md](2_Technical_Components_and_Data_Flow.md) - Data flow diagrams
- [3_Algorithms_and_Analysis_Methodologies.md](3_Algorithms_and_Analysis_Methodologies.md) - Analysis algorithms
- [4_Sentiment_Analysis_Integration.md](4_Sentiment_Analysis_Integration.md) - Sentiment features
- [5_Automation_and_Trading_Features.md](5_Automation_and_Trading_Features.md) - Trading automation
- [6_Configuration_and_Extension_Points.md](6_Configuration_and_Extension_Points.md) - Configuration options
- [7_Performance_Considerations_and_Best_Practices.md](7_Performance_Considerations_and_Best_Practices.md) - Performance optimization
