# Automated Feature Engineering Pipeline for Quantra

This document explains how to use the automated feature engineering pipeline for machine learning models in the Quantra project.

## Overview

The feature engineering pipeline automates the process of creating, selecting, and transforming features for financial machine learning models. It provides the following functionality:

- **Feature Generation**: Create technical indicators and financial features from OHLCV data
- **Feature Selection**: Select the most important features using various methods
- **Feature Transformation**: Transform features using dimensionality reduction techniques
- **Interaction Term Generation**: Create polynomial and custom interaction features
- **Feature Scaling**: Scale features for optimal model performance

## Getting Started

### Requirements

Make sure you have the required dependencies installed:

```bash
pip install -r python/requirements.txt
```

### Basic Usage

The feature engineering pipeline is integrated with the `stock_predictor.py` module and can be enabled by setting the `UseFeatureEngineering` flag in the input JSON:

```json
{
  "Features": {
    "open": 150.5,
    "high": 152.3,
    "low": 149.8,
    "close": 151.2,
    "volume": 1250000
  },
  "ModelType": "auto",
  "UseFeatureEngineering": true,
  "FeatureType": "balanced"
}
```

### Feature Types

You can choose from three feature complexity levels:

- `minimal`: Basic set of features (faster processing)
- `balanced`: Medium complexity with good performance (default)
- `full`: Comprehensive feature set (more compute-intensive)

## Standalone Usage

You can use the feature engineering pipeline independently from the stock predictor. The `feature_engineering_example.py` script demonstrates this:

```bash
python feature_engineering_example.py your_data.csv output_features.csv
```

## API Reference

### FeatureEngineer Class

Main class for orchestrating the automated feature engineering pipeline.

```python
from feature_engineering import FeatureEngineer, build_default_pipeline

# Use a pre-configured pipeline
pipeline = build_default_pipeline(target_type='regression', feature_type='balanced')

# Or create a custom pipeline
pipeline = FeatureEngineer(steps=[
    ('generate', {'include_basic': True, 'include_trend': True}),
    ('select', {'method': 'variance', 'params': {'threshold': 0.001}}),
    ('scale', {'method': 'robust'})
])

# Apply to your data
X_transformed = pipeline.fit_transform(data)

# Save and load pipelines
pipeline.save('my_pipeline.pkl')
loaded_pipeline = FeatureEngineer.load('my_pipeline.pkl')
```

### Feature Generation

```python
from feature_engineering import FinancialFeatureGenerator

generator = FinancialFeatureGenerator(
    include_basic=True,
    include_trend=True,
    include_volatility=True,
    include_volume=True,
    include_momentum=True,
    rolling_windows=[5, 10, 20, 50]
)

features = generator.fit_transform(data)
```

### Feature Selection

```python
from feature_engineering import FeatureSelector

selector = FeatureSelector(
    method='kbest',  # Options: 'variance', 'correlation', 'kbest', 'percentile', 'model', 'rfe'
    params={'k': 20, 'score_func': f_regression}
)

selected_features = selector.fit_transform(X, y)
importance = selector.get_feature_importances()
```

### Feature Transformation

```python
from feature_engineering import DimensionalityReducer

reducer = DimensionalityReducer(
    method='pca',  # Options: 'pca', 'tsne'
    n_components=10
)

reduced_features = reducer.fit_transform(X)
var_explained = reducer.get_explained_variance()  # For PCA only
```

### Interaction Feature Generation

```python
from feature_engineering import InteractionFeatureGenerator

interaction_generator = InteractionFeatureGenerator(
    interaction_type='polynomial',  # Options: 'polynomial', 'custom', 'financial'
    degree=2,
    interaction_only=True
)

interaction_features = interaction_generator.fit_transform(X)
```

## Advanced Usage

### Creating Train/Test Sets with Feature Engineering

```python
from feature_engineering import create_train_test_features

X_train, X_test, y_train, y_test, pipeline = create_train_test_features(
    data, 
    target_col='close',
    target_shift=5,  # Predict 5 days ahead
    test_size=0.2,  # 20% test set
    pipeline=None  # Will create a new pipeline if None
)
```

### Evaluating Feature Importance

```python
# Evaluate feature importance and model performance
evaluation = pipeline.evaluate_features(X_train, y_train, cv=5)
print(f"CV Score: {evaluation['mean_score']:.4f}")

# Visualize feature importance
pipeline.visualize_feature_importance(
    evaluation['feature_importance'],
    top_n=20,
    save_path="feature_importance.png"
)
```

## Integration with Models

The feature engineering pipeline is designed to work with all machine learning models in Quantra:

- RandomForest models
- PyTorch deep learning models
- TensorFlow LSTM models

It automatically adapts the feature set to the model requirements.

## Best Practices

1. Start with a `minimal` feature set for rapid prototyping
2. Use `balanced` or `full` for production models
3. Use the feature importance visualization to identify key predictors
4. Consider custom interaction terms for specific trading strategies
5. Save pipelines for reproducibility and consistency in production