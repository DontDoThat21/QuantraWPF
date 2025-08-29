# Python Ensemble Learning: Decision Pipeline and Weighting Algorithms

## Overview

This document provides technical documentation for the ensemble learning decision pipeline implemented in the Quantra Python modules. It covers the algorithms and methodologies used for model selection, ensemble creation, and dynamic weighting value determination.

## Table of Contents

1. [Ensemble Creation Decision Process](#ensemble-creation-decision-process)
2. [Dynamic Weighting Algorithms](#dynamic-weighting-algorithms)
3. [Model Selection Criteria](#model-selection-criteria)
4. [Performance-Based Weight Adjustment](#performance-based-weight-adjustment)
5. [Stacking Meta-Model Training Pipeline](#stacking-meta-model-training-pipeline)
6. [Feature Importance Aggregation](#feature-importance-aggregation)
7. [Ensemble Method Selection Guidelines](#ensemble-method-selection-guidelines)

## Ensemble Creation Decision Process

### Automatic Model Selection Pipeline

The `train_model_ensemble()` function implements an automated pipeline for creating optimal ensembles:

```python
def train_model_ensemble(
    X, y, models_to_train=None, test_size=0.2, 
    ensemble_method='weighted_average', dynamic_weighting=True
):
```

#### Decision Flow:
1. **Model Diversity Assessment**: The system evaluates model diversity by training multiple model types (Random Forest, Gradient Boosting, ElasticNet)
2. **Performance Validation**: Each model's performance is evaluated on validation data
3. **Ensemble Compatibility Check**: Models are validated for ensemble compatibility based on prediction output types
4. **Weight Initialization**: Initial weights are assigned based on cross-validation performance

#### Model Type Detection Algorithm:

```python
# Automatic model type detection logic
if hasattr(model, 'predict') and hasattr(model, 'fit'):
    model_type = 'sklearn'
elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
    model_type = 'tensorflow'  
elif PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
    model_type = 'pytorch'
else:
    model_type = 'custom'
```

## Dynamic Weighting Algorithms

### Performance-Based Weight Update

The `update_weights()` method implements performance-based weight adjustment:

```python
def update_weights(self, X: np.ndarray, y: np.ndarray):
    for model_wrapper in self.models:
        preds = model_wrapper.predict(X)
        
        if self.task_type == 'regression':
            mse = mean_squared_error(y, preds)
            error = np.sqrt(mse)  # RMSE
            model_wrapper.record_performance(error, 'rmse')
            # Inverse relationship: higher error = lower weight
            model_wrapper.update_weight(1.0 / (error + 1e-10))
        else:  # classification
            accuracy = accuracy_score(y, preds)
            model_wrapper.record_performance(accuracy, 'accuracy')
            # Direct relationship: higher accuracy = higher weight
            model_wrapper.update_weight(accuracy + 1e-10)
```

### Weight Calculation Methodology:

#### For Regression Tasks:
- **Error Metric**: Root Mean Square Error (RMSE)
- **Weight Formula**: `weight = 1.0 / (RMSE + ε)` where ε = 1e-10 prevents division by zero
- **Rationale**: Models with lower prediction error receive higher weights

#### For Classification Tasks:
- **Performance Metric**: Accuracy Score
- **Weight Formula**: `weight = accuracy + ε` where ε = 1e-10 ensures minimum weight
- **Rationale**: Models with higher accuracy receive proportionally higher weights

### Weight Normalization

Weights are normalized during prediction to ensure they sum to 1:

```python
# Weighted average prediction
weights = weights.reshape(-1, 1)
weighted_sum = np.sum(predictions * weights, axis=0)
return weighted_sum / np.sum(weights)
```

## Model Selection Criteria

### Diversity Assessment

The ensemble framework prioritizes model diversity through:

1. **Algorithm Diversity**: Different model types (tree-based, linear, neural networks)
2. **Parameter Diversity**: Varied hyperparameters within model types
3. **Training Diversity**: Different random states and training subsets

### Inclusion Criteria

Models are included in ensembles based on:

1. **Minimum Performance Threshold**: Models must exceed baseline performance
2. **Prediction Compatibility**: Output formats must be ensemble-compatible
3. **Error Correlation**: Models with uncorrelated errors are preferred
4. **Computational Efficiency**: Balance between performance and inference time

### Exclusion Criteria

Models are excluded if they:

1. **Fail Compatibility Checks**: Cannot produce valid predictions
2. **Show Poor Performance**: Consistently underperform compared to ensemble average
3. **Exhibit High Correlation**: Predictions are highly correlated with existing models
4. **Have Resource Constraints**: Exceed computational or memory limits

## Performance-Based Weight Adjustment

### Historical Performance Tracking

Each `ModelWrapper` maintains performance history:

```python
def record_performance(self, performance: float, metric: str):
    """Record model performance for weight adjustment"""
    self.performance_history.append({
        'performance': performance,
        'metric': metric,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
```

### Adaptive Weight Updates

The system uses sliding window performance evaluation:

1. **Recent Performance Weighting**: More recent performance has higher influence
2. **Performance Variance Consideration**: Models with consistent performance receive higher weights
3. **Degradation Detection**: Models showing performance degradation receive reduced weights

### Weight Bounds and Constraints

- **Minimum Weight**: 1e-10 to prevent complete model exclusion
- **Maximum Weight**: No upper bound, allowing dominant high-performing models
- **Normalization**: Weights are normalized before ensemble prediction

## Stacking Meta-Model Training Pipeline

### Meta-Feature Generation

The `_generate_meta_features()` method creates training data for the meta-model:

```python
def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
    meta_features = []
    
    for model_wrapper in self.models:
        if self.task_type == 'regression':
            preds = model_wrapper.predict(X)
            meta_features.append(preds)
        else:  # classification
            probs = model_wrapper.predict_proba(X)
            if probs is not None:
                # Use probability distributions as features
                meta_features.extend([probs[:, i] for i in range(probs.shape[1])])
            else:
                # Fallback to hard predictions
                preds = model_wrapper.predict(X)
                meta_features.append(preds)
    
    return np.column_stack(meta_features)
```

### Meta-Model Selection Logic

#### For Regression:
- **Default Choice**: Ridge Regression with α = 1.0
- **Rationale**: Ridge provides regularization to prevent overfitting to base model predictions
- **Alternative**: Can be configured to use other linear models

#### For Classification:
- **Default Choice**: Logistic Regression with C = 1.0, max_iter = 1000
- **Rationale**: Logistic regression handles probability inputs well and provides interpretable coefficients
- **Alternative**: Can be configured to use other classification algorithms

### Cross-Validation Strategy

The stacking implementation uses:
1. **Base Model Training**: On full training set
2. **Meta-Model Training**: On validation predictions to prevent overfitting
3. **Final Prediction**: Meta-model combines base model predictions

## Feature Importance Aggregation

### Individual Model Importance Extraction

```python
def get_feature_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Get combined feature importance from all models"""
    combined_importance = {}
    total_weight = 0.0
    
    for model_wrapper in self.models:
        importance = model_wrapper.get_feature_importance(X)
        if importance:
            # Weight importance by model performance
            for feature, score in importance.items():
                weighted_score = score * model_wrapper.weight
                combined_importance[feature] = combined_importance.get(feature, 0) + weighted_score
            total_weight += model_wrapper.weight
    
    # Normalize by total weight
    if total_weight > 0:
        for feature in combined_importance:
            combined_importance[feature] /= total_weight
    
    return combined_importance
```

### Aggregation Methodology

1. **Weight-Based Scaling**: Each model's feature importance is scaled by its ensemble weight
2. **Feature Alignment**: Features are aligned across different model types
3. **Normalization**: Final importance scores are normalized by total model weight
4. **Consensus Building**: Features important across multiple models receive higher scores

### Importance Score Interpretation

- **High Scores (>0.1)**: Features consistently important across ensemble
- **Medium Scores (0.01-0.1)**: Features important in subset of models
- **Low Scores (<0.01)**: Features with minimal predictive value

## Ensemble Method Selection Guidelines

### Method Characteristics

#### Simple Average (`simple_average`)
- **Use Case**: When all models have similar performance
- **Assumption**: Equal contribution from all models
- **Advantage**: Simple, no training required
- **Disadvantage**: Ignores individual model performance

#### Weighted Average (`weighted_average`)
- **Use Case**: When models have different performance levels
- **Assumption**: Better models should have more influence
- **Advantage**: Adapts to model performance differences
- **Disadvantage**: Requires performance evaluation

#### Stacking (`stacking`)
- **Use Case**: When complex non-linear combinations are beneficial
- **Assumption**: Meta-model can learn optimal combination
- **Advantage**: Can capture complex model interactions
- **Disadvantage**: Risk of overfitting, requires additional training

#### Majority Vote (`majority_vote`)
- **Use Case**: Classification tasks with categorical outputs
- **Assumption**: Democratic decision making
- **Advantage**: Robust to individual model errors
- **Disadvantage**: Limited to classification, ignores confidence

### Selection Decision Tree

```
1. Task Type?
   ├─ Regression → Consider: simple_average, weighted_average, stacking
   └─ Classification → Consider: weighted_average, stacking, majority_vote

2. Model Performance Variance?
   ├─ High Variance → Prefer: weighted_average, stacking
   └─ Low Variance → Consider: simple_average

3. Training Data Availability?
   ├─ Limited → Prefer: simple_average, weighted_average
   └─ Abundant → Consider: stacking

4. Computational Constraints?
   ├─ High Constraints → Prefer: simple_average, weighted_average
   └─ Low Constraints → Consider: stacking

5. Interpretability Requirements?
   ├─ High → Prefer: weighted_average
   └─ Low → Consider: stacking
```

## Best Practices for Ensemble Decision Making

### Model Selection
1. **Diversify Model Types**: Include different algorithm families
2. **Validate Performance**: Use cross-validation for model selection
3. **Monitor Correlation**: Avoid highly correlated models
4. **Consider Computational Cost**: Balance performance vs. inference time

### Weighting Strategy
1. **Use Dynamic Weighting**: Enable adaptive weight adjustment
2. **Regular Retraining**: Update weights with new data
3. **Performance Monitoring**: Track individual model degradation
4. **Robustness Testing**: Validate ensemble on out-of-sample data

### Meta-Model Training
1. **Prevent Overfitting**: Use separate validation data for meta-model
2. **Regularization**: Apply appropriate regularization to meta-model
3. **Cross-Validation**: Use cross-validation for meta-model selection
4. **Feature Engineering**: Consider transforming meta-features

## Integration with Quantra Trading System

The ensemble decision pipeline integrates with Quantra's trading system through:

1. **Real-Time Inference**: Ensemble predictions are generated for live trading
2. **Strategy Integration**: Ensemble outputs feed into trading strategy decisions
3. **Risk Management**: Ensemble confidence scores inform position sizing
4. **Performance Tracking**: Ensemble performance is monitored for strategy optimization

This documentation provides the technical foundation for understanding and extending the ensemble learning capabilities within the Quantra algorithmic trading platform.