# Ensemble Learning Framework

## Overview

The ensemble learning framework provides comprehensive tools for combining multiple machine learning models to improve prediction accuracy and robustness. It supports both **homogeneous ensembles** (multiple instances of the same model) and **heterogeneous ensembles** (different model types) with advanced combining strategies including bagging, stacking, blending, and dynamic selection.

## Features

### Core Capabilities
- Support for different model types (scikit-learn, PyTorch, TensorFlow)
- **Homogeneous ensemble methods**:
  - Bagging (Bootstrap Aggregating)
  - Random Subspace Method (feature bagging)
  - Pasting (sampling without replacement)
  - Extra Trees (extremely randomized trees)
- **Heterogeneous ensemble methods**:
  - Simple averaging/voting
  - Weighted averaging based on confidence or performance
  - Stacking (meta-learning)
  - Cross-validation stacking
  - Blending (holdout validation set)
  - Dynamic ensemble selection
- Comprehensive evaluation metrics
- Transparent feature importance analysis across models
- Flexible integration with existing prediction modules

### New in Latest Update
- **HomogeneousEnsemble** class for bagging, random subspace, pasting, and extra trees
- **HeterogeneousEnsemble** class with advanced stacking, blending, and dynamic selection
- **EnsembleFactory** for easy ensemble creation with factory methods
- Enhanced parameter validation and error handling
- Comprehensive test suite with 13+ test cases

## Installation

The ensemble learning framework is included in the Quantra Python package. No additional installation is required beyond the standard dependencies.

## Quick Start Examples

### Homogeneous Ensembles

#### Bagging Ensemble
```python
from python.ensemble_learning import EnsembleFactory
from sklearn.tree import DecisionTreeRegressor

# Create bagging ensemble
bagging_ensemble = EnsembleFactory.create_bagging_ensemble(
    base_estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=10,
    max_samples=0.8,
    random_state=42
)

# Fit and predict
bagging_ensemble.fit(X_train, y_train)
predictions = bagging_ensemble.predict(X_test)
```

#### Random Subspace Ensemble
```python
# Create random subspace ensemble
rs_ensemble = EnsembleFactory.create_random_subspace_ensemble(
    base_estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=10,
    max_features=0.6,  # Use 60% of features
    random_state=42
)

rs_ensemble.fit(X_train, y_train)
predictions = rs_ensemble.predict(X_test)
```

### Heterogeneous Ensembles

#### Stacking Ensemble with Cross-Validation
```python
from python.ensemble_learning import EnsembleFactory
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Train different base models
models = [
    RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train),
    GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train),
    LinearRegression().fit(X_train, y_train)
]

# Create stacking ensemble
stacking_ensemble = EnsembleFactory.create_stacking_ensemble(
    models=models,
    cv_folds=5,
    task_type='regression'
)

stacking_ensemble.fit(X_train, y_train)
predictions = stacking_ensemble.predict(X_test)
```

#### Blending Ensemble
```python
# Create blending ensemble
blending_ensemble = EnsembleFactory.create_blending_ensemble(
    models=models,
    holdout_size=0.2,  # Use 20% for meta-model training
    task_type='regression'
)

blending_ensemble.fit(X_train, y_train)
predictions = blending_ensemble.predict(X_test)
```

## Usage Examples

### Basic Ensemble Example

```python
from python.ensemble_learning import EnsembleModel, ModelWrapper
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Create and train some models
rf_model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
lr_model = LinearRegression().fit(X_train, y_train)

# Create model wrappers
rf_wrapper = ModelWrapper(rf_model, model_type='sklearn', name='RandomForest')
lr_wrapper = ModelWrapper(lr_model, model_type='sklearn', name='LinearRegression')

# Create ensemble
ensemble = EnsembleModel(
    models=[rf_wrapper, lr_wrapper],
    ensemble_method='weighted_average',
    task_type='regression'
)

# Make predictions
predictions = ensemble.predict(X_test)

# Evaluate the ensemble
evaluation = ensemble.evaluate(X_test, y_test)
print(f"Ensemble RMSE: {evaluation['ensemble']['rmse']}")
```

### Training an Ensemble with Multiple Models

```python
from python.ensemble_learning import train_model_ensemble
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

# Define models for the ensemble
models = [
    {'model_class': RandomForestRegressor, 'n_estimators': 100, 'random_state': 42},
    {'model_class': GradientBoostingRegressor, 'n_estimators': 100, 'random_state': 42},
    {'model_class': ElasticNet, 'alpha': 0.01, 'random_state': 42}
]

# Train ensemble
ensemble, results = train_model_ensemble(
    X_train, y_train,
    models_to_train=models,
    ensemble_method='weighted_average',
    task_type='regression',
    dynamic_weighting=True
)

# Print results
print(f"Ensemble metrics: {results['ensemble_metrics']}")
print(f"Individual model metrics: {results['model_metrics']}")
```

### Classification Example

```python
from python.ensemble_learning import EnsembleModel, ModelWrapper
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create and train classification models
rf_cls = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
log_cls = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# Create ensemble for classification
ensemble = EnsembleModel(
    models=[rf_cls, log_cls],
    model_types=['sklearn', 'sklearn'],
    ensemble_method='weighted_average',
    voting='soft',
    task_type='classification'
)

# Make class predictions
class_predictions = ensemble.predict(X_test)

# Get class probabilities
probabilities = ensemble.predict_proba(X_test)
```

### Using Stacking Ensemble

```python
from python.ensemble_learning import EnsembleModel, ModelWrapper
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Create and train base models
rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
gbm = GradientBoostingRegressor(n_estimators=100).fit(X_train, y_train)

# Create model wrappers
rf_wrapper = ModelWrapper(rf, model_type='sklearn', name='RandomForest')
gbm_wrapper = ModelWrapper(gbm, model_type='sklearn', name='GradientBoosting')

# Create stacking ensemble
stacking_ensemble = EnsembleModel(
    models=[rf_wrapper, gbm_wrapper],
    ensemble_method='stacking',
    task_type='regression'
)

# Train meta-model
stacking_ensemble.fit(X_val, y_val)

# Make predictions
predictions = stacking_ensemble.predict(X_test)
```

### Integration with Existing Modules

The ensemble learning framework can be easily integrated with existing Quantra modules:

```python
from python.model_ensemble_integration import integrate_all_modules

# Integrate ensemble learning with all compatible modules
results = integrate_all_modules()

# Now you can use enhanced ensemble capabilities in stock_predictor, 
# market_regime_detection, and anomaly_detection modules
```

## Advanced Features

### Dynamic Model Weighting

Models in an ensemble can have their weights adjusted dynamically based on performance:

```python
# Create ensemble with dynamic weighting
dynamic_ensemble = EnsembleModel(
    models=models,
    ensemble_method='weighted_average',
    dynamic_weighting=True,
    task_type='regression'
)

# Update weights based on validation data performance
dynamic_ensemble.update_weights(X_val, y_val)
```

### Feature Importance Analysis

Get combined feature importance across all models in the ensemble:

```python
# Get feature importance
importance = ensemble.get_feature_importance(X_test)

# Print top features
sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("Top important features:")
for feature, score in sorted_features[:5]:
    print(f"  {feature}: {score:.4f}")
```

### Saving and Loading Ensembles

Save and load ensemble models for later use:

```python
# Save ensemble
ensemble.save('/path/to/ensemble_model')

# Load ensemble later
loaded_ensemble = EnsembleModel.load('/path/to/ensemble_model')
```

## Ensemble Method Comparison

| Method | Type | Use Case | Advantages | Disadvantages |
|--------|------|----------|------------|---------------|
| **Bagging** | Homogeneous | High variance models | Reduces overfitting | May underfit with stable models |
| **Random Subspace** | Homogeneous | High-dimensional data | Handles feature noise | Needs many features |
| **Pasting** | Homogeneous | Large datasets | No bootstrap bias | Less diversity than bagging |
| **Extra Trees** | Homogeneous | Tree-based models | High randomness | May lose interpretability |
| **Simple Average** | Heterogeneous | Similar model performance | Simple and robust | Ignores model quality differences |
| **Weighted Average** | Heterogeneous | Varied model performance | Adapts to model quality | Requires performance evaluation |
| **Stacking** | Heterogeneous | Complex patterns | Learns optimal combination | Risk of overfitting |
| **Blending** | Heterogeneous | Limited data | Simpler than stacking | Less robust than CV stacking |
| **Dynamic Selection** | Heterogeneous | Changing patterns | Adapts to local performance | Requires performance history |

## Best Practices

1. **Choose the right ensemble type**:
   - Use **homogeneous ensembles** when you have a good base model but want to reduce variance
   - Use **heterogeneous ensembles** when you have diverse models with different strengths

2. **Model diversity is key**: 
   - For homogeneous: vary training data, features, or parameters
   - For heterogeneous: use fundamentally different algorithm types

3. **Balance complexity**: More models isn't always better. Focus on quality over quantity.

4. **Use appropriate validation**:
   - Cross-validation for stacking
   - Holdout sets for blending
   - Separate validation for dynamic weighting

5. **Monitor performance**: Regularly evaluate both ensemble and individual model performance.

6. **Consider computational cost**: Ensemble methods increase training and prediction time.

## Factory Methods Reference

The `EnsembleFactory` class provides convenient methods for creating ensembles:

```python
# Homogeneous ensembles
EnsembleFactory.create_bagging_ensemble(base_estimator, n_estimators=10, max_samples=1.0, ...)
EnsembleFactory.create_random_subspace_ensemble(base_estimator, n_estimators=10, max_features=0.5, ...)

# Heterogeneous ensembles
EnsembleFactory.create_stacking_ensemble(models, cv_folds=5, ...)
EnsembleFactory.create_blending_ensemble(models, holdout_size=0.2, ...)
EnsembleFactory.create_heterogeneous_ensemble(models, ensemble_type='weighted_average', ...)
```

## Extending the Framework

The ensemble learning framework is designed to be easily extended:

1. **Custom models**: Wrap any model with `ModelWrapper` to add it to an ensemble.
2. **New ensemble methods**: Add new combination strategies by extending the ensemble classes.
3. **Custom evaluation metrics**: Add domain-specific metrics in the `evaluate()` method.

For more information, refer to the code documentation and examples in the `ensemble_learning.py` module.