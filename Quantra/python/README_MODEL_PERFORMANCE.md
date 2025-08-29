# ML Model Performance Tracking and Evaluation Framework

This framework provides comprehensive tools for tracking, evaluating, and visualizing
the performance of machine learning models over time. It integrates with the existing ML
infrastructure in Quantra and provides a consistent interface for evaluating diverse model types.

## Features

- **Real-time metrics collection and visualization**
- **Historical performance comparison across model versions**
- **Model drift detection and alerting**
- **Performance data persistence and retrieval**
- **Dashboard generation for monitoring key performance indicators**
- **Integration with existing ensemble learning and real-time inference modules**

## Core Components

### 1. ModelPerformanceTracker

The main class for tracking model performance metrics:

```python
from model_performance_tracking import create_performance_tracker

# Create a tracker for a classification model
tracker = create_performance_tracker(
    model_name="my_model",
    model_type="classification",  # or "regression"
    version="v1.0"
)

# Record predictions and actual values
y_pred = model.predict(X_test)
tracker.record_prediction(y_pred, y_test)

# Get current metrics
metrics = tracker.get_current_metrics()
print(f"Accuracy: {metrics['metrics']['accuracy']}")

# Add feature importance information
tracker.set_feature_importances({
    "feature1": 0.5,
    "feature2": 0.3,
    "feature3": 0.2
})
```

### 2. PerformanceVisualizer

Creates visualizations for model performance metrics:

```python
from model_performance_tracking import PerformanceVisualizer

# Create visualizer
visualizer = PerformanceVisualizer(tracker)

# Plot a specific metric over time
visualizer.plot_metric_over_time("accuracy", "accuracy_trend.png")

# Plot feature importances
visualizer.plot_feature_importance(top_n=10, save_path="feature_importance.png")

# Create a comprehensive dashboard
visualizer.create_performance_dashboard("dashboard.png")
```

### 3. ModelDriftDetector

Detects changes in model performance that may indicate drift:

```python
# The drift detector is automatically created with the tracker
# Establish a baseline after initial training
tracker.drift_detector.establish_baseline()

# Later, check if new predictions indicate drift
drift_detected = tracker.drift_detector.check_drift(current_metrics)

# Get drift events history
drift_events = tracker.drift_detector.get_drift_events()
```

### 4. Integration with Existing ML Modules

Enhanced versions of existing ML classes with integrated performance tracking:

```python
from model_performance_integration import (
    enhance_model_wrapper,
    enhance_ensemble_model,
    enhance_inference_pipeline
)

# Enhance an existing model wrapper
enhanced_wrapper = enhance_model_wrapper(original_wrapper, model_type="classification")

# Enhance an ensemble model
enhanced_ensemble = enhance_ensemble_model(original_ensemble, name="my_ensemble")

# Enhance a real-time inference pipeline
enhanced_pipeline = enhance_inference_pipeline(original_pipeline)
```

## Dashboard Generation

The framework includes tools for creating both image-based and HTML dashboards:

```python
# Create a simple performance dashboard
from model_performance_tracking import create_performance_tracker
from model_performance_dashboard_example import create_html_dashboard

# Create tracker and record predictions
tracker = create_performance_tracker("my_model")
tracker.record_prediction(y_pred, y_true)

# Generate HTML dashboard
dashboard_path = create_html_dashboard(tracker, "dashboard_output")
print(f"Dashboard created at: {dashboard_path}")
```

## Usage Examples

For detailed usage examples, see:

1. `model_performance_example.py` - Basic examples of using the framework
2. `model_performance_dashboard_example.py` - Dashboard generation examples
3. `model_performance_integration.py` - Integration with existing ML components

## Quick Start

```python
from model_performance_tracking import evaluate_classification_model

# Train a model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate and track performance
metrics = evaluate_classification_model(
    model_name="my_random_forest",
    y_pred=y_pred,
    y_true=y_test,
    feature_importances={f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
)

print(f"Tracked metrics: {metrics}")
```

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn (for enhanced visualizations)