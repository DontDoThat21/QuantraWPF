#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model Performance Tracking and Evaluation Example

This script demonstrates how to use the model performance tracking and evaluation
framework to monitor, evaluate, and visualize ML model performance.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import our modules
from model_performance_tracking import (
    ModelPerformanceTracker,
    PerformanceVisualizer,
    PerformanceDataManager,
    create_performance_tracker,
    evaluate_classification_model
)

from model_performance_integration import (
    EnhancedModelWrapper,
    EnhancedEnsembleModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_performance_example')


def simple_tracking_example():
    """Demonstrate basic performance tracking for a single model."""
    logger.info("Running simple tracking example...")
    
    # Create synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create a performance tracker
    tracker = create_performance_tracker(
        model_name="example_random_forest",
        model_type="classification"
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Record predictions
    tracker.record_prediction(y_pred, y_test)
    
    # Add feature importances
    feature_importances = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
    tracker.set_feature_importances(feature_importances)
    
    # Get current metrics
    metrics = tracker.get_current_metrics()
    logger.info(f"Model metrics: {metrics}")
    
    # Create a visualizer
    visualizer = PerformanceVisualizer(tracker)
    
    # Create dashboard image
    output_dir = os.path.join(os.getcwd(), "performance_output")
    os.makedirs(output_dir, exist_ok=True)
    
    dashboard_path = os.path.join(output_dir, "rf_dashboard.png")
    visualizer.create_performance_dashboard(dashboard_path, show_plots=False)
    logger.info(f"Performance dashboard created at: {dashboard_path}")
    
    return tracker


def batch_evaluation_example():
    """Demonstrate batch evaluation with metrics tracking over time."""
    logger.info("Running batch evaluation example...")
    
    # Create synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create a performance tracker
    tracker = create_performance_tracker(
        model_name="example_logistic_regression",
        model_type="classification"
    )
    
    # Train and evaluate in batches to simulate evolving performance
    batch_size = 40
    num_batches = len(X_test) // batch_size
    
    model = LogisticRegression(random_state=42)
    
    for i in range(num_batches):
        # Get batch data
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        
        X_batch_train = X_train[:batch_start + 100]  # Gradually increase training data
        y_batch_train = y_train[:batch_start + 100]
        
        X_batch_test = X_test[batch_start:batch_end]
        y_batch_test = y_test[batch_start:batch_end]
        
        # Train on current batch
        model.fit(X_batch_train, y_batch_train)
        
        # Predict and record performance
        y_pred = model.predict(X_batch_test)
        tracker.record_prediction(y_pred, y_batch_test)
        
        # Log progress
        current_metrics = tracker.get_current_metrics()
        if 'metrics' in current_metrics and 'accuracy' in current_metrics['metrics']:
            logger.info(f"Batch {i+1}/{num_batches} - Accuracy: {current_metrics['metrics']['accuracy']:.4f}")
    
    # Create visualizations for the metrics over time
    visualizer = PerformanceVisualizer(tracker)
    
    output_dir = os.path.join(os.getcwd(), "performance_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy over time
    accuracy_path = os.path.join(output_dir, "accuracy_over_time.png")
    visualizer.plot_metric_over_time("accuracy", accuracy_path)
    logger.info(f"Accuracy trend plot created at: {accuracy_path}")
    
    return tracker


def model_comparison_example():
    """Demonstrate comparing multiple models."""
    logger.info("Running model comparison example...")
    
    # Create synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create three different models
    models = [
        {
            "name": "random_forest",
            "instance": RandomForestClassifier(n_estimators=100, random_state=42)
        },
        {
            "name": "random_forest_deeper",
            "instance": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        },
        {
            "name": "logistic_regression",
            "instance": LogisticRegression(random_state=42)
        }
    ]
    
    trackers = []
    
    # Train and evaluate each model
    for model_info in models:
        model = model_info["instance"]
        model.fit(X_train, y_train)
        
        # Create performance tracker
        tracker = create_performance_tracker(
            model_name=f"example_{model_info['name']}",
            model_type="classification"
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Record predictions
        tracker.record_prediction(y_pred, y_test)
        
        # Add feature importances if available
        if hasattr(model, 'feature_importances_'):
            feature_importances = {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
            tracker.set_feature_importances(feature_importances)
        
        # Get current metrics
        metrics = tracker.get_current_metrics()
        logger.info(f"Model {model_info['name']} metrics: {metrics}")
        
        trackers.append(tracker)
    
    # Compare models
    data_manager = PerformanceDataManager()
    
    model_names = [tracker.model_name for tracker in trackers]
    comparison = data_manager.compare_models(model_names)
    
    logger.info("Model comparison results:")
    for metric_name, metric_data in comparison['metrics'].items():
        logger.info(f"Metric: {metric_name}")
        logger.info(f"  Best model: {metric_data['best_model']} ({metric_data['values'].get(metric_data['best_model'], 'N/A')})")
        logger.info(f"  Worst model: {metric_data['worst_model']} ({metric_data['values'].get(metric_data['worst_model'], 'N/A')})")
    
    return trackers


def drift_detection_example():
    """Demonstrate model drift detection."""
    logger.info("Running drift detection example...")
    
    # Create synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Further split test data into two parts (normal and drifted)
    X_test_normal, X_test_drift, y_test_normal, y_test_drift = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    # Introduce artificial drift by flipping some labels
    drift_indices = np.random.choice(len(y_test_drift), size=int(len(y_test_drift) * 0.3), replace=False)
    y_test_drift_modified = y_test_drift.copy()
    y_test_drift_modified[drift_indices] = 1 - y_test_drift_modified[drift_indices]
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create a tracker
    tracker = create_performance_tracker(
        model_name="drift_detection_example",
        model_type="classification"
    )
    
    # First, establish baseline performance
    logger.info("Establishing baseline performance...")
    for i in range(5):  # Multiple batches to establish baseline
        # Select a batch
        batch_indices = np.random.choice(len(X_test_normal), size=50, replace=False)
        X_batch = X_test_normal[batch_indices]
        y_batch = y_test_normal[batch_indices]
        
        # Predict and record
        y_pred = model.predict(X_batch)
        tracker.record_prediction(y_pred, y_batch)
    
    # Establish baseline for drift detection
    tracker.drift_detector.establish_baseline()
    
    # Now, introduce drift gradually
    logger.info("Introducing performance drift...")
    for i in range(10):
        # Mix increasing proportion of drifted data
        drift_ratio = i / 10
        normal_size = int(50 * (1 - drift_ratio))
        drift_size = 50 - normal_size
        
        # Select samples from both normal and drift datasets
        normal_indices = np.random.choice(len(X_test_normal), size=normal_size, replace=False)
        drift_indices = np.random.choice(len(X_test_drift), size=drift_size, replace=False)
        
        X_batch = np.vstack([
            X_test_normal[normal_indices],
            X_test_drift[drift_indices]
        ])
        
        y_batch = np.concatenate([
            y_test_normal[normal_indices],
            y_test_drift_modified[drift_indices]
        ])
        
        # Predict and record
        y_pred = model.predict(X_batch)
        tracker.record_prediction(y_pred, y_batch)
        
        # Check if drift was detected
        drift_events = tracker.drift_detector.get_drift_events()
        if drift_events:
            latest_drift = drift_events[-1]
            logger.info(f"Drift detected at batch {i+1}! Details: {latest_drift['details']}")
    
    # Create visualization showing drift
    visualizer = PerformanceVisualizer(tracker)
    
    output_dir = os.path.join(os.getcwd(), "performance_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot metrics showing drift
    drift_path = os.path.join(output_dir, "drift_detection.png")
    visualizer.plot_metric_over_time("accuracy", drift_path)
    logger.info(f"Drift visualization created at: {drift_path}")
    
    return tracker


def ensemble_tracking_example():
    """Demonstrate performance tracking with ensemble models."""
    logger.info("Running ensemble tracking example...")
    
    # Create synthetic classification data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Create individual models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    # Train models
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Create enhanced model wrappers
    rf_wrapper = EnhancedModelWrapper(rf_model, "random_forest", "classification")
    lr_wrapper = EnhancedModelWrapper(lr_model, "logistic", "classification")
    
    # Create enhanced ensemble
    ensemble = EnhancedEnsembleModel(
        models=[rf_wrapper, lr_wrapper],
        weights=[0.7, 0.3],
        task_type="classification",
        name="example_ensemble"
    )
    
    # Evaluate the ensemble
    evaluation = ensemble.evaluate(X_test, y_test)
    
    # Log results
    logger.info("Ensemble evaluation results:")
    logger.info(f"  Ensemble accuracy: {evaluation['ensemble']['accuracy']:.4f}")
    logger.info(f"  RF model accuracy: {evaluation['models']['random_forest']['accuracy']:.4f}")
    logger.info(f"  LR model accuracy: {evaluation['models']['logistic']['accuracy']:.4f}")
    
    # Get performance comparison
    comparison = ensemble.get_performance_comparison()
    logger.info(f"Top component model: {comparison['top_component']}")
    
    # Create visualizations
    output_dir = os.path.join(os.getcwd(), "performance_output")
    os.makedirs(output_dir, exist_ok=True)
    
    dashboard_path = os.path.join(output_dir, "ensemble_dashboard.png")
    ensemble.create_comparative_dashboard(dashboard_path)
    logger.info(f"Ensemble dashboard created at: {dashboard_path}")
    
    return ensemble


def main():
    """Run the different examples."""
    output_dir = os.path.join(os.getcwd(), "performance_output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Example 1: Simple tracking
        simple_tracking_example()
        
        # Example 2: Batch evaluation
        batch_evaluation_example()
        
        # Example 3: Model comparison
        model_comparison_example()
        
        # Example 4: Drift detection
        drift_detection_example()
        
        # Example 5: Ensemble tracking
        ensemble_tracking_example()
        
        logger.info(f"All examples completed. Output saved to: {output_dir}")
        logger.info("You can view the generated visualizations to see model performance.")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()