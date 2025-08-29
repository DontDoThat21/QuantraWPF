#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model Performance Tracking and Evaluation Framework

This module provides comprehensive tools for tracking, evaluating, and visualizing
the performance of machine learning models over time. It supports:

- Real-time metrics collection and visualization
- Historical performance comparison across model versions
- Model drift detection and alerting
- Performance data persistence and retrieval
- Dashboard generation for monitoring key performance indicators

The framework integrates with existing ML infrastructure and provides a consistent
interface for evaluating diverse model types (classification, regression, etc.)
"""

import os
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple, Callable, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, precision_recall_curve, confusion_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_performance_tracking')

# Set up paths for data persistence
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERFORMANCE_DATA_DIR = os.path.join(BASE_DIR, 'models', 'performance_data')
os.makedirs(PERFORMANCE_DATA_DIR, exist_ok=True)


class ModelPerformanceTracker:
    """
    Tracks and stores performance metrics for machine learning models.
    
    This class collects, analyzes, and persists performance metrics for ML models.
    It supports both classification and regression models, and provides methods
    for comparing performance across model versions and over time.
    """
    
    def __init__(
        self, 
        model_name: str, 
        model_type: str = 'classification',
        version: str = None,
        max_history: int = 1000,
        metrics_to_track: Set[str] = None,
        persistence_enabled: bool = True
    ):
        """
        Initialize the performance tracker.
        
        Args:
            model_name: Unique identifier for the model
            model_type: Type of model ('classification' or 'regression')
            version: Version identifier for the model
            max_history: Maximum number of prediction records to keep in memory
            metrics_to_track: Set of metric names to track
            persistence_enabled: Whether to persist metrics to disk
        """
        self.model_name = model_name
        self.model_type = model_type
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_history = max_history
        self.persistence_enabled = persistence_enabled
        
        # Determine which metrics to track based on model type
        self.metrics_to_track = metrics_to_track or self._default_metrics()
        
        # Initialize data structures
        self.predictions = deque(maxlen=max_history)
        self.actuals = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.inference_times = deque(maxlen=max_history)
        self.metric_history = defaultdict(list)
        self.aggregated_metrics = {}
        self.feature_importances = {}
        
        # Additional tracking for time-series analysis
        self.batch_metrics = []  # Stores metrics for each evaluation batch
        
        # Threading protection
        self._lock = threading.Lock()
        
        # Model drift detection
        self.drift_detector = ModelDriftDetector(self)
        
        # Initialize persistence
        self._init_persistence()
        
        logger.info(f"Initialized performance tracker for model: {model_name} (v{version})")
    
    def _default_metrics(self) -> Set[str]:
        """Return the default set of metrics based on model type."""
        if self.model_type == 'classification':
            return {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        elif self.model_type == 'regression':
            return {'rmse', 'mae', 'r2', 'mape'}
        else:
            return {'custom'}
    
    def _init_persistence(self):
        """Initialize the persistence layer."""
        if self.persistence_enabled:
            self.model_dir = os.path.join(PERFORMANCE_DATA_DIR, self.model_name)
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Check if we have existing data to load
            self._load_latest_metrics()
    
    def record_prediction(
        self, 
        y_pred: Union[np.ndarray, List], 
        y_true: Union[np.ndarray, List], 
        inference_time: float = None,
        metadata: Dict = None
    ):
        """
        Record a new prediction and its actual value.
        
        Args:
            y_pred: Predicted values
            y_true: Actual/true values
            inference_time: Time taken to make the prediction (seconds)
            metadata: Additional metadata about the prediction
        """
        with self._lock:
            timestamp = datetime.now()
            
            # Convert to numpy arrays for consistent handling
            y_pred_array = np.array(y_pred)
            y_true_array = np.array(y_true)
            
            # Store the prediction
            self.predictions.append(y_pred_array)
            self.actuals.append(y_true_array)
            self.timestamps.append(timestamp)
            
            if inference_time is not None:
                self.inference_times.append(inference_time)
            
            # Update metrics if we have enough data
            if len(self.predictions) % 10 == 0:  # Update metrics every 10 predictions
                self._update_metrics()
    
    def _update_metrics(self):
        """Calculate and update all tracked metrics."""
        if not self.predictions:
            return
        
        # Convert deques to numpy arrays for metric calculation
        y_pred = np.array(list(self.predictions))
        y_true = np.array(list(self.actuals))
        
        # Calculate metrics based on model type
        metrics = {}
        
        if self.model_type == 'classification':
            # Handle multi-class vs binary classification
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # Multi-class with probability outputs
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                # Binary classification
                y_pred_classes = (y_pred > 0.5).astype(int)
            
            # Calculate classification metrics
            if 'accuracy' in self.metrics_to_track:
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred_classes))
            
            if 'precision' in self.metrics_to_track:
                metrics['precision'] = float(precision_score(y_true, y_pred_classes, average='weighted', zero_division=0))
            
            if 'recall' in self.metrics_to_track:
                metrics['recall'] = float(recall_score(y_true, y_pred_classes, average='weighted', zero_division=0))
            
            if 'f1' in self.metrics_to_track:
                metrics['f1'] = float(f1_score(y_true, y_pred_classes, average='weighted', zero_division=0))
            
            # ROC AUC - only for binary classification with probability outputs
            if 'roc_auc' in self.metrics_to_track and len(np.unique(y_true)) == 2:
                try:
                    # Ensure we have probability outputs for ROC AUC
                    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                        roc_scores = y_pred
                    else:
                        roc_scores = y_pred[:, 1]  # Use positive class probability
                    
                    fpr, tpr, _ = roc_curve(y_true, roc_scores)
                    metrics['roc_auc'] = float(auc(fpr, tpr))
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {e}")
        
        elif self.model_type == 'regression':
            # Calculate regression metrics
            if 'rmse' in self.metrics_to_track:
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            
            if 'mae' in self.metrics_to_track:
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            
            if 'r2' in self.metrics_to_track:
                metrics['r2'] = float(r2_score(y_true, y_pred))
            
            if 'mape' in self.metrics_to_track:
                # Mean Absolute Percentage Error
                mask = y_true != 0  # Avoid division by zero
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics['mape'] = float(mape)
        
        # Store metrics in history
        current_time = datetime.now()
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append({
                'timestamp': current_time,
                'value': value
            })
        
        # Update aggregated metrics
        self.aggregated_metrics = {
            'last_updated': current_time,
            'metrics': metrics,
            'sample_count': len(self.predictions)
        }
        
        # Add to batch metrics for time series analysis
        self.batch_metrics.append({
            'timestamp': current_time,
            'metrics': metrics,
            'sample_count': len(self.predictions)
        })
        
        # Check for model drift
        self.drift_detector.check_drift(metrics)
        
        # Persist metrics if enabled
        if self.persistence_enabled and len(self.batch_metrics) % 5 == 0:
            self._persist_metrics()
    
    def _persist_metrics(self):
        """Save the current metrics to disk."""
        if not self.persistence_enabled:
            return
        
        try:
            # Prepare data for saving
            data_to_save = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'version': self.version,
                'last_updated': datetime.now().isoformat(),
                'aggregated_metrics': self.aggregated_metrics,
                'batch_metrics': self.batch_metrics[-100:],  # Save only the last 100 batches
                'feature_importances': self.feature_importances
            }
            
            # Generate filename with timestamp to avoid overwriting
            filename = f"{self.model_name}_v{self.version}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            filepath = os.path.join(self.model_dir, filename)
            
            # Save to disk
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            # Also save as latest for quick loading
            latest_filepath = os.path.join(self.model_dir, f"{self.model_name}_latest.json")
            with open(latest_filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2, default=str)
            
            logger.debug(f"Persisted metrics to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def _load_latest_metrics(self):
        """Load the latest metrics from disk."""
        if not self.persistence_enabled:
            return
        
        try:
            latest_filepath = os.path.join(self.model_dir, f"{self.model_name}_latest.json")
            
            if os.path.exists(latest_filepath):
                with open(latest_filepath, 'r') as f:
                    data = json.load(f)
                
                # Restore metrics
                if 'aggregated_metrics' in data:
                    self.aggregated_metrics = data['aggregated_metrics']
                
                if 'batch_metrics' in data:
                    self.batch_metrics = data['batch_metrics']
                
                if 'feature_importances' in data:
                    self.feature_importances = data['feature_importances']
                
                logger.info(f"Loaded latest metrics for model {self.model_name}")
        
        except Exception as e:
            logger.warning(f"Failed to load latest metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the current performance metrics."""
        with self._lock:
            return {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'version': self.version,
                'metrics': self.aggregated_metrics.get('metrics', {}),
                'sample_count': self.aggregated_metrics.get('sample_count', 0),
                'last_updated': self.aggregated_metrics.get('last_updated', None)
            }
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get historical values for a specific metric."""
        with self._lock:
            return self.metric_history.get(metric_name, [])
    
    def get_all_metric_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical values for all metrics."""
        with self._lock:
            return dict(self.metric_history)
    
    def set_feature_importances(self, feature_importances: Dict[str, float]):
        """Set feature importance values."""
        with self._lock:
            self.feature_importances = feature_importances
            
            if self.persistence_enabled:
                self._persist_metrics()
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance values."""
        with self._lock:
            return self.feature_importances
    
    def compare_with_version(self, other_version: str) -> Dict[str, Any]:
        """Compare metrics with another version of the same model."""
        result = {'model_name': self.model_name, 'comparisons': {}}
        
        try:
            other_filepath = os.path.join(
                self.model_dir, 
                f"{self.model_name}_v{other_version}_latest.json"
            )
            
            if not os.path.exists(other_filepath):
                # Try to find any file for this version
                candidate_files = [
                    f for f in os.listdir(self.model_dir) 
                    if f.startswith(f"{self.model_name}_v{other_version}_")
                ]
                
                if not candidate_files:
                    return {
                        'error': f"No data found for version {other_version}",
                        'model_name': self.model_name,
                        'current_version': self.version
                    }
                
                other_filepath = os.path.join(self.model_dir, sorted(candidate_files)[-1])
            
            # Load the other version's metrics
            with open(other_filepath, 'r') as f:
                other_data = json.load(f)
            
            other_metrics = other_data.get('aggregated_metrics', {}).get('metrics', {})
            current_metrics = self.aggregated_metrics.get('metrics', {})
            
            # Calculate differences
            for metric_name in set(current_metrics.keys()).union(set(other_metrics.keys())):
                current_value = current_metrics.get(metric_name, None)
                other_value = other_metrics.get(metric_name, None)
                
                if current_value is not None and other_value is not None:
                    # Both versions have this metric
                    absolute_diff = current_value - other_value
                    relative_diff = absolute_diff / abs(other_value) if other_value != 0 else float('inf')
                    
                    result['comparisons'][metric_name] = {
                        'current_value': current_value,
                        'other_value': other_value,
                        'absolute_diff': absolute_diff,
                        'relative_diff': relative_diff,
                        'improved': (absolute_diff > 0 and metric_name not in ['rmse', 'mae', 'mape']) or 
                                    (absolute_diff < 0 and metric_name in ['rmse', 'mae', 'mape'])
                    }
                else:
                    # One version is missing this metric
                    result['comparisons'][metric_name] = {
                        'current_value': current_value,
                        'other_value': other_value,
                        'note': 'Metric not available in both versions'
                    }
            
            return result
        
        except Exception as e:
            logger.error(f"Error comparing with version {other_version}: {e}")
            return {
                'error': str(e),
                'model_name': self.model_name,
                'current_version': self.version
            }
    
    def get_avg_inference_time(self) -> float:
        """Get the average inference time in milliseconds."""
        with self._lock:
            if not self.inference_times:
                return 0.0
            return sum(self.inference_times) / len(self.inference_times) * 1000  # Convert to ms
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get the confusion matrix for classification models."""
        if self.model_type != 'classification' or not self.predictions:
            return None
        
        with self._lock:
            try:
                y_pred = np.array(list(self.predictions))
                y_true = np.array(list(self.actuals))
                
                # Handle multi-class vs binary classification
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Multi-class with probability outputs
                    y_pred_classes = np.argmax(y_pred, axis=1)
                else:
                    # Binary classification
                    y_pred_classes = (y_pred > 0.5).astype(int)
                
                return confusion_matrix(y_true, y_pred_classes)
            except Exception as e:
                logger.error(f"Error calculating confusion matrix: {e}")
                return None


class ModelDriftDetector:
    """
    Detects changes in model performance that may indicate drift.
    
    This class monitors model metrics over time and raises alerts when
    significant changes are detected that may indicate model drift.
    """
    
    def __init__(
        self, 
        performance_tracker: ModelPerformanceTracker,
        window_size: int = 10,
        threshold_multiplier: float = 2.0,
        min_samples: int = 50
    ):
        """
        Initialize the model drift detector.
        
        Args:
            performance_tracker: Reference to the associated performance tracker
            window_size: Number of metric points to use for drift detection
            threshold_multiplier: How many standard deviations constitute drift
            min_samples: Minimum samples needed before drift detection
        """
        self.tracker = performance_tracker
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.min_samples = min_samples
        
        # Baseline metrics (for comparing against)
        self.baseline_metrics = {}
        self.baseline_std = {}
        self.established = False
        
        # Drift history
        self.drift_events = []
    
    def establish_baseline(self):
        """Establish a baseline for drift detection."""
        metrics_history = self.tracker.get_all_metric_history()
        
        # Check if we have enough data
        for metric_name, history in metrics_history.items():
            if len(history) < self.min_samples:
                logger.info(f"Not enough data to establish baseline for {metric_name}")
                continue
            
            # Extract values for this metric
            values = [entry['value'] for entry in history[-self.min_samples:]]
            
            # Store baseline mean and std
            self.baseline_metrics[metric_name] = np.mean(values)
            self.baseline_std[metric_name] = np.std(values)
        
        if self.baseline_metrics:
            self.established = True
            logger.info("Established performance baseline for drift detection")
    
    def check_drift(self, current_metrics: Dict[str, float]) -> bool:
        """
        Check if current metrics indicate model drift.
        
        Args:
            current_metrics: Dictionary of current metric values
            
        Returns:
            True if drift detected, False otherwise
        """
        # If we don't have a baseline yet, try to establish one
        if not self.established:
            self.establish_baseline()
            return False
        
        drift_detected = False
        drift_details = {}
        
        # Check each metric for drift
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                std = max(self.baseline_std[metric_name], 1e-10)  # Avoid division by zero
                
                # Calculate z-score (deviation from baseline in terms of std)
                z_score = abs(current_value - baseline) / std
                
                # Check if beyond threshold
                if z_score > self.threshold_multiplier:
                    drift_details[metric_name] = {
                        'baseline': baseline,
                        'current': current_value,
                        'z_score': z_score,
                        'threshold': self.threshold_multiplier
                    }
                    drift_detected = True
        
        if drift_detected:
            drift_event = {
                'timestamp': datetime.now(),
                'details': drift_details
            }
            self.drift_events.append(drift_event)
            
            # Log the drift event
            logger.warning(f"Model drift detected for {self.tracker.model_name} - details: {drift_details}")
        
        return drift_detected
    
    def get_drift_events(self) -> List[Dict[str, Any]]:
        """Get the history of detected drift events."""
        return self.drift_events
    
    def reset_baseline(self):
        """Reset the baseline to establish a new one."""
        self.baseline_metrics = {}
        self.baseline_std = {}
        self.established = False
        self.establish_baseline()


class PerformanceVisualizer:
    """
    Generates visualizations for model performance metrics.
    
    This class creates various plots and charts to visualize model 
    performance metrics, feature importances, and drift events.
    """
    
    def __init__(self, performance_tracker: ModelPerformanceTracker):
        """
        Initialize the performance visualizer.
        
        Args:
            performance_tracker: Reference to the associated performance tracker
        """
        self.tracker = performance_tracker
    
    def plot_metric_over_time(self, metric_name: str, save_path: str = None, show_plot: bool = False):
        """
        Plot a metric's value over time.
        
        Args:
            metric_name: Name of the metric to plot
            save_path: Path to save the plot (None for no saving)
            show_plot: Whether to display the plot
            
        Returns:
            Figure and axes objects
        """
        metric_history = self.tracker.get_metric_history(metric_name)
        
        if not metric_history:
            logger.warning(f"No history available for metric: {metric_name}")
            return None, None
        
        # Extract timestamps and values
        timestamps = [entry['timestamp'] for entry in metric_history]
        values = [entry['value'] for entry in metric_history]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, values, '-o', markersize=4)
        plt.title(f"{metric_name.upper()} over time - {self.tracker.model_name} (v{self.tracker.version})")
        plt.xlabel('Time')
        plt.ylabel(metric_name.upper())
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis for better readability
        plt.gcf().autofmt_xdate()
        
        # Add a trend line
        z = np.polyfit(range(len(values)), values, 1)
        p = np.poly1d(z)
        plt.plot(timestamps, p(range(len(values))), "r--", alpha=0.8)
        
        # Mark drift events if available
        drift_events = self.tracker.drift_detector.get_drift_events()
        drift_events_for_metric = [
            event for event in drift_events 
            if metric_name in event['details']
        ]
        
        if drift_events_for_metric:
            drift_timestamps = [event['timestamp'] for event in drift_events_for_metric]
            plt.scatter(drift_timestamps, [values[timestamps.index(ts)] if ts in timestamps else None for ts in drift_timestamps], 
                      color='red', s=100, marker='*', label='Drift Detected')
            plt.legend()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf(), plt.gca()
    
    def plot_all_metrics(self, save_dir: str = None, show_plots: bool = False):
        """
        Plot all metrics over time.
        
        Args:
            save_dir: Directory to save plots (None for no saving)
            show_plots: Whether to display the plots
            
        Returns:
            Dictionary mapping metric names to figure objects
        """
        metrics = self.tracker.aggregated_metrics.get('metrics', {}).keys()
        figures = {}
        
        for metric_name in metrics:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{self.tracker.model_name}_{metric_name}_trend.png")
            else:
                save_path = None
            
            fig, ax = self.plot_metric_over_time(metric_name, save_path, show_plots)
            if fig:
                figures[metric_name] = fig
        
        return figures
    
    def plot_feature_importance(self, top_n: int = 10, save_path: str = None, show_plot: bool = False):
        """
        Plot feature importance values.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot (None for no saving)
            show_plot: Whether to display the plot
            
        Returns:
            Figure and axes objects
        """
        feature_importances = self.tracker.get_feature_importances()
        
        if not feature_importances:
            logger.warning("No feature importance data available")
            return None, None
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top N features
        top_features = sorted_features[:top_n]
        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.title(f"Top {top_n} Feature Importances - {self.tracker.model_name}")
        plt.grid(True, linestyle='--', axis='x', alpha=0.7)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf(), plt.gca()
    
    def plot_confusion_matrix(self, save_path: str = None, show_plot: bool = False):
        """
        Plot confusion matrix for classification models.
        
        Args:
            save_path: Path to save the plot (None for no saving)
            show_plot: Whether to display the plot
            
        Returns:
            Figure and axes objects
        """
        if self.tracker.model_type != 'classification':
            logger.warning("Confusion matrix is only available for classification models")
            return None, None
        
        conf_matrix = self.tracker.get_confusion_matrix()
        
        if conf_matrix is None:
            logger.warning("No data available for confusion matrix")
            return None, None
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {self.tracker.model_name}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return plt.gcf(), plt.gca()
    
    def create_performance_dashboard(self, save_path: str = None, show_plots: bool = False):
        """
        Create a comprehensive performance dashboard with multiple plots.
        
        Args:
            save_path: Path to save the dashboard (None for no saving)
            show_plots: Whether to display the plots
            
        Returns:
            Figure object
        """
        metrics = list(self.tracker.aggregated_metrics.get('metrics', {}).keys())
        
        if not metrics:
            logger.warning("No metrics available for dashboard creation")
            return None
        
        # Create subplots based on number of metrics
        n_metrics = len(metrics)
        n_rows = (n_metrics + 1) // 2  # +1 for feature importance
        
        fig = plt.figure(figsize=(15, 5 * n_rows))
        
        # Plot each metric
        for i, metric_name in enumerate(metrics):
            ax = fig.add_subplot(n_rows, 2, i+1)
            
            metric_history = self.tracker.get_metric_history(metric_name)
            if metric_history:
                timestamps = [entry['timestamp'] for entry in metric_history]
                values = [entry['value'] for entry in metric_history]
                
                ax.plot(timestamps, values, '-o', markersize=3)
                ax.set_title(f"{metric_name.upper()} over time")
                ax.set_ylabel(metric_name.upper())
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Add feature importance plot if available
        if self.tracker.feature_importances:
            ax = fig.add_subplot(n_rows, 2, n_metrics+1)
            
            # Sort features by importance
            sorted_features = sorted(
                self.tracker.feature_importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Take top 10 features
            top_features = sorted_features[:10]
            feature_names = [item[0] for item in top_features]
            importance_values = [item[1] for item in top_features]
            
            ax.barh(range(len(feature_names)), importance_values, align='center')
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Importance')
            ax.set_title("Top Feature Importances")
            ax.grid(True, linestyle='--', axis='x', alpha=0.7)
        
        # Add confusion matrix if available
        if self.tracker.model_type == 'classification':
            conf_matrix = self.tracker.get_confusion_matrix()
            if conf_matrix is not None:
                ax = fig.add_subplot(n_rows, 2, n_metrics+2)
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
        
        plt.suptitle(f"Performance Dashboard - {self.tracker.model_name} (v{self.tracker.version})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save or show the dashboard
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return fig


class PerformanceDataManager:
    """
    Manages storage and retrieval of performance data across model versions.
    
    This class provides centralized access to performance data for all models
    and versions, enabling cross-model comparisons and bulk operations.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the performance data manager.
        
        Args:
            data_dir: Directory for performance data storage
        """
        self.data_dir = data_dir or PERFORMANCE_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
    
    def list_models(self) -> List[str]:
        """List all model names with tracked performance data."""
        return [
            d for d in os.listdir(self.data_dir) 
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a specific model."""
        model_dir = os.path.join(self.data_dir, model_name)
        
        if not os.path.exists(model_dir):
            return []
        
        # Extract version information from filenames
        version_pattern = f"{model_name}_v(.*?)_"
        versions = set()
        
        for filename in os.listdir(model_dir):
            if filename.startswith(f"{model_name}_v") and filename.endswith(".json"):
                import re
                match = re.search(version_pattern, filename)
                if match:
                    versions.add(match.group(1))
        
        return sorted(list(versions))
    
    def get_model_performance(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Get performance data for a specific model and version.
        
        Args:
            model_name: Name of the model
            version: Specific version (None for latest)
            
        Returns:
            Performance data dictionary
        """
        model_dir = os.path.join(self.data_dir, model_name)
        
        if not os.path.exists(model_dir):
            return {'error': f"No data found for model: {model_name}"}
        
        # Determine file to load
        if version:
            # Try to find the specific version's latest data
            candidate_files = [
                f for f in os.listdir(model_dir) 
                if f.startswith(f"{model_name}_v{version}_")
            ]
            
            if not candidate_files:
                return {'error': f"No data found for version: {version}"}
            
            # Use the most recent file for this version
            filepath = os.path.join(model_dir, sorted(candidate_files)[-1])
        else:
            # Use the latest overall file
            latest_path = os.path.join(model_dir, f"{model_name}_latest.json")
            
            if not os.path.exists(latest_path):
                # Fall back to most recent file
                all_files = [
                    f for f in os.listdir(model_dir)
                    if f.endswith(".json") and not f.endswith("_latest.json")
                ]
                
                if not all_files:
                    return {'error': f"No data files found for model: {model_name}"}
                
                filepath = os.path.join(model_dir, sorted(all_files)[-1])
            else:
                filepath = latest_path
        
        # Load the data
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {'error': f"Error loading performance data: {str(e)}"}
    
    def compare_models(self, model_names: List[str], metric_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare performance across different models.
        
        Args:
            model_names: List of model names to compare
            metric_names: Specific metrics to compare (None for all)
            
        Returns:
            Comparison data dictionary
        """
        result = {'models': {}, 'metrics': {}}
        
        # Fetch latest data for each model
        for model_name in model_names:
            model_data = self.get_model_performance(model_name)
            
            if 'error' in model_data:
                result['models'][model_name] = {'error': model_data['error']}
                continue
            
            # Extract core information
            model_info = {
                'version': model_data.get('version', 'unknown'),
                'model_type': model_data.get('model_type', 'unknown'),
                'last_updated': model_data.get('last_updated', 'unknown')
            }
            
            # Extract metrics
            metrics = model_data.get('aggregated_metrics', {}).get('metrics', {})
            model_info['metrics'] = metrics
            
            result['models'][model_name] = model_info
        
        # Compare metrics across models
        available_metrics = set()
        for model_name, model_info in result['models'].items():
            if 'metrics' in model_info:
                available_metrics.update(model_info['metrics'].keys())
        
        # Filter to requested metrics if specified
        if metric_names:
            metrics_to_compare = set(metric_names).intersection(available_metrics)
        else:
            metrics_to_compare = available_metrics
        
        # Build metric comparison data
        for metric in metrics_to_compare:
            metric_data = {
                'values': {},
                'best_model': None,
                'worst_model': None
            }
            
            valid_models = []
            
            for model_name, model_info in result['models'].items():
                if 'metrics' in model_info and metric in model_info['metrics']:
                    value = model_info['metrics'][metric]
                    metric_data['values'][model_name] = value
                    valid_models.append((model_name, value))
            
            # Determine best and worst (assumes higher is better except for error metrics)
            if valid_models:
                error_metrics = {'rmse', 'mae', 'mape'}
                reverse = metric.lower() in error_metrics
                
                sorted_models = sorted(valid_models, key=lambda x: x[1], reverse=not reverse)
                metric_data['best_model'] = sorted_models[0][0]
                metric_data['worst_model'] = sorted_models[-1][0]
            
            result['metrics'][metric] = metric_data
        
        return result
    
    def get_performance_trends(self, model_name: str, metric_names: List[str] = None) -> Dict[str, Any]:
        """
        Get performance trends for a model across all versions.
        
        Args:
            model_name: Name of the model
            metric_names: Specific metrics to include (None for all)
            
        Returns:
            Trend data dictionary
        """
        versions = self.list_versions(model_name)
        
        if not versions:
            return {'error': f"No versions found for model: {model_name}"}
        
        result = {
            'model_name': model_name,
            'versions': versions,
            'metrics': {}
        }
        
        # Determine metrics to track
        if not metric_names:
            # Get metrics from the latest version
            latest_data = self.get_model_performance(model_name)
            if 'error' not in latest_data:
                available_metrics = latest_data.get('aggregated_metrics', {}).get('metrics', {}).keys()
                metric_names = list(available_metrics)
            else:
                return {'error': "Could not determine available metrics"}
        
        # Collect metric values across versions
        for metric in metric_names:
            metric_values = []
            
            for version in versions:
                version_data = self.get_model_performance(model_name, version)
                
                if 'error' not in version_data:
                    metric_value = version_data.get('aggregated_metrics', {}).get('metrics', {}).get(metric)
                    if metric_value is not None:
                        metric_values.append({
                            'version': version,
                            'value': metric_value
                        })
            
            result['metrics'][metric] = metric_values
        
        return result


def create_performance_tracker(
    model_name: str,
    model_type: str = 'classification',
    version: str = None,
    metrics: Set[str] = None
) -> ModelPerformanceTracker:
    """
    Factory function to create a performance tracker.
    
    Args:
        model_name: Name of the model
        model_type: Type of model ('classification' or 'regression')
        version: Specific version (None for auto-generated)
        metrics: Set of metrics to track (None for default)
        
    Returns:
        Configured ModelPerformanceTracker
    """
    tracker = ModelPerformanceTracker(
        model_name=model_name,
        model_type=model_type,
        version=version,
        metrics_to_track=metrics
    )
    
    return tracker


def evaluate_classification_model(
    model_name: str,
    y_pred: Union[np.ndarray, List],
    y_true: Union[np.ndarray, List],
    version: str = None,
    feature_importances: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Evaluate a classification model and track its performance.
    
    Args:
        model_name: Name of the model
        y_pred: Predicted values (probabilities or classes)
        y_true: Actual/true values
        version: Model version
        feature_importances: Feature importance values
        
    Returns:
        Dictionary of performance metrics
    """
    # Create tracker
    tracker = create_performance_tracker(
        model_name=model_name,
        model_type='classification',
        version=version
    )
    
    # Convert inputs to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Record the prediction batch
    tracker.record_prediction(y_pred, y_true)
    
    # Set feature importances if provided
    if feature_importances:
        tracker.set_feature_importances(feature_importances)
    
    # Return the current metrics
    return tracker.get_current_metrics()


def evaluate_regression_model(
    model_name: str,
    y_pred: Union[np.ndarray, List],
    y_true: Union[np.ndarray, List],
    version: str = None,
    feature_importances: Dict[str, float] = None
) -> Dict[str, float]:
    """
    Evaluate a regression model and track its performance.
    
    Args:
        model_name: Name of the model
        y_pred: Predicted values
        y_true: Actual/true values
        version: Model version
        feature_importances: Feature importance values
        
    Returns:
        Dictionary of performance metrics
    """
    # Create tracker
    tracker = create_performance_tracker(
        model_name=model_name,
        model_type='regression',
        version=version
    )
    
    # Convert inputs to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Record the prediction batch
    tracker.record_prediction(y_pred, y_true)
    
    # Set feature importances if provided
    if feature_importances:
        tracker.set_feature_importances(feature_importances)
    
    # Return the current metrics
    return tracker.get_current_metrics()


def create_simple_dashboard(
    model_name: str,
    save_path: str = None,
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Create a simple performance dashboard for a model.
    
    Args:
        model_name: Name of the model
        save_path: Path to save the dashboard image
        show_plots: Whether to display the plots
        
    Returns:
        Dictionary with dashboard information
    """
    # Get the latest data
    data_manager = PerformanceDataManager()
    model_data = data_manager.get_model_performance(model_name)
    
    if 'error' in model_data:
        return model_data
    
    # Create a temporary tracker with the loaded data
    tracker = ModelPerformanceTracker(
        model_name=model_data.get('model_name', model_name),
        model_type=model_data.get('model_type', 'classification'),
        version=model_data.get('version', 'latest'),
        persistence_enabled=False
    )
    
    # Initialize with existing data
    if 'batch_metrics' in model_data:
        tracker.batch_metrics = model_data['batch_metrics']
    
    if 'aggregated_metrics' in model_data:
        tracker.aggregated_metrics = model_data['aggregated_metrics']
    
    if 'feature_importances' in model_data:
        tracker.feature_importances = model_data['feature_importances']
    
    # Create visualizer and dashboard
    visualizer = PerformanceVisualizer(tracker)
    dashboard = visualizer.create_performance_dashboard(save_path, show_plots)
    
    return {
        'model_name': model_name,
        'version': model_data.get('version', 'latest'),
        'dashboard_created': dashboard is not None,
        'save_path': save_path
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Performance Tracking and Evaluation')
    parser.add_argument('--model', type=str, help='Model name to analyze')
    parser.add_argument('--dashboard', action='store_true', help='Create performance dashboard')
    parser.add_argument('--list-models', action='store_true', help='List all tracked models')
    parser.add_argument('--compare', nargs='+', help='Compare models (space-separated list)')
    
    args = parser.parse_args()
    
    if args.list_models:
        data_manager = PerformanceDataManager()
        models = data_manager.list_models()
        
        print("Available models:")
        for model in models:
            versions = data_manager.list_versions(model)
            print(f"- {model} (versions: {', '.join(versions)})")
    
    elif args.model and args.dashboard:
        result = create_simple_dashboard(
            model_name=args.model,
            save_path=f"{args.model}_dashboard.png",
            show_plots=True
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Dashboard created for {args.model} v{result['version']}")
    
    elif args.compare:
        data_manager = PerformanceDataManager()
        comparison = data_manager.compare_models(args.compare)
        
        print("Model Comparison:")
        for model_name, model_info in comparison['models'].items():
            if 'error' in model_info:
                print(f"- {model_name}: {model_info['error']}")
            else:
                print(f"- {model_name} (v{model_info['version']}):")
                for metric, value in model_info.get('metrics', {}).items():
                    print(f"  - {metric}: {value}")
        
        print("\nMetric Comparison:")
        for metric, metric_data in comparison['metrics'].items():
            print(f"- {metric}:")
            print(f"  - Best: {metric_data['best_model']} ({metric_data['values'].get(metric_data['best_model'], 'N/A')})")
            print(f"  - Worst: {metric_data['worst_model']} ({metric_data['values'].get(metric_data['worst_model'], 'N/A')})")
    
    else:
        print("Please specify an action. See --help for options.")