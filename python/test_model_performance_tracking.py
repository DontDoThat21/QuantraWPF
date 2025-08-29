#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the ML Model Performance Tracking and Evaluation Framework.

This module tests the functionality of the model performance tracking framework including:
- Performance metric collection and tracking
- Historical performance comparison
- Model drift detection
- Visualization functionality
- Performance data management
"""

import unittest
import os
import json
import shutil
import numpy as np
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the module to test
from model_performance_tracking import (
    ModelPerformanceTracker,
    ModelDriftDetector,
    PerformanceVisualizer,
    PerformanceDataManager,
    evaluate_classification_model,
    evaluate_regression_model
)


class TestModelPerformanceTracker(unittest.TestCase):
    """Test the ModelPerformanceTracker class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the PERFORMANCE_DATA_DIR to use the temporary directory
        self.patcher = patch('model_performance_tracking.PERFORMANCE_DATA_DIR', self.test_dir)
        self.mock_data_dir = self.patcher.start()
        
        # Create a test tracker
        self.tracker = ModelPerformanceTracker(
            model_name="test_model",
            model_type="classification",
            version="test_version",
            persistence_enabled=True
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.patcher.stop()
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.model_name, "test_model")
        self.assertEqual(self.tracker.model_type, "classification")
        self.assertEqual(self.tracker.version, "test_version")
        self.assertTrue(self.tracker.persistence_enabled)
        
        # Check default metrics for classification
        expected_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        self.assertEqual(self.tracker.metrics_to_track, expected_metrics)
    
    def test_record_prediction(self):
        """Test recording predictions and actuals."""
        # Record some test predictions
        y_pred = np.array([1, 0, 1, 1, 0])
        y_true = np.array([1, 0, 0, 1, 1])
        self.tracker.record_prediction(y_pred, y_true, inference_time=0.05)
        
        # Check that data was stored
        self.assertEqual(len(self.tracker.predictions), 5)
        self.assertEqual(len(self.tracker.actuals), 5)
        self.assertEqual(len(self.tracker.timestamps), 5)
        self.assertEqual(len(self.tracker.inference_times), 1)
    
    def test_update_metrics(self):
        """Test metrics update."""
        # Record predictions that will trigger metrics update
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # 10 predictions (trigger threshold)
        y_true = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
        self.tracker.record_prediction(y_pred, y_true)
        
        # Check that metrics were calculated
        self.assertIn('metrics', self.tracker.aggregated_metrics)
        metrics = self.tracker.aggregated_metrics['metrics']
        self.assertIn('accuracy', metrics)
        self.assertEqual(metrics['accuracy'], 0.7)  # 7/10 correct predictions
    
    def test_get_current_metrics(self):
        """Test retrieving current metrics."""
        # Add some test data
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_true = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
        self.tracker.record_prediction(y_pred, y_true)
        
        # Get current metrics
        metrics = self.tracker.get_current_metrics()
        
        # Check structure
        self.assertEqual(metrics['model_name'], "test_model")
        self.assertEqual(metrics['model_type'], "classification")
        self.assertEqual(metrics['version'], "test_version")
        self.assertIn('metrics', metrics)
        self.assertIn('sample_count', metrics)
    
    def test_feature_importances(self):
        """Test setting and getting feature importances."""
        # Define test feature importances
        importances = {
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2
        }
        
        # Set feature importances
        self.tracker.set_feature_importances(importances)
        
        # Get feature importances
        retrieved = self.tracker.get_feature_importances()
        
        # Check that they match
        self.assertEqual(retrieved, importances)
    
    def test_metrics_persistence(self):
        """Test that metrics are persisted to disk."""
        # Add some test data
        y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_true = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0])
        self.tracker.record_prediction(y_pred, y_true)
        
        # Force persistence
        self.tracker._persist_metrics()
        
        # Check that a file was created
        model_dir = os.path.join(self.test_dir, "test_model")
        self.assertTrue(os.path.exists(model_dir))
        
        # Check that the latest file exists
        latest_file = os.path.join(model_dir, "test_model_latest.json")
        self.assertTrue(os.path.exists(latest_file))
        
        # Check file content
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['model_name'], "test_model")
        self.assertEqual(data['version'], "test_version")
        self.assertIn('aggregated_metrics', data)
        self.assertIn('metrics', data['aggregated_metrics'])


class TestModelDriftDetector(unittest.TestCase):
    """Test the ModelDriftDetector class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock tracker
        self.mock_tracker = MagicMock()
        self.mock_tracker.model_name = "test_model"
        
        # Create drift detector
        self.detector = ModelDriftDetector(
            self.mock_tracker,
            window_size=5,
            threshold_multiplier=2.0,
            min_samples=5
        )
    
    def test_initialization(self):
        """Test drift detector initialization."""
        self.assertEqual(self.detector.tracker, self.mock_tracker)
        self.assertEqual(self.detector.window_size, 5)
        self.assertEqual(self.detector.threshold_multiplier, 2.0)
        self.assertEqual(self.detector.min_samples, 5)
        self.assertFalse(self.detector.established)
        self.assertEqual(self.detector.drift_events, [])
    
    def test_establish_baseline(self):
        """Test establishing a baseline for drift detection."""
        # Mock metric history
        metric_history = {
            'accuracy': [{'value': 0.8, 'timestamp': datetime.now()}] * 10
        }
        self.mock_tracker.get_all_metric_history.return_value = metric_history
        
        # Establish baseline
        self.detector.establish_baseline()
        
        # Check that baseline was set
        self.assertTrue(self.detector.established)
        self.assertIn('accuracy', self.detector.baseline_metrics)
        self.assertEqual(self.detector.baseline_metrics['accuracy'], 0.8)
    
    def test_drift_detection(self):
        """Test detecting drift in metrics."""
        # Set up baseline
        self.detector.established = True
        self.detector.baseline_metrics = {'accuracy': 0.8}
        self.detector.baseline_std = {'accuracy': 0.05}
        
        # Test with normal value (within threshold)
        no_drift = self.detector.check_drift({'accuracy': 0.78})
        self.assertFalse(no_drift)
        self.assertEqual(len(self.detector.drift_events), 0)
        
        # Test with drift value (beyond threshold)
        drift = self.detector.check_drift({'accuracy': 0.6})  # More than 2 std devs
        self.assertTrue(drift)
        self.assertEqual(len(self.detector.drift_events), 1)
        
        # Check drift event details
        event = self.detector.drift_events[0]
        self.assertIn('timestamp', event)
        self.assertIn('details', event)
        self.assertIn('accuracy', event['details'])


class TestPerformanceVisualizer(unittest.TestCase):
    """Test the PerformanceVisualizer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test output
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock tracker with test data
        self.mock_tracker = MagicMock()
        self.mock_tracker.model_name = "test_model"
        self.mock_tracker.model_type = "classification"
        self.mock_tracker.version = "test_version"
        
        # Mock metric history with 10 data points
        timestamps = [datetime.now() for _ in range(10)]
        values = [0.75 + 0.01 * i for i in range(10)]  # Increasing values
        
        self.mock_tracker.get_metric_history.return_value = [
            {'timestamp': ts, 'value': val} for ts, val in zip(timestamps, values)
        ]
        
        # Mock feature importances
        self.mock_tracker.get_feature_importances.return_value = {
            f"feature{i}": 0.1 * (10 - i) for i in range(10)
        }
        
        # Create visualizer
        self.visualizer = PerformanceVisualizer(self.mock_tracker)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_metric_over_time(self, mock_close, mock_savefig):
        """Test plotting a metric over time."""
        # Create a test plot
        save_path = os.path.join(self.test_dir, "test_plot.png")
        fig, ax = self.visualizer.plot_metric_over_time("accuracy", save_path)
        
        # Check that plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        mock_savefig.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_feature_importance(self, mock_close, mock_savefig):
        """Test plotting feature importances."""
        # Create a test plot
        save_path = os.path.join(self.test_dir, "feature_plot.png")
        fig, ax = self.visualizer.plot_feature_importance(top_n=5, save_path=save_path)
        
        # Check that plot was created
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        mock_savefig.assert_called_once()


class TestPerformanceDataManager(unittest.TestCase):
    """Test the PerformanceDataManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create data manager with test directory
        self.data_manager = PerformanceDataManager(data_dir=self.test_dir)
        
        # Create test model data
        self.model_dir = os.path.join(self.test_dir, "test_model")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create test data files
        test_data = {
            "model_name": "test_model",
            "model_type": "classification",
            "version": "v1",
            "last_updated": datetime.now().isoformat(),
            "aggregated_metrics": {
                "metrics": {
                    "accuracy": 0.85,
                    "precision": 0.83,
                    "recall": 0.82
                },
                "sample_count": 100
            }
        }
        
        with open(os.path.join(self.model_dir, "test_model_v1_202201.json"), 'w') as f:
            json.dump(test_data, f)
        
        with open(os.path.join(self.model_dir, "test_model_latest.json"), 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.test_dir)
    
    def test_list_models(self):
        """Test listing available models."""
        models = self.data_manager.list_models()
        self.assertEqual(len(models), 1)
        self.assertIn("test_model", models)
    
    def test_list_versions(self):
        """Test listing available versions for a model."""
        versions = self.data_manager.list_versions("test_model")
        self.assertEqual(len(versions), 1)
        self.assertIn("v1", versions)
    
    def test_get_model_performance(self):
        """Test retrieving model performance data."""
        # Test getting latest data
        data = self.data_manager.get_model_performance("test_model")
        self.assertNotIn("error", data)
        self.assertEqual(data["model_name"], "test_model")
        self.assertEqual(data["version"], "v1")
        
        # Test getting specific version
        data = self.data_manager.get_model_performance("test_model", "v1")
        self.assertNotIn("error", data)
        self.assertEqual(data["model_name"], "test_model")
        
        # Test getting non-existent version
        data = self.data_manager.get_model_performance("test_model", "v2")
        self.assertIn("error", data)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions in the module."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.patcher = patch('model_performance_tracking.PERFORMANCE_DATA_DIR', self.test_dir)
        self.mock_data_dir = self.patcher.start()
    
    def tearDown(self):
        """Clean up after each test."""
        self.patcher.stop()
        shutil.rmtree(self.test_dir)
    
    @patch('model_performance_tracking.ModelPerformanceTracker._update_metrics')
    @patch('model_performance_tracking.ModelPerformanceTracker._persist_metrics')
    def test_evaluate_classification_model(self, mock_persist, mock_update):
        """Test helper function for evaluating classification models."""
        # Define test data
        y_pred = np.array([1, 0, 1, 1, 0])
        y_true = np.array([1, 0, 0, 1, 1])
        feature_importances = {"feature1": 0.5, "feature2": 0.3}
        
        # Call the function
        metrics = evaluate_classification_model(
            model_name="test_classification",
            y_pred=y_pred,
            y_true=y_true,
            feature_importances=feature_importances
        )
        
        # Check that it returns the expected structure
        self.assertIn('model_name', metrics)
        self.assertEqual(metrics['model_name'], "test_classification")
        self.assertEqual(metrics['model_type'], "classification")
    
    @patch('model_performance_tracking.ModelPerformanceTracker._update_metrics')
    @patch('model_performance_tracking.ModelPerformanceTracker._persist_metrics')
    def test_evaluate_regression_model(self, mock_persist, mock_update):
        """Test helper function for evaluating regression models."""
        # Define test data
        y_pred = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
        feature_importances = {"feature1": 0.5, "feature2": 0.3}
        
        # Call the function
        metrics = evaluate_regression_model(
            model_name="test_regression",
            y_pred=y_pred,
            y_true=y_true,
            feature_importances=feature_importances
        )
        
        # Check that it returns the expected structure
        self.assertIn('model_name', metrics)
        self.assertEqual(metrics['model_name'], "test_regression")
        self.assertEqual(metrics['model_type'], "regression")


if __name__ == '__main__':
    unittest.main()