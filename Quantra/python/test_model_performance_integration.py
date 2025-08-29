#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the ML Model Performance Integration Module

This module tests the integration with existing ML components including:
- Enhanced model wrapper functionality
- Enhanced ensemble model functionality
- Enhanced inference pipeline functionality
"""

import unittest
import os
import json
import shutil
import numpy as np
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

# Check if dependencies are available
DEPENDENCIES_AVAILABLE = False
try:
    from ensemble_learning import ModelWrapper, EnsembleModel
    from real_time_inference import RealTimeInferencePipeline
    from model_performance_tracking import ModelPerformanceTracker
    from model_performance_integration import (
        EnhancedModelWrapper,
        EnhancedEnsembleModel,
        EnhancedRealTimeInferencePipeline,
        enhance_model_wrapper,
        enhance_ensemble_model,
        enhance_inference_pipeline
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Integration test dependencies not available: {e}")


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Integration dependencies not available")
class TestEnhancedModelWrapper(unittest.TestCase):
    """Test the EnhancedModelWrapper class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock model
        self.mock_model = MockModel()
        
        # Create enhanced wrapper
        self.wrapper = EnhancedModelWrapper(
            model=self.mock_model,
            name="test_model",
            model_type="classification",
            version="test_version"
        )
    
    def test_initialization(self):
        """Test enhanced wrapper initialization."""
        self.assertEqual(self.wrapper.model, self.mock_model)
        self.assertEqual(self.wrapper.name, "test_model")
        self.assertEqual(self.wrapper.model_type, "classification")
        self.assertEqual(self.wrapper.version, "test_version")
        self.assertIsNotNone(self.wrapper.performance_tracker)
        self.assertEqual(self.wrapper.prediction_history, [])
    
    def test_predict(self):
        """Test the predict method with tracking."""
        # Make a prediction
        result = self.wrapper.predict(np.array([1, 2, 3]))
        
        # Check that the mock model was called
        self.assertTrue(self.mock_model.predict_called)
        
        # Check that prediction was added to history
        self.assertEqual(len(self.wrapper.prediction_history), 1)
        self.assertIn('timestamp', self.wrapper.prediction_history[0])
        self.assertIn('features', self.wrapper.prediction_history[0])
        self.assertIn('prediction', self.wrapper.prediction_history[0])
    
    def test_evaluate_performance(self):
        """Test performance evaluation."""
        # Make predictions
        self.wrapper.predict(np.array([1, 2, 3]))
        self.wrapper.predict(np.array([4, 5, 6]))
        
        # Set mock model to return known values
        self.mock_model.prediction_value = 1
        
        # Evaluate with known true values
        y_true = np.array([1, 0])  # Second prediction is wrong
        metrics = self.wrapper.evaluate_performance(y_true)
        
        # Check that metrics were returned
        self.assertIn('model_name', metrics)
        self.assertEqual(metrics['model_name'], "test_model")


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Integration dependencies not available")
class TestEnhancedEnsembleModel(unittest.TestCase):
    """Test the EnhancedEnsembleModel class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create mock models
        self.mock_model1 = MockModel()
        self.mock_model2 = MockModel()
        
        # Create model wrappers
        self.wrapper1 = EnhancedModelWrapper(self.mock_model1, "model1", "classification")
        self.wrapper2 = EnhancedModelWrapper(self.mock_model2, "model2", "classification")
        
        # Create enhanced ensemble
        self.ensemble = EnhancedEnsembleModel(
            models=[self.wrapper1, self.wrapper2],
            weights=[0.6, 0.4],
            task_type="classification",
            name="test_ensemble",
            version="test_version"
        )
    
    def test_initialization(self):
        """Test enhanced ensemble initialization."""
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertEqual(self.ensemble.name, "test_ensemble")
        self.assertEqual(self.ensemble.version, "test_version")
        self.assertIsNotNone(self.ensemble.performance_tracker)
    
    @patch('ensemble_learning.EnsembleModel.evaluate')
    def test_evaluate(self, mock_evaluate):
        """Test the evaluate method with tracking."""
        # Set up mock to return a result structure
        mock_evaluate.return_value = {
            'ensemble': {'accuracy': 0.85},
            'models': {
                'model1': {'accuracy': 0.8},
                'model2': {'accuracy': 0.7}
            }
        }
        
        # Evaluate with test data
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        result = self.ensemble.evaluate(X, y)
        
        # Check that parent method was called
        mock_evaluate.assert_called_once()
        
        # Check that tracking was added
        self.assertIn('tracking', result)
        self.assertIn('ensemble', result['tracking'])
        self.assertIn('models', result['tracking'])


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Integration dependencies not available")
class TestEnhancedInferencePipeline(unittest.TestCase):
    """Test the EnhancedRealTimeInferencePipeline class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create enhanced pipeline
        self.pipeline = EnhancedRealTimeInferencePipeline(
            model_types=["test_model"],
            enable_monitoring=True,
            enable_tracking=True,
            name="test_pipeline"
        )
    
    def test_initialization(self):
        """Test enhanced pipeline initialization."""
        self.assertEqual(self.pipeline.model_types, ["test_model"])
        self.assertEqual(self.pipeline.name, "test_pipeline")
        self.assertTrue(self.pipeline.enable_tracking)
        self.assertIn("test_model", self.pipeline.model_trackers)
    
    @patch('real_time_inference.RealTimeInferencePipeline._process_request')
    def test_process_request(self, mock_process):
        """Test the _process_request method with tracking."""
        # Create a test request
        request = {
            'id': 'test_id',
            'model_type': 'test_model',
            'market_data': {'close': 100},
            'ground_truth': 1  # This is needed for tracking
        }
        
        # Add a mock result to the cache
        self.pipeline.result_cache['test_id_test_model'] = {
            'prediction': {
                'action': 'BUY',
                'confidence': 0.8,
                'inference_time_ms': 10
            },
            'cached_at': datetime.now()
        }
        
        # Process the request
        self.pipeline._process_request(request)
        
        # Check that parent method was called
        mock_process.assert_called_once()


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Integration dependencies not available")
class TestHelperFunctions(unittest.TestCase):
    """Test helper functions in the integration module."""
    
    def test_enhance_model_wrapper(self):
        """Test enhancing a standard model wrapper."""
        # Create a standard wrapper
        standard = ModelWrapper(MockModel(), "test_model")
        
        # Enhance it
        enhanced = enhance_model_wrapper(standard, "classification", "test_version")
        
        # Check that it was enhanced
        self.assertIsInstance(enhanced, EnhancedModelWrapper)
        self.assertEqual(enhanced.name, "test_model")
        self.assertEqual(enhanced.model_type, "classification")
        self.assertEqual(enhanced.version, "test_version")
    
    @patch('model_performance_integration.enhance_model_wrapper')
    def test_enhance_ensemble_model(self, mock_enhance):
        """Test enhancing a standard ensemble model."""
        # Mock the enhance_model_wrapper function
        mock_enhance.side_effect = lambda model, *args, **kwargs: model
        
        # Create a standard ensemble
        model1 = ModelWrapper(MockModel(), "model1")
        model2 = ModelWrapper(MockModel(), "model2")
        standard = EnsembleModel([model1, model2], task_type="classification")
        
        # Enhance it
        enhanced = enhance_ensemble_model(standard, "test_ensemble", "test_version")
        
        # Check that it was enhanced
        self.assertIsInstance(enhanced, EnhancedEnsembleModel)
        self.assertEqual(enhanced.name, "test_ensemble")
        self.assertEqual(enhanced.version, "test_version")
        
        # Check that enhance_model_wrapper was called for each model
        self.assertEqual(mock_enhance.call_count, 2)
    
    def test_enhance_inference_pipeline(self):
        """Test enhancing a standard inference pipeline."""
        # Create a standard pipeline
        standard = RealTimeInferencePipeline(["test_model"])
        
        # Enhance it
        enhanced = enhance_inference_pipeline(standard, "test_pipeline")
        
        # Check that it was enhanced
        self.assertIsInstance(enhanced, EnhancedRealTimeInferencePipeline)
        self.assertEqual(enhanced.name, "test_pipeline")
        self.assertEqual(enhanced.model_types, ["test_model"])


class MockModel:
    """Mock model class for testing."""
    
    def __init__(self):
        self.predict_called = False
        self.prediction_value = 1
    
    def predict(self, X):
        """Mock predict method."""
        self.predict_called = True
        return np.ones(X.shape[0]) * self.prediction_value


if __name__ == '__main__':
    unittest.main()