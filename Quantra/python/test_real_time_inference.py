#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the Real-Time Inference Pipeline

This module tests the real-time ML inference functionality including:
- Basic inference operations
- Performance monitoring
- Model caching
- Asynchronous processing
- Error handling
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch
import numpy as np

# Import the module to test
from real_time_inference import (
    RealTimeInferencePipeline,
    PerformanceMonitor,
    ModelCache,
    MockModel,
    create_inference_pipeline
)


class TestPerformanceMonitor(unittest.TestCase):
    """Test the PerformanceMonitor class."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor(max_history=100)
    
    def test_record_inference_success(self):
        """Test recording successful inference operations."""
        self.monitor.record_inference(0.05, success=True)
        self.monitor.record_inference(0.03, success=True)
        
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics['total_predictions'], 2)
        self.assertEqual(metrics['total_errors'], 0)
        self.assertGreater(metrics['avg_inference_time_ms'], 0)
    
    def test_record_inference_failure(self):
        """Test recording failed inference operations."""
        self.monitor.record_inference(0.05, success=True)
        self.monitor.record_inference(0.10, success=False)
        
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics['total_predictions'], 2)
        self.assertEqual(metrics['total_errors'], 1)
        self.assertEqual(metrics['error_rate'], 0.5)
    
    def test_metrics_calculation(self):
        """Test metrics calculation with multiple data points."""
        # Record multiple inferences
        inference_times = [0.01, 0.02, 0.05, 0.03, 0.08]
        for t in inference_times:
            self.monitor.record_inference(t, success=True)
        
        metrics = self.monitor.get_metrics()
        expected_avg = np.mean(inference_times) * 1000  # Convert to ms
        self.assertAlmostEqual(metrics['avg_inference_time_ms'], expected_avg, places=2)
        self.assertEqual(metrics['predictions_per_minute'], 5)


class TestModelCache(unittest.TestCase):
    """Test the ModelCache class."""
    
    def setUp(self):
        self.cache = ModelCache(max_cache_size=3)
    
    def test_cache_model_loading(self):
        """Test loading and caching models."""
        # Load a model
        model = self.cache.get_model('test_model')
        self.assertIsNotNone(model)
        
        # Verify it's cached
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['cache_size'], 1)
        self.assertIn('test_model', stats['cached_models'])
    
    def test_cache_eviction(self):
        """Test cache eviction when max size is reached."""
        # Fill cache to max capacity
        for i in range(4):  # One more than max_cache_size
            self.cache.get_model(f'model_{i}')
        
        stats = self.cache.get_cache_stats()
        self.assertEqual(stats['cache_size'], 3)  # Should not exceed max
    
    def test_model_access_tracking(self):
        """Test that model access times are tracked."""
        model1 = self.cache.get_model('model_1')
        time.sleep(0.01)  # Small delay
        model2 = self.cache.get_model('model_2')
        
        # Access model1 again to update its access time
        model1_again = self.cache.get_model('model_1')
        
        # Should be the same object
        self.assertIs(model1, model1_again)


class TestMockModel(unittest.TestCase):
    """Test the MockModel class."""
    
    def setUp(self):
        self.model = MockModel('test_model')
    
    def test_mock_prediction(self):
        """Test mock model prediction generation."""
        features = {'close': 150.0, 'volume': 1000000}
        prediction = self.model.predict(features)
        
        # Check required fields
        self.assertIn('action', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('predicted_price', prediction)
        self.assertIn('model_type', prediction)
        
        # Check value ranges
        self.assertIn(prediction['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(prediction['confidence'], 0.6)
        self.assertLessEqual(prediction['confidence'], 0.95)
        self.assertEqual(prediction['model_type'], 'test_model')


class TestRealTimeInferencePipeline(unittest.TestCase):
    """Test the main RealTimeInferencePipeline class."""
    
    def setUp(self):
        self.pipeline = RealTimeInferencePipeline(
            model_types=['test_model'],
            max_queue_size=100,
            prediction_timeout=1.0,
            enable_monitoring=True
        )
    
    def tearDown(self):
        if self.pipeline.running:
            self.pipeline.stop()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertFalse(self.pipeline.running)
        self.assertEqual(self.pipeline.model_types, ['test_model'])
        self.assertIsNotNone(self.pipeline.model_cache)
        self.assertIsNotNone(self.pipeline.monitor)
    
    def test_pipeline_start_stop(self):
        """Test starting and stopping the pipeline."""
        self.pipeline.start()
        self.assertTrue(self.pipeline.running)
        self.assertGreater(len(self.pipeline.worker_threads), 0)
        
        self.pipeline.stop()
        self.assertFalse(self.pipeline.running)
    
    def test_synchronous_prediction(self):
        """Test synchronous prediction."""
        self.pipeline.start()
        
        market_data = {
            'close': 150.0,
            'open': 148.0,
            'high': 152.0,
            'low': 147.0,
            'volume': 1000000
        }
        
        result = self.pipeline.predict_sync(market_data, timeout=2.0)
        
        self.assertIsNotNone(result)
        self.assertIn('action', result)
        self.assertIn('confidence', result)
        self.assertIn('inference_time_ms', result)
    
    def test_asynchronous_prediction(self):
        """Test asynchronous prediction."""
        self.pipeline.start()
        
        market_data = {
            'close': 150.0,
            'volume': 1000000
        }
        
        result_container = {}
        event = threading.Event()
        
        def callback(prediction):
            result_container['prediction'] = prediction
            event.set()
        
        request_id = self.pipeline.predict_async(market_data, callback=callback)
        
        self.assertIsNotNone(request_id)
        
        # Wait for result
        self.assertTrue(event.wait(timeout=2.0))
        self.assertIn('prediction', result_container)
        
        prediction = result_container['prediction']
        self.assertIn('action', prediction)
        self.assertIn('confidence', prediction)
    
    def test_feature_preparation(self):
        """Test feature preparation from market data."""
        market_data = {
            'close': 150.0,
            'open': 148.0,
            'high': 152.0,
            'low': 147.0,
            'volume': 1000000,
            'rsi': 65.0
        }
        
        features = self.pipeline._prepare_features(market_data)
        
        self.assertEqual(features['close'], 150.0)
        self.assertEqual(features['open'], 148.0)
        self.assertEqual(features['volume'], 1000000)
        self.assertEqual(features['rsi'], 65.0)
    
    def test_feature_preparation_fallback(self):
        """Test feature preparation with missing data."""
        market_data = {}  # Empty data
        
        features = self.pipeline._prepare_features(market_data)
        
        # Should provide fallback values
        self.assertIn('close', features)
        self.assertIn('volume', features)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        self.pipeline.start()
        
        # Generate some predictions to create metrics
        market_data = {'close': 150.0, 'volume': 1000000}
        self.pipeline.predict_sync(market_data)
        
        metrics = self.pipeline.get_performance_metrics()
        
        self.assertIn('inference', metrics)
        self.assertIn('cache', metrics)
        self.assertIn('queue_size', metrics)
        self.assertIn('pipeline_status', metrics)
        
        self.assertEqual(metrics['pipeline_status'], 'running')
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        # Add some cache entries
        self.pipeline.result_cache['test_key'] = {
            'prediction': {'action': 'BUY'},
            'cached_at': time.time() - 10  # Expired entry
        }
        
        self.pipeline.result_cache['fresh_key'] = {
            'prediction': {'action': 'SELL'},
            'cached_at': time.time()  # Fresh entry
        }
        
        # Cleanup should remove expired entries
        self.pipeline.cleanup_cache()
        
        self.assertNotIn('test_key', self.pipeline.result_cache)
        self.assertIn('fresh_key', self.pipeline.result_cache)
    
    def test_queue_full_handling(self):
        """Test handling when request queue is full."""
        # Create a pipeline with very small queue
        small_pipeline = RealTimeInferencePipeline(max_queue_size=1)
        
        try:
            # Don't start the pipeline so requests won't be processed
            market_data = {'close': 150.0}
            
            # First request should succeed
            request_id1 = small_pipeline.predict_async(market_data)
            self.assertIsNotNone(request_id1)
            
            # Second request should fail (queue full)
            request_id2 = small_pipeline.predict_async(market_data)
            self.assertIsNone(request_id2)
        
        finally:
            small_pipeline.stop()


class TestInferencePipelineIntegration(unittest.TestCase):
    """Integration tests for the inference pipeline."""
    
    def test_create_inference_pipeline_factory(self):
        """Test the factory function for creating pipelines."""
        config = {
            'model_types': ['ensemble', 'random_forest'],
            'max_queue_size': 500,
            'prediction_timeout': 0.2,
            'enable_monitoring': False
        }
        
        pipeline = create_inference_pipeline(config)
        
        self.assertEqual(pipeline.model_types, ['ensemble', 'random_forest'])
        self.assertEqual(pipeline.max_queue_size, 500)
        self.assertEqual(pipeline.prediction_timeout, 0.2)
        self.assertIsNone(pipeline.monitor)  # Monitoring disabled
    
    def test_create_inference_pipeline_defaults(self):
        """Test factory function with default configuration."""
        pipeline = create_inference_pipeline()
        
        self.assertEqual(pipeline.model_types, ['auto'])
        self.assertEqual(pipeline.max_queue_size, 1000)
        self.assertEqual(pipeline.prediction_timeout, 0.1)
        self.assertIsNotNone(pipeline.monitor)  # Monitoring enabled by default
    
    def test_concurrent_predictions(self):
        """Test multiple concurrent prediction requests."""
        pipeline = create_inference_pipeline()
        pipeline.start()
        
        try:
            results = []
            events = []
            
            def make_prediction(i):
                event = threading.Event()
                events.append(event)
                
                def callback(prediction):
                    results.append((i, prediction))
                    event.set()
                
                market_data = {
                    'close': 100.0 + i,
                    'volume': 1000000 + i * 1000
                }
                
                pipeline.predict_async(market_data, callback=callback)
            
            # Submit multiple concurrent requests
            threads = []
            for i in range(5):
                thread = threading.Thread(target=make_prediction, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Wait for all predictions to complete
            for event in events:
                self.assertTrue(event.wait(timeout=3.0))
            
            # Verify we got all results
            self.assertEqual(len(results), 5)
            
            # Verify all predictions have required fields
            for i, prediction in results:
                self.assertIn('action', prediction)
                self.assertIn('confidence', prediction)
        
        finally:
            pipeline.stop()


class TestErrorHandling(unittest.TestCase):
    """Test error handling in the inference pipeline."""
    
    def test_invalid_market_data(self):
        """Test handling of invalid market data."""
        pipeline = RealTimeInferencePipeline()
        pipeline.start()
        
        try:
            # Test with invalid data types
            invalid_data = {
                'close': 'invalid_string',
                'volume': 'not_a_number'
            }
            
            result = pipeline.predict_sync(invalid_data)
            
            # Should get a fallback prediction with error
            self.assertIsNotNone(result)
            self.assertEqual(result['action'], 'HOLD')
            self.assertIn('error', result)
        
        finally:
            pipeline.stop()
    
    def test_prediction_timeout(self):
        """Test handling of prediction timeouts."""
        pipeline = RealTimeInferencePipeline(prediction_timeout=0.001)  # Very short timeout
        
        # Don't start the pipeline so predictions will timeout
        market_data = {'close': 150.0}
        
        result = pipeline.predict_sync(market_data, timeout=0.001)
        self.assertIsNone(result)  # Should timeout


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)