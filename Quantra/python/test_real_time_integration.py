#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the real-time inference integration with C# service simulation.
"""

import unittest
import json
import subprocess
import time
import threading
from unittest.mock import Mock, patch


class TestRealTimeInferenceIntegration(unittest.TestCase):
    """Test integration scenarios for real-time inference."""
    
    def test_basic_inference_pipeline(self):
        """Test the basic inference pipeline functionality."""
        from real_time_inference import create_inference_pipeline
        
        # Create and start pipeline
        pipeline = create_inference_pipeline({
            'model_types': ['auto'],
            'max_queue_size': 100,
            'prediction_timeout': 1.0
        })
        
        pipeline.start()
        
        try:
            # Test market data similar to what C# would send
            market_data = {
                'close': 150.50,
                'open': 149.75,
                'high': 151.25,
                'low': 149.25,
                'volume': 2500000,
                'symbol': 'AAPL'
            }
            
            # Get synchronous prediction
            result = pipeline.predict_sync(market_data, timeout=2.0)
            
            # Verify result structure
            self.assertIsNotNone(result)
            self.assertIn('action', result)
            self.assertIn('confidence', result)
            self.assertIn('inference_time_ms', result)
            
            # Verify action is valid
            self.assertIn(result['action'], ['BUY', 'SELL', 'HOLD'])
            
            # Verify confidence is in valid range
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)
            
            # Verify inference time is reasonable (less than 1 second)
            self.assertLess(result['inference_time_ms'], 1000.0)
            
        finally:
            pipeline.stop()
    
    def test_performance_under_load(self):
        """Test performance under multiple concurrent requests."""
        from real_time_inference import create_inference_pipeline
        
        pipeline = create_inference_pipeline({
            'model_types': ['auto'],
            'max_queue_size': 500,
            'prediction_timeout': 0.5
        })
        
        pipeline.start()
        
        try:
            import concurrent.futures
            
            def make_prediction(symbol_id):
                market_data = {
                    'close': 100.0 + symbol_id,
                    'volume': 1000000 + symbol_id * 10000,
                    'symbol': f'TEST{symbol_id:03d}'
                }
                
                start_time = time.time()
                result = pipeline.predict_sync(market_data, timeout=2.0)
                end_time = time.time()
                
                return {
                    'result': result,
                    'latency': (end_time - start_time) * 1000,
                    'symbol_id': symbol_id
                }
            
            # Submit 20 concurrent predictions
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_prediction, i) for i in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify all predictions completed
            self.assertEqual(len(results), 20)
            
            # Verify all have valid results
            for test_result in results:
                self.assertIsNotNone(test_result['result'])
                self.assertIn('action', test_result['result'])
                
                # Verify latency is reasonable (less than 2 seconds)
                self.assertLess(test_result['latency'], 2000.0)
            
            # Get performance metrics
            metrics = pipeline.get_performance_metrics()
            self.assertGreaterEqual(metrics['inference']['total_predictions'], 20)
            
        finally:
            pipeline.stop()
    
    def test_error_handling(self):
        """Test error handling in the pipeline."""
        from real_time_inference import create_inference_pipeline
        
        pipeline = create_inference_pipeline()
        pipeline.start()
        
        try:
            # Test with invalid data
            invalid_data = {
                'close': 'invalid_string',
                'volume': None
            }
            
            result = pipeline.predict_sync(invalid_data, timeout=2.0)
            
            # Should get a fallback result, not crash
            self.assertIsNotNone(result)
            self.assertEqual(result['action'], 'HOLD')  # Fallback action
            
            # Test with empty data
            empty_data = {}
            result = pipeline.predict_sync(empty_data, timeout=2.0)
            
            # Should still get a result
            self.assertIsNotNone(result)
            self.assertIn('action', result)
            
        finally:
            pipeline.stop()
    
    def test_model_caching(self):
        """Test model caching functionality."""
        from real_time_inference import ModelCache
        
        cache = ModelCache(max_cache_size=2)
        
        # Load first model
        model1 = cache.get_model('test_model_1')
        self.assertIsNotNone(model1)
        
        # Load second model
        model2 = cache.get_model('test_model_2')
        self.assertIsNotNone(model2)
        
        # Cache should have 2 models
        stats = cache.get_cache_stats()
        self.assertEqual(stats['cache_size'], 2)
        
        # Load third model (should evict oldest)
        model3 = cache.get_model('test_model_3')
        self.assertIsNotNone(model3)
        
        # Cache should still have 2 models (evicted oldest)
        stats = cache.get_cache_stats()
        self.assertEqual(stats['cache_size'], 2)
    
    def test_feature_preparation(self):
        """Test feature preparation from various market data formats."""
        from real_time_inference import RealTimeInferencePipeline
        
        pipeline = RealTimeInferencePipeline()
        
        # Test with complete data
        complete_data = {
            'close': 150.0,
            'open': 149.0,
            'high': 151.0,
            'low': 148.5,
            'volume': 1000000,
            'rsi': 65.0,
            'macd': 1.5
        }
        
        features = pipeline._prepare_features(complete_data)
        
        self.assertEqual(features['close'], 150.0)
        self.assertEqual(features['volume'], 1000000)
        self.assertEqual(features['rsi'], 65.0)
        self.assertEqual(features['macd'], 1.5)
        
        # Test with minimal data
        minimal_data = {'close': 100.0}
        
        features = pipeline._prepare_features(minimal_data)
        
        self.assertEqual(features['close'], 100.0)
        # Note: volume may not be added automatically unless specified in fallback logic
        
        # Test with no data (should provide safe fallbacks)
        features = pipeline._prepare_features({})
        
        self.assertIn('close', features)
        self.assertIn('volume', features)
    
    def test_metrics_collection(self):
        """Test performance metrics collection."""
        from real_time_inference import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # Record some inference operations
        monitor.record_inference(0.05, success=True)
        monitor.record_inference(0.03, success=True)
        monitor.record_inference(0.10, success=False)
        
        metrics = monitor.get_metrics()
        
        self.assertEqual(metrics['total_predictions'], 3)
        self.assertEqual(metrics['total_errors'], 1)
        self.assertAlmostEqual(metrics['error_rate'], 1/3, places=2)
        self.assertGreater(metrics['avg_inference_time_ms'], 0)


def run_integration_tests():
    """Run all integration tests."""
    print("Running real-time inference integration tests...")
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRealTimeInferenceIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    if success:
        print("\nAll integration tests passed!")
    else:
        print("\nSome integration tests failed!")
        exit(1)