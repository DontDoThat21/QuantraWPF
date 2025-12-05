#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-Time Model Inference Pipeline

This module provides a real-time inference pipeline for ML models, designed to:
- Process live market data with minimal latency
- Deliver instant predictions 
- Enable rapid response to dynamic market conditions
- Maintain high throughput for trading applications

Key Features:
- Model caching for fast inference
- Streaming data processing
- Performance monitoring
- Graceful error handling
- Hot model swapping capabilities
"""

import sys
import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import queue
import numpy as np
import pandas as pd

# Import debugpy utilities for remote debugging support
try:
    from debugpy_utils import init_debugpy_if_enabled
    DEBUGPY_UTILS_AVAILABLE = True
except ImportError:
    DEBUGPY_UTILS_AVAILABLE = False

# Import existing ML modules
try:
    from stock_predictor import predict_stock, load_or_train_model
    from feature_engineering import build_default_pipeline, FeatureEngineer
    from ensemble_learning import EnsembleModel
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some ML dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('real_time_inference')


class PerformanceMonitor:
    """Monitor and track inference pipeline performance metrics."""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.inference_times = deque(maxlen=max_history)
        self.prediction_counts = deque(maxlen=max_history)
        self.error_counts = deque(maxlen=max_history)
        self.start_time = time.time()
        self.total_predictions = 0
        self.total_errors = 0
        self._lock = threading.Lock()
    
    def record_inference(self, inference_time: float, success: bool = True):
        """Record an inference operation."""
        with self._lock:
            current_time = time.time()
            self.inference_times.append((current_time, inference_time))
            self.total_predictions += 1
            
            if not success:
                self.total_errors += 1
                self.error_counts.append(current_time)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        with self._lock:
            now = time.time()
            recent_inferences = [t for timestamp, t in self.inference_times 
                               if now - timestamp < 60]  # Last minute
            
            recent_errors = [t for t in self.error_counts 
                           if now - t < 60]  # Last minute
            
            return {
                'avg_inference_time_ms': np.mean(recent_inferences) * 1000 if recent_inferences else 0,
                'p95_inference_time_ms': np.percentile(recent_inferences, 95) * 1000 if recent_inferences else 0,
                'predictions_per_minute': len(recent_inferences),
                'error_rate': len(recent_errors) / max(len(recent_inferences), 1),
                'total_predictions': self.total_predictions,
                'total_errors': self.total_errors,
                'uptime_seconds': now - self.start_time
            }


class ModelCache:
    """Cache for loaded ML models to enable fast inference."""
    
    def __init__(self, max_cache_size=5):
        self.max_cache_size = max_cache_size
        self.models = {}
        self.model_metadata = {}
        self.access_times = {}
        self._lock = threading.Lock()
    
    def get_model(self, model_key: str):
        """Get a model from cache or load it."""
        with self._lock:
            if model_key in self.models:
                self.access_times[model_key] = time.time()
                return self.models[model_key]
            
            # Model not in cache, need to load
            return self._load_model(model_key)
    
    def _load_model(self, model_key: str):
        """Load a model and add it to cache."""
        try:
            # Evict oldest model if cache is full
            if len(self.models) >= self.max_cache_size:
                oldest_key = min(self.access_times.keys(), key=self.access_times.get)
                self._evict_model(oldest_key)
            
            # Load the model (simplified - in practice this would load from specific paths)
            if DEPENDENCIES_AVAILABLE:
                model = load_or_train_model(model_type=model_key)
                self.models[model_key] = model
                self.model_metadata[model_key] = {
                    'loaded_at': time.time(),
                    'model_type': model_key
                }
                self.access_times[model_key] = time.time()
                logger.info(f"Loaded model {model_key} into cache")
                return model
            else:
                # Fallback mock model for testing
                mock_model = MockModel(model_key)
                self.models[model_key] = mock_model
                self.access_times[model_key] = time.time()
                return mock_model
                
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}")
            return None
    
    def _evict_model(self, model_key: str):
        """Remove a model from cache."""
        if model_key in self.models:
            del self.models[model_key]
            del self.model_metadata[model_key]
            del self.access_times[model_key]
            logger.info(f"Evicted model {model_key} from cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cached_models': list(self.models.keys()),
                'cache_size': len(self.models),
                'max_cache_size': self.max_cache_size,
                'model_metadata': self.model_metadata.copy()
            }


class MockModel:
    """Mock model for testing when real models aren't available."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.created_at = time.time()
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate mock prediction."""
        return {
            'action': np.random.choice(['BUY', 'SELL', 'HOLD']),
            'confidence': np.random.uniform(0.6, 0.95),
            'predicted_price': features.get('close', 100.0) * np.random.uniform(0.95, 1.05),
            'model_type': self.model_type,
            'prediction_time': time.time()
        }


class RealTimeInferencePipeline:
    """
    Main real-time inference pipeline that processes streaming market data
    and delivers fast ML predictions.
    """
    
    def __init__(self, 
                 model_types: List[str] = None,
                 max_queue_size: int = 1000,
                 prediction_timeout: float = 0.1,
                 enable_monitoring: bool = True):
        """
        Initialize the real-time inference pipeline.
        
        Args:
            model_types: List of model types to support (e.g., ['random_forest', 'ensemble'])
            max_queue_size: Maximum size of the input data queue
            prediction_timeout: Maximum time to wait for a prediction (seconds)
            enable_monitoring: Whether to enable performance monitoring
        """
        self.model_types = model_types or ['auto']
        self.max_queue_size = max_queue_size
        self.prediction_timeout = prediction_timeout
        
        # Initialize components
        self.model_cache = ModelCache()
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        
        # Data processing queue
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_cache = {}
        self.cache_ttl = 5.0  # Cache results for 5 seconds
        
        # Threading control
        self.running = False
        self.worker_threads = []
        self.num_workers = 2
        
        # Feature engineering pipeline
        self.feature_pipeline = None
        self._initialize_feature_pipeline()
        
        logger.info(f"Initialized RealTimeInferencePipeline with models: {self.model_types}")
    
    def _initialize_feature_pipeline(self):
        """Initialize the feature engineering pipeline."""
        try:
            if DEPENDENCIES_AVAILABLE:
                self.feature_pipeline = build_default_pipeline()
                logger.info("Feature engineering pipeline initialized")
            else:
                logger.warning("Feature engineering not available, using basic features")
        except Exception as e:
            logger.error(f"Failed to initialize feature pipeline: {e}")
    
    def start(self):
        """Start the real-time inference pipeline."""
        if self.running:
            logger.warning("Pipeline is already running")
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"InferenceWorker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started real-time inference pipeline with {self.num_workers} workers")
    
    def stop(self):
        """Stop the real-time inference pipeline."""
        self.running = False
        
        # Signal workers to stop
        for _ in self.worker_threads:
            try:
                self.input_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=2.0)
        
        self.worker_threads.clear()
        logger.info("Stopped real-time inference pipeline")
    
    def _worker_loop(self):
        """Main worker loop for processing inference requests."""
        while self.running:
            try:
                # Get next request from queue
                request = self.input_queue.get(timeout=1.0)
                
                if request is None:  # Shutdown signal
                    break
                
                # Process the request
                self._process_request(request)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                if self.monitor:
                    self.monitor.record_inference(0, success=False)
    
    def _process_request(self, request: Dict[str, Any]):
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            request_id = request.get('id', 'unknown')
            market_data = request.get('market_data', {})
            model_type = request.get('model_type', 'auto')
            callback = request.get('callback')
            
            # Generate prediction
            prediction = self._generate_prediction(market_data, model_type)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            # Add timing information to prediction
            prediction['inference_time_ms'] = inference_time * 1000
            prediction['request_id'] = request_id
            prediction['timestamp'] = datetime.now().isoformat()
            
            # Cache the result
            cache_key = f"{request_id}_{model_type}"
            self.result_cache[cache_key] = {
                'prediction': prediction,
                'cached_at': time.time()
            }
            
            # Call callback if provided
            if callback:
                callback(prediction)
            
            # Record performance metrics
            if self.monitor:
                self.monitor.record_inference(inference_time, success=True)
            
            logger.debug(f"Processed request {request_id} in {inference_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            if self.monitor:
                self.monitor.record_inference(time.time() - start_time, success=False)
    
    def _generate_prediction(self, market_data: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Generate a prediction for the given market data."""
        try:
            # Prepare features
            features = self._prepare_features(market_data)
            
            # Get model from cache
            model = self.model_cache.get_model(model_type)
            if model is None:
                raise ValueError(f"Could not load model: {model_type}")
            
            # Generate prediction
            if hasattr(model, 'predict') and callable(model.predict):
                prediction = model.predict(features)
            elif DEPENDENCIES_AVAILABLE:
                # Use the existing predict_stock function
                prediction = predict_stock(
                    features, 
                    model_type=model_type,
                    use_feature_engineering=True
                )
            else:
                # Fallback to mock prediction
                prediction = MockModel(model_type).predict(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            # Return safe fallback prediction
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'error': str(e),
                'model_type': model_type
            }
    
    def _prepare_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Prepare features from raw market data."""
        try:
            # Convert market data to features
            features = {}
            
            # Basic price features
            if 'close' in market_data:
                features['close'] = float(market_data['close'])
            if 'open' in market_data:
                features['open'] = float(market_data['open'])
            if 'high' in market_data:
                features['high'] = float(market_data['high'])
            if 'low' in market_data:
                features['low'] = float(market_data['low'])
            if 'volume' in market_data:
                features['volume'] = float(market_data['volume'])
            
            # Add technical indicators if available
            if 'rsi' in market_data:
                features['rsi'] = float(market_data['rsi'])
            if 'macd' in market_data:
                features['macd'] = float(market_data['macd'])
            if 'bb_upper' in market_data:
                features['bb_upper'] = float(market_data['bb_upper'])
            if 'bb_lower' in market_data:
                features['bb_lower'] = float(market_data['bb_lower'])
            
            # If no features available, create basic ones
            if not features:
                features = {'close': 100.0, 'volume': 1000000}
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return {'close': 100.0, 'volume': 1000000}  # Safe fallback
    
    def predict_async(self, 
                     market_data: Dict[str, Any], 
                     model_type: str = 'auto',
                     callback: Callable = None) -> Optional[str]:
        """
        Submit a prediction request asynchronously.
        
        Args:
            market_data: Current market data
            model_type: Type of model to use for prediction
            callback: Function to call with results
            
        Returns:
            Request ID for tracking, or None if queue is full
        """
        try:
            request_id = f"req_{int(time.time()*1000000)}"
            
            request = {
                'id': request_id,
                'market_data': market_data,
                'model_type': model_type,
                'callback': callback,
                'submitted_at': time.time()
            }
            
            self.input_queue.put(request, timeout=0.1)
            return request_id
            
        except queue.Full:
            logger.warning("Inference queue is full, dropping request")
            return None
        except Exception as e:
            logger.error(f"Error submitting async prediction: {e}")
            return None
    
    def predict_sync(self, 
                    market_data: Dict[str, Any], 
                    model_type: str = 'auto',
                    timeout: float = None) -> Optional[Dict[str, Any]]:
        """
        Get a prediction synchronously.
        
        Args:
            market_data: Current market data
            model_type: Type of model to use for prediction
            timeout: Maximum time to wait for result
            
        Returns:
            Prediction result or None if timeout
        """
        if timeout is None:
            timeout = self.prediction_timeout
        
        result_container = {}
        event = threading.Event()
        
        def callback(prediction):
            result_container['prediction'] = prediction
            event.set()
        
        request_id = self.predict_async(market_data, model_type, callback)
        if request_id is None:
            return None
        
        # Wait for result
        if event.wait(timeout):
            return result_container.get('prediction')
        else:
            logger.warning(f"Prediction timeout for request {request_id}")
            return None
    
    def get_cached_prediction(self, request_id: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Get a cached prediction result."""
        cache_key = f"{request_id}_{model_type}"
        cached = self.result_cache.get(cache_key)
        
        if cached:
            # Check if cache is still valid
            if time.time() - cached['cached_at'] < self.cache_ttl:
                return cached['prediction']
            else:
                # Cache expired, remove it
                del self.result_cache[cache_key]
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {}
        
        if self.monitor:
            metrics['inference'] = self.monitor.get_metrics()
        
        metrics['cache'] = self.model_cache.get_cache_stats()
        metrics['queue_size'] = self.input_queue.qsize()
        metrics['pipeline_status'] = 'running' if self.running else 'stopped'
        
        return metrics
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, cached in self.result_cache.items()
            if current_time - cached['cached_at'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.result_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")


def create_inference_pipeline(config: Dict[str, Any] = None) -> RealTimeInferencePipeline:
    """
    Factory function to create a configured inference pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured RealTimeInferencePipeline instance
    """
    if config is None:
        config = {}
    
    pipeline = RealTimeInferencePipeline(
        model_types=config.get('model_types', ['auto']),
        max_queue_size=config.get('max_queue_size', 1000),
        prediction_timeout=config.get('prediction_timeout', 0.1),
        enable_monitoring=config.get('enable_monitoring', True)
    )
    
    return pipeline


def main():
    """Main function for testing the real-time inference pipeline."""
    # Initialize debugpy remote debugging if DEBUGPY environment variable is set
    if DEBUGPY_UTILS_AVAILABLE:
        init_debugpy_if_enabled()
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time ML Inference Pipeline')
    parser.add_argument('--test', action='store_true', help='Run test scenarios')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    # Create pipeline
    pipeline = create_inference_pipeline(config)
    
    try:
        if args.test:
            run_test_scenarios(pipeline)
        elif args.benchmark:
            run_benchmark(pipeline)
        else:
            # Run interactive mode
            run_interactive_mode(pipeline)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        pipeline.stop()


def run_test_scenarios(pipeline: RealTimeInferencePipeline):
    """Run test scenarios for the inference pipeline."""
    logger.info("Running test scenarios...")
    
    pipeline.start()
    
    # Test basic prediction
    test_data = {
        'close': 150.0,
        'open': 148.0,
        'high': 152.0,
        'low': 147.0,
        'volume': 1000000,
        'rsi': 65.0
    }
    
    print("Testing synchronous prediction...")
    result = pipeline.predict_sync(test_data)
    if result:
        print(f"Prediction: {result['action']} (confidence: {result['confidence']:.2f})")
        print(f"Inference time: {result.get('inference_time_ms', 0):.2f}ms")
    else:
        print("Failed to get prediction")
    
    # Test async prediction
    print("\nTesting asynchronous prediction...")
    results = []
    
    def callback(prediction):
        results.append(prediction)
        print(f"Async result: {prediction['action']} (confidence: {prediction['confidence']:.2f})")
    
    request_id = pipeline.predict_async(test_data, callback=callback)
    if request_id:
        print(f"Submitted async request: {request_id}")
        time.sleep(0.5)  # Wait for result
    
    # Test performance metrics
    print("\nPerformance metrics:")
    metrics = pipeline.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def run_benchmark(pipeline: RealTimeInferencePipeline):
    """Run performance benchmark."""
    logger.info("Running performance benchmark...")
    
    pipeline.start()
    
    # Generate test data
    test_requests = []
    for i in range(100):
        test_data = {
            'close': 100.0 + np.random.normal(0, 5),
            'volume': 1000000 + np.random.randint(-100000, 100000),
            'rsi': np.random.uniform(20, 80)
        }
        test_requests.append(test_data)
    
    # Benchmark synchronous predictions
    print("Benchmarking synchronous predictions...")
    start_time = time.time()
    
    for i, test_data in enumerate(test_requests):
        result = pipeline.predict_sync(test_data)
        if i % 20 == 0:
            print(f"Processed {i+1}/100 requests...")
    
    sync_time = time.time() - start_time
    print(f"Sync benchmark: {len(test_requests)} predictions in {sync_time:.2f}s")
    print(f"Average: {sync_time/len(test_requests)*1000:.2f}ms per prediction")
    
    # Final metrics
    print("\nFinal performance metrics:")
    metrics = pipeline.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def run_interactive_mode(pipeline: RealTimeInferencePipeline):
    """Run interactive mode for manual testing."""
    logger.info("Starting interactive mode...")
    pipeline.start()
    
    print("Real-Time ML Inference Pipeline")
    print("Commands: predict, metrics, status, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'predict':
                # Get market data from user
                try:
                    close = float(input("Close price: "))
                    volume = float(input("Volume (or press Enter for default): ") or "1000000")
                    
                    market_data = {'close': close, 'volume': volume}
                    
                    result = pipeline.predict_sync(market_data)
                    if result:
                        print(f"Prediction: {result['action']}")
                        print(f"Confidence: {result['confidence']:.2f}")
                        print(f"Inference time: {result.get('inference_time_ms', 0):.2f}ms")
                    else:
                        print("Failed to get prediction")
                
                except ValueError:
                    print("Invalid input")
            
            elif command == 'metrics':
                metrics = pipeline.get_performance_metrics()
                print("Performance Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            elif command == 'status':
                print(f"Pipeline status: {'running' if pipeline.running else 'stopped'}")
                print(f"Queue size: {pipeline.input_queue.qsize()}")
                print(f"Worker threads: {len(pipeline.worker_threads)}")
            
            else:
                print("Unknown command")
        
        except EOFError:
            break
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()