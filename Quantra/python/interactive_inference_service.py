#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Real-Time Inference Service

This script runs as an interactive service that receives commands from C#
and provides real-time ML inference responses. It maintains a persistent
inference pipeline for low-latency predictions.
"""

import sys
import json
import logging
import signal
import threading
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from real_time_inference import RealTimeInferencePipeline, create_inference_pipeline
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    logging.warning("Real-time inference module not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('interactive_inference')


class InteractiveInferenceService:
    """Interactive service for real-time ML inference."""
    
    def __init__(self):
        self.pipeline = None
        self.running = True
        self.command_handlers = {
            'initialize': self._handle_initialize,
            'predict_sync': self._handle_predict_sync,
            'predict_async': self._handle_predict_async,
            'metrics': self._handle_metrics,
            'health_check': self._handle_health_check,
            'shutdown': self._handle_shutdown
        }
    
    def start(self):
        """Start the interactive service."""
        logger.info("Starting interactive inference service...")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            self._main_loop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self._cleanup()
    
    def _main_loop(self):
        """Main loop to process commands from stdin."""
        while self.running:
            try:
                # Read command from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse command
                try:
                    command_data = json.loads(line)
                    command = command_data.get('command', '')
                    
                    if command in self.command_handlers:
                        response = self.command_handlers[command](command_data)
                    else:
                        response = {
                            'status': 'error',
                            'error': f'Unknown command: {command}',
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Send response to stdout
                    response_json = json.dumps(response)
                    print(response_json, flush=True)
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        'status': 'error',
                        'error': f'Invalid JSON: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }
                    print(json.dumps(error_response), flush=True)
                    
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    def _handle_initialize(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the inference pipeline."""
        try:
            config = command_data.get('config', {})
            
            if INFERENCE_AVAILABLE:
                self.pipeline = create_inference_pipeline(config)
                self.pipeline.start()
                
                return {
                    'status': 'success',
                    'message': 'Pipeline initialized successfully',
                    'config': config,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'error': 'Inference module not available',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_predict_sync(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronous prediction request."""
        try:
            if not self.pipeline:
                return {
                    'status': 'error',
                    'error': 'Pipeline not initialized',
                    'timestamp': datetime.now().isoformat()
                }
            
            request_id = command_data.get('request_id', 'unknown')
            market_data = command_data.get('market_data', {})
            model_type = command_data.get('model_type', 'auto')
            timeout = command_data.get('timeout', 5.0)
            
            # Get prediction
            prediction = self.pipeline.predict_sync(
                market_data=market_data,
                model_type=model_type,
                timeout=timeout
            )
            
            if prediction:
                # Convert to the format expected by C#
                response = {
                    'status': 'success',
                    'request_id': request_id,
                    'symbol': market_data.get('symbol', 'UNKNOWN'),
                    'action': prediction.get('action', 'HOLD'),
                    'confidence': prediction.get('confidence', 0.5),
                    'current_price': prediction.get('currentPrice', market_data.get('close', 0)),
                    'predicted_price': prediction.get('predictedPrice', market_data.get('close', 0)),
                    'inference_time_ms': prediction.get('inference_time_ms', 0),
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type
                }
                
                # Add risk metrics if available
                if 'risk' in prediction:
                    response['risk'] = prediction['risk']
                
                return response
            else:
                return {
                    'status': 'error',
                    'error': 'Prediction timeout or failed',
                    'request_id': request_id,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'request_id': command_data.get('request_id', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_predict_async(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle asynchronous prediction request."""
        try:
            if not self.pipeline:
                return {
                    'status': 'error',
                    'error': 'Pipeline not initialized',
                    'timestamp': datetime.now().isoformat()
                }
            
            request_id = command_data.get('request_id', 'unknown')
            market_data = command_data.get('market_data', {})
            model_type = command_data.get('model_type', 'auto')
            
            def callback(prediction):
                # This would need to be handled differently in a real implementation
                # For now, we'll just log it
                logger.info(f"Async prediction complete for {request_id}: {prediction.get('action')}")
            
            submitted_id = self.pipeline.predict_async(
                market_data=market_data,
                model_type=model_type,
                callback=callback
            )
            
            if submitted_id:
                return {
                    'status': 'success',
                    'message': 'Async prediction submitted',
                    'request_id': request_id,
                    'submitted_id': submitted_id,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'error': 'Failed to submit async prediction',
                    'request_id': request_id,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'request_id': command_data.get('request_id', 'unknown'),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_metrics(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            if not self.pipeline:
                return {
                    'status': 'error',
                    'error': 'Pipeline not initialized',
                    'timestamp': datetime.now().isoformat()
                }
            
            metrics = self.pipeline.get_performance_metrics()
            
            return {
                'status': 'success',
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_health_check(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check."""
        try:
            is_healthy = True
            errors = []
            
            if not INFERENCE_AVAILABLE:
                is_healthy = False
                errors.append("Inference module not available")
            
            if not self.pipeline:
                is_healthy = False
                errors.append("Pipeline not initialized")
            elif not self.pipeline.running:
                is_healthy = False
                errors.append("Pipeline not running")
            
            return {
                'status': 'success',
                'healthy': is_healthy,
                'errors': errors,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'healthy': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_shutdown(self, command_data: Dict[str, Any]) -> Dict[str, Any]:
        """Shutdown the service."""
        try:
            self.running = False
            
            return {
                'status': 'success',
                'message': 'Shutdown initiated',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _cleanup(self):
        """Clean up resources."""
        try:
            if self.pipeline:
                self.pipeline.stop()
                logger.info("Pipeline stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # If arguments provided, run the original real_time_inference.py
        from real_time_inference import main as original_main
        original_main()
    else:
        # Run as interactive service
        service = InteractiveInferenceService()
        service.start()


if __name__ == "__main__":
    main()