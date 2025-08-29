#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model Performance Integration Module

This module integrates the model performance tracking and evaluation framework
with existing ML infrastructure including ensemble learning and real-time inference.
It provides utility functions for connecting different components and ensuring
performance data flows through the entire ML pipeline.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Union, Optional
from datetime import datetime

# Import existing ML modules
try:
    from ensemble_learning import ModelWrapper, EnsembleModel
    from real_time_inference import RealTimeInferencePipeline, PerformanceMonitor
    from model_performance_tracking import (
        ModelPerformanceTracker, 
        PerformanceVisualizer,
        PerformanceDataManager,
        create_performance_tracker
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some ML dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_performance_integration')


class EnhancedModelWrapper(ModelWrapper):
    """
    Enhanced version of ModelWrapper that integrates with performance tracking.
    
    This class extends the existing ModelWrapper to include comprehensive
    performance tracking and evaluation capabilities.
    """
    
    def __init__(self, model, name, model_type='classification', version=None, metadata=None):
        """
        Initialize the enhanced model wrapper.
        
        Args:
            model: The underlying model object
            name: Name identifier for this model
            model_type: Type of model ('classification' or 'regression')
            version: Version identifier for the model
            metadata: Additional metadata for the model
        """
        # Initialize the parent ModelWrapper
        super().__init__(model, name, metadata)
        
        # Set additional properties
        self.model_type = model_type
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create performance tracker
        self.performance_tracker = create_performance_tracker(
            model_name=name,
            model_type=model_type,
            version=self.version
        )
        
        # Track historical predictions in addition to performance metrics
        self.prediction_history = []
        
        logger.info(f"Created enhanced model wrapper for {name} (v{self.version})")
    
    def predict(self, X):
        """
        Make a prediction and track its performance.
        
        Args:
            X: Features to predict on
            
        Returns:
            Model predictions
        """
        # Get the original prediction
        prediction = super().predict(X)
        
        # Store prediction for future evaluation
        self.prediction_history.append({
            'timestamp': datetime.now(),
            'features': X,
            'prediction': prediction
        })
        
        return prediction
    
    def evaluate_performance(self, y_true, y_pred=None):
        """
        Evaluate and record model performance.
        
        Args:
            y_true: Actual/true values
            y_pred: Predicted values (uses last predictions if None)
            
        Returns:
            Dictionary of performance metrics
        """
        if y_pred is None and not self.prediction_history:
            logger.warning("No predictions available for evaluation")
            return {}
        
        # Use provided predictions or get from history
        if y_pred is None:
            # Use the most recent predictions
            preds = [entry['prediction'] for entry in self.prediction_history[-len(y_true):]]
            y_pred = np.array(preds)
        
        # Record predictions in the tracker
        self.performance_tracker.record_prediction(y_pred, y_true)
        
        # Extract feature importances if available
        feature_importances = {}
        if hasattr(self.model, 'feature_importances_'):
            # For sklearn-like models
            feature_importances = {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}
            self.performance_tracker.set_feature_importances(feature_importances)
        
        # Get the current metrics
        return self.performance_tracker.get_current_metrics()
    
    def get_performance_dashboard(self, save_path=None, show_plots=False):
        """
        Generate a performance dashboard for this model.
        
        Args:
            save_path: Path to save the dashboard
            show_plots: Whether to display the plots
            
        Returns:
            Path to saved dashboard if successful, None otherwise
        """
        try:
            # Create visualizer
            visualizer = PerformanceVisualizer(self.performance_tracker)
            
            # Generate dashboard
            dashboard = visualizer.create_performance_dashboard(save_path, show_plots)
            
            if dashboard and save_path:
                return save_path
                
            return None
        
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
            return None


class EnhancedEnsembleModel(EnsembleModel):
    """
    Enhanced version of EnsembleModel with integrated performance tracking.
    
    This class extends EnsembleModel to provide comprehensive performance
    monitoring, evaluation, and visualization capabilities.
    """
    
    def __init__(self, models=None, weights=None, task_type='classification', 
                 voting='soft', name='ensemble', version=None):
        """
        Initialize the enhanced ensemble model.
        
        Args:
            models: List of model wrappers
            weights: List of weights for models
            task_type: Type of task ('classification' or 'regression')
            voting: Voting strategy for classification
            name: Name identifier for this model
            version: Version identifier
        """
        # Initialize the parent EnsembleModel
        super().__init__(models, weights, task_type, voting)
        
        # Set additional properties
        self.name = name
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create performance tracker for the ensemble
        self.performance_tracker = create_performance_tracker(
            model_name=name,
            model_type=task_type,
            version=self.version
        )
        
        # Enhanced tracking for individual models
        for model in self.models:
            if not hasattr(model, 'performance_tracker'):
                model_name = f"{name}_component_{model.name}"
                model.performance_tracker = create_performance_tracker(
                    model_name=model_name,
                    model_type=task_type,
                    version=self.version
                )
        
        logger.info(f"Created enhanced ensemble model: {name} (v{self.version})")
    
    def evaluate(self, X, y):
        """
        Evaluate the ensemble and its component models.
        
        Args:
            X: Features to evaluate on
            y: Actual/true values
            
        Returns:
            Dictionary of performance metrics
        """
        # Get original evaluation results
        results = super().evaluate(X, y)
        
        # Track ensemble performance
        if self.task_type == 'classification':
            ensemble_preds = self.predict(X)
            self.performance_tracker.record_prediction(ensemble_preds, y)
            
            # Set feature importances if available
            if hasattr(self, 'feature_importances_'):
                self.performance_tracker.set_feature_importances(
                    {f"feature_{i}": imp for i, imp in enumerate(self.feature_importances_)}
                )
        
        else:  # regression
            ensemble_preds = self.predict(X)
            self.performance_tracker.record_prediction(ensemble_preds, y)
        
        # Track component model performances
        for model_wrapper in self.models:
            if hasattr(model_wrapper, 'performance_tracker'):
                model_preds = model_wrapper.predict(X)
                model_wrapper.performance_tracker.record_prediction(model_preds, y)
        
        # Add tracker metrics to results
        results['tracking'] = {
            'ensemble': self.performance_tracker.get_current_metrics(),
            'models': {
                model.name: model.performance_tracker.get_current_metrics()
                for model in self.models
                if hasattr(model, 'performance_tracker')
            }
        }
        
        return results
    
    def get_performance_comparison(self):
        """
        Compare performance between ensemble and component models.
        
        Returns:
            Comparison metrics dictionary
        """
        ensemble_metrics = self.performance_tracker.get_current_metrics()
        
        component_metrics = {
            model.name: model.performance_tracker.get_current_metrics()
            for model in self.models
            if hasattr(model, 'performance_tracker')
        }
        
        # Determine top-performing component model
        top_model = None
        top_score = -float('inf')
        
        for model_name, metrics in component_metrics.items():
            score = 0
            if self.task_type == 'classification':
                score = metrics.get('metrics', {}).get('accuracy', 0)
            else:  # regression
                # For regression, higher R² is better
                score = metrics.get('metrics', {}).get('r2', 0)
            
            if score > top_score:
                top_score = score
                top_model = model_name
        
        return {
            'ensemble': ensemble_metrics,
            'components': component_metrics,
            'top_component': top_model
        }
    
    def create_comparative_dashboard(self, save_path=None, show_plots=False):
        """
        Create a dashboard comparing ensemble vs component models.
        
        Args:
            save_path: Path to save the dashboard
            show_plots: Whether to display the plots
            
        Returns:
            Path to saved dashboard if successful, None otherwise
        """
        try:
            # Set up performance data manager
            data_manager = PerformanceDataManager()
            
            # Get component model names
            component_names = [
                f"{self.name}_component_{model.name}" 
                for model in self.models 
                if hasattr(model, 'performance_tracker')
            ]
            
            # Compare models
            comparison = data_manager.compare_models(
                model_names=[self.name] + component_names
            )
            
            # Get visualizer for ensemble
            visualizer = PerformanceVisualizer(self.performance_tracker)
            
            # Create dashboard
            dashboard = visualizer.create_performance_dashboard(save_path, show_plots)
            
            if dashboard and save_path:
                return save_path
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to create comparative dashboard: {e}")
            return None


class EnhancedRealTimeInferencePipeline(RealTimeInferencePipeline):
    """
    Enhanced real-time inference pipeline with integrated performance tracking.
    
    This class extends RealTimeInferencePipeline to provide comprehensive
    performance monitoring, evaluation, and visualization capabilities.
    """
    
    def __init__(self, model_types=None, max_queue_size=1000, prediction_timeout=0.1,
                 enable_monitoring=True, enable_tracking=True, name='real_time_pipeline'):
        """
        Initialize the enhanced inference pipeline.
        
        Args:
            model_types: List of model types to support
            max_queue_size: Maximum size of the input data queue
            prediction_timeout: Maximum time to wait for a prediction
            enable_monitoring: Whether to enable basic performance monitoring
            enable_tracking: Whether to enable comprehensive performance tracking
            name: Name identifier for this pipeline
        """
        # Initialize the parent RealTimeInferencePipeline
        super().__init__(model_types, max_queue_size, prediction_timeout, enable_monitoring)
        
        self.name = name
        self.enable_tracking = enable_tracking
        
        # Create performance trackers
        if enable_tracking:
            self.model_trackers = {}
            
            for model_type in self.model_types:
                self.model_trackers[model_type] = create_performance_tracker(
                    model_name=f"{name}_{model_type}",
                    model_type='classification'  # Default, will be updated based on actual use
                )
        
        logger.info(f"Created enhanced real-time inference pipeline: {name}")
    
    def _process_request(self, request: Dict[str, Any]):
        """
        Process a single inference request with enhanced tracking.
        
        Args:
            request: Inference request dictionary
        """
        # Track the original timestamp for enhanced metrics
        original_timestamp = datetime.now()
        
        # Call the original method
        super()._process_request(request)
        
        # Enhanced tracking if enabled and we have a result
        if self.enable_tracking:
            try:
                request_id = request.get('id', 'unknown')
                model_type = request.get('model_type', 'auto')
                
                # Look for the result in cache
                cache_key = f"{request_id}_{model_type}"
                cached = self.result_cache.get(cache_key)
                
                if cached:
                    # We have a result to track
                    prediction_result = cached['prediction']
                    
                    # Track for this model type if available
                    if model_type in self.model_trackers:
                        # Extract metrics to track
                        inference_time = prediction_result.get('inference_time_ms', 0) / 1000  # Convert to seconds
                        
                        # Update tracker with request/result pairs that include ground truth
                        if 'ground_truth' in request:
                            y_true = request['ground_truth']
                            y_pred = prediction_result.get('prediction_value', 
                                     prediction_result.get('action', 'HOLD'))
                            
                            # Track performance
                            self.model_trackers[model_type].record_prediction(
                                y_pred=[y_pred], 
                                y_true=[y_true],
                                inference_time=inference_time
                            )
                        
                        # Calculate additional metrics
                        processing_time = (datetime.now() - original_timestamp).total_seconds()
                        request_size = len(str(request))
                        response_size = len(str(prediction_result))
                        
                        # Log enhanced metrics
                        logger.debug(f"Enhanced metrics - model: {model_type}, " +
                                    f"inference: {inference_time:.3f}s, " +
                                    f"total: {processing_time:.3f}s, " +
                                    f"req size: {request_size}, " +
                                    f"res size: {response_size}")
                
            except Exception as e:
                logger.error(f"Error in enhanced request tracking: {e}")
    
    def get_tracking_metrics(self, model_type=None):
        """
        Get comprehensive tracking metrics for models.
        
        Args:
            model_type: Specific model type to get metrics for (None for all)
            
        Returns:
            Dictionary of tracking metrics
        """
        if not self.enable_tracking:
            return {"error": "Performance tracking not enabled"}
        
        if model_type:
            # Return metrics for specific model type
            if model_type in self.model_trackers:
                return self.model_trackers[model_type].get_current_metrics()
            else:
                return {"error": f"No tracker for model type: {model_type}"}
        else:
            # Return metrics for all model types
            return {
                model_type: tracker.get_current_metrics()
                for model_type, tracker in self.model_trackers.items()
            }
    
    def create_tracking_dashboard(self, model_type=None, save_path=None, show_plots=False):
        """
        Create a performance tracking dashboard.
        
        Args:
            model_type: Specific model type to visualize (None for all)
            save_path: Path to save the dashboard
            show_plots: Whether to display the plots
            
        Returns:
            Path to saved dashboard if successful, None otherwise
        """
        if not self.enable_tracking:
            logger.error("Performance tracking not enabled")
            return None
        
        try:
            if model_type:
                # Create dashboard for specific model type
                if model_type in self.model_trackers:
                    visualizer = PerformanceVisualizer(self.model_trackers[model_type])
                    save_path = save_path or f"{self.name}_{model_type}_dashboard.png"
                    dashboard = visualizer.create_performance_dashboard(save_path, show_plots)
                    return save_path if dashboard else None
                else:
                    logger.error(f"No tracker for model type: {model_type}")
                    return None
            else:
                # Create consolidated dashboard for all model types
                # Currently just creates separate files for each model type
                dashboard_paths = []
                
                for mt, tracker in self.model_trackers.items():
                    mt_save_path = save_path.replace('.png', f'_{mt}.png') if save_path else f"{self.name}_{mt}_dashboard.png"
                    visualizer = PerformanceVisualizer(tracker)
                    dashboard = visualizer.create_performance_dashboard(mt_save_path, show_plots)
                    if dashboard:
                        dashboard_paths.append(mt_save_path)
                
                return dashboard_paths if dashboard_paths else None
                
        except Exception as e:
            logger.error(f"Failed to create tracking dashboard: {e}")
            return None


# Helper functions for easy integration

def enhance_model_wrapper(model_wrapper, model_type='classification', version=None):
    """
    Convert a standard ModelWrapper to an EnhancedModelWrapper.
    
    Args:
        model_wrapper: Original ModelWrapper instance
        model_type: Type of model ('classification' or 'regression')
        version: Version identifier for the model
        
    Returns:
        EnhancedModelWrapper instance
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Dependencies not available for model enhancement")
        return model_wrapper
    
    try:
        # Create enhanced wrapper with same underlying model
        enhanced = EnhancedModelWrapper(
            model=model_wrapper.model,
            name=model_wrapper.name,
            model_type=model_type,
            version=version,
            metadata=model_wrapper.metadata
        )
        
        # Copy any relevant attributes
        enhanced.performance_history = model_wrapper.performance_history
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Failed to enhance model wrapper: {e}")
        return model_wrapper


def enhance_ensemble_model(ensemble_model, name='enhanced_ensemble', version=None):
    """
    Convert a standard EnsembleModel to an EnhancedEnsembleModel.
    
    Args:
        ensemble_model: Original EnsembleModel instance
        name: Name for the enhanced ensemble
        version: Version identifier
        
    Returns:
        EnhancedEnsembleModel instance
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Dependencies not available for ensemble enhancement")
        return ensemble_model
    
    try:
        # Enhance component models first
        enhanced_models = []
        for model_wrapper in ensemble_model.models:
            enhanced_models.append(enhance_model_wrapper(
                model_wrapper,
                model_type=ensemble_model.task_type,
                version=version
            ))
        
        # Create enhanced ensemble
        enhanced = EnhancedEnsembleModel(
            models=enhanced_models,
            weights=ensemble_model.weights,
            task_type=ensemble_model.task_type,
            voting=ensemble_model.voting,
            name=name,
            version=version
        )
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Failed to enhance ensemble model: {e}")
        return ensemble_model


def enhance_inference_pipeline(pipeline, name='enhanced_pipeline'):
    """
    Convert a standard RealTimeInferencePipeline to an EnhancedRealTimeInferencePipeline.
    
    Args:
        pipeline: Original RealTimeInferencePipeline instance
        name: Name for the enhanced pipeline
        
    Returns:
        EnhancedRealTimeInferencePipeline instance
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Dependencies not available for pipeline enhancement")
        return pipeline
    
    try:
        # Create enhanced pipeline with same configuration
        enhanced = EnhancedRealTimeInferencePipeline(
            model_types=pipeline.model_types,
            max_queue_size=pipeline.max_queue_size,
            prediction_timeout=pipeline.prediction_timeout,
            enable_monitoring=pipeline.monitor is not None,
            name=name
        )
        
        # Copy any relevant attributes or state
        if hasattr(pipeline, 'model_cache'):
            enhanced.model_cache = pipeline.model_cache
        
        if hasattr(pipeline, 'feature_pipeline'):
            enhanced.feature_pipeline = pipeline.feature_pipeline
        
        # Restart the pipeline if it was running
        if pipeline.running:
            enhanced.start()
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Failed to enhance inference pipeline: {e}")
        return pipeline


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Performance Integration')
    parser.add_argument('--test', action='store_true', help='Run integration test')
    
    args = parser.parse_args()
    
    if args.test:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            # Generate test data
            X, y = make_classification(n_samples=100, n_features=5, random_state=42)
            
            # Create a simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Create a standard ModelWrapper
            standard_wrapper = ModelWrapper(model, "test_rf")
            
            # Enhance it
            enhanced_wrapper = enhance_model_wrapper(standard_wrapper)
            
            # Make predictions
            y_pred = enhanced_wrapper.predict(X)
            
            # Evaluate performance
            metrics = enhanced_wrapper.evaluate_performance(y, y_pred)
            
            print(f"Enhanced model metrics: {metrics}")
            
            # Generate a dashboard
            dashboard_path = enhanced_wrapper.get_performance_dashboard("test_dashboard.png")
            
            if dashboard_path:
                print(f"Dashboard created at: {dashboard_path}")
            else:
                print("Dashboard creation failed")
            
        except Exception as e:
            print(f"Integration test failed: {e}")
    else:
        print("No action specified. Use --test to run the integration test.")