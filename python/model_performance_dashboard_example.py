#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model Performance Dashboard Example

This script demonstrates how to create performance dashboards for ML models using
the model performance tracking and evaluation framework. It provides a simple
web interface for viewing performance metrics and visualizations.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional

# Import from our modules
from model_performance_tracking import (
    ModelPerformanceTracker,
    PerformanceVisualizer,
    PerformanceDataManager,
    create_performance_tracker
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_dashboard')


def generate_sample_data(model_name="sample_model", n_days=30, model_type="classification"):
    """
    Generate sample performance data for demonstration purposes.
    
    Args:
        model_name: Name for the sample model
        n_days: Number of days of historical data to generate
        model_type: Type of model ('classification' or 'regression')
        
    Returns:
        ModelPerformanceTracker instance with sample data
    """
    np.random.seed(42)  # For reproducibility
    
    # Create tracker
    tracker = create_performance_tracker(
        model_name=model_name,
        model_type=model_type,
        version="sample"
    )
    
    # Generate time series of dates
    end_date = datetime.now()
    dates = [end_date - timedelta(days=n_days-i) for i in range(n_days)]
    
    # Generate sample metrics that improve over time with some noise
    for i, date in enumerate(dates):
        base_accuracy = 0.7 + (i / n_days) * 0.15  # Starts at 0.7, improves to 0.85
        noise = np.random.normal(0, 0.03)  # Small random noise
        
        if model_type == "classification":
            # Generate random predictions with improving accuracy
            n_samples = 100
            accuracy = base_accuracy + noise
            
            # Create synthetic true values
            y_true = np.random.randint(0, 2, n_samples)
            
            # Create predictions with controlled accuracy
            y_pred = y_true.copy()
            
            # Introduce errors based on accuracy
            error_indices = np.random.choice(
                range(n_samples), 
                size=int(n_samples * (1 - accuracy)), 
                replace=False
            )
            
            for idx in error_indices:
                y_pred[idx] = 1 - y_true[idx]  # Flip the prediction
            
            # Record predictions
            with tracker._lock:  # Direct access for sample generation
                tracker.timestamps = dates[:i+1]
                tracker._update_metrics = lambda: None  # Disable automatic updates
            
            tracker.record_prediction(y_pred, y_true)
            
            # Manually insert metrics for this date
            metrics = {
                'accuracy': accuracy,
                'precision': base_accuracy + np.random.normal(0, 0.05),
                'recall': base_accuracy + np.random.normal(0, 0.05),
                'f1': base_accuracy + np.random.normal(0, 0.04),
            }
            
            current_time = date
            for metric_name, value in metrics.items():
                tracker.metric_history[metric_name].append({
                    'timestamp': current_time,
                    'value': value
                })
            
            # Update aggregated metrics
            tracker.aggregated_metrics = {
                'last_updated': end_date,
                'metrics': metrics,
                'sample_count': n_samples * (i + 1)
            }
                
        else:  # regression
            # Generate random predictions with improving accuracy
            n_samples = 100
            base_rmse = 0.5 - (i / n_days) * 0.3  # Starts at 0.5, improves to 0.2
            rmse = max(0.1, base_rmse + np.random.normal(0, 0.05))
            
            # Generate synthetic data
            y_true = np.random.normal(5, 2, n_samples)
            
            # Create predictions with controlled RMSE
            errors = np.random.normal(0, rmse, n_samples)
            y_pred = y_true + errors
            
            # Record predictions
            with tracker._lock:  # Direct access for sample generation
                tracker.timestamps = dates[:i+1]
                tracker._update_metrics = lambda: None  # Disable automatic updates
            
            tracker.record_prediction(y_pred, y_true)
            
            # Manually insert metrics for this date
            metrics = {
                'rmse': rmse,
                'mae': base_rmse * 0.8 + np.random.normal(0, 0.03),
                'r2': 0.7 + (i / n_days) * 0.2 + np.random.normal(0, 0.05),
                'mape': 10 - (i / n_days) * 5 + np.random.normal(0, 1),
            }
            
            current_time = date
            for metric_name, value in metrics.items():
                tracker.metric_history[metric_name].append({
                    'timestamp': current_time,
                    'value': value
                })
            
            # Update aggregated metrics
            tracker.aggregated_metrics = {
                'last_updated': end_date,
                'metrics': metrics,
                'sample_count': n_samples * (i + 1)
            }
    
    # Generate sample feature importances
    feature_importances = {}
    n_features = 10
    
    # Total importance should sum to approximately 1.0
    importances = np.random.dirichlet(np.ones(n_features) * 2, size=1)[0]
    
    for i in range(n_features):
        feature_name = f"feature_{i}"
        feature_importances[feature_name] = float(importances[i])
    
    tracker.set_feature_importances(feature_importances)
    
    # Simulate a drift event
    if n_days > 10:
        drift_date = end_date - timedelta(days=n_days//3)
        drift_metrics = tracker.aggregated_metrics['metrics'].copy()
        
        if model_type == "classification":
            # Sudden drop in performance
            drift_metrics['accuracy'] -= 0.15
            drift_metrics['precision'] -= 0.15
            drift_metrics['recall'] -= 0.15
            drift_metrics['f1'] -= 0.15
        else:
            # Sudden increase in error
            drift_metrics['rmse'] += 0.2
            drift_metrics['mae'] += 0.15
            drift_metrics['r2'] -= 0.2
        
        # Record drift event
        drift_details = {}
        for metric, value in drift_metrics.items():
            tracker.drift_detector.baseline_metrics[metric] = tracker.aggregated_metrics['metrics'][metric]
            tracker.drift_detector.baseline_std[metric] = 0.05  # Arbitrary small std
            
            # Calculate z-score
            z_score = abs(value - tracker.drift_detector.baseline_metrics[metric]) / tracker.drift_detector.baseline_std[metric]
            
            drift_details[metric] = {
                'baseline': tracker.drift_detector.baseline_metrics[metric],
                'current': value,
                'z_score': z_score,
                'threshold': 2.0
            }
        
        # Add the drift event
        tracker.drift_detector.established = True
        tracker.drift_detector.drift_events.append({
            'timestamp': drift_date,
            'details': drift_details
        })
    
    return tracker


def create_html_dashboard(tracker, output_dir="performance_dashboard"):
    """
    Create a simple HTML dashboard to display performance metrics.
    
    Args:
        tracker: ModelPerformanceTracker instance
        output_dir: Directory to save the HTML and images
        
    Returns:
        Path to the HTML dashboard file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = PerformanceVisualizer(tracker)
    
    # Generate plots
    plots = {}
    for metric_name in tracker.aggregated_metrics.get('metrics', {}).keys():
        plot_path = os.path.join(output_dir, f"{metric_name}_plot.png")
        fig, _ = visualizer.plot_metric_over_time(metric_name, plot_path)
        plots[metric_name] = f"{metric_name}_plot.png"
    
    # Generate feature importance plot
    feat_path = os.path.join(output_dir, "feature_importance.png")
    visualizer.plot_feature_importance(top_n=10, save_path=feat_path)
    
    # Generate confusion matrix if applicable
    conf_matrix_path = None
    if tracker.model_type == 'classification':
        conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        visualizer.plot_confusion_matrix(save_path=conf_matrix_path)
    
    # Get current metrics
    current_metrics = tracker.get_current_metrics()
    
    # Get drift events
    drift_events = tracker.drift_detector.get_drift_events()
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Performance Dashboard - {tracker.model_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: white;
            }}
            .metrics-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .metric-card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                color: #2c3e50;
            }}
            .plot-container {{
                margin-top: 30px;
                margin-bottom: 30px;
            }}
            .plot {{
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                display: block;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .drift-event {{
                background-color: #fff3cd;
                border: 1px solid #ffeeba;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Performance Dashboard</h1>
            <h2>{tracker.model_name} (v{tracker.version})</h2>
            <p>Last updated: {current_metrics.get('last_updated', 'Unknown')}</p>
            <p>Model type: {tracker.model_type}</p>
            <p>Sample count: {current_metrics.get('sample_count', 0)}</p>
        </div>
        
        <div class="card">
            <h2>Current Metrics</h2>
            <div class="metrics-container">
    """
    
    # Add metric cards
    for metric_name, value in current_metrics.get('metrics', {}).items():
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
        html_content += f"""
                <div class="metric-card">
                    <h3>{metric_name.upper()}</h3>
                    <div class="metric-value">{formatted_value}</div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <div class="card">
            <h2>Performance Trends</h2>
    """
    
    # Add plots
    for metric_name, plot_path in plots.items():
        html_content += f"""
            <div class="plot-container">
                <h3>{metric_name.upper()} Trend</h3>
                <img src="{plot_path}" alt="{metric_name} Plot" class="plot">
            </div>
        """
    
    html_content += """
        </div>
    """
    
    # Add feature importance
    if os.path.exists(feat_path):
        html_content += f"""
        <div class="card">
            <h2>Feature Importance</h2>
            <div class="plot-container">
                <img src="feature_importance.png" alt="Feature Importance" class="plot">
            </div>
        </div>
        """
    
    # Add confusion matrix if available
    if conf_matrix_path and os.path.exists(conf_matrix_path):
        html_content += f"""
        <div class="card">
            <h2>Confusion Matrix</h2>
            <div class="plot-container">
                <img src="confusion_matrix.png" alt="Confusion Matrix" class="plot">
            </div>
        </div>
        """
    
    # Add drift events if any
    if drift_events:
        html_content += f"""
        <div class="card">
            <h2>Drift Events ({len(drift_events)})</h2>
        """
        
        for i, event in enumerate(drift_events):
            timestamp = event['timestamp']
            details = event['details']
            
            html_content += f"""
            <div class="drift-event">
                <h3>Drift Event #{i+1} - {timestamp}</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Baseline</th>
                        <th>Current</th>
                        <th>Z-Score</th>
                    </tr>
            """
            
            for metric, detail in details.items():
                html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{detail['baseline']:.4f}</td>
                        <td>{detail['current']:.4f}</td>
                        <td>{detail['z_score']:.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
        </div>
        """
    
    # Close HTML
    html_content += """
        <div class="card">
            <h2>About</h2>
            <p>This dashboard was generated automatically by the Model Performance Tracking and Evaluation Framework.</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    html_path = os.path.join(output_dir, "dashboard.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML dashboard created at {html_path}")
    return html_path


def create_multi_model_dashboard(models, output_dir="performance_dashboard"):
    """
    Create a dashboard comparing multiple models.
    
    Args:
        models: List of ModelPerformanceTracker instances
        output_dir: Directory to save the HTML and images
        
    Returns:
        Path to the HTML dashboard file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data for each model
    model_data = []
    for tracker in models:
        model_data.append({
            'name': tracker.model_name,
            'version': tracker.version,
            'type': tracker.model_type,
            'metrics': tracker.get_current_metrics(),
        })
    
    # Create plots for each model
    model_plots = {}
    for tracker in models:
        model_name = tracker.model_name
        model_plots[model_name] = {}
        
        visualizer = PerformanceVisualizer(tracker)
        
        # Create directory for this model
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate metric plots
        for metric_name in tracker.aggregated_metrics.get('metrics', {}).keys():
            plot_path = os.path.join(model_dir, f"{metric_name}_plot.png")
            rel_path = os.path.join(model_name, f"{metric_name}_plot.png")
            fig, _ = visualizer.plot_metric_over_time(metric_name, plot_path)
            model_plots[model_name][metric_name] = rel_path
        
        # Generate feature importance plot
        feat_path = os.path.join(model_dir, "feature_importance.png")
        rel_feat_path = os.path.join(model_name, "feature_importance.png")
        visualizer.plot_feature_importance(top_n=10, save_path=feat_path)
        model_plots[model_name]['feature_importance'] = rel_feat_path
    
    # Create comparison tables
    common_metrics = set()
    for tracker in models:
        common_metrics.update(tracker.aggregated_metrics.get('metrics', {}).keys())
    
    comparison_data = {}
    for metric in common_metrics:
        comparison_data[metric] = []
        
        for tracker in models:
            value = tracker.aggregated_metrics.get('metrics', {}).get(metric, None)
            if value is not None:
                comparison_data[metric].append({
                    'model': tracker.model_name,
                    'value': value
                })
        
        # Sort by value
        if comparison_data[metric]:
            if metric in ['rmse', 'mae', 'mape']:
                # For these metrics, lower is better
                comparison_data[metric].sort(key=lambda x: x['value'])
            else:
                # For most metrics, higher is better
                comparison_data[metric].sort(key=lambda x: x['value'], reverse=True)
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: #f8f9fa;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: white;
            }
            .tab {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 5px 5px 0 0;
            }
            .tab button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 10px 20px;
                transition: 0.3s;
                font-size: 16px;
            }
            .tab button:hover {
                background-color: #ddd;
            }
            .tab button.active {
                background-color: #ccc;
            }
            .tabcontent {
                display: none;
                padding: 20px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 5px 5px;
                animation: fadeEffect 1s;
            }
            @keyframes fadeEffect {
                from {opacity: 0;}
                to {opacity: 1;}
            }
            .plot {
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                display: block;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .best {
                background-color: #d4edda;
                font-weight: bold;
            }
            .metric-container {
                margin-bottom: 40px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Model Comparison Dashboard</h1>
            <p>Compare performance metrics across multiple models</p>
        </div>
        
        <div class="card">
            <h2>Models Overview</h2>
            <table>
                <tr>
                    <th>Model Name</th>
                    <th>Version</th>
                    <th>Type</th>
                    <th>Sample Count</th>
                    <th>Last Updated</th>
                </tr>
    """
    
    for data in model_data:
        html_content += f"""
                <tr>
                    <td>{data['name']}</td>
                    <td>{data['version']}</td>
                    <td>{data['type']}</td>
                    <td>{data['metrics'].get('sample_count', 'N/A')}</td>
                    <td>{data['metrics'].get('last_updated', 'Unknown')}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="card">
            <h2>Metrics Comparison</h2>
    """
    
    # Add comparison tables for each metric
    for metric_name, metric_data in comparison_data.items():
        html_content += f"""
            <div class="metric-container">
                <h3>{metric_name.upper()}</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Value</th>
                    </tr>
        """
        
        for i, item in enumerate(metric_data):
            best_class = " class='best'" if i == 0 else ""
            html_content += f"""
                    <tr{best_class}>
                        <td>{i+1}</td>
                        <td>{item['model']}</td>
                        <td>{item['value']:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="card">
            <h2>Model Details</h2>
            <div class="tab">
    """
    
    # Create tabs for each model
    for i, tracker in enumerate(models):
        active = " class='active'" if i == 0 else ""
        html_content += f"""
                <button id="tab-{tracker.model_name}" onclick="openModel(event, '{tracker.model_name}')"{active}>{tracker.model_name}</button>
        """
    
    html_content += """
            </div>
    """
    
    # Create tab content for each model
    for i, tracker in enumerate(models):
        display = "block" if i == 0 else "none"
        
        html_content += f"""
            <div id="{tracker.model_name}" class="tabcontent" style="display: {display};">
                <h3>{tracker.model_name} (v{tracker.version})</h3>
                <p>Model type: {tracker.model_type}</p>
                
                <h4>Current Metrics</h4>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        metrics = tracker.aggregated_metrics.get('metrics', {})
        for metric_name, value in metrics.items():
            html_content += f"""
                    <tr>
                        <td>{metric_name.upper()}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h4>Performance Trends</h4>
        """
        
        # Add plots for this model
        model_name = tracker.model_name
        for metric_name, plot_path in model_plots.get(model_name, {}).items():
            if metric_name != 'feature_importance':
                html_content += f"""
                <div>
                    <h5>{metric_name.upper()}</h5>
                    <img src="{plot_path}" alt="{metric_name} Plot" class="plot">
                </div>
                """
        
        # Add feature importance
        if 'feature_importance' in model_plots.get(model_name, {}):
            html_content += f"""
                <h4>Feature Importance</h4>
                <img src="{model_plots[model_name]['feature_importance']}" alt="Feature Importance" class="plot">
            """
        
        html_content += """
            </div>
        """
    
    # Add JavaScript for tab functionality
    html_content += """
        </div>
        
        <script>
        function openModel(evt, modelName) {
          // Declare variables
          var i, tabcontent, tablinks;
          
          // Get all elements with class="tabcontent" and hide them
          tabcontent = document.getElementsByClassName("tabcontent");
          for (i = 0; i < tabcontent.length; i++) {
            tabcontent[i].style.display = "none";
          }
          
          // Get all elements with class="tablinks" and remove the class "active"
          tablinks = document.getElementsByClassName("tablinks");
          for (i = 0; i < tablinks.length; i++) {
            tablinks[i].className = tablinks[i].className.replace(" active", "");
          }
          
          // Show the current tab, and add an "active" class to the button that opened the tab
          document.getElementById(modelName).style.display = "block";
          evt.currentTarget.className += " active";
        }
        </script>
    </body>
    </html>
    """
    
    # Write HTML to file
    html_path = os.path.join(output_dir, "comparison_dashboard.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Multi-model comparison dashboard created at {html_path}")
    return html_path


def main():
    """Main function for the dashboard example."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Performance Dashboard Example')
    parser.add_argument('--output-dir', type=str, default='performance_dashboard', 
                       help='Directory to save dashboard files')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days of historical data to generate')
    parser.add_argument('--compare', action='store_true',
                       help='Create a comparison dashboard with multiple models')
    
    args = parser.parse_args()
    
    # Generate sample data
    if args.compare:
        # Generate multiple models for comparison
        trackers = [
            generate_sample_data("model_a", args.days, "classification"),
            generate_sample_data("model_b", args.days, "classification"),
            generate_sample_data("model_c", args.days, "regression")
        ]
        
        # Create comparison dashboard
        dashboard_path = create_multi_model_dashboard(trackers, args.output_dir)
        print(f"Comparison dashboard created at: {dashboard_path}")
    else:
        # Generate single model
        tracker = generate_sample_data("sample_model", args.days, "classification")
        
        # Create dashboard
        dashboard_path = create_html_dashboard(tracker, args.output_dir)
        print(f"Dashboard created at: {dashboard_path}")
        print(f"Open this file in a web browser to view the dashboard")


if __name__ == "__main__":
    main()