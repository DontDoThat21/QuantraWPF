#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering Example

This script demonstrates how to use the automated feature engineering pipeline
for financial data to generate features for machine learning models.

Usage:
    python feature_engineering_example.py input.csv [output.csv]

Example:
    python feature_engineering_example.py stock_data.csv features.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engineering import (
    FeatureEngineer, FinancialFeatureGenerator,
    build_default_pipeline, create_train_test_features
)

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python feature_engineering_example.py input.csv [output.csv]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "features.csv"
    
    # Load input data
    print(f"Loading data from {input_file}...")
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded {len(data)} rows with {len(data.columns)} columns")
        print(f"Columns: {', '.join(data.columns)}")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)
        
    # Check if we have required OHLCV columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {', '.join(missing_cols)}")
        print("Attempting to adapt data...")
        
        # Try to adapt data - often CSV files have different column names
        rename_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 
            'Volume': 'volume', 'Date': 'date'
        }
        data = data.rename(columns={col: col.lower() for col in data.columns 
                                   if col in rename_map})

    # Convert date column if present
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
    # Basic feature generation
    print("Generating basic financial features...")
    basic_generator = FinancialFeatureGenerator(
        include_basic=True,
        include_trend=True,
        include_volatility=True,
        include_volume='volume' in data.columns,
        include_momentum=True
    )
    
    basic_features = basic_generator.fit_transform(data)
    print(f"Generated {len(basic_features.columns)} basic features")
    print(f"First 5 basic features: {', '.join(basic_features.columns[:5])}")
    
    # Advanced feature engineering pipeline
    print("\nBuilding advanced feature engineering pipeline...")
    pipeline = build_default_pipeline(feature_type='full')
    
    # Transform data using the pipeline
    print("Applying feature engineering pipeline...")
    features = pipeline.fit_transform(data)
    print(f"Final feature set: {len(features.columns)} features")
    
    # Create train/test sets for modeling
    print("\nCreating train/test datasets...")
    X_train, X_test, y_train, y_test, _ = create_train_test_features(
        data, target_col='close', target_shift=5, test_size=0.2
    )
    print(f"Train set: {X_train.shape[0]} samples with {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples with {X_test.shape[1]} features")
    
    # Evaluate feature importance
    print("\nEvaluating features...")
    evaluation = pipeline.evaluate_features(X_train, y_train, cv=5)
    print(f"Cross-validation score: {evaluation['mean_score']:.4f} ± {evaluation['std_score']:.4f}")
    
    # Get top features by importance
    importance = evaluation['feature_importance']
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features by importance:")
    for feature, imp in top_features:
        print(f"  {feature}: {imp:.4f}")
    
    # Save results to output file
    print(f"\nSaving features to {output_file}...")
    features.to_csv(output_file)
    print(f"Saved {len(features)} samples with {len(features.columns)} features")
    
    # Create feature importance visualization
    print("Creating feature importance visualization...")
    fig = pipeline.visualize_feature_importance(
        importance, top_n=20, figsize=(10, 8),
        save_path="feature_importance.png"
    )
    
    print("Done! Feature engineering complete.")
    print("Output files:")
    print(f"  - {output_file} (engineered features)")
    print(f"  - feature_importance.png (feature importance visualization)")
    
if __name__ == "__main__":
    main()