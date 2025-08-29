#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of L1 regularization feature selection methods.

This script demonstrates how to use the new Lasso, ElasticNet, and 
Adaptive Lasso feature selection methods in the Quantra trading platform.
"""

import numpy as np
import pandas as pd
from feature_engineering import FeatureSelector

def demonstrate_lasso_selection():
    """Demonstrate Lasso feature selection usage."""
    print("=== Lasso Feature Selection Example ===")
    
    # Example: Create sample financial data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # Generate synthetic financial indicators
    feature_names = [
        'rsi_14', 'macd_signal', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12',
        'volume_sma', 'atr_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci_20',
        'momentum_10', 'roc_12', 'tsi', 'mfi_14', 'adx_14', 'aroon_up',
        'aroon_down', 'obv_ratio'
    ]
    
    X = np.random.randn(n_samples, n_features)
    # Make first 5 features more predictive
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.2 + 
         X[:, 3] * 0.8 + X[:, 4] * 0.5 + np.random.randn(n_samples) * 0.1)
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Apply Lasso feature selection
    lasso_selector = FeatureSelector(
        method='lasso',
        params={
            'alpha': 0.1,  # Regularization strength
            'selection_threshold': 1e-3,  # Minimum coefficient threshold
            'max_iter': 1000
        }
    )
    
    lasso_selector.fit(X_df, y)
    X_selected = lasso_selector.transform(X_df)
    
    print(f"Original features: {X_df.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected indicators: {lasso_selector.selected_features_}")
    
    # Show feature importances
    importances = lasso_selector.get_feature_importances()
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 important features: {top_features}")
    
    return lasso_selector, X_selected

def demonstrate_elastic_net_selection():
    """Demonstrate ElasticNet feature selection usage."""
    print("\n=== ElasticNet Feature Selection Example ===")
    
    # Use same synthetic data as above
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    feature_names = [
        'rsi_14', 'macd_signal', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12',
        'volume_sma', 'atr_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci_20',
        'momentum_10', 'roc_12', 'tsi', 'mfi_14', 'adx_14', 'aroon_up',
        'aroon_down', 'obv_ratio'
    ]
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.2 + 
         X[:, 3] * 0.8 + X[:, 4] * 0.5 + np.random.randn(n_samples) * 0.1)
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Apply ElasticNet feature selection
    elastic_selector = FeatureSelector(
        method='elastic_net',
        params={
            'alpha': 0.1,  # Overall regularization strength  
            'l1_ratio': 0.7,  # Balance between L1 (0.7) and L2 (0.3)
            'selection_threshold': 1e-3,
            'max_iter': 1000
        }
    )
    
    elastic_selector.fit(X_df, y)
    X_selected = elastic_selector.transform(X_df)
    
    print(f"Original features: {X_df.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected indicators: {elastic_selector.selected_features_}")
    
    # Show feature importances
    importances = elastic_selector.get_feature_importances()
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 important features: {top_features}")
    
    return elastic_selector, X_selected

def demonstrate_adaptive_lasso_selection():
    """Demonstrate Adaptive Lasso feature selection usage."""
    print("\n=== Adaptive Lasso Feature Selection Example ===")
    
    # Use same synthetic data as above
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    feature_names = [
        'rsi_14', 'macd_signal', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12',
        'volume_sma', 'atr_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci_20',
        'momentum_10', 'roc_12', 'tsi', 'mfi_14', 'adx_14', 'aroon_up',
        'aroon_down', 'obv_ratio'
    ]
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.2 + 
         X[:, 3] * 0.8 + X[:, 4] * 0.5 + np.random.randn(n_samples) * 0.1)
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Apply Adaptive Lasso feature selection
    adaptive_selector = FeatureSelector(
        method='adaptive_lasso',
        params={
            'alpha': 0.1,  # Lasso regularization strength
            'gamma': 1.0,  # Adaptive penalty parameter
            'selection_threshold': 1e-3,
            'max_iter': 1000
            # initial_estimator defaults to Ridge regression
        }
    )
    
    adaptive_selector.fit(X_df, y)
    X_selected = adaptive_selector.transform(X_df)
    
    print(f"Original features: {X_df.shape[1]}")
    print(f"Selected features: {X_selected.shape[1]}")
    print(f"Selected indicators: {adaptive_selector.selected_features_}")
    
    # Show feature importances
    importances = adaptive_selector.get_feature_importances()
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 important features: {top_features}")
    
    return adaptive_selector, X_selected

def compare_methods():
    """Compare all three L1-based methods."""
    print("\n=== Comparison of L1 Regularization Methods ===")
    
    # Use same data for fair comparison
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    X = np.random.randn(n_samples, n_features)
    # Make first 5 features informative
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.2 + 
         X[:, 3] * 0.8 + X[:, 4] * 0.5 + np.random.randn(n_samples) * 0.1)
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    methods = [
        ('lasso', {'alpha': 0.1, 'selection_threshold': 1e-3}),
        ('elastic_net', {'alpha': 0.1, 'l1_ratio': 0.7, 'selection_threshold': 1e-3}),
        ('adaptive_lasso', {'alpha': 0.1, 'gamma': 1.0, 'selection_threshold': 1e-3})
    ]
    
    results = {}
    
    for method_name, params in methods:
        selector = FeatureSelector(method=method_name, params=params)
        selector.fit(X_df, y)
        X_selected = selector.transform(X_df)
        
        results[method_name] = {
            'n_selected': X_selected.shape[1],
            'selected_features': selector.selected_features_,
            'importances': selector.get_feature_importances()
        }
    
    # Display comparison
    for method_name, result in results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Features selected: {result['n_selected']}")
        print(f"  Top features: {result['selected_features'][:5]}")
    
    return results

def main():
    """Run all examples."""
    print("L1 Regularization Feature Selection Examples")
    print("=" * 50)
    
    try:
        # Individual method demonstrations
        demonstrate_lasso_selection()
        demonstrate_elastic_net_selection()
        demonstrate_adaptive_lasso_selection()
        
        # Comparison
        compare_methods()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nThese methods are now available in the FeatureSelector class:")
        print("- method='lasso': Pure L1 regularization")
        print("- method='elastic_net': Combined L1/L2 regularization")
        print("- method='adaptive_lasso': Feature-specific penalty weights")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()