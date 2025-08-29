#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic test to verify the integration of hyperparameter optimization with stock_predictor.py
"""

import os
import sys

# Directory where the module is located
module_path = os.path.dirname(os.path.abspath(__file__))
stock_predictor_file = os.path.join(module_path, 'stock_predictor.py')
hyperparameter_file = os.path.join(module_path, 'hyperparameter_optimization.py')

def test_files_exist():
    """Verify that both files exist."""
    stock_exists = os.path.isfile(stock_predictor_file)
    hyperparameter_exists = os.path.isfile(hyperparameter_file)
    print(f"stock_predictor.py exists: {stock_exists}")
    print(f"hyperparameter_optimization.py exists: {hyperparameter_exists}")
    return stock_exists and hyperparameter_exists

def test_integration():
    """Verify that stock_predictor.py has integration with hyperparameter_optimization."""
    with open(stock_predictor_file, 'r') as f:
        content = f.read()
    
    integration_elements = [
        'from hyperparameter_optimization import',
        'HYPERPARAMETER_OPTIMIZATION_AVAILABLE',
        'optimize_model',
        'hyperparameterOptimization',
        'OptimizationResult'
    ]
    
    results = []
    for element in integration_elements:
        found = element in content
        print(f"Contains {element}: {found}")
        results.append(found)
    
    return all(results)

def main():
    print("\nTesting integration between stock_predictor.py and hyperparameter_optimization.py...\n")
    files_exist = test_files_exist()
    if files_exist:
        integration_valid = test_integration()
        if integration_valid:
            print("\nIntegration test passed!")
            return 0
        else:
            print("\nIntegration test failed!")
            return 1
    else:
        print("\nOne or more files do not exist!")
        return 1

if __name__ == "__main__":
    sys.exit(main())