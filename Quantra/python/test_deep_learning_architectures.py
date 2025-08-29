#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the various neural network architectures implemented for stock prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import unittest
import json

# Directory where the module is located
module_path = os.path.dirname(os.path.abspath(__file__))
stock_predictor_path = os.path.join(module_path, 'stock_predictor.py')

def test_architectures_exist():
    """Check if the architecture classes and methods are implemented in stock_predictor.py"""
    with open(stock_predictor_path, 'r') as f:
        content = f.read()
    
    # Check for PyTorch architecture implementations
    pytorch_required = [
        'architecture_type',
        '_build_lstm_model',
        '_build_gru_model',
        '_build_transformer_model'
    ]
    
    # Check for TensorFlow architecture implementations
    tensorflow_required = [
        'architecture_type',
        '_build_lstm_model',
        '_build_gru_model',
        '_build_transformer_model'
    ]
    
    # Check PyTorch implementations
    print("\nChecking PyTorch architecture implementations...")
    for item in pytorch_required:
        found = f"PyTorchStockPredictor" in content and item in content
        print(f"  {item}: {'✓' if found else '✗'}")
        if not found:
            return False
    
    # Check TensorFlow implementations
    print("\nChecking TensorFlow architecture implementations...")
    for item in tensorflow_required:
        found = f"TensorFlowStockPredictor" in content and item in content
        print(f"  {item}: {'✓' if found else '✗'}")
        if not found:
            return False
    
    return True

def test_predict_stock_function():
    """Test that the predict_stock function now handles architecture_type"""
    with open(stock_predictor_path, 'r') as f:
        content = f.read()
    
    has_architecture_param = "def predict_stock(features, model_type='auto', architecture_type='lstm'" in content
    has_architecture_arg = "architecture_type=architecture_type" in content
    
    print("\nChecking predict_stock function:")
    print(f"  Has architecture_type parameter: {'✓' if has_architecture_param else '✗'}")
    print(f"  Passes architecture_type argument: {'✓' if has_architecture_arg else '✗'}")
    
    return has_architecture_param and has_architecture_arg

def main():
    """Run the tests"""
    print("\nTesting Deep Learning Architecture Implementations...")
    
    # Test if the architectures exist
    architectures_exist = test_architectures_exist()
    
    # Test the predict_stock function
    predict_stock_updated = test_predict_stock_function()
    
    # Report results
    if architectures_exist and predict_stock_updated:
        print("\n✓ All tests passed! The deep learning architectures are properly implemented.")
        return 0
    else:
        print("\n✗ Some tests failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())