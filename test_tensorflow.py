#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script to verify TensorFlow is working correctly."""

import sys

def test_tensorflow():
    """Test TensorFlow import and basic functionality."""
    try:
        import tensorflow as tf
        print(f"? TensorFlow version: {tf.__version__}")
        print(f"? TensorFlow installed at: {tf.__file__}")
        
        # Test basic operations
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)
        print(f"? Basic tensor operations working")
        print(f"  Matrix multiplication result: {c.numpy().tolist()}")
        
        # Test Keras
        from tensorflow import keras
        print(f"? Keras is available: {keras.__version__}")
        
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print(f"? Created a simple Keras model successfully")
        
        return True
        
    except Exception as e:
        print(f"? Error testing TensorFlow: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_numpy():
    """Test NumPy compatibility."""
    try:
        import numpy as np
        print(f"\n? NumPy version: {np.__version__}")
        
        # Create a simple array
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        print(f"? NumPy array operations working")
        print(f"  Array shape: {arr.shape}, mean: {arr.mean()}")
        
        return True
        
    except Exception as e:
        print(f"? Error testing NumPy: {e}")
        return False


def test_stock_predictor_imports():
    """Test that stock_predictor.py can import TensorFlow."""
    try:
        import sys
        import os
        
        # Add the python directory to the path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Quantra', 'python'))
        
        # Try to import (this will execute the module-level imports)
        print(f"\n? Testing stock_predictor.py imports...")
        
        # We can't actually import it as a module since it has command-line behavior
        # But we can check if the file exists and has correct syntax
        stock_predictor_path = os.path.join('Quantra', 'python', 'stock_predictor.py')
        if os.path.exists(stock_predictor_path):
            print(f"? stock_predictor.py found at: {stock_predictor_path}")
            
            # Check if it can be compiled (syntax check)
            with open(stock_predictor_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, stock_predictor_path, 'exec')
            print(f"? stock_predictor.py has valid Python syntax")
            
            return True
        else:
            print(f"? stock_predictor.py not found")
            return False
            
    except Exception as e:
        print(f"? Error testing stock_predictor imports: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TensorFlow Installation Verification Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_numpy()
    all_tests_passed &= test_tensorflow()
    all_tests_passed &= test_stock_predictor_imports()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("? All tests passed! TensorFlow is ready to use.")
        sys.exit(0)
    else:
        print("? Some tests failed. Please review the errors above.")
        sys.exit(1)
