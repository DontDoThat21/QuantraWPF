#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for hyperparameter_optimization.py.
This is a simple test that only checks if the module imports correctly.
"""

import os
import sys
import unittest

# Add the parent directory to sys.path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class TestHyperparameterOptimization(unittest.TestCase):
    """Test cases for hyperparameter_optimization.py."""
    
    def test_imports(self):
        """Test that the module can be imported."""
        try:
            from hyperparameter_optimization import (
                OptimizationResult,
                optimize_sklearn_model,
                optimize_sklearn_model_optuna,
                optimize_pytorch_model,
                optimize_tensorflow_model,
                visualize_optimization_results
            )
            success = True
        except ImportError as e:
            success = False
            print(f"Error importing hyperparameter_optimization: {str(e)}")
        
        self.assertTrue(success, "Failed to import hyperparameter_optimization module")

if __name__ == "__main__":
    unittest.main()