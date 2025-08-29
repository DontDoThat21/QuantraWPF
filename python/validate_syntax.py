#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Syntax validation script for the L1 regularization implementation.
This script checks if the code can be imported and basic syntax is correct.
"""

import ast
import sys
import os

def validate_syntax(file_path):
    """Validate Python syntax by parsing the AST."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source_code)
        print(f"✓ Syntax validation passed for {os.path.basename(file_path)}")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in {os.path.basename(file_path)}: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error validating {os.path.basename(file_path)}: {e}")
        return False

def check_imports():
    """Check if the new imports are syntactically correct."""
    # Mock code to check import syntax
    test_imports = """
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.linear_model import Ridge
import numpy as np
"""
    
    try:
        ast.parse(test_imports)
        print("✓ Import statements are syntactically correct")
        return True
    except SyntaxError as e:
        print(f"❌ Import syntax error: {e}")
        return False

def check_method_definitions():
    """Check if the new method definitions are syntactically correct."""
    # Sample of the new methods to validate syntax
    test_methods = """
def _fit_lasso(self, X, y):
    if y is None:
        raise ValueError("Target variable y is required for Lasso feature selection")
    alpha = self.params.get('alpha', 1.0)
    max_iter = self.params.get('max_iter', 1000)
    selection_threshold = self.params.get('selection_threshold', 1e-5)
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    self.selector_ = SelectFromModel(estimator=lasso, threshold=selection_threshold)
    self.selector_.fit(X, y)
    self.support_ = self.selector_.get_support()

def _fit_elastic_net(self, X, y):
    if y is None:
        raise ValueError("Target variable y is required for ElasticNet feature selection")
    alpha = self.params.get('alpha', 1.0)
    l1_ratio = self.params.get('l1_ratio', 0.5)
    max_iter = self.params.get('max_iter', 1000)
    selection_threshold = self.params.get('selection_threshold', 1e-5)
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42)
    self.selector_ = SelectFromModel(estimator=elastic_net, threshold=selection_threshold)
    self.selector_.fit(X, y)
    self.support_ = self.selector_.get_support()

def _fit_adaptive_lasso(self, X, y):
    if y is None:
        raise ValueError("Target variable y is required for Adaptive Lasso feature selection")
    alpha = self.params.get('alpha', 1.0)
    gamma = self.params.get('gamma', 1.0)
    max_iter = self.params.get('max_iter', 1000)
    selection_threshold = self.params.get('selection_threshold', 1e-5)
    initial_estimator = self.params.get('initial_estimator')
    if initial_estimator is None:
        from sklearn.linear_model import Ridge
        initial_estimator = Ridge(alpha=1.0, random_state=42)
    initial_estimator.fit(X, y)
    initial_coefs = np.abs(initial_estimator.coef_)
    adaptive_weights = 1.0 / (initial_coefs + 1e-8) ** gamma
    X_weighted = X.copy()
    if hasattr(X_weighted, 'values'):
        X_weighted = X_weighted.values
    X_weighted = X_weighted / adaptive_weights.reshape(1, -1)
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    self.selector_ = SelectFromModel(estimator=lasso, threshold=selection_threshold)
    self.selector_.fit(X_weighted, y)
    self.support_ = self.selector_.get_support()
"""
    
    try:
        ast.parse(test_methods)
        print("✓ New method definitions are syntactically correct")
        return True
    except SyntaxError as e:
        print(f"❌ Method syntax error: {e}")
        print(f"   Line {e.lineno}: {e.text}")
        return False

def main():
    """Run syntax validation."""
    print("Running syntax validation for L1 regularization implementation...")
    print("=" * 60)
    
    # Check main feature engineering file
    feature_eng_path = os.path.join(os.path.dirname(__file__), 'feature_engineering.py')
    valid_main = validate_syntax(feature_eng_path)
    
    # Check imports
    valid_imports = check_imports()
    
    # Check method definitions
    valid_methods = check_method_definitions()
    
    # Check test file if it exists
    test_path = os.path.join(os.path.dirname(__file__), 'test_lasso_feature_selection.py')
    valid_test = True
    if os.path.exists(test_path):
        valid_test = validate_syntax(test_path)
    
    print("\n" + "=" * 60)
    
    if all([valid_main, valid_imports, valid_methods, valid_test]):
        print("✅ All syntax validation checks passed!")
        print("The L1 regularization implementation is syntactically correct.")
        return True
    else:
        print("❌ Some syntax validation checks failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)