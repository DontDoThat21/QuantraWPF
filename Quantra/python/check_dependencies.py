#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dependency Checker for Quantra Python Environment

This script checks if all required Python dependencies are installed and working.
Run this to diagnose import issues with feature_engineering and other modules.
"""

import sys
import importlib

def check_module(module_name, package_name=None, optional=False):
    """
    Check if a Python module can be imported.
    
    Args:
        module_name (str): Name of the module to import
        package_name (str): Name of the pip package (if different from module)
        optional (bool): Whether the module is optional
        
    Returns:
        bool: True if module is available, False otherwise
    """
    if package_name is None:
        package_name = module_name
        
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        status = "? OK" if not optional else "? OK (optional)"
        print(f"{status:20} {module_name:30} (version: {version})")
        return True
    except ImportError as e:
        status = "? MISSING" if not optional else "? MISSING (optional)"
        print(f"{status:20} {module_name:30} pip install {package_name}")
        if not optional:
            print(f"    Error: {e}")
        return False
    except Exception as e:
        status = "? ERROR" if not optional else "? ERROR (optional)"
        print(f"{status:20} {module_name:30} Error: {e}")
        return False


def main():
    print("=" * 80)
    print("Quantra Python Dependency Checker")
    print("=" * 80)
    print()
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print()
    
    # Core dependencies
    print("Core Dependencies:")
    print("-" * 80)
    core_deps = [
        ('numpy', None, False),
        ('pandas', None, False),
        ('sklearn', 'scikit-learn', False),
        ('scipy', None, False),
        ('joblib', None, False),
        ('matplotlib', None, False),
    ]
    
    core_ok = True
    for module, package, optional in core_deps:
        if not check_module(module, package, optional):
            core_ok = False
    
    print()
    
    # Machine Learning Frameworks
    print("Machine Learning Frameworks:")
    print("-" * 80)
    ml_deps = [
        ('torch', None, True),
        ('tensorflow', None, True),
    ]
    
    ml_ok = True
    for module, package, optional in ml_deps:
        if not check_module(module, package, optional):
            ml_ok = False
    
    print()
    
    # Optimization and Analysis
    print("Optimization and Analysis:")
    print("-" * 80)
    opt_deps = [
        ('optuna', None, True),
        ('plotly', None, True),
        ('hmmlearn', None, True),
    ]
    
    for module, package, optional in opt_deps:
        check_module(module, package, optional)
    
    print()
    
    # Check custom modules
    print("Custom Modules:")
    print("-" * 80)
    
    # Try to import feature_engineering
    try:
        from feature_engineering import (
            FeatureEngineer, FinancialFeatureGenerator,
            build_default_pipeline, create_train_test_features
        )
        print(f"? OK                 feature_engineering module")
    except ImportError as e:
        print(f"? MISSING            feature_engineering module")
        print(f"    Error: {e}")
        core_ok = False
    except Exception as e:
        print(f"? ERROR              feature_engineering module")
        print(f"    Error: {e}")
        core_ok = False
    
    # Try to import stock_predictor
    try:
        import stock_predictor
        print(f"? OK                 stock_predictor module")
    except ImportError as e:
        print(f"? MISSING            stock_predictor module")
        print(f"    Error: {e}")
    except Exception as e:
        print(f"? ERROR              stock_predictor module")
        print(f"    Error: {e}")
    
    print()
    print("=" * 80)
    
    # Summary
    if core_ok:
        print("? All core dependencies are installed!")
    else:
        print("? Some core dependencies are missing. Run:")
        print("  pip install -r requirements.txt")
    
    print()
    print("To install all dependencies, run:")
    print("  pip install -r requirements.txt")
    print()
    print("For GPU support (optional), also install:")
    print("  pip install cudf cuml cupy  # Requires NVIDIA CUDA")
    print("=" * 80)
    
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
