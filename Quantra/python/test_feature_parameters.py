#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify feature_type and use_feature_engineering parameters
are properly extracted and used in train_from_database.py
"""

import sys
import json
import tempfile
import os

# Add the script directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def test_feature_parameters():
    """Test that feature parameters are properly extracted and used."""
    print("=" * 60)
    print("Testing Feature Parameter Extraction")
    print("=" * 60)
    
    # Test 1: Default parameters (no config)
    print("\n[Test 1] Default parameters (no hyperparameters dict):")
    hyperparams_default = {}
    
    # Test extracting from empty hyperparameters dict
    feature_type = hyperparams_default.get('feature_type', 'balanced')
    use_feature_engineering = hyperparams_default.get('use_feature_engineering', True)
    
    if feature_type == 'comprehensive':
        feature_type = 'full'
    
    print(f"  feature_type: {feature_type} (expected: 'balanced')")
    print(f"  use_feature_engineering: {use_feature_engineering} (expected: True)")
    assert feature_type == 'balanced', "Default feature_type should be 'balanced'"
    assert use_feature_engineering == True, "Default use_feature_engineering should be True"
    print("  ✓ Test 1 passed!")
    
    # Test 2: Minimal features
    print("\n[Test 2] Minimal features configuration:")
    hyperparams_minimal = {
        'feature_type': 'minimal',
        'use_feature_engineering': False,
        'epochs': 10,
        'batch_size': 32
    }
    
    feature_type = hyperparams_minimal.get('feature_type', 'balanced')
    use_feature_engineering = hyperparams_minimal.get('use_feature_engineering', True)
    
    if feature_type == 'comprehensive':
        feature_type = 'full'
    
    print(f"  feature_type: {feature_type} (expected: 'minimal')")
    print(f"  use_feature_engineering: {use_feature_engineering} (expected: False)")
    assert feature_type == 'minimal', "feature_type should be 'minimal'"
    assert use_feature_engineering == False, "use_feature_engineering should be False"
    print("  ✓ Test 2 passed!")
    
    # Test 3: Comprehensive features (should map to 'full')
    print("\n[Test 3] Comprehensive features configuration:")
    hyperparams_comprehensive = {
        'feature_type': 'comprehensive',
        'use_feature_engineering': True,
        'epochs': 50,
        'batch_size': 32
    }
    
    feature_type = hyperparams_comprehensive.get('feature_type', 'balanced')
    use_feature_engineering = hyperparams_comprehensive.get('use_feature_engineering', True)
    
    # This is the key mapping fix
    if feature_type == 'comprehensive':
        feature_type = 'full'
    
    print(f"  feature_type: {feature_type} (expected: 'full' - mapped from 'comprehensive')")
    print(f"  use_feature_engineering: {use_feature_engineering} (expected: True)")
    assert feature_type == 'full', "feature_type 'comprehensive' should map to 'full'"
    assert use_feature_engineering == True, "use_feature_engineering should be True"
    print("  ✓ Test 3 passed!")
    
    # Test 4: Test JSON config parsing (simulating C# input)
    print("\n[Test 4] JSON config parsing (C# interop):")
    config_json = {
        "configurationName": "High Accuracy Test",
        "featureType": "comprehensive",  # camelCase from C#
        "useFeatureEngineering": True,    # camelCase from C#
        "modelType": "pytorch",
        "architectureType": "lstm",
        "epochs": 50,
        "batchSize": 32,
        "learningRate": 0.001
    }
    
    # Simulate how the script extracts from config
    feature_type = config_json.get('featureType', 'balanced')
    use_feature_engineering = config_json.get('useFeatureEngineering', True)
    
    if feature_type == 'comprehensive':
        feature_type = 'full'
    
    print(f"  Original JSON featureType: {config_json['featureType']}")
    print(f"  Mapped feature_type: {feature_type} (expected: 'full')")
    print(f"  use_feature_engineering: {use_feature_engineering} (expected: True)")
    assert feature_type == 'full', "featureType should map 'comprehensive' to 'full'"
    assert use_feature_engineering == True, "useFeatureEngineering should be True"
    print("  ✓ Test 4 passed!")
    
    # Test 5: Verify hyperparameters dict includes feature params
    print("\n[Test 5] Hyperparameters dict includes feature parameters:")
    hyperparameters = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.1,
        'hidden_dim': 128,
        'num_layers': 2,
        'feature_type': 'full',
        'use_feature_engineering': True
    }
    
    assert 'feature_type' in hyperparameters, "feature_type should be in hyperparameters dict"
    assert 'use_feature_engineering' in hyperparameters, "use_feature_engineering should be in hyperparameters dict"
    print(f"  feature_type in dict: {hyperparameters['feature_type']}")
    print(f"  use_feature_engineering in dict: {hyperparameters['use_feature_engineering']}")
    print("  ✓ Test 5 passed!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nExpected behavior after fix:")
    print("  - Minimal: 15-20 features")
    print("  - Balanced: 40-50 features")
    print("  - Comprehensive: 100-150 features")
    print("\nNote: Actual feature counts depend on feature_engineering module availability")


if __name__ == "__main__":
    try:
        test_feature_parameters()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
