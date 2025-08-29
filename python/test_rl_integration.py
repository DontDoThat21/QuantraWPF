#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test for Reinforcement Learning Module

This script tests the basic functionality of the RL module without requiring
external dependencies to be installed.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that the RL module can be imported"""
    try:
        import reinforcement_learning as rl
        print("✓ Reinforcement learning module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import RL module: {e}")
        return False

def test_data_preparation():
    """Test data preparation functionality"""
    try:
        import reinforcement_learning as rl
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000000, 1100000, 900000, 1200000, 950000]
        })
        
        prepared = rl.prepare_data_for_rl(sample_data)
        assert len(prepared) == 5
        assert all(col in prepared.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        print("✓ Data preparation works correctly")
        return True
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

def test_trading_environment():
    """Test trading environment creation and basic operations"""
    try:
        import reinforcement_learning as rl
        
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [100 + i for i in range(50)],
            'high': [101 + i for i in range(50)],
            'low': [99 + i for i in range(50)],
            'close': [100.5 + i for i in range(50)],
            'volume': [1000000] * 50
        })
        
        # Create environment
        env = rl.TradingEnvironment(
            data=sample_data,
            initial_balance=10000.0,
            lookback_window=10
        )
        
        # Test reset
        state = env.reset()
        assert isinstance(state, rl.TradingState)
        assert len(state.prices) > 0
        assert len(state.indicators) > 0
        
        # Test step
        next_state, reward, done, info = env.step(rl.ACTION_HOLD)
        assert isinstance(next_state, rl.TradingState)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        print("✓ Trading environment works correctly")
        return True
    except Exception as e:
        print(f"✗ Trading environment test failed: {e}")
        return False

def test_state_size_calculation():
    """Test state size calculation"""
    try:
        import reinforcement_learning as rl
        
        lookback_window = 20
        num_indicators = 5
        state_size = rl.calculate_state_size(lookback_window, num_indicators)
        
        expected_size = lookback_window + num_indicators + 3  # +3 for position, cash, portfolio
        assert state_size == expected_size
        
        print("✓ State size calculation works correctly")
        return True
    except Exception as e:
        print(f"✗ State size calculation failed: {e}")
        return False

def test_basic_agent_creation():
    """Test basic agent creation (without PyTorch)"""
    try:
        import reinforcement_learning as rl
        
        # This should create a BasicAgent since PyTorch likely isn't available
        agent = rl.create_rl_agent('dqn', state_size=25)
        assert agent is not None
        
        # Test basic agent methods exist
        assert hasattr(agent, 'act')
        assert hasattr(agent, 'train')
        assert hasattr(agent, 'save')
        assert hasattr(agent, 'load')
        
        print("✓ Basic agent creation works correctly")
        return True
    except Exception as e:
        print(f"✗ Basic agent creation failed: {e}")
        return False

def test_experience_and_replay_buffer():
    """Test Experience and ReplayBuffer classes"""
    try:
        import reinforcement_learning as rl
        
        # Create sample states
        state = rl.TradingState(
            prices=np.array([100, 101, 102]),
            indicators=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            position=0.5,
            cash=0.2,
            portfolio_value=0.1,
            regime="TRENDING_UP",
            step=10
        )
        
        next_state = rl.TradingState(
            prices=np.array([101, 102, 103]),
            indicators=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            position=0.6,
            cash=0.1,
            portfolio_value=0.15,
            regime="TRENDING_UP",
            step=11
        )
        
        # Create experience
        experience = rl.Experience(
            state=state,
            action=rl.ACTION_BUY,
            reward=1.5,
            next_state=next_state,
            done=False
        )
        
        # Test replay buffer
        buffer = rl.ReplayBuffer(capacity=100)
        buffer.push(experience)
        assert len(buffer) == 1
        
        sampled = buffer.sample(1)
        assert len(sampled) == 1
        assert sampled[0].action == rl.ACTION_BUY
        
        print("✓ Experience and ReplayBuffer work correctly")
        return True
    except Exception as e:
        print(f"✗ Experience and ReplayBuffer test failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("Running Reinforcement Learning Integration Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_data_preparation,
        test_trading_environment,
        test_state_size_calculation,
        test_basic_agent_creation,
        test_experience_and_replay_buffer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The RL module is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)