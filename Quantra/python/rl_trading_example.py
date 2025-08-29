#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example Usage of Reinforcement Learning for Trading

This script demonstrates how to use the reinforcement learning module
to train and evaluate trading strategies on market data.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import reinforcement_learning as rl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('rl_example')

def load_sample_market_data():
    """
    Load or generate sample market data for demonstration.
    In practice, this would load real market data from your data source.
    """
    logger.info("Generating sample market data...")
    
    # Generate sample data with realistic market characteristics
    np.random.seed(42)
    n_days = 2000  # About 5.5 years of daily data
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate price series with trend and volatility clustering
    initial_price = 100.0
    prices = [initial_price]
    
    # Parameters for realistic price generation
    mean_return = 0.0005  # Daily return mean (about 12% annually)
    base_volatility = 0.015  # Base daily volatility
    
    for i in range(1, n_days):
        # Add some trend and volatility clustering
        trend = 0.0001 * np.sin(i / 100)  # Long-term trend component
        volatility = base_volatility * (1 + 0.5 * np.sin(i / 50))  # Volatility clustering
        
        # Generate return
        daily_return = np.random.normal(mean_return + trend, volatility)
        
        # Apply return to price
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1.0))  # Ensure price doesn't go negative
    
    # Generate OHLC data from close prices
    data = []
    for i, close_price in enumerate(prices):
        # Generate intraday high/low/open based on close
        high_factor = 1 + abs(np.random.normal(0, 0.01))
        low_factor = 1 - abs(np.random.normal(0, 0.01))
        open_factor = 1 + np.random.normal(0, 0.005)
        
        if i == 0:
            open_price = close_price * open_factor
        else:
            open_price = prices[i-1] * (1 + np.random.normal(0, 0.002))  # Gap from previous close
        
        high_price = max(open_price, close_price) * high_factor
        low_price = min(open_price, close_price) * low_factor
        
        # Generate volume (inversely correlated with price changes)
        price_change = abs(close_price - open_price) / open_price if open_price > 0 else 0
        base_volume = 1000000
        volume = int(base_volume * (1 + 2 * price_change) * np.random.lognormal(0, 0.3))
        
        data.append({
            'date': dates[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} days of market data")
    logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"Average daily volume: {df['volume'].mean():,.0f}")
    
    return df

def train_dqn_example():
    """Example of training a DQN agent"""
    logger.info("\n=== Training DQN Agent ===")
    
    # Load market data
    data = load_sample_market_data()
    
    # Split data into train/test
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index].copy()
    test_data = data[split_index:].copy()
    
    logger.info(f"Training data: {len(train_data)} days")
    logger.info(f"Test data: {len(test_data)} days")
    
    # Prepare data for RL
    prepared_data = rl.prepare_data_for_rl(train_data)
    
    # Create trading environment
    environment = rl.TradingEnvironment(
        data=prepared_data,
        initial_balance=10000.0,
        transaction_cost=0.001,  # 0.1% transaction cost
        max_position=1.0,
        lookback_window=20
    )
    
    # Calculate state size
    state_size = rl.calculate_state_size(environment.lookback_window)
    logger.info(f"State size: {state_size}")
    
    # Create DQN agent
    agent = rl.create_rl_agent(
        'dqn', 
        state_size,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32
    )
    
    # Create trainer
    trainer = rl.RLTrainer(
        agent=agent,
        environment=environment,
        episodes=300,  # Reduced for example
        max_steps_per_episode=len(prepared_data) - environment.lookback_window - 1
    )
    
    # Train the agent
    logger.info("Starting DQN training...")
    training_history = trainer.train(verbose=True)
    
    # Save the trained agent
    model_path = os.path.join(rl.MODEL_DIR, 'dqn_agent.pth')
    agent.save(model_path)
    logger.info(f"DQN agent saved to {model_path}")
    
    # Evaluate on test data
    logger.info("\nEvaluating DQN agent on test data...")
    test_environment = rl.TradingEnvironment(
        data=rl.prepare_data_for_rl(test_data),
        initial_balance=10000.0,
        transaction_cost=0.001,
        max_position=1.0,
        lookback_window=20
    )
    
    test_trainer = rl.RLTrainer(agent, test_environment, episodes=1)
    evaluation_results = test_trainer.evaluate(episodes=1)
    
    # Print results
    print(f"\n=== DQN Training Results ===")
    print(f"Final training episode reward: {training_history['episode_rewards'][-1]:.2f}")
    print(f"Average reward (last 50 episodes): {np.mean(training_history['episode_rewards'][-50:]):.2f}")
    print(f"Test evaluation reward: {evaluation_results['mean_reward']:.2f}")
    
    return agent, training_history, evaluation_results

def train_a2c_example():
    """Example of training an A2C agent"""
    logger.info("\n=== Training A2C Agent ===")
    
    # Load market data
    data = load_sample_market_data()
    
    # Split data
    split_index = int(len(data) * 0.8)
    train_data = data[:split_index].copy()
    test_data = data[split_index:].copy()
    
    # Prepare data
    prepared_data = rl.prepare_data_for_rl(train_data)
    
    # Create environment
    environment = rl.TradingEnvironment(
        data=prepared_data,
        initial_balance=10000.0,
        transaction_cost=0.001,
        max_position=1.0,
        lookback_window=20
    )
    
    # Calculate state size
    state_size = rl.calculate_state_size(environment.lookback_window)
    
    # Create A2C agent
    agent = rl.create_rl_agent(
        'a2c',
        state_size,
        learning_rate=0.001,
        gamma=0.95,
        entropy_coef=0.01,
        value_coef=0.5
    )
    
    # Create trainer
    trainer = rl.RLTrainer(
        agent=agent,
        environment=environment,
        episodes=300,
        max_steps_per_episode=len(prepared_data) - environment.lookback_window - 1
    )
    
    # Train the agent
    logger.info("Starting A2C training...")
    training_history = trainer.train(verbose=True)
    
    # Save the trained agent
    model_path = os.path.join(rl.MODEL_DIR, 'a2c_agent.pth')
    agent.save(model_path)
    logger.info(f"A2C agent saved to {model_path}")
    
    # Evaluate on test data
    logger.info("\nEvaluating A2C agent on test data...")
    test_environment = rl.TradingEnvironment(
        data=rl.prepare_data_for_rl(test_data),
        initial_balance=10000.0,
        transaction_cost=0.001,
        max_position=1.0,
        lookback_window=20
    )
    
    test_trainer = rl.RLTrainer(agent, test_environment, episodes=1)
    evaluation_results = test_trainer.evaluate(episodes=1)
    
    # Print results
    print(f"\n=== A2C Training Results ===")
    print(f"Final training episode reward: {training_history['episode_rewards'][-1]:.2f}")
    print(f"Average reward (last 50 episodes): {np.mean(training_history['episode_rewards'][-50:]):.2f}")
    print(f"Test evaluation reward: {evaluation_results['mean_reward']:.2f}")
    
    return agent, training_history, evaluation_results

def compare_with_baseline():
    """Compare RL strategies with simple baseline strategies"""
    logger.info("\n=== Comparing with Baseline Strategies ===")
    
    # Load test data
    data = load_sample_market_data()
    test_data = data[-400:].copy()  # Last 400 days for testing
    prepared_data = rl.prepare_data_for_rl(test_data)
    
    initial_balance = 10000.0
    
    # Buy and Hold strategy
    buy_hold_balance = initial_balance
    shares = buy_hold_balance / prepared_data.iloc[0]['close']
    buy_hold_final_value = shares * prepared_data.iloc[-1]['close']
    buy_hold_return = (buy_hold_final_value - initial_balance) / initial_balance * 100
    
    print(f"Buy and Hold Return: {buy_hold_return:.2f}%")
    
    # Load trained DQN agent if available
    dqn_model_path = os.path.join(rl.MODEL_DIR, 'dqn_agent.pth')
    if os.path.exists(dqn_model_path):
        state_size = rl.calculate_state_size(20)
        dqn_agent = rl.create_rl_agent('dqn', state_size)
        dqn_agent.load(dqn_model_path)
        
        # Test DQN agent
        test_environment = rl.TradingEnvironment(
            data=prepared_data,
            initial_balance=initial_balance,
            transaction_cost=0.001,
            max_position=1.0,
            lookback_window=20
        )
        
        test_trainer = rl.RLTrainer(dqn_agent, test_environment, episodes=1)
        dqn_results = test_trainer.evaluate(episodes=1)
        
        # Calculate DQN return
        final_info = dqn_results['episode_infos'][0][-1]
        dqn_final_value = final_info['portfolio_value']
        dqn_return = (dqn_final_value - initial_balance) / initial_balance * 100
        
        print(f"DQN Agent Return: {dqn_return:.2f}%")
        print(f"DQN vs Buy and Hold: {dqn_return - buy_hold_return:.2f}% difference")
    else:
        print("No trained DQN agent found. Train a model first.")

def demonstrate_regime_integration():
    """Demonstrate integration with market regime detection"""
    logger.info("\n=== Demonstrating Market Regime Integration ===")
    
    try:
        # Try to load regime detector
        regime_detector = rl.integrate_with_regime_detection()
        
        if regime_detector:
            logger.info("Market regime detector loaded successfully")
            
            # Create environment with regime detection
            data = load_sample_market_data()[-500:]  # Last 500 days
            prepared_data = rl.prepare_data_for_rl(data)
            
            environment = rl.TradingEnvironment(
                data=prepared_data,
                initial_balance=10000.0,
                regime_detector=regime_detector,
                lookback_window=20
            )
            
            # Get a few states to show regime detection
            state = environment.reset()
            print(f"Initial state regime: {state.regime}")
            
            for i in range(5):
                action = rl.ACTION_HOLD  # Just hold to see regime changes
                next_state, reward, done, info = environment.step(action)
                print(f"Step {i+1} regime: {next_state.regime}")
                if done:
                    break
                state = next_state
                
        else:
            logger.info("Market regime detection not available in this demo")
            
    except Exception as e:
        logger.warning(f"Error in regime integration demo: {e}")

def main():
    """Main example execution"""
    print("=== Reinforcement Learning for Trading - Example Usage ===")
    print("This example demonstrates the RL trading system capabilities.")
    
    try:
        # Check if PyTorch is available
        if rl.PYTORCH_AVAILABLE:
            print("PyTorch is available - using neural network agents")
            
            # Train DQN agent
            dqn_agent, dqn_history, dqn_eval = train_dqn_example()
            
            # Train A2C agent
            a2c_agent, a2c_history, a2c_eval = train_a2c_example()
            
            # Compare with baseline
            compare_with_baseline()
            
        else:
            print("PyTorch not available - using basic Q-learning agent")
            
            # Create basic agent example
            data = load_sample_market_data()[-1000:]  # Use smaller dataset
            prepared_data = rl.prepare_data_for_rl(data)
            
            environment = rl.TradingEnvironment(
                data=prepared_data,
                initial_balance=10000.0,
                lookback_window=10  # Smaller lookback for basic agent
            )
            
            basic_agent = rl.BasicAgent()
            trainer = rl.RLTrainer(basic_agent, environment, episodes=100)
            
            print("Training basic Q-learning agent...")
            training_history = trainer.train()
            
            print(f"Basic agent final reward: {training_history['episode_rewards'][-1]:.2f}")
        
        # Demonstrate regime integration
        demonstrate_regime_integration()
        
        print("\n=== Example completed successfully! ===")
        print("You can now:")
        print("1. Modify the training parameters to experiment with different settings")
        print("2. Use your own market data by replacing the sample data generation")
        print("3. Integrate with the existing market regime detection system")
        print("4. Deploy trained models for live trading (with proper risk management)")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Example failed with error: {e}")
        print("Please check the error messages above and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()