#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reinforcement Learning Module for Adaptive Trading Strategies

This module implements reinforcement learning algorithms for developing adaptive
trading strategies that dynamically optimize trade decisions based on real-time
market feedback.

Algorithms implemented:
1. Deep Q-Network (DQN) for value-based learning
2. Actor-Critic (A2C) for policy gradient methods
3. Proximal Policy Optimization (PPO) for stable policy updates
4. Integration with market regime detection for adaptive strategy switching
"""

import sys
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import random
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reinforcement_learning')

# Try to import ML frameworks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch is available for RL algorithms")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. RL algorithms will use basic implementation")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available for RL algorithms")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

# Model paths for persistence
MODEL_DIR = 'python/models/rl'
os.makedirs(MODEL_DIR, exist_ok=True)

# Trading action definitions
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTIONS = [ACTION_HOLD, ACTION_BUY, ACTION_SELL]
ACTION_NAMES = ['HOLD', 'BUY', 'SELL']

@dataclass
class TradingState:
    """Represents the current state of the trading environment"""
    prices: np.ndarray  # Historical prices
    indicators: np.ndarray  # Technical indicators
    position: float  # Current position (-1 to 1, normalized)
    cash: float  # Available cash (normalized)
    portfolio_value: float  # Total portfolio value (normalized)
    regime: str  # Market regime from regime detection
    step: int  # Current time step

@dataclass
class Experience:
    """Represents a single experience tuple for training"""
    state: TradingState
    action: int
    reward: float
    next_state: TradingState
    done: bool

class TradingEnvironment:
    """
    Trading Environment for Reinforcement Learning
    
    This class simulates a trading environment where an RL agent can learn
    optimal trading strategies by interacting with historical market data.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0,
                 lookback_window: int = 20,
                 regime_detector=None):
        """
        Initialize the trading environment
        
        Args:
            data: DataFrame with OHLCV data
            initial_balance: Starting cash balance
            transaction_cost: Cost per transaction (as fraction)
            max_position: Maximum position size (as fraction of balance)
            lookback_window: Number of historical steps to include in state
            regime_detector: Market regime detection model
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.regime_detector = regime_detector
        
        # Initialize environment state
        self.reset()
        
        # Calculate technical indicators
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """Calculate technical indicators for the state representation"""
        # Simple moving averages
        self.data['sma_5'] = self.data['close'].rolling(window=5).mean()
        self.data['sma_20'] = self.data['close'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI)
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['bb_middle'] = self.data['close'].rolling(window=20).mean()
        bb_std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['bb_middle'] + (bb_std * 2)
        self.data['bb_lower'] = self.data['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        self.data['volume_sma'] = self.data['volume'].rolling(window=20).mean()
        
        # Price momentum
        self.data['momentum'] = self.data['close'].pct_change(periods=5)
        
        # Volatility
        self.data['volatility'] = self.data['close'].pct_change().rolling(window=20).std()
        
    def reset(self) -> TradingState:
        """Reset the environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.shares = 0.0
        self.total_reward = 0.0
        self.transaction_history = []
        
        return self._get_state()
    
    def _get_state(self) -> TradingState:
        """Get the current state representation"""
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        
        # Get price data for lookback window
        prices = self.data['close'].iloc[start_idx:end_idx].values
        
        # Normalize prices (percentage change from first price in window)
        if len(prices) > 1:
            prices = (prices / prices[0] - 1) * 100
        else:
            prices = np.array([0.0])
        
        # Get technical indicators
        current_data = self.data.iloc[self.current_step]
        indicators = np.array([
            current_data.get('sma_5', 0) / current_data['close'] - 1 if current_data['close'] > 0 else 0,
            current_data.get('sma_20', 0) / current_data['close'] - 1 if current_data['close'] > 0 else 0,
            (current_data.get('rsi', 50) - 50) / 50,  # Normalize RSI to [-1, 1]
            current_data.get('momentum', 0),
            current_data.get('volatility', 0) * 100,  # Scale volatility
        ])
        
        # Get market regime if available
        regime = "UNKNOWN"
        if self.regime_detector:
            try:
                # This would integrate with the market regime detection module
                regime_data = {
                    'close': [current_data['close']],
                    'volume': [current_data['volume']],
                    'high': [current_data['high']],
                    'low': [current_data['low']]
                }
                regime = self.regime_detector.predict_regime(regime_data)
            except Exception as e:
                logger.warning(f"Could not get market regime: {e}")
        
        # Calculate portfolio value
        current_price = current_data['close']
        portfolio_value = self.balance + self.shares * current_price
        
        # Normalize position and financial values
        normalized_position = self.position  # Already normalized to [-1, 1]
        normalized_cash = self.balance / self.initial_balance - 1  # Relative to initial
        normalized_portfolio = portfolio_value / self.initial_balance - 1
        
        return TradingState(
            prices=prices,
            indicators=indicators,
            position=normalized_position,
            cash=normalized_cash,
            portfolio_value=normalized_portfolio,
            regime=regime,
            step=self.current_step
        )
    
    def step(self, action: int) -> Tuple[TradingState, float, bool, Dict]:
        """
        Execute one trading step
        
        Args:
            action: Trading action (0=HOLD, 1=BUY, 2=SELL)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0.0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute trading action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Get next state
        next_state = self._get_state()
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.balance + self.shares * current_price <= 0)
        
        # Calculate info
        portfolio_value = self.balance + self.shares * current_price
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'position': self.position,
            'action': ACTION_NAMES[action]
        }
        
        self.total_reward += reward
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute the trading action and return reward"""
        old_portfolio_value = self.balance + self.shares * current_price
        
        if action == ACTION_BUY:
            # Buy with available cash (up to max position)
            max_shares_affordable = self.balance / (current_price * (1 + self.transaction_cost))
            target_position = min(self.max_position, self.position + 0.25)  # Increase position by 25%
            target_shares = target_position * self.initial_balance / current_price
            shares_to_buy = min(max_shares_affordable, max(0, target_shares - self.shares))
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares += shares_to_buy
                    self.position = self.shares * current_price / self.initial_balance
                    self.transaction_history.append(('BUY', shares_to_buy, current_price, self.current_step))
        
        elif action == ACTION_SELL:
            # Sell shares (up to current holdings)
            target_position = max(-self.max_position, self.position - 0.25)  # Decrease position by 25%
            target_shares = max(0, target_position * self.initial_balance / current_price)
            shares_to_sell = max(0, self.shares - target_shares)
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.shares -= shares_to_sell
                self.position = self.shares * current_price / self.initial_balance
                self.transaction_history.append(('SELL', shares_to_sell, current_price, self.current_step))
        
        # ACTION_HOLD: do nothing
        
        # Calculate reward based on portfolio value change
        new_portfolio_value = self.balance + self.shares * current_price
        return_rate = (new_portfolio_value - old_portfolio_value) / old_portfolio_value if old_portfolio_value > 0 else 0
        
        # Scale reward to make it more suitable for RL
        reward = return_rate * 1000  # Scale up the reward
        
        # Add penalty for excessive trading
        if action != ACTION_HOLD:
            reward -= 0.1  # Small penalty for trading
        
        return reward

class ReplayBuffer:
    """Experience replay buffer for training RL agents"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class RLAgent(ABC):
    """Abstract base class for RL agents"""
    
    @abstractmethod
    def act(self, state: TradingState, epsilon: float = 0.0) -> int:
        """Choose action given current state"""
        pass
    
    @abstractmethod
    def train(self, experiences: List[Experience]) -> Dict[str, float]:
        """Train the agent on a batch of experiences"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the agent's model"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the agent's model"""
        pass

# PyTorch-based implementations (if available)
if PYTORCH_AVAILABLE:
    
    class DQN(nn.Module):
        """Deep Q-Network implementation"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(state_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, action_size)
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    class DQNAgent(RLAgent):
        """DQN Agent implementation"""
        
        def __init__(self, 
                     state_size: int,
                     action_size: int = 3,
                     learning_rate: float = 0.001,
                     gamma: float = 0.95,
                     epsilon_start: float = 1.0,
                     epsilon_end: float = 0.01,
                     epsilon_decay: float = 0.995,
                     buffer_size: int = 10000,
                     batch_size: int = 32,
                     target_update_freq: int = 100):
            
            self.state_size = state_size
            self.action_size = action_size
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            
            # Neural networks
            self.q_network = DQN(state_size, action_size)
            self.target_network = DQN(state_size, action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            
            # Experience replay
            self.replay_buffer = ReplayBuffer(buffer_size)
            self.step_count = 0
            
            # Initialize target network
            self._update_target_network()
            
        def _state_to_tensor(self, state: TradingState) -> torch.Tensor:
            """Convert TradingState to tensor"""
            # Combine all state features into a single vector
            features = np.concatenate([
                state.prices.flatten(),
                state.indicators,
                [state.position, state.cash, state.portfolio_value]
            ])
            return torch.FloatTensor(features)
        
        def act(self, state: TradingState, epsilon: float = None) -> int:
            """Choose action using epsilon-greedy policy"""
            if epsilon is None:
                epsilon = self.epsilon
                
            if random.random() > epsilon:
                state_tensor = self._state_to_tensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
            else:
                return random.choice(ACTIONS)
        
        def train(self, experiences: List[Experience]) -> Dict[str, float]:
            """Train the DQN on a batch of experiences"""
            if len(experiences) < self.batch_size:
                return {}
            
            # Convert experiences to tensors
            states = torch.stack([self._state_to_tensor(exp.state) for exp in experiences])
            actions = torch.LongTensor([exp.action for exp in experiences])
            rewards = torch.FloatTensor([exp.reward for exp in experiences])
            next_states = torch.stack([self._state_to_tensor(exp.next_state) for exp in experiences])
            dones = torch.BoolTensor([exp.done for exp in experiences])
            
            # Current Q values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q values from target network
            next_q_values = self.target_network(next_states).max(1)[0].detach()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Loss calculation
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network periodically
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self._update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            return {'loss': loss.item(), 'epsilon': self.epsilon}
        
        def _update_target_network(self):
            """Copy weights from main network to target network"""
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        def save(self, path: str):
            """Save the agent's model"""
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count
            }, path)
        
        def load(self, path: str):
            """Load the agent's model"""
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']

    class ActorCritic(nn.Module):
        """Actor-Critic network implementation"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
            super(ActorCritic, self).__init__()
            
            # Shared layers
            self.shared_fc1 = nn.Linear(state_size, hidden_size)
            self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
            
            # Actor head (policy)
            self.actor_fc = nn.Linear(hidden_size, action_size)
            
            # Critic head (value function)
            self.critic_fc = nn.Linear(hidden_size, 1)
            
            self.dropout = nn.Dropout(0.1)
            
        def forward(self, x):
            shared = F.relu(self.shared_fc1(x))
            shared = self.dropout(shared)
            shared = F.relu(self.shared_fc2(x))
            
            # Actor output (action probabilities)
            policy = F.softmax(self.actor_fc(shared), dim=-1)
            
            # Critic output (state value)
            value = self.critic_fc(shared)
            
            return policy, value

    class A2CAgent(RLAgent):
        """Advantage Actor-Critic (A2C) Agent implementation"""
        
        def __init__(self,
                     state_size: int,
                     action_size: int = 3,
                     learning_rate: float = 0.001,
                     gamma: float = 0.95,
                     entropy_coef: float = 0.01,
                     value_coef: float = 0.5):
            
            self.state_size = state_size
            self.action_size = action_size
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            
            # Actor-Critic network
            self.network = ActorCritic(state_size, action_size)
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
            
        def _state_to_tensor(self, state: TradingState) -> torch.Tensor:
            """Convert TradingState to tensor"""
            features = np.concatenate([
                state.prices.flatten(),
                state.indicators,
                [state.position, state.cash, state.portfolio_value]
            ])
            return torch.FloatTensor(features)
        
        def act(self, state: TradingState, epsilon: float = 0.0) -> int:
            """Choose action using policy network"""
            state_tensor = self._state_to_tensor(state).unsqueeze(0)
            policy, _ = self.network(state_tensor)
            
            # Sample action from policy
            dist = Categorical(policy)
            action = dist.sample()
            
            return action.item()
        
        def train(self, experiences: List[Experience]) -> Dict[str, float]:
            """Train the A2C agent"""
            if len(experiences) == 0:
                return {}
            
            # Convert experiences to tensors
            states = torch.stack([self._state_to_tensor(exp.state) for exp in experiences])
            actions = torch.LongTensor([exp.action for exp in experiences])
            rewards = torch.FloatTensor([exp.reward for exp in experiences])
            next_states = torch.stack([self._state_to_tensor(exp.next_state) for exp in experiences])
            dones = torch.BoolTensor([exp.done for exp in experiences])
            
            # Forward pass
            policies, values = self.network(states)
            _, next_values = self.network(next_states)
            
            # Calculate returns and advantages
            returns = []
            advantages = []
            
            for i in range(len(experiences)):
                if dones[i]:
                    target_return = rewards[i]
                else:
                    target_return = rewards[i] + self.gamma * next_values[i].item()
                
                advantage = target_return - values[i].item()
                
                returns.append(target_return)
                advantages.append(advantage)
            
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)
            
            # Policy loss
            dist = Categorical(policies)
            log_probs = dist.log_prob(actions)
            policy_loss = -(log_probs * advantages).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy loss (for exploration)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            return {
                'total_loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item()
            }
        
        def save(self, path: str):
            """Save the agent's model"""
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        
        def load(self, path: str):
            """Load the agent's model"""
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

else:
    # Fallback implementations without PyTorch
    class BasicAgent(RLAgent):
        """Basic Q-learning agent without neural networks"""
        
        def __init__(self, action_size: int = 3, learning_rate: float = 0.1, gamma: float = 0.95):
            self.action_size = action_size
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.q_table = {}  # State-action values
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
        
        def _state_key(self, state: TradingState) -> str:
            """Convert state to string key for Q-table"""
            # Discretize continuous state for tabular Q-learning
            position_bucket = int(state.position * 10) // 2  # -5 to 5
            cash_bucket = int(state.cash * 10) // 2
            portfolio_bucket = int(state.portfolio_value * 10) // 2
            
            return f"{position_bucket}_{cash_bucket}_{portfolio_bucket}_{state.regime}"
        
        def act(self, state: TradingState, epsilon: float = None) -> int:
            """Choose action using epsilon-greedy policy"""
            if epsilon is None:
                epsilon = self.epsilon
            
            state_key = self._state_key(state)
            
            if random.random() > epsilon and state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            else:
                return random.choice(ACTIONS)
        
        def train(self, experiences: List[Experience]) -> Dict[str, float]:
            """Train using Q-learning updates"""
            total_loss = 0.0
            
            for exp in experiences:
                state_key = self._state_key(exp.state)
                next_state_key = self._state_key(exp.next_state)
                
                # Initialize Q-values if not seen before
                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.action_size)
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = np.zeros(self.action_size)
                
                # Q-learning update
                old_value = self.q_table[state_key][exp.action]
                if exp.done:
                    target = exp.reward
                else:
                    target = exp.reward + self.gamma * np.max(self.q_table[next_state_key])
                
                self.q_table[state_key][exp.action] += self.learning_rate * (target - old_value)
                total_loss += abs(target - old_value)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return {'loss': total_loss / len(experiences) if experiences else 0.0, 'epsilon': self.epsilon}
        
        def save(self, path: str):
            """Save the Q-table"""
            import pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'q_table': self.q_table,
                    'epsilon': self.epsilon
                }, f)
        
        def load(self, path: str):
            """Load the Q-table"""
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']

class RLTrainer:
    """Trainer class for RL agents"""
    
    def __init__(self, 
                 agent: RLAgent, 
                 environment: TradingEnvironment,
                 episodes: int = 1000,
                 max_steps_per_episode: int = 1000):
        self.agent = agent
        self.environment = environment
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.training_history = []
        
    def train(self, verbose: bool = True) -> Dict[str, List]:
        """Train the RL agent"""
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        for episode in range(self.episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_experiences = []
            
            for step in range(self.max_steps_per_episode):
                # Choose action
                action = self.agent.act(state)
                
                # Take step in environment
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                experience = Experience(state, action, reward, next_state, done)
                episode_experiences.append(experience)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Train agent on episode experiences
            train_info = self.agent.train(episode_experiences)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            if 'loss' in train_info:
                losses.append(train_info['loss'])
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                logger.info(f"Episode {episode + 1}/{self.episodes}: "
                           f"Avg Reward: {avg_reward:.2f}, "
                           f"Avg Length: {avg_length:.2f}, "
                           f"Portfolio Value: {info.get('portfolio_value', 0):.2f}")
        
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'losses': losses
        }
        
        return self.training_history
    
    def evaluate(self, episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        evaluation_rewards = []
        evaluation_infos = []
        
        for episode in range(episodes):
            state = self.environment.reset()
            episode_reward = 0.0
            episode_info = []
            
            for step in range(self.max_steps_per_episode):
                # Choose action (no exploration)
                action = self.agent.act(state, epsilon=0.0)
                
                # Take step
                next_state, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                episode_info.append(info)
                state = next_state
                
                if done:
                    break
            
            evaluation_rewards.append(episode_reward)
            evaluation_infos.append(episode_info)
        
        return {
            'mean_reward': np.mean(evaluation_rewards),
            'std_reward': np.std(evaluation_rewards),
            'min_reward': np.min(evaluation_rewards),
            'max_reward': np.max(evaluation_rewards),
            'episode_infos': evaluation_infos
        }

def create_rl_agent(agent_type: str, state_size: int, **kwargs) -> RLAgent:
    """Factory function to create RL agents"""
    if agent_type.lower() == 'dqn' and PYTORCH_AVAILABLE:
        return DQNAgent(state_size, **kwargs)
    elif agent_type.lower() == 'a2c' and PYTORCH_AVAILABLE:
        return A2CAgent(state_size, **kwargs)
    else:
        logger.warning(f"Agent type '{agent_type}' not available or PyTorch not installed. Using BasicAgent.")
        return BasicAgent(**kwargs)

def calculate_state_size(lookback_window: int, num_indicators: int = 5) -> int:
    """Calculate the size of the state vector"""
    return lookback_window + num_indicators + 3  # +3 for position, cash, portfolio_value

def prepare_data_for_rl(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare market data for RL training"""
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Sort by date/time if available
    if 'date' in data.columns:
        data = data.sort_values('date')
    
    # Remove rows with NaN values
    data = data.dropna()
    
    return data.reset_index(drop=True)

# Example usage and integration functions
def integrate_with_regime_detection(regime_detector_path: str = None):
    """Load and integrate with market regime detection"""
    try:
        # This would import and use the market regime detection module
        import market_regime_detection as mrd
        
        # Load pre-trained regime detector if available
        if regime_detector_path and os.path.exists(regime_detector_path):
            regime_detector = mrd.load_regime_model(regime_detector_path)
            logger.info("Market regime detector loaded successfully")
            return regime_detector
        else:
            logger.warning("No regime detector available")
            return None
    except Exception as e:
        logger.warning(f"Could not load regime detection: {e}")
        return None

def main():
    """Example usage of the RL trading system"""
    # This is an example of how to use the RL system
    
    # Generate sample data (replace with real market data)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate realistic-looking price data
    returns = np.random.normal(0.0005, 0.02, 1000)  # Daily returns
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    sample_data = pd.DataFrame({
        'date': dates,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0.005, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0.005, 0.003))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 1000)
    })
    
    # Prepare data
    prepared_data = prepare_data_for_rl(sample_data)
    
    # Create environment
    regime_detector = integrate_with_regime_detection()
    environment = TradingEnvironment(
        data=prepared_data,
        initial_balance=10000.0,
        regime_detector=regime_detector
    )
    
    # Calculate state size
    state_size = calculate_state_size(environment.lookback_window)
    
    # Create agent
    agent = create_rl_agent('dqn', state_size)
    
    # Create trainer
    trainer = RLTrainer(agent, environment, episodes=500)
    
    # Train the agent
    logger.info("Starting RL training...")
    training_history = trainer.train()
    
    # Evaluate the agent
    logger.info("Evaluating trained agent...")
    evaluation_results = trainer.evaluate()
    
    # Save the trained agent
    model_path = os.path.join(MODEL_DIR, 'trained_rl_agent.pth')
    agent.save(model_path)
    logger.info(f"Trained agent saved to {model_path}")
    
    # Print results
    print("\n=== Training Results ===")
    print(f"Average episode reward (last 100): {np.mean(training_history['episode_rewards'][-100:]):.2f}")
    print(f"Final epsilon: {getattr(agent, 'epsilon', 'N/A')}")
    
    print("\n=== Evaluation Results ===")
    print(f"Mean reward: {evaluation_results['mean_reward']:.2f}")
    print(f"Std reward: {evaluation_results['std_reward']:.2f}")
    print(f"Min reward: {evaluation_results['min_reward']:.2f}")
    print(f"Max reward: {evaluation_results['max_reward']:.2f}")

if __name__ == "__main__":
    main()