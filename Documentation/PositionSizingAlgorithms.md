# Position Sizing Algorithms

This document explains the various position sizing algorithms implemented in the Quantra trading platform.

## Overview

Position sizing is a critical component of risk management in trading. A good position sizing strategy helps to:
- Manage risk per trade
- Optimize capital allocation
- Maintain consistency in trading
- Adapt to market conditions

The Quantra platform offers several position sizing algorithms to accommodate different trading styles and risk preferences.

## Available Position Sizing Methods

### Fixed Risk

This is the standard method where you risk a fixed percentage of your account on each trade.

```csharp
// Calculate position size (risk 1% of $100,000 account with $5 risk per share)
int shares = tradingBot.CalculatePositionSizeByRisk(
    "AAPL",         // Symbol
    150.00,         // Current price
    145.00,         // Stop loss price
    0.01,           // Risk percentage (1%)
    100000          // Account size
);
// Result: 200 shares
```

**Calculation:**
- Risk per share = |Entry price - Stop loss price|
- Risk amount = Account size × Risk percentage
- Position size = Risk amount ÷ Risk per share

**Advantages:**
- Consistent risk exposure across all trades
- Simple to calculate and implement
- Automatically adjusts position size based on stop loss distance

### Percentage of Equity

This method allocates a fixed percentage of your account to each position, regardless of stop loss distance.

```csharp
var parameters = new PositionSizingParameters {
    Symbol = "MSFT",
    Price = 350.00,
    AccountSize = 100000,
    RiskPercentage = 0.05,    // 5% of account
    Method = PositionSizingMethod.PercentageOfEquity
};
int shares = tradingBot.CalculatePositionSize(parameters);
// Result: 14 shares
```

**Calculation:**
- Position value = Account size × Allocation percentage
- Position size = Position value ÷ Current price

**Advantages:**
- Simple and consistent position sizing
- Works well for portfolio allocation strategies
- Not dependent on stop loss placement

### Volatility-Based (ATR)

This method uses the Average True Range (ATR) to adjust position size based on market volatility.

```csharp
var parameters = new PositionSizingParameters {
    Symbol = "TSLA",
    Price = 250.00,
    AccountSize = 100000,
    RiskPercentage = 0.01,    // 1% of account
    Method = PositionSizingMethod.VolatilityBased,
    ATR = 10.50,              // Current ATR value
    ATRMultiple = 2.0         // 2 × ATR for stop distance
};
int shares = tradingBot.CalculatePositionSize(parameters);
```

**Calculation:**
- Risk per share = ATR × ATR multiple
- Risk amount = Account size × Risk percentage
- Position size = Risk amount ÷ Risk per share

**Advantages:**
- Adapts to changing market volatility
- Smaller positions in volatile markets
- Larger positions in less volatile markets
- More objective stop placement based on actual market conditions

### Kelly Formula

This method uses the Kelly Criterion to optimize position size based on the probability of winning and the reward/risk ratio.

```csharp
var parameters = new PositionSizingParameters {
    Symbol = "AMZN",
    Price = 180.00,
    StopLossPrice = 175.00,
    AccountSize = 100000,
    Method = PositionSizingMethod.KellyFormula,
    WinRate = 0.60,           // 60% win rate
    RewardRiskRatio = 2.0,    // 2:1 reward-to-risk ratio
    KellyFractionMultiplier = 0.5  // Half-Kelly for more conservative sizing
};
int shares = tradingBot.CalculatePositionSize(parameters);
```

**Calculation:**
- Kelly percentage = (Win rate × Reward/risk ratio - (1 - Win rate)) ÷ Reward/risk ratio
- Adjusted Kelly = Kelly percentage × Kelly fraction multiplier
- Position value = Account size × Adjusted Kelly
- Position size = Position value ÷ (Risk per share × Reward/risk ratio)

**Advantages:**
- Mathematically optimal position sizing
- Accounts for both win rate and reward/risk ratio
- Helps maximize long-term growth
- Can be adjusted using the Kelly fraction multiplier for more conservative sizing

### Fixed Amount

This method simply allocates a fixed dollar amount to each trade.

```csharp
var parameters = new PositionSizingParameters {
    Symbol = "QQQ",
    Price = 400.00,
    Method = PositionSizingMethod.FixedAmount,
    FixedAmount = 10000      // $10,000 per trade
};
int shares = tradingBot.CalculatePositionSize(parameters);
// Result: 25 shares
```

**Calculation:**
- Position size = Fixed amount ÷ Current price

**Advantages:**
- Extremely simple to implement
- Consistent dollar exposure
- Easy to track and manage

### Tier-Based

This method adjusts position size based on the confidence level of the trading signal.

```csharp
var parameters = new PositionSizingParameters {
    Symbol = "NVDA",
    Price = 800.00,
    StopLossPrice = 780.00,
    AccountSize = 100000,
    RiskPercentage = 0.01,   // 1% base risk
    Method = PositionSizingMethod.TierBased,
    Confidence = 0.92        // 92% confidence in the signal
};
int shares = tradingBot.CalculatePositionSize(parameters);
```

**Calculation:**
- Tier determination based on confidence level
- Adjusted risk percentage based on confidence tier
- Position size calculated using the adjusted risk percentage

**Advantages:**
- Adapts position size to signal quality
- Larger positions for higher-probability setups
- Smaller positions for lower-probability setups
- Maintains risk management while exploiting better opportunities

### Adaptive Risk

This advanced method dynamically calculates position size based on multiple factors including market volatility, recent performance, and trend strength.

```csharp
// Advanced adaptive risk position sizing
int shares = tradingBot.CalculatePositionSizeByAdaptiveRisk(
    "AAPL",         // Symbol
    150.00,         // Current price
    145.00,         // Stop loss price
    0.01,           // Base position percentage (1%)
    100000,         // Account size
    -0.3,           // Market volatility factor (-1.0 to 1.0, negative means decreasing volatility)
    0.4,            // Recent performance factor (-1.0 to 1.0, positive means recent gains)
    0.7             // Trend strength factor (0.0 to 1.0, higher means stronger trend)
);
```

**Calculation:**
- Start with base position percentage
- Apply volatility adjustment (reduce size in high volatility, increase in low volatility)
- Apply performance adjustment (reduce after losses, carefully increase after gains)
- Apply trend strength adjustment (increase in stronger trends)
- Apply minimum and maximum constraints for safety
- Calculate shares based on adjusted risk percentage

**Advantages:**
- Truly dynamic position sizing that adapts to changing market conditions
- Automatically scales down in high-risk environments
- Capitalizes on favorable market conditions with larger positions
- Responds to performance feedback to manage drawdowns
- Combines multiple factors for a more holistic approach to risk management
- Maintains risk boundaries with min/max constraints

## Risk Modes

The platform supports different risk modes that can further adjust position sizing:

- **Normal**: Standard risk parameters
- **Conservative**: Reduced risk (50% of normal risk percentage, 70% of normal maximum size)
- **Moderate**: Slightly reduced risk (80% of normal risk)
- **Aggressive**: Increased risk (150% of normal risk percentage, 130% of normal maximum size)
- **GoodFaithValue**: Uses available cash rather than full account size

## Maximum Position Size Limitations

All position sizing methods are subject to a maximum position size constraint to prevent over-concentration in a single position. By default, no position can exceed 20% of the account size, but this can be configured in the user settings.

## Configuration

Users can configure position sizing methods and parameters in the Configuration screen:
- Account size
- Base risk percentage
- Position sizing method
- Maximum position size percentage
- Fixed trade amount
- ATR multiple
- Kelly Criterion parameters (win rate, reward/risk ratio, Kelly fraction)
- Adaptive risk parameters (volatility sensitivity, performance impact, trend strength weight)