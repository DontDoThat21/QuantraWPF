# Bracket Order Trading Guide

## Overview
Bracket orders provide a comprehensive trade management solution by automatically placing stop loss and take profit orders alongside your entry order. This documentation covers how to use the bracket order functionality in Quantra.

## Benefits of Bracket Orders
- **Risk Management**: Automatically set stop loss levels to limit potential losses
- **Profit Targeting**: Define take profit levels for systematic profit capture
- **Simplified Trade Management**: Set and forget - no need to manually place exit orders
- **Emotion-Free Trading**: Exit decisions are predetermined, removing emotional decision making

## Using Bracket Orders in Quantra

### Manual Trading
1. Open the Orders page
2. Enter your trade details (Symbol, Quantity, Order Type, Price)
3. Set your Stop Loss level
4. Set your Take Profit level
5. Click "Place Order" to execute the bracket order

When both Stop Loss and Take Profit values are set, the system will automatically create a bracket order.

### Automated Trading
The prediction system automatically calculates appropriate stop loss and take profit levels based on:
- Order type (Buy/Sell)
- Current market conditions
- Asset volatility
- Target prices from predictions

## Stop Loss Calculation Logic
- For Buy orders: Stop loss is typically set at 5% below entry price
- For Sell orders: Stop loss is typically set at 5% above entry price

## Take Profit Calculation Logic
- For Buy orders: Take profit is set to the target price from prediction or 10% above entry price
- For Sell orders: Take profit is set to the target price from prediction or 10% below entry price

## Advanced Features
- **Trailing Stops**: Stop loss levels can automatically move as price moves favorably
- **OCO Orders (One-Cancels-Other)**: When either stop loss or take profit is triggered, the other is automatically canceled
- **Time-Based Exits**: Automatically exit positions at specific times
- **Position Sizing**: Calculate position size based on risk parameters
- **Multi-Leg Orders**: Place related orders as part of a coordinated strategy
- **Order Splitting**: Break large orders into smaller chunks to minimize market impact

## Code Usage

### Basic Bracket Order
```csharp
// Example of placing a bracket order in code
bool success = await tradingBot.PlaceBracketOrder(
    "AAPL",          // Symbol
    100,             // Quantity
    "BUY",           // Order type
    150.00,          // Entry price
    142.50,          // Stop loss price (5% below entry)
    165.00           // Take profit price (10% above entry)
);
```

### Trailing Stop
```csharp
// Set a trailing stop with 5% trailing distance
tradingBot.SetTrailingStop(
    "MSFT",          // Symbol
    300.00,          // Current price
    0.05             // Trailing distance (5%)
);
```

### Time-Based Exit
```csharp
// Set a time-based exit for market close
tradingBot.SetTimeBasedExit(
    "AMZN",                          // Symbol
    DateTime.Today.AddHours(16)      // Exit at 4:00 PM
);
```

### Position Sizing Based on Risk
```csharp
// Calculate position size (risk 1% of $100,000 account)
int shares = tradingBot.CalculatePositionSizeByRisk(
    "TSLA",          // Symbol
    900.00,          // Current price
    855.00,          // Stop loss price
    0.01,            // Risk percentage (1%)
    100000           // Account size
);
```

### Order Splitting
```csharp
// Split a large order of 10000 shares into 5 chunks, 15 minutes apart
tradingBot.SplitLargeOrder(
    "SPY",           // Symbol
    10000,           // Total quantity
    "BUY",           // Order type
    420.00,          // Price
    5,               // Number of chunks
    15               // Minutes between chunks
);
```

## Best Practices
1. Always use bracket orders for risk management
2. Set stop loss based on technical levels, not arbitrary percentages
3. Consider using a risk-reward ratio of at least 1:2 (risk 1% to gain 2%)
4. Adjust take profit levels based on key resistance/support levels
5. For volatile stocks, consider wider stop loss levels to avoid premature exits