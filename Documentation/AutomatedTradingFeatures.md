# Automated Trading Execution Features Guide

## Overview
Quantra provides advanced automated trading execution capabilities to help you implement and manage your trading strategies efficiently. This guide covers the key features of the automated trading system and how to use them in your trading applications.

## Key Features

### 1. Bracket Orders

Bracket orders provide a comprehensive trade management solution by automatically placing stop loss and take profit orders alongside your entry order.

```csharp
// Place a bracket order
await tradingBot.PlaceBracketOrder(
    "AAPL",          // Symbol
    100,             // Quantity
    "BUY",           // Order type
    150.00,          // Entry price
    142.50,          // Stop loss price (5% below entry)
    165.00           // Take profit price (10% above entry)
);
```

For more details, see the [Bracket Order Guide](BracketOrderGuide.md).

### 2. Trailing Stop Orders

Trailing stops automatically adjust as the price moves in your favor, helping to lock in profits while giving trades room to grow.

```csharp
// Set a trailing stop for a long position with 5% trailing distance
tradingBot.SetTrailingStop(
    "MSFT",          // Symbol
    300.00,          // Current price
    0.05,            // Trailing distance (5%)
    "SELL"           // Order type (SELL for long positions)
);

// Set a trailing stop for a short position with 5% trailing distance
tradingBot.SetTrailingStop(
    "MSFT",          // Symbol
    300.00,          // Current price
    0.05,            // Trailing distance (5%)
    "BUY"            // Order type (BUY for short positions)
);
```

The trailing distance (0.05 = 5%) determines how far the stop price will trail behind the highest price (for long positions) or above the lowest price (for short positions). As the price moves favorably, the stop level automatically adjusts to maintain the specified distance, securing profits without limiting upside potential.

You can also retrieve information about currently active trailing stops:

```csharp
// Get information about a trailing stop
var stopInfo = tradingBot.GetTrailingStopInfo("MSFT");
if (stopInfo.HasValue)
{
    var (initialPrice, trailingDistance, currentTriggerPrice) = stopInfo.Value;
    Console.WriteLine($"Trailing stop for MSFT: Initial price {initialPrice:C2}, " +
                      $"Distance {trailingDistance:P2}, Trigger at {currentTriggerPrice:C2}");
}

// Get all symbols with active trailing stops
var symbolsWithTrailingStops = tradingBot.GetSymbolsWithTrailingStops();

// Remove a trailing stop if needed
tradingBot.RemoveTrailingStop("MSFT");
```

### 3. Time-Based Exit Strategies

Schedule automatic exits from positions at specific times or after predefined durations, useful for day trading or avoiding overnight positions.

```csharp
// Set a time-based exit for market close
tradingBot.SetTimeBasedExit(
    "AMZN",                          // Symbol
    DateTime.Today.AddHours(16)      // Exit at 4:00 PM
);

// Set end-of-day exit (automatically exits at 4:00 PM)
tradingBot.SetEndOfDayExit("AMZN");

// Set end-of-week exit (automatically exits at 4:00 PM on Friday)
tradingBot.SetEndOfWeekExit("NVDA");

// Exit after specific duration (e.g., 30 minutes after entry)
tradingBot.SetDurationBasedExit("TSLA", 30);

// Exit at specific time of day (e.g., 3:30 PM every day)
tradingBot.SetSpecificTimeExit("AAPL", new TimeOnly(15, 30));
```

For more advanced configurations, you can use the generic method:

```csharp
// Generic time-based exit strategy method
tradingBot.SetTimeBasedExitStrategy(
    "META",                                    // Symbol
    TimeBasedExitStrategy.SpecificTimeOfDay,   // Exit strategy type
    null,                                      // Duration (not used for this strategy)
    new TimeOnly(15, 45)                       // Specific time (3:45 PM)
);
```

### 4. Position Sizing Based on Risk

Calculate optimal position sizes based on your risk tolerance and account size using multiple algorithms:

```csharp
// Basic fixed risk method (risk 1% of $100,000 account)
int shares = tradingBot.CalculatePositionSizeByRisk(
    "TSLA",          // Symbol
    900.00,          // Current price
    855.00,          // Stop loss price
    0.01,            // Risk percentage (1%)
    100000           // Account size
);

// Advanced position sizing with full parameter control
var parameters = new PositionSizingParameters
{
    Symbol = "AAPL",
    Price = 190.00,
    StopLossPrice = 180.00,
    AccountSize = 100000,
    RiskPercentage = 0.01,
    Method = PositionSizingMethod.VolatilityBased,
    ATR = 5.20,
    ATRMultiple = 2.0,
    RiskMode = RiskMode.Conservative
};
int shares = tradingBot.CalculatePositionSize(parameters);

// Dynamic adaptive risk sizing based on market conditions
int adaptiveShares = tradingBot.CalculatePositionSizeByAdaptiveRisk(
    "MSFT",          // Symbol
    350.00,          // Current price
    340.00,          // Stop loss price
    0.01,            // Base position percentage (1%)
    100000,          // Account size
    -0.3,            // Market volatility factor (-1.0 to 1.0)
    0.4,             // Performance factor (-1.0 to 1.0)
    0.7              // Trend strength factor (0.0 to 1.0)
);
```

Multiple sizing methods are available:
- **FixedRisk**: Standard method based on account risk percentage and stop distance
- **PercentageOfEquity**: Allocates a fixed percentage of account to each position
- **VolatilityBased**: Adjusts position size based on market volatility (ATR)
- **KellyFormula**: Advanced method that optimizes size based on win rate and reward/risk
- **FixedAmount**: Allocates the same dollar amount to each trade
- **TierBased**: Scales position size based on setup confidence
- **AdaptiveRisk**: Dynamic sizing that adjusts to market conditions, volatility, and performance
- **TierBased**: Scales position size based on setup confidence

Risk modes allow for global adjustment of risk parameters:
- **Conservative**: Reduced risk with smaller positions
- **Normal**: Standard risk settings
- **Moderate**: Slightly reduced risk
- **Aggressive**: Higher risk with larger positions
- **GoodFaithValue**: Uses available cash for calculations

See the [Position Sizing Algorithms](PositionSizingAlgorithms.md) documentation for details on each method.

### 5. Dollar-Cost Averaging

Implement dollar-cost averaging strategies to gradually build positions over time.

```csharp
// Set up dollar-cost averaging for 1000 shares over 10 orders, every 7 days
tradingBot.SetupDollarCostAveraging(
    "SPY",           // Symbol
    1000,            // Total shares to acquire
    10,              // Number of orders
    7                // Days between orders
);
```

### 6. Portfolio Rebalancing

Automatically rebalance your portfolio to maintain target allocations.

```csharp
// Set target allocations for portfolio
var allocations = new Dictionary<string, double>
{
    { "AAPL", 0.25 },    // 25% in Apple
    { "MSFT", 0.25 },    // 25% in Microsoft
    { "AMZN", 0.20 },    // 20% in Amazon
    { "GOOGL", 0.20 },   // 20% in Google
    { "TSLA", 0.10 }     // 10% in Tesla
};
tradingBot.SetPortfolioAllocations(allocations);

// Rebalance with 2% tolerance
await tradingBot.RebalancePortfolio(0.02);
```

#### Advanced Rebalancing with Risk Profiles

Create custom rebalancing profiles with risk levels, market condition adjustments, and automatic scheduling:

```csharp
// Create a rebalancing profile
var growthProfile = new RebalancingProfile
{
    Name = "Growth Allocation",
    TargetAllocations = new Dictionary<string, double>
    {
        { "VTI", 0.70 },   // 70% stocks
        { "AGG", 0.20 },   // 20% bonds
        { "GLD", 0.10 }    // 10% gold
    },
    RiskLevel = RebalancingRiskLevel.Growth,
    TolerancePercentage = 0.05,         // 5% drift tolerance
    Schedule = RebalancingSchedule.Monthly,  // Monthly rebalancing
    EnableMarketConditionAdjustments = true  // Auto-adjust for market conditions
};

// Add profile to trading bot
tradingBot.AddRebalancingProfile(growthProfile);

// Set as active profile
tradingBot.SetActiveRebalancingProfile(growthProfile.ProfileId);

// Manually trigger rebalancing
await tradingBot.RebalancePortfolio();   // Uses active profile settings
```

#### Market-Condition Adaptive Rebalancing

The rebalancing system can automatically adjust allocations based on market conditions:

* During high volatility: Shifts toward defensive assets 
* During bearish trends: Reduces exposure to risk assets
* During bullish markets: Maintains or increases growth allocations

Profile settings determine how aggressively allocations adapt to changing market conditions.

### 7. Multi-Leg Orders

Execute complex strategies requiring multiple related orders.

```csharp
// Create a list of orders for a multi-leg strategy
var orders = new List<ScheduledOrder>
{
    new ScheduledOrder { Symbol = "AAPL", Quantity = 100, OrderType = "BUY", Price = 150.00 },
    new ScheduledOrder { Symbol = "MSFT", Quantity = 50, OrderType = "SELL", Price = 300.00 }
};

// Place the multi-leg order
await tradingBot.PlaceMultiLegOrder(orders);
```

### 8. Multi-Leg Strategy Trading

Quantra supports sophisticated multi-leg trading strategies such as spreads, straddles, and pairs trading. These strategies allow you to execute complex, coordinated trades as a single unit.

```csharp
// Create and execute a bull call spread
var verticalSpread = tradingBot.CreateVerticalSpread(
    "AAPL",                   // Symbol
    1,                        // Contracts
    true,                     // Bull call spread (true=call, false=put)
    180.0,                    // Lower strike price
    185.0,                    // Upper strike price
    expirationDate,           // Option expiration
    1.50                      // Desired net debit (optional)
);

// Execute the vertical spread
await tradingBot.PlaceMultiLegStrategy(verticalSpread);

// Create and execute a straddle
var straddle = tradingBot.CreateStraddle(
    "AAPL",                   // Symbol
    1,                        // Contracts
    180.0,                    // Strike price
    expirationDate            // Option expiration
);

// Execute the straddle
await tradingBot.PlaceMultiLegStrategy(straddle);

// Create and execute a pairs trade
var pairsTrade = tradingBot.CreatePairsTrade(
    "AAPL",                   // Long symbol
    "MSFT",                   // Short symbol
    100,                      // Long quantity
    50,                       // Short quantity
    0.85                      // Correlation coefficient
);

// Execute the pairs trade
await tradingBot.PlaceMultiLegStrategy(pairsTrade);

// Create and execute a basket order
var basketOrder = tradingBot.CreateBasketOrder(
    new List<string> { "AAPL", "MSFT", "GOOG", "AMZN" },    // Symbols
    new List<int> { 10, 5, 3, 2 },                          // Quantities
    new List<string> { "BUY", "BUY", "BUY", "BUY" }         // Order types
);

// Execute the basket order
await tradingBot.PlaceMultiLegStrategy(basketOrder);
```

Supported multi-leg strategy types:

- **VerticalSpread**: Bull call spreads and bear put spreads
- **CalendarSpread**: Options with same strike but different expirations
- **DiagonalSpread**: Options with different strikes and expirations
- **Straddle**: Put and call at the same strike and expiration
- **Strangle**: Put and call at different strikes but same expiration
- **IronCondor**: Combination of bull put spread and bear call spread
- **ButterflySpread**: Three strike prices with ratio 1:2:1
- **CoveredCall**: Long stock with short call
- **PairsTrade**: Long one security and short a correlated security
- **BasketOrder**: Multiple related securities traded simultaneously

You can validate a strategy before execution:

```csharp
// Validate the strategy before execution
bool isValid = tradingBot.ValidateMultiLegStrategy(verticalSpread);
if (isValid)
{
    await tradingBot.PlaceMultiLegStrategy(verticalSpread);
}
else
{
    Console.WriteLine("Strategy validation failed");
}
```

### 9. Order Splitting for Large Positions

Split large orders into smaller chunks to minimize market impact.

```csharp
// Basic: Split a large order of 10000 shares into 5 chunks, 15 minutes apart
tradingBot.SplitLargeOrder(
    "SPY",           // Symbol
    10000,           // Total quantity
    "BUY",           // Order type
    420.00,          // Price
    5,               // Number of chunks
    15               // Minutes between chunks
);

// Advanced: Split with price variance and randomized timing to reduce footprint
tradingBot.SplitLargeOrder(
    "SPY",                           // Symbol
    10000,                           // Total quantity
    "BUY",                           // Order type
    420.00,                          // Base price
    5,                               // Number of chunks
    15,                              // Base minutes between chunks
    1.5,                             // Price variance percentage (±1.5%)
    true,                            // Randomize intervals
    OrderDistributionType.Normal     // Bell curve distribution (middle chunks larger)
);

// Example: Front-loaded distribution (larger chunks first)
tradingBot.SplitLargeOrder(
    "AAPL",                          // Symbol
    5000,                            // Total quantity
    "SELL",                          // Order type
    185.00,                          // Base price
    4,                               // Number of chunks
    10,                              // Base minutes between chunks
    0.75,                            // Price variance percentage (±0.75%)
    false,                           // Fixed intervals
    OrderDistributionType.FrontLoaded // Larger chunks at the beginning
);

// Cancel a split order by its group ID
int cancelledChunks = tradingBot.CancelSplitOrderGroup("AAPL-a1b2c3d4e5f6");
```

The enhanced order splitting functionality offers several distribution patterns:

- **Equal**: Orders evenly distributed (default)
- **FrontLoaded**: Larger orders at the beginning, tapering off
- **BackLoaded**: Smaller orders at the beginning, larger at the end
- **Normal**: Bell curve distribution with larger orders in the middle

Combined with price variance and randomized timing, these options help minimize market impact and detection of your trading activity.

### 9. Emergency Stop Functionality

Quickly halt all trading activity in case of emergencies.

```csharp
// Activate emergency stop
tradingBot.ActivateEmergencyStop();

// Check if emergency stop is active
bool isActive = tradingBot.IsEmergencyStopActive();

// Deactivate emergency stop when ready to resume trading
tradingBot.DeactivateEmergencyStop();
```

### 10. Trading Hour Restrictions

Restrict trading to specific market hours and sessions.

```csharp
// Set trading hours to regular market hours (9:30 AM - 4:00 PM)
tradingBot.SetTradingHourRestrictions(
    new TimeOnly(9, 30),     // Market open at 9:30 AM
    new TimeOnly(16, 0)      // Market close at 4:00 PM
);

// Enable specific market sessions (pre-market, regular, after-hours)
tradingBot.SetEnabledMarketSessions(MarketSession.Regular | MarketSession.AfterHours);

// Enable all market sessions
tradingBot.SetEnabledMarketSessions(MarketSession.All);

// Configure custom market session time boundaries
tradingBot.SetMarketSessionTimes(
    new TimeOnly(4, 0),      // Pre-market open at 4:00 AM
    new TimeOnly(9, 30),     // Regular market open at 9:30 AM
    new TimeOnly(16, 0),     // Regular market close at 4:00 PM
    new TimeOnly(20, 0)      // After-hours close at 8:00 PM
);

// Get current market session times
var sessionTimes = tradingBot.GetMarketSessionTimes();
Console.WriteLine($"Pre-market: {sessionTimes.preMarketOpen} to {sessionTimes.regularMarketOpen}");
Console.WriteLine($"Regular: {sessionTimes.regularMarketOpen} to {sessionTimes.regularMarketClose}");
Console.WriteLine($"After-hours: {sessionTimes.regularMarketClose} to {sessionTimes.afterHoursClose}");

// Check if trading is currently allowed based on time and enabled sessions
bool canTrade = tradingBot.IsTradingAllowed();
```

The market session filters allow you to determine which parts of the trading day to enable:

- `MarketSession.PreMarket` - Trading during pre-market hours (customizable, default: 4:00 AM - 9:30 AM)
- `MarketSession.Regular` - Trading during regular market hours (customizable, default: 9:30 AM - 4:00 PM)
- `MarketSession.AfterHours` - Trading during after-hours (customizable, default: 4:00 PM - 8:00 PM)
- `MarketSession.All` - Trading during all sessions
- `MarketSession.None` - Trading disabled in all sessions
- `MarketSession.Regular` - Trading during regular market hours (9:30 AM - 4:00 PM)
- `MarketSession.AfterHours` - Trading during after-hours (4:00 PM - 8:00 PM)
- `MarketSession.All` - Trading during all sessions
- `MarketSession.None` - Trading disabled in all sessions

## Best Practices

1. **Risk Management**: Always use position sizing and stop losses to manage risk.
2. **Testing**: Test automated strategies in paper trading mode before using real money.
3. **Monitoring**: Regularly monitor automated trading systems for unexpected behavior.
4. **Logging**: Enable detailed logging to track all trading decisions and actions.
5. **Emergency Procedures**: Have clear procedures for activating emergency stops and handling system failures.
6. **Time-Based Exit Strategy Selection**: Choose appropriate time-based exit strategies based on your trading style:
   - Use EndOfDay for day trading to avoid overnight risk
   - Use Duration for short-term scalping or momentum trades
   - Use EndOfWeek for swing trades you want to close before the weekend
   - Use SpecificTimeOfDay for trades that should exit before specific market events

## Error Handling

The trading bot methods return boolean values or tasks with boolean results to indicate success or failure. Always check these return values and handle errors appropriately.

```csharp
// Example error handling
try
{
    bool success = await tradingBot.PlaceBracketOrder("AAPL", 100, "BUY", 150.00, 142.50, 165.00);
    if (!success)
    {
        // Handle failure to place order
        LogError("Failed to place bracket order for AAPL");
    }
}
catch (Exception ex)
{
    // Handle unexpected exceptions
    LogError($"Exception placing bracket order: {ex.Message}");
}
```

## Integration with Prediction Analysis

The automated trading system integrates with Quantra's prediction analysis system to make data-driven trading decisions. See the `PredictionAnalysisControl.Trading.cs` file for examples of how to use prediction models to drive trading decisions.

```csharp
// Example integration with prediction model
private async Task ExecuteAutomatedTrade(PredictionModel prediction)
{
    // Calculate position size based on prediction confidence
    int quantity = CalculatePositionSize(prediction);
    
    // Determine if a trailing stop should be used
    bool useTrailingStop = ShouldUseTrailingStop(prediction);
    
    if (useTrailingStop)
    {
        // Place the order with a trailing stop
        await tradingBot.PlaceLimitOrder(
            prediction.Symbol,
            quantity,
            prediction.PredictedAction,
            prediction.CurrentPrice
        );
        
        // Calculate appropriate trailing distance based on volatility and confidence
        double trailingDistance = CalculateTrailingStopDistance(prediction);
        
        // For BUY orders, we use SELL for trailing stop
        // For SELL orders, we use BUY for trailing stop
        string trailingStopOrderType = prediction.PredictedAction == "BUY" ? "SELL" : "BUY";
        
        tradingBot.SetTrailingStop(
            prediction.Symbol,
            prediction.CurrentPrice,
            trailingDistance,
            trailingStopOrderType
        );
    }
    else
    {
        // Place a bracket order with fixed stop loss and take profit
        double stopLoss = CalculateStopLoss(prediction);
        double takeProfit = CalculateTakeProfit(prediction);
        
        await tradingBot.PlaceBracketOrder(
            prediction.Symbol,
            quantity,
            prediction.PredictedAction,
            prediction.CurrentPrice,
            stopLoss,
            takeProfit
        );
    }
}
```

## Conclusion

Quantra's automated trading execution system provides a comprehensive set of tools for implementing sophisticated trading strategies. By leveraging these features, you can build robust, automated trading solutions that manage risk effectively while capturing opportunities in the market.