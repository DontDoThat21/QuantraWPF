# Spread Strategy Backtesting Usage Example

This document provides examples of how to use the new spread strategy backtesting functionality.

## Basic Usage

```csharp
// 1. Create a spread configuration
var spreadConfig = new SpreadConfiguration
{
    SpreadType = MultiLegStrategyType.VerticalSpread,
    UnderlyingSymbol = "AAPL",
    UnderlyingPrice = 150.0,
    Legs = new List<OptionLeg>
    {
        // Bull Call Spread: Buy 145 Call, Sell 155 Call
        new OptionLeg
        {
            Option = new OptionData { StrikePrice = 145, OptionType = "CALL" },
            Action = "BUY",
            Quantity = 1,
            Price = 8.0 // Estimated premium
        },
        new OptionLeg
        {
            Option = new OptionData { StrikePrice = 155, OptionType = "CALL" },
            Action = "SELL", 
            Quantity = 1,
            Price = 3.0 // Estimated premium
        }
    }
};

// 2. Create a spread strategy profile
var spreadStrategy = new SpreadStrategyProfile(spreadConfig, "Bull Call Spread")
{
    TargetProfitPercentage = 0.5,  // Take profit at 50% of max gain
    StopLossPercentage = -2.0,     // Stop loss at 200% of premium paid
    RiskFreeRate = 0.02            // 2% risk-free rate
};

// 3. Get historical data (example)
var historicalData = await historicalDataService.GetHistoricalData("AAPL", "daily");

// 4. Run the spread backtest
var backtestEngine = new BacktestingEngine();
var result = await backtestEngine.RunSpreadBacktestAsync(
    "AAPL", 
    historicalData, 
    spreadStrategy, 
    initialCapital: 10000);

// 5. Analyze results
Console.WriteLine($"Total Return: {result.TotalReturn:P2}");
Console.WriteLine($"Max Drawdown: {result.MaxDrawdown:P2}");
Console.WriteLine($"Spread Trades: {result.SpreadResults.SpreadTrades.Count}");
Console.WriteLine($"Win Rate: {result.SpreadResults.ProfitableTradePercentage:P1}");
Console.WriteLine($"Outperformance vs Equity: {result.SpreadResults.OutperformanceVsEquity:P2}");
```

## Advanced Usage with Transaction Costs

```csharp
// Create a realistic transaction cost model for options
var costModel = new TransactionCostModel
{
    FixedCommission = 1.0,        // $1 per contract
    PercentageCommission = 0.001,  // 0.1% of premium
    BidAskSpread = 0.05           // 5% bid-ask spread
};

var result = await backtestEngine.RunSpreadBacktestAsync(
    "AAPL", 
    historicalData, 
    spreadStrategy,
    initialCapital: 10000,
    costModel: costModel);
```

## Different Spread Types

### Long Straddle
```csharp
var straddleConfig = new SpreadConfiguration
{
    SpreadType = MultiLegStrategyType.Straddle,
    Legs = new List<OptionLeg>
    {
        new OptionLeg { Option = new OptionData { StrikePrice = 150, OptionType = "CALL" }, Action = "BUY" },
        new OptionLeg { Option = new OptionData { StrikePrice = 150, OptionType = "PUT" }, Action = "BUY" }
    }
};
```

### Iron Condor
```csharp
var ironCondorConfig = new SpreadConfiguration
{
    SpreadType = MultiLegStrategyType.IronCondor,
    Legs = new List<OptionLeg>
    {
        new OptionLeg { Option = new OptionData { StrikePrice = 140, OptionType = "PUT" }, Action = "BUY" },
        new OptionLeg { Option = new OptionData { StrikePrice = 145, OptionType = "PUT" }, Action = "SELL" },
        new OptionLeg { Option = new OptionData { StrikePrice = 155, OptionType = "CALL" }, Action = "SELL" },
        new OptionLeg { Option = new OptionData { StrikePrice = 160, OptionType = "CALL" }, Action = "BUY" }
    }
};
```

## Analyzing Results

```csharp
// Access spread-specific metrics
var spreadResults = result.SpreadResults;

// Trade analysis
foreach (var trade in spreadResults.SpreadTrades)
{
    Console.WriteLine($"Entry: {trade.EntryDate:yyyy-MM-dd}, " +
                     $"Exit: {trade.ExitDate:yyyy-MM-dd}, " +
                     $"P&L: {trade.ProfitLoss:C}, " +
                     $"Days Held: {trade.DaysHeld:F1}, " +
                     $"Exit Reason: {trade.ExitReason}");
}

// Rolling P&L visualization data
var rollingPnL = spreadResults.RollingPnL;
foreach (var point in rollingPnL.Take(10)) // First 10 points
{
    Console.WriteLine($"{point.Date:yyyy-MM-dd}: " +
                     $"Cumulative P&L: {point.CumulativePnL:C}, " +
                     $"Open Positions: {point.OpenPositions}, " +
                     $"Underlying Price: {point.UnderlyingPrice:C}");
}

// Performance metrics
Console.WriteLine($"Average Time in Trade: {spreadResults.AverageTimeInTrade:F1} days");
Console.WriteLine($"Average Winning Trade: {spreadResults.AverageWinningTrade:C}");
Console.WriteLine($"Average Losing Trade: {spreadResults.AverageLosingTrade:C}");
Console.WriteLine($"Max Profit: {spreadResults.MaxProfit:C}");
Console.WriteLine($"Max Loss: {spreadResults.MaxLoss:C}");
```

## Comparison with Equity Strategy

```csharp
// Run equivalent equity backtest for comparison
var equityStrategy = new SmaCrossoverStrategy();
var equityResult = await backtestEngine.RunBacktestAsync(
    "AAPL", 
    historicalData, 
    equityStrategy, 
    initialCapital: 10000);

// Compare results
Console.WriteLine($"Spread Strategy Return: {result.TotalReturn:P2}");
Console.WriteLine($"Equity Strategy Return: {equityResult.TotalReturn:P2}");
Console.WriteLine($"Spread Outperformance: {result.SpreadResults.OutperformanceVsEquity:P2}");
```

## Key Features

- **Historical Option Pricing**: Uses Black-Scholes simulation for realistic option valuation
- **Rolling P&L**: Tracks unrealized and realized P&L over time
- **Multiple Exit Conditions**: Target profit, stop loss, time decay, and expiration
- **Risk Management**: Position limits and capital allocation constraints
- **Transaction Costs**: Realistic modeling of commissions and bid-ask spreads
- **Performance Metrics**: Comprehensive spread-specific analytics
- **Equity Comparison**: Direct comparison with buy-and-hold strategy performance