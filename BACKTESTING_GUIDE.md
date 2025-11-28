# Backtesting Module - Complete Implementation Guide

## 📋 Overview

The Backtesting module has been completely restructured and is now **fully functional**. It provides an end-to-end workflow for:
1. Configuring backtest parameters
2. Selecting trading strategies
3. Running backtests on historical data
4. Analyzing comprehensive results with benchmarks

---

## 🗂️ Directory Structure

**Location:** `Quantra\Views\Backtesting\`

### ✅ Active Files (KEEP)

1. **BacktestConfiguration.xaml/cs** ⭐ **NEW - Main Entry Point**
   - Primary UI for configuring and running backtests
   - User selects symbol, strategy, dates, capital, etc.
   - Executes backtests and displays results
   - **This is where users start**

2. **BacktestResults.xaml/cs**
   - Comprehensive results visualization
   - Shows performance metrics, charts, drawdowns
   - Benchmark comparison with multiple indices
   - Monte Carlo simulation support
   - **Automatically loaded after backtest completes**

3. **CustomBenchmarkManager.xaml/cs**
   - Manage custom benchmark portfolios
   - Create weighted combinations of stocks/ETFs
   - Compare strategy against custom indices
   - **Opened from BacktestResults**

4. **CustomBenchmarkDialog.xaml/cs**
   - Dialog for creating/editing custom benchmarks
   - Add/remove components with weights
   - Validate benchmark configurations
   - **Opened from CustomBenchmarkManager**

### ❌ Removed Files (DELETED)

5. ~~**MultiStrategyComparisonControl.xaml/cs**~~ ✂️ **DELETED**
   - Was never used in the codebase
   - Functionality not needed (replaced by better workflow)

---

## 🔄 Complete Workflow

### Step 1: User Opens Backtesting Tab
```
MainWindow → Backtesting Tab → BacktestConfiguration Control Loads
```

### Step 2: User Configures Backtest
**In BacktestConfiguration UI:**
- Enter stock symbol (e.g., "AAPL")
- Select strategy from dropdown (e.g., "SMA Crossover 20/50")
- Set initial capital ($10,000 default)
- Set trade size (100 shares default)
- Choose date range (1 year back by default)
- Select cost model (Retail/Zero Cost/Fixed/Percentage)
- Adjust strategy-specific parameters (if available)

### Step 3: Run Backtest
**User clicks "Run Backtest" or "Quick Test (6 months)"**
```
BacktestConfiguration.cs:
  1. Validates inputs
  2. Fetches historical data via HistoricalDataService
  3. Creates BacktestingEngine
  4. Runs backtest with selected strategy
  5. Generates BacktestResult object
```

### Step 4: View Results
**Results automatically display in embedded BacktestResults control:**
- **Performance Metrics:** Total Return, CAGR, Sharpe Ratio, Win Rate, etc.
- **Charts:** Equity curve, Drawdown, Price with trade markers
- **Benchmark Comparison:** Compare vs SPY, QQQ, IWM, DIA, or custom
- **Advanced Analysis:** Monte Carlo simulation, correlation, alpha/beta

### Step 5: (Optional) Create Custom Benchmarks
**User clicks "Manage Custom Benchmarks" in BacktestResults:**
```
CustomBenchmarkManager → CustomBenchmarkDialog
  - Create weighted portfolio (e.g., 60% SPY + 40% QQQ)
  - Save and compare against strategy
```

---

## 🎯 Key Features

### BacktestConfiguration Features
✅ **Strategy Selection**
- Dropdown of all available strategies from StrategyProfileManager
- Built-in strategies: SMA Crossover, MACD, RSI Divergence, Bollinger Bands, etc.
- Dynamic parameter controls based on selected strategy

✅ **Flexible Configuration**
- Any stock symbol (auto-fetches data)
- Custom date ranges or Quick Test (6 months)
- Multiple cost models (Zero, Retail, Fixed, Percentage)
- Adjustable capital and trade size

✅ **User-Friendly UX**
- Real-time validation
- Progress indicators during execution
- Clear error messages
- Empty state when no results

### BacktestResults Features
✅ **Comprehensive Metrics**
- Total Return, Max Drawdown, Win Rate
- Advanced: Sharpe, Sortino, Calmar, Information Ratio
- CAGR (Compound Annual Growth Rate)
- Profit Factor

✅ **Visual Analysis**
- Price chart with buy/sell markers
- Equity curve visualization
- Drawdown timeline
- Multiple benchmark overlays

✅ **Benchmark Comparison (6 Tabs)**
1. Performance Summary
2. Cumulative Returns
3. Drawdown Comparison
4. Volatility Comparison
5. Performance Attribution
6. Risk-Adjusted Metrics
7. Monte Carlo Simulation

✅ **Monte Carlo Simulation**
- Run 100-10,000 simulations
- Return distribution histogram
- Percentile analysis (5%, 25%, 50%, 75%, 95%)
- Value at Risk (VaR) calculations
- Probability metrics

---

## 💻 Code Architecture

### Data Flow
```
User Input (BacktestConfiguration)
    ↓
HistoricalDataService.GetComprehensiveHistoricalData(symbol)
    ↓
BacktestingEngine.RunBacktestAsync(symbol, data, strategy, params)
    ↓
BacktestResult Object (trades, equity curve, metrics)
    ↓
BacktestResults.LoadResults(result, historical)
    ↓
Charts & Analysis Display
```

### Key Classes

**BacktestConfiguration.xaml.cs**
```csharp
public BacktestConfiguration(
    IUserSettingsService userSettingsService,
    LoggingService loggingService,
    IAlphaVantageService alphaVantageService)
```
- Loads strategies via `StrategyProfileManager.Instance.GetProfiles()`
- Creates `BacktestingEngine` and runs backtest
- Instantiates `BacktestResults` control and loads data

**BacktestingEngine.cs** (Quantra.DAL)
```csharp
public async Task<BacktestResult> RunBacktestAsync(
    string symbol,
    List<HistoricalPrice> historical,
    StrategyProfile strategy,
    double initialCapital = 10000,
    int tradeSize = 1,
    TransactionCostModel costModel = null)
```
- Core backtesting logic
- Simulates trades based on strategy signals
- Calculates all performance metrics
- Supports transaction costs (commission, spread, slippage)

**StrategyProfile.cs** (Base Class)
```csharp
public abstract class StrategyProfile
{
    public abstract string GenerateSignal(List<HistoricalPrice> prices, int? index = null);
    public abstract bool ValidateConditions(Dictionary<string, double> indicators);
}
```
- All strategies inherit from this
- Implement `GenerateSignal()` to return "BUY", "SELL", or null

### Available Strategies (Built-in)
1. **SmaCrossoverStrategy** - Simple Moving Average crossover
2. **MacdCrossoverStrategy** - MACD indicator strategy
3. **RsiDivergenceStrategy** - RSI-based mean reversion
4. **BollingerBandsMeanReversionStrategy** - Bollinger Bands
5. **IchimokuCloudStrategy** - Ichimoku indicator
6. **ParabolicSARStrategy** - Parabolic SAR
7. **StochRsiSwingStrategy** - Stochastic RSI
8. **VolumeBreakoutStrategy** - Volume-based entries
9. **EmaCrossoverStrategy** - Exponential MA crossover
10. **AdxTrendStrengthStrategy** - ADX trend following
11. **FibonacciRetracementStrategy** - Fibonacci levels
12. **SupportResistanceStrategy** - S/R breakouts
13. **VwapAnchoredStrategy** - VWAP-based
14. **WmaCrossoverStrategy** - Weighted MA crossover
15. **EmaSmaCrossStrategy** - EMA/SMA combination
16. **AggregatedStrategyProfile** - Multi-strategy aggregator
17. **SpreadStrategyProfile** - Options spread strategies

---

## 🚀 Usage Examples

### Example 1: Simple SMA Crossover Backtest
```csharp
// In BacktestConfiguration UI:
Symbol: AAPL
Strategy: SMA Crossover (20/50)
Initial Capital: $10,000
Trade Size: 100 shares
Start Date: 1 year ago
End Date: Today
Cost Model: Retail Brokerage

// Click "Run Backtest"
// Results automatically display below
```

### Example 2: Programmatic Backtest
```csharp
// Create services
var historicalService = new HistoricalDataService(userSettings, logging);
var engine = new BacktestingEngine(historicalService);

// Get data
var historical = await historicalService.GetComprehensiveHistoricalData("MSFT");

// Create strategy
var strategy = new SmaCrossoverStrategy
{
    FastPeriod = 20,
    SlowPeriod = 50
};

// Run backtest
var result = await engine.RunBacktestAsync(
    "MSFT",
    historical,
    strategy,
    initialCapital: 10000,
    tradeSize: 50,
    TransactionCostModel.CreateRetailBrokerageModel()
);

// Access results
Console.WriteLine($"Total Return: {result.TotalReturn:P2}");
Console.WriteLine($"Sharpe Ratio: {result.SharpeRatio:F2}");
Console.WriteLine($"Max Drawdown: {result.MaxDrawdown:P2}");
Console.WriteLine($"Win Rate: {result.WinRate:P2}");
Console.WriteLine($"Total Trades: {result.TotalTrades}");
```

### Example 3: Custom Strategy
```csharp
public class MyCustomStrategy : StrategyProfile
{
    public MyCustomStrategy()
    {
        Name = "My Custom Strategy";
        Description = "Buy on breakout, sell on mean reversion";
    }

    public override string GenerateSignal(List<HistoricalPrice> prices, int? index = null)
    {
        int idx = index ?? prices.Count - 1;
        if (idx < 20) return null;

        // Calculate 20-day moving average
        double ma20 = prices.Skip(idx - 20).Take(20).Average(p => p.Close);

        // Buy if price crosses above MA
        if (prices[idx].Close > ma20 && prices[idx - 1].Close <= ma20)
            return "BUY";

        // Sell if price crosses below MA
        if (prices[idx].Close < ma20 && prices[idx - 1].Close >= ma20)
            return "SELL";

        return null;
    }

    public override bool ValidateConditions(Dictionary<string, double> indicators)
    {
        return true;
    }
}

// Use in BacktestConfiguration:
// 1. Register strategy: StrategyProfileManager.Instance.SaveProfile(new MyCustomStrategy());
// 2. It will appear in the dropdown
// 3. Run backtest as normal
```

---

## 📊 Understanding Results

### Performance Metrics Explained

**Total Return**: Overall profit/loss as percentage of initial capital
**CAGR**: Annualized return (accounts for compounding)
**Max Drawdown**: Largest peak-to-trough decline
**Win Rate**: Percentage of winning trades
**Sharpe Ratio**: Return per unit of risk (>1 is good, >2 is excellent)
**Sortino Ratio**: Like Sharpe but only penalizes downside volatility
**Calmar Ratio**: Annual return / Max Drawdown (higher is better)
**Profit Factor**: Gross profit / Gross loss (>1 is profitable)
**Information Ratio**: Excess return vs benchmark per unit of tracking error

### Benchmark Comparison

**Standard Benchmarks:**
- SPY: S&P 500 ETF
- QQQ: NASDAQ-100 ETF
- IWM: Russell 2000 ETF
- DIA: Dow Jones ETF

**Custom Benchmarks:**
Create weighted portfolios like:
- 60% SPY + 40% AGG (balanced portfolio)
- 50% QQQ + 30% IWM + 20% DIA (custom index)

### Monte Carlo Simulation

**Purpose:** Assess strategy robustness by randomizing trade order

**Key Metrics:**
- **Median Return (50%)**: Most likely outcome
- **VaR 95%**: Maximum expected loss at 95% confidence
- **CVaR 95%**: Average of worst 5% outcomes
- **Probability of Profit**: Chance of positive returns

---

## 🛠️ Troubleshooting

### Issue: "No historical data found for symbol"
**Solution:**
- Verify symbol is correct (e.g., "AAPL" not "Apple")
- Check internet connection (data fetched from Alpha Vantage)
- Ensure Alpha Vantage API key is configured in settings

### Issue: "Insufficient data. Need at least 30 days"
**Solution:**
- Expand date range (need minimum 30 trading days)
- Choose a more liquid stock with complete history

### Issue: Strategy dropdown is empty
**Solution:**
- Built-in strategies are automatically loaded
- If using custom strategies, ensure they're registered:
  ```csharp
  StrategyProfileManager.Instance.SaveProfile(myStrategy);
  ```

### Issue: Backtest runs but shows poor performance
**Suggestions:**
- Try different strategies
- Adjust strategy parameters
- Test on different time periods
- Compare against benchmarks to validate
- Check if transaction costs are too high

---

## 🎓 Best Practices

1. **Always Use Transaction Costs**
   - Zero-cost backtests are unrealistic
   - Use "Retail Brokerage" for most accurate simulation

2. **Test Multiple Time Periods**
   - Bull markets (2019-2021)
   - Bear markets (2022)
   - Volatile periods (2020)

3. **Compare Against Benchmarks**
   - A good strategy should beat buy-and-hold
   - Check correlation with SPY (low is better for diversification)

4. **Run Monte Carlo**
   - Validates strategy isn't just lucky
   - Shows range of possible outcomes

5. **Watch for Overfitting**
   - If results are "too good to be true", they probably are
   - Test on out-of-sample data

6. **Consider Max Drawdown**
   - High returns with 50% drawdown = risky
   - Target DD < 20% for conservative strategies

---

## 🔮 Future Enhancements

Potential additions (not yet implemented):
- Walk-forward optimization
- Parameter optimization grid search
- Multi-symbol portfolio backtesting
- Options strategies backtesting (partial support exists)
- ML-based strategy integration
- Export results to PDF/Excel
- Strategy comparison side-by-side

---

## 📝 Summary

**You are now ready to use the Backtesting Module!**

1. ✅ Open the Backtesting tab in MainWindow
2. ✅ Configure your backtest settings
3. ✅ Select a strategy
4. ✅ Click "Run Backtest"
5. ✅ Analyze comprehensive results
6. ✅ Compare against benchmarks
7. ✅ Run Monte Carlo for validation

**The complete workflow is now functional from start to finish!**

---

*Generated: 2025-11-28*
*Module Version: 2.0 (Complete Rewrite)*
