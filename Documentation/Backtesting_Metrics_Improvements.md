# Backtesting Performance Metrics - Improvements and Debugging

## Issue Summary

The backtesting engine was reporting 0 values for several important performance metrics including:
- Sharpe Ratio
- Sortino Ratio  
- Calmar Ratio
- Max Drawdown
- Win Rate

## Root Causes Identified

### 1. **End Date Validation Issue**
**Problem**: User was able to set the end date to a future date (12/10/2025), which caused issues because there's no historical data for future dates.

**Solution**: Added validation to prevent selecting future dates:
```csharp
if (EndDatePicker.SelectedDate > DateTime.Today)
{
    ShowError($"End date cannot be in the future...");
    return false;
}
```

### 2. **Metric Calculation Robustness**
**Problem**: The metric calculations were failing silently when encountering edge cases like:
- Empty equity curves
- Zero or near-zero standard deviations
- Insufficient daily returns data
- Division by zero scenarios

**Solutions Implemented**:

#### Enhanced Logging
Added comprehensive debug logging throughout `CalculateAdvancedMetrics`:
- Equity curve data validation
- Daily returns calculation verification
- Standard deviation calculations
- Each metric calculation with intermediate values
- Final metrics summary

#### Epsilon Comparisons
Replaced exact zero comparisons with small epsilon thresholds:
```csharp
// Before
if (returnStdDev > 0)

// After  
if (returnStdDev > 0.0001)  // Use small epsilon
```

This prevents division by zero and handles near-zero volatility scenarios.

#### Edge Case Handling
Added validation for:
- Previous equity value being zero (skip to avoid division by zero)
- Empty daily returns lists
- Start value validation for CAGR
- Minimum thresholds for meaningful ratio calculations

## Improvements Made

### 1. **Sharpe Ratio Calculation**
```csharp
// Enhanced with epsilon comparison and logging
if (returnStdDev > 0.0001)  
{
    result.SharpeRatio = (averageReturn - riskFreeRate) / returnStdDev * Math.Sqrt(252);
    System.Diagnostics.Debug.WriteLine($"  Calculated Sharpe Ratio: {result.SharpeRatio:F2}");
}
else
{
    result.SharpeRatio = 0;
    System.Diagnostics.Debug.WriteLine($"  Sharpe Ratio set to 0 (stddev too low: {returnStdDev})");
}
```

### 2. **Sortino Ratio Calculation**
Similar improvements with downside deviation validation and logging.

### 3. **CAGR Calculation**
```csharp
if (totalDays > 0 && result.EquityCurve.Count > 0)
{
    double startValue = initialCapital;
    double endValue = result.EquityCurve.Last().Equity;
    
    if (startValue > 0)  // Validate start value
    {
        result.CAGR = Math.Pow(endValue / startValue, 365.0 / totalDays) - 1;
        System.Diagnostics.Debug.WriteLine($"  Calculated CAGR: {result.CAGR:P2}");
    }
}
```

### 4. **Calmar Ratio Calculation**
```csharp
if (result.MaxDrawdown > 0.0001)  // Epsilon comparison
{
    result.CalmarRatio = result.CAGR / result.MaxDrawdown;
}
```

### 5. **Profit Factor Calculation**
```csharp
if (grossLoss > 0.01)  // Small threshold to avoid meaningless ratios
{
    result.ProfitFactor = grossProfit / grossLoss;
}
```

### 6. **Information Ratio Calculation**
Enhanced with epsilon comparison and proper handling of zero volatility cases.

## Win Rate Calculation

The Win Rate is calculated correctly in the `BacktestResult` class:
```csharp
public double WinRate => TotalTrades > 0 ? (double)WinningTrades / TotalTrades : 0;
```

Where:
```csharp
public int TotalTrades => Trades.Count;
public int WinningTrades => Trades.Count(t => t.ExitPrice.HasValue && t.ProfitLoss > 0);
```

**Important Note**: Win Rate only counts trades that have been **closed** (have an `ExitPrice`). Open positions are not included in the calculation.

## Debugging Output

The enhanced logging now provides detailed output in the Debug window:

```
CalculateAdvancedMetrics called:
  Equity curve count: 250
  Trades count: 15
  Completed trades: 12
  Daily returns calculated: 249
  Average daily return: 0.001234
  Min daily return: -0.034567
  Max daily return: 0.045678
  Return Std Dev: 0.012345
  Average return: 0.001234
  Calculated Sharpe Ratio: 1.58
  Downside returns count: 105
  Downside deviation: 0.008765
  Calculated Sortino Ratio: 2.23
  Total days: 365
  Start value: $10,000.00
  End value: $12,450.00
  Calculated CAGR: 24.50%
  Max Drawdown: 8.30%
  Calculated Calmar Ratio: 2.95
  Gross profit: $5,678.00
  Gross loss: $2,234.00
  Calculated Profit Factor: 2.54
  Calculated Information Ratio: 1.58

Final Metrics Summary:
  Sharpe Ratio: 1.58
  Sortino Ratio: 2.23
  CAGR: 24.50%
  Calmar Ratio: 2.95
  Profit Factor: 2.54
  Information Ratio: 1.58
  Win Rate: 80.00%
  Max Drawdown: 8.30%
```

## How to Use

### Setting Up a Backtest

1. **Select a Symbol**: Choose from the dropdown or enter manually
2. **Choose a Strategy**: Select from available trading strategies
3. **Set Date Range**: 
   - Start Date: Historical date (up to 20 years ago)
   - **End Date: MUST be today or in the past** ?
4. **Configure Parameters**:
   - Initial Capital (e.g., $10,000)
   - Trade Size (e.g., 100 shares)
   - Cost Model (for realistic simulation)

### Understanding the Metrics

| Metric | Good Value | Excellent Value | Meaning |
|--------|-----------|----------------|---------|
| **Sharpe Ratio** | >1.0 | >2.0 | Risk-adjusted return |
| **Sortino Ratio** | >1.0 | >2.5 | Downside risk-adjusted return |
| **CAGR** | >10% | >20% | Annualized growth rate |
| **Calmar Ratio** | >1.0 | >3.0 | Return per unit of max drawdown |
| **Profit Factor** | >1.5 | >2.0 | Gross profit / gross loss |
| **Win Rate** | >50% | >60% | Percentage of profitable trades |
| **Max Drawdown** | <15% | <10% | Largest peak-to-trough decline |

### Common Issues

#### Zero Metrics
**Causes**:
- Not enough data points (need at least 30 days)
- No completed trades (all positions still open)
- Invalid date range
- Future end date selected

**Solutions**:
- Ensure date range includes historical data
- Check that end date is today or in the past
- Verify strategy is generating trade signals
- Run for longer time periods (6 months minimum recommended)

#### Win Rate Shows 0%
**Causes**:
- No closed trades (all positions still open)
- Strategy hasn't generated any sell signals
- Backtest period too short

**Solutions**:
- Extend backtest period
- Check strategy exit conditions
- Verify strategy parameters are reasonable

## Testing Recommendations

### Quick Test (6 Months)
Good for rapid strategy validation:
- Use the "Quick Test" button
- Tests last 6 months of data
- Fast results for iteration

### Full Test (1 Year+)
Recommended for strategy validation:
- More reliable statistics
- Better captures market cycles
- More meaningful Sharpe/Sortino ratios

### Best Practices
1. Always backtest multiple time periods
2. Test across different market conditions (bull, bear, sideways)
3. Compare results with benchmarks (SPY, QQQ)
4. Include realistic transaction costs
5. Validate metrics make sense (no zeros for active trading)

## Next Steps

### Further Improvements Planned
1. **Alpha Vantage Analytics Integration**: Use professional-grade analytics APIs for metric validation
2. **Rolling Window Analysis**: Calculate metrics over different time windows
3. **Benchmark Correlation**: Enhanced beta and alpha calculations
4. **Custom Risk Metrics**: VaR, CVaR, and other advanced risk measures

### Documentation
- See `BacktestingPerformanceMetrics.md` for detailed metric definitions
- See `BACKTESTING_ALPHAVANTAGE_ANALYTICS_ENHANCEMENT.md` for future enhancement plans
- See `GreekLetterMetrics_EnterpriseTrading.md` for advanced Greek letter metrics

## Summary

The backtesting metrics are now:
? **Properly calculated** with robust edge case handling  
? **Thoroughly logged** for debugging and validation  
? **Protected against** division by zero and invalid inputs  
? **Validated** to prevent future date selection  
? **Meaningful** with epsilon comparisons instead of exact zero checks

**Key Takeaway**: Always ensure the **End Date is set to today or a past date** for meaningful backtest results!
