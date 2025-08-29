# Backtesting Performance Metrics

This document provides an overview of the advanced performance metrics used in the backtesting module of Quantra.

## Basic Metrics

### Total Return
The overall percentage return achieved during the backtest period.

### Max Drawdown
The largest percentage drop from a peak to a subsequent trough in the portfolio's value.

### Win Rate
The percentage of trades that were profitable.

## Advanced Performance Metrics

### Sharpe Ratio
A measure of risk-adjusted return. It quantifies the return of an investment compared to its risk.

- **Formula**: (Average Return - Risk-Free Rate) / Standard Deviation of Returns
- **Interpretation**: Higher is better. A Sharpe ratio of 1.0 is considered good, 2.0 is very good, and 3.0 is excellent.
- **Usage**: Helps compare strategies with different risk profiles.

### Sortino Ratio
Similar to the Sharpe ratio but only considers downside risk (negative returns).

- **Formula**: (Average Return - Risk-Free Rate) / Standard Deviation of Negative Returns
- **Interpretation**: Higher is better. It focuses on harmful volatility.
- **Usage**: Useful when upside volatility is not a concern.

### CAGR (Compound Annual Growth Rate)
The annualized rate of return that would have been required for an investment to grow from its beginning value to its ending value.

- **Formula**: (Ending Value / Beginning Value)^(1/Years) - 1
- **Interpretation**: Higher is better. Represents the "smoothed" annual growth rate.
- **Usage**: Comparing strategies over different time periods.

### Calmar Ratio
Measures the relationship between the CAGR and the maximum drawdown.

- **Formula**: CAGR / Maximum Drawdown
- **Interpretation**: Higher is better. Indicates return per unit of downside risk.
- **Usage**: Particularly useful for strategies that aim to minimize drawdowns.

### Profit Factor
The ratio of gross profits to gross losses.

- **Formula**: Gross Profits / Gross Losses
- **Interpretation**: Greater than 1 is profitable. The higher, the more profitable the strategy.
- **Usage**: Quick assessment of strategy profitability.

### Information Ratio
Measures the risk-adjusted excess return relative to a benchmark.

- **Formula**: (Portfolio Return - Benchmark Return) / Tracking Error
- **Interpretation**: Higher is better. Shows how much excess return is generated per unit of risk relative to the benchmark.
- **Usage**: Evaluating a strategy's performance against a benchmark.

## Benchmark Comparison Metrics

### Beta
A measure of the volatility, or systematic risk, of a security or portfolio compared to the market as a whole.

- **Formula**: Covariance(Strategy Returns, Benchmark Returns) / Variance(Benchmark Returns)
- **Interpretation**: 
  - Beta = 1: The strategy moves with the market
  - Beta < 1: The strategy is less volatile than the market
  - Beta > 1: The strategy is more volatile than the market

### Alpha
The excess return of an investment relative to the return of a benchmark index.

- **Formula**: Strategy Return - [Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate)]
- **Interpretation**: Positive alpha indicates the strategy outperformed the benchmark on a risk-adjusted basis.

### Correlation
The degree to which the strategy's returns move in relation to the benchmark's returns.

- **Formula**: Covariance(Strategy Returns, Benchmark Returns) / (Standard Deviation(Strategy Returns) * Standard Deviation(Benchmark Returns))
- **Interpretation**: 
  - 1.0: Perfect positive correlation
  - 0.0: No correlation
  - -1.0: Perfect negative correlation

## Custom Benchmarks

Quantra supports creating custom benchmarks for more relevant performance comparisons. Custom benchmarks can be created in several ways:

### Types of Custom Benchmarks

1. **Sector/Industry Benchmarks**: Create benchmarks that represent specific sectors or industries relevant to your strategy.
2. **Asset Class Benchmarks**: Create multi-asset benchmarks that better match your asset allocation.
3. **Strategy-Specific Benchmarks**: Create benchmarks tailored to the specific markets or instruments your strategy trades.
4. **Weighted Indices**: Create custom-weighted combinations of existing indices.

### Creating a Custom Benchmark

To create a custom benchmark:

1. Click on "Manage Custom Benchmarks..." in the Backtesting Results view.
2. Click "Create New Benchmark" in the Benchmark Manager.
3. Provide a name, description, and category for your benchmark.
4. Add components to your benchmark, specifying symbol, name, and weight for each.
5. Use "Equal Weight" to assign equal weighting to all components.
6. Use "Normalize Weights" to ensure weights sum to 100%.
7. Save the benchmark for future use.

### Using Custom Benchmarks

Once created, custom benchmarks can be selected from the dropdown in the Backtesting Results view, and will be included in all performance comparisons and charts alongside standard benchmarks.

### Performance Calculation

Performance metrics for custom benchmarks are calculated by:

1. Loading historical data for all components.
2. Aligning dates across components to find a common date range.
3. Creating a weighted average price for each date based on component weights.
4. Calculating performance metrics (return, drawdown, volatility, etc.) from this composite data.
5. Comparing the strategy's performance to this custom benchmark.

Custom benchmarks provide a more accurate assessment of your strategy's performance against relevant competitors or market segments.

## Advanced Greek Letter Metrics

For enterprise-tier trading applications, Quantra supports advanced Greek letter metrics that go beyond traditional performance measures. These sophisticated metrics enable systematic outperformance through mathematical precision and comprehensive risk management.

### Quick Reference to Advanced Greeks

- **Alpha (α)**: Excess return generation and factor-based strategies
- **Beta (β)**: Smart beta strategies and dynamic market exposure
- **Sigma (σ)**: Volatility trading and risk premium capture  
- **Omega (Ω)**: Advanced risk-return optimization considering all distribution moments
- **Gamma (Γ)**: Convexity trading and portfolio acceleration
- **Delta (Δ)**: Directional exposure and delta-neutral strategies
- **Theta (Θ)**: Time decay strategies and income generation
- **Vega (ν)**: Volatility sensitivity and vol surface trading
- **Rho (ρ)**: Interest rate sensitivity and macro strategies

For comprehensive documentation on implementing these advanced metrics for optimal trading performance, see [Greek Letter Metrics for Enterprise Trading](GreekLetterMetrics_EnterpriseTrading.md).

### Enterprise Applications

The advanced Greek metrics enable:
- **Systematic Alpha Generation**: Factor-based models and cross-asset arbitrage
- **Dynamic Risk Management**: Real-time Greek-neutral positioning
- **Volatility as an Asset Class**: Sophisticated volatility trading strategies
- **Portfolio Optimization**: Multi-dimensional risk-return optimization
- **Institutional-Grade Strategies**: Enterprise-tier performance analytics