# Backtesting with Premium Alpha Vantage API

This guide explains how to utilize the extended backtesting capabilities in Quantra that are enabled by the Alpha Vantage Premium API.

## Premium API Features

The Alpha Vantage Premium API integration provides several advantages for backtesting:

- **More comprehensive historical data** with adjusted prices
- **Multiple asset classes** including stocks, forex, and cryptocurrencies
- **Extended timeframes** for more robust testing
- **Adjusted price information** for accurate dividend and split handling
- **Higher API rate limits** for faster data retrieval

## Using Premium Features in Your Code

### Example: Running a Comprehensive Backtest

```csharp
// Create instances of required services
var backtestingEngine = new BacktestingEngine();
var strategy = new SmaCrossoverStrategy 
{
    FastPeriod = 10,
    SlowPeriod = 30
};

// For stock backtesting
var stockResult = await backtestingEngine.RunComprehensiveBacktestAsync(
    "MSFT",     // Symbol
    strategy,   // Strategy to test
    "daily",    // Interval (1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)
    "stock",    // Asset class
    100000      // Initial capital
);

// For forex backtesting
var forexResult = await backtestingEngine.RunComprehensiveBacktestAsync(
    "EUR/USD",  // Currency pair
    strategy,   // Strategy to test
    "daily",    // Interval
    "forex",    // Asset class
    100000      // Initial capital
);

// For cryptocurrency backtesting
var cryptoResult = await backtestingEngine.RunComprehensiveBacktestAsync(
    "BTC",      // Symbol
    strategy,   // Strategy to test
    "daily",    // Interval
    "crypto",   // Asset class
    100000      // Initial capital
);

// Let the engine auto-detect the asset class
var autoResult = await backtestingEngine.RunComprehensiveBacktestAsync(
    "AAPL",     // Symbol
    strategy,   // Strategy to test
    "daily",    // Interval
    "auto",     // Auto-detect asset class
    100000      // Initial capital
);
```

### Manual Access to Extended Historical Data

If you need to directly access the historical data:

```csharp
var historicalDataService = new HistoricalDataService();

// Get extended stock data
var stockData = await historicalDataService.GetComprehensiveHistoricalData("AAPL", "daily", "stock");

// Get forex data
var forexData = await historicalDataService.GetForexHistoricalData("EUR", "USD", "daily");

// Get cryptocurrency data
var cryptoData = await historicalDataService.GetCryptoHistoricalData("BTC", "USD", "daily");
```

## Premium API Key Configuration

To utilize the premium features, set your Alpha Vantage Premium API key in one of the following ways:

1. **Environment Variable**: Set the `ALPHA_VANTAGE_API_KEY` environment variable with your premium key.

2. **API Key Format**: If your API key starts with "PREMIUM_", it will automatically be recognized as a premium key.

3. **Premium Flag**: Set the `ALPHA_VANTAGE_PREMIUM` environment variable to "true".

## Checking Premium Status

You can check if you're using a premium API key:

```csharp
var historicalDataService = new HistoricalDataService();
bool isPremium = historicalDataService.IsPremiumKey();

// Or directly from the AlphaVantageService
var alphaVantageService = new AlphaVantageService();
bool isPremium = alphaVantageService.IsPremiumKey;
```

## Benefits of Premium API for Backtesting

1. **More accurate results** due to complete historical data
2. **Dividend-adjusted prices** for more realistic equity returns
3. **Split-adjusted data** to avoid misleading price jumps
4. **Multi-asset backtesting** for diversified strategies
5. **Longer historical periods** for better strategy validation