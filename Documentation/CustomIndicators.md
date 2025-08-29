# Custom Indicators for Quantra

This document describes how to create, customize, and use custom indicators in the Quantra trading platform.

## Overview

Quantra allows you to create your own technical indicators through a composable interface. You can:

- Create indicators from scratch
- Combine existing indicators
- Customize parameters
- Visualize results
- Use custom indicators in trading strategies and backtests

## Creating Custom Indicators

### Using the Indicator Builder

The easiest way to create a custom indicator is through the Indicator Builder UI:

1. Open the Indicator Builder from the main menu (Analysis → Indicator Builder)
2. Drag existing indicators or operations from the left panel to the design canvas
3. Connect indicators to create a workflow
4. Configure parameters
5. Test your indicator against historical data
6. Save the indicator for later use

### Through Code

For advanced users, indicators can be created programmatically:

```csharp
// Simple example: Creating a composite indicator that averages RSI and MACD
var avgIndicator = new CompositeIndicator(
    name: "RSI-MACD Average", 
    calculationMethod: (inputs) => {
        var rsiValue = inputs["RSI"]["Value"];
        var macdValue = inputs["MACD"]["Value"];
        return (rsiValue + macdValue) / 2;
    },
    dependencies: new List<string> { "RSI", "MACD" },
    category: "Custom");

// Register with the indicator service
var indicatorService = ServiceLocator.GetService<ITechnicalIndicatorService>();
await indicatorService.RegisterIndicatorAsync(avgIndicator);
```

## Indicator Types

### Built-in Indicators

All standard technical indicators can be used as building blocks:
- RSI, MACD, Bollinger Bands, etc.

### Composite Indicators

These combine multiple indicators with operations:
- Average of multiple indicators
- Ratio between indicators
- Threshold-based signals
- Custom formulas

### Template Indicators

Pre-configured popular custom indicators:
- MACD Histogram with RSI Filter
- Triple Moving Average System
- Volatility-adjusted momentum

## Example Indicators

### Weighted Average Indicator

```csharp
// Combines two indicators with weighted average
var weightedAvg = new AverageIndicator(
    firstIndicatorId: "RSI",
    secondIndicatorId: "MFI",
    weightFirst: 0.7,
    name: "RSI-MFI Weighted"
);
```

### Relative Strength Indicator

```csharp
// Measures one symbol/indicator relative to another
var relativeStrength = new RelativeStrengthIndicator(
    primaryIndicatorId: "AAPL_Close",  
    referenceIndicatorId: "SPY_Close",  
    name: "AAPL vs SPY Strength"
);
```

## Parameter Customization

Each indicator can expose parameters that can be adjusted:

- Period lengths (like SMA periods)
- Thresholds for signals
- Weights for component indicators
- Sources of price data (Open, High, Low, Close)

## Backtesting with Custom Indicators

Custom indicators can be used in backtesting:

```csharp
// Create a strategy using custom indicator
var strategy = new StrategyProfile();
strategy.AddIndicatorCondition("MyCustomIndicator", value => value > 70, "BUY");
strategy.AddIndicatorCondition("MyCustomIndicator", value => value < 30, "SELL");

// Run backtest
var engine = new BacktestingEngine();
var result = await engine.RunBacktestAsync("AAPL", historicalPrices, strategy);
```

## Best Practices

1. **Keep it simple** - Start with simple combinations before building complex indicators
2. **Validate with history** - Test indicators against historical data
3. **Avoid overfitting** - Focus on robust indicators that work across market conditions
4. **Document your indicators** - Add clear descriptions and usage notes
5. **Share with community** - Export and share successful indicators

## Indicator Visualization

Custom indicators can be visualized using the standard charting tools:

1. Open any chart view
2. Right-click and select "Add Indicator"
3. Choose your custom indicator from the list
4. Configure display properties (color, style, placement)

## Troubleshooting

Common issues:

- **Circular references**: Ensure indicators don't create circular dependencies
- **Performance**: Complex indicators may slow down analysis; consider optimization
- **Parameter sensitivity**: Test different parameter values to ensure stability