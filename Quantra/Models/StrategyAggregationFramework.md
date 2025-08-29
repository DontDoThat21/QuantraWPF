# Strategy Aggregation Framework

This document describes the Strategy Aggregation Framework, which allows combining signals from multiple trading strategies.

## Overview

The Strategy Aggregation Framework provides a way to combine trading signals from multiple strategies using different aggregation methods. This can help reduce false signals and increase confidence in trading decisions.

## Key Components

### AggregatedStrategyProfile

The `AggregatedStrategyProfile` class is the core component of the framework. It:
- Inherits from `StrategyProfile` 
- Manages a collection of underlying strategies
- Implements various methods to combine signals from these strategies

### StrategyWeight

The `StrategyWeight` class represents a strategy with an associated weight. Higher weights give strategies more influence in the aggregation process.

## Aggregation Methods

The framework supports these aggregation methods:

1. **Majority Vote**: The signal returned by the most strategies wins
   ```csharp
   // Example: 3 BUY, 2 SELL signals = BUY signal returned
   ```

2. **Consensus**: Only returns a signal if a certain percentage of strategies agree
   ```csharp
   // Example with 75% threshold: 4 BUY, 1 SELL signals = BUY signal returned
   // Example with 75% threshold: 3 BUY, 2 SELL signals = no signal returned
   ```

3. **Weighted Vote**: Like majority vote but strategies have weights
   ```csharp
   // Example: Strategy 1 (weight 3) = BUY, Strategy 2 (weight 2) = SELL, Strategy 3 (weight 1) = SELL
   // Result: BUY signal (3 vs 3) with tie-breaking based on implementation
   ```

4. **Priority Based**: Higher priority strategies can override lower ones
   ```csharp
   // Example: Strategy 1 (high priority) = BUY, Strategy 2 (low priority) = SELL
   // Result: BUY signal
   ```

## Using the Framework

### Creating an Aggregated Strategy

```csharp
// Create a new aggregated strategy
var aggregatedStrategy = new AggregatedStrategyProfile
{
    Name = "Combined Strategy",
    Description = "Combines RSI and MACD signals",
    Method = AggregatedStrategyProfile.AggregationMethod.MajorityVote
};

// Add underlying strategies with weights
aggregatedStrategy.AddStrategy(rsiStrategy, 1.0);
aggregatedStrategy.AddStrategy(macdStrategy, 1.5);  // MACD has higher weight
```

### Setting the Aggregation Method

```csharp
// Use consensus-based aggregation that requires 70% agreement
aggregatedStrategy.Method = AggregatedStrategyProfile.AggregationMethod.Consensus;
aggregatedStrategy.ConsensusThreshold = 0.7;
```

### Generating Signals

The aggregated strategy is used just like any other strategy:

```csharp
string signal = aggregatedStrategy.GenerateSignal(historicalPrices);
```

## Benefits

1. **Reduced false signals**: By requiring multiple strategies to agree
2. **Increased robustness**: Less dependent on a single strategy's potential weaknesses
3. **Adaptability**: Can be configured for different market conditions
4. **Flexibility**: Easy to add/remove/weight strategies as needed

## Considerations

- Adding more strategies increases computational requirements
- Too many strategies may create "analysis paralysis" with conflicting signals
- Weight calibration requires careful testing and optimization