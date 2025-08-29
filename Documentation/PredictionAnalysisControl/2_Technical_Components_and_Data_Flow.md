# Prediction Analysis Control: Technical Components and Data Flow

## Component Architecture

The Prediction Analysis Control (PAC) employs a component-based architecture with specialized modules handling different aspects of the analysis workflow. This document provides an in-depth look at the technical components and data flow within the system.

## Core Components

### 1. PredictionAnalysisViewModel

The ViewModel serves as the central coordinator for the PAC, managing data flow between UI components and services:

```csharp
public class PredictionAnalysisViewModel : INotifyPropertyChanged
{
    private readonly ITechnicalIndicatorService _indicatorService;
    private readonly PredictionAnalysisRepository _analysisRepository;
    private readonly IAlphaVantageService _alphaVantageService;
    private readonly IEmailService _emailService;
    public ITradingService TradingService { get; }
    
    // Collection properties exposed to the view
    public ObservableCollection<PredictionModel> Predictions { get; set; }
    public ObservableCollection<PredictionModel> Models { get; private set; }
}
```

Key responsibilities:
- Managing prediction models collection
- Coordinating analysis operations
- Applying filters to prediction results
- Managing service interactions
- Executing trading operations

### 2. PredictionModel

The core data model used throughout the system for prediction data:

```csharp
public class PredictionModel
{
    // Basic prediction info
    public string Symbol { get; set; }
    public string PredictedAction { get; set; } // "BUY", "SELL", "HOLD"
    public double Confidence { get; set; } // 0.0 to 1.0
    public double CurrentPrice { get; set; }
    public double TargetPrice { get; set; }
    public DateTime PredictionDate { get; set; }
    
    // Risk metrics
    public double RiskScore { get; set; }
    public double ValueAtRisk { get; set; }
    public double MaxDrawdown { get; set; }
    public double SharpeRatio { get; set; }
    public double? Volatility { get; set; }
    
    // Technical indicators used for the prediction
    public Dictionary<string, double> Indicators { get; set; }
}
```

Extensions to PredictionModel provide advanced capabilities:
- `CalculateTrendDirection()`: Determines trend direction from indicators
- `EstimateSignalStrength()`: Calculates normalized signal strength
- `IsAlgorithmicTradingSignal()`: Evaluates if prediction is suitable for automated trading
- `AnalyzePredictionQuality()`: Assesses prediction quality based on historical accuracy

### 3. Indicator Modules

The `IndicatorDisplayModule` manages technical indicator visualization:

```csharp
public partial class IndicatorDisplayModule : UserControl, INotifyPropertyChanged
{
    private readonly ITechnicalIndicatorService _indicatorService;
    private readonly INotificationService _notificationService;
    
    // Indicator data
    public Dictionary<string, double> IndicatorValues { get; private set; }
    public List<IndicatorVisualization> VisualizationItems { get; set; }
}
```

This module provides:
- Visual representation of technical indicators
- Correlation analysis between indicators
- Signal strength visualization
- Customizable indicator display

### 4. Chart Module

The `PredictionChartModule` handles time-series visualizations:

```csharp
public partial class PredictionChartModule : UserControl, INotifyPropertyChanged, IDisposable
{
    private readonly ITechnicalIndicatorService _indicatorService;
    private readonly INotificationService _notificationService;
    
    // Chart data
    public SeriesCollection ChartSeries { get; private set; }
    public List<string> Labels { get; private set; }
    public Func<double, string> YFormatter { get; set; }
}
```

Features include:
- Price chart visualization
- Indicator overlay capabilities
- Prediction visualization
- Pattern recognition highlighting
- Interactive zooming and panning

### 5. Sentiment Visualization Control

```csharp
private SentimentVisualizationControl _sentimentVisualizationControl;
```

This specialized control displays:
- Sentiment score visualization
- Sentiment trends over time
- Source-specific sentiment breakdown
- Sentiment-price correlation metrics

### 6. Repository Layer

The `PredictionAnalysisRepository` manages data persistence and retrieval:

```csharp
public class PredictionAnalysisRepository
{
    public void SaveAnalysisResults(IEnumerable<PredictionAnalysisResult> results) { /* ... */ }
    public List<PredictionAnalysisResult> GetLatestAnalyses(int count = 50) { /* ... */ }
    public List<string> GetSymbols() { /* ... */ }
    public PredictionAnalysisResult AnalyzeSymbol(string symbol, Models.StrategyProfile strategy) { /* ... */ }
    public List<HistoricalPrice> GetHistoricalPrices(string symbol) { /* ... */ }
}
```

This component:
- Provides database access for prediction data
- Maintains historical analysis results
- Performs strategy-based symbol analysis
- Manages historical price data

## Data Flow Sequences

### Analysis Flow

The complete data flow during a prediction analysis operation:

```
┌────────────┐  1. Request Analysis  ┌───────────────────┐
│    User    │ ─────────────────────►│ PredictionAnalysis│
│            │                       │     Control       │
└────────────┘                       └───────────────────┘
                                              │
                                              ▼ 2. Delegate to ViewModel
                                     ┌───────────────────┐  3. Get Symbols   ┌───────────────┐
                                     │ PredictionAnalysis│ ────────────────► │ AnalysisRepo  │
                                     │    ViewModel      │                   │               │
                                     └───────────────────┘ ◄────────────────┐└───────────────┘
                                              │             4. Return Symbols
                                              ▼ 5. For Each Symbol
                 ┌────────────────────┬───────┴────────┬───────────────────┐ 
                 │                    │                │                   │
   ┌─────────────▼─────────┐ ┌────────▼───────┐ ┌─────▼───────────┐ ┌─────▼──────────┐
   │ TechnicalIndicatorSvc │ │ AlphaVantage   │ │ SentimentSvc    │ │ PatternRecog   │
   └───────────────────────┘ └────────────────┘ └─────────────────┘ └────────────────┘
   6a. Get Indicators        6b. Get Market    6c. Get Sentiment    6d. Analyze Patterns
          │                      Data               │                     │
          └───────────────┬──────────┬───────────────┘                    │
                          │          │                                    │
                          ▼          ▼                                    │
                ┌───────────────────────────────────┐ 8. Get Risk  ┌─────────────────┐
                │    AnalyzeStockWithAllAlgorithms  │◄────────────►│  Risk Analysis  │
                └───────────────────────────────────┘   Metrics    └─────────────────┘
                                   │
                                   ▼ 9. Create Model
                           ┌──────────────────┐
                           │  PredictionModel │
                           └──────────────────┘
                                   │
               10. Add to Collection │
                           ┌──────────────────┐ 11. Update UI  ┌───────────────────┐
                           │   Predictions    │ ──────────────►│PredictionAnalysis  │
                           │  Collection      │                │    Control UI      │
                           └──────────────────┘                └───────────────────┘
```

### Automated Analysis Workflow

The sequence for automated analysis:

1. `AutoModeToggle_Checked` event triggers automation
2. `StartAutomatedAnalysisTimer` initializes a timer
3. Timer triggers `RunAutomatedAnalysis` at specified intervals
4. Symbol determination (single symbol or multiple)
5. Predictions are generated and added to collection
6. Optional: Execute trades based on predictions

### Trading Execution Flow

When a prediction triggers a trading action:

1. `ExecuteTrade` method is called with prediction model
2. `ValidateTradingConditions` checks if trade is viable
3. `TradingService.ExecuteTradeAsync` sends trade to broker
4. Trade confirmation and status update
5. Results are persisted to repository

## Key Interfaces and Services

The PAC integrates with multiple service interfaces:

```csharp
// Technical indicator calculations
public interface ITechnicalIndicatorService {
    Task<Dictionary<string, double>> GetIndicatorsForPrediction(string symbol, string timeframe);
    Task<Dictionary<string, double>> GetAlgorithmicTradingSignals(string symbol);
}

// Market data provider
public interface IAlphaVantageService {
    Task<Dictionary<string, double>> GetAllTechnicalIndicatorsAsync(string symbol);
}

// Trade execution
public interface ITradingService {
    Task<bool> ExecuteTradeAsync(string symbol, string action, double currentPrice, double targetPrice);
}

// Sentiment analysis
public interface ISocialMediaSentimentService {
    Task<double> GetSentimentScoreAsync(string symbol);
}
```

## Communication Patterns

The PAC employs several communication patterns:

1. **Property Change Notification**: Via `INotifyPropertyChanged` for UI updates
2. **Command Pattern**: Using `ICommand` implementations for user actions
3. **Dependency Injection**: For service integration
4. **Repository Pattern**: For data access abstraction
5. **Asynchronous Tasks**: For non-blocking operations

## Error Handling and Logging

Error handling follows a structured approach:

```csharp
try {
    // Operation code
}
catch (Exception ex) {
    LoggingService.LogErrorWithContext(ex, "Operation description", details,
        memberName, sourceFilePath, sourceLineNumber);
    StatusText = "User-friendly error message";
}
```

Errors are categorized and handled differently based on severity:
- Critical errors: Logged and displayed to user
- Non-critical errors: Logged with reduced visibility
- Service unavailability: Handled with fallbacks

## Next Steps

For more information on the specific algorithms and methodologies used in the PAC, refer to [Algorithms and Analysis Methodologies](3_Algorithms_and_Analysis_Methodologies.md).