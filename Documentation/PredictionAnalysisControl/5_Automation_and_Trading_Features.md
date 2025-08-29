# Prediction Analysis Control: Automation and Trading Features

## Introduction

The Prediction Analysis Control (PAC) includes advanced automation and algorithmic trading capabilities, allowing for scheduled analysis, automatic trade execution, and continuous market monitoring. This document details the implementation of these features and their integration with the core prediction framework.

## Automated Mode Architecture

The automated mode functionality is implemented through a dedicated timer-based system:

```csharp
// Automated mode flag
private bool isAutomatedMode = false;

// Timer for automated analysis
private System.Windows.Threading.DispatcherTimer automatedAnalysisTimer;

// Automation interval in minutes (configurable)
private int automationIntervalMinutes = 15; // Default: 15 minutes
```

### Automation Initialization

```csharp
// Initialize the automation timer
private void InitializeAutomationTimer()
{
    automatedAnalysisTimer = new System.Windows.Threading.DispatcherTimer();
    automatedAnalysisTimer.Interval = TimeSpan.FromMinutes(automationIntervalMinutes);
    automatedAnalysisTimer.Tick += AutomatedAnalysisTimer_Tick;
}
```

### Automation Event Handlers

```csharp
/// <summary>
/// Handles the AutoModeToggle checked event, enabling auto mode functionality
/// </summary>
private void AutoModeToggle_Checked(object sender, RoutedEventArgs e)
{
    isAutomatedMode = true;
    StatusText.Text = "Auto mode enabled. Will analyze at regular intervals.";
    DatabaseMonolith.Log("Info", "Auto mode enabled");
    StartAutomatedAnalysisTimer();
}

/// <summary>
/// Handles the AutoModeToggle unchecked event, disabling auto mode functionality
/// </summary>
private void AutoModeToggle_Unchecked(object sender, RoutedEventArgs e)
{
    isAutomatedMode = false;
    StatusText.Text = "Auto mode disabled.";
    DatabaseMonolith.Log("Info", "Auto mode disabled");
    StopAutomatedAnalysisTimer();
}

// Timer tick event handler
private async void AutomatedAnalysisTimer_Tick(object sender, EventArgs e)
{
    await RunAutomatedAnalysis();
}

// Start the automation timer
private void StartAutomatedAnalysisTimer()
{
    if (!automatedAnalysisTimer.IsEnabled)
    {
        automatedAnalysisTimer.Start();
        DatabaseMonolith.Log("Info", $"Started automated analysis timer with interval: {automationIntervalMinutes} minutes");
        
        // Run analysis immediately once when enabling
        Task.Run(async () => await RunAutomatedAnalysis());
    }
}

// Stop the automation timer
private void StopAutomatedAnalysisTimer()
{
    if (automatedAnalysisTimer.IsEnabled)
    {
        automatedAnalysisTimer.Stop();
        DatabaseMonolith.Log("Info", "Stopped automated analysis timer");
    }
}
```

## Automated Analysis Implementation

The core of the automated functionality is the `RunAutomatedAnalysis` method:

```csharp
private async Task RunAutomatedAnalysis()
{
    try
    {
        StatusText.Text = "Running automated analysis...";

        // Get selected symbol
        string selectedSymbol = null;
        if (SymbolFilterComboBox?.SelectedItem is ComboBoxItem symbolItem)
        {
            var content = symbolItem.Content?.ToString();
            if (content != "All Symbols" && content != "Top Market Cap" && content != "Watchlist")
            {
                selectedSymbol = content;
            }
        }

        // Clear existing predictions
        Predictions.Clear();

        // If we have a specific symbol selected
        if (!string.IsNullOrEmpty(selectedSymbol))
        {
            // Run prediction analysis for the selected symbol
            var prediction = await _viewModel.PredictForSymbolAsync(selectedSymbol);
            if (prediction != null)
            {
                Predictions.Add(prediction);

                // Perform sentiment analysis for the selected symbol
                await AnalyzeSentimentPriceCorrelation(selectedSymbol);
            }
        }
        else
        {
            // Run analysis for multiple symbols
            var allPredictions = await _viewModel.PredictForMultipleSymbolsAsync();

            foreach (var prediction in allPredictions)
            {
                Predictions.Add(prediction);
            }

            // If we have predictions, perform sentiment analysis for the first symbol
            if (Predictions.Count > 0)
            {
                await AnalyzeSentimentPriceCorrelation(Predictions[0].Symbol);
            }
        }

        // Set the DataGrid's ItemsSource to the Predictions collection
        PredictionDataGrid.ItemsSource = Predictions;

        // Update UI
        if (LastUpdatedText != null)
            LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm}";

        StatusText.Text = $"Analysis complete. Found {Predictions.Count} predictions.";
        
        // Check for trading signals if auto-trading is enabled
        if (isAutomatedTrading)
        {
            await ProcessTradingSignals();
        }
    }
    catch (Exception ex)
    {
        DatabaseMonolith.Log("Error", "Error in RunAutomatedAnalysis", ex.ToString());
        StatusText.Text = "Error during automated analysis.";
    }
}
```

## Auto-Trading System

The PAC includes sophisticated auto-trading capabilities:

```csharp
// Auto-trading flag
private bool isAutomatedTrading = false;

// Trading parameters
private double minimumTradeConfidence = 0.8;
private int maxConcurrentTrades = 3;
private double maxPositionSize = 0.05; // 5% of portfolio per position
```

### Trading Signal Processing

```csharp
private async Task ProcessTradingSignals()
{
    try
    {
        // Get strong trading signals above confidence threshold
        var tradingSignals = Predictions
            .Where(p => p.Confidence >= minimumTradeConfidence && 
                   p.IsAlgorithmicTradingSignal(minimumTradeConfidence))
            .OrderByDescending(p => p.Confidence)
            .ToList();
            
        StatusText.Text = $"Processing {tradingSignals.Count} trading signals...";
        
        // Get current open positions
        var openPositions = await _viewModel.TradingService.GetOpenPositionsAsync();
        int openPositionsCount = openPositions.Count;
        
        // Check if we can open new positions
        if (openPositionsCount >= maxConcurrentTrades)
        {
            StatusText.Text = $"Maximum concurrent trades reached ({maxConcurrentTrades}). No new positions will be opened.";
            return;
        }
        
        // Available slots for new positions
        int availableSlots = maxConcurrentTrades - openPositionsCount;
        
        // Process signals up to available slots
        int executedTrades = 0;
        foreach (var signal in tradingSignals.Take(availableSlots))
        {
            // Skip if we already have a position in this symbol
            if (openPositions.Any(p => p.Symbol == signal.Symbol))
                continue;
                
            // Calculate position size based on available capital and risk
            double positionSize = CalculatePositionSize(signal);
            
            // Execute the trade
            bool success = await _viewModel.TradingService.ExecuteTradeAsync(
                signal.Symbol,
                signal.PredictedAction,
                signal.CurrentPrice,
                signal.TargetPrice,
                positionSize);
                
            if (success)
            {
                executedTrades++;
                
                // Send notification
                await _notificationService.SendNotificationAsync(
                    $"Auto-Trade Executed: {signal.PredictedAction} {signal.Symbol}",
                    $"Auto-trading system executed {signal.PredictedAction} for {signal.Symbol} at ${signal.CurrentPrice:F2} with target ${signal.TargetPrice:F2}. Confidence: {signal.Confidence:P0}");
            }
        }
        
        StatusText.Text = $"Processed trading signals. Executed {executedTrades} trades.";
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Error processing trading signals");
        StatusText.Text = "Error processing trading signals.";
    }
}

private double CalculatePositionSize(PredictionModel signal)
{
    // Get available capital
    double availableCapital = _viewModel.TradingService.GetAvailableCash();
    
    // Base position size as percentage of portfolio
    double baseSize = availableCapital * maxPositionSize;
    
    // Adjust based on confidence and risk
    double adjustedSize = baseSize * (0.75 + signal.Confidence * 0.5);
    
    // Scale down for higher risk
    if (signal.RiskScore > 0.5)
    {
        adjustedSize *= (1.25 - signal.RiskScore * 0.5);
    }
    
    // Ensure minimum and maximum constraints
    double minSize = availableCapital * 0.01; // Min 1% of portfolio
    double maxSize = availableCapital * maxPositionSize;
    return Math.Min(maxSize, Math.Max(minSize, adjustedSize));
}
```

### Risk Management System

The auto-trading system incorporates sophisticated risk management:

```csharp
private double CalculateStopLossPrice(PredictionModel signal)
{
    if (signal.PredictedAction == "BUY")
    {
        // For buy signals, calculate stop loss based on volatility and risk
        double volatility = signal.Volatility ?? 0.02; // Default to 2% if not available
        double atr = GetAverageTrueRange(signal.Symbol);
        
        // Stop loss is typically 2-3 ATR below entry for long positions
        double stopDistance = Math.Max(atr * 2.5, signal.CurrentPrice * volatility * 2);
        return signal.CurrentPrice - stopDistance;
    }
    else // SELL
    {
        // For short signals, calculate stop above entry
        double volatility = signal.Volatility ?? 0.02;
        double atr = GetAverageTrueRange(signal.Symbol);
        
        double stopDistance = Math.Max(atr * 2.5, signal.CurrentPrice * volatility * 2);
        return signal.CurrentPrice + stopDistance;
    }
}

private double GetAverageTrueRange(string symbol)
{
    try
    {
        // Get ATR from technical indicator service
        return _indicatorService.GetIndicator(symbol, "ATR", 14);
    }
    catch
    {
        // Fallback to 2% of current price if ATR calculation fails
        var price = _alphaVantageService.GetCurrentPrice(symbol);
        return price * 0.02;
    }
}
```

## Trade Order Management

The PAC integrates with order management systems:

```csharp
/// <summary>
/// Executes a trade based on the prediction model
/// </summary>
/// <param name="prediction">The prediction model containing trade details</param>
/// <returns>True if trade successful, false otherwise</returns>
public async Task<bool> ExecuteTrade(PredictionModel prediction)
{
    if (prediction == null)
        return false;

    try
    {
        // Get trade parameters
        string symbol = prediction.Symbol;
        string action = prediction.PredictedAction;
        double price = prediction.CurrentPrice;
        double targetPrice = prediction.TargetPrice;
        
        // Calculate position size
        double positionSize = CalculatePositionSize(prediction);
        
        // Calculate stop loss
        double stopLoss = CalculateStopLossPrice(prediction);
        
        // Create bracket order with target and stop
        var order = new BracketOrderModel
        {
            Symbol = symbol,
            Action = action,
            Quantity = CalculateQuantity(symbol, price, positionSize),
            EntryPrice = price,
            TargetPrice = targetPrice,
            StopPrice = stopLoss,
            OrderType = "MARKET",
            TimeInForce = "GTC",
            ExtendedHours = false
        };
        
        // Submit the order
        string orderId = await _tradingService.SubmitOrderAsync(order);
        
        if (!string.IsNullOrEmpty(orderId))
        {
            LoggingService.Log("Info", $"Successfully submitted trade: {action} {symbol}, OrderId: {orderId}");
            return true;
        }
        
        return false;
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, $"Failed to execute trade for {prediction.Symbol}");
        return false;
    }
}

private int CalculateQuantity(string symbol, double price, double positionSize)
{
    // Round down to whole shares
    return (int)Math.Floor(positionSize / price);
}
```

### Position Monitoring

The PAC continuously monitors open positions:

```csharp
private async Task MonitorOpenPositions()
{
    try
    {
        var openPositions = await _viewModel.TradingService.GetOpenPositionsAsync();
        
        foreach (var position in openPositions)
        {
            // Get latest price
            double currentPrice = await _alphaVantageService.GetCurrentPriceAsync(position.Symbol);
            
            // Calculate profit/loss
            double profitLossPercent = position.Direction == "BUY" ?
                (currentPrice - position.EntryPrice) / position.EntryPrice :
                (position.EntryPrice - currentPrice) / position.EntryPrice;
                
            // Update position tracking
            _positionProfitLoss[position.Symbol] = profitLossPercent;
            
            // Check for exit conditions
            if (ShouldExitPosition(position, currentPrice, profitLossPercent))
            {
                await _viewModel.TradingService.ClosePositionAsync(
                    position.Symbol, position.PositionId);
                    
                LoggingService.Log("Info", $"Automated exit: {position.Symbol} with P/L: {profitLossPercent:P2}");
            }
        }
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Error monitoring open positions");
    }
}

private bool ShouldExitPosition(PositionModel position, double currentPrice, double profitLossPercent)
{
    // Take profit at significant gains
    if (profitLossPercent >= 0.15) // 15% profit
        return true;
        
    // Stop loss at significant drawdown
    if (profitLossPercent <= -0.07) // 7% loss
        return true;
        
    // Time-based exit - close positions held too long
    TimeSpan holdingTime = DateTime.Now - position.EntryTime;
    if (holdingTime.TotalDays > 10) // Max 10-day holding period
        return true;
        
    // Technical indicator based exit
    var indicators = _indicatorService.GetIndicatorsForPrediction(position.Symbol, "1day");
    
    if (position.Direction == "BUY" && 
        indicators.TryGetValue("RSI", out double rsi) && rsi > 75)
        return true;
        
    if (position.Direction == "SELL" && 
        indicators.TryGetValue("RSI", out rsi) && rsi < 25)
        return true;
        
    return false;
}
```

## Automation Settings Management

The PAC includes a settings management system for automation parameters:

```csharp
// Settings management
private void LoadAutomationSettings()
{
    try
    {
        var userSettings = SettingsService.GetDefaultSettingsProfile();
        
        // Load automation interval
        automationIntervalMinutes = userSettings.AutomationIntervalMinutes;
        
        // Load trading parameters
        minimumTradeConfidence = userSettings.MinimumTradeConfidence;
        maxConcurrentTrades = userSettings.MaxConcurrentTrades;
        maxPositionSize = userSettings.MaxPositionSize;
        
        // Update UI controls to reflect settings
        if (AutomationIntervalSlider != null)
            AutomationIntervalSlider.Value = automationIntervalMinutes;
            
        if (TradeConfidenceSlider != null)
            TradeConfidenceSlider.Value = minimumTradeConfidence;
            
        if (MaxTradesNumeric != null)
            MaxTradesNumeric.Value = maxConcurrentTrades;
            
        if (PositionSizeSlider != null)
            PositionSizeSlider.Value = maxPositionSize * 100; // Convert to percentage
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Failed to load automation settings");
    }
}

private void SaveAutomationSettings()
{
    try
    {
        var userSettings = SettingsService.GetDefaultSettingsProfile();
        
        // Update settings
        userSettings.AutomationIntervalMinutes = automationIntervalMinutes;
        userSettings.MinimumTradeConfidence = minimumTradeConfidence;
        userSettings.MaxConcurrentTrades = maxConcurrentTrades;
        userSettings.MaxPositionSize = maxPositionSize;
        
        // Save settings
        SettingsService.SaveUserSettings(userSettings);
        
        // Log success
        LoggingService.Log("Info", "Automation settings saved successfully");
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Failed to save automation settings");
    }
}
```

## Trading Strategy Profiles

The PAC supports configurable trading strategy profiles:

```csharp
// Load strategy profiles
private void LoadStrategyProfiles()
{
    try
    {
        var profiles = StrategyProfileManager.GetAllProfiles();
        
        // Clear existing items
        StrategyProfileComboBox.Items.Clear();
        
        // Add a default option
        StrategyProfileComboBox.Items.Add(new ComboBoxItem 
        { 
            Content = "Default Strategy", 
            Tag = null 
        });
        
        // Add each strategy profile
        foreach (var profile in profiles)
        {
            StrategyProfileComboBox.Items.Add(new ComboBoxItem 
            { 
                Content = profile.Name,
                Tag = profile
            });
        }
        
        // Select default
        StrategyProfileComboBox.SelectedIndex = 0;
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Failed to load strategy profiles");
    }
}

// Handle strategy profile selection
private void StrategyProfileComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
{
    if (StrategyProfileComboBox.SelectedItem is ComboBoxItem item && item.Tag is StrategyProfile profile)
    {
        // Set the selected profile for use in analysis
        _viewModel.SelectedStrategyProfile = profile;
        
        // Update parameters display
        DisplayStrategyParameters(profile);
    }
    else
    {
        // Clear selected profile (use default)
        _viewModel.SelectedStrategyProfile = null;
        ClearStrategyParametersDisplay();
    }
}
```

## Backtesting Integration

The PAC integrates with a backtesting system to validate strategies:

```csharp
private async Task RunBacktest()
{
    try
    {
        StatusText.Text = "Running backtest...";
        
        // Get selected strategy
        var strategy = _viewModel.SelectedStrategyProfile ?? 
            new SmaCrossoverStrategy(); // Default strategy
        
        // Get selected symbol
        string symbol = SymbolFilterComboBox.Text;
        if (string.IsNullOrEmpty(symbol) || symbol == "All Symbols")
        {
            StatusText.Text = "Please select a specific symbol for backtesting.";
            return;
        }
        
        // Get backtest parameters
        DateTime startDate = BacktestStartDatePicker.SelectedDate ?? 
            DateTime.Now.AddYears(-1);
        DateTime endDate = BacktestEndDatePicker.SelectedDate ?? 
            DateTime.Now;
        
        // Run backtest
        var backtestEngine = new BacktestingEngine();
        var result = await backtestEngine.RunBacktest(
            symbol, strategy, startDate, endDate);
        
        // Display backtest results
        DisplayBacktestResults(result);
        
        StatusText.Text = $"Backtest complete. Total return: {result.TotalReturn:P2}";
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Error running backtest");
        StatusText.Text = "Error running backtest.";
    }
}

private void DisplayBacktestResults(BacktestResult result)
{
    // Update results display
    if (BacktestReturnValue != null)
        BacktestReturnValue.Text = result.TotalReturn.ToString("P2");
        
    if (SharpeRatioValue != null)
        SharpeRatioValue.Text = result.SharpeRatio.ToString("F2");
        
    if (MaxDrawdownValue != null)
        MaxDrawdownValue.Text = result.MaxDrawdown.ToString("P2");
        
    if (WinRateValue != null)
        WinRateValue.Text = result.WinRate.ToString("P2");
        
    // Update equity curve chart
    BacktestEquityCurve.Series = new SeriesCollection
    {
        new LineSeries
        {
            Title = "Equity Curve",
            Values = new ChartValues<double>(result.EquityCurve),
            PointGeometry = null
        }
    };
    
    BacktestEquityCurve.AxisX.Labels = result.Dates
        .Select(d => d.ToString("MM/dd")).ToList();
}
```

## Notification System

The PAC includes a comprehensive notification system for automation events:

```csharp
private async Task SendTradeNotification(string symbol, string action, double price, double targetPrice)
{
    try
    {
        // Email notification
        if (_userSettings.EnableEmailAlerts)
        {
            await _emailService.SendEmailAsync(
                _userSettings.AlertEmailAddress,
                $"{action} Alert: {symbol}",
                $"Automated trading system executed {action} for {symbol} at ${price:F2}. Target price: ${targetPrice:F2}");
        }
        
        // SMS notification
        if (_userSettings.EnableSmsAlerts)
        {
            await _smsService.SendSmsAsync(
                _userSettings.AlertPhoneNumber,
                $"{symbol} {action} at ${price:F2}, Target: ${targetPrice:F2}");
        }
        
        // Push notification
        if (_userSettings.EnablePushNotifications)
        {
            await _pushNotificationService.SendNotificationAsync(
                _userSettings.UserId,
                $"{action} Alert: {symbol}",
                $"Trade executed at ${price:F2}");
        }
        
        // Log notification
        LoggingService.Log("Info", $"Sent trade notifications for {symbol} {action}");
    }
    catch (Exception ex)
    {
        LoggingService.LogErrorWithContext(ex, "Failed to send trade notifications");
    }
}
```

## Next Steps

For information on how to configure and extend the PAC for custom implementations, refer to [Configuration and Extension Points](6_Configuration_and_Extension_Points.md).