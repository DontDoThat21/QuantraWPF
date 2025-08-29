using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using Quantra.Enums;
using Quantra.Models;
using Quantra.Services;

namespace Quantra.Controls
{
    public partial class PredictionAnalysisControl
    {
        private WebullTradingBot _tradingBot;
        private StockDataCacheService _stockDataCache;
        private Dictionary<string, DateTime> _lastTradeTime = new Dictionary<string, DateTime>();
        private bool _isAutoTradingEnabled = false;

        /// <summary>
        /// Initializes trading components for the PredictionAnalysisControl
        /// </summary>
        private void InitializeTradingComponents()
        {
            _tradingBot = new WebullTradingBot();
            _stockDataCache = new StockDataCacheService();
            
            // Default to paper trading mode
            _tradingBot.SetTradingMode(Quantra.Enums.TradingMode.Paper);
            
            DatabaseMonolith.Log("Info", "Trading components initialized in PredictionAnalysisControl");
        }

        /// <summary>
        /// Handles enabling auto trading mode after auto mode is enabled
        /// </summary>
        internal void HandleAutoModeEnabled()
        {
            _isAutoTradingEnabled = true;
            DatabaseMonolith.Log("Info", "Auto trading mode enabled");

            // Ensure trading components are initialized before starting auto trading
            if (_tradingBot == null || _stockDataCache == null)
                InitializeTradingComponents();

            // Update UI elements to indicate auto trading is enabled
            if (StatusText != null)
                StatusText.Text = "Auto trading mode enabled - Monitoring predictions...";

            // Start the auto trading process
            StartAutoTrading();
        }

        /// <summary>
        /// Handles disabling auto trading mode after auto mode is disabled
        /// </summary>
        internal void HandleAutoModeDisabled()
        {
            _isAutoTradingEnabled = false;
            DatabaseMonolith.Log("Info", "Auto trading mode disabled");
            
            // Update UI elements to indicate auto trading is disabled
            if (StatusText != null)
                StatusText.Text = "Auto trading mode disabled";
            
            // Stop the auto trading process
            StopAutoTrading();
        }

        /// <summary>
        /// Starts auto trading based on predictions with trading rules
        /// </summary>
        private async void StartAutoTrading()
        {
            if (!_isAutoTradingEnabled) return;

            // Ensure trading components are initialized before auto trading
            if (_tradingBot == null || _stockDataCache == null)
                InitializeTradingComponents();

            try
            {
                // Get all predictions with trading rules
                var tradablePredictions = predictions?
                    .Where(p => !string.IsNullOrWhiteSpace(p.TradingRule))
                    .ToList() ?? new List<PredictionModel>();
                
                if (tradablePredictions.Count == 0)
                {
                    DatabaseMonolith.Log("Info", "No predictions with trading rules found for auto trading");
                    return;
                }
                
                DatabaseMonolith.Log("Info", $"Starting auto trading for {tradablePredictions.Count} predictions");
                
                foreach (var prediction in tradablePredictions)
                {
                    // Check if we've traded this symbol too recently
                    if (HasTradedRecently(prediction.Symbol))
                    {
                        continue;
                    }
                    
                    // Check for stock data
                    var stockData = await GetStockDataForSymbol(prediction.Symbol);
                    if (stockData == null || stockData.Count == 0)
                    {
                        DatabaseMonolith.Log("Warning", $"No stock data available for {prediction.Symbol}");
                        continue;
                    }
                    
                    // Evaluate trading rule
                    if (ShouldExecuteTrade(prediction, stockData))
                    {
                        // Execute trade 
                        await ExecuteAutomatedTrade(prediction);
                        
                        // Record the time of this trade to avoid excessive trading
                        _lastTradeTime[prediction.Symbol] = DateTime.Now;
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error during auto trading", ex.ToString());
            }
        }

        /// <summary>
        /// Stops the auto trading process
        /// </summary>
        private void StopAutoTrading()
        {
            // Nothing to specifically stop if using event-based monitoring
            DatabaseMonolith.Log("Info", "Auto trading stopped");
        }

        /// <summary>
        /// Gets historical stock data for a symbol, using cache if available
        /// </summary>
        private async Task<List<HistoricalPrice>> GetStockDataForSymbol(string symbol)
        {
            try
            {
                // Get stock data from cache or API
                return await _stockDataCache.GetStockDataAsync(symbol);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting stock data for {symbol}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Determines if a trade should be executed based on prediction and current data
        /// </summary>
        private bool ShouldExecuteTrade(PredictionModel prediction, List<HistoricalPrice> stockData)
        {
            // Simple implementation for a trading rule
            // In a real system, this would parse the TradingRule text and apply logic
            
            try
            {
                // Example rule parsing (very basic)
                var rule = prediction.TradingRule?.ToLower() ?? "";
                var latestPrice = stockData.LastOrDefault()?.Close ?? 0;
                
                // Basic string parsing - in a real system this would be more sophisticated
                if (prediction.PredictedAction == "BUY")
                {
                    // For buy signals, check if the current price is at or below our target entry
                    if (rule.Contains("buy below") || rule.Contains("entry below"))
                    {
                        // Try to parse a price value
                        var pricePart = rule.Split(new[] { "below" }, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
                        if (pricePart != null && double.TryParse(pricePart.Trim(), out double targetPrice))
                        {
                            // If current price is below the specified entry point, trade
                            return latestPrice <= targetPrice;
                        }
                    }
                    
                    // Default buy rule - if confidence is high and no specific rule parsed
                    return prediction.Confidence > 0.85;
                }
                else if (prediction.PredictedAction == "SELL")
                {
                    // For sell signals, check if the current price is at or above our target entry
                    if (rule.Contains("sell above") || rule.Contains("entry above"))
                    {
                        // Try to parse a price value
                        var pricePart = rule.Split(new[] { "above" }, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
                        if (pricePart != null && double.TryParse(pricePart.Trim(), out double targetPrice))
                        {
                            // If current price is above the specified entry point, trade
                            return latestPrice >= targetPrice;
                        }
                    }
                    
                    // Default sell rule - if confidence is high and no specific rule parsed
                    return prediction.Confidence > 0.85;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error evaluating trading rule for {prediction.Symbol}", ex.ToString());
            }
            
            // Default to not trading if there's any issue
            return false;
        }

        /// <summary>
        /// Executes a trade based on a prediction
        /// </summary>
        private async Task ExecuteAutomatedTrade(PredictionModel prediction)
        {
            try
            {
                // Ensure trading components are initialized before executing a trade
                if (_tradingBot == null)
                {
                    DatabaseMonolith.Log("Error", "Trading bot is not initialized. Cannot execute trade.");
                    return;
                }

                // Calculate standard position size (simple implementation)
                int quantity = CalculatePositionSize(prediction);
                
                // Calculate price - use current price for market orders
                double price = prediction.CurrentPrice;
                
                // Calculate stop loss and take profit levels
                double stopLoss = CalculateStopLoss(prediction);
                double takeProfit = CalculateTakeProfit(prediction);
                
                // Determine if we should use a trailing stop for this prediction
                bool useTrailingStop = ShouldUseTrailingStop(prediction);
                
                DatabaseMonolith.Log("Info", $"Executing automated {prediction.PredictedAction} trade for {prediction.Symbol}: {quantity} shares at {price:C2}");
                
                // Create order model
                var order = new OrderModel
                {
                    Symbol = prediction.Symbol,
                    OrderType = prediction.PredictedAction,
                    Quantity = quantity,
                    Price = price,
                    StopLoss = stopLoss,
                    TakeProfit = takeProfit,
                    IsPaperTrade = true,  // Always use paper trade mode for automated trades
                    PredictionSource = $"Auto: {prediction.Symbol} ({prediction.Confidence:P0})",
                    Status = "New",
                    Timestamp = DateTime.Now
                };

                // Execute the trade via the trading bot
                if (useTrailingStop)
                {
                    // For trailing stops, we first place the regular order
                    await _tradingBot.PlaceLimitOrder(
                        order.Symbol,
                        order.Quantity,
                        order.OrderType,
                        order.Price
                    );
                    
                    // Then set up the trailing stop
                    double trailingDistance = CalculateTrailingStopDistance(prediction);
                    
                    // For BUY orders, we use SELL for trailing stop
                    // For SELL orders, we use BUY for trailing stop
                    string trailingStopOrderType = order.OrderType == "BUY" ? "SELL" : "BUY";
                    
                    bool trailingStopSet = _tradingBot.SetTrailingStop(
                        order.Symbol,
                        order.Price,
                        trailingDistance,
                        trailingStopOrderType
                    );
                    
                    if (trailingStopSet)
                    {
                        DatabaseMonolith.Log("Info", $"Trailing stop set for {order.Symbol} with {trailingDistance:P2} trailing distance");
                    }
                    else
                    {
                        DatabaseMonolith.Log("Warning", $"Failed to set trailing stop for {order.Symbol}");
                    }
                }
                else
                {
                    // Use standard bracket order with fixed stop loss and take profit
                    await _tradingBot.PlaceBracketOrder(
                        order.Symbol,
                        order.Quantity,
                        order.OrderType,
                        order.Price,
                        order.StopLoss,
                        order.TakeProfit
                    );
                }
                
                // Update order status
                order.Status = "Executed";
                
                // Save to order history
                DatabaseMonolith.AddOrderToHistory(order);
                
                // Update UI to show the trade was executed
                UpdateUIAfterTrade(prediction);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error executing automated trade for {prediction.Symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Updates the UI after a trade is executed
        /// </summary>
        private void UpdateUIAfterTrade(PredictionModel prediction)
        {
            try
            {
                // Update status text
                if (StatusText != null)
                    StatusText.Text = $"Auto trade executed: {prediction.PredictedAction} {prediction.Symbol}";
                
                // Update last updated text
                if (LastUpdatedText != null)
                    LastUpdatedText.Text = $"Last updated: {DateTime.Now:MM/dd/yyyy HH:mm:ss}";
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error updating UI after trade", ex.ToString());
            }
        }

        /// <summary>
        /// Calculates position size based on the prediction and risk parameters
        /// </summary>
        private int CalculatePositionSize(PredictionModel prediction)
        {
            try
            {
                // If trading bot isn't initialized, use basic position sizing
                if (_tradingBot == null)
                {
                    // Basic position sizing - fallback implementation
                    int baseSize = 100;
                    int scaledSize = (int)(baseSize * prediction.Confidence);
                    return Math.Max(1, Math.Min(scaledSize, 1000));
                }
                
                // Get user settings for account size and risk parameters
                var userSettings = DatabaseMonolith.GetUserSettings();
                double accountSize = userSettings.AccountSize;
                double baseRiskPercentage = userSettings.BaseRiskPercentage;
                
                // Calculate appropriate stop loss price based on prediction
                double stopLossPrice = CalculateStopLoss(prediction);
                
                // Create position sizing parameters
                var parameters = new PositionSizingParameters
                {
                    Symbol = prediction.Symbol,
                    Price = prediction.CurrentPrice,
                    StopLossPrice = stopLossPrice,
                    RiskPercentage = baseRiskPercentage,
                    AccountSize = accountSize,
                    Method = GetPositionSizingMethodFromSettings(userSettings),
                    Confidence = prediction.Confidence
                };
                
                // Set volatility if available in the prediction
                if (prediction.Volatility.HasValue)
                {
                    parameters.ATR = prediction.CurrentPrice * prediction.Volatility.Value;
                }
                
                // Add historical trading metrics if available
                if (userSettings.UseKellyCriterion)
                {
                    // In a real implementation, these would come from actual trade history
                    parameters.WinRate = userSettings.HistoricalWinRate > 0 ? 
                        userSettings.HistoricalWinRate : 0.5;
                    
                    parameters.RewardRiskRatio = userSettings.HistoricalRewardRiskRatio > 0 ? 
                        userSettings.HistoricalRewardRiskRatio : 2.0;
                }
                
                // Calculate position size using the trading bot
                int shares = _tradingBot.CalculatePositionSize(parameters);
                
                DatabaseMonolith.Log("Info", $"Calculated position size for {prediction.Symbol}: {shares} shares using {parameters.Method}");
                
                return shares;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error calculating position size for {prediction.Symbol}", ex.ToString());
                
                // Fallback to basic position sizing
                int baseSize = 100;
                int scaledSize = (int)(baseSize * prediction.Confidence);
                return Math.Max(1, Math.Min(scaledSize, 1000));
            }
        }
        
        /// <summary>
        /// Determines the position sizing method to use based on user settings
        /// </summary>
        private PositionSizingMethod GetPositionSizingMethodFromSettings(UserSettings settings)
        {
            // Get the position sizing method from settings
            string methodName = settings.PositionSizingMethod ?? "FixedRisk";
            
            // Parse the method name to get the enum value
            if (Enum.TryParse<PositionSizingMethod>(methodName, out var method))
            {
                return method;
            }
            
            // Default to fixed risk if the method name is invalid
            return PositionSizingMethod.FixedRisk;
        }

        /// <summary>
        /// Calculates stop loss level for a prediction
        /// </summary>
        private double CalculateStopLoss(PredictionModel prediction)
        {
            if (prediction.PredictedAction == "BUY")
            {
                // For buy orders, stop loss is below entry price (e.g., 5% below)
                return prediction.CurrentPrice * 0.95;
            }
            else
            {
                // For sell orders, stop loss is above entry price (e.g., 5% above)
                return prediction.CurrentPrice * 1.05;
            }
        }

        /// <summary>
        /// Calculates take profit level for a prediction
        /// </summary>
        private double CalculateTakeProfit(PredictionModel prediction)
        {
            if (prediction.PredictedAction == "BUY")
            {
                // For buy orders, take profit is above entry price (e.g., target price or 10% above)
                return prediction.TargetPrice > prediction.CurrentPrice ? 
                    prediction.TargetPrice : prediction.CurrentPrice * 1.1;
            }
            else
            {
                // For sell orders, take profit is below entry price (e.g., target price or 10% below)
                return prediction.TargetPrice < prediction.CurrentPrice ? 
                    prediction.TargetPrice : prediction.CurrentPrice * 0.9;
            }
        }

        /// <summary>
        /// Checks if we've traded this symbol too recently to avoid excessive trading
        /// </summary>
        private bool HasTradedRecently(string symbol)
        {
            // Check if we have a record of trading this symbol
            if (_lastTradeTime.TryGetValue(symbol, out DateTime lastTradeTime))
            {
                // Don't trade the same symbol more than once per hour
                // This threshold could be made configurable
                TimeSpan timeSinceLastTrade = DateTime.Now - lastTradeTime;
                return timeSinceLastTrade.TotalHours < 1;
            }
            
            // Haven't traded this symbol recently
            return false;
        }

        /// <summary>
        /// Determines whether to use a trailing stop for this prediction
        /// </summary>
        /// <param name="prediction">The prediction model</param>
        /// <returns>True if a trailing stop should be used</returns>
        private bool ShouldUseTrailingStop(PredictionModel prediction)
        {
            // Use trailing stops for high confidence predictions
            if (prediction.Confidence >= 0.90)
            {
                return true;
            }
            
            // Use trailing stops for predictions with larger expected price movements
            if (prediction.PredictedAction == "BUY" && 
                prediction.TargetPrice > prediction.CurrentPrice * 1.15) // 15% potential upside
            {
                return true;
            }
            
            // Use trailing stops for predictions that have trend-following keywords in the trading rule
            if (!string.IsNullOrEmpty(prediction.TradingRule))
            {
                string rule = prediction.TradingRule.ToLower();
                if (rule.Contains("trend") || rule.Contains("follow") || rule.Contains("momentum") || 
                    rule.Contains("trailing") || rule.Contains("dynamic"))
                {
                    return true;
                }
            }
            
            // Default to traditional fixed stop-loss
            return false;
        }

        /// <summary>
        /// Updates the ExecuteAutomatedTrade method to use trailing stop orders when appropriate
        /// This method is called for testing purposes to verify integration of trailing stop functionality
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <param name="price">Current price</param>
        /// <param name="confidence">Prediction confidence</param>
        /// <param name="targetPrice">Target price</param>
        /// <returns>True if trailing stop would be used</returns>
        internal bool TestTrailingStopIntegration(string symbol, double price, double confidence, double targetPrice)
        {
            // Create a test prediction
            var prediction = new PredictionModel
            {
                Symbol = symbol,
                PredictedAction = "BUY",
                CurrentPrice = price,
                Confidence = confidence,
                TargetPrice = targetPrice,
                TradingRule = "Trend following strategy with trailing stop"
            };
            
            // Test if we would use a trailing stop for this prediction
            bool useTrailingStop = ShouldUseTrailingStop(prediction);
            
            if (useTrailingStop)
            {
                // Calculate the trailing distance that would be used
                double trailingDistance = CalculateTrailingStopDistance(prediction);
                
                // Verify the trailing distance is reasonable
                if (trailingDistance < 0.02 || trailingDistance > 0.15)
                {
                    return false;
                }
            }
            
            return useTrailingStop;
        }
        
        /// <summary>
        /// Calculates the appropriate trailing distance for a trailing stop
        /// </summary>
        /// <param name="prediction">The prediction model</param>
        /// <returns>The trailing distance as a percentage (e.g., 0.05 for 5%)</returns>
        private double CalculateTrailingStopDistance(PredictionModel prediction)
        {
            // Start with a base trailing distance
            double baseDistance = 0.05; // 5%
            
            try
            {
                // Adjust based on prediction confidence
                // Lower confidence = wider trailing stop
                if (prediction.Confidence < 0.95)
                {
                    baseDistance += (0.95 - prediction.Confidence) * 0.1; // Up to 1% extra
                }
                
                // Adjust based on volatility if available
                // Higher volatility = wider trailing stop
                if (prediction.Volatility.HasValue)
                {
                    baseDistance += prediction.Volatility.Value * 0.5;
                }
                else
                {
                    // If volatility isn't directly available, try to retrieve it
                    if (_tradingBot != null)
                    {
                        try
                        {
                            // Try to get an indicator of volatility like ATR from the trading bot
                            var technicalService = _tradingBot.GetTechnicalIndicatorService();
                            double atr = technicalService.GetATR(prediction.Symbol, "1d").Result;
                            double price = prediction.CurrentPrice;
                            
                            // Normalize ATR as a percentage of price
                            double atrPercent = atr / price;
                            baseDistance += atrPercent * 2; // Adjust by twice the ATR
                        }
                        catch
                        {
                            // If we can't get volatility data, use a slightly wider default
                            baseDistance += 0.02;
                        }
                    }
                }
                
                // Ensure the trailing stop distance is within reasonable bounds
                baseDistance = Math.Max(0.02, Math.Min(baseDistance, 0.15)); // Between 2% and 15%
                
                return baseDistance;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", "Error calculating trailing stop distance, using default", ex.ToString());
                return 0.05; // Default to 5% if there's an error
            }
        }

        /// <summary>
        /// Executes trades based on a list of predictions using WebullTradingBot for market data and trading.
        /// </summary>
        /// <param name="predictions">List of predictions to act on</param>
        private async Task ExecuteTradesFromPredictionsAsync(List<PredictionModel> predictions)
        {
            // Ensure trading components are initialized before executing trades
            if (_tradingBot == null)
                InitializeTradingComponents();

            foreach (var prediction in predictions)
            {
                try
                {
                    // Only trade if not traded recently and confidence is high
                    if (!HasTradedRecently(prediction.Symbol) && prediction.Confidence >= 0.8)
                    {
                        int quantity = CalculatePositionSize(prediction);
                        double stopLoss = CalculateStopLoss(prediction);
                        double takeProfit = CalculateTakeProfit(prediction);
                        
                        // Determine if we should use a trailing stop for this prediction
                        bool useTrailingStop = ShouldUseTrailingStop(prediction);

                        // Create order record
                        var order = new OrderModel
                        {
                            Symbol = prediction.Symbol,
                            OrderType = prediction.PredictedAction,
                            Quantity = quantity,
                            Price = prediction.CurrentPrice,
                            StopLoss = stopLoss,
                            TakeProfit = takeProfit,
                            IsPaperTrade = true,
                            PredictionSource = $"Auto: {prediction.Symbol} ({prediction.Confidence:P0})",
                            Status = "Executed",
                            Timestamp = DateTime.Now
                        };

                        if (useTrailingStop)
                        {
                            // For trailing stops, first place the regular order
                            await _tradingBot.PlaceLimitOrder(
                                order.Symbol,
                                order.Quantity,
                                order.OrderType,
                                order.Price
                            );
                            
                            // Then set up the trailing stop
                            double trailingDistance = CalculateTrailingStopDistance(prediction);
                            
                            // For BUY orders, we use SELL for trailing stop
                            // For SELL orders, we use BUY for trailing stop
                            string trailingStopOrderType = order.OrderType == "BUY" ? "SELL" : "BUY";
                            
                            bool trailingStopSet = _tradingBot.SetTrailingStop(
                                order.Symbol,
                                order.Price,
                                trailingDistance,
                                trailingStopOrderType
                            );
                            
                            if (trailingStopSet)
                            {
                                DatabaseMonolith.Log("Info", $"Trailing stop set for {order.Symbol} with {trailingDistance:P2} trailing distance");
                            }
                            else
                            {
                                DatabaseMonolith.Log("Warning", $"Failed to set trailing stop for {order.Symbol}");
                            }
                        }
                        else
                        {
                            // Use standard bracket order with fixed stop loss and take profit
                            await _tradingBot.PlaceBracketOrder(
                                order.Symbol,
                                order.Quantity,
                                order.OrderType,
                                order.Price,
                                order.StopLoss,
                                order.TakeProfit
                            );
                        }

                        // Log and update last trade time
                        _lastTradeTime[prediction.Symbol] = DateTime.Now;
                        DatabaseMonolith.Log("Info", $"Trade executed for {prediction.Symbol}: {prediction.PredictedAction} {quantity} @ {prediction.CurrentPrice}");
                        
                        // Save order history
                        DatabaseMonolith.AddOrderToHistory(order);

                        // Update UI after trade
                        UpdateUIAfterTrade(prediction);
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error executing trade for {prediction.Symbol}", ex.ToString());
                }
            }
        }
    }
}
