using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for detecting sentiment extremes and generating trading signals
    /// </summary>
    public class SentimentExtremeSignalService
    {
        private readonly FinancialNewsSentimentService _financialNewsSentimentService;
        private readonly ISocialMediaSentimentService _twitterSentimentService;
        private readonly IAnalystRatingService _analystRatingService;
        private readonly IInsiderTradingService _insiderTradingService;
        private readonly SentimentPriceCorrelationAnalysis _sentimentPriceCorrelationAnalysis;
        private readonly ITradingService _tradingService;
        private readonly IEmailService _emailService;
        private readonly UserSettings _userSettings;

        // Default thresholds
        private readonly double _extremeSentimentThreshold = 0.65; // Sentiment above this is considered extreme
        private readonly double _significantSourcesThreshold = 2; // At least this many sources must agree
        private readonly double _minimumConfidenceLevel = 0.7; // Minimum confidence for signal generation
        private readonly double _minimumPotentialReturn = 5.0; // Minimum expected return percentage

        public SentimentExtremeSignalService(
            ITradingService tradingService = null,
            IEmailService emailService = null,
            UserSettings userSettings = null)
        {
            _financialNewsSentimentService = new FinancialNewsSentimentService(userSettings);
            _twitterSentimentService = new TwitterSentimentService();
            _analystRatingService = ServiceLocator.Resolve<IAnalystRatingService>();
            _insiderTradingService = ServiceLocator.Resolve<IInsiderTradingService>();
            _sentimentPriceCorrelationAnalysis = new SentimentPriceCorrelationAnalysis();
            _tradingService = tradingService;
            _emailService = emailService;
            _userSettings = userSettings ?? new UserSettings();
        }

        /// <summary>
        /// Analyzes sentiment across multiple sources and generates a trading signal if extremes are detected
        /// </summary>
        /// <param name="symbol">Stock symbol to analyze</param>
        /// <param name="autoExecuteTrade">Whether to automatically execute the trade</param>
        /// <returns>A signal model if sentiment extreme is detected, otherwise null</returns>
        public async Task<SentimentExtremeSignalModel> AnalyzeAndGenerateSignalAsync(
            string symbol, 
            bool autoExecuteTrade = false)
        {
            try
            {
                // 1. Gather sentiment data from all sources
                var sentimentData = await GatherSentimentDataAsync(symbol);
                
                if (!sentimentData.Any())
                {
                    DatabaseMonolith.Log("Info", $"No sentiment data available for {symbol}");
                    return null;
                }
                
                // 2. Check for sentiment extremes
                var extremeSentimentSources = sentimentData
                    .Where(s => Math.Abs(s.Value) >= _extremeSentimentThreshold)
                    .ToDictionary(s => s.Key, s => s.Value);
                
                // 3. If not enough sources show extreme sentiment, exit
                if (extremeSentimentSources.Count < _significantSourcesThreshold)
                {
                    return null;
                }
                
                // 4. Determine direction (positive/negative sentiment)
                bool isPositiveSentiment = extremeSentimentSources.Values.Average() > 0;
                string action = isPositiveSentiment ? "BUY" : "SELL";
                
                // 5. Calculate average strength of sentiment from extreme sources
                double averageExtremeSentiment = extremeSentimentSources.Values.Average();
                
                // 6. Calculate confidence level
                double confidenceLevel = CalculateConfidenceLevel(extremeSentimentSources, sentimentData);
                
                // 7. If confidence level is too low, exit
                if (confidenceLevel < _minimumConfidenceLevel)
                {
                    return null;
                }
                
                // 8. Get current price and calculate target price
                double currentPrice = await GetCurrentPriceAsync(symbol);
                double targetPrice = CalculateTargetPrice(currentPrice, averageExtremeSentiment);
                
                // 9. Calculate potential return
                double potentialReturn = Math.Abs((targetPrice - currentPrice) / currentPrice * 100.0);
                
                // 10. If potential return is too low, exit
                if (potentialReturn < _minimumPotentialReturn)
                {
                    return null;
                }
                
                // 11. Generate signal model
                var signal = new SentimentExtremeSignalModel
                {
                    Symbol = symbol,
                    GeneratedDate = DateTime.Now,
                    RecommendedAction = action,
                    CurrentPrice = currentPrice,
                    TargetPrice = targetPrice,
                    SentimentScore = averageExtremeSentiment,
                    ConfidenceLevel = confidenceLevel,
                    SourceSentiments = sentimentData,
                    ContributingSources = extremeSentimentSources.Keys.ToList(),
                    SignalReason = $"Extreme {(isPositiveSentiment ? "positive" : "negative")} sentiment detected across multiple sources"
                };
                
                // 12. Create alert
                CreateSignalAlert(signal);
                
                // 13. Execute trade if requested
                if (autoExecuteTrade && _tradingService != null)
                {
                    bool success = await _tradingService.ExecuteTradeAsync(
                        symbol,
                        action,
                        currentPrice,
                        targetPrice
                    );
                    
                    signal.IsActedUpon = success;
                    
                    // Log trade execution result
                    DatabaseMonolith.Log(
                        success ? "Info" : "Warning",
                        $"Automated trade execution for sentiment extreme signal on {symbol}",
                        $"Action: {action}, Result: {(success ? "Success" : "Failed")}"
                    );
                }
                
                return signal;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error analyzing sentiment extremes for {symbol}", ex.ToString());
                return null;
            }
        }
        
        /// <summary>
        /// Runs sentiment extreme analysis on a list of symbols and returns signals that meet the criteria
        /// </summary>
        /// <param name="symbols">List of symbols to analyze</param>
        /// <param name="autoExecuteTrades">Whether to automatically execute trades for signals</param>
        /// <returns>List of generated signals</returns>
        public async Task<List<SentimentExtremeSignalModel>> AnalyzeBatchForSignalsAsync(
            List<string> symbols, 
            bool autoExecuteTrades = false)
        {
            var signals = new List<SentimentExtremeSignalModel>();
            
            foreach (var symbol in symbols)
            {
                try
                {
                    var signal = await AnalyzeAndGenerateSignalAsync(symbol, autoExecuteTrades);
                    if (signal != null)
                    {
                        signals.Add(signal);
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Error", $"Error processing sentiment extreme analysis for {symbol}", ex.ToString());
                }
            }
            
            return signals;
        }

        #region Helper Methods
        
        /// <summary>
        /// Gathers sentiment data from all available sources
        /// </summary>
        private async Task<Dictionary<string, double>> GatherSentimentDataAsync(string symbol)
        {
            var sentimentData = new Dictionary<string, double>();
            
            try
            {
                // Get news sentiment
                double newsSentiment = await _financialNewsSentimentService.GetSymbolSentimentAsync(symbol);
                sentimentData["News"] = newsSentiment;
                
                // Get Twitter sentiment
                double twitterSentiment = await _twitterSentimentService.GetSymbolSentimentAsync(symbol);
                sentimentData["Twitter"] = twitterSentiment;
                
                // Get analyst sentiment
                double analystSentiment = await _analystRatingService.GetAnalystSentimentAsync(symbol);
                sentimentData["AnalystRatings"] = analystSentiment;
                
                // Get insider trading sentiment
                double insiderSentiment = await _insiderTradingService.GetInsiderSentimentAsync(symbol);
                sentimentData["InsiderTrading"] = insiderSentiment;
                
                // Remove any that failed or returned 0
                sentimentData = sentimentData
                    .Where(s => Math.Abs(s.Value) > 0.01) // Filter out zero/near-zero values
                    .ToDictionary(s => s.Key, s => s.Value);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error gathering sentiment data for {symbol}", ex.ToString());
            }
            
            return sentimentData;
        }
        
        /// <summary>
        /// Calculates the confidence level for a sentiment extreme signal
        /// </summary>
        private double CalculateConfidenceLevel(
            Dictionary<string, double> extremeSources, 
            Dictionary<string, double> allSources)
        {
            // Factors affecting confidence:
            // 1. Number of sources showing extreme sentiment relative to all sources
            double sourceRatio = (double)extremeSources.Count / allSources.Count;
            
            // 2. Agreement level among extreme sources (standard deviation)
            double avgSentiment = extremeSources.Values.Average();
            double standardDeviation = Math.Sqrt(
                extremeSources.Values.Sum(v => Math.Pow(v - avgSentiment, 2)) / extremeSources.Count
            );
            // Convert standard deviation to a normalized value (0-1 where 0 is perfect agreement)
            double agreementScore = Math.Max(0, 1 - standardDeviation * 2);
            
            // 3. Strength of sentiment (how extreme is it)
            double avgStrength = extremeSources.Values.Average(v => Math.Abs(v));
            double strengthScore = Math.Min(1, avgStrength / 0.8); // Normalize to 0-1, max at 0.8
            
            // Weighted factors for final confidence
            double confidence = sourceRatio * 0.4 + agreementScore * 0.3 + strengthScore * 0.3;
            
            // Ensure it's in the range 0-1
            return Math.Max(0, Math.Min(1, confidence));
        }
        
        /// <summary>
        /// Gets the current price for a symbol
        /// </summary>
        private async Task<double> GetCurrentPriceAsync(string symbol)
        {
            // In a real implementation, this would call a market data service
            // For now, use a mock price
            Random random = new Random();
            return 100.0 + random.NextDouble() * 50.0;
        }
        
        /// <summary>
        /// Calculates target price based on current price and sentiment strength
        /// </summary>
        private double CalculateTargetPrice(double currentPrice, double sentimentScore)
        {
            // Calculate expected move based on sentiment strength
            // Stronger sentiment scores predict larger price movements
            double movePercentage = Math.Abs(sentimentScore) * 15.0; // Up to 15% move for max sentiment
            
            // Direction depends on sentiment sign
            double moveDirection = sentimentScore > 0 ? 1 : -1;
            
            // Calculate target
            return currentPrice * (1 + movePercentage / 100.0 * moveDirection);
        }
        
        /// <summary>
        /// Creates an alert from a sentiment extreme signal
        /// </summary>
        private void CreateSignalAlert(SentimentExtremeSignalModel signal)
        {
            var alert = new AlertModel
            {
                Name = $"Sentiment Extreme Signal: {signal.Symbol}",
                Symbol = signal.Symbol,
                Condition = signal.SignalReason,
                AlertType = "SentimentExtreme",
                Notes = $"Action: {signal.RecommendedAction}\n" +
                        $"Current Price: {signal.CurrentPrice:C}\n" +
                        $"Target Price: {signal.TargetPrice:C}\n" +
                        $"Potential Return: {signal.PotentialReturn:F2}%\n" +
                        $"Confidence: {signal.ConfidenceLevel:P0}\n" +
                        $"Contributing Sources: {string.Join(", ", signal.ContributingSources)}",
                TriggerPrice = signal.CurrentPrice,
                Category = AlertCategory.Opportunity,
                IsTriggered = true,
                TriggeredDate = DateTime.Now,
                Priority = CalculateAlertPriority(signal)
            };
            
            // Emit global alert
            AlertManager.EmitGlobalAlert(alert);
        }
        
        /// <summary>
        /// Calculates alert priority based on signal parameters
        /// </summary>
        private int CalculateAlertPriority(SentimentExtremeSignalModel signal)
        {
            // Priority 1 (highest) to 3 (lowest)
            if (signal.ConfidenceLevel >= 0.85 && Math.Abs(signal.PotentialReturn) >= 10.0)
                return 1;
            else if (signal.ConfidenceLevel >= 0.75)
                return 2;
            else
                return 3;
        }
        
        #endregion
    }
}