using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for monitoring sentiment shifts across various sources and generating alerts
    /// </summary>
    public class SentimentShiftAlertService : Interfaces.ISentimentShiftAlertService
    {
        private readonly SentimentPriceCorrelationAnalysis _sentimentAnalysis;
        private readonly Dictionary<string, Dictionary<string, double>> _lastSentimentValues;
        private readonly Models.UserSettings _userSettings;

        // Threshold for significant sentiment change that triggers an alert
        private const double DefaultSentimentShiftThreshold = 0.2;

        public SentimentShiftAlertService(Models.UserSettings userSettings = null)
        {
            _userSettings = userSettings ?? new Models.UserSettings();
            _sentimentAnalysis = new SentimentPriceCorrelationAnalysis();
            _lastSentimentValues = new Dictionary<string, Dictionary<string, double>>();
        }

        /// <summary>
        /// Monitors sentiment for a specific symbol and generates alerts for significant shifts
        /// </summary>
        /// <param name="symbol">The stock symbol to monitor</param>
        /// <param name="sources">Specific sentiment sources to monitor (null for all)</param>
        public async Task MonitorSentimentShiftsAsync(string symbol, List<string> sources = null)
        {
            try
            {
                // Default to all sources if none specified
                sources ??= new List<string> {
                    "News", "Twitter", "Reddit", "AnalystRatings", "InsiderTrading"
                };

                // Get current sentiment from correlation analysis
                var correlationResult = await _sentimentAnalysis.AnalyzeSentimentPriceCorrelation(
                    symbol,
                    lookbackDays: 7, // Only need recent data
                    sentimentSources: sources);

                if (correlationResult?.AlignedData == null)
                {
                    return; // nothing to process
                }

                if (!_lastSentimentValues.ContainsKey(symbol))
                {
                    // First time monitoring this symbol, just store values as baseline
                    _lastSentimentValues[symbol] = new Dictionary<string, double>();
                    foreach (var source in correlationResult.AlignedData.SentimentBySource.Keys)
                    {
                        var sentimentValues = correlationResult.AlignedData.SentimentBySource[source];
                        if (sentimentValues != null && sentimentValues.Count > 0)
                        {
                            _lastSentimentValues[symbol][source] = sentimentValues.Last();
                        }
                    }
                }
                else
                {
                    // Check for significant changes
                    foreach (var source in correlationResult.AlignedData.SentimentBySource.Keys)
                    {
                        var sentimentValues = correlationResult.AlignedData.SentimentBySource[source];
                        if (sentimentValues == null || sentimentValues.Count == 0) continue;

                        double currentSentiment = sentimentValues.Last();

                        // If we have a previous value for this source
                        if (_lastSentimentValues[symbol].ContainsKey(source))
                        {
                            double previousSentiment = _lastSentimentValues[symbol][source];
                            double shift = currentSentiment - previousSentiment;

                            // Check if shift exceeds threshold
                            if (Math.Abs(shift) >= GetShiftThreshold(source))
                            {
                                // Create alert for significant shift
                                CreateSentimentShiftAlert(symbol, source, previousSentiment, currentSentiment, shift, correlationResult);
                            }

                            // Update the stored value
                            _lastSentimentValues[symbol][source] = currentSentiment;
                        }
                        else
                        {
                            // First time seeing this source for this symbol
                            _lastSentimentValues[symbol][source] = currentSentiment;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error monitoring sentiment shifts for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Creates an alert for a significant sentiment shift
        /// </summary>
        private void CreateSentimentShiftAlert(
            string symbol,
            string source,
            double previousSentiment,
            double currentSentiment,
            double shift,
            SentimentPriceCorrelationResult correlationResult)
        {
            // Determine if it's a positive or negative shift
            string direction = shift > 0 ? "positive" : "negative";

            // Format the sentiment values to be more readable
            string formattedPrevious = FormatSentimentValue(previousSentiment);
            string formattedCurrent = FormatSentimentValue(currentSentiment);

            // Get insight on how this source historically correlates with price
            double sourceCorrelation = 0;
            if (correlationResult.SourceCorrelations.ContainsKey(source))
            {
                sourceCorrelation = correlationResult.SourceCorrelations[source];
            }

            // Create a message based on shift direction and correlation
            string impactMessage = GenerateImpactMessage(shift, sourceCorrelation);

            // Calculate alert priority based on shift magnitude and source importance
            int priority = CalculateAlertPriority(source, Math.Abs(shift));

            // Create the alert
            var alert = new Models.AlertModel
            {
                Name = $"Sentiment Shift: {symbol} {source}",
                Symbol = symbol,
                Condition = $"{source} sentiment shifted {direction} ({formattedPrevious} → {formattedCurrent})",
                AlertType = "SentimentShift",
                Notes = $"Source: {source}\n" +
                        $"Previous Sentiment: {formattedPrevious}\n" +
                        $"Current Sentiment: {formattedCurrent}\n" +
                        $"Shift Magnitude: {Math.Abs(shift):F2}\n" +
                        $"Historical Correlation: {sourceCorrelation:F2}\n" +
                        $"{impactMessage}",
                Category = AlertCategory.Sentiment,
                IsActive = true,
                IsTriggered = true,
                CreatedDate = DateTime.Now,
                TriggeredDate = DateTime.Now,
                Priority = priority
            };

            // Fallback emission: log alert details; higher-level layers may subscribe and forward
            //DatabaseMonolith.Log("Alert", alert.Name, alert.Notes);
        }

        /// <summary>
        /// Formats a sentiment value to be more readable
        /// </summary>
        private string FormatSentimentValue(double sentiment)
        {
            if (sentiment >= 0.5) return "Very Positive";
            if (sentiment >= 0.2) return "Positive";
            if (sentiment >= -0.2) return "Neutral";
            if (sentiment >= -0.5) return "Negative";
            return "Very Negative";
        }

        /// <summary>
        /// Generates a message describing potential price impact based on shift and correlation
        /// </summary>
        private string GenerateImpactMessage(double shift, double correlation)
        {
            if (Math.Abs(correlation) < 0.2)
            {
                return "Historically, this source has shown minimal correlation with price movements.";
            }

            string directionHint;
            if (shift > 0 && correlation > 0 || shift < 0 && correlation < 0)
            {
                directionHint = "potential upward";
            }
            else
            {
                directionHint = "potential downward";
            }

            string strengthHint = Math.Abs(correlation) > 0.5 ? "strong" : "moderate";

            return $"Historical analysis suggests {strengthHint} {directionHint} price pressure based on this sentiment shift.";
        }

        /// <summary>
        /// Gets the threshold for considering a shift significant based on source
        /// </summary>
        private double GetShiftThreshold(string source)
        {
            // Different sources may have different volatility in sentiment
            switch (source)
            {
                case "Twitter":
                case "Reddit":
                    return 0.25; // Social media more volatile, need higher threshold
                case "InsiderTrading":
                    return 0.15; // Insider trading less volatile, lower threshold
                case "News":
                case "AnalystRatings":
                default:
                    return DefaultSentimentShiftThreshold;
            }
        }

        /// <summary>
        /// Calculates priority for the alert based on source and shift magnitude
        /// </summary>
        private int CalculateAlertPriority(string source, double shiftMagnitude)
        {
            // Base priority on magnitude
            int basePriority = 2; // Default to medium

            if (shiftMagnitude >= 0.4) basePriority = 1; // High
            else if (shiftMagnitude >= 0.3) basePriority = 2; // Medium
            else basePriority = 3; // Low

            // Adjust based on source reliability
            switch (source)
            {
                case "AnalystRatings":
                case "InsiderTrading":
                    return Math.Max(1, basePriority - 1); // Increase priority (lower number)
                case "Twitter":
                case "Reddit":
                    return Math.Min(3, basePriority + 1); // Decrease priority (higher number)
                default:
                    return basePriority;
            }
        }

        /// <summary>
        /// Monitors sentiment shifts for a watchlist of symbols
        /// </summary>
        /// <param name="symbols">List of symbols to monitor</param>
        public async Task MonitorWatchlistAsync(List<string> symbols)
        {
            if (symbols == null || symbols.Count == 0) return;

            foreach (var symbol in symbols)
            {
                await MonitorSentimentShiftsAsync(symbol);
            }
        }
    }
}