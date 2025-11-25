using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Model representing a trading signal generated from sentiment extremes
    /// </summary>
    public class SentimentExtremeSignalModel
    {
        /// <summary>
        /// Symbol the signal applies to
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Date when the signal was generated
        /// </summary>
        public DateTime GeneratedDate { get; set; }

        /// <summary>
        /// Recommended trading action (BUY, SELL)
        /// </summary>
        public string RecommendedAction { get; set; }

        /// <summary>
        /// Current price at time of signal
        /// </summary>
        public double CurrentPrice { get; set; }

        /// <summary>
        /// Target price based on sentiment analysis
        /// </summary>
        public double TargetPrice { get; set; }

        /// <summary>
        /// Overall sentiment score from all sources (-1.0 to 1.0)
        /// </summary>
        public double SentimentScore { get; set; }

        /// <summary>
        /// Signal confidence level from 0.0 to 1.0
        /// </summary>
        public double ConfidenceLevel { get; set; }

        /// <summary>
        /// Individual sentiment scores from each source
        /// </summary>
        public Dictionary<string, double> SourceSentiments { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Reason for the signal generation
        /// </summary>
        public string SignalReason { get; set; }

        /// <summary>
        /// Whether the signal has been acted upon
        /// </summary>
        public bool IsActedUpon { get; set; }

        /// <summary>
        /// Sources that contributed to this sentiment extreme
        /// </summary>
        public List<string> ContributingSources { get; set; } = new List<string>();

        /// <summary>
        /// Potential return percentage based on current price vs target price
        /// </summary>
        public double PotentialReturn => (TargetPrice - CurrentPrice) / CurrentPrice * 100.0;

        /// <summary>
        /// Returns a human-readable description of the sentiment extreme
        /// </summary>
        public string GetDescriptionForHumans()
        {
            string direction = SentimentScore > 0 ? "bullish" : "bearish";
            string magnitude = Math.Abs(SentimentScore) > 0.6 ? "extremely" : "strongly";

            return $"{magnitude} {direction} sentiment detected across " +
                   $"{ContributingSources.Count} sources with {ConfidenceLevel:P0} confidence";
        }

        /// <summary>
        /// Returns a summary of the signal for alerts or notifications
        /// </summary>
        public string GetSignalSummary()
        {
            string direction = RecommendedAction == "BUY" ? "bullish" : "bearish";
            return $"Sentiment Extreme Signal: {direction.ToUpper()} {Symbol} at {CurrentPrice:C} " +
                   $"with target {TargetPrice:C} ({PotentialReturn:F2}% potential return)";
        }
    }
}