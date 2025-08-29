using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a confirmation pattern among multiple technical indicators
    /// </summary>
    public class IndicatorConfirmationPattern
    {
        /// <summary>
        /// Symbol this pattern applies to
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Type of pattern identified
        /// </summary>
        public string PatternType { get; set; }

        /// <summary>
        /// Strength of the confirmation (0.0 to 1.0)
        /// </summary>
        public double ConfirmationStrength { get; set; }

        /// <summary>
        /// Signal direction: "Bullish", "Bearish", or "Neutral"
        /// </summary>
        public string SignalDirection { get; set; }

        /// <summary>
        /// Timestamp when the pattern was identified
        /// </summary>
        public DateTime IdentificationTime { get; set; } = DateTime.Now;

        /// <summary>
        /// List of indicators that form this confirmation pattern
        /// </summary>
        public List<string> ConfirmingIndicators { get; set; } = new List<string>();

        /// <summary>
        /// Values of the indicators at pattern identification time
        /// </summary>
        public Dictionary<string, double> IndicatorValues { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Correlation results that support this pattern (if applicable)
        /// </summary>
        public List<IndicatorCorrelationResult> SupportingCorrelations { get; set; } = new List<IndicatorCorrelationResult>();

        /// <summary>
        /// Time horizon for which this pattern is relevant (short/medium/long term)
        /// </summary>
        public string TimeHorizon { get; set; }

        /// <summary>
        /// Expected reliability of the pattern (0.0 to 1.0)
        /// </summary>
        public double Reliability { get; set; }

        /// <summary>
        /// Returns a trading recommendation based on this pattern
        /// </summary>
        public string TradingRecommendation
        {
            get
            {
                if (ConfirmationStrength < 0.3)
                    return "Insufficient confirmation - avoid trading";

                if (SignalDirection == "Bullish" && ConfirmationStrength > 0.7)
                    return "Strong buy signal";
                else if (SignalDirection == "Bullish")
                    return "Consider buying";
                else if (SignalDirection == "Bearish" && ConfirmationStrength > 0.7)
                    return "Strong sell signal";
                else if (SignalDirection == "Bearish")
                    return "Consider selling";
                else
                    return "No clear trading action recommended";
            }
        }

        /// <summary>
        /// Returns a description of the confirmation pattern
        /// </summary>
        public string GetPatternDescription()
        {
            string description = $"{ConfirmingIndicators.Count} indicators confirm a {SignalDirection.ToLower()} signal ";
            description += $"with {GetStrengthDescription()} strength ";
            description += $"for {TimeHorizon.ToLower()} term trading.";
            return description;
        }

        private string GetStrengthDescription()
        {
            if (ConfirmationStrength > 0.8) return "very strong";
            if (ConfirmationStrength > 0.6) return "strong";
            if (ConfirmationStrength > 0.4) return "moderate";
            return "weak";
        }
    }
}