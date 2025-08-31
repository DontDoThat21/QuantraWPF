using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Represents correlation analysis results between technical indicators
    /// </summary>
    public class IndicatorCorrelationResult
    {
        /// <summary>
        /// Symbol for which correlation was calculated
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// First indicator in the correlation
        /// </summary>
        public string FirstIndicator { get; set; }

        /// <summary>
        /// Second indicator in the correlation
        /// </summary>
        public string SecondIndicator { get; set; }

        /// <summary>
        /// Correlation coefficient between the indicators (-1.0 to 1.0)
        /// </summary>
        public double CorrelationCoefficient { get; set; }

        /// <summary>
        /// Confidence level for the correlation (0.0 to 1.0)
        /// </summary>
        public double ConfidenceLevel { get; set; }

        /// <summary>
        /// Number of data points used in correlation calculation
        /// </summary>
        public int DataPointsCount { get; set; }

        /// <summary>
        /// Timeframe used for correlation calculation
        /// </summary>
        public string Timeframe { get; set; }

        /// <summary>
        /// Original historical values for first indicator
        /// </summary>
        public List<double> FirstIndicatorValues { get; set; } = new List<double>();

        /// <summary>
        /// Original historical values for second indicator
        /// </summary>
        public List<double> SecondIndicatorValues { get; set; } = new List<double>();

        /// <summary>
        /// Date when the correlation was calculated
        /// </summary>
        public DateTime CalculationDate { get; set; } = DateTime.Now;

        /// <summary>
        /// Returns textual description of correlation strength
        /// </summary>
        public string CorrelationStrength
        {
            get
            {
                double absCorr = Math.Abs(CorrelationCoefficient);
                
                if (absCorr > 0.8) return "Very Strong";
                if (absCorr > 0.6) return "Strong";
                if (absCorr > 0.4) return "Moderate";
                if (absCorr > 0.2) return "Weak";
                return "Very Weak";
            }
        }

        /// <summary>
        /// Returns the direction of correlation
        /// </summary>
        public string CorrelationDirection => CorrelationCoefficient > 0 ? "Positive" : "Negative";

        /// <summary>
        /// Determines if correlation provides a confirmation pattern
        /// </summary>
        /// <returns>True if indicators confirm each other with high confidence</returns>
        public bool IsConfirmationPattern()
        {
            // Confirmation requires reasonably strong correlation and sufficient data
            return Math.Abs(CorrelationCoefficient) > 0.6 && 
                   ConfidenceLevel > 0.7 && 
                   DataPointsCount >= 10;
        }
    }
}