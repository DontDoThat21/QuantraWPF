using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Result from Alpha Vantage ANALYTICS_FIXED_WINDOW API
    /// Contains statistical metrics calculated across a fixed date range
    /// </summary>
    public class AnalyticsFixedWindowResult
    {
        /// <summary>
        /// Comma-separated list of symbols included in the analysis
        /// </summary>
        public string Symbols { get; set; }

        /// <summary>
        /// Date range of the analysis
        /// </summary>
        public string Range { get; set; }

        /// <summary>
        /// Time interval used (DAILY, WEEKLY, MONTHLY)
        /// </summary>
        public string Interval { get; set; }

        /// <summary>
        /// OHLC type used for calculations (close, open, high, low)
        /// </summary>
        public string OhlcType { get; set; }

        /// <summary>
        /// Mean values for each symbol
        /// </summary>
        public Dictionary<string, double> MeanValues { get; set; }

        /// <summary>
        /// Standard deviation for each symbol
        /// </summary>
        public Dictionary<string, double> StdDev { get; set; }

        /// <summary>
        /// Annualized standard deviation (volatility) for each symbol
        /// </summary>
        public Dictionary<string, double> AnnualizedStdDev { get; set; }

        /// <summary>
        /// Variance for each symbol
        /// </summary>
        public Dictionary<string, double> Variance { get; set; }

        /// <summary>
        /// Annualized variance for each symbol
        /// </summary>
        public Dictionary<string, double> AnnualizedVariance { get; set; }

        /// <summary>
        /// Correlation matrix between all symbols
        /// Dictionary of Symbol -> Dictionary of correlations with other symbols
        /// </summary>
        public Dictionary<string, Dictionary<string, double>> CorrelationMatrix { get; set; }

        /// <summary>
        /// Covariance matrix between all symbols
        /// </summary>
        public Dictionary<string, Dictionary<string, double>> CovarianceMatrix { get; set; }

        /// <summary>
        /// API response metadata
        /// </summary>
        public AnalyticsMetadata Metadata { get; set; }

        public AnalyticsFixedWindowResult()
        {
            MeanValues = new Dictionary<string, double>();
            StdDev = new Dictionary<string, double>();
            AnnualizedStdDev = new Dictionary<string, double>();
            Variance = new Dictionary<string, double>();
            AnnualizedVariance = new Dictionary<string, double>();
            CorrelationMatrix = new Dictionary<string, Dictionary<string, double>>();
            CovarianceMatrix = new Dictionary<string, Dictionary<string, double>>();
            Metadata = new AnalyticsMetadata();
        }

        /// <summary>
        /// Get correlation between two symbols
        /// </summary>
        public double? GetCorrelation(string symbol1, string symbol2)
        {
            if (CorrelationMatrix != null &&
                CorrelationMatrix.ContainsKey(symbol1) &&
                CorrelationMatrix[symbol1].ContainsKey(symbol2))
            {
                return CorrelationMatrix[symbol1][symbol2];
            }
            return null;
        }

        /// <summary>
        /// Get volatility for a specific symbol
        /// </summary>
        public double? GetVolatility(string symbol, bool annualized = true)
        {
            var dict = annualized ? AnnualizedStdDev : StdDev;
            return dict != null && dict.ContainsKey(symbol) ? dict[symbol] : null;
        }

        /// <summary>
        /// Check if the result contains valid data
        /// </summary>
        public bool IsValid => !string.IsNullOrEmpty(Symbols) && MeanValues != null && MeanValues.Count > 0;
    }

    /// <summary>
    /// Metadata about the analytics API response
    /// </summary>
    public class AnalyticsMetadata
    {
        /// <summary>
        /// Information message from the API
        /// </summary>
        public string Information { get; set; }

        /// <summary>
        /// Timestamp of the response
        /// </summary>
        public DateTime? ResponseTime { get; set; }

        /// <summary>
        /// Any error message from the API
        /// </summary>
        public string Error { get; set; }

        /// <summary>
        /// Note from the API (e.g., rate limit warnings)
        /// </summary>
        public string Note { get; set; }

        /// <summary>
        /// Check if there was an error
        /// </summary>
        public bool HasError => !string.IsNullOrEmpty(Error) || !string.IsNullOrEmpty(Note);
    }
}
