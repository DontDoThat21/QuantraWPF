using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Response model for Alpha Vantage Analytics Fixed Window API
    /// Provides advanced analytics metrics like Sharpe Ratio, Sortino Ratio, etc.
    /// </summary>
    public class AnalyticsFixedWindowResponse
    {
        /// <summary>
        /// Meta information about the analytics response
        /// </summary>
        public AnalyticsMetaData MetaData { get; set; }

        /// <summary>
        /// List of analytics metric results
        /// </summary>
        public List<AnalyticsMetricResult> Metrics { get; set; } = new List<AnalyticsMetricResult>();

        /// <summary>
        /// Indicates if the response was successful
        /// </summary>
        public bool IsSuccess { get; set; }

        /// <summary>
        /// Error message if the request failed
        /// </summary>
        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// Meta data for the analytics response
    /// </summary>
    public class AnalyticsMetaData
    {
        /// <summary>
        /// Symbol(s) analyzed
        /// </summary>
        public string Symbols { get; set; }

        /// <summary>
        /// Time range used for analysis
        /// </summary>
        public string Range { get; set; }

        /// <summary>
        /// Data interval (e.g., DAILY, WEEKLY, MONTHLY)
        /// </summary>
        public string Interval { get; set; }

        /// <summary>
        /// Benchmark symbol used for comparison (if any)
        /// </summary>
        public string Benchmark { get; set; }

        /// <summary>
        /// Date when the analytics were calculated
        /// </summary>
        public DateTime CalculationDate { get; set; }
    }

    /// <summary>
    /// Individual metric result from the analytics
    /// </summary>
    public class AnalyticsMetricResult
    {
        /// <summary>
        /// Name of the metric (e.g., SHARPE_RATIO, SORTINO_RATIO)
        /// </summary>
        public string MetricName { get; set; }

        /// <summary>
        /// Calculated value of the metric
        /// </summary>
        public double Value { get; set; }

        /// <summary>
        /// Symbol this metric applies to (for multi-symbol requests)
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Additional context or description for the metric
        /// </summary>
        public string Description { get; set; }
    }

    /// <summary>
    /// Performance metrics calculated from Alpha Vantage Analytics API
    /// </summary>
    public class PerformanceMetrics
    {
        /// <summary>
        /// Sharpe Ratio - Risk-adjusted return (higher is better)
        /// </summary>
        public double SharpeRatio { get; set; }

        /// <summary>
        /// Sortino Ratio - Downside risk-adjusted return (higher is better)
        /// </summary>
        public double SortinoRatio { get; set; }

        /// <summary>
        /// Calmar Ratio - Return over maximum drawdown (higher is better)
        /// </summary>
        public double CalmarRatio { get; set; }

        /// <summary>
        /// Information Ratio - Risk-adjusted excess return vs benchmark
        /// </summary>
        public double InformationRatio { get; set; }

        /// <summary>
        /// Maximum Drawdown - Largest peak-to-trough decline
        /// </summary>
        public double MaxDrawdown { get; set; }

        /// <summary>
        /// Cumulative return over the period
        /// </summary>
        public double CumulativeReturn { get; set; }

        /// <summary>
        /// Annualized return (CAGR equivalent)
        /// </summary>
        public double AnnualizedReturn { get; set; }

        /// <summary>
        /// Annualized volatility (standard deviation of returns)
        /// </summary>
        public double AnnualizedVolatility { get; set; }

        /// <summary>
        /// Beta relative to benchmark
        /// </summary>
        public double Beta { get; set; }

        /// <summary>
        /// Alpha (Jensen's alpha) - excess return vs expected from CAPM
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Treynor Ratio - Return per unit of systematic risk
        /// </summary>
        public double TreynorRatio { get; set; }

        /// <summary>
        /// R-squared - Correlation with benchmark
        /// </summary>
        public double RSquared { get; set; }

        /// <summary>
        /// Symbol this metrics apply to
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Start date of the analysis period
        /// </summary>
        public DateTime StartDate { get; set; }

        /// <summary>
        /// End date of the analysis period
        /// </summary>
        public DateTime EndDate { get; set; }

        /// <summary>
        /// Indicates if the metrics were successfully calculated
        /// </summary>
        public bool IsValid { get; set; }

        /// <summary>
        /// Error message if metrics calculation failed
        /// </summary>
        public string ErrorMessage { get; set; }
    }
}
