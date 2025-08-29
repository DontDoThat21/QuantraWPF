using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Profile information for tracked insiders
    /// </summary>
    public class InsiderProfile
    {
        /// <summary>
        /// Full name of the insider
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Current title or position
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// Company or organization associated with
        /// </summary>
        public string Organization { get; set; }

        /// <summary>
        /// Category of the insider
        /// </summary>
        public NotableFigureCategory Category { get; set; }

        /// <summary>
        /// Influence score (0-100) representing importance of their trades
        /// </summary>
        public int InfluenceScore { get; set; }

        /// <summary>
        /// Historical performance metrics
        /// </summary>
        public InsiderPerformanceMetrics PerformanceMetrics { get; set; }

        /// <summary>
        /// Additional information
        /// </summary>
        public string Notes { get; set; }

        /// <summary>
        /// Whether to highlight this person's trades with high priority
        /// </summary>
        public bool IsPriority { get; set; }
    }

    /// <summary>
    /// Performance metrics for a tracked insider
    /// </summary>
    public class InsiderPerformanceMetrics
    {
        /// <summary>
        /// Number of transactions tracked
        /// </summary>
        public int TotalTransactions { get; set; }

        /// <summary>
        /// Average return following their buy transactions (over 3 months)
        /// </summary>
        public double AverageBuyReturn { get; set; }
        
        /// <summary>
        /// Average return following their sell transactions (over 3 months)
        /// </summary>
        public double AverageSellReturn { get; set; }

        /// <summary>
        /// Success rate of trades (percentage where price moved favorably)
        /// </summary>
        public double SuccessRate { get; set; }

        /// <summary>
        /// Average value of transactions in USD
        /// </summary>
        public double AverageTransactionValue { get; set; }
    }
}