using System;
using System.Collections.Generic;

namespace Quantra.Models
{
    /// <summary>
    /// Class for storing benchmark comparison data for backtesting results
    /// </summary>
    public class BenchmarkComparisonData
    {
        /// <summary>
        /// Name of the benchmark (e.g., "S&P 500", "NASDAQ", "Russell 2000")
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Symbol used to track the benchmark (e.g., "SPY", "QQQ", "IWM")
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Historical price data for the benchmark
        /// </summary>
        public List<HistoricalPrice> HistoricalData { get; set; } = new List<HistoricalPrice>();

        /// <summary>
        /// Normalized returns for benchmark (starting from 1.0)
        /// </summary>
        public List<double> NormalizedReturns { get; set; } = new List<double>();

        /// <summary>
        /// Dates corresponding to the normalized returns
        /// </summary>
        public List<DateTime> Dates { get; set; } = new List<DateTime>();

        /// <summary>
        /// Total return percentage over the backtest period
        /// </summary>
        public double TotalReturn { get; set; }

        /// <summary>
        /// Maximum drawdown percentage over the backtest period
        /// </summary>
        public double MaxDrawdown { get; set; }

        /// <summary>
        /// Volatility (standard deviation of returns) over the backtest period
        /// </summary>
        public double Volatility { get; set; }

        /// <summary>
        /// Sharpe ratio for the benchmark over the backtest period
        /// </summary>
        public double SharpeRatio { get; set; }

        /// <summary>
        /// Sortino ratio for the benchmark over the backtest period
        /// </summary>
        public double SortinoRatio { get; set; }

        /// <summary>
        /// Calmar ratio for the benchmark over the backtest period
        /// </summary>
        public double CalmarRatio { get; set; }

        /// <summary>
        /// Information ratio for the benchmark over the backtest period
        /// </summary>
        public double InformationRatio { get; set; }

        /// <summary>
        /// Beta of the strategy compared to this benchmark
        /// </summary>
        public double Beta { get; set; }

        /// <summary>
        /// Alpha of the strategy compared to this benchmark
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Correlation between the strategy and the benchmark
        /// </summary>
        public double Correlation { get; set; }

        /// <summary>
        /// Compound Annual Growth Rate (CAGR) for the benchmark over the backtest period
        /// </summary>
        public double CAGR { get; set; }
    }
}