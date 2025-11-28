using System;
using System.Collections.Generic;

namespace Quantra.DAL.Models
{
    /// <summary>
    /// Represents the result of a multi-symbol comparative analysis.
    /// Used by Market Chat for handling queries like "Compare predictions for AAPL, MSFT, and GOOGL".
    /// (MarketChat story 7)
    /// </summary>
    public class MultiSymbolComparisonResult
    {
        /// <summary>
        /// The symbols that were compared
        /// </summary>
        public List<string> Symbols { get; set; } = new List<string>();

        /// <summary>
        /// Individual analysis data for each symbol
        /// </summary>
        public Dictionary<string, SymbolAnalysisData> SymbolData { get; set; } = new Dictionary<string, SymbolAnalysisData>();

        /// <summary>
        /// Timestamp when the comparison was generated
        /// </summary>
        public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Whether any data was retrieved from cache
        /// </summary>
        public bool UsedCachedData { get; set; }

        /// <summary>
        /// Signal highlights identifying strongest/weakest performers
        /// </summary>
        public SignalHighlights Highlights { get; set; }

        /// <summary>
        /// Composite scores for each symbol (0-100)
        /// </summary>
        public Dictionary<string, double> CompositeScores { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Any errors encountered during comparison
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Returns true if the comparison completed successfully with data for at least one symbol
        /// </summary>
        public bool IsSuccessful => SymbolData.Count > 0;

        /// <summary>
        /// Creates an empty result indicating no data was found
        /// </summary>
        public static MultiSymbolComparisonResult Empty => new MultiSymbolComparisonResult
        {
            GeneratedAt = DateTime.UtcNow,
            UsedCachedData = false
        };
    }

    /// <summary>
    /// Represents analysis data for a single symbol within a multi-symbol comparison.
    /// </summary>
    public class SymbolAnalysisData
    {
        /// <summary>
        /// Stock symbol (e.g., "AAPL")
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Current stock price
        /// </summary>
        public double CurrentPrice { get; set; }

        /// <summary>
        /// Predicted action from ML model (BUY, SELL, HOLD)
        /// </summary>
        public string PredictedAction { get; set; }

        /// <summary>
        /// Target price from prediction
        /// </summary>
        public double TargetPrice { get; set; }

        /// <summary>
        /// Prediction confidence (0-1)
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Potential return percentage
        /// </summary>
        public double PotentialReturn { get; set; }

        /// <summary>
        /// Key technical indicators for the symbol
        /// </summary>
        public Dictionary<string, double> Indicators { get; set; } = new Dictionary<string, double>();

        /// <summary>
        /// Risk metrics for the symbol
        /// </summary>
        public SymbolRiskMetrics RiskMetrics { get; set; }

        /// <summary>
        /// Historical price context summary
        /// </summary>
        public HistoricalContextSummary HistoricalContext { get; set; }

        /// <summary>
        /// Whether data was retrieved from cache
        /// </summary>
        public bool IsCached { get; set; }

        /// <summary>
        /// When the prediction was generated
        /// </summary>
        public DateTime? PredictionTimestamp { get; set; }

        /// <summary>
        /// Composite score (0-100) based on weighted factors
        /// </summary>
        public double CompositeScore { get; set; }

        /// <summary>
        /// Any errors retrieving data for this symbol
        /// </summary>
        public string Error { get; set; }

        /// <summary>
        /// Returns true if data was successfully retrieved for this symbol
        /// </summary>
        public bool HasData => string.IsNullOrEmpty(Error) && !string.IsNullOrEmpty(PredictedAction);
    }

    /// <summary>
    /// Risk metrics for a symbol in the comparison.
    /// </summary>
    public class SymbolRiskMetrics
    {
        /// <summary>
        /// Overall risk score (0-100, higher = more risky)
        /// </summary>
        public double RiskScore { get; set; }

        /// <summary>
        /// Historical volatility
        /// </summary>
        public double Volatility { get; set; }

        /// <summary>
        /// Average True Range
        /// </summary>
        public double ATR { get; set; }

        /// <summary>
        /// Maximum drawdown percentage
        /// </summary>
        public double MaxDrawdown { get; set; }

        /// <summary>
        /// Sharpe ratio if available
        /// </summary>
        public double? SharpeRatio { get; set; }

        /// <summary>
        /// Value at Risk (VaR) if available
        /// </summary>
        public double? VaR { get; set; }
    }

    /// <summary>
    /// Summary of historical price context for comparison.
    /// </summary>
    public class HistoricalContextSummary
    {
        /// <summary>
        /// 5-day price change percentage
        /// </summary>
        public double FiveDayChange { get; set; }

        /// <summary>
        /// 20-day price change percentage
        /// </summary>
        public double TwentyDayChange { get; set; }

        /// <summary>
        /// 5-day moving average
        /// </summary>
        public double FiveDayMA { get; set; }

        /// <summary>
        /// 20-day moving average
        /// </summary>
        public double TwentyDayMA { get; set; }

        /// <summary>
        /// 50-day moving average
        /// </summary>
        public double FiftyDayMA { get; set; }

        /// <summary>
        /// Current price position relative to 20-day MA ("above", "below", "at")
        /// </summary>
        public string PriceVsMA { get; set; }

        /// <summary>
        /// Average daily volume
        /// </summary>
        public double AverageVolume { get; set; }

        /// <summary>
        /// Recent volume compared to average ("high", "normal", "low")
        /// </summary>
        public string VolumePattern { get; set; }

        /// <summary>
        /// Trend direction ("bullish", "bearish", "neutral")
        /// </summary>
        public string TrendDirection { get; set; }

        /// <summary>
        /// Momentum score (-100 to 100)
        /// </summary>
        public double MomentumScore { get; set; }
    }

    /// <summary>
    /// Highlights of strongest and weakest signals across compared symbols.
    /// </summary>
    public class SignalHighlights
    {
        /// <summary>
        /// Symbol with the strongest bullish signal
        /// </summary>
        public string StrongestBullish { get; set; }

        /// <summary>
        /// Reason for strongest bullish signal
        /// </summary>
        public string StrongestBullishReason { get; set; }

        /// <summary>
        /// Symbol with the strongest bearish signal
        /// </summary>
        public string StrongestBearish { get; set; }

        /// <summary>
        /// Reason for strongest bearish signal
        /// </summary>
        public string StrongestBearishReason { get; set; }

        /// <summary>
        /// Symbol with highest confidence
        /// </summary>
        public string HighestConfidence { get; set; }

        /// <summary>
        /// Confidence value for highest confidence symbol
        /// </summary>
        public double HighestConfidenceValue { get; set; }

        /// <summary>
        /// Symbol with lowest risk
        /// </summary>
        public string LowestRisk { get; set; }

        /// <summary>
        /// Risk score for lowest risk symbol
        /// </summary>
        public double LowestRiskValue { get; set; }

        /// <summary>
        /// Symbol with highest risk
        /// </summary>
        public string HighestRisk { get; set; }

        /// <summary>
        /// Risk score for highest risk symbol
        /// </summary>
        public double HighestRiskValue { get; set; }

        /// <summary>
        /// Symbol with highest momentum
        /// </summary>
        public string HighestMomentum { get; set; }

        /// <summary>
        /// Momentum score for highest momentum symbol
        /// </summary>
        public double HighestMomentumValue { get; set; }

        /// <summary>
        /// Symbol with highest potential return
        /// </summary>
        public string HighestPotentialReturn { get; set; }

        /// <summary>
        /// Potential return value for highest return symbol
        /// </summary>
        public double HighestPotentialReturnValue { get; set; }

        /// <summary>
        /// Overall recommended pick (best risk-adjusted opportunity)
        /// </summary>
        public string RecommendedPick { get; set; }

        /// <summary>
        /// Reason for the recommendation
        /// </summary>
        public string RecommendedPickReason { get; set; }
    }
}
