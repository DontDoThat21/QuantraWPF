using Quantra.Enums;

namespace Quantra.Models
{
    /// <summary>
    /// Parameters used for various position sizing algorithms
    /// </summary>
    public class PositionSizingParameters
    {
        /// <summary>
        /// The stock symbol
        /// </summary>
        public string Symbol { get; set; }
        
        /// <summary>
        /// Current market price of the security
        /// </summary>
        public double Price { get; set; }
        
        /// <summary>
        /// Stop loss price for the trade
        /// </summary>
        public double StopLossPrice { get; set; }
        
        /// <summary>
        /// Percentage of account to risk per trade (e.g., 0.01 = 1%)
        /// </summary>
        public double RiskPercentage { get; set; } = 0.01; // Default 1%
        
        /// <summary>
        /// Total account size in dollars
        /// </summary>
        public double AccountSize { get; set; }
        
        /// <summary>
        /// Method to use for position sizing calculation
        /// </summary>
        public PositionSizingMethod Method { get; set; } = PositionSizingMethod.FixedRisk;
        
        /// <summary>
        /// Risk mode defining risk tolerance
        /// </summary>
        public RiskMode RiskMode { get; set; } = RiskMode.Normal;
        
        /// <summary>
        /// Average True Range (ATR) value for volatility-based sizing
        /// </summary>
        public double? ATR { get; set; }
        
        /// <summary>
        /// ATR multiple for volatility-based position sizing (usually 1-3)
        /// </summary>
        public double ATRMultiple { get; set; } = 2.0;
        
        /// <summary>
        /// Maximum position size as a percentage of account (e.g., 0.20 = 20%)
        /// </summary>
        public double MaxPositionSizePercent { get; set; } = 0.20; // Default 20%
        
        /// <summary>
        /// Fixed dollar amount for fixed-amount position sizing
        /// </summary>
        public double FixedAmount { get; set; } = 5000.0; // Default $5,000
        
        /// <summary>
        /// Historical win rate for Kelly formula (e.g., 0.60 = 60%)
        /// </summary>
        public double WinRate { get; set; } = 0.50; // Default 50%
        
        /// <summary>
        /// Average reward/risk ratio for Kelly formula (e.g., 2.0 = 2:1)
        /// </summary>
        public double RewardRiskRatio { get; set; } = 2.0; // Default 2:1
        
        /// <summary>
        /// Multiplier for adjusting Kelly formula (usually 0.25 to 0.5 to be conservative)
        /// </summary>
        public double KellyFractionMultiplier { get; set; } = 0.5; // Default half-Kelly
        
        /// <summary>
        /// Trade setup confidence (0.0-1.0) for tier-based sizing
        /// </summary>
        public double Confidence { get; set; } = 0.5; // Default 50%
        
        /// <summary>
        /// Market volatility factor for adaptive risk sizing (-1.0 to 1.0)
        /// Negative values indicate decreasing volatility, positive values indicate increasing volatility
        /// </summary>
        public double MarketVolatilityFactor { get; set; } = 0.0;
        
        /// <summary>
        /// Recent performance factor for adaptive risk sizing (-1.0 to 1.0)
        /// Negative values indicate recent losses, positive values indicate recent gains
        /// </summary>
        public double PerformanceFactor { get; set; } = 0.0;
        
        /// <summary>
        /// Market trend strength factor for adaptive risk sizing (0.0 to 1.0)
        /// Higher values indicate stronger trend
        /// </summary>
        public double TrendStrengthFactor { get; set; } = 0.5;
        
        /// <summary>
        /// Base position size percentage for adaptive risk sizing
        /// </summary>
        public double BasePositionPercentage { get; set; } = 0.01; // Default 1%
        
        /// <summary>
        /// Equity percentage for percentage-of-equity position sizing (e.g., 0.05 = 5%)
        /// </summary>
        public double EquityPercentage { get; set; } = 0.05; // Default 5%
        
        /// <summary>
        /// Probability of winning for Kelly formula (e.g., 0.60 = 60%)
        /// </summary>
        public double WinProbability { get; set; } = 0.50; // Default 50%
        
        /// <summary>
        /// Probability of losing for Kelly formula (e.g., 0.40 = 40%)
        /// </summary>
        public double LossProbability { get; set; } = 0.50; // Default 50%
        
        /// <summary>
        /// Average win amount for Kelly formula calculation
        /// </summary>
        public double AvgWin { get; set; } = 2.0; // Default 2:1 reward ratio
        
        /// <summary>
        /// Current market volatility for adaptive risk sizing
        /// </summary>
        public double CurrentVolatility { get; set; } = 0.20; // Default 20%
        
        /// <summary>
        /// Baseline volatility for comparison in adaptive risk sizing
        /// </summary>
        public double BaselineVolatility { get; set; } = 0.20; // Default 20%
        
        /// <summary>
        /// Current market trend direction for adaptive adjustments ("Bullish", "Bearish", "Neutral")
        /// </summary>
        public string MarketTrend { get; set; } = "Neutral"; // Default neutral
        
        /// <summary>
        /// Recent win rate for adaptive risk adjustments (e.g., 0.60 = 60%)
        /// </summary>
        public double RecentWinRate { get; set; } = 0.50; // Default 50%
    }
}