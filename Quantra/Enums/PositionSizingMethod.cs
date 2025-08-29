namespace Quantra.Enums
{
    /// <summary>
    /// Defines different methods for calculating position sizes
    /// </summary>
    public enum PositionSizingMethod
    {
        /// <summary>
        /// Fixed risk percentage of account (standard risk-based sizing)
        /// </summary>
        FixedRisk,
        
        /// <summary>
        /// Fixed percentage of account equity regardless of risk
        /// </summary>
        PercentageOfEquity,
        
        /// <summary>
        /// Position size based on market volatility (e.g., ATR)
        /// </summary>
        VolatilityBased,
        
        /// <summary>
        /// Position size based on Kelly formula (optimal sizing based on win rate and reward/risk ratio)
        /// </summary>
        KellyFormula,
        
        /// <summary>
        /// Fixed dollar amount per trade
        /// </summary>
        FixedAmount,
        
        /// <summary>
        /// Tier-based position size that increases with higher probability setups
        /// </summary>
        TierBased,
        
        /// <summary>
        /// Adaptive position sizing that dynamically adjusts based on multiple factors
        /// including market volatility, recent performance, and market conditions
        /// </summary>
        AdaptiveRisk
    }
}