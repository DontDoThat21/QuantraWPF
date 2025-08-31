namespace Quantra.Enums
{
    /// <summary>
    /// Defines different risk modes for trading operations
    /// </summary>
    public enum RiskMode
    {
        /// <summary>
        /// Standard risk tolerance mode (default)
        /// </summary>
        Normal,
        
        /// <summary>
        /// Uses Good Faith Value for risk calculations
        /// </summary>
        GoodFaithValue,
        
        /// <summary>
        /// Very low risk tolerance - smaller position sizes and tighter risk control
        /// </summary>
        Conservative,
        
        /// <summary>
        /// Moderate risk tolerance - balanced position sizing
        /// </summary>
        Moderate,
        
        /// <summary>
        /// Higher risk tolerance - larger position sizes and wider stops
        /// </summary>
        Aggressive
    }
}
