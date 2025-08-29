using System;

namespace Quantra.Enums
{
    /// <summary>
    /// Defines the distribution strategies for dollar-cost averaging investments
    /// </summary>
    public enum DCAStrategyType
    {
        /// <summary>
        /// Equal distribution across all periods (default)
        /// </summary>
        Equal = 0,
        
        /// <summary>
        /// Front-loaded distribution with larger investments at the beginning
        /// </summary>
        FrontLoaded = 1,
        
        /// <summary>
        /// Back-loaded distribution with larger investments at the end
        /// </summary>
        BackLoaded = 2,
        
        /// <summary>
        /// Normal (bell curve) distribution with larger investments in the middle
        /// </summary>
        Normal = 3,
        
        /// <summary>
        /// Value-based distribution that invests more when prices are lower
        /// </summary>
        ValueBased = 4,
        
        /// <summary>
        /// Volatility-based distribution that adjusts based on market volatility
        /// </summary>
        VolatilityBased = 5,
        
        /// <summary>
        /// Custom distribution provided by the user
        /// </summary>
        Custom = 6
    }
}