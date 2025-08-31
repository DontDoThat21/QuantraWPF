using System;

namespace Quantra.Enums
{
    /// <summary>
    /// Defines the various market sessions for trading
    /// </summary>
    [Flags]
    public enum MarketSession
    {
        /// <summary>
        /// No sessions enabled
        /// </summary>
        None = 0,
        
        /// <summary>
        /// Pre-market trading session (4:00 AM - 9:30 AM)
        /// </summary>
        PreMarket = 1,
        
        /// <summary>
        /// Regular market trading session (9:30 AM - 4:00 PM)
        /// </summary>
        Regular = 2,
        
        /// <summary>
        /// After-hours trading session (4:00 PM - 8:00 PM)
        /// </summary>
        AfterHours = 4,
        
        /// <summary>
        /// All trading sessions enabled
        /// </summary>
        All = PreMarket | Regular | AfterHours
    }
}