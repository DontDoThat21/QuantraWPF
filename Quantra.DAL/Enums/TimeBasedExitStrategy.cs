using System;

namespace Quantra.Enums
{
    /// <summary>
    /// Defines the different types of time-based exit strategies
    /// </summary>
    public enum TimeBasedExitStrategy
    {
        /// <summary>
        /// Custom exit at a specific datetime
        /// </summary>
        Custom = 0,
        
        /// <summary>
        /// Exit at the end of the current trading day (4:00 PM)
        /// </summary>
        EndOfDay = 1,
        
        /// <summary>
        /// Exit at the end of the week (Friday at 4:00 PM)
        /// </summary>
        EndOfWeek = 2,
        
        /// <summary>
        /// Exit after a specific duration (minutes) from entry
        /// </summary>
        Duration = 3,
        
        /// <summary>
        /// Exit at a specific time of day
        /// </summary>
        SpecificTimeOfDay = 4
    }
}