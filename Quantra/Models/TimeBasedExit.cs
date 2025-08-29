using System;
using Quantra.Enums;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a time-based exit strategy for a position
    /// </summary>
    public class TimeBasedExit
    {
        /// <summary>
        /// The time at which to exit the position
        /// </summary>
        public DateTime ExitTime { get; set; }
        
        /// <summary>
        /// The type of exit strategy
        /// </summary>
        public TimeBasedExitStrategy Strategy { get; set; }
        
        /// <summary>
        /// Duration in minutes for Duration-type strategies
        /// </summary>
        public int? DurationMinutes { get; set; }
        
        /// <summary>
        /// Entry time for the position, used for Duration-type strategies
        /// </summary>
        public DateTime? EntryTime { get; set; }
        
        /// <summary>
        /// Time of day for SpecificTimeOfDay-type strategies (e.g., "15:30:00" for 3:30 PM)
        /// </summary>
        public TimeOnly? SpecificTime { get; set; }
    }
}