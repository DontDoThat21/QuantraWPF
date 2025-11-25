using System;
using Quantra.Enums;

namespace Quantra.Models
{
    /// <summary>
    /// Represents a dollar-cost averaging strategy configuration
    /// </summary>
    public class DCAStrategy
    {
        /// <summary>
        /// Unique identifier for this DCA strategy
        /// </summary>
        public string StrategyId { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Symbol of the security to trade
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Optional name or description for this strategy
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// Date and time when the strategy was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.Now;

        /// <summary>
        /// Date and time of the last order execution
        /// </summary>
        public DateTime? LastExecutedAt { get; set; }

        /// <summary>
        /// Date and time of the next scheduled order
        /// </summary>
        public DateTime? NextExecutionAt { get; set; }

        /// <summary>
        /// Whether this strategy is share-based (true) or dollar-based (false)
        /// </summary>
        public bool IsShareBased { get; set; } = true;

        /// <summary>
        /// Total shares to acquire (for share-based strategies)
        /// </summary>
        public int TotalShares { get; set; }

        /// <summary>
        /// Total dollar amount to invest (for dollar-based strategies)
        /// </summary>
        public double TotalAmount { get; set; }

        /// <summary>
        /// Shares to buy in each order (for share-based strategies)
        /// </summary>
        public int SharesPerOrder { get; set; }

        /// <summary>
        /// Dollar amount for each order (for dollar-based strategies)
        /// </summary>
        public double AmountPerOrder { get; set; }

        /// <summary>
        /// Number of orders remaining to execute
        /// </summary>
        public int OrdersRemaining { get; set; }

        /// <summary>
        /// Number of days between each order
        /// </summary>
        public int IntervalDays { get; set; }

        /// <summary>
        /// Distribution strategy type for the orders
        /// </summary>
        public DCAStrategyType StrategyType { get; set; } = DCAStrategyType.Equal;

        /// <summary>
        /// Whether the strategy is currently paused
        /// </summary>
        public bool IsPaused { get; set; } = false;

        /// <summary>
        /// Date and time when the strategy was paused
        /// </summary>
        public DateTime? PausedAt { get; set; }

        /// <summary>
        /// Number of orders that have already been executed
        /// </summary>
        public int OrdersExecuted { get; set; } = 0;

        /// <summary>
        /// Total number of shares acquired so far
        /// </summary>
        public int SharesAcquired { get; set; } = 0;

        /// <summary>
        /// Total amount invested so far
        /// </summary>
        public double AmountInvested { get; set; } = 0;

        /// <summary>
        /// Average price paid per share
        /// </summary>
        public double AveragePricePerShare
        {
            get
            {
                return SharesAcquired > 0 ? AmountInvested / SharesAcquired : 0;
            }
        }
    }
}