using System;
using Quantra.Enums;

namespace Quantra.Models
{
    /// <summary>
    /// Represents an order scheduled for future execution
    /// </summary>
    public class ScheduledOrder
    {
        /// <summary>
        /// Symbol of the security to trade
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Number of shares to trade
        /// </summary>
        public int Quantity { get; set; }

        /// <summary>
        /// BUY or SELL
        /// </summary>
        public string OrderType { get; set; }

        /// <summary>
        /// Limit price for the order
        /// </summary>
        public double Price { get; set; }

        /// <summary>
        /// When the order should be executed
        /// </summary>
        public DateTime ExecutionTime { get; set; }

        /// <summary>
        /// Whether the order is part of a dollar-cost averaging strategy
        /// </summary>
        public bool IsDollarCostAveraging { get; set; }

        /// <summary>
        /// Whether the order is part of a portfolio rebalancing strategy
        /// </summary>
        public bool IsRebalancing { get; set; }

        /// <summary>
        /// Whether this order is a time-based exit
        /// </summary>
        public bool IsTimeBasedExit { get; set; }

        /// <summary>
        /// The time-based exit strategy for this order (if applicable)
        /// </summary>
        public TimeBasedExitStrategy? ExitStrategy { get; set; }

        /// <summary>
        /// Duration in minutes for Duration-type exits
        /// </summary>
        public int? DurationMinutes { get; set; }

        /// <summary>
        /// Optional stop loss price
        /// </summary>
        public double? StopLoss { get; set; }

        /// <summary>
        /// Optional take profit price
        /// </summary>
        public double? TakeProfit { get; set; }

        /// <summary>
        /// Whether this order is part of a split large order
        /// </summary>
        public bool IsSplitOrder { get; set; }

        /// <summary>
        /// Identifier to group related split orders together
        /// </summary>
        public string SplitOrderGroupId { get; set; }

        /// <summary>
        /// Current sequence number in the split order series (e.g., 1 of 5)
        /// </summary>
        public int SplitOrderSequence { get; set; }

        /// <summary>
        /// Total number of chunks in the split order
        /// </summary>
        public int SplitOrderTotalChunks { get; set; }

        /// <summary>
        /// Whether this order is part of a multi-leg strategy
        /// </summary>
        public bool IsMultiLegStrategy { get; set; }

        /// <summary>
        /// Identifier linking to the parent multi-leg strategy
        /// </summary>
        public string MultiLegStrategyId { get; set; }

        /// <summary>
        /// Position of this order in the multi-leg strategy sequence (1-based)
        /// </summary>
        public int LegPosition { get; set; }

        /// <summary>
        /// Strike price (for options legs)
        /// </summary>
        public double? StrikePrice { get; set; }

        /// <summary>
        /// Expiration date (for options legs)
        /// </summary>
        public DateTime? ExpirationDate { get; set; }

        /// <summary>
        /// Option type (for options legs): "CALL" or "PUT"
        /// </summary>
        public string OptionType { get; set; }

        /// <summary>
        /// Whether this leg is an option (true) or equity (false)
        /// </summary>
        public bool IsOption { get; set; }
    }
}