using System;

namespace Quantra.DAL.TradingEngine.Orders
{
    /// <summary>
    /// Represents a fill (execution) of an order
    /// </summary>
    public class OrderFill
    {
        /// <summary>
        /// Unique identifier for this fill
        /// </summary>
        public Guid FillId { get; set; } = Guid.NewGuid();

        /// <summary>
        /// Reference to the order that was filled
        /// </summary>
        public Guid OrderId { get; set; }

        /// <summary>
        /// Symbol that was traded
        /// </summary>
        public string Symbol { get; set; } = string.Empty;

        /// <summary>
        /// Quantity filled in this execution
        /// </summary>
        public int Quantity { get; set; }

        /// <summary>
        /// Execution price
        /// </summary>
        public decimal Price { get; set; }

        /// <summary>
        /// Side of the trade (Buy/Sell)
        /// </summary>
        public OrderSide Side { get; set; }

        /// <summary>
        /// Time of the fill
        /// </summary>
        public DateTime FillTime { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Commission charged for this fill
        /// </summary>
        public decimal Commission { get; set; }

        /// <summary>
        /// Slippage incurred (difference from expected price)
        /// </summary>
        public decimal Slippage { get; set; }

        /// <summary>
        /// Exchange or venue where the fill occurred
        /// </summary>
        public string Exchange { get; set; } = "PAPER";

        /// <summary>
        /// Total value of this fill (Quantity * Price)
        /// </summary>
        public decimal TotalValue => Quantity * Price;

        /// <summary>
        /// Net value after commissions
        /// </summary>
        public decimal NetValue => Side == OrderSide.Buy 
            ? -(TotalValue + Commission) 
            : TotalValue - Commission;

        /// <summary>
        /// Indicates if this is a simulated (paper) fill
        /// </summary>
        public bool IsPaperTrade { get; set; } = true;
    }
}
