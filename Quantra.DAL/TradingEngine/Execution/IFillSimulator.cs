using System;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Orders;

namespace Quantra.DAL.TradingEngine.Execution
{
    /// <summary>
    /// Result of a fill simulation
    /// </summary>
    public class FillResult
    {
        /// <summary>
        /// Whether the order was filled (at least partially)
        /// </summary>
        public bool IsFilled { get; set; }

        /// <summary>
        /// Quantity filled
        /// </summary>
        public int FilledQuantity { get; set; }

        /// <summary>
        /// Execution price
        /// </summary>
        public decimal ExecutionPrice { get; set; }

        /// <summary>
        /// Slippage incurred
        /// </summary>
        public decimal Slippage { get; set; }

        /// <summary>
        /// Commission charged
        /// </summary>
        public decimal Commission { get; set; }

        /// <summary>
        /// Time of the fill
        /// </summary>
        public DateTime FillTime { get; set; }

        /// <summary>
        /// Reason if order was not filled
        /// </summary>
        public string? RejectReason { get; set; }

        /// <summary>
        /// Creates a successful fill result
        /// </summary>
        public static FillResult Success(int quantity, decimal price, decimal slippage, decimal commission, DateTime time)
        {
            return new FillResult
            {
                IsFilled = true,
                FilledQuantity = quantity,
                ExecutionPrice = price,
                Slippage = slippage,
                Commission = commission,
                FillTime = time
            };
        }

        /// <summary>
        /// Creates a rejected fill result
        /// </summary>
        public static FillResult Rejected(string reason)
        {
            return new FillResult
            {
                IsFilled = false,
                RejectReason = reason
            };
        }
    }

    /// <summary>
    /// Interface for fill simulation
    /// </summary>
    public interface IFillSimulator
    {
        /// <summary>
        /// Attempts to fill an order given the current market quote
        /// </summary>
        /// <param name="order">Order to fill</param>
        /// <param name="quote">Current market quote</param>
        /// <param name="time">Current time</param>
        /// <returns>Fill result</returns>
        FillResult TryFill(Order order, Quote quote, DateTime time);

        /// <summary>
        /// Calculates the expected execution price for an order
        /// </summary>
        /// <param name="order">Order to price</param>
        /// <param name="quote">Current market quote</param>
        /// <returns>Expected execution price</returns>
        decimal GetExpectedPrice(Order order, Quote quote);
    }
}
