using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;

namespace Quantra.DAL.TradingEngine.Core
{
    /// <summary>
    /// Interface for portfolio management
    /// </summary>
    public interface IPortfolioManager
    {
        /// <summary>
        /// Gets the current cash balance
        /// </summary>
        decimal CashBalance { get; }

        /// <summary>
        /// Gets the total portfolio value (cash + positions)
        /// </summary>
        decimal TotalValue { get; }

        /// <summary>
        /// Gets the total unrealized P&L
        /// </summary>
        decimal UnrealizedPnL { get; }

        /// <summary>
        /// Gets the total realized P&L
        /// </summary>
        decimal RealizedPnL { get; }

        /// <summary>
        /// Gets all current positions
        /// </summary>
        IReadOnlyDictionary<string, TradingPosition> Positions { get; }

        /// <summary>
        /// Gets the buying power (available for new trades)
        /// </summary>
        decimal BuyingPower { get; }

        /// <summary>
        /// Gets a position for a symbol
        /// </summary>
        TradingPosition? GetPosition(string symbol);

        /// <summary>
        /// Updates a position with a new fill
        /// </summary>
        void ProcessFill(OrderFill fill);

        /// <summary>
        /// Updates all positions with current market prices
        /// </summary>
        Task UpdatePricesAsync(IDataSource dataSource, DateTime time);

        /// <summary>
        /// Takes a snapshot of the current portfolio state
        /// </summary>
        PortfolioSnapshot TakeSnapshot(DateTime time);

        /// <summary>
        /// Resets the portfolio to initial state
        /// </summary>
        void Reset(decimal initialCash);

        /// <summary>
        /// Event raised when portfolio value changes
        /// </summary>
        event EventHandler<PortfolioChangedEventArgs>? PortfolioChanged;
    }

    /// <summary>
    /// Event args for portfolio changes
    /// </summary>
    public class PortfolioChangedEventArgs : EventArgs
    {
        public decimal OldValue { get; set; }
        public decimal NewValue { get; set; }
        public string Reason { get; set; } = string.Empty;
        public DateTime Time { get; set; }
    }

    /// <summary>
    /// Snapshot of portfolio state at a point in time
    /// </summary>
    public class PortfolioSnapshot
    {
        public DateTime Time { get; set; }
        public decimal CashBalance { get; set; }
        public decimal PositionsValue { get; set; }
        public decimal TotalValue { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal RealizedPnL { get; set; }
        public Dictionary<string, PositionSnapshot> Positions { get; set; } = new Dictionary<string, PositionSnapshot>();
    }

    /// <summary>
    /// Snapshot of a single position
    /// </summary>
    public class PositionSnapshot
    {
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal AverageCost { get; set; }
        public decimal CurrentPrice { get; set; }
        public decimal MarketValue { get; set; }
        public decimal UnrealizedPnL { get; set; }
    }
}
