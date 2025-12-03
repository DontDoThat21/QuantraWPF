using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.TradingEngine.Data;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;
using Quantra.DAL.TradingEngine.Time;

namespace Quantra.DAL.TradingEngine.Core
{
    /// <summary>
    /// Interface for the unified trading engine supporting paper trading and backtesting
    /// </summary>
    public interface ITradingEngine
    {
        /// <summary>
        /// Initializes the trading engine with data source, clock, and portfolio
        /// </summary>
        void Initialize(IDataSource dataSource, IClock clock, IPortfolioManager portfolio);

        /// <summary>
        /// Places a new order
        /// </summary>
        /// <param name="order">Order to place</param>
        /// <returns>Order ID if successful</returns>
        Task<Guid> PlaceOrderAsync(Order order);

        /// <summary>
        /// Cancels an existing order
        /// </summary>
        /// <param name="orderId">ID of the order to cancel</param>
        /// <returns>True if cancelled successfully</returns>
        bool CancelOrder(Guid orderId);

        /// <summary>
        /// Modifies an existing order
        /// </summary>
        /// <param name="orderId">ID of the order to modify</param>
        /// <param name="newPrice">New limit/stop price (null to keep current)</param>
        /// <param name="newQuantity">New quantity (null to keep current)</param>
        /// <returns>True if modified successfully</returns>
        bool ModifyOrder(Guid orderId, decimal? newPrice, int? newQuantity);

        /// <summary>
        /// Gets an order by ID
        /// </summary>
        Order? GetOrder(Guid orderId);

        /// <summary>
        /// Gets all active (non-terminal) orders
        /// </summary>
        IEnumerable<Order> GetActiveOrders();

        /// <summary>
        /// Gets all orders (including completed/cancelled)
        /// </summary>
        IEnumerable<Order> GetAllOrders();

        /// <summary>
        /// Gets all current positions
        /// </summary>
        IEnumerable<TradingPosition> GetPositions();

        /// <summary>
        /// Gets the portfolio manager
        /// </summary>
        IPortfolioManager GetPortfolio();

        /// <summary>
        /// Gets the current data source
        /// </summary>
        IDataSource? GetDataSource();

        /// <summary>
        /// Gets the current clock
        /// </summary>
        IClock? GetClock();

        /// <summary>
        /// Processes a single time step (for backtesting)
        /// </summary>
        Task ProcessTimeStepAsync(DateTime time);

        /// <summary>
        /// Starts the engine (for paper trading)
        /// </summary>
        void Start();

        /// <summary>
        /// Stops the engine
        /// </summary>
        void Stop();

        /// <summary>
        /// Gets whether the engine is running
        /// </summary>
        bool IsRunning { get; }

        /// <summary>
        /// Gets whether this is a paper trading engine
        /// </summary>
        bool IsPaperTrading { get; }

        /// <summary>
        /// Event raised when an order is filled
        /// </summary>
        event EventHandler<OrderFilledEventArgs>? OrderFilled;

        /// <summary>
        /// Event raised when an order state changes
        /// </summary>
        event EventHandler<OrderStateChangedEventArgs>? OrderStateChanged;
    }

    /// <summary>
    /// Event args for order filled events
    /// </summary>
    public class OrderFilledEventArgs : EventArgs
    {
        public Order Order { get; set; } = null!;
        public OrderFill Fill { get; set; } = null!;
        public DateTime Time { get; set; }
    }

    /// <summary>
    /// Event args for order state change events
    /// </summary>
    public class OrderStateChangedEventArgs : EventArgs
    {
        public Order Order { get; set; } = null!;
        public OrderState OldState { get; set; }
        public OrderState NewState { get; set; }
        public DateTime Time { get; set; }
        public string Reason { get; set; } = string.Empty;
    }
}
