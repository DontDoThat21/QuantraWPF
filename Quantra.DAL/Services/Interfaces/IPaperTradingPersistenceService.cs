using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for paper trading persistence operations
    /// </summary>
    public interface IPaperTradingPersistenceService
    {
        #region Session Operations

        /// <summary>
        /// Creates a new paper trading session
        /// </summary>
        /// <param name="name">Optional session name</param>
        /// <param name="initialCash">Initial cash balance</param>
        /// <returns>The created session entity</returns>
        Task<PaperTradingSessionEntity> CreateSessionAsync(string name, decimal initialCash);

        /// <summary>
        /// Gets the current active session or null if none exists
        /// </summary>
        /// <returns>The active session or null</returns>
        Task<PaperTradingSessionEntity> GetActiveSessionAsync();

        /// <summary>
        /// Gets a session by ID
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>The session entity or null</returns>
        Task<PaperTradingSessionEntity> GetSessionByIdAsync(int sessionId);

        /// <summary>
        /// Gets a session by its unique session identifier
        /// </summary>
        /// <param name="sessionGuid">The session GUID string</param>
        /// <returns>The session entity or null</returns>
        Task<PaperTradingSessionEntity> GetSessionByGuidAsync(string sessionGuid);

        /// <summary>
        /// Updates session state (cash balance, realized P&L, etc.)
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <param name="cashBalance">Current cash balance</param>
        /// <param name="realizedPnL">Total realized P&L</param>
        /// <returns>True if update was successful</returns>
        Task<bool> UpdateSessionStateAsync(int sessionId, decimal cashBalance, decimal realizedPnL);

        /// <summary>
        /// Ends a session (sets IsActive to false)
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>True if successful</returns>
        Task<bool> EndSessionAsync(int sessionId);

        /// <summary>
        /// Gets all sessions
        /// </summary>
        /// <param name="includeInactive">Whether to include inactive sessions</param>
        /// <returns>List of session entities</returns>
        Task<List<PaperTradingSessionEntity>> GetAllSessionsAsync(bool includeInactive = false);

        #endregion

        #region Position Operations

        /// <summary>
        /// Saves or updates a position
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <param name="position">The trading position to save</param>
        /// <returns>The saved position entity</returns>
        Task<PaperTradingPositionEntity> SavePositionAsync(int sessionId, TradingPosition position);

        /// <summary>
        /// Gets all open positions for a session
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>List of open positions</returns>
        Task<List<PaperTradingPositionEntity>> GetOpenPositionsAsync(int sessionId);

        /// <summary>
        /// Gets all positions for a session (including closed)
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>List of all positions</returns>
        Task<List<PaperTradingPositionEntity>> GetAllPositionsAsync(int sessionId);

        /// <summary>
        /// Marks a position as closed
        /// </summary>
        /// <param name="positionId">The position database ID</param>
        /// <returns>True if successful</returns>
        Task<bool> ClosePositionAsync(int positionId);

        /// <summary>
        /// Removes all positions for a session (for reset)
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>Number of positions removed</returns>
        Task<int> ClearPositionsAsync(int sessionId);

        #endregion

        #region Order Operations

        /// <summary>
        /// Saves an order to the database
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <param name="order">The order to save</param>
        /// <returns>The saved order entity</returns>
        Task<PaperTradingOrderEntity> SaveOrderAsync(int sessionId, Order order);

        /// <summary>
        /// Updates an order's state
        /// </summary>
        /// <param name="orderId">The order GUID</param>
        /// <param name="order">The updated order</param>
        /// <returns>True if successful</returns>
        Task<bool> UpdateOrderAsync(Guid orderId, Order order);

        /// <summary>
        /// Gets all orders for a session
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>List of all orders</returns>
        Task<List<PaperTradingOrderEntity>> GetAllOrdersAsync(int sessionId);

        /// <summary>
        /// Gets active (non-terminal) orders for a session
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>List of active orders</returns>
        Task<List<PaperTradingOrderEntity>> GetActiveOrdersAsync(int sessionId);

        /// <summary>
        /// Gets an order by its GUID
        /// </summary>
        /// <param name="orderId">The order GUID</param>
        /// <returns>The order entity or null</returns>
        Task<PaperTradingOrderEntity> GetOrderByIdAsync(Guid orderId);

        /// <summary>
        /// Removes all orders for a session (for reset)
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>Number of orders removed</returns>
        Task<int> ClearOrdersAsync(int sessionId);

        #endregion

        #region Fill Operations

        /// <summary>
        /// Saves a fill to the database
        /// </summary>
        /// <param name="orderEntityId">The order database ID</param>
        /// <param name="positionEntityId">The position database ID (optional)</param>
        /// <param name="fill">The fill to save</param>
        /// <returns>The saved fill entity</returns>
        Task<PaperTradingFillEntity> SaveFillAsync(int orderEntityId, int? positionEntityId, OrderFill fill);

        /// <summary>
        /// Gets all fills for an order
        /// </summary>
        /// <param name="orderEntityId">The order database ID</param>
        /// <returns>List of fills</returns>
        Task<List<PaperTradingFillEntity>> GetFillsByOrderAsync(int orderEntityId);

        /// <summary>
        /// Gets all fills for a session
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>List of all fills</returns>
        Task<List<PaperTradingFillEntity>> GetAllFillsAsync(int sessionId);

        #endregion

        #region Restoration Operations

        /// <summary>
        /// Restores trading positions from the database into a PortfolioManager
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>Dictionary of symbol to TradingPosition</returns>
        Task<Dictionary<string, TradingPosition>> RestorePositionsAsync(int sessionId);

        /// <summary>
        /// Restores orders from the database
        /// </summary>
        /// <param name="sessionId">The session database ID</param>
        /// <returns>Dictionary of order ID to Order</returns>
        Task<Dictionary<Guid, Order>> RestoreOrdersAsync(int sessionId);

        #endregion
    }
}
