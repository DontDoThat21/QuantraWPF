using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.TradingEngine.Orders;
using Quantra.DAL.TradingEngine.Positions;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for persisting paper trading state to the database
    /// </summary>
    public class PaperTradingPersistenceService : IPaperTradingPersistenceService, IDisposable
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<PaperTradingPersistenceService> _logger;
        private readonly bool _ownsContext;
        private bool _disposed;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public PaperTradingPersistenceService(QuantraDbContext context, ILogger<PaperTradingPersistenceService> logger = null)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger;
            _ownsContext = false; // Context managed externally
        }

        /// <summary>
        /// Parameterless constructor for backward compatibility
        /// </summary>
        public PaperTradingPersistenceService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            _context = new QuantraDbContext(optionsBuilder.Options);
            _ownsContext = true; // We own this context and must dispose it
        }

        #region Session Operations

        public async Task<PaperTradingSessionEntity> CreateSessionAsync(string name, decimal initialCash)
        {
            try
            {
                // Deactivate any existing active sessions
                var activeSessions = await _context.PaperTradingSessions
                    .Where(s => s.IsActive)
                    .ToListAsync();

                foreach (var session in activeSessions)
                {
                    session.IsActive = false;
                    session.EndedAt = DateTime.UtcNow;
                }

                // Create new session
                var newSession = new PaperTradingSessionEntity
                {
                    SessionId = Guid.NewGuid().ToString(),
                    Name = name ?? $"Paper Trading Session {DateTime.UtcNow:yyyy-MM-dd HH:mm}",
                    InitialCash = initialCash,
                    CashBalance = initialCash,
                    RealizedPnL = 0,
                    IsActive = true,
                    StartedAt = DateTime.UtcNow,
                    LastUpdatedAt = DateTime.UtcNow
                };

                await _context.PaperTradingSessions.AddAsync(newSession);
                await _context.SaveChangesAsync();

                _logger?.LogInformation("Created new paper trading session: {SessionId}", newSession.SessionId);

                return newSession;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to create paper trading session");
                throw new InvalidOperationException("Failed to create paper trading session", ex);
            }
        }

        public async Task<PaperTradingSessionEntity> GetActiveSessionAsync()
        {
            try
            {
                return await _context.PaperTradingSessions
                    .Include(s => s.Positions.Where(p => !p.IsClosed))
                    .Include(s => s.Orders)
                    .FirstOrDefaultAsync(s => s.IsActive);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get active session");
                throw new InvalidOperationException("Failed to get active session", ex);
            }
        }

        public async Task<PaperTradingSessionEntity> GetSessionByIdAsync(int sessionId)
        {
            try
            {
                return await _context.PaperTradingSessions
                    .Include(s => s.Positions)
                    .Include(s => s.Orders)
                    .FirstOrDefaultAsync(s => s.Id == sessionId);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get session by ID: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get session by ID: {sessionId}", ex);
            }
        }

        public async Task<PaperTradingSessionEntity> GetSessionByGuidAsync(string sessionGuid)
        {
            try
            {
                return await _context.PaperTradingSessions
                    .Include(s => s.Positions)
                    .Include(s => s.Orders)
                    .FirstOrDefaultAsync(s => s.SessionId == sessionGuid);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get session by GUID: {SessionGuid}", sessionGuid);
                throw new InvalidOperationException($"Failed to get session by GUID: {sessionGuid}", ex);
            }
        }

        public async Task<bool> UpdateSessionStateAsync(int sessionId, decimal cashBalance, decimal realizedPnL)
        {
            try
            {
                var session = await _context.PaperTradingSessions.FindAsync(sessionId);
                if (session == null)
                {
                    _logger?.LogWarning("Session not found for update: {SessionId}", sessionId);
                    return false;
                }

                session.CashBalance = cashBalance;
                session.RealizedPnL = realizedPnL;
                session.LastUpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to update session state: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to update session state: {sessionId}", ex);
            }
        }

        public async Task<bool> EndSessionAsync(int sessionId)
        {
            try
            {
                var session = await _context.PaperTradingSessions.FindAsync(sessionId);
                if (session == null)
                {
                    _logger?.LogWarning("Session not found for ending: {SessionId}", sessionId);
                    return false;
                }

                session.IsActive = false;
                session.EndedAt = DateTime.UtcNow;
                session.LastUpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();
                _logger?.LogInformation("Ended paper trading session: {SessionId}", sessionId);
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to end session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to end session: {sessionId}", ex);
            }
        }

        public async Task<List<PaperTradingSessionEntity>> GetAllSessionsAsync(bool includeInactive = false)
        {
            try
            {
                var query = _context.PaperTradingSessions.AsNoTracking();

                if (!includeInactive)
                {
                    query = query.Where(s => s.IsActive);
                }

                return await query.OrderByDescending(s => s.StartedAt).ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get all sessions");
                throw new InvalidOperationException("Failed to get all sessions", ex);
            }
        }

        #endregion

        #region Position Operations

        public async Task<PaperTradingPositionEntity> SavePositionAsync(int sessionId, TradingPosition position)
        {
            if (position == null)
            {
                throw new ArgumentNullException(nameof(position));
            }

            try
            {
                // Check if position exists for this session and symbol
                var existingPosition = await _context.PaperTradingPositions
                    .FirstOrDefaultAsync(p => p.SessionId == sessionId && 
                                              p.Symbol == position.Symbol && 
                                              !p.IsClosed);

                if (existingPosition != null)
                {
                    // Update existing position
                    existingPosition.Quantity = position.Quantity;
                    existingPosition.AverageCost = position.AverageCost;
                    existingPosition.CurrentPrice = position.CurrentPrice;
                    existingPosition.UnrealizedPnL = position.UnrealizedPnL;
                    existingPosition.RealizedPnL = position.RealizedPnL;
                    existingPosition.LastUpdatedAt = DateTime.UtcNow;

                    // If quantity is 0, mark as closed
                    if (position.Quantity == 0)
                    {
                        existingPosition.IsClosed = true;
                        existingPosition.ClosedAt = DateTime.UtcNow;
                    }

                    await _context.SaveChangesAsync();
                    return existingPosition;
                }
                else
                {
                    // Create new position
                    var newPosition = new PaperTradingPositionEntity
                    {
                        SessionId = sessionId,
                        Symbol = position.Symbol,
                        Quantity = position.Quantity,
                        AverageCost = position.AverageCost,
                        CurrentPrice = position.CurrentPrice,
                        UnrealizedPnL = position.UnrealizedPnL,
                        RealizedPnL = position.RealizedPnL,
                        AssetType = position.AssetType.ToString(),
                        IsClosed = position.IsFlat,
                        OpenedAt = position.OpenedTime,
                        ClosedAt = position.IsFlat ? DateTime.UtcNow : null,
                        LastUpdatedAt = DateTime.UtcNow
                    };

                    await _context.PaperTradingPositions.AddAsync(newPosition);
                    await _context.SaveChangesAsync();

                    _logger?.LogInformation("Saved position: {Symbol} qty={Quantity}", position.Symbol, position.Quantity);
                    return newPosition;
                }
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to save position for {Symbol}", position.Symbol);
                throw new InvalidOperationException($"Failed to save position for {position.Symbol}", ex);
            }
        }

        public async Task<List<PaperTradingPositionEntity>> GetOpenPositionsAsync(int sessionId)
        {
            try
            {
                return await _context.PaperTradingPositions
                    .AsNoTracking()
                    .Where(p => p.SessionId == sessionId && !p.IsClosed)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get open positions for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get open positions for session: {sessionId}", ex);
            }
        }

        public async Task<List<PaperTradingPositionEntity>> GetAllPositionsAsync(int sessionId)
        {
            try
            {
                return await _context.PaperTradingPositions
                    .AsNoTracking()
                    .Where(p => p.SessionId == sessionId)
                    .OrderByDescending(p => p.OpenedAt)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get all positions for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get all positions for session: {sessionId}", ex);
            }
        }

        public async Task<bool> ClosePositionAsync(int positionId)
        {
            try
            {
                var position = await _context.PaperTradingPositions.FindAsync(positionId);
                if (position == null)
                {
                    _logger?.LogWarning("Position not found for closing: {PositionId}", positionId);
                    return false;
                }

                position.IsClosed = true;
                position.ClosedAt = DateTime.UtcNow;
                position.LastUpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to close position: {PositionId}", positionId);
                throw new InvalidOperationException($"Failed to close position: {positionId}", ex);
            }
        }

        public async Task<int> ClearPositionsAsync(int sessionId)
        {
            try
            {
                var positions = await _context.PaperTradingPositions
                    .Where(p => p.SessionId == sessionId)
                    .ToListAsync();

                _context.PaperTradingPositions.RemoveRange(positions);
                await _context.SaveChangesAsync();

                _logger?.LogInformation("Cleared {Count} positions for session: {SessionId}", positions.Count, sessionId);
                return positions.Count;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to clear positions for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to clear positions for session: {sessionId}", ex);
            }
        }

        #endregion

        #region Order Operations

        public async Task<PaperTradingOrderEntity> SaveOrderAsync(int sessionId, Order order)
        {
            if (order == null)
            {
                throw new ArgumentNullException(nameof(order));
            }

            try
            {
                // Check if order already exists
                var existingOrder = await _context.PaperTradingOrders
                    .FirstOrDefaultAsync(o => o.OrderId == order.Id.ToString());

                if (existingOrder != null)
                {
                    // Update existing order
                    return await UpdateOrderEntityAsync(existingOrder, order);
                }

                // Create new order
                var newOrder = new PaperTradingOrderEntity
                {
                    OrderId = order.Id.ToString(),
                    SessionId = sessionId,
                    Symbol = order.Symbol,
                    OrderType = order.OrderType.ToString(),
                    Side = order.Side.ToString(),
                    State = order.State.ToString(),
                    Quantity = order.Quantity,
                    FilledQuantity = order.FilledQuantity,
                    LimitPrice = order.LimitPrice,
                    StopPrice = order.StopPrice,
                    AvgFillPrice = order.AvgFillPrice,
                    TimeInForce = order.TimeInForce.ToString(),
                    AssetType = order.AssetType.ToString(),
                    RejectReason = order.RejectReason,
                    Notes = order.Notes,
                    CreatedAt = order.CreatedTime,
                    SubmittedAt = order.SubmittedTime,
                    FilledAt = order.FilledTime,
                    ExpirationTime = order.ExpirationTime,
                    LastUpdatedAt = DateTime.UtcNow
                };

                await _context.PaperTradingOrders.AddAsync(newOrder);
                await _context.SaveChangesAsync();

                _logger?.LogInformation("Saved order: {OrderId} {Side} {Quantity} {Symbol}", 
                    order.Id, order.Side, order.Quantity, order.Symbol);
                return newOrder;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to save order: {OrderId}", order.Id);
                throw new InvalidOperationException($"Failed to save order: {order.Id}", ex);
            }
        }

        public async Task<bool> UpdateOrderAsync(Guid orderId, Order order)
        {
            try
            {
                var existingOrder = await _context.PaperTradingOrders
                    .FirstOrDefaultAsync(o => o.OrderId == orderId.ToString());

                if (existingOrder == null)
                {
                    _logger?.LogWarning("Order not found for update: {OrderId}", orderId);
                    return false;
                }

                await UpdateOrderEntityAsync(existingOrder, order);
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to update order: {OrderId}", orderId);
                throw new InvalidOperationException($"Failed to update order: {orderId}", ex);
            }
        }

        private async Task<PaperTradingOrderEntity> UpdateOrderEntityAsync(PaperTradingOrderEntity entity, Order order)
        {
            entity.State = order.State.ToString();
            entity.FilledQuantity = order.FilledQuantity;
            entity.AvgFillPrice = order.AvgFillPrice;
            entity.SubmittedAt = order.SubmittedTime;
            entity.FilledAt = order.FilledTime;
            entity.RejectReason = order.RejectReason;
            entity.Notes = order.Notes;
            entity.LastUpdatedAt = DateTime.UtcNow;

            await _context.SaveChangesAsync();
            return entity;
        }

        public async Task<List<PaperTradingOrderEntity>> GetAllOrdersAsync(int sessionId)
        {
            try
            {
                return await _context.PaperTradingOrders
                    .AsNoTracking()
                    .Where(o => o.SessionId == sessionId)
                    .OrderByDescending(o => o.CreatedAt)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get all orders for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get all orders for session: {sessionId}", ex);
            }
        }

        public async Task<List<PaperTradingOrderEntity>> GetActiveOrdersAsync(int sessionId)
        {
            try
            {
                var terminalStates = new[] { "Filled", "Cancelled", "Expired", "Rejected" };

                return await _context.PaperTradingOrders
                    .AsNoTracking()
                    .Where(o => o.SessionId == sessionId && !terminalStates.Contains(o.State))
                    .OrderByDescending(o => o.CreatedAt)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get active orders for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get active orders for session: {sessionId}", ex);
            }
        }

        public async Task<PaperTradingOrderEntity> GetOrderByIdAsync(Guid orderId)
        {
            try
            {
                return await _context.PaperTradingOrders
                    .AsNoTracking()
                    .Include(o => o.Fills)
                    .FirstOrDefaultAsync(o => o.OrderId == orderId.ToString());
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get order by ID: {OrderId}", orderId);
                throw new InvalidOperationException($"Failed to get order by ID: {orderId}", ex);
            }
        }

        public async Task<int> ClearOrdersAsync(int sessionId)
        {
            try
            {
                var orders = await _context.PaperTradingOrders
                    .Where(o => o.SessionId == sessionId)
                    .ToListAsync();

                _context.PaperTradingOrders.RemoveRange(orders);
                await _context.SaveChangesAsync();

                _logger?.LogInformation("Cleared {Count} orders for session: {SessionId}", orders.Count, sessionId);
                return orders.Count;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to clear orders for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to clear orders for session: {sessionId}", ex);
            }
        }

        #endregion

        #region Fill Operations

        public async Task<PaperTradingFillEntity> SaveFillAsync(int orderEntityId, int? positionEntityId, OrderFill fill)
        {
            if (fill == null)
            {
                throw new ArgumentNullException(nameof(fill));
            }

            try
            {
                var newFill = new PaperTradingFillEntity
                {
                    FillId = fill.FillId.ToString(),
                    OrderEntityId = orderEntityId,
                    PositionEntityId = positionEntityId,
                    Symbol = fill.Symbol,
                    Quantity = fill.Quantity,
                    Price = fill.Price,
                    Side = fill.Side.ToString(),
                    Commission = fill.Commission,
                    Slippage = fill.Slippage,
                    Exchange = fill.Exchange,
                    FillTime = fill.FillTime
                };

                await _context.PaperTradingFills.AddAsync(newFill);
                await _context.SaveChangesAsync();

                _logger?.LogInformation("Saved fill: {FillId} {Side} {Quantity} {Symbol} @ {Price}", 
                    fill.FillId, fill.Side, fill.Quantity, fill.Symbol, fill.Price);
                return newFill;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to save fill: {FillId}", fill.FillId);
                throw new InvalidOperationException($"Failed to save fill: {fill.FillId}", ex);
            }
        }

        public async Task<List<PaperTradingFillEntity>> GetFillsByOrderAsync(int orderEntityId)
        {
            try
            {
                return await _context.PaperTradingFills
                    .AsNoTracking()
                    .Where(f => f.OrderEntityId == orderEntityId)
                    .OrderBy(f => f.FillTime)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get fills for order: {OrderEntityId}", orderEntityId);
                throw new InvalidOperationException($"Failed to get fills for order: {orderEntityId}", ex);
            }
        }

        public async Task<List<PaperTradingFillEntity>> GetAllFillsAsync(int sessionId)
        {
            try
            {
                return await _context.PaperTradingFills
                    .AsNoTracking()
                    .Where(f => f.Order.SessionId == sessionId)
                    .OrderByDescending(f => f.FillTime)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to get all fills for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to get all fills for session: {sessionId}", ex);
            }
        }

        #endregion

        #region Restoration Operations

        public async Task<Dictionary<string, TradingPosition>> RestorePositionsAsync(int sessionId)
        {
            try
            {
                var positions = await GetOpenPositionsAsync(sessionId);
                var result = new Dictionary<string, TradingPosition>(StringComparer.OrdinalIgnoreCase);

                foreach (var entity in positions)
                {
                    var position = new TradingPosition
                    {
                        Symbol = entity.Symbol,
                        Quantity = entity.Quantity,
                        AverageCost = entity.AverageCost,
                        CurrentPrice = entity.CurrentPrice,
                        RealizedPnL = entity.RealizedPnL,
                        OpenedTime = entity.OpenedAt,
                        LastUpdateTime = entity.LastUpdatedAt,
                        AssetType = Enum.TryParse<AssetType>(entity.AssetType, out var at) ? at : AssetType.Stock
                    };

                    result[entity.Symbol] = position;
                }

                _logger?.LogInformation("Restored {Count} positions for session: {SessionId}", result.Count, sessionId);
                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to restore positions for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to restore positions for session: {sessionId}", ex);
            }
        }

        public async Task<Dictionary<Guid, Order>> RestoreOrdersAsync(int sessionId)
        {
            try
            {
                var orders = await GetAllOrdersAsync(sessionId);
                var result = new Dictionary<Guid, Order>();

                foreach (var entity in orders)
                {
                    if (!Guid.TryParse(entity.OrderId, out var orderId))
                    {
                        continue;
                    }

                    var order = new Order
                    {
                        Id = orderId,
                        Symbol = entity.Symbol,
                        OrderType = Enum.TryParse<OrderType>(entity.OrderType, out var ot) ? ot : OrderType.Market,
                        Side = Enum.TryParse<OrderSide>(entity.Side, out var os) ? os : OrderSide.Buy,
                        State = Enum.TryParse<OrderState>(entity.State, out var st) ? st : OrderState.Pending,
                        Quantity = entity.Quantity,
                        FilledQuantity = entity.FilledQuantity,
                        LimitPrice = entity.LimitPrice,
                        StopPrice = entity.StopPrice,
                        AvgFillPrice = entity.AvgFillPrice,
                        TimeInForce = Enum.TryParse<TimeInForce>(entity.TimeInForce, out var tif) ? tif : TimeInForce.Day,
                        AssetType = Enum.TryParse<AssetType>(entity.AssetType, out var at) ? at : AssetType.Stock,
                        CreatedTime = entity.CreatedAt,
                        SubmittedTime = entity.SubmittedAt,
                        FilledTime = entity.FilledAt,
                        ExpirationTime = entity.ExpirationTime,
                        RejectReason = entity.RejectReason ?? string.Empty,
                        Notes = entity.Notes ?? string.Empty,
                        IsPaperTrade = true
                    };

                    result[orderId] = order;
                }

                _logger?.LogInformation("Restored {Count} orders for session: {SessionId}", result.Count, sessionId);
                return result;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to restore orders for session: {SessionId}", sessionId);
                throw new InvalidOperationException($"Failed to restore orders for session: {sessionId}", ex);
            }
        }

        #endregion

        #region IDisposable

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing && _ownsContext)
                {
                    _context?.Dispose();
                }
                _disposed = true;
            }
        }

        #endregion
    }
}
