using System;
using Quantra.Models;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing order history operations using Entity Framework Core
    /// </summary>
    public class OrderHistoryService
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<OrderHistoryService> _logger;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public OrderHistoryService(QuantraDbContext context, ILogger<OrderHistoryService> logger = null)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger;
        }

        /// <summary>
        /// Parameterless constructor for backward compatibility
        /// </summary>
        public OrderHistoryService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        /// <summary>
        /// Adds an order to the history using Entity Framework Core
        /// </summary>
        /// <param name="order">The order to add to history</param>
        public void AddOrderToHistory(OrderModel order)
        {
            if (order == null)
            {
                _logger?.LogError("Cannot add null order to history");
                return;
            }

            try
            {
                // Map OrderModel to OrderHistoryEntity
                var entity = new OrderHistoryEntity
                {
                    Symbol = order.Symbol,
                    OrderType = order.OrderType,
                    Quantity = order.Quantity,
                    Price = order.Price,
                    StopLoss = order.StopLoss > 0 ? order.StopLoss : null,
                    TakeProfit = order.TakeProfit > 0 ? order.TakeProfit : null,
                    IsPaperTrade = order.IsPaperTrade,
                    Status = order.Status,
                    PredictionSource = order.PredictionSource ?? string.Empty,
                    Timestamp = order.Timestamp
                };

                // Add to DbSet and save changes
                _context.OrderHistory.Add(entity);
                _context.SaveChanges();

                _logger?.LogInformation(
                     "Order added to history: {Symbol} {OrderType} {Quantity} @ {Price:C2}",
                 order.Symbol, order.OrderType, order.Quantity, order.Price);
            }
            catch (DbUpdateException dbEx)
            {
                _logger?.LogError(dbEx,
                     "Database error while adding order to history: {Symbol}",
                              order.Symbol);
                throw;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex,
                "Failed to add order to history: {Symbol}",
                      order.Symbol);
                throw;
            }
        }

        /// <summary>
        /// Asynchronously adds an order to the history
        /// </summary>
        /// <param name="order">The order to add to history</param>
        public async System.Threading.Tasks.Task AddOrderToHistoryAsync(OrderModel order)
        {
            if (order == null)
            {
                _logger?.LogError("Cannot add null order to history");
                return;
            }

            try
            {
                // Map OrderModel to OrderHistoryEntity
                var entity = new OrderHistoryEntity
                {
                    Symbol = order.Symbol,
                    OrderType = order.OrderType,
                    Quantity = order.Quantity,
                    Price = order.Price,
                    StopLoss = order.StopLoss > 0 ? order.StopLoss : null,
                    TakeProfit = order.TakeProfit > 0 ? order.TakeProfit : null,
                    IsPaperTrade = order.IsPaperTrade,
                    Status = order.Status,
                    PredictionSource = order.PredictionSource ?? string.Empty,
                    Timestamp = order.Timestamp
                };

                // Add to DbSet and save changes asynchronously
                await _context.OrderHistory.AddAsync(entity);
                await _context.SaveChangesAsync();

                _logger?.LogInformation(
                 "Order added to history: {Symbol} {OrderType} {Quantity} @ {Price:C2}",
                order.Symbol, order.OrderType, order.Quantity, order.Price);
            }
            catch (DbUpdateException dbEx)
            {
                _logger?.LogError(dbEx,
         "Database error while adding order to history: {Symbol}",
order.Symbol);
                throw;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex,
                         "Failed to add order to history: {Symbol}",
                       order.Symbol);
                throw;
            }
        }
    }
}
