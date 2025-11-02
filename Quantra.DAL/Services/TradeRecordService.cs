using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing trade records using Entity Framework Core
    /// </summary>
    public class TradeRecordService : ITradeRecordService
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<TradeRecordService> _logger;

        /// <summary>
        /// Constructor with dependency injection
        /// </summary>
        public TradeRecordService(QuantraDbContext context, ILogger<TradeRecordService> logger = null)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger;
        }

        /// <summary>
        /// Parameterless constructor for backward compatibility
        /// </summary>
        public TradeRecordService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        /// <summary>
        /// Saves a trade record synchronously using Entity Framework Core
        /// </summary>
        /// <param name="trade">The trade record to save</param>
        public void SaveTradeRecord(TradeRecord trade)
        {
            if (trade == null)
            {
                _logger?.LogError("Cannot save null TradeRecord");
                return;
            }

            try
            {
                // Map TradeRecord model to TradeRecordEntity
                var entity = MapToEntity(trade);

                // Add to DbSet and save changes
                _context.TradeRecords.Add(entity);
                _context.SaveChanges();

                // Update the model with the generated ID
                trade.Id = entity.Id;

                _logger?.LogInformation(
                    "TradeRecord saved: {Symbol} {Action} @ {Price:C}",
                    trade.Symbol, trade.Action, trade.Price);
            }
            catch (DbUpdateException dbEx)
            {
                _logger?.LogError(dbEx,
                    "Database error while saving TradeRecord for {Symbol}",
                    trade.Symbol);
                throw new InvalidOperationException($"Failed to save TradeRecord for {trade.Symbol} to database", dbEx);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex,
                    "Failed to save TradeRecord for {Symbol}",
                    trade.Symbol);
                throw new InvalidOperationException($"Failed to save TradeRecord for {trade.Symbol}", ex);
            }
        }

        /// <summary>
        /// Saves a trade record asynchronously using Entity Framework Core
        /// </summary>
        /// <param name="trade">The trade record to save</param>
        public async Task SaveTradeRecordAsync(TradeRecord trade)
        {
            if (trade == null)
            {
                _logger?.LogError("Cannot save null TradeRecord");
                throw new ArgumentNullException(nameof(trade));
            }

            try
            {
                // Map TradeRecord model to TradeRecordEntity
                var entity = MapToEntity(trade);

                // Add to DbSet and save changes asynchronously
                await _context.TradeRecords.AddAsync(entity);
                await _context.SaveChangesAsync();

                // Update the model with the generated ID
                trade.Id = entity.Id;

                _logger?.LogInformation(
                    "TradeRecord saved: {Symbol} {Action} @ {Price:C}",
                    trade.Symbol, trade.Action, trade.Price);
            }
            catch (DbUpdateException dbEx)
            {
                _logger?.LogError(dbEx,
                    "Database error while saving TradeRecord for {Symbol}",
                    trade.Symbol);
                throw new InvalidOperationException($"Failed to save TradeRecord for {trade.Symbol} to database", dbEx);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex,
                    "Failed to save TradeRecord for {Symbol}",
                    trade.Symbol);
                throw new InvalidOperationException($"Failed to save TradeRecord for {trade.Symbol}", ex);
            }
        }

        /// <summary>
        /// Gets trade records, optionally filtered by symbol
        /// </summary>
        /// <param name="symbol">Optional stock symbol to filter records (null for all records)</param>
        /// <returns>List of trade records matching the criteria</returns>
        public async Task<List<TradeRecord>> GetTradeRecordsAsync(string symbol = null)
        {
            try
            {
                var query = _context.TradeRecords.AsNoTracking();

                if (!string.IsNullOrWhiteSpace(symbol))
                {
                    query = query.Where(r => r.Symbol == symbol.ToUpper());
                }

                var entities = await query
                    .OrderByDescending(r => r.ExecutionTime)
                    .ToListAsync();

                return entities.Select(MapToModel).ToList();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to retrieve trade records for symbol '{Symbol}'", symbol);
                throw new InvalidOperationException($"Failed to retrieve trade records for symbol '{symbol}'", ex);
            }
        }

        /// <summary>
        /// Gets trade records within a specific date range
        /// </summary>
        /// <param name="startDate">Start date of the range</param>
        /// <param name="endDate">End date of the range</param>
        /// <returns>List of trade records in the date range</returns>
        public async Task<List<TradeRecord>> GetTradeRecordsByDateRangeAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                var entities = await _context.TradeRecords
                    .AsNoTracking()
                    .Where(r => r.ExecutionTime >= startDate && r.ExecutionTime <= endDate)
                    .OrderByDescending(r => r.ExecutionTime)
                    .ToListAsync();

                return entities.Select(MapToModel).ToList();
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to retrieve trade records by date range");
                throw new InvalidOperationException("Failed to retrieve trade records by date range", ex);
            }
        }

        /// <summary>
        /// Gets a specific trade record by ID
        /// </summary>
        /// <param name="id">The ID of the trade record to retrieve</param>
        /// <returns>The trade record or null if not found</returns>
        public async Task<TradeRecord> GetTradeRecordByIdAsync(int id)
        {
            try
            {
                var entity = await _context.TradeRecords
                    .AsNoTracking()
                    .FirstOrDefaultAsync(r => r.Id == id);

                return entity != null ? MapToModel(entity) : null;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to retrieve trade record with ID {Id}", id);
                throw new InvalidOperationException($"Failed to retrieve trade record with ID {id}", ex);
            }
        }

        /// <summary>
        /// Deletes a trade record by ID
        /// </summary>
        /// <param name="id">The ID of the trade record to delete</param>
        /// <returns>True if deleted successfully, false if record not found</returns>
        public async Task<bool> DeleteTradeRecordAsync(int id)
        {
            try
            {
                var entity = await _context.TradeRecords.FindAsync(id);
                if (entity == null)
                {
                    _logger?.LogWarning("Trade record with ID {Id} not found for deletion", id);
                    return false;
                }

                _context.TradeRecords.Remove(entity);
                await _context.SaveChangesAsync();

                _logger?.LogInformation("Trade record with ID {Id} deleted successfully", id);
                return true;
            }
            catch (DbUpdateException ex)
            {
                _logger?.LogError(ex, "Database error while deleting trade record with ID {Id}", id);
                throw new InvalidOperationException($"Failed to delete trade record with ID {id} from database", ex);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to delete trade record with ID {Id}", id);
                throw new InvalidOperationException($"Failed to delete trade record with ID {id}", ex);
            }
        }

        #region Mapping Methods

        /// <summary>
        /// Maps TradeRecordEntity to TradeRecord model
        /// </summary>
        private TradeRecord MapToModel(TradeRecordEntity entity)
        {
            if (entity == null)
                return null;

            return new TradeRecord
            {
                Id = entity.Id,
                Symbol = entity.Symbol,
                Action = entity.Action,
                Price = entity.Price,
                TargetPrice = entity.TargetPrice,
                Confidence = entity.Confidence,
                ExecutionTime = entity.ExecutionTime,
                Status = entity.Status,
                Notes = entity.Notes,
                // Note: Quantity and TimeStamp are in TradeRecord model but not in TradeRecordEntity
                // These may need to be added to the entity or handled differently
                Quantity = 0, // Default value - may need to be added to entity
                TimeStamp = entity.ExecutionTime // Use ExecutionTime as TimeStamp
            };
        }

        /// <summary>
        /// Maps TradeRecord model to TradeRecordEntity
        /// </summary>
        private TradeRecordEntity MapToEntity(TradeRecord model)
        {
            if (model == null)
                return null;

            return new TradeRecordEntity
            {
                Id = model.Id,
                Symbol = model.Symbol?.ToUpper(),
                Action = model.Action,
                Price = model.Price,
                TargetPrice = model.TargetPrice,
                Confidence = model.Confidence,
                ExecutionTime = model.ExecutionTime,
                Status = model.Status ?? "Executed",
                Notes = model.Notes
            };
        }

        #endregion
    }
}
