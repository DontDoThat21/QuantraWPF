using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    public class TransactionService : ITransactionService
    {
        private readonly QuantraDbContext _context;

        public TransactionService(QuantraDbContext context)
        {
            ArgumentNullException.ThrowIfNull(context);
            _context = context;
        }

        public TransactionService()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            _context = new QuantraDbContext(optionsBuilder.Options);
        }

        public List<TransactionModel> GetTransactions()
        {
            return ResilienceHelper.Retry(() =>
            {
                try
                {
                    return GetOrdersFromDatabase();
                }
                catch (Exception)
                {
                    return new List<TransactionModel>();
                }
            }, RetryOptions.ForUserFacingOperation());
        }

        public async Task<List<TransactionModel>> GetTransactionsAsync()
        {
            try
            {
                _context.Database.EnsureCreated();

                var orderEntities = await _context.OrderHistory
                    .AsNoTracking()
                    .OrderByDescending(o => o.Timestamp)
                    .ToListAsync();

                return orderEntities.Select(MapToTransactionModel).ToList();
            }
            catch (Exception)
            {
                return new List<TransactionModel>();
            }
        }

        private List<TransactionModel> GetOrdersFromDatabase()
        {
            _context.Database.EnsureCreated();

            var orderEntities = _context.OrderHistory
                .AsNoTracking()
                .OrderByDescending(o => o.Timestamp)
                .ToList();

            return orderEntities.Select(MapToTransactionModel).ToList();
        }

        private static TransactionModel MapToTransactionModel(OrderHistoryEntity entity)
        {
            return new TransactionModel
            {
                Symbol = entity.Symbol,
                TransactionType = entity.OrderType,
                Quantity = entity.Quantity,
                ExecutionPrice = (double)entity.Price,
                TotalValue = (double)(entity.Price * entity.Quantity),
                ExecutionTime = entity.Timestamp,
                IsPaperTrade = entity.IsPaperTrade,
                Fees = 0.0,
                RealizedPnL = 0.0,
                RealizedPnLPercentage = 0.0,
                Notes = entity.PredictionSource ?? string.Empty,
                OrderSource = string.IsNullOrEmpty(entity.PredictionSource) ? "Manual" : "Automated",
                Status = entity.Status
            };
        }

        public TransactionModel GetTransaction(int id)
        {
            try
            {
                var entity = _context.OrderHistory
                    .AsNoTracking()
                    .FirstOrDefault(o => o.Id == id);

                return entity == null ? null : MapToTransactionModel(entity);
            }
            catch (Exception)
            {
                throw;
            }
        }

        public async Task<TransactionModel> GetTransactionAsync(int id)
        {
            try
            {
                var entity = await _context.OrderHistory
                    .AsNoTracking()
                    .FirstOrDefaultAsync(o => o.Id == id);

                return entity == null ? null : MapToTransactionModel(entity);
            }
            catch (Exception)
            {
                throw;
            }
        }

        public List<TransactionModel> GetTransactionsByDateRange(DateTime startDate, DateTime endDate)
        {
            try
            {
                var entities = _context.OrderHistory
                    .AsNoTracking()
                    .Where(o => o.Timestamp >= startDate && o.Timestamp <= endDate)
                    .OrderByDescending(o => o.Timestamp)
                    .ToList();

                return entities.Select(MapToTransactionModel).ToList();
            }
            catch (Exception)
            {
                throw;
            }
        }

        public async Task<List<TransactionModel>> GetTransactionsByDateRangeAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                var entities = await _context.OrderHistory
                    .AsNoTracking()
                    .Where(o => o.Timestamp >= startDate && o.Timestamp <= endDate)
                    .OrderByDescending(o => o.Timestamp)
                    .ToListAsync();

                return entities.Select(MapToTransactionModel).ToList();
            }
            catch (Exception)
            {
                throw;
            }
        }

        public List<TransactionModel> GetTransactionsBySymbol(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                throw new ArgumentException("Symbol cannot be null or empty.", nameof(symbol));
            }

            try
            {
                var entities = _context.OrderHistory
                    .AsNoTracking()
                    .Where(o => o.Symbol == symbol)
                    .OrderByDescending(o => o.Timestamp)
                    .ToList();

                return entities.Select(MapToTransactionModel).ToList();
            }
            catch (Exception)
            {
                throw;
            }
        }

        public async Task<List<TransactionModel>> GetTransactionsBySymbolAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                throw new ArgumentException("Symbol cannot be null or empty.", nameof(symbol));
            }

            try
            {
                var entities = await _context.OrderHistory
                    .AsNoTracking()
                    .Where(o => o.Symbol == symbol)
                    .OrderByDescending(o => o.Timestamp)
                    .ToListAsync();

                return entities.Select(MapToTransactionModel).ToList();
            }
            catch (Exception)
            {
                throw;
            }
        }

        public void SaveTransaction(TransactionModel transaction)
        {
            ArgumentNullException.ThrowIfNull(transaction);
            if (string.IsNullOrWhiteSpace(transaction.Symbol))
            {
                throw new ArgumentException("Transaction symbol cannot be null or empty.", nameof(transaction));
            }

            ResilienceHelper.Retry(() =>
            {
                try
                {
                    var entity = new OrderHistoryEntity
                    {
                        Symbol = transaction.Symbol,
                        OrderType = transaction.TransactionType,
                        Quantity = transaction.Quantity,
                        Price = (float)transaction.ExecutionPrice,
                        StopLoss = null,
                        TakeProfit = null,
                        IsPaperTrade = transaction.IsPaperTrade,
                        Status = string.IsNullOrWhiteSpace(transaction.Status) ? "Executed" : transaction.Status,
                        PredictionSource = transaction.Notes ?? string.Empty,
                        Timestamp = transaction.ExecutionTime
                    };

                    _context.OrderHistory.Add(entity);
                    _context.SaveChanges();
                }
                catch (Exception)
                {
                    throw;
                }
            }, RetryOptions.ForCriticalOperation());
        }

        public async Task SaveTransactionAsync(TransactionModel transaction)
        {
            ArgumentNullException.ThrowIfNull(transaction);
            if (string.IsNullOrWhiteSpace(transaction.Symbol))
            {
                throw new ArgumentException("Transaction symbol cannot be null or empty.", nameof(transaction));
            }

            try
            {
                var entity = new OrderHistoryEntity
                {
                    Symbol = transaction.Symbol,
                    OrderType = transaction.TransactionType,
                    Quantity = transaction.Quantity,
                    Price = (float)transaction.ExecutionPrice,
                    StopLoss = null,
                    TakeProfit = null,
                    IsPaperTrade = transaction.IsPaperTrade,
                    Status = string.IsNullOrWhiteSpace(transaction.Status) ? "Executed" : transaction.Status,
                    PredictionSource = transaction.Notes ?? string.Empty,
                    Timestamp = transaction.ExecutionTime
                };

                await _context.OrderHistory.AddAsync(entity);
                await _context.SaveChangesAsync();
            }
            catch (Exception)
            {
                throw;
            }
        }

        public void DeleteTransaction(int id)
        {
            ResilienceHelper.Retry(() =>
            {
                try
                {
                    var entity = _context.OrderHistory.Find(id);
                    if (entity != null)
                    {
                        _context.OrderHistory.Remove(entity);
                        _context.SaveChanges();
                    }
                }
                catch (Exception)
                {
                    throw;
                }
            }, RetryOptions.ForCriticalOperation());
        }

        public async Task DeleteTransactionAsync(int id)
        {
            try
            {
                var entity = await _context.OrderHistory.FindAsync(id);
                if (entity != null)
                {
                    _context.OrderHistory.Remove(entity);
                    await _context.SaveChangesAsync();
                }
            }
            catch (Exception)
            {
                throw;
            }
        }
    }
}
