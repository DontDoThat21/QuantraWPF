using Quantra.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Data;

namespace Quantra.DAL.Services
{
    public class TransactionService : ITransactionService
    {
        private readonly QuantraDbContext _context;

        public TransactionService(QuantraDbContext context)
        {
            _context = context;
        }

        // Parameterless constructor for backward compatibility
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
                    // Use Entity Framework to get orders from the OrderHistory table
                    return GetOrdersFromDatabase();
                }
                catch (Exception ex)
                {
                    // Log the error
                    //DatabaseMonolith.Log("Error", "Failed to retrieve transactions from database", ex.ToString());

                    // Return empty list in case of error
                    return new List<TransactionModel>();
                }
            }, RetryOptions.ForUserFacingOperation());
        }

        private List<TransactionModel> GetOrdersFromDatabase()
        {
            // Ensure database is created
            _context.Database.EnsureCreated();

            // Query OrderHistory using Entity Framework
            var orderEntities = _context.OrderHistory
                .AsNoTracking()
                .OrderByDescending(o => o.Timestamp)
                .ToList();

            // Map entities to TransactionModel
            var transactions = orderEntities.Select(order => new TransactionModel
            {
                Symbol = order.Symbol,
                TransactionType = order.OrderType,
                Quantity = order.Quantity,
                ExecutionPrice = order.Price,
                TotalValue = order.Price * order.Quantity,
                ExecutionTime = order.Timestamp,
                IsPaperTrade = order.IsPaperTrade,
                Fees = 0.0, // Default for now, could be updated later
                RealizedPnL = 0.0, // Default for now, could be calculated
                RealizedPnLPercentage = 0.0, // Default for now
                Notes = order.PredictionSource ?? string.Empty,
                OrderSource = string.IsNullOrEmpty(order.PredictionSource) ? "Manual" : "Automated",
                Status = order.Status
            }).ToList();

            //DatabaseMonolith.Log("Info", $"Retrieved {transactions.Count} transactions from database");
            return transactions;
        }

        // Sample data method preserved for reference or testing
        private List<TransactionModel> GetSampleTransactions()
        {
            var transactions = new List<TransactionModel>();
            //
            //// Generate sample data
            //transactions.Add(new TransactionModel
            //{
            //    Symbol = "AAPL",
            //    TransactionType = "BUY",
            //    Quantity = 100,
            //    ExecutionPrice = 182.50,
            //    TotalValue = 18250.00,
            //    ExecutionTime = DateTime.Now.AddDays(-30),
            //    IsPaperTrade = false,
            //    Fees = 4.95,
            //    RealizedPnL = 650.50,
            //    RealizedPnLPercentage = 0.0356,
            //    Notes = "Quarterly earnings expectation"
            //});
            // ... other sample transactions
            return transactions;
        }

        public TransactionModel GetTransaction(int id)
        {
            try
            {
                var entity = _context.OrderHistory
                    .AsNoTracking()
                    .FirstOrDefault(o => o.Id == id);

                if (entity == null)
                    return null;

                return new TransactionModel
                {
                    Symbol = entity.Symbol,
                    TransactionType = entity.OrderType,
                    Quantity = entity.Quantity,
                    ExecutionPrice = entity.Price,
                    TotalValue = entity.Price * entity.Quantity,
                    ExecutionTime = entity.Timestamp,
                    IsPaperTrade = entity.IsPaperTrade,
                    Fees = 0.0,
                    RealizedPnL = 0.0,
                    RealizedPnLPercentage = 0.0,
                    Notes = entity.PredictionSource ?? string.Empty,
                    Status = entity.Status
                };
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get transaction with ID {id}", ex.ToString());
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

                return entities.Select(entity => new TransactionModel
                {
                    Symbol = entity.Symbol,
                    TransactionType = entity.OrderType,
                    Quantity = entity.Quantity,
                    ExecutionPrice = entity.Price,
                    TotalValue = entity.Price * entity.Quantity,
                    ExecutionTime = entity.Timestamp,
                    IsPaperTrade = entity.IsPaperTrade,
                    Fees = 0.0,
                    RealizedPnL = 0.0,
                    RealizedPnLPercentage = 0.0,
                    Notes = entity.PredictionSource ?? string.Empty,
                    Status = entity.Status
                }).ToList();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to get transactions by date range", ex.ToString());
                throw;
            }
        }

        public List<TransactionModel> GetTransactionsBySymbol(string symbol)
        {
            try
            {
                var entities = _context.OrderHistory
                    .AsNoTracking()
                    .Where(o => o.Symbol == symbol)
                    .OrderByDescending(o => o.Timestamp)
                    .ToList();

                return entities.Select(entity => new TransactionModel
                {
                    Symbol = entity.Symbol,
                    TransactionType = entity.OrderType,
                    Quantity = entity.Quantity,
                    ExecutionPrice = entity.Price,
                    TotalValue = entity.Price * entity.Quantity,
                    ExecutionTime = entity.Timestamp,
                    IsPaperTrade = entity.IsPaperTrade,
                    Fees = 0.0,
                    RealizedPnL = 0.0,
                    RealizedPnLPercentage = 0.0,
                    Notes = entity.PredictionSource ?? string.Empty,
                    Status = entity.Status
                }).ToList();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to get transactions for symbol {symbol}", ex.ToString());
                throw;
            }
        }

        public void SaveTransaction(TransactionModel transaction)
        {
            ResilienceHelper.Retry(() =>
            {
                try
                {
                    var entity = new Data.Entities.OrderHistoryEntity
                    {
                        Symbol = transaction.Symbol,
                        OrderType = transaction.TransactionType,
                        Quantity = transaction.Quantity,
                        Price = transaction.ExecutionPrice,
                        StopLoss = null, // Default value, can be updated
                        TakeProfit = null, // Default value, can be updated
                        IsPaperTrade = transaction.IsPaperTrade,
                        Status = "Executed",
                        PredictionSource = transaction.Notes ?? string.Empty,
                        Timestamp = transaction.ExecutionTime
                    };

                    _context.OrderHistory.Add(entity);
                    _context.SaveChanges();

                    //DatabaseMonolith.Log("Info", $"Saved transaction for {transaction.Symbol} ({transaction.TransactionType})");
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", "Failed to save transaction", ex.ToString());
                    throw;
                }
            }, RetryOptions.ForCriticalOperation());
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

                    //DatabaseMonolith.Log("Info", $"Deleted transaction with ID {id}");
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Error", $"Failed to delete transaction with ID {id}", ex.ToString());
                    throw;
                }
            }, RetryOptions.ForCriticalOperation());
        }
    }
}
