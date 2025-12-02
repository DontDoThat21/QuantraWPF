using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing backtest results persistence
    /// </summary>
    public class BacktestResultService : IBacktestResultService
    {
        private readonly LoggingService _loggingService;

        public BacktestResultService(LoggingService loggingService)
        {
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Creates a new QuantraDbContext with SQL Server connection
        /// </summary>
        private QuantraDbContext CreateDbContext()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            return new QuantraDbContext(optionsBuilder.Options);
        }

        /// <inheritdoc/>
        public async Task<BacktestResultEntity> SaveResultAsync(BacktestResultEntity result)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            try
            {
                using (var dbContext = CreateDbContext())
                {
                    // Sanitize numeric values to prevent SQL Server errors
                    SanitizeNumericValues(result);
                    
                    result.CreatedAt = DateTime.Now;
                    dbContext.BacktestResults.Add(result);
                    await dbContext.SaveChangesAsync();

                    _loggingService.Log("Info", $"Saved backtest result for {result.Symbol} using {result.StrategyName}");
                    return result;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to save backtest result: {ex.Message}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Sanitizes numeric values to prevent SQL Server errors with NaN, Infinity, or out-of-range values
        /// </summary>
        private void SanitizeNumericValues(BacktestResultEntity result)
        {
            // Replace invalid values with 0
            result.InitialCapital = SanitizeDouble(result.InitialCapital);
            result.FinalEquity = SanitizeDouble(result.FinalEquity);
            result.TotalReturn = SanitizeDouble(result.TotalReturn);
            result.MaxDrawdown = SanitizeDouble(result.MaxDrawdown);
            result.WinRate = SanitizeDouble(result.WinRate);
            result.SharpeRatio = SanitizeDouble(result.SharpeRatio);
            result.SortinoRatio = SanitizeDouble(result.SortinoRatio);
            result.CAGR = SanitizeDouble(result.CAGR);
            result.CalmarRatio = SanitizeDouble(result.CalmarRatio);
            
            // ProfitFactor needs special handling - cap at reasonable maximum
            result.ProfitFactor = SanitizeProfitFactor(result.ProfitFactor);
            
            result.InformationRatio = SanitizeDouble(result.InformationRatio);
            result.TotalTransactionCosts = SanitizeDouble(result.TotalTransactionCosts);
            result.GrossReturn = SanitizeDouble(result.GrossReturn);
        }

        /// <summary>
        /// Sanitizes a double value, replacing NaN, Infinity, or extreme values with 0
        /// </summary>
        private double SanitizeDouble(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return 0;
            }
            
            // Cap extremely large values that might cause issues
            const double maxValue = 1e10; // 10 billion
            if (Math.Abs(value) > maxValue)
            {
                return 0;
            }
            
            return value;
        }

        /// <summary>
        /// Sanitizes profit factor, capping at 999 for cases with no losses
        /// </summary>
        private double SanitizeProfitFactor(double profitFactor)
        {
            if (double.IsNaN(profitFactor) || double.IsInfinity(profitFactor))
            {
                return 0;
            }
            
            // Cap profit factor at 999 (extremely high but represents "perfect" trading)
            if (profitFactor > 999)
            {
                return 999;
            }
            
            return profitFactor;
        }

        /// <inheritdoc/>
        public async Task<BacktestResultEntity> GetByIdAsync(int id)
        {
            try
            {
                using (var dbContext = CreateDbContext())
                {
                    return await dbContext.BacktestResults
                        .FirstOrDefaultAsync(b => b.Id == id);
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get backtest result by ID {id}: {ex.Message}", ex.ToString());
                return null;
            }
        }

        /// <inheritdoc/>
        public async Task<List<BacktestResultEntity>> GetBySymbolAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return new List<BacktestResultEntity>();
            }

            try
            {
                using (var dbContext = CreateDbContext())
                {
                    return await dbContext.BacktestResults
                        .Where(b => b.Symbol == symbol)
                        .OrderByDescending(b => b.CreatedAt)
                        .ToListAsync();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get backtest results for {symbol}: {ex.Message}", ex.ToString());
                return new List<BacktestResultEntity>();
            }
        }

        /// <inheritdoc/>
        public async Task<List<BacktestResultEntity>> GetByStrategyAsync(string strategyName)
        {
            if (string.IsNullOrWhiteSpace(strategyName))
            {
                return new List<BacktestResultEntity>();
            }

            try
            {
                using (var dbContext = CreateDbContext())
                {
                    return await dbContext.BacktestResults
                        .Where(b => b.StrategyName == strategyName)
                        .OrderByDescending(b => b.CreatedAt)
                        .ToListAsync();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get backtest results for strategy {strategyName}: {ex.Message}", ex.ToString());
                return new List<BacktestResultEntity>();
            }
        }

        /// <inheritdoc/>
        public async Task<List<BacktestResultEntity>> GetByDateRangeAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                using (var dbContext = CreateDbContext())
                {
                    return await dbContext.BacktestResults
                        .Where(b => b.CreatedAt >= startDate && b.CreatedAt <= endDate)
                        .OrderByDescending(b => b.CreatedAt)
                        .ToListAsync();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get backtest results for date range: {ex.Message}", ex.ToString());
                return new List<BacktestResultEntity>();
            }
        }

        /// <inheritdoc/>
        public async Task<List<BacktestResultEntity>> GetRecentAsync(int count = 10)
        {
            try
            {
                using (var dbContext = CreateDbContext())
                {
                    return await dbContext.BacktestResults
                        .OrderByDescending(b => b.CreatedAt)
                        .Take(count)
                        .ToListAsync();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get recent backtest results: {ex.Message}", ex.ToString());
                return new List<BacktestResultEntity>();
            }
        }

        /// <inheritdoc/>
        public async Task<List<BacktestResultEntity>> GetAllAsync()
        {
            try
            {
                using (var dbContext = CreateDbContext())
                {
                    return await dbContext.BacktestResults
                        .OrderByDescending(b => b.CreatedAt)
                        .ToListAsync();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get all backtest results: {ex.Message}", ex.ToString());
                return new List<BacktestResultEntity>();
            }
        }

        /// <inheritdoc/>
        public async Task<bool> DeleteAsync(int id)
        {
            try
            {
                using (var dbContext = CreateDbContext())
                {
                    var result = await dbContext.BacktestResults.FindAsync(id);
                    if (result == null)
                    {
                        return false;
                    }

                    dbContext.BacktestResults.Remove(result);
                    await dbContext.SaveChangesAsync();

                    _loggingService.Log("Info", $"Deleted backtest result with ID {id}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to delete backtest result {id}: {ex.Message}", ex.ToString());
                return false;
            }
        }

        /// <inheritdoc/>
        public async Task<int> DeleteBySymbolAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return 0;
            }

            try
            {
                using (var dbContext = CreateDbContext())
                {
                    var results = await dbContext.BacktestResults
                        .Where(b => b.Symbol == symbol)
                        .ToListAsync();

                    int count = results.Count;
                    dbContext.BacktestResults.RemoveRange(results);
                    await dbContext.SaveChangesAsync();

                    _loggingService.Log("Info", $"Deleted {count} backtest results for {symbol}");
                    return count;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to delete backtest results for {symbol}: {ex.Message}", ex.ToString());
                return 0;
            }
        }

        /// <inheritdoc/>
        public async Task<BacktestResultEntity> UpdateAsync(BacktestResultEntity result)
        {
            if (result == null)
            {
                throw new ArgumentNullException(nameof(result));
            }

            try
            {
                using (var dbContext = CreateDbContext())
                {
                    dbContext.BacktestResults.Update(result);
                    await dbContext.SaveChangesAsync();

                    _loggingService.Log("Info", $"Updated backtest result with ID {result.Id}");
                    return result;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to update backtest result: {ex.Message}", ex.ToString());
                throw;
            }
        }

        /// <inheritdoc/>
        public BacktestResultEntity ConvertFromEngineResult(
            BacktestingEngine.BacktestResult engineResult,
            string strategyName,
            double initialCapital,
            string runName = null,
            string notes = null)
        {
            if (engineResult == null)
            {
                throw new ArgumentNullException(nameof(engineResult));
            }

            // Serialize equity curve data
            string equityCurveJson = null;
            if (engineResult.EquityCurve != null && engineResult.EquityCurve.Count > 0)
            {
                var equityCurveData = engineResult.EquityCurve.Select(e => new
                {
                    Date = e.Date,
                    Equity = e.Equity
                }).ToList();
                equityCurveJson = JsonConvert.SerializeObject(equityCurveData);
            }

            // Serialize trades data
            string tradesJson = null;
            if (engineResult.Trades != null && engineResult.Trades.Count > 0)
            {
                var tradesData = engineResult.Trades.Select(t => new
                {
                    EntryDate = t.EntryDate,
                    EntryPrice = t.EntryPrice,
                    ExitDate = t.ExitDate,
                    ExitPrice = t.ExitPrice,
                    Action = t.Action,
                    Quantity = t.Quantity,
                    EntryCosts = t.EntryCosts,
                    ExitCosts = t.ExitCosts,
                    ProfitLoss = t.ProfitLoss
                }).ToList();
                tradesJson = JsonConvert.SerializeObject(tradesData);
            }

            var result = new BacktestResultEntity
            {
                Symbol = engineResult.Symbol,
                StrategyName = strategyName,
                TimeFrame = engineResult.TimeFrame,
                StartDate = engineResult.StartDate,
                EndDate = engineResult.EndDate,
                InitialCapital = initialCapital,
                FinalEquity = engineResult.EquityCurve?.LastOrDefault()?.Equity ?? initialCapital,
                TotalReturn = engineResult.TotalReturn,
                MaxDrawdown = engineResult.MaxDrawdown,
                WinRate = engineResult.WinRate,
                TotalTrades = engineResult.TotalTrades,
                WinningTrades = engineResult.WinningTrades,
                SharpeRatio = engineResult.SharpeRatio,
                SortinoRatio = engineResult.SortinoRatio,
                CAGR = engineResult.CAGR,
                CalmarRatio = engineResult.CalmarRatio,
                ProfitFactor = engineResult.ProfitFactor,
                InformationRatio = engineResult.InformationRatio,
                TotalTransactionCosts = engineResult.TotalTransactionCosts,
                GrossReturn = engineResult.GrossReturn,
                EquityCurveJson = equityCurveJson,
                TradesJson = tradesJson,
                RunName = runName,
                Notes = notes,
                CreatedAt = DateTime.Now
            };

            // Sanitize numeric values before returning
            SanitizeNumericValues(result);
            
            return result;
        }
    }
}
