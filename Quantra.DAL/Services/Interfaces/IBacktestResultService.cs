using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for managing backtest results persistence
    /// </summary>
    public interface IBacktestResultService
    {
        /// <summary>
        /// Saves a backtest result to the database
        /// </summary>
        /// <param name="result">The backtest result entity to save</param>
        /// <returns>The saved entity with its generated ID</returns>
        Task<BacktestResultEntity> SaveResultAsync(BacktestResultEntity result);

        /// <summary>
        /// Gets a backtest result by its ID
        /// </summary>
        /// <param name="id">The ID of the result to retrieve</param>
        /// <returns>The backtest result or null if not found</returns>
        Task<BacktestResultEntity> GetByIdAsync(int id);

        /// <summary>
        /// Gets all backtest results for a specific symbol
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <returns>List of backtest results for the symbol</returns>
        Task<List<BacktestResultEntity>> GetBySymbolAsync(string symbol);

        /// <summary>
        /// Gets all backtest results for a specific strategy
        /// </summary>
        /// <param name="strategyName">The strategy name</param>
        /// <returns>List of backtest results for the strategy</returns>
        Task<List<BacktestResultEntity>> GetByStrategyAsync(string strategyName);

        /// <summary>
        /// Gets all backtest results within a date range
        /// </summary>
        /// <param name="startDate">Start date for filtering</param>
        /// <param name="endDate">End date for filtering</param>
        /// <returns>List of backtest results within the date range</returns>
        Task<List<BacktestResultEntity>> GetByDateRangeAsync(DateTime startDate, DateTime endDate);

        /// <summary>
        /// Gets the most recent backtest results
        /// </summary>
        /// <param name="count">Number of results to retrieve</param>
        /// <returns>List of most recent backtest results</returns>
        Task<List<BacktestResultEntity>> GetRecentAsync(int count = 10);

        /// <summary>
        /// Gets all backtest results
        /// </summary>
        /// <returns>List of all backtest results</returns>
        Task<List<BacktestResultEntity>> GetAllAsync();

        /// <summary>
        /// Deletes a backtest result by its ID
        /// </summary>
        /// <param name="id">The ID of the result to delete</param>
        /// <returns>True if deleted successfully, false otherwise</returns>
        Task<bool> DeleteAsync(int id);

        /// <summary>
        /// Deletes all backtest results for a specific symbol
        /// </summary>
        /// <param name="symbol">The stock symbol</param>
        /// <returns>Number of results deleted</returns>
        Task<int> DeleteBySymbolAsync(string symbol);

        /// <summary>
        /// Updates an existing backtest result
        /// </summary>
        /// <param name="result">The backtest result entity to update</param>
        /// <returns>The updated entity</returns>
        Task<BacktestResultEntity> UpdateAsync(BacktestResultEntity result);

        /// <summary>
        /// Converts a BacktestingEngine.BacktestResult to a BacktestResultEntity
        /// </summary>
        /// <param name="engineResult">The backtest result from the engine</param>
        /// <param name="strategyName">Name of the strategy used</param>
        /// <param name="initialCapital">Initial capital for the backtest</param>
        /// <param name="runName">Optional name for this backtest run</param>
        /// <param name="notes">Optional notes about the backtest</param>
        /// <returns>A BacktestResultEntity ready for persistence</returns>
        BacktestResultEntity ConvertFromEngineResult(
            BacktestingEngine.BacktestResult engineResult,
            string strategyName,
            double initialCapital,
            string runName = null,
            string notes = null);
    }
}
