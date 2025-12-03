using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for managing trading signals
    /// </summary>
    public interface ITradingSignalService
    {
        /// <summary>
        /// Gets all trading signals
        /// </summary>
        Task<List<TradingSignal>> GetAllSignalsAsync();

        /// <summary>
        /// Gets a trading signal by ID
        /// </summary>
        Task<TradingSignal> GetSignalByIdAsync(int signalId);

        /// <summary>
        /// Gets all enabled trading signals
        /// </summary>
        Task<List<TradingSignal>> GetEnabledSignalsAsync();

        /// <summary>
        /// Saves a trading signal (creates new or updates existing)
        /// </summary>
        Task<bool> SaveSignalAsync(TradingSignal signal);

        /// <summary>
        /// Deletes a trading signal by ID
        /// </summary>
        Task<bool> DeleteSignalAsync(int signalId);

        /// <summary>
        /// Validates a stock symbol against the Alpha Vantage API
        /// </summary>
        Task<(bool IsValid, string Message)> ValidateSymbolAsync(string symbol);

        /// <summary>
        /// Evaluates a trading signal against current market data
        /// </summary>
        Task<bool> EvaluateSignalAsync(TradingSignal signal);
    }
}
