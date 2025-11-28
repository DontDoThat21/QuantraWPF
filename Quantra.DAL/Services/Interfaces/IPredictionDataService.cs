using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.DAL.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for querying ML prediction data from the database
    /// Used by Market Chat to provide AI-generated forecast context in conversations
    /// </summary>
    public interface IPredictionDataService
    {
        /// <summary>
        /// Gets prediction context for a specific symbol to include in Market Chat conversations
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL", "MSFT")</param>
        /// <returns>Formatted prediction context string for AI prompts, or null if no predictions exist</returns>
        Task<string> GetPredictionContextAsync(string symbol);

        /// <summary>
        /// Gets prediction context with cache metadata for a specific symbol.
        /// This method first checks the PredictionCache table before querying fresh predictions.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., "AAPL", "MSFT")</param>
        /// <returns>PredictionContextResult containing the context and cache metadata</returns>
        Task<PredictionContextResult> GetPredictionContextWithCacheAsync(string symbol);

        /// <summary>
        /// Warms the prediction cache for popular symbols during market hours.
        /// This pre-populates the cache to improve response times for frequently requested symbols.
        /// </summary>
        /// <param name="symbols">List of symbols to warm the cache for</param>
        /// <returns>Number of symbols successfully warmed</returns>
        Task<int> WarmCacheForSymbolsAsync(IEnumerable<string> symbols);
    }
}
