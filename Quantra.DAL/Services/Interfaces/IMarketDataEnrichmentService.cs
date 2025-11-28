using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for enriching market data with historical context
    /// Used by Market Chat to provide AI with context-aware responses
    /// </summary>
    public interface IMarketDataEnrichmentService
    {
        /// <summary>
        /// Gets historical context for a symbol including price trends and statistics
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="days">Number of days of historical data to include</param>
        /// <returns>Formatted historical context string for AI prompts</returns>
        Task<string> GetHistoricalContextAsync(string symbol, int days = 60);

        /// <summary>
        /// Gets a brief summary of historical context for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>Brief historical context summary</returns>
        Task<string> GetHistoricalContextSummaryAsync(string symbol);

        /// <summary>
        /// Clears the historical context cache for a specific symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        void ClearCacheForSymbol(string symbol);

        /// <summary>
        /// Clears all cached historical context data
        /// </summary>
        void ClearAllCache();
    }
}
