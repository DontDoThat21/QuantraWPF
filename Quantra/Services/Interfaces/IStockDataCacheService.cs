using Quantra.Models;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Quantra.Services.Interfaces
{
    /// <summary>
    /// Service interface for caching and retrieving stock data
    /// </summary>
    public interface IStockDataCacheService
    {
        /// <summary>
        /// Gets stock data for a symbol, with optional timeframe and interval
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="timeframe">Timeframe (e.g., "1mo", "6mo", "1y", "5y")</param>
        /// <param name="interval">Interval (e.g., "1d", "1h", "15min")</param>
        /// <param name="forceRefresh">Force refresh from source instead of using cache</param>
        /// <returns>List of historical prices</returns>
        Task<List<HistoricalPrice>> GetStockDataAsync(string symbol, string timeframe = "1mo", string interval = "1d", bool forceRefresh = false);
        
        /// <summary>
        /// Gets the latest quote data for a symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="forceRefresh">Force refresh from source instead of using cache</param>
        /// <returns>Quote data for the symbol</returns>
        Task<QuoteData> GetQuoteDataAsync(string symbol, bool forceRefresh = false);
        
        /// <summary>
        /// Gets historical indicator data for a symbol and indicator type
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="indicatorType">Type of indicator (e.g., "RSI", "MACD", "BB")</param>
        /// <param name="timeframe">Timeframe (e.g., "1mo", "6mo", "1y", "5y")</param>
        /// <param name="interval">Interval (e.g., "1d", "1h", "15min")</param>
        /// <param name="forceRefresh">Force refresh from source instead of using cache</param>
        /// <returns>Dictionary with indicator values</returns>
        Task<Dictionary<string, List<double>>> GetIndicatorDataAsync(
            string symbol, 
            string indicatorType, 
            string timeframe = "1mo", 
            string interval = "1d", 
            bool forceRefresh = false);
            
        /// <summary>
        /// Clears the cache for a specific symbol
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <returns>True if successful</returns>
        Task<bool> ClearCacheForSymbolAsync(string symbol);
        
        /// <summary>
        /// Clears all cached data
        /// </summary>
        /// <returns>True if successful</returns>
        Task<bool> ClearAllCacheAsync();
    }
}