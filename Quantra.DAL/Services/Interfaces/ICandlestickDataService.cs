using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Service interface for retrieving candlestick chart data.
    /// Abstracts the data source for improved testability and maintainability.
    /// </summary>
    public interface ICandlestickDataService
    {
        /// <summary>
        /// Gets candlestick data for a specific symbol and interval
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Time interval (e.g., "1min", "5min", "15min")</param>
        /// <param name="forceRefresh">Force refresh from API instead of using cache</param>
        /// <param name="cancellationToken">Cancellation token for async operation</param>
        /// <returns>List of historical price data</returns>
        Task<List<HistoricalPrice>> GetCandlestickDataAsync(
            string symbol, 
            string interval, 
            bool forceRefresh = false,
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Gets the current API usage count for rate limiting
        /// </summary>
        /// <returns>Number of API calls made</returns>
        int GetApiUsageCount();

        /// <summary>
        /// Logs an API usage event
        /// </summary>
        void LogApiUsage();
    }
}
