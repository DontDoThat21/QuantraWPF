using System.Collections.Generic;
using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Interface for sentiment shift alert service
    /// </summary>
    public interface ISentimentShiftAlertService
    {
        /// <summary>
        /// Monitors sentiment for a specific symbol and generates alerts for significant shifts
        /// </summary>
        /// <param name="symbol">The stock symbol to monitor</param>
        /// <param name="sources">Specific sentiment sources to monitor (null for all)</param>
        Task MonitorSentimentShiftsAsync(string symbol, List<string> sources = null);
        
        /// <summary>
        /// Monitors sentiment shifts for a watchlist of symbols
        /// </summary>
        /// <param name="symbols">List of symbols to monitor</param>
        Task MonitorWatchlistAsync(List<string> symbols);
    }
}