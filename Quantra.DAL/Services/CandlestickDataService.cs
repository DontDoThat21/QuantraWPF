using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Implementation of ICandlestickDataService that wraps AlphaVantageService
    /// and StockDataCacheService to provide candlestick data.
    /// </summary>
    public class CandlestickDataService : ICandlestickDataService
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly StockDataCacheService _stockDataCacheService;
        private readonly LoggingService _loggingService;

        public CandlestickDataService(
            AlphaVantageService alphaVantageService,
            StockDataCacheService stockDataCacheService,
            LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _stockDataCacheService = stockDataCacheService ?? throw new ArgumentNullException(nameof(stockDataCacheService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
        }

        /// <summary>
        /// Gets candlestick data for a specific symbol and interval
        /// </summary>
        public async Task<List<HistoricalPrice>> GetCandlestickDataAsync(
            string symbol,
            string interval,
            bool forceRefresh = false,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
            }

            if (string.IsNullOrWhiteSpace(interval))
            {
                throw new ArgumentException("Interval cannot be null or empty", nameof(interval));
            }

            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                // Convert interval format for StockDataCacheService (1min, 5min, etc.)
                string timeRange = ConvertIntervalToTimeRange(interval);

                // Get data from cache/API
                var data = await _stockDataCacheService.GetStockData(
                    symbol,
                    timeRange,
                    interval,
                    forceRefresh);

                if (data == null || data.Count == 0)
                {
                    _loggingService?.Log("Warning", $"No candlestick data available for {symbol} with interval {interval}");
                    return new List<HistoricalPrice>();
                }

                return data;
            }
            catch (OperationCanceledException)
            {
                _loggingService?.Log("Info", $"Candlestick data request cancelled for {symbol}");
                throw;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error fetching candlestick data for {symbol} with interval {interval}");
                throw;
            }
        }

        /// <summary>
        /// Gets the current API usage count for rate limiting
        /// </summary>
        public int GetApiUsageCount()
        {
            return _alphaVantageService?.GetAlphaVantageApiUsageCount(DateTime.UtcNow) ?? 0;
        }

        /// <summary>
        /// Logs an API usage event
        /// </summary>
        public void LogApiUsage()
        {
            _alphaVantageService?.LogApiUsage();
        }

        /// <summary>
        /// Converts interval format to time range for StockDataCacheService
        /// </summary>
        private string ConvertIntervalToTimeRange(string interval)
        {
            return interval switch
            {
                "1min" => "1d",      // 1-minute data: last 1 day
                "5min" => "5d",      // 5-minute data: last 5 days
                "15min" => "1mo",    // 15-minute data: last month
                "30min" => "1mo",    // 30-minute data: last month
                "60min" => "2mo",    // 60-minute data: last 2 months
                _ => "1mo"           // Default: 1 month
            };
        }
    }
}
