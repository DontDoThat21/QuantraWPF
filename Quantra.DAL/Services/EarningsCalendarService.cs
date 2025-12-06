using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json.Linq;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to fetch and manage earnings calendar data for TFT model known future inputs.
    /// Uses Alpha Vantage EARNINGS_CALENDAR endpoint with database caching.
    /// </summary>
    public class EarningsCalendarService
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        
        // In-memory cache for earnings data with 24-hour expiry
        private static readonly Dictionary<string, (List<EarningsCalendarEntity> Data, DateTime Timestamp)> _earningsCache 
            = new Dictionary<string, (List<EarningsCalendarEntity>, DateTime)>();
        private static readonly object _cacheLock = new object();
        private const int CacheExpirationHours = 24;

        public EarningsCalendarService(AlphaVantageService alphaVantageService, LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _loggingService = loggingService;
        }

        /// <summary>
        /// Get next earnings date for a symbol.
        /// Uses Alpha Vantage EARNINGS_CALENDAR endpoint with database caching.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., AAPL)</param>
        /// <returns>Next earnings date or null if unknown</returns>
        public async Task<DateTime?> GetNextEarningsDateAsync(string symbol)
        {
            try
            {
                var earningsData = await GetEarningsDataAsync(symbol);
                
                DateTime now = DateTime.UtcNow.Date;
                
                // Find the next future earnings date
                var nextEarnings = earningsData
                    .Where(e => e.EarningsDate >= now)
                    .OrderBy(e => e.EarningsDate)
                    .FirstOrDefault();

                return nextEarnings?.EarningsDate;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get next earnings date for {symbol}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Get last earnings date for a symbol.
        /// Returns the most recent earnings date in the past.
        /// </summary>
        /// <param name="symbol">Stock symbol (e.g., AAPL)</param>
        /// <returns>Last earnings date or null if unknown</returns>
        public async Task<DateTime?> GetLastEarningsDateAsync(string symbol)
        {
            try
            {
                var earningsData = await GetEarningsDataAsync(symbol);
                
                DateTime now = DateTime.UtcNow.Date;
                
                // Find the most recent past earnings date
                var lastEarnings = earningsData
                    .Where(e => e.EarningsDate < now)
                    .OrderByDescending(e => e.EarningsDate)
                    .FirstOrDefault();

                return lastEarnings?.EarningsDate;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get last earnings date for {symbol}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Calculate trading days until next earnings (excluding weekends).
        /// </summary>
        /// <param name="nextEarningsDate">The next earnings date</param>
        /// <returns>Number of trading days until earnings</returns>
        public int GetTradingDaysToEarnings(DateTime nextEarningsDate)
        {
            DateTime startDate = DateTime.UtcNow.Date;
            DateTime endDate = nextEarningsDate.Date;

            if (endDate <= startDate)
            {
                return 0;
            }

            int tradingDays = 0;
            DateTime current = startDate;

            while (current < endDate)
            {
                current = current.AddDays(1);
                // Count weekdays only (Monday-Friday)
                if (current.DayOfWeek != DayOfWeek.Saturday && current.DayOfWeek != DayOfWeek.Sunday)
                {
                    tradingDays++;
                }
            }

            return tradingDays;
        }

        /// <summary>
        /// Calculate trading days since last earnings (excluding weekends).
        /// </summary>
        /// <param name="lastEarningsDate">The last earnings date</param>
        /// <returns>Number of trading days since earnings</returns>
        public int GetTradingDaysSinceEarnings(DateTime lastEarningsDate)
        {
            DateTime startDate = lastEarningsDate.Date;
            DateTime endDate = DateTime.UtcNow.Date;

            if (startDate >= endDate)
            {
                return 0;
            }

            int tradingDays = 0;
            DateTime current = startDate;

            while (current < endDate)
            {
                current = current.AddDays(1);
                // Count weekdays only (Monday-Friday)
                if (current.DayOfWeek != DayOfWeek.Saturday && current.DayOfWeek != DayOfWeek.Sunday)
                {
                    tradingDays++;
                }
            }

            return tradingDays;
        }

        /// <summary>
        /// Get earnings data for a symbol, using cache first then API.
        /// </summary>
        private async Task<List<EarningsCalendarEntity>> GetEarningsDataAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return new List<EarningsCalendarEntity>();
            }

            symbol = symbol.ToUpperInvariant();

            // Check in-memory cache first
            lock (_cacheLock)
            {
                if (_earningsCache.TryGetValue(symbol, out var cacheEntry))
                {
                    if (DateTime.UtcNow - cacheEntry.Timestamp < TimeSpan.FromHours(CacheExpirationHours))
                    {
                        _loggingService?.Log("Debug", $"Using cached earnings data for {symbol}");
                        return cacheEntry.Data;
                    }
                }
            }

            // Try to get from database first
            var dbData = await GetFromDatabaseAsync(symbol);
            if (dbData != null && dbData.Count > 0)
            {
                // Check if database cache is fresh enough (within 24 hours)
                var mostRecent = dbData.Max(e => e.LastUpdated);
                if (DateTime.UtcNow - mostRecent < TimeSpan.FromHours(CacheExpirationHours))
                {
                    UpdateMemoryCache(symbol, dbData);
                    return dbData;
                }
            }

            // Fetch from Alpha Vantage API
            var apiData = await FetchFromAlphaVantageAsync(symbol);
            
            if (apiData != null && apiData.Count > 0)
            {
                // Save to database
                await SaveToDatabaseAsync(apiData);
                
                // Update memory cache
                UpdateMemoryCache(symbol, apiData);
                
                return apiData;
            }

            // Return any existing data as fallback
            if (dbData != null && dbData.Count > 0)
            {
                return dbData;
            }

            return new List<EarningsCalendarEntity>();
        }

        /// <summary>
        /// Fetch earnings calendar data from Alpha Vantage API.
        /// </summary>
        private async Task<List<EarningsCalendarEntity>> FetchFromAlphaVantageAsync(string symbol)
        {
            try
            {
                _loggingService?.Log("Info", $"Fetching earnings calendar for {symbol} from Alpha Vantage");

                // Use AlphaVantageService's HttpClient and API key through a method
                // Alpha Vantage EARNINGS endpoint: https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={key}
                var earningsData = await _alphaVantageService.GetEarningsCalendarAsync(symbol);
                
                if (earningsData == null || earningsData.Count == 0)
                {
                    _loggingService?.Log("Warning", $"No earnings calendar data returned for {symbol}");
                    return new List<EarningsCalendarEntity>();
                }

                var entities = earningsData.Select(e => new EarningsCalendarEntity
                {
                    Symbol = symbol,
                    EarningsDate = e.ReportDate,
                    FiscalQuarter = e.FiscalQuarter,
                    EPSEstimate = e.EstimatedEPS,
                    LastUpdated = DateTime.UtcNow
                }).ToList();

                _loggingService?.Log("Info", $"Fetched {entities.Count} earnings dates for {symbol}");
                
                return entities;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to fetch earnings calendar from Alpha Vantage for {symbol}: {ex.Message}");
                return new List<EarningsCalendarEntity>();
            }
        }

        /// <summary>
        /// Get earnings data from database.
        /// </summary>
        private async Task<List<EarningsCalendarEntity>> GetFromDatabaseAsync(string symbol)
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                
                using (var context = new QuantraDbContext(optionsBuilder.Options))
                {
                    return await context.EarningsCalendar
                        .Where(e => e.Symbol == symbol)
                        .OrderBy(e => e.EarningsDate)
                        .ToListAsync();
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get earnings data from database for {symbol}: {ex.Message}");
                return new List<EarningsCalendarEntity>();
            }
        }

        /// <summary>
        /// Save earnings data to database.
        /// </summary>
        private async Task SaveToDatabaseAsync(List<EarningsCalendarEntity> earningsData)
        {
            if (earningsData == null || earningsData.Count == 0)
            {
                return;
            }

            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
                
                using (var context = new QuantraDbContext(optionsBuilder.Options))
                {
                    string symbol = earningsData.First().Symbol;
                    
                    // Remove existing data for this symbol
                    var existingData = await context.EarningsCalendar
                        .Where(e => e.Symbol == symbol)
                        .ToListAsync();
                    
                    if (existingData.Count > 0)
                    {
                        context.EarningsCalendar.RemoveRange(existingData);
                    }

                    // Add new data
                    await context.EarningsCalendar.AddRangeAsync(earningsData);
                    await context.SaveChangesAsync();
                    
                    _loggingService?.Log("Debug", $"Saved {earningsData.Count} earnings dates to database for {symbol}");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to save earnings data to database: {ex.Message}");
            }
        }

        /// <summary>
        /// Update the in-memory cache with new data.
        /// </summary>
        private void UpdateMemoryCache(string symbol, List<EarningsCalendarEntity> data)
        {
            lock (_cacheLock)
            {
                _earningsCache[symbol] = (data, DateTime.UtcNow);
            }
        }

        /// <summary>
        /// Clear the in-memory cache for a specific symbol or all symbols.
        /// </summary>
        /// <param name="symbol">Symbol to clear, or null to clear all</param>
        public void ClearCache(string symbol = null)
        {
            lock (_cacheLock)
            {
                if (string.IsNullOrEmpty(symbol))
                {
                    _earningsCache.Clear();
                }
                else
                {
                    _earningsCache.Remove(symbol.ToUpperInvariant());
                }
            }
        }
    }
}
