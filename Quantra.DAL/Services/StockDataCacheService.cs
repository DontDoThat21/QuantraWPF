using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Quantra.Models;
using System.Linq;
using System.Data;
using Quantra.DAL.Services.Interfaces;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Microsoft.Data.SqlClient;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Cache information for tracking cache state
    /// </summary>
    internal class CacheInfo
    {
        public DateTime CacheTime { get; set; }
        public int DataCount { get; set; }
        public string Symbol { get; set; }
        public string TimeRange { get; set; }
        public string Interval { get; set; }
    }
    /// <summary>
    /// Service for caching and retrieving stock data to minimize API calls
    /// </summary>
    public class StockDataCacheService : IStockDataCacheService
    {
        private readonly HistoricalDataService _historicalDataService;
        private readonly UserSettingsService _userSettingsService;
        private readonly LoggingService _loggingService;

        public StockDataCacheService(UserSettingsService userSettingsService, LoggingService loggingService)
        {
            _historicalDataService = new HistoricalDataService(userSettingsService, loggingService);
            _userSettingsService = userSettingsService;
            _loggingService = loggingService;
            EnsureCacheTableExists();
        }

        /// <summary>
        /// Ensures the stock data cache table exists in the database using Entity Framework Core
        /// </summary>
        private void EnsureCacheTableExists()
        {
            try
            {
                // Create DbContext using SQL Server configuration
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Ensure the database and tables are created
                    // This will create all tables defined in the DbContext including StockDataCache
                    dbContext.Database.EnsureCreated();

                    _loggingService.Log("Info", "Stock data cache tables created or verified using EF Core with SQL Server");
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to ensure stock data cache table exists", ex.ToString());
            }
        }

        /// <summary>
        /// Gets stock data for a symbol, either from cache or from the API
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range (e.g., "1d", "5d", "1mo", "1y")</param>
        /// <param name="interval">Data interval (e.g., "1m", "5m", "1d")</param>
        /// <param name="forceRefresh">Force refresh from API even if cache exists</param>
        /// <returns>List of historical price data</returns>
        public async Task<List<HistoricalPrice>> GetStockData(string symbol, string range = "1mo", string interval = "1d", bool forceRefresh = false)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
            }

            //DatabaseMonolith.Log("Debug", $"GetStockData called for {symbol} (range={range}, interval={interval}, forceRefresh={forceRefresh})");

            // First, check if we have cached data that's still valid
            if (!forceRefresh)
            {
                var cachedData = GetCachedData(symbol, range, interval);
                if (cachedData != null && cachedData.Count > 0)
                {
                    //DatabaseMonolith.Log("Info", $"Retrieved stock data for {symbol} from cache");

                    // Check if we should perform incremental update in background
                    var cacheInfo = GetCacheInfo(symbol, range, interval);
                    if (cacheInfo != null && ShouldPerformIncrementalUpdate(cacheInfo))
                    {
                        // Perform incremental update in background without blocking
                        _ = Task.Run(async () => await PerformIncrementalUpdateAsync(symbol, range, interval, cachedData));
                    }

                    return cachedData;
                }
            }

            // If we don't have cached data or force refresh is true, get from API
            var freshData = await _historicalDataService.GetHistoricalPrices(symbol, range, interval);

            // Cache the new data if we received it
            if (freshData != null && freshData.Count > 0)
            {
                CacheStockData(symbol, range, interval, freshData);
                //DatabaseMonolith.Log("Info", $"Fetched and cached fresh stock data for {symbol}");
            }
            else
            {
                //DatabaseMonolith.Log("Warning", $"Failed to fetch stock data for {symbol}");
            }

            return freshData;
        }

        /// <summary>
        /// Gets stock data from cache if it exists and is not expired
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range</param>
        /// <param name="interval">Data interval</param>
        /// <returns>Cached historical price data or null if not found or expired</returns>
        private List<HistoricalPrice> GetCachedData(string symbol, string range, string interval)
        {
            try
            {
                // Get user preference for cache duration (default 15 minutes)
                var settings = _userSettingsService.GetUserSettings();
                var cacheDurationMinutes = settings.CacheDurationMinutes;
                var expiryTime = DateTime.Now.AddMinutes(-cacheDurationMinutes);

                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Query the cache using LINQ and EF Core
                    var cacheEntry = dbContext.StockDataCache
                     .Where(c => c.Symbol == symbol
                        && c.TimeRange == range
                        && c.Interval == interval
                        && c.CachedAt > expiryTime)
                      .OrderByDescending(c => c.CachedAt)
                      .FirstOrDefault();

                    if (cacheEntry != null)
                    {
                        var storedData = cacheEntry.Data;
                        string jsonData;

                        // Check if data is compressed and decompress if needed
                        if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                        {
                            jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                        }
                        else
                        {
                            jsonData = storedData;
                        }

                        return Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
                    }

                    return null;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error retrieving cached stock data for {symbol}", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Caches stock data in the database
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range</param>
        /// <param name="interval">Data interval</param>
        /// <param name="data">Historical price data to cache</param>
        private void CacheStockData(string symbol, string range, string interval, List<HistoricalPrice> data)
        {
            try
            {
                // Serialize data to JSON and compress it
                var jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(data);
                var compressedData = Quantra.Utilities.CompressionHelper.CompressString(jsonData);

                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Check if cache entry already exists
                    var existing = dbContext.StockDataCache
                   .FirstOrDefault(c => c.Symbol == symbol && c.TimeRange == range && c.Interval == interval);

                    if (existing != null)
                    {
                        // Update existing entry
                        existing.Data = compressedData;
                        existing.CachedAt = DateTime.Now;
                    }
                    else
                    {
                        // Create new entry
                        dbContext.StockDataCache.Add(new StockDataCache
                        {
                            Symbol = symbol,
                            TimeRange = range,
                            Interval = interval,
                            Data = compressedData,
                            CachedAt = DateTime.Now
                        });
                    }

                    dbContext.SaveChanges();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error caching stock data for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Caches historical price data asynchronously
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range</param>
        /// <param name="interval">Data interval</param>
        /// <param name="data">Historical price data to cache</param>
        public async Task CacheHistoricalDataAsync(string symbol, string range, string interval, List<HistoricalPrice> data)
        {
            await Task.Run(() => CacheStockData(symbol, range, interval, data)).ConfigureAwait(false);
        }

        /// <summary>
        /// Checks if data exists for a symbol in the cache
        /// </summary>
        /// <param name="symbol">Stock symbol to check</param>
        /// <returns>True if cache contains data for the symbol</returns>
        public bool HasCachedData(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                return false;
            }

            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Query the cache using LINQ and EF Core
                    return dbContext.StockDataCache.Any(c => c.Symbol == symbol);
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error checking cached data for {symbol}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Gets cache information for a specific symbol and time range
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range</param>
        /// <param name="interval">Data interval</param>
        /// <returns>Cache information including timestamp and data count</returns>
        private CacheInfo GetCacheInfo(string symbol, string range, string interval)
        {
            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Query the cache using LINQ and EF Core
                    var cacheEntry = dbContext.StockDataCache
          .Where(c => c.Symbol == symbol && c.TimeRange == range && c.Interval == interval)
                          .OrderByDescending(c => c.CachedAt)
                      .FirstOrDefault();

                    if (cacheEntry != null)
                    {
                        var storedData = cacheEntry.Data;
                        string jsonData;

                        // Check if data is compressed and decompress if needed
                        if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                        {
                            jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                        }
                        else
                        {
                            jsonData = storedData;
                        }

                        var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);

                        return new CacheInfo
                        {
                            CacheTime = cacheEntry.CachedAt,
                            DataCount = prices?.Count ?? 0,
                            Symbol = symbol,
                            TimeRange = range,
                            Interval = interval
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error getting cache info for {symbol}", ex.ToString());
            }

            return null;
        }

        /// <summary>
        /// Determines if an incremental update should be performed for cached data
        /// </summary>
        /// <param name="cacheInfo">Cache information</param>
        /// <returns>True if incremental update is recommended</returns>
        private bool ShouldPerformIncrementalUpdate(CacheInfo cacheInfo)
        {
            if (cacheInfo == null) return false;

            var settings = _userSettingsService.GetUserSettings();
            var cacheDurationMinutes = settings.CacheDurationMinutes;

            // If cache is more than half expired, perform incremental update
            var halfExpiry = DateTime.Now.AddMinutes(-cacheDurationMinutes / 2);
            return cacheInfo.CacheTime < halfExpiry;
        }

        /// <summary>
        /// Performs incremental update by fetching only recent data and merging with cache
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="range">Time range</param>
        /// <param name="interval">Data interval</param>
        /// <param name="existingData">Existing cached data</param>
        private async Task PerformIncrementalUpdateAsync(string symbol, string range, string interval, List<HistoricalPrice> existingData)
        {
            try
            {
                //DatabaseMonolith.Log("Debug", $"Performing incremental update for {symbol}");

                // For daily data, fetch only last few days to update recent prices
                string incrementalRange = interval == "1d" ? "5d" : range;

                var recentData = await _historicalDataService.GetHistoricalPrices(symbol, incrementalRange, interval);

                if (recentData != null && recentData.Count > 0 && existingData != null)
                {
                    // Merge new data with existing data
                    var mergedData = MergeHistoricalData(existingData, recentData);

                    // Update cache with merged data
                    CacheStockData(symbol, range, interval, mergedData);

                    //DatabaseMonolith.Log("Info", $"Incremental update completed for {symbol}. Added {recentData.Count} recent records");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Incremental update failed for {symbol}: {ex.Message}");
            }
        }

        /// <summary>
        /// Merges new historical data with existing cached data, avoiding duplicates
        /// </summary>
        /// <param name="existingData">Existing cached data</param>
        /// <param name="newData">New data to merge</param>
        /// <returns>Merged data sorted by date</returns>
        private List<HistoricalPrice> MergeHistoricalData(List<HistoricalPrice> existingData, List<HistoricalPrice> newData)
        {
            // Create a dictionary for efficient lookups using full DateTime (including time component)
            // This supports both daily data (one entry per day) and intraday data (multiple entries per day)
            var mergedDict = existingData.ToDictionary(p => p.Date, p => p);

            foreach (var newPrice in newData)
            {
                // Check if this exact datetime already exists in the dictionary
                if (mergedDict.ContainsKey(newPrice.Date))
                {
                    // Update existing entry with newer data
                    mergedDict[newPrice.Date] = newPrice;
                }
                else
                {
                    // Add new entry
                    mergedDict[newPrice.Date] = newPrice;
                }
            }

            // Convert dictionary back to a list, sort by date, and return
            return mergedDict.Values.OrderBy(p => p.Date).ToList();
        }

        /// <summary>
        /// Clears expired cache entries
        /// </summary>
        /// <param name="maxAgeMinutes">Maximum age of cache entries in minutes</param>
        /// <returns>Number of entries cleared</returns>
        public int ClearExpiredCache(int maxAgeMinutes = 60)
        {
            try
            {
                var expiryTime = DateTime.Now.AddMinutes(-maxAgeMinutes);

                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Find all expired entries
                    var expiredEntries = dbContext.StockDataCache
                        .Where(c => c.CachedAt < expiryTime)
                              .ToList();

                    var count = expiredEntries.Count;

                    // Remove expired entries
                    dbContext.StockDataCache.RemoveRange(expiredEntries);
                    dbContext.SaveChanges();

                    _loggingService.Log("Info", $"Cleared {count} expired cache entries");
                    return count;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error clearing expired cache", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Deletes all cached data for a specific symbol.
        /// </summary>
        /// <param name="symbol">The stock symbol for which to delete cached data.</param>
        /// <returns>The number of cache entries deleted.</returns>
        public int DeleteCachedDataForSymbol(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                _loggingService.Log("Warning", "DeleteCachedDataForSymbol called with null or empty symbol.");
                return 0;
            }

            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Find all cache entries for the symbol
                    var entriesToDelete = dbContext.StockDataCache
                        .Where(c => c.Symbol == symbol)
                     .ToList();

                    var count = entriesToDelete.Count;

                    // Remove the entries
                    dbContext.StockDataCache.RemoveRange(entriesToDelete);
                    dbContext.SaveChanges();

                    _loggingService.Log("Info", $"Deleted {count} cache entries for symbol {symbol}");
                    return count;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error deleting cached data for symbol {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Returns a list of all cached stocks (latest data for each symbol, with all indicators)
        /// </summary>
        /// <remarks>
        /// NOTE: This method requires QuoteDataCache table which is not yet implemented in EF Core.
        /// Currently returns data from StockDataCache only.
        /// </remarks>
        public List<QuoteData> GetAllCachedStocks()
        {
            var stocks = new List<QuoteData>();
            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get unique symbols from StockDataCache
                    var symbols = dbContext.StockDataCache
                      .Select(c => c.Symbol)
                    .Distinct()
           .ToList();

                    // Batch load P/E ratios from FundamentalDataCache for all symbols
                    var peRatios = dbContext.FundamentalDataCache
                        .Where(f => symbols.Contains(f.Symbol) && f.DataType == "PERatio")
                        .ToList();
                    
                    // Create a dictionary for quick P/E ratio lookup
                    var peRatioDict = peRatios.ToDictionary(f => f.Symbol, f => f.Value ?? 0.0);

                    foreach (var symbol in symbols)
                    {
                        // Get the latest cache entry for each symbol
                        var latest = dbContext.StockDataCache
                .Where(c => c.Symbol == symbol)
                 .OrderByDescending(c => c.CachedAt)
                             .FirstOrDefault();

                        if (latest != null)
                        {
                            var storedData = latest.Data;
                            string jsonData;

                            // Check if data is compressed and decompress if needed
                            if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                            {
                                jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                            }
                            else
                            {
                                jsonData = storedData;
                            }

                            var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
                            if (prices != null && prices.Count > 0)
                            {
                                var last = prices.Last();
                                
                                // Get P/E ratio from cache if available
                                var peRatio = peRatioDict.ContainsKey(symbol) ? peRatioDict[symbol] : 0.0;
                                
                                stocks.Add(new QuoteData
                                {
                                    Symbol = symbol,
                                    Price = last.Close,
                                    Timestamp = last.Date,
                                    // Other fields will be default values
                                    Change = 0,
                                    ChangePercent = 0,
                                    DayHigh = 0,
                                    DayLow = 0,
                                    MarketCap = 0,
                                    Volume = 0,
                                    RSI = 0,
                                    PERatio = peRatio, // Set from FundamentalDataCache
                                    Date = last.Date,
                                    LastUpdated = DateTime.Now,
                                    LastAccessed = DateTime.Now
                                });
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error retrieving all cached stocks", ex.ToString());
            }
            return stocks;
        }

        /// <summary>
        /// Returns a paginated list of cached stocks (latest data for each symbol)
        /// </summary>
        /// <param name="page">Page number (1-based)</param>
        /// <param name="pageSize">Number of items per page</param>
        /// <returns>Paginated result with stocks and total count</returns>
        public async Task<(List<QuoteData> Stocks, int TotalCount)> GetCachedStocksPaginatedAsync(int page = 1, int pageSize = 25)
        {
            var stocks = new List<QuoteData>();
            int totalCount = 0;

            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get count and paginated symbols in efficient queries
                    var distinctSymbols = dbContext.StockDataCache
                        .Select(c => c.Symbol)
                        .Distinct();

                    totalCount = await distinctSymbols.CountAsync().ConfigureAwait(false);

                    // Apply pagination at database level
                    var paginatedSymbols = await distinctSymbols
                        .OrderBy(s => s)
                        .Skip((page - 1) * pageSize)
                        .Take(pageSize)
                        .ToListAsync()
                        .ConfigureAwait(false);

                    if (paginatedSymbols.Count == 0)
                    {
                        return (stocks, totalCount);
                    }

                    // Fetch latest cache entries for all paginated symbols
                    // Get all entries for paginated symbols, then filter in memory to get latest per symbol
                    var allEntriesForSymbols = await dbContext.StockDataCache
                        .Where(c => paginatedSymbols.Contains(c.Symbol))
                        .ToListAsync()
                        .ConfigureAwait(false);

                    // Group in memory and get the latest entry for each symbol
                    var latestEntries = allEntriesForSymbols
                        .GroupBy(c => c.Symbol)
                        .Select(g => g.OrderByDescending(c => c.CachedAt).First())
                        .ToList();

                    // Get all symbols from latest entries to batch-load fundamental data
                    var symbolsList = latestEntries.Select(e => e.Symbol).ToList();
                    
                    // Batch load P/E ratios from FundamentalDataCache for all symbols
                    var peRatios = await dbContext.FundamentalDataCache
                        .Where(f => symbolsList.Contains(f.Symbol) && f.DataType == "PERatio")
                        .ToListAsync()
                        .ConfigureAwait(false);
                    
                    // Batch load EPS from FundamentalDataCache for all symbols
                    var epsValues = await dbContext.FundamentalDataCache
                        .Where(f => symbolsList.Contains(f.Symbol) && f.DataType == "EPS")
                        .ToListAsync()
                        .ConfigureAwait(false);
                    
                    // Create dictionaries for quick lookup
                    var peRatioDict = peRatios.ToDictionary(f => f.Symbol, f => f.Value ?? 0.0);
                    var epsDict = epsValues.ToDictionary(f => f.Symbol, f => f.Value);

                    // Process entries to build QuoteData list
                    foreach (var entry in latestEntries)
                    {
                        if (entry == null) continue;

                        var storedData = entry.Data;
                        string jsonData;

                        // Check if data is compressed and decompress if needed
                        if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                        {
                            jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                        }
                        else
                        {
                            jsonData = storedData;
                        }

                        var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
                        if (prices != null && prices.Count > 0)
                        {
                            var last = prices.Last();
                            
                            // Get P/E ratio from cache if available
                            var peRatio = peRatioDict.ContainsKey(entry.Symbol) ? peRatioDict[entry.Symbol] : 0.0;
                            
                            // Get EPS from cache if available
                            var eps = epsDict.ContainsKey(entry.Symbol) ? epsDict[entry.Symbol] : null;
                            
                            stocks.Add(new QuoteData
                            {
                                Symbol = entry.Symbol,
                                Price = last.Close,
                                Timestamp = last.Date,
                                // Other fields will be default values
                                Change = 0,
                                ChangePercent = 0,
                                DayHigh = 0,
                                DayLow = 0,
                                MarketCap = 0,
                                Volume = 0,
                                RSI = 0,
                                PERatio = peRatio, // Set from FundamentalDataCache
                                EPS = eps, // Set from FundamentalDataCache
                                Date = last.Date,
                                LastUpdated = DateTime.Now,
                                LastAccessed = DateTime.Now,
                                CacheTime = entry.CachedAt // Set from database cache timestamp
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error retrieving paginated cached stocks", ex.ToString());
            }

            return (stocks, totalCount);
        }

        /// <summary>
        /// Gets the total count of cached stocks (unique symbols)
        /// </summary>
        /// <returns>Total number of unique symbols in cache</returns>
        public int GetCachedStocksCount()
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    return dbContext.StockDataCache
                        .Select(c => c.Symbol)
                        .Distinct()
                        .Count();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error getting cached stocks count", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Returns cached stock data for a specific symbol (complete data with all indicators) - Async version
        /// </summary>
        public async Task<QuoteData?> GetCachedStockAsync(string symbol)
        {
            return await Task.Run(() => GetCachedStock(symbol)).ConfigureAwait(false);
        }

        /// <summary>
        /// Returns cached stock data for a specific symbol (complete data with all indicators)
        /// </summary>
        /// <remarks>
        /// NOTE: This method requires QuoteDataCache table which is not yet implemented in EF Core.
        /// Currently returns data from StockDataCache only.
        /// </remarks>
        public QuoteData? GetCachedStock(string symbol)
        {
            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get the latest cache entry for the symbol
                    var latest = dbContext.StockDataCache
             .Where(c => c.Symbol == symbol)
                    .OrderByDescending(c => c.CachedAt)
               .FirstOrDefault();

                    if (latest != null)
                    {
                        var storedData = latest.Data;
                        string jsonData;

                        // Check if data is compressed and decompress if needed
                        if (Quantra.Utilities.CompressionHelper.IsCompressed(storedData))
                        {
                            jsonData = Quantra.Utilities.CompressionHelper.DecompressString(storedData);
                        }
                        else
                        {
                            jsonData = storedData;
                        }

                        var prices = Newtonsoft.Json.JsonConvert.DeserializeObject<List<HistoricalPrice>>(jsonData);
                        if (prices != null && prices.Count > 0)
                        {
                            var last = prices.Last();
                            
                            // Get P/E ratio from FundamentalDataCache if available
                            var peRatio = dbContext.FundamentalDataCache
                                .Where(f => f.Symbol == symbol && f.DataType == "PERatio")
                                .Select(f => f.Value)
                                .FirstOrDefault() ?? 0.0;
                            
                            // Get EPS from FundamentalDataCache if available
                            var eps = dbContext.FundamentalDataCache
                                .Where(f => f.Symbol == symbol && f.DataType == "EPS")
                                .Select(f => f.Value)
                                .FirstOrDefault();
                            
                            return new QuoteData
                            {
                                Symbol = symbol,
                                Price = last.Close,
                                Timestamp = last.Date,
                                // Other fields will be default values
                                Change = 0,
                                ChangePercent = 0,
                                DayHigh = 0,
                                DayLow = 0,
                                MarketCap = 0,
                                Volume = 0,
                                RSI = 0,
                                PERatio = peRatio, // Set from FundamentalDataCache
                                EPS = eps, // Set from FundamentalDataCache
                                Date = last.Date,
                                LastUpdated = DateTime.Now,
                                LastAccessed = DateTime.Now,
                                CacheTime = latest.CachedAt // Set from database cache timestamp
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error retrieving cached stock for {symbol}", ex.ToString());
            }
            return null;
        }

        /// <summary>
        /// Caches QuoteData - Async version
        /// </summary>
        public async Task CacheQuoteDataAsync(QuoteData quoteData)
        {
            await Task.Run(() => CacheQuoteData(quoteData)).ConfigureAwait(false);
        }

        /// <summary>
        /// Caches QuoteData
        /// </summary>
        /// <remarks>
        /// NOTE: This method requires QuoteDataCache table which is not yet implemented in EF Core.
        /// Currently stores data in StockDataCache only.
        /// </remarks>
        public void CacheQuoteData(QuoteData quoteData)
        {
            if (quoteData == null || string.IsNullOrEmpty(quoteData.Symbol))
                return;

            try
            {
                // Cache as historical data in StockDataCache
                var priceList = new List<HistoricalPrice>
    {
            new HistoricalPrice
      {
 Date = quoteData.Timestamp,
        Close = quoteData.Price,
     Open = quoteData.Price,
    High = quoteData.DayHigh,
          Low = quoteData.DayLow,
       Volume = (long)quoteData.Volume
    }
     };

                CacheStockData(quoteData.Symbol, "1mo", "1d", priceList);
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to cache QuoteData for {quoteData.Symbol}", ex.ToString());
            }
        }

        // IStockDataCacheService implementation

        public async Task<List<HistoricalPrice>> GetStockDataAsync(string symbol, string timeframe = "1mo", string interval = "1d", bool forceRefresh = false)
        {
            return await GetStockData(symbol, timeframe, interval, forceRefresh);
        }

        public async Task<QuoteData> GetQuoteDataAsync(string symbol, bool forceRefresh = false)
        {
            if (forceRefresh || !HasCachedData(symbol))
            {
                // This would typically call an API to get fresh data
                // For now just return cached data if available or null
                var cachedData = GetCachedStock(symbol);
                return await Task.FromResult(cachedData);
            }

            return await Task.FromResult(GetCachedStock(symbol));
        }

        public async Task<Dictionary<string, List<double>>> GetIndicatorDataAsync(
            string symbol,
            string indicatorType,
            string timeframe = "1mo",
            string interval = "1d",
            bool forceRefresh = false)
        {
            // This is a placeholder implementation
            // In a real implementation, we would calculate the indicator or retrieve it from cache
            return await Task.FromResult(new Dictionary<string, List<double>>());
        }

        public async Task<bool> ClearCacheForSymbolAsync(string symbol)
        {
            int deletedCount = DeleteCachedDataForSymbol(symbol);
            return await Task.FromResult(deletedCount > 0);
        }

        /// <summary>
        /// Performs background preloading for frequently accessed symbols
        /// </summary>
        /// <param name="symbols">List of symbols to preload</param>
        /// <param name="timeRange">Time range for data</param>
        /// <param name="interval">Data interval</param>
        public async Task PreloadSymbolsAsync(List<string> symbols, string timeRange = "1mo", string interval = "1d")
        {
            if (symbols == null || symbols.Count == 0)
                return;

            //DatabaseMonolith.Log("Info", $"Starting background preload for {symbols.Count} symbols");

            // Process symbols in small batches to avoid overwhelming the API
            const int batchSize = 3;
            for (int i = 0; i < symbols.Count; i += batchSize)
            {
                var batch = symbols.Skip(i).Take(batchSize);
                var tasks = batch.Select(symbol => PreloadSingleSymbolAsync(symbol, timeRange, interval));

                try
                {
                    await Task.WhenAll(tasks);

                    // Small delay between batches to be respectful to API limits
                    if (i + batchSize < symbols.Count)
                    {
                        await Task.Delay(2000); // 2 second delay between batches
                    }
                }
                catch (Exception ex)
                {
                    //DatabaseMonolith.Log("Warning", $"Error in preload batch starting at index {i}", ex.ToString());
                }
            }

            //DatabaseMonolith.Log("Info", "Background preload completed");
        }

        /// <summary>
        /// Preloads a single symbol in the background
        /// </summary>
        /// <param name="symbol">Symbol to preload</param>
        /// <param name="timeRange">Time range</param>
        /// <param name="interval">Interval</param>
        private async Task PreloadSingleSymbolAsync(string symbol, string timeRange, string interval)
        {
            try
            {
                // Check if we already have recent data
                var cacheInfo = GetCacheInfo(symbol, timeRange, interval);
                if (cacheInfo != null && !ShouldPerformIncrementalUpdate(cacheInfo))
                {
                    // Data is fresh, no need to preload
                    return;
                }

                // Fetch data in background (don't force refresh to respect existing cache)
                await GetStockData(symbol, timeRange, interval, forceRefresh: false);

                //DatabaseMonolith.Log("Debug", $"Preloaded data for {symbol}");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Warning", $"Failed to preload {symbol}: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets a list of frequently accessed symbols based on cache access patterns
        /// </summary>
        /// <param name="maxCount">Maximum number of symbols to return</param>
        /// <returns>List of frequently accessed symbols</returns>
        public List<string> GetFrequentlyAccessedSymbols(int maxCount = 10)
        {
            var symbols = new List<string>();

            try
            {
                var recentThreshold = DateTime.Now.AddDays(-7); // Last week

                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get symbols ordered by most recent cache time (proxy for frequency)
                    symbols = dbContext.StockDataCache
                          .Where(c => c.CachedAt > recentThreshold)
                          .GroupBy(c => c.Symbol)
                          .Select(g => new
                              {
                                  Symbol = g.Key,
                                  AccessCount = g.Count(),
                                  LastAccess = g.Max(c => c.CachedAt)
                              })
                          .OrderByDescending(x => x.AccessCount)
                          .ThenByDescending(x => x.LastAccess)
                          .Take(maxCount)
                          .Select(x => x.Symbol)
                          .ToList();
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error getting frequently accessed symbols", ex.ToString());
            }

            return symbols;
        }

        public async Task<bool> ClearAllCacheAsync()
        {
            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get all cache entries
                    var allEntries = dbContext.StockDataCache.ToList();
                    var count = allEntries.Count;

                    // Remove all entries
                    dbContext.StockDataCache.RemoveRange(allEntries);
                    dbContext.SaveChanges();

                    _loggingService.Log("Info", $"Cleared {count} cache entries");
                    return await Task.FromResult(true);
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to clear cache", ex.ToString());
                return await Task.FromResult(false);
            }
        }

        /// <summary>
        /// Gets a list of all symbols that have cached historical data
        /// </summary>
        /// <returns>List of symbols with cached data</returns>
        public List<string> GetAllCachedSymbols()
        {
            var symbols = new List<string>();

            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get distinct symbols from the cache
                    symbols = dbContext.StockDataCache
                        .Select(c => c.Symbol)
                        .Distinct()
                        .OrderBy(s => s)
                        .ToList();
                }

                _loggingService.Log("Info", $"Retrieved {symbols.Count} cached symbols");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error getting cached symbols", ex.ToString());
            }

            return symbols;
        }

        /// <summary>
        /// Gets a list of all symbols that have cached historical data (async version)
        /// </summary>
        /// <returns>List of symbols with cached data</returns>
        public async Task<List<string>> GetAllCachedSymbolsAsync()
        {
            var symbols = new List<string>();

            try
            {
                // Use Entity Framework Core with SQL Server
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    // Get distinct symbols from the cache asynchronously
                    symbols = await dbContext.StockDataCache
                        .Select(c => c.Symbol)
                        .Distinct()
                        .OrderBy(s => s)
                        .ToListAsync();
                }

                _loggingService.Log("Info", $"Retrieved {symbols.Count} cached symbols");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error getting cached symbols", ex.ToString());
            }

            return symbols;
        }

        /// <summary>
        /// Gets recent historical sequence for TFT model inference.
        /// Returns the most recent N days of OHLCV data for a symbol.
        /// CRITICAL for TFT: This provides real temporal sequences instead of synthetic repeated values.
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="days">Number of days to retrieve (default 60 for TFT lookback)</param>
        /// <param name="range">Time range to search (default "1y" to ensure enough history)</param>
        /// <param name="interval">Data interval (default "1d" for daily data)</param>
        /// <returns>List of historical prices ordered from oldest to newest (ready for TFT)</returns>
        public async Task<List<HistoricalPrice>> GetRecentHistoricalSequenceAsync(
            string symbol, 
            int days = 60, 
            string range = "1y", 
            string interval = "1d")
        {
            try
            {
                // Get cached data or fetch if needed
                var allData = await GetStockData(symbol, range, interval, forceRefresh: false);

                if (allData == null || allData.Count == 0)
                {
                    _loggingService.Log("Warning", $"No historical data available for {symbol}");
                    return new List<HistoricalPrice>();
                }

                // Sort by date ascending (oldest first) and take the last N days
                var recentData = allData
                    .OrderBy(p => p.Date)
                    .TakeLast(days)
                    .ToList();

                if (recentData.Count < days)
                {
                    _loggingService.Log("Warning", 
                        $"Requested {days} days for {symbol} but only {recentData.Count} available. " +
                        $"TFT performance may be degraded with insufficient lookback window.");
                }

                _loggingService.Log("Info", 
                    $"Retrieved {recentData.Count} days of historical sequence for {symbol} " +
                    $"(from {recentData.First().Date:yyyy-MM-dd} to {recentData.Last().Date:yyyy-MM-dd})");

                return recentData;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", 
                    $"Error retrieving historical sequence for {symbol}: {ex.Message}", 
                    ex.ToString());
                return new List<HistoricalPrice>();
            }
        }

        /// <summary>
        /// Gets recent historical sequence with calendar features for TFT model.
        /// Returns OHLCV data plus known-future covariates (day of week, month, etc.).
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="days">Number of days to retrieve (default 60 for TFT lookback)</param>
        /// <param name="futureHorizon">Additional days to project calendar features into future (default 30)</param>
        /// <returns>Dictionary with "prices" and "calendar_features" arrays</returns>
        public async Task<Dictionary<string, object>> GetHistoricalSequenceWithFeaturesAsync(
            string symbol,
            int days = 60,
            int futureHorizon = 30)
        {
            var result = new Dictionary<string, object>();

            try
            {
                // Get historical prices
                var prices = await GetRecentHistoricalSequenceAsync(symbol, days);

                if (prices == null || prices.Count == 0)
                {
                    result["prices"] = new List<HistoricalPrice>();
                    result["calendar_features"] = new List<Dictionary<string, object>>();
                    return result;
                }

                // Generate calendar features for historical period + future horizon
                var calendarFeatures = new List<Dictionary<string, object>>();
                var startDate = prices.First().Date;
                var endDate = prices.Last().Date.AddDays(futureHorizon);

                for (var date = startDate; date <= endDate; date = date.AddDays(1))
                {
                    // Skip weekends for stock market data
                    if (date.DayOfWeek == DayOfWeek.Saturday || date.DayOfWeek == DayOfWeek.Sunday)
                        continue;

                    calendarFeatures.Add(new Dictionary<string, object>
                    {
                        ["date"] = date.ToString("yyyy-MM-dd"),
                        ["dayofweek"] = (int)date.DayOfWeek,
                        ["day"] = date.Day,
                        ["month"] = date.Month,
                        ["quarter"] = (date.Month - 1) / 3 + 1,
                        ["year"] = date.Year,
                        ["is_month_end"] = date.Day >= DateTime.DaysInMonth(date.Year, date.Month) - 2 ? 1 : 0,
                        ["is_quarter_end"] = (date.Month % 3 == 0 && date.Day >= 28) ? 1 : 0,
                        ["is_year_end"] = (date.Month == 12 && date.Day >= 29) ? 1 : 0,
                        ["is_month_start"] = date.Day <= 5 ? 1 : 0,
                        ["is_friday"] = date.DayOfWeek == DayOfWeek.Friday ? 1 : 0,
                        ["is_monday"] = date.DayOfWeek == DayOfWeek.Monday ? 1 : 0,
                        ["is_potential_holiday_week"] = new[] { 1, 5, 7, 9, 11, 12 }.Contains(date.Month) ? 1 : 0
                    });
                }

                result["prices"] = prices;
                result["calendar_features"] = calendarFeatures;
                result["lookback_days"] = prices.Count;
                result["future_horizon_days"] = futureHorizon;
                result["total_calendar_days"] = calendarFeatures.Count;

                _loggingService.Log("Info", 
                    $"Generated historical sequence with features for {symbol}: " +
                    $"{prices.Count} historical days + {futureHorizon} future calendar days");
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", 
                    $"Error generating historical sequence with features for {symbol}: {ex.Message}",
                    ex.ToString());
                result["error"] = ex.Message;
            }

            return result;
        }
    }
}
