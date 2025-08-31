using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Threading.Tasks;
using Quantra.Models;
using System.Linq;
using System.Data;
using Quantra.DAL.Services.Interfaces;

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
        private static readonly string ConnectionString = "Data Source=Quantra.db;Version=3;";

        public StockDataCacheService()
        {
            _historicalDataService = new HistoricalDataService();
            EnsureCacheTableExists();
        }

        /// <summary>
        /// Ensures the stock data cache table exists in the database
        /// </summary>
        private void EnsureCacheTableExists()
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    // Add columns if missing (ALTER TABLE is limited in SQLite, so check and add if needed)
                    var command = new SQLiteCommand(
                        @"CREATE TABLE IF NOT EXISTS StockDataCache (
                            Symbol TEXT NOT NULL,
                            TimeRange TEXT NOT NULL,
                            Interval TEXT NOT NULL,
                            Data TEXT NOT NULL,
                            CacheTime DATETIME NOT NULL,
                            PRIMARY KEY (Symbol, TimeRange, Interval)
                        )", connection);
                    command.ExecuteNonQuery();

                    // Create QuoteDataCache table for complete quote data with indicators and predictions
                    command = new SQLiteCommand(
                        @"CREATE TABLE IF NOT EXISTS QuoteDataCache (
                            Symbol TEXT PRIMARY KEY,
                            Price REAL NOT NULL,
                            Change REAL NOT NULL DEFAULT 0,
                            ChangePercent REAL NOT NULL DEFAULT 0,
                            DayHigh REAL NOT NULL DEFAULT 0,
                            DayLow REAL NOT NULL DEFAULT 0,
                            MarketCap REAL NOT NULL DEFAULT 0,
                            Volume REAL NOT NULL DEFAULT 0,
                            RSI REAL NOT NULL DEFAULT 0,
                            PERatio REAL NOT NULL DEFAULT 0,
                            Date TEXT NOT NULL,
                            LastUpdated TEXT NOT NULL,
                            Timestamp TEXT NOT NULL,
                            CacheTime DATETIME NOT NULL,
                            PredictedPrice REAL NULL,
                            PredictedAction TEXT NULL,
                            PredictionConfidence REAL NULL,
                            PredictionTimestamp TEXT NULL,
                            PredictionModelVersion TEXT NULL
                        )", connection);
                    command.ExecuteNonQuery();



                    // Example: Add new columns if needed in the future
                    // command = new SQLiteCommand("ALTER TABLE StockDataCache ADD COLUMN NewColumn TEXT", connection);
                    // try { command.ExecuteNonQuery(); } catch { /* Ignore if already exists */ }
                }
                DatabaseMonolith.Log("Info", "Stock data cache tables created or verified");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to create stock data cache table", ex.ToString());
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

            DatabaseMonolith.Log("Debug", $"GetStockData called for {symbol} (range={range}, interval={interval}, forceRefresh={forceRefresh})");

            // First, check if we have cached data that's still valid
            if (!forceRefresh)
            {
                var cachedData = GetCachedData(symbol, range, interval);
                if (cachedData != null && cachedData.Count > 0)
                {
                    DatabaseMonolith.Log("Info", $"Retrieved stock data for {symbol} from cache");
                    
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
                DatabaseMonolith.Log("Info", $"Fetched and cached fresh stock data for {symbol}");
            }
            else
            {
                DatabaseMonolith.Log("Warning", $"Failed to fetch stock data for {symbol}");
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
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    // Get user preference for cache duration (default 15 minutes)
                    var settings = DatabaseMonolith.GetUserSettings();
                    var cacheDurationMinutes = settings.CacheDurationMinutes;
                    
                    var command = new SQLiteCommand(
                        @"SELECT Data FROM StockDataCache 
                          WHERE Symbol = @Symbol 
                          AND TimeRange = @Range 
                          AND Interval = @Interval 
                          AND CacheTime > @ExpiryTime", connection);
                    
                    command.Parameters.AddWithValue("@Symbol", symbol);
                    command.Parameters.AddWithValue("@Range", range);
                    command.Parameters.AddWithValue("@Interval", interval);
                    command.Parameters.AddWithValue("@ExpiryTime", DateTime.Now.AddMinutes(-cacheDurationMinutes));
                    
                    var result = command.ExecuteScalar();
                    if (result != null)
                    {
                        var storedData = result.ToString();
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
                DatabaseMonolith.Log("Error", $"Error retrieving cached stock data for {symbol}", ex.ToString());
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
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    // Serialize data to JSON and compress it
                    var jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(data);
                    var compressedData = Quantra.Utilities.CompressionHelper.CompressString(jsonData);
                    
                    var command = new SQLiteCommand(
                        @"INSERT OR REPLACE INTO StockDataCache 
                          (Symbol, TimeRange, Interval, Data, CacheTime) 
                          VALUES (@Symbol, @Range, @Interval, @Data, @CacheTime)", connection);
                    
                    command.Parameters.AddWithValue("@Symbol", symbol);
                    command.Parameters.AddWithValue("@Range", range);
                    command.Parameters.AddWithValue("@Interval", interval);
                    command.Parameters.AddWithValue("@Data", compressedData);
                    command.Parameters.AddWithValue("@CacheTime", DateTime.Now); // Save the API fetch time as LastUpdated in the DB, not DateTime.Now
                    
                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error caching stock data for {symbol}", ex.ToString());
            }
        }

        /// <summary>
        /// Checks if data exists for a symbol in the cache
        /// </summary>
        /// <param name="symbol">Stock symbol to check</param>
        /// <returns>True if cache contains data for the symbol</returns>
        public bool HasCachedData(string symbol)
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    var command = new SQLiteCommand(
                        "SELECT COUNT(*) FROM StockDataCache WHERE Symbol = @Symbol", connection);
                    command.Parameters.AddWithValue("@Symbol", symbol);
                    
                    var count = Convert.ToInt32(command.ExecuteScalar());
                    return count > 0;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error checking cached data for {symbol}", ex.ToString());
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
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    var command = new SQLiteCommand(
                        @"SELECT CacheTime, Data FROM StockDataCache 
                          WHERE Symbol = @Symbol 
                          AND TimeRange = @Range 
                          AND Interval = @Interval", connection);
                    
                    command.Parameters.AddWithValue("@Symbol", symbol);
                    command.Parameters.AddWithValue("@Range", range);
                    command.Parameters.AddWithValue("@Interval", interval);
                    
                    using (var reader = command.ExecuteReader())
                    {
                        if (reader.Read())
                        {
                            var cacheTime = reader.GetDateTime(0);
                            var storedData = reader.GetString(1);
                            
                            // Get data count without full deserialization
                            string jsonData;
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
                                CacheTime = cacheTime,
                                DataCount = prices?.Count ?? 0,
                                Symbol = symbol,
                                TimeRange = range,
                                Interval = interval
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error getting cache info for {symbol}", ex.ToString());
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
            
            var settings = DatabaseMonolith.GetUserSettings();
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
                DatabaseMonolith.Log("Debug", $"Performing incremental update for {symbol}");
                
                // For daily data, fetch only last few days to update recent prices
                string incrementalRange = interval == "1d" ? "5d" : range;
                
                var recentData = await _historicalDataService.GetHistoricalPrices(symbol, incrementalRange, interval);
                
                if (recentData != null && recentData.Count > 0 && existingData != null)
                {
                    // Merge new data with existing data
                    var mergedData = MergeHistoricalData(existingData, recentData);
                    
                    // Update cache with merged data
                    CacheStockData(symbol, range, interval, mergedData);
                    
                    DatabaseMonolith.Log("Info", $"Incremental update completed for {symbol}. Added {recentData.Count} recent records");
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Incremental update failed for {symbol}: {ex.Message}");
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
            // Create a dictionary for efficient lookups
            var mergedDict = existingData.ToDictionary(p => p.Date.Date, p => p);
            
            foreach (var newPrice in newData)
            {
                // Check if this date already exists in the dictionary
                if (mergedDict.ContainsKey(newPrice.Date.Date))
                {
                    // Update existing entry with newer data
                    mergedDict[newPrice.Date.Date] = newPrice;
                }
                else
                {
                    // Add new entry
                    mergedDict[newPrice.Date.Date] = newPrice;
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
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    var command = new SQLiteCommand(
                        "DELETE FROM StockDataCache WHERE CacheTime < @ExpiryTime", connection);
                    command.Parameters.AddWithValue("@ExpiryTime", DateTime.Now.AddMinutes(-maxAgeMinutes));
                    
                    var count = command.ExecuteNonQuery();
                    DatabaseMonolith.Log("Info", $"Cleared {count} expired cache entries");
                    return count;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error clearing expired cache", ex.ToString());
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
                DatabaseMonolith.Log("Warning", "DeleteCachedDataForSymbol called with null or empty symbol.");
                return 0;
            }

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    var command = new SQLiteCommand(
                        "DELETE FROM StockDataCache WHERE Symbol = @Symbol", connection);
                    command.Parameters.AddWithValue("@Symbol", symbol);
                    
                    var count = command.ExecuteNonQuery();
                    DatabaseMonolith.Log("Info", $"Deleted {count} cache entries for symbol {symbol}");
                    return count;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error deleting cached data for symbol {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Returns a list of all cached stocks (latest data for each symbol, with all indicators)
        /// </summary>
        public List<QuoteData> GetAllCachedStocks()
        {
            var stocks = new List<QuoteData>();
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    // First try to get from QuoteDataCache table (complete data with predictions)
                    var command = new SQLiteCommand(
                        @"SELECT Symbol, Price, Change, ChangePercent, DayHigh, DayLow, MarketCap, Volume, RSI, PERatio, Date, LastUpdated, Timestamp,
                                 PredictedPrice, PredictedAction, PredictionConfidence, PredictionTimestamp, PredictionModelVersion
                          FROM QuoteDataCache
                          ORDER BY CacheTime DESC", connection);

                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            var stock = new QuoteData
                            {
                                Symbol = reader.GetString(0),
                                Price = reader.GetDouble(1),
                                Change = reader.GetDouble(2),
                                ChangePercent = reader.GetDouble(3),
                                DayHigh = reader.GetDouble(4),
                                DayLow = reader.GetDouble(5),
                                MarketCap = reader.GetDouble(6),
                                Volume = reader.GetDouble(7),
                                RSI = reader.GetDouble(8),
                                PERatio = reader.GetDouble(9),
                                Date = DateTime.Parse(reader.GetString(10)),
                                LastUpdated = DateTime.Parse(reader.GetString(11)),
                                LastAccessed = DateTime.Now,
                                Timestamp = DateTime.Parse(reader.GetString(12))
                            };

                            // Load prediction data if available
                            if (!reader.IsDBNull(13))
                                stock.PredictedPrice = reader.GetDouble(13);
                            if (!reader.IsDBNull(14))
                                stock.PredictedAction = reader.GetString(14);
                            if (!reader.IsDBNull(15))
                                stock.PredictionConfidence = reader.GetDouble(15);
                            if (!reader.IsDBNull(16))
                                stock.PredictionTimestamp = DateTime.Parse(reader.GetString(16));
                            if (!reader.IsDBNull(17))
                                stock.PredictionModelVersion = reader.GetString(17);

                            stocks.Add(stock);
                        }
                    }
                    
                    // If no data in QuoteDataCache, fall back to old table
                    if (stocks.Count == 0)
                    {
                        command = new SQLiteCommand(
                            @"SELECT Symbol, Data FROM StockDataCache
                              GROUP BY Symbol
                              ORDER BY CacheTime DESC", connection);

                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                var symbol = reader.GetString(0);
                                var storedData = reader.GetString(1);
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
                                        PERatio = 0,
                                        Date = last.Date,
                                        LastUpdated = DateTime.Now,
                                        LastAccessed = DateTime.Now
                                    });
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error retrieving all cached stocks", ex.ToString());
            }
            return stocks;
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
        public QuoteData? GetCachedStock(string symbol)
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    // First try to get from QuoteDataCache table (complete data with predictions)
                    var command = new SQLiteCommand(
                        @"SELECT Symbol, Price, Change, ChangePercent, DayHigh, DayLow, MarketCap, Volume, RSI, PERatio, Date, LastUpdated, Timestamp,
                                 PredictedPrice, PredictedAction, PredictionConfidence, PredictionTimestamp, PredictionModelVersion
                          FROM QuoteDataCache
                          WHERE Symbol = @Symbol
                          ORDER BY CacheTime DESC
                          LIMIT 1", connection);
                    command.Parameters.AddWithValue("@Symbol", symbol);

                    using (var reader = command.ExecuteReader())
                    {
                        if (reader.Read())
                        {
                            var stock = new QuoteData
                            {
                                Symbol = reader.GetString(0),
                                Price = reader.GetDouble(1),
                                Change = reader.GetDouble(2),
                                ChangePercent = reader.GetDouble(3),
                                DayHigh = reader.GetDouble(4),
                                DayLow = reader.GetDouble(5),
                                MarketCap = reader.GetDouble(6),
                                Volume = reader.GetDouble(7),
                                RSI = reader.GetDouble(8),
                                PERatio = reader.GetDouble(9),
                                Date = DateTime.Parse(reader.GetString(10)),
                                LastUpdated = DateTime.Parse(reader.GetString(11)),
                                LastAccessed = DateTime.Now,
                                Timestamp = DateTime.Parse(reader.GetString(12))
                            };

                            // Load prediction data if available
                            if (!reader.IsDBNull(13))
                                stock.PredictedPrice = reader.GetDouble(13);
                            if (!reader.IsDBNull(14))
                                stock.PredictedAction = reader.GetString(14);
                            if (!reader.IsDBNull(15))
                                stock.PredictionConfidence = reader.GetDouble(15);
                            if (!reader.IsDBNull(16))
                                stock.PredictionTimestamp = DateTime.Parse(reader.GetString(16));
                            if (!reader.IsDBNull(17))
                                stock.PredictionModelVersion = reader.GetString(17);

                            return stock;
                        }
                    }
                    
                    // Fall back to old table if not found in QuoteDataCache
                    command = new SQLiteCommand(
                        @"SELECT Data FROM StockDataCache
                          WHERE Symbol = @Symbol
                          ORDER BY CacheTime DESC
                          LIMIT 1", connection);
                    command.Parameters.AddWithValue("@Symbol", symbol);

                    var result = command.ExecuteScalar();
                    if (result != null)
                    {
                        var storedData = result.ToString();
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
                                PERatio = 0,
                                Date = last.Date,
                                LastUpdated = DateTime.Now,
                                LastAccessed = DateTime.Now
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error retrieving cached stock for {symbol}", ex.ToString());
            }
            return null;
        }

        // Add this public method to allow caching of QuoteData directly - Async version
        public async Task CacheQuoteDataAsync(QuoteData quoteData)
        {
            await Task.Run(() => CacheQuoteData(quoteData)).ConfigureAwait(false);
        }

        // Add this public method to allow caching of QuoteData directly
        public void CacheQuoteData(QuoteData quoteData)
        {
            if (quoteData == null || string.IsNullOrEmpty(quoteData.Symbol))
                return;

            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    // Cache complete quote data in QuoteDataCache table with predictions
                    var command = new SQLiteCommand(
                        @"INSERT OR REPLACE INTO QuoteDataCache 
                          (Symbol, Price, Change, ChangePercent, DayHigh, DayLow, MarketCap, Volume, RSI, PERatio, Date, LastUpdated, Timestamp, CacheTime, 
                           PredictedPrice, PredictedAction, PredictionConfidence, PredictionTimestamp, PredictionModelVersion) 
                          VALUES (@Symbol, @Price, @Change, @ChangePercent, @DayHigh, @DayLow, @MarketCap, @Volume, @RSI, @PERatio, @Date, @LastUpdated, @Timestamp, @CacheTime,
                           @PredictedPrice, @PredictedAction, @PredictionConfidence, @PredictionTimestamp, @PredictionModelVersion)", connection);

                    command.Parameters.AddWithValue("@Symbol", quoteData.Symbol);
                    command.Parameters.AddWithValue("@Price", quoteData.Price);
                    command.Parameters.AddWithValue("@Change", quoteData.Change);
                    command.Parameters.AddWithValue("@ChangePercent", quoteData.ChangePercent);
                    command.Parameters.AddWithValue("@DayHigh", quoteData.DayHigh);
                    command.Parameters.AddWithValue("@DayLow", quoteData.DayLow);
                    command.Parameters.AddWithValue("@MarketCap", quoteData.MarketCap);
                    command.Parameters.AddWithValue("@Volume", quoteData.Volume);
                    command.Parameters.AddWithValue("@RSI", quoteData.RSI);
                    command.Parameters.AddWithValue("@PERatio", quoteData.PERatio);
                    command.Parameters.AddWithValue("@Date", quoteData.Date.ToString("yyyy-MM-dd HH:mm:ss"));
                    command.Parameters.AddWithValue("@LastUpdated", quoteData.LastUpdated.ToString("yyyy-MM-dd HH:mm:ss"));
                    command.Parameters.AddWithValue("@Timestamp", quoteData.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"));
                    command.Parameters.AddWithValue("@CacheTime", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));

                    // Add prediction fields (nullable)
                    command.Parameters.AddWithValue("@PredictedPrice", (object)quoteData.PredictedPrice ?? DBNull.Value);
                    command.Parameters.AddWithValue("@PredictedAction", (object)quoteData.PredictedAction ?? DBNull.Value);
                    command.Parameters.AddWithValue("@PredictionConfidence", (object)quoteData.PredictionConfidence ?? DBNull.Value);
                    command.Parameters.AddWithValue("@PredictionTimestamp", quoteData.PredictionTimestamp?.ToString("yyyy-MM-dd HH:mm:ss") ?? (object)DBNull.Value);
                    command.Parameters.AddWithValue("@PredictionModelVersion", (object)quoteData.PredictionModelVersion ?? DBNull.Value);

                    command.ExecuteNonQuery();

                    // Also cache as historical data for backwards compatibility
                    var priceList = new List<HistoricalPrice>
                    {
                        new HistoricalPrice
                        {
                            Date = quoteData.Timestamp,
                            Close = quoteData.Price
                        }
                    };
                    var jsonData = Newtonsoft.Json.JsonConvert.SerializeObject(priceList);
                    var compressedData = Quantra.Utilities.CompressionHelper.CompressString(jsonData);

                    command = new SQLiteCommand(
                        @"INSERT OR REPLACE INTO StockDataCache 
                          (Symbol, TimeRange, Interval, Data, CacheTime) 
                          VALUES (@Symbol, @Range, @Interval, @Data, @CacheTime)", connection);

                    command.Parameters.AddWithValue("@Symbol", quoteData.Symbol);
                    command.Parameters.AddWithValue("@Range", "1mo");
                    command.Parameters.AddWithValue("@Interval", "1d");
                    command.Parameters.AddWithValue("@Data", compressedData);
                    command.Parameters.AddWithValue("@CacheTime", quoteData.Timestamp);

                    command.ExecuteNonQuery();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to cache QuoteData for {quoteData.Symbol}", ex.ToString());
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

            DatabaseMonolith.Log("Info", $"Starting background preload for {symbols.Count} symbols");

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
                    DatabaseMonolith.Log("Warning", $"Error in preload batch starting at index {i}", ex.ToString());
                }
            }

            DatabaseMonolith.Log("Info", "Background preload completed");
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
                
                DatabaseMonolith.Log("Debug", $"Preloaded data for {symbol}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Warning", $"Failed to preload {symbol}: {ex.Message}");
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
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    
                    // Get symbols ordered by most recent cache time (proxy for frequency)
                    var command = new SQLiteCommand(
                        @"SELECT Symbol, COUNT(*) as AccessCount 
                          FROM StockDataCache 
                          WHERE CacheTime > @RecentThreshold
                          GROUP BY Symbol 
                          ORDER BY AccessCount DESC, MAX(CacheTime) DESC 
                          LIMIT @MaxCount", connection);
                    
                    command.Parameters.AddWithValue("@RecentThreshold", DateTime.Now.AddDays(-7)); // Last week
                    command.Parameters.AddWithValue("@MaxCount", maxCount);
                    
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            symbols.Add(reader.GetString(0));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error getting frequently accessed symbols", ex.ToString());
            }
            
            return symbols;
        }
        
        public async Task<bool> ClearAllCacheAsync()
        {
            try
            {
                using (var connection = new SQLiteConnection(ConnectionString))
                {
                    connection.Open();
                    var command = new SQLiteCommand("DELETE FROM StockDataCache", connection);
                    int count = command.ExecuteNonQuery();
                    DatabaseMonolith.Log("Info", $"Cleared {count} cache entries");
                    return await Task.FromResult(true);
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to clear cache", ex.ToString());
                return await Task.FromResult(false);
            }
        }
    }
}

// ...existing code...
// No changes required for database persistence here unless you want to cache analysis results.
// ...existing code...
