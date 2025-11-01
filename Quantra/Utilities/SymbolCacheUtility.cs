using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Linq;

namespace Quantra
{
    /// <summary>
    /// Utility class for managing stock symbol caching operations
    /// </summary>
    public static class SymbolCacheUtility
    {
        /// <summary>
        /// Checks if the symbol cache is stale based on specified criteria
        /// </summary>
        /// <param name="maxAgeHours">Maximum age of cache in hours before considered stale (default: 24)</param>
        /// <returns>True if cache is stale or unusable, false if cache is valid</returns>
        public static bool IsSymbolCacheStale(int maxAgeHours = 24)
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    // Check if table exists
                    bool tableExists = DatabaseMonolith.ExecuteScalar<int>(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='StockSymbols'") > 0;
                    
                    if (!tableExists)
                    {
                        //DatabaseMonolith.Log("Info", "StockSymbols table does not exist - cache is stale");
                        return true;
                    }
                    
                    // Check if table has symbols
                    int symbolCount = DatabaseMonolith.ExecuteScalar<int>("SELECT COUNT(*) FROM StockSymbols");
                    if (symbolCount == 0)
                    {
                        //DatabaseMonolith.Log("Info", "StockSymbols table is empty - cache is stale");
                        return true;
                    }
                    
                    // Check age of cache
                    var query = "SELECT MIN(LastUpdated) FROM StockSymbols";
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        var result = command.ExecuteScalar();
                        if (result == null || result == DBNull.Value)
                        {
                            //DatabaseMonolith.Log("Warning", "Could not determine cache age - assuming stale");
                            return true;
                        }
                        
                        DateTime oldestUpdate = Convert.ToDateTime(result);
                        TimeSpan age = DateTime.Now - oldestUpdate;
                        
                        bool isStale = age.TotalHours > maxAgeHours;
                        if (isStale)
                        {
                            //DatabaseMonolith.Log("Info", $"Symbol cache is stale (age: {age.TotalHours:F1} hours, max: {maxAgeHours} hours)");
                        }
                        
                        return isStale;
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error checking symbol cache status", ex.ToString());
                return true; // Assume cache is stale on error
            }
        }

        /// <summary>
        /// Finds symbols that match a search pattern, prioritizing exact and prefix matches
        /// </summary>
        /// <param name="searchPattern">The pattern to search for</param>
        /// <param name="maxResults">Maximum number of results to return (default: 20)</param>
        /// <returns>List of matching StockSymbol objects in priority order</returns>
        public static List<Quantra.Models.StockSymbol> FindSymbolMatches(string searchPattern, int maxResults = 20)
        {
            var results = new List<Quantra.Models.StockSymbol>();
            
            if (string.IsNullOrWhiteSpace(searchPattern))
            {
                return results;
            }
            
            searchPattern = searchPattern.Trim().ToUpper();
            
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    // Use a sophisticated query that prioritizes matching using a CASE expression
                    string query = @"
                        SELECT Symbol, Name, Sector, Industry, LastUpdated
                        FROM StockSymbols
                        WHERE Symbol LIKE @SearchPattern OR Name LIKE @SearchPattern
                        ORDER BY
                            CASE
                                WHEN Symbol = @ExactMatch THEN 1                           -- Exact match has highest priority
                                WHEN Symbol LIKE @StartsWith THEN 2                        -- Starts with has next priority
                                WHEN Symbol LIKE @Contains THEN 3                          -- Contains has third priority
                                WHEN Name LIKE @StartsWith OR Name LIKE @ExactMatch THEN 4 -- Name matches have lower priority
                                ELSE 5                                                     -- Everything else
                            END,
                            LENGTH(Symbol),  -- Shorter symbols come first
                            Symbol           -- Alphabetical within each category
                        LIMIT @MaxResults";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        // Add parameters with appropriate wildcards for different matching scenarios
                        command.Parameters.AddWithValue("@SearchPattern", $"%{searchPattern}%");
                        command.Parameters.AddWithValue("@ExactMatch", searchPattern);
                        command.Parameters.AddWithValue("@StartsWith", $"{searchPattern}%");
                        command.Parameters.AddWithValue("@Contains", $"%{searchPattern}%");
                        command.Parameters.AddWithValue("@MaxResults", maxResults);
                        
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                results.Add(new Quantra.Models.StockSymbol
                                {
                                    Symbol = reader["Symbol"].ToString(),
                                    Name = reader["Name"]?.ToString() ?? string.Empty,
                                    Sector = reader["Sector"]?.ToString() ?? string.Empty,
                                    Industry = reader["Industry"]?.ToString() ?? string.Empty,
                                    LastUpdated = reader["LastUpdated"] != DBNull.Value ? 
                                        Convert.ToDateTime(reader["LastUpdated"]) : DateTime.MinValue
                                });
                            }
                        }
                    }
                    
                    //DatabaseMonolith.Log("Info", $"Found {results.Count} symbol matches for pattern '{searchPattern}'");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error finding symbol matches for pattern '{searchPattern}'", ex.ToString());
            }
            
            return results;
        }

        /// <summary>
        /// Updates the LastUpdated timestamp for all symbols to refresh the cache
        /// </summary>
        /// <returns>Number of symbols updated</returns>
        public static int RefreshSymbolCache()
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    string query = "UPDATE StockSymbols SET LastUpdated = @Now";
                    
                    using (var command = new SQLiteCommand(query, connection))
                    {
                        command.Parameters.AddWithValue("@Now", DateTime.Now);
                        int count = command.ExecuteNonQuery();
                        //DatabaseMonolith.Log("Info", $"Refreshed symbol cache: updated {count} symbols");
                        return count;
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to refresh symbol cache", ex.ToString());
                return 0;
            }
        }
        
        /// <summary>
        /// Retrieves the total count of symbols in the cache
        /// </summary>
        /// <returns>Number of symbols in the cache</returns>
        public static int GetSymbolCount()
        {
            try
            {
                return DatabaseMonolith.ExecuteScalar<int>("SELECT COUNT(*) FROM StockSymbols");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error getting symbol count", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Ensures VIX is available in the symbol cache for searching
        /// </summary>
        public static void EnsureVixInCache()
        {
            try
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    // Check if VIX already exists
                    string checkQuery = "SELECT COUNT(*) FROM StockSymbols WHERE Symbol = @Symbol";
                    using (var checkCommand = new SQLiteCommand(checkQuery, connection))
                    {
                        checkCommand.Parameters.AddWithValue("@Symbol", "VIX");
                        int count = Convert.ToInt32(checkCommand.ExecuteScalar());
                        
                        if (count == 0)
                        {
                            // Insert VIX if it doesn't exist
                            string insertQuery = @"
                                INSERT INTO StockSymbols (Symbol, Name, Sector, Industry, LastUpdated)
                                VALUES (@Symbol, @Name, @Sector, @Industry, @LastUpdated)";
                            
                            using (var insertCommand = new SQLiteCommand(insertQuery, connection))
                            {
                                insertCommand.Parameters.AddWithValue("@Symbol", "VIX");
                                insertCommand.Parameters.AddWithValue("@Name", "CBOE Volatility Index");
                                insertCommand.Parameters.AddWithValue("@Sector", "Index");
                                insertCommand.Parameters.AddWithValue("@Industry", "Volatility Index");
                                insertCommand.Parameters.AddWithValue("@LastUpdated", DateTime.Now);
                                insertCommand.ExecuteNonQuery();
                                
                                //DatabaseMonolith.Log("Info", "Added VIX to symbol cache");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Error ensuring VIX in cache", ex.ToString());
            }
        }
    }
}
