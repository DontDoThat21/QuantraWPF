using System;
using System.Collections.Generic;
//using System.Data.SQLite;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra
{
    /// <summary>
    /// Utility class for managing stock symbol caching operations
    /// </summary>
    public static class SymbolCacheUtility
    {
        private static QuantraDbContext CreateContext()
        {
            // Use default OnConfiguring in QuantraDbContext when no options are provided
            var options = new DbContextOptionsBuilder<QuantraDbContext>().Options;
            return new QuantraDbContext(options);
        }

        /// <summary>
        /// Checks if the symbol cache is stale based on specified criteria
        /// </summary>
        /// <param name="maxAgeHours">Maximum age of cache in hours before considered stale (default:24)</param>
        /// <returns>True if cache is stale or unusable, false if cache is valid</returns>
        public static bool IsSymbolCacheStale(int maxAgeHours = 24)
        {
            try
            {
                using (var context = CreateContext())
                {
                    // Try to connect to the database
                    if (!context.Database.CanConnect())
                    {
                        return true;
                    }

                    // If table doesn't exist or has no symbols, consider stale
                    if (!context.StockSymbols.Any())
                    {
                        return true;
                    }

                    // Determine the oldest LastUpdated value
                    DateTime? oldestUpdate = context.StockSymbols.Min(s => s.LastUpdated);
                    if (!oldestUpdate.HasValue)
                    {
                        return true;
                    }

                    TimeSpan age = DateTime.Now - oldestUpdate.Value;
                    return age.TotalHours > maxAgeHours;
                }
            }
            catch (Exception)
            {
                return true; // Assume stale on error
            }
        }

        /// <summary>
        /// Finds symbols that match a search pattern, prioritizing exact and prefix matches
        /// </summary>
        /// <param name="searchPattern">The pattern to search for</param>
        /// <param name="maxResults">Maximum number of results to return (default:20)</param>
        /// <returns>List of matching StockSymbol objects in priority order</returns>
        public static List<Quantra.Models.StockSymbol> FindSymbolMatches(string searchPattern, int maxResults = 20)
        {
            var results = new List<Quantra.Models.StockSymbol>();

            if (string.IsNullOrWhiteSpace(searchPattern))
            {
                return results;
            }

            string pattern = searchPattern.Trim();
            string patternUpper = pattern.ToUpperInvariant();

            try
            {
                using (var context = CreateContext())
                {
                    // Load symbols into memory to avoid complex EF translations
                    var all = context.StockSymbols.AsNoTracking().ToList();

                    var matches = all
                        .Where(s => (!string.IsNullOrEmpty(s.Symbol) && s.Symbol.IndexOf(pattern, StringComparison.OrdinalIgnoreCase) >= 0) ||
                                    (!string.IsNullOrEmpty(s.Name) && s.Name.IndexOf(pattern, StringComparison.OrdinalIgnoreCase) >= 0))
                        .Select(s => new
                        {
                            Entity = s,
                            Priority = GetPriority(s, patternUpper)
                        })
                        .OrderBy(x => x.Priority)
                        .ThenBy(x => string.IsNullOrEmpty(x.Entity.Symbol) ? int.MaxValue : x.Entity.Symbol.Length)
                        .ThenBy(x => x.Entity.Symbol)
                        .Take(maxResults)
                        .ToList();

                    foreach (var item in matches)
                    {
                        var s = item.Entity;
                        results.Add(new Quantra.Models.StockSymbol
                        {
                            Symbol = s.Symbol,
                            Name = s.Name ?? string.Empty,
                            Sector = s.Sector ?? string.Empty,
                            Industry = s.Industry ?? string.Empty,
                            LastUpdated = s.LastUpdated ?? DateTime.MinValue
                        });
                    }
                }
            }
            catch (Exception)
            {
                // ignore and return what we have
            }

            return results;
        }

        private static int GetPriority(StockSymbolEntity s, string patternUpper)
        {
            if (!string.IsNullOrEmpty(s.Symbol))
            {
                var symUpper = s.Symbol.ToUpperInvariant();
                if (symUpper == patternUpper) return 1;
                if (symUpper.StartsWith(patternUpper)) return 2;
                if (symUpper.Contains(patternUpper)) return 3;
            }

            if (!string.IsNullOrEmpty(s.Name))
            {
                var nameUpper = s.Name.ToUpperInvariant();
                if (nameUpper.StartsWith(patternUpper) || nameUpper == patternUpper) return 4;
            }

            return 5;
        }

        /// <summary>
        /// Updates the LastUpdated timestamp for all symbols to refresh the cache
        /// </summary>
        /// <returns>Number of symbols updated</returns>
        public static int RefreshSymbolCache()
        {
            try
            {
                using (var context = CreateContext())
                {
                    var all = context.StockSymbols.ToList();
                    var now = DateTime.Now;
                    foreach (var s in all)
                    {
                        s.LastUpdated = now;
                    }

                    context.SaveChanges();
                    return all.Count;
                }
            }
            catch (Exception)
            {
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
                using (var context = CreateContext())
                {
                    return context.StockSymbols.Count();
                }
            }
            catch (Exception)
            {
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
                using (var context = CreateContext())
                {
                    bool exists = context.StockSymbols.Any(s => s.Symbol == "VIX");
                    if (!exists)
                    {
                        var entity = new StockSymbolEntity
                        {
                            Symbol = "VIX",
                            Name = "CBOE Volatility Index",
                            Sector = "Index",
                            Industry = "Volatility Index",
                            LastUpdated = DateTime.Now
                        };

                        context.StockSymbols.Add(entity);
                        context.SaveChanges();
                    }
                }
            }
            catch (Exception)
            {
                // ignore errors
            }
        }
    }
}
