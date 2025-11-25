using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Quantra.Models
{
    /// <summary>
    /// Class for managing stock symbol data with caching and fallback mechanisms
    /// </summary>
    public static class StockSymbols
    {
        private static readonly string CacheFilePath = "stocksymbols_cache.json";
        private static List<string> _cachedSymbols;

        /// <summary>
        /// Gets the major stock indices and their constituents
        /// </summary>
        public static Dictionary<string, List<string>> Indices => new Dictionary<string, List<string>>
        {
            {
                "NASDAQ100", new List<string>
                {
                    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "FB", "TSLA", "NVDA", "PYPL", "ADBE",
                    "NFLX", "CMCSA", "INTC", "CSCO", "PEP", "AVGO", "COST", "TXN", "QCOM", "AMD"
                }
            },
            {
                "DOW", new List<string>
                {
                    "AAPL", "MSFT", "JPM", "V", "WMT", "UNH", "HD", "DIS", "INTC", "CSCO",
                    "VZ", "PG", "MCD", "MRK", "BA", "KO", "CRM", "CAT", "GS", "AXP"
                }
            },
            {
                "S&P500Top", new List<string>
                {
                    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "FB", "TSLA", "BRK.B", "JNJ", "JPM",
                    "V", "PG", "UNH", "HD", "MA", "NVDA", "BAC", "DIS", "PYPL", "ADBE"
                }
            },
            {
                "Popular", new List<string>
                {
                    "AAPL", "MSFT", "TSLA", "AMC", "GME", "PLTR", "NIO", "LCID", "BB", "NOK",
                    "NVDA", "AMD", "AMZN", "NFLX", "BABA", "FB", "GOOGL", "COIN", "MRNA", "PFE"
                }
            }
        };

        /// <summary>
        /// Gets a comprehensive list of the most common stock symbols as a fallback
        /// </summary>
        public static List<string> CommonSymbols => new List<string>
        {
            // Major US Indices components (top components from S&P 500, Dow, and NASDAQ)
            "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "BRK.B", "JPM",
            "JNJ", "V", "PG", "UNH", "HD", "BAC", "MA", "XOM", "DIS", "CSCO", "VZ", "ADBE",
            "CRM", "NFLX", "CMCSA", "PEP", "INTC", "ABT", "KO", "MRK", "PFE", "TMO", "COST",
            "WMT", "CVX", "AVGO", "ACN", "DHR", "MCD", "LLY", "TXN", "NEE", "NKE", "PM", "T",
            "WFC", "BMY", "QCOM", "UPS", "AMD", "PYPL", "MS", "C", "SBUX", "AMGN", "LMT",
            "BA", "HON", "IBM", "GS", "INTU", "MMM", "CAT", "GE", "SPGI", "BKNG", "BLK", "AXP",
            
            // Popular ETFs
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "BND", "VGK",
            "IEFA", "AGG", "EFA", "LQD", "XLF", "XLE", "XLV", "XLK", "XLI", "XLU"
        };

        /// <summary>
        /// Load cached symbols from disk if available
        /// </summary>
        public static List<string> GetCachedSymbols()
        {
            if (_cachedSymbols != null && _cachedSymbols.Count > 0)
            {
                return _cachedSymbols;
            }

            try
            {
                if (File.Exists(CacheFilePath))
                {
                    var json = File.ReadAllText(CacheFilePath);
                    _cachedSymbols = JsonConvert.DeserializeObject<List<string>>(json);

                    if (_cachedSymbols != null && _cachedSymbols.Count > 0)
                    {
                        return _cachedSymbols;
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to read symbol cache: {ex.Message}");
            }

            // If cache loading failed, return the common symbols
            return CommonSymbols;
        }

        /// <summary>
        /// Save symbols to cache file
        /// </summary>
        public static void SaveSymbolsToCache(List<string> symbols)
        {
            if (symbols == null || symbols.Count == 0)
            {
                return;
            }

            try
            {
                _cachedSymbols = symbols;
                File.WriteAllText(CacheFilePath, JsonConvert.SerializeObject(symbols));
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to write symbol cache: {ex.Message}");
            }
        }

        /// <summary>
        /// Search for symbols that match the query
        /// </summary>
        public static List<string> SearchSymbols(string query, List<string> symbolSource, int maxResults = 10)
        {
            if (string.IsNullOrWhiteSpace(query) || symbolSource == null || symbolSource.Count == 0)
            {
                return new List<string>();
            }

            query = query.ToUpperInvariant().Trim();

            // First, find exact matches at the beginning
            var exactMatches = symbolSource
                .Where(s => s.StartsWith(query, StringComparison.OrdinalIgnoreCase))
                .OrderBy(s => s.Length) // Shorter symbols first
                .ThenBy(s => s)
                .Take(maxResults / 2);

            // Then add symbols that contain the query
            var containsMatches = symbolSource
                .Where(s => !s.StartsWith(query, StringComparison.OrdinalIgnoreCase) &&
                           s.Contains(query, StringComparison.OrdinalIgnoreCase))
                .OrderBy(s => s.Length)
                .ThenBy(s => s)
                .Take(maxResults / 2);

            return exactMatches.Concat(containsMatches).Take(maxResults).ToList();
        }
    }
}
