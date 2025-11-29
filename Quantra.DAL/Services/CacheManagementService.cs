using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing prediction cache through Market Chat commands (MarketChat Story 10).
    /// Provides methods for clearing cache, viewing statistics, and managing cache lifecycle.
    /// Integrates with PredictionCacheService for cache operations.
    /// </summary>
    public class CacheManagementService : ICacheManagementService
    {
        private readonly LoggingService _loggingService;
        private readonly PredictionCacheService _predictionCacheService;
        private readonly TimeSpan _cacheValidityPeriod;

        // Compiled regex patterns for cache command detection
        private static readonly Regex ClearCachePattern = new Regex(
            @"\b(clear|delete|remove|purge|flush)\s+(cache|cached|prediction\s*cache)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex ShowStatsPattern = new Regex(
            @"\b(show|display|get|view|list)\s+(cache\s*)?(stats|statistics|metrics|info|information|status)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex CacheInfoPattern = new Regex(
            @"\b(cache\s*info|cache\s*status|cache\s*details)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex SymbolPattern = new Regex(
            @"\b(for|of)\s+([A-Z]{1,5})\b|\$([A-Z]{1,5})\b|\b([A-Z]{1,5})\s+(cache|cached)\b",
            RegexOptions.Compiled);

        private static readonly Regex AllCachePattern = new Regex(
            @"\b(all|entire|complete|full)\s+(cache|cached)\b|\bcache\s+(all|entirely)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex ExpiredCachePattern = new Regex(
            @"\b(expired|stale|old)\s+(cache|entries)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private static readonly Regex RefreshCachePattern = new Regex(
            @"\b(refresh|update|renew)\s+(cache|prediction)\b",
            RegexOptions.IgnoreCase | RegexOptions.Compiled);

        /// <summary>
        /// Constructor for CacheManagementService
        /// </summary>
        /// <param name="loggingService">Logging service for error tracking</param>
        /// <param name="predictionCacheService">Prediction cache service for cache operations</param>
        /// <param name="cacheValidityPeriod">Optional cache validity period (default 1 hour)</param>
        public CacheManagementService(
            LoggingService loggingService,
            PredictionCacheService predictionCacheService = null,
            TimeSpan? cacheValidityPeriod = null)
        {
            _loggingService = loggingService;
            _predictionCacheService = predictionCacheService;
            _cacheValidityPeriod = cacheValidityPeriod ?? TimeSpan.FromHours(1);
        }

        /// <inheritdoc/>
        public async Task<CacheManagementResult> ClearCacheAsync(string symbol)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(symbol))
                {
                    return new CacheManagementResult
                    {
                        Success = false,
                        Message = "Please specify a valid stock symbol to clear cache for.",
                        OperationType = "Clear"
                    };
                }

                var normalizedSymbol = symbol.ToUpperInvariant().Trim();
                var entriesRemoved = 0;

                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    var cacheEntries = await dbContext.PredictionCache
                        .Where(pc => pc.Symbol == normalizedSymbol)
                        .ToListAsync();

                    entriesRemoved = cacheEntries.Count;

                    if (entriesRemoved > 0)
                    {
                        dbContext.PredictionCache.RemoveRange(cacheEntries);
                        await dbContext.SaveChangesAsync();
                    }
                }

                _loggingService?.Log("Info", $"Cleared {entriesRemoved} cache entries for {normalizedSymbol}");

                var message = entriesRemoved > 0
                    ? $"Successfully cleared {entriesRemoved} cache {(entriesRemoved == 1 ? "entry" : "entries")} for {normalizedSymbol}."
                    : $"No cache entries found for {normalizedSymbol}.";

                return new CacheManagementResult
                {
                    Success = true,
                    Message = message,
                    OperationType = "Clear",
                    Symbol = normalizedSymbol,
                    EntriesAffected = entriesRemoved,
                    MarkdownContent = $"✅ **Cache Cleared**\n\n{message}"
                };
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Error clearing cache for {symbol}", ex.ToString());
                return new CacheManagementResult
                {
                    Success = false,
                    Message = $"Error clearing cache for {symbol}: {ex.Message}",
                    OperationType = "Clear",
                    Symbol = symbol
                };
            }
        }

        /// <inheritdoc/>
        public async Task<CacheManagementResult> ClearExpiredCacheAsync()
        {
            try
            {
                var expiryDate = DateTime.Now - _cacheValidityPeriod;
                var entriesRemoved = 0;

                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    var expiredEntries = await dbContext.PredictionCache
                        .Where(pc => pc.CreatedAt < expiryDate)
                        .ToListAsync();

                    entriesRemoved = expiredEntries.Count;

                    if (entriesRemoved > 0)
                    {
                        dbContext.PredictionCache.RemoveRange(expiredEntries);
                        await dbContext.SaveChangesAsync();
                    }
                }

                // Also call the PredictionCacheService's method if available
                _predictionCacheService?.ClearExpiredCache();

                _loggingService?.Log("Info", $"Cleared {entriesRemoved} expired cache entries");

                var message = entriesRemoved > 0
                    ? $"Successfully cleared {entriesRemoved} expired cache {(entriesRemoved == 1 ? "entry" : "entries")}."
                    : "No expired cache entries found.";

                return new CacheManagementResult
                {
                    Success = true,
                    Message = message,
                    OperationType = "ClearExpired",
                    EntriesAffected = entriesRemoved,
                    MarkdownContent = $"🧹 **Expired Cache Cleared**\n\n{message}"
                };
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Error clearing expired cache", ex.ToString());
                return new CacheManagementResult
                {
                    Success = false,
                    Message = $"Error clearing expired cache: {ex.Message}",
                    OperationType = "ClearExpired"
                };
            }
        }

        /// <inheritdoc/>
        public async Task<CacheManagementResult> ClearAllCacheAsync(bool confirmed = false)
        {
            if (!confirmed)
            {
                return new CacheManagementResult
                {
                    Success = false,
                    Message = "⚠️ **Warning**: This will delete ALL cached predictions. " +
                              "Please confirm by saying \"Yes, clear all cache\" or \"Confirm clear all cache\".",
                    OperationType = "ClearAll",
                    RequiresConfirmation = true,
                    MarkdownContent = "⚠️ **Confirmation Required**\n\n" +
                                     "This will delete **ALL** cached predictions from the database.\n\n" +
                                     "To confirm, please respond with:\n" +
                                     "- \"Yes, clear all cache\"\n" +
                                     "- \"Confirm clear all cache\"\n\n" +
                                     "_This action cannot be undone._"
                };
            }

            try
            {
                var entriesRemoved = 0;

                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    var allEntries = await dbContext.PredictionCache.ToListAsync();
                    entriesRemoved = allEntries.Count;

                    if (entriesRemoved > 0)
                    {
                        dbContext.PredictionCache.RemoveRange(allEntries);
                        await dbContext.SaveChangesAsync();
                    }
                }

                _loggingService?.Log("Warning", $"Cleared ALL {entriesRemoved} cache entries by user request");

                return new CacheManagementResult
                {
                    Success = true,
                    Message = $"Successfully cleared all {entriesRemoved} cache entries.",
                    OperationType = "ClearAll",
                    EntriesAffected = entriesRemoved,
                    MarkdownContent = $"✅ **All Cache Cleared**\n\n" +
                                     $"Successfully removed **{entriesRemoved}** cached prediction {(entriesRemoved == 1 ? "entry" : "entries")} from the database."
                };
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Error clearing all cache", ex.ToString());
                return new CacheManagementResult
                {
                    Success = false,
                    Message = $"Error clearing all cache: {ex.Message}",
                    OperationType = "ClearAll"
                };
            }
        }

        /// <inheritdoc/>
        public async Task<CacheStatistics> GetCacheStatsAsync()
        {
            try
            {
                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    var now = DateTime.Now;
                    var expiryDate = now - _cacheValidityPeriod;

                    var allEntries = await dbContext.PredictionCache.ToListAsync();

                    if (allEntries.Count == 0)
                    {
                        return new CacheStatistics
                        {
                            TotalEntries = 0,
                            ValidEntries = 0,
                            ExpiredEntries = 0,
                            TotalCacheHits = 0,
                            CacheValidityPeriod = _cacheValidityPeriod,
                            UniqueSymbols = 0
                        };
                    }

                    var validEntries = allEntries.Where(e => e.CreatedAt >= expiryDate).ToList();
                    var expiredEntries = allEntries.Where(e => e.CreatedAt < expiryDate).ToList();
                    var uniqueSymbols = allEntries.Select(e => e.Symbol).Distinct().Count();
                    var totalHits = allEntries.Sum(e => e.AccessCount);
                    var ages = allEntries.Select(e => now - e.CreatedAt).ToList();

                    // Estimate storage: roughly 200 bytes per entry
                    var estimatedStorage = allEntries.Count * 200L;

                    var avgConfidence = allEntries
                        .Where(e => e.Confidence.HasValue)
                        .Select(e => e.Confidence.Value)
                        .DefaultIfEmpty(0)
                        .Average();

                    var latestOperation = allEntries.Max(e => e.LastAccessedAt ?? e.CreatedAt);

                    return new CacheStatistics
                    {
                        TotalEntries = allEntries.Count,
                        ValidEntries = validEntries.Count,
                        ExpiredEntries = expiredEntries.Count,
                        TotalCacheHits = totalHits,
                        AverageAge = TimeSpan.FromTicks((long)ages.Average(a => a.Ticks)),
                        OldestEntryAge = ages.Max(),
                        NewestEntryAge = ages.Min(),
                        CacheValidityPeriod = _cacheValidityPeriod,
                        EstimatedStorageBytes = estimatedStorage,
                        UniqueSymbols = uniqueSymbols,
                        AverageConfidence = avgConfidence,
                        LastCacheOperation = latestOperation
                    };
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", "Error getting cache statistics", ex.ToString());
                return new CacheStatistics
                {
                    TotalEntries = 0,
                    CacheValidityPeriod = _cacheValidityPeriod
                };
            }
        }

        /// <inheritdoc/>
        public async Task<SymbolCacheInfo> GetSymbolCacheInfoAsync(string symbol)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(symbol))
                {
                    return null;
                }

                var normalizedSymbol = symbol.ToUpperInvariant().Trim();

                var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                optionsBuilder.UseSqlite("Data Source=Quantra.db;Journal Mode=WAL;Busy Timeout=30000;");

                using (var dbContext = new QuantraDbContext(optionsBuilder.Options))
                {
                    var entries = await dbContext.PredictionCache
                        .Where(pc => pc.Symbol == normalizedSymbol)
                        .OrderByDescending(pc => pc.CreatedAt)
                        .ToListAsync();

                    if (entries.Count == 0)
                    {
                        return new SymbolCacheInfo
                        {
                            Symbol = normalizedSymbol,
                            EntryCount = 0,
                            IsValid = false,
                            Recommendation = $"No cache exists for {normalizedSymbol}. Run a prediction to create cache."
                        };
                    }

                    var latestEntry = entries.First();
                    var now = DateTime.Now;
                    var age = now - latestEntry.CreatedAt;
                    var isValid = age <= _cacheValidityPeriod;
                    var totalAccess = entries.Sum(e => e.AccessCount);
                    var lastAccessed = entries.Max(e => e.LastAccessedAt);

                    var info = new SymbolCacheInfo
                    {
                        Symbol = normalizedSymbol,
                        EntryCount = entries.Count,
                        Age = age,
                        IsValid = isValid,
                        AccessCount = totalAccess,
                        LastAccessed = lastAccessed,
                        ModelVersion = latestEntry.ModelVersion,
                        PredictedAction = latestEntry.PredictedAction,
                        Confidence = latestEntry.Confidence,
                        TargetPrice = latestEntry.PredictedPrice
                    };

                    info.Recommendation = GenerateRecommendation(info);

                    return info;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Error", $"Error getting cache info for {symbol}", ex.ToString());
                return new SymbolCacheInfo
                {
                    Symbol = symbol,
                    EntryCount = 0,
                    IsValid = false,
                    Recommendation = "Error retrieving cache information."
                };
            }
        }

        /// <inheritdoc/>
        public string FormatStatsAsMarkdown(CacheStatistics stats)
        {
            if (stats == null)
            {
                return "❌ Unable to retrieve cache statistics.";
            }

            var sb = new StringBuilder();
            sb.AppendLine("📊 **Prediction Cache Statistics**");
            sb.AppendLine();
            sb.AppendLine("| Metric | Value |");
            sb.AppendLine("|--------|-------|");
            sb.AppendLine($"| Total Entries | {stats.TotalEntries} |");
            sb.AppendLine($"| Valid Entries | {stats.ValidEntries} |");
            sb.AppendLine($"| Expired Entries | {stats.ExpiredEntries} |");
            sb.AppendLine($"| Unique Symbols | {stats.UniqueSymbols} |");
            sb.AppendLine($"| Total Cache Hits | {stats.TotalCacheHits} |");
            sb.AppendLine($"| Average Confidence | {stats.AverageConfidence:P0} |");
            sb.AppendLine($"| Cache Validity Period | {FormatTimeSpan(stats.CacheValidityPeriod)} |");
            sb.AppendLine();

            if (stats.TotalEntries > 0)
            {
                sb.AppendLine("**Cache Age Distribution:**");
                sb.AppendLine($"- Average Age: {FormatTimeSpan(stats.AverageAge)}");
                sb.AppendLine($"- Oldest Entry: {FormatTimeSpan(stats.OldestEntryAge)}");
                sb.AppendLine($"- Newest Entry: {FormatTimeSpan(stats.NewestEntryAge)}");
                sb.AppendLine();

                sb.AppendLine("**Storage:**");
                sb.AppendLine($"- Estimated Size: {FormatBytes(stats.EstimatedStorageBytes)}");
                if (stats.LastCacheOperation.HasValue)
                {
                    sb.AppendLine($"- Last Activity: {stats.LastCacheOperation:yyyy-MM-dd HH:mm:ss}");
                }
                sb.AppendLine();
            }

            // Recommendations
            if (stats.ExpiredEntries > 0)
            {
                sb.AppendLine($"💡 **Recommendation:** {stats.ExpiredEntries} expired entries can be cleared. Say \"Clear expired cache\" to clean up.");
            }

            if (stats.TotalEntries > 100)
            {
                sb.AppendLine("💡 **Recommendation:** Consider clearing old cache entries to improve performance.");
            }

            return sb.ToString();
        }

        /// <inheritdoc/>
        public string FormatSymbolInfoAsMarkdown(SymbolCacheInfo info)
        {
            if (info == null)
            {
                return "❌ Unable to retrieve cache information.";
            }

            if (info.EntryCount == 0)
            {
                return $"❌ **No Cache Found for {info.Symbol}**\n\n{info.Recommendation}";
            }

            var sb = new StringBuilder();
            sb.AppendLine($"📈 **Cache Information for {info.Symbol}**");
            sb.AppendLine();
            sb.AppendLine("| Property | Value |");
            sb.AppendLine("|----------|-------|");
            sb.AppendLine($"| Cache Entries | {info.EntryCount} |");
            sb.AppendLine($"| Cache Age | {FormatTimeSpan(info.Age)} |");
            sb.AppendLine($"| Status | {(info.IsValid ? "✅ Valid" : "⚠️ Expired")} |");
            sb.AppendLine($"| Access Count | {info.AccessCount} |");

            if (info.LastAccessed.HasValue)
            {
                sb.AppendLine($"| Last Accessed | {info.LastAccessed:yyyy-MM-dd HH:mm:ss} |");
            }

            sb.AppendLine();

            if (!string.IsNullOrEmpty(info.PredictedAction))
            {
                sb.AppendLine("**Cached Prediction:**");
                sb.AppendLine($"- Action: {info.PredictedAction}");
                if (info.Confidence.HasValue)
                {
                    sb.AppendLine($"- Confidence: {info.Confidence:P0}");
                }
                if (info.TargetPrice.HasValue)
                {
                    sb.AppendLine($"- Target Price: ${info.TargetPrice:F2}");
                }
                if (!string.IsNullOrEmpty(info.ModelVersion))
                {
                    sb.AppendLine($"- Model Version: {info.ModelVersion}");
                }
                sb.AppendLine();
            }

            if (!string.IsNullOrEmpty(info.Recommendation))
            {
                sb.AppendLine($"💡 **Recommendation:** {info.Recommendation}");
            }

            return sb.ToString();
        }

        /// <inheritdoc/>
        public bool IsCacheManagementRequest(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return false;
            }

            // Check for clear cache commands
            if (ClearCachePattern.IsMatch(message))
            {
                return true;
            }

            // Check for show stats commands
            if (ShowStatsPattern.IsMatch(message))
            {
                return true;
            }

            // Check for cache info commands
            if (CacheInfoPattern.IsMatch(message))
            {
                return true;
            }

            // Check for refresh cache commands
            if (RefreshCachePattern.IsMatch(message))
            {
                return true;
            }

            // Check for explicit cache-related phrases
            var lowerMessage = message.ToLowerInvariant();
            return lowerMessage.Contains("cache statistics") ||
                   lowerMessage.Contains("cache hit") ||
                   lowerMessage.Contains("cache expiry") ||
                   lowerMessage.Contains("cache storage") ||
                   lowerMessage.Contains("prediction cache") ||
                   lowerMessage.Contains("clear cache") ||
                   lowerMessage.Contains("show cache") ||
                   lowerMessage.Contains("cache status");
        }

        /// <inheritdoc/>
        public string ExtractOperationType(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return "unknown";
            }

            var lowerMessage = message.ToLowerInvariant();

            // Check for confirmation of clear all
            if (lowerMessage.Contains("yes") && lowerMessage.Contains("clear all") ||
                lowerMessage.Contains("confirm") && lowerMessage.Contains("clear all"))
            {
                return "clearall_confirmed";
            }

            // Check for clear operations
            if (ClearCachePattern.IsMatch(message))
            {
                if (AllCachePattern.IsMatch(message))
                {
                    return "clearall";
                }
                if (ExpiredCachePattern.IsMatch(message))
                {
                    return "clearexpired";
                }
                // Check if a specific symbol is mentioned
                if (SymbolPattern.IsMatch(message))
                {
                    return "clearsymbol";
                }
                return "clearexpired"; // Default to clearing expired
            }

            // Check for stats/info operations
            if (ShowStatsPattern.IsMatch(message) || CacheInfoPattern.IsMatch(message))
            {
                // Check if asking about a specific symbol
                if (SymbolPattern.IsMatch(message))
                {
                    return "symbolinfo";
                }
                return "stats";
            }

            // Check for refresh operations
            if (RefreshCachePattern.IsMatch(message))
            {
                return "refresh";
            }

            return "stats"; // Default to showing stats
        }

        /// <inheritdoc/>
        public string ExtractSymbol(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
            {
                return null;
            }

            // Try to match "for SYMBOL" or "$SYMBOL" patterns
            var match = SymbolPattern.Match(message);
            if (match.Success)
            {
                // Find the first non-empty group
                for (int i = 2; i < match.Groups.Count; i++)
                {
                    if (match.Groups[i].Success && !string.IsNullOrEmpty(match.Groups[i].Value))
                    {
                        var symbol = match.Groups[i].Value.ToUpperInvariant();
                        // Filter out common words
                        if (!IsCommonWord(symbol))
                        {
                            return symbol;
                        }
                    }
                }
            }

            // Also look for standalone uppercase symbols in the message
            var standaloneMatch = Regex.Match(message, @"\b([A-Z]{1,5})\b");
            if (standaloneMatch.Success)
            {
                var symbol = standaloneMatch.Groups[1].Value;
                if (!IsCommonWord(symbol))
                {
                    return symbol;
                }
            }

            return null;
        }

        /// <inheritdoc/>
        public string GenerateRecommendation(SymbolCacheInfo info)
        {
            if (info == null)
            {
                return null;
            }

            if (info.EntryCount == 0)
            {
                return $"No cache exists for {info.Symbol}. Run a prediction to create cache.";
            }

            var ageHours = info.Age.TotalHours;

            if (!info.IsValid)
            {
                return $"{info.Symbol} cache is {FormatTimeSpan(info.Age)} old and expired. Suggest refresh.";
            }

            if (ageHours > _cacheValidityPeriod.TotalHours * 0.75)
            {
                return $"{info.Symbol} cache is {FormatTimeSpan(info.Age)} old and will expire soon. Consider refreshing.";
            }

            if (ageHours > _cacheValidityPeriod.TotalHours * 0.5)
            {
                return $"{info.Symbol} cache is {FormatTimeSpan(info.Age)} old. Still valid but may benefit from refresh.";
            }

            if (info.AccessCount > 10)
            {
                return $"{info.Symbol} cache is fresh ({FormatTimeSpan(info.Age)} old) and frequently accessed ({info.AccessCount} hits).";
            }

            return $"{info.Symbol} cache is fresh ({FormatTimeSpan(info.Age)} old).";
        }

        /// <summary>
        /// Checks if a word is a common English word (not a stock symbol)
        /// </summary>
        private static bool IsCommonWord(string word)
        {
            var commonWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "I", "A", "AN", "THE", "IN", "ON", "AT", "TO", "FOR", "OF", "AND", "OR", "IS", "IT",
                "BE", "AS", "BY", "IF", "DO", "GO", "SO", "NO", "UP", "MY", "ME", "WE", "US", "AM",
                "CAN", "ALL", "NEW", "ONE", "TWO", "NOW", "HOW", "WHY", "WHAT", "WHEN", "WHO",
                "NOT", "BUT", "OUT", "HAS", "HAD", "GET", "GOT", "MAY", "SAY", "SEE", "SET",
                "YES", "SHOW", "CLEAR", "DELETE", "REMOVE", "CACHE", "STATS", "INFO"
            };
            return commonWords.Contains(word);
        }

        /// <summary>
        /// Formats a TimeSpan into a human-readable string
        /// </summary>
        private static string FormatTimeSpan(TimeSpan timeSpan)
        {
            if (timeSpan.TotalDays >= 1)
            {
                var days = (int)timeSpan.TotalDays;
                var hours = timeSpan.Hours;
                return hours > 0 ? $"{days}d {hours}h" : $"{days}d";
            }
            if (timeSpan.TotalHours >= 1)
            {
                var hours = (int)timeSpan.TotalHours;
                var minutes = timeSpan.Minutes;
                return minutes > 0 ? $"{hours}h {minutes}m" : $"{hours}h";
            }
            if (timeSpan.TotalMinutes >= 1)
            {
                return $"{(int)timeSpan.TotalMinutes}m";
            }
            return $"{(int)timeSpan.TotalSeconds}s";
        }

        /// <summary>
        /// Formats bytes into a human-readable string
        /// </summary>
        private static string FormatBytes(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB" };
            double len = bytes;
            int order = 0;
            while (len >= 1024 && order < sizes.Length - 1)
            {
                order++;
                len = len / 1024;
            }
            return $"{len:F1} {sizes[order]}";
        }
    }
}
