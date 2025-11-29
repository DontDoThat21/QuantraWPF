using System;
using System.Threading.Tasks;

namespace Quantra.DAL.Services.Interfaces
{
    /// <summary>
    /// Result of a cache management operation for display in Market Chat (MarketChat Story 10).
    /// </summary>
    public class CacheManagementResult
    {
        /// <summary>
        /// Whether the operation was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Human-readable message describing the result
        /// </summary>
        public string Message { get; set; }

        /// <summary>
        /// The operation that was performed (Clear, Stats, Refresh, etc.)
        /// </summary>
        public string OperationType { get; set; }

        /// <summary>
        /// Symbol affected by the operation (if applicable)
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Number of cache entries affected
        /// </summary>
        public int EntriesAffected { get; set; }

        /// <summary>
        /// Whether the operation required confirmation
        /// </summary>
        public bool RequiresConfirmation { get; set; }

        /// <summary>
        /// Formatted markdown content for display in chat
        /// </summary>
        public string MarkdownContent { get; set; }

        /// <summary>
        /// Recommendations based on cache analysis
        /// </summary>
        public string Recommendation { get; set; }
    }

    /// <summary>
    /// Statistics about the prediction cache (MarketChat Story 10).
    /// </summary>
    public class CacheStatistics
    {
        /// <summary>
        /// Total number of cached prediction entries
        /// </summary>
        public int TotalEntries { get; set; }

        /// <summary>
        /// Number of entries that are still valid (not expired)
        /// </summary>
        public int ValidEntries { get; set; }

        /// <summary>
        /// Number of entries that have expired
        /// </summary>
        public int ExpiredEntries { get; set; }

        /// <summary>
        /// Total number of cache hits (access count)
        /// </summary>
        public long TotalCacheHits { get; set; }

        /// <summary>
        /// Average age of cached entries
        /// </summary>
        public TimeSpan AverageAge { get; set; }

        /// <summary>
        /// Oldest cache entry age
        /// </summary>
        public TimeSpan OldestEntryAge { get; set; }

        /// <summary>
        /// Newest cache entry age
        /// </summary>
        public TimeSpan NewestEntryAge { get; set; }

        /// <summary>
        /// Cache validity period configured in the system
        /// </summary>
        public TimeSpan CacheValidityPeriod { get; set; }

        /// <summary>
        /// Estimated storage usage in bytes
        /// </summary>
        public long EstimatedStorageBytes { get; set; }

        /// <summary>
        /// Number of unique symbols in cache
        /// </summary>
        public int UniqueSymbols { get; set; }

        /// <summary>
        /// Average confidence of cached predictions
        /// </summary>
        public double AverageConfidence { get; set; }

        /// <summary>
        /// Timestamp of the last cache operation
        /// </summary>
        public DateTime? LastCacheOperation { get; set; }
    }

    /// <summary>
    /// Cache information for a specific symbol (MarketChat Story 10).
    /// </summary>
    public class SymbolCacheInfo
    {
        /// <summary>
        /// Stock symbol
        /// </summary>
        public string Symbol { get; set; }

        /// <summary>
        /// Number of cache entries for this symbol
        /// </summary>
        public int EntryCount { get; set; }

        /// <summary>
        /// Age of the most recent cache entry
        /// </summary>
        public TimeSpan Age { get; set; }

        /// <summary>
        /// Whether the cache entry is still valid
        /// </summary>
        public bool IsValid { get; set; }

        /// <summary>
        /// Number of times this symbol's cache has been accessed
        /// </summary>
        public int AccessCount { get; set; }

        /// <summary>
        /// Last time this cache entry was accessed
        /// </summary>
        public DateTime? LastAccessed { get; set; }

        /// <summary>
        /// Model version used for the cached prediction
        /// </summary>
        public string ModelVersion { get; set; }

        /// <summary>
        /// Predicted action from the cached prediction
        /// </summary>
        public string PredictedAction { get; set; }

        /// <summary>
        /// Confidence score from the cached prediction
        /// </summary>
        public double? Confidence { get; set; }

        /// <summary>
        /// Target price from the cached prediction
        /// </summary>
        public double? TargetPrice { get; set; }

        /// <summary>
        /// Recommendation based on cache state (e.g., "suggest refresh")
        /// </summary>
        public string Recommendation { get; set; }
    }

    /// <summary>
    /// Service for managing prediction cache through Market Chat commands (MarketChat Story 10).
    /// Provides methods for clearing cache, viewing statistics, and managing cache lifecycle.
    /// </summary>
    public interface ICacheManagementService
    {
        /// <summary>
        /// Clears cache entries for a specific symbol.
        /// </summary>
        /// <param name="symbol">Stock symbol to clear cache for</param>
        /// <returns>Result of the clear operation</returns>
        Task<CacheManagementResult> ClearCacheAsync(string symbol);

        /// <summary>
        /// Clears all expired cache entries from the database.
        /// </summary>
        /// <returns>Result of the clear operation</returns>
        Task<CacheManagementResult> ClearExpiredCacheAsync();

        /// <summary>
        /// Clears all cache entries (requires authorization).
        /// </summary>
        /// <param name="confirmed">Whether the operation has been confirmed</param>
        /// <returns>Result of the clear operation</returns>
        Task<CacheManagementResult> ClearAllCacheAsync(bool confirmed = false);

        /// <summary>
        /// Gets overall cache statistics.
        /// </summary>
        /// <returns>Cache statistics</returns>
        Task<CacheStatistics> GetCacheStatsAsync();

        /// <summary>
        /// Gets cache information for a specific symbol.
        /// </summary>
        /// <param name="symbol">Stock symbol to get cache info for</param>
        /// <returns>Cache information for the symbol</returns>
        Task<SymbolCacheInfo> GetSymbolCacheInfoAsync(string symbol);

        /// <summary>
        /// Formats cache statistics as a markdown table for display in chat.
        /// </summary>
        /// <param name="stats">Cache statistics to format</param>
        /// <returns>Markdown formatted string</returns>
        string FormatStatsAsMarkdown(CacheStatistics stats);

        /// <summary>
        /// Formats symbol cache info as markdown for display in chat.
        /// </summary>
        /// <param name="info">Symbol cache info to format</param>
        /// <returns>Markdown formatted string</returns>
        string FormatSymbolInfoAsMarkdown(SymbolCacheInfo info);

        /// <summary>
        /// Determines if a user message is a cache management request.
        /// </summary>
        /// <param name="message">User message to analyze</param>
        /// <returns>True if the message is a cache management request</returns>
        bool IsCacheManagementRequest(string message);

        /// <summary>
        /// Extracts the cache operation type from a user message.
        /// </summary>
        /// <param name="message">User message to analyze</param>
        /// <returns>Operation type (clear, stats, info, refresh)</returns>
        string ExtractOperationType(string message);

        /// <summary>
        /// Extracts the target symbol from a cache management message, if any.
        /// </summary>
        /// <param name="message">User message to analyze</param>
        /// <returns>Symbol or null if no specific symbol is mentioned</returns>
        string ExtractSymbol(string message);

        /// <summary>
        /// Generates a recommendation based on cache state for a symbol.
        /// </summary>
        /// <param name="info">Symbol cache info</param>
        /// <returns>Recommendation string</returns>
        string GenerateRecommendation(SymbolCacheInfo info);
    }
}
