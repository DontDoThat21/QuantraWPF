using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Data;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for querying ML prediction data from the database.
    /// Used by Market Chat to provide AI-generated forecast context in conversations.
    /// Integrates with PredictionCacheService to leverage cached predictions (MarketChat story 3).
    /// </summary>
    public class PredictionDataService : IPredictionDataService
    {
        private readonly QuantraDbContext _context;
        private readonly ILogger<PredictionDataService> _logger;
        private readonly PredictionCacheService _cacheService;

        // Configuration constants for cache behavior
        private const string DefaultModelVersion = "v1.0";
        private static readonly TimeSpan CacheExpiryPeriod = TimeSpan.FromHours(1);
        private const int CacheWarmingDelayMs = 50;

        // Popular symbols for cache warming during market hours
        private static readonly string[] PopularSymbols = new[]
        {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ",
            "WMT", "PG", "MA", "UNH", "HD", "DIS", "BAC", "ADBE", "CRM", "NFLX"
        };

        /// <summary>
        /// Constructor for PredictionDataService with dependency injection.
        /// All dependencies are required and should be provided by the DI container.
        /// </summary>
        public PredictionDataService(QuantraDbContext context, ILogger<PredictionDataService> logger, PredictionCacheService cacheService)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _cacheService = cacheService ?? throw new ArgumentNullException(nameof(cacheService));
        }

        /// <inheritdoc/>
        public async Task<string> GetPredictionContextAsync(string symbol)
        {
            var result = await GetPredictionContextWithCacheAsync(symbol);
            return result?.Context;
        }

        /// <inheritdoc/>
        public async Task<PredictionContextResult> GetPredictionContextWithCacheAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return PredictionContextResult.Empty;
            }

            try
            {
                symbol = symbol.ToUpperInvariant().Trim();
                _logger?.LogInformation("Fetching prediction context with cache support for {Symbol}", symbol);

                // First, check the PredictionCache table for a cached prediction
                var cachedResult = await TryGetFromCacheAsync(symbol);
                if (cachedResult != null)
                {
                    _logger?.LogInformation("Cache HIT for {Symbol} - returning cached prediction from {Age} ago", 
                        symbol, cachedResult.CacheAge);
                    return cachedResult;
                }

                _logger?.LogInformation("Cache MISS for {Symbol} - fetching fresh prediction", symbol);

                // Cache miss - get fresh prediction from StockPredictions table
                var freshResult = await GetFreshPredictionAsync(symbol);
                
                // If we got a fresh prediction, cache it for future requests
                if (freshResult != null && !string.IsNullOrEmpty(freshResult.Context))
                {
                    await CachePredictionResultAsync(symbol, freshResult);
                }

                return freshResult ?? PredictionContextResult.Empty;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error fetching prediction context for {Symbol}", symbol);
                return PredictionContextResult.Empty;
            }
        }

        /// <inheritdoc/>
        public async Task<int> WarmCacheForSymbolsAsync(IEnumerable<string> symbols)
        {
            if (symbols == null)
            {
                symbols = PopularSymbols;
            }

            var symbolList = symbols.ToList();
            _logger?.LogInformation("Starting cache warming for {Count} symbols", symbolList.Count);

            int warmedCount = 0;

            foreach (var symbol in symbolList)
            {
                try
                {
                    var result = await GetPredictionContextWithCacheAsync(symbol);
                    if (result != null && !string.IsNullOrEmpty(result.Context))
                    {
                        warmedCount++;
                        _logger?.LogDebug("Cache warmed for {Symbol}", symbol);
                    }

                    // Configurable delay to avoid overwhelming the system
                    await Task.Delay(CacheWarmingDelayMs);
                }
                catch (Exception ex)
                {
                    _logger?.LogWarning(ex, "Failed to warm cache for {Symbol}", symbol);
                }
            }

            _logger?.LogInformation("Cache warming complete: {Warmed}/{Total} symbols", warmedCount, symbolList.Count);
            return warmedCount;
        }

        /// <summary>
        /// Gets popular symbols for cache warming
        /// </summary>
        public static IEnumerable<string> GetPopularSymbols() => PopularSymbols;

        /// <summary>
        /// Attempts to retrieve a cached prediction from the PredictionCache table
        /// </summary>
        private async Task<PredictionContextResult> TryGetFromCacheAsync(string symbol)
        {
            try
            {
                // Query the PredictionCache table for a valid cached entry
                var cachedEntry = await _context.PredictionCache
                    .AsNoTracking()
                    .Where(pc => pc.Symbol == symbol)
                    .OrderByDescending(pc => pc.CreatedAt)
                    .FirstOrDefaultAsync();

                if (cachedEntry == null)
                {
                    return null;
                }

                // Check if the cache entry is still valid using configurable expiry period
                var cacheAge = DateTime.Now - cachedEntry.CreatedAt;
                if (cacheAge > CacheExpiryPeriod)
                {
                    _logger?.LogDebug("Cache entry for {Symbol} is stale ({Age} old, expiry: {Expiry})", 
                        symbol, cacheAge, CacheExpiryPeriod);
                    return null;
                }

                // Update access count for metrics
                await UpdateCacheAccessMetricsAsync(cachedEntry.Id);

                // Build the context string from cached data
                var contextBuilder = new StringBuilder();
                contextBuilder.AppendLine($"ML Prediction Data for {symbol} (Cached):");
                contextBuilder.AppendLine($"- Predicted Action: {cachedEntry.PredictedAction ?? "N/A"}");
                contextBuilder.AppendLine($"- Confidence: {(cachedEntry.Confidence ?? 0):P0}");
                contextBuilder.AppendLine($"- Target Price: ${(cachedEntry.PredictedPrice ?? 0):F2}");
                contextBuilder.AppendLine($"- Prediction Date: {(cachedEntry.PredictionTimestamp ?? cachedEntry.CreatedAt):yyyy-MM-dd HH:mm}");
                contextBuilder.AppendLine($"- Model Version: {cachedEntry.ModelVersion}");

                return new PredictionContextResult
                {
                    Context = contextBuilder.ToString(),
                    IsCached = true,
                    CacheAge = cacheAge,
                    PredictionTimestamp = cachedEntry.PredictionTimestamp ?? cachedEntry.CreatedAt,
                    ModelVersion = cachedEntry.ModelVersion
                };
            }
            catch (Exception ex)
            {
                _logger?.LogWarning(ex, "Error checking prediction cache for {Symbol}", symbol);
                return null;
            }
        }

        /// <summary>
        /// Updates the access count and last accessed timestamp for a cache entry
        /// </summary>
        private async Task UpdateCacheAccessMetricsAsync(int cacheEntryId)
        {
            try
            {
                var entry = await _context.PredictionCache.FindAsync(cacheEntryId);
                if (entry != null)
                {
                    entry.AccessCount++;
                    entry.LastAccessedAt = DateTime.Now;
                    await _context.SaveChangesAsync();
                }
            }
            catch (Exception ex)
            {
                // Non-critical operation - just log and continue
                _logger?.LogDebug(ex, "Failed to update cache access metrics for entry {Id}", cacheEntryId);
            }
        }

        /// <summary>
        /// Gets a fresh prediction from the StockPredictions table
        /// </summary>
        private async Task<PredictionContextResult> GetFreshPredictionAsync(string symbol)
        {
            // Get the most recent prediction for the symbol
            var prediction = await _context.StockPredictions
                .AsNoTracking()
                .Where(p => p.Symbol == symbol)
                .OrderByDescending(p => p.CreatedDate)
                .FirstOrDefaultAsync();

            if (prediction == null)
            {
                _logger?.LogInformation("No predictions found for {Symbol}", symbol);
                return null;
            }

            // Get the indicators for this prediction
            var indicators = await _context.PredictionIndicators
                .AsNoTracking()
                .Where(i => i.PredictionId == prediction.Id)
                .ToListAsync();

            // Build the prediction context string
            var contextBuilder = new StringBuilder();
            contextBuilder.AppendLine($"ML Prediction Data for {symbol}:");
            contextBuilder.AppendLine($"- Predicted Action: {prediction.PredictedAction}");
            contextBuilder.AppendLine($"- Confidence: {prediction.Confidence:P0}");
            contextBuilder.AppendLine($"- Current Price: ${prediction.CurrentPrice:F2}");
            contextBuilder.AppendLine($"- Target Price: ${prediction.TargetPrice:F2}");
            contextBuilder.AppendLine($"- Potential Return: {prediction.PotentialReturn:P2}");
            contextBuilder.AppendLine($"- Prediction Date: {prediction.CreatedDate:yyyy-MM-dd HH:mm}");

            // Add indicator rationale if available
            if (indicators.Any())
            {
                contextBuilder.AppendLine();
                contextBuilder.AppendLine("Technical Indicators Used:");
                foreach (var indicator in indicators)
                {
                    contextBuilder.AppendLine($"- {indicator.IndicatorName}: {indicator.IndicatorValue:F4}");
                }

                // Add indicator interpretation guidance
                contextBuilder.AppendLine();
                contextBuilder.AppendLine("Indicator Interpretation:");
                AppendIndicatorInterpretation(contextBuilder, indicators);
            }

            _logger?.LogInformation("Successfully built fresh prediction context for {Symbol}", symbol);

            return new PredictionContextResult
            {
                Context = contextBuilder.ToString(),
                IsCached = false,
                CacheAge = null,
                PredictionTimestamp = prediction.CreatedDate,
                ModelVersion = DefaultModelVersion
            };
        }

        /// <summary>
        /// Caches a prediction result for future requests
        /// </summary>
        private async Task CachePredictionResultAsync(string symbol, PredictionContextResult result)
        {
            try
            {
                // Check if an entry already exists
                var existingEntry = await _context.PredictionCache
                    .FirstOrDefaultAsync(pc => pc.Symbol == symbol && pc.ModelVersion == DefaultModelVersion);

                if (existingEntry != null)
                {
                    // Update existing entry
                    existingEntry.PredictionTimestamp = result.PredictionTimestamp;
                    existingEntry.CreatedAt = DateTime.Now;
                    existingEntry.AccessCount = 0;
                    existingEntry.LastAccessedAt = null;
                }
                else
                {
                    // Create new cache entry
                    var newEntry = new Data.Entities.PredictionCacheEntity
                    {
                        Symbol = symbol,
                        ModelVersion = DefaultModelVersion,
                        InputDataHash = GenerateSimpleHash(symbol + DateTime.Today.ToString("yyyyMMdd")),
                        PredictionTimestamp = result.PredictionTimestamp,
                        CreatedAt = DateTime.Now,
                        AccessCount = 0,
                        LastAccessedAt = null
                    };
                    _context.PredictionCache.Add(newEntry);
                }

                await _context.SaveChangesAsync();
                _logger?.LogDebug("Cached prediction for {Symbol}", symbol);
            }
            catch (Exception ex)
            {
                // Non-critical operation - just log and continue
                _logger?.LogWarning(ex, "Failed to cache prediction for {Symbol}", symbol);
            }
        }

        /// <summary>
        /// Generates a simple hash for cache key purposes.
        /// Uses SHA256.HashData for better performance (single allocation).
        /// </summary>
        private static string GenerateSimpleHash(string input)
        {
            var bytes = System.Text.Encoding.UTF8.GetBytes(input);
            var hashBytes = System.Security.Cryptography.SHA256.HashData(bytes);
            return Convert.ToHexString(hashBytes).Substring(0, 32);
        }

        /// <summary>
        /// Appends interpretation guidance for common technical indicators
        /// </summary>
        private void AppendIndicatorInterpretation(StringBuilder builder, List<Data.Entities.PredictionIndicatorEntity> indicators)
        {
            foreach (var indicator in indicators)
            {
                string interpretation = GetIndicatorInterpretation(indicator.IndicatorName, indicator.IndicatorValue);
                if (!string.IsNullOrEmpty(interpretation))
                {
                    builder.AppendLine($"  • {interpretation}");
                }
            }
        }

        /// <summary>
        /// Gets human-readable interpretation for a specific indicator
        /// </summary>
        private string GetIndicatorInterpretation(string indicatorName, double value)
        {
            var name = indicatorName.ToUpperInvariant();

            if (name.Contains("RSI"))
            {
                if (value < 30) return $"RSI at {value:F1} indicates oversold conditions (bullish signal)";
                if (value > 70) return $"RSI at {value:F1} indicates overbought conditions (bearish signal)";
                return $"RSI at {value:F1} is in neutral territory";
            }

            if (name.Contains("MACD"))
            {
                if (value > 0) return $"MACD at {value:F4} is positive (bullish momentum)";
                if (value < 0) return $"MACD at {value:F4} is negative (bearish momentum)";
                return $"MACD at {value:F4} is near zero (neutral)";
            }

            if (name.Contains("ADX"))
            {
                if (value > 25) return $"ADX at {value:F1} indicates strong trend";
                return $"ADX at {value:F1} indicates weak trend";
            }

            if (name.Contains("BOLLINGER") || name.Contains("BB"))
            {
                return $"Bollinger Band position at {value:F2}";
            }

            if (name.Contains("VWAP"))
            {
                return $"VWAP at ${value:F2}";
            }

            if (name.Contains("EMA") || name.Contains("SMA"))
            {
                return $"{indicatorName} at ${value:F2}";
            }

            if (name.Contains("STOCH"))
            {
                if (value < 20) return $"Stochastic at {value:F1} indicates oversold";
                if (value > 80) return $"Stochastic at {value:F1} indicates overbought";
                return $"Stochastic at {value:F1}";
            }

            // Default: return the indicator name and value
            return null;
        }
    }
}
