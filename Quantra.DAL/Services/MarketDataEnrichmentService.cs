using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.DAL.Services.Interfaces;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for enriching market data with historical context for AI-powered market chat.
    /// Queries HistoricalDataResponse/HistoricalPrice tables and provides moving averages,
    /// volatility, volume patterns, and price ranges to enhance AI prompts.
    /// </summary>
    public class MarketDataEnrichmentService : IMarketDataEnrichmentService
    {
        private readonly IStockDataCacheService _stockDataCacheService;
        private readonly ILogger<MarketDataEnrichmentService> _logger;
        
        // Cache for historical context to minimize database calls
        private readonly ConcurrentDictionary<string, CachedHistoricalContext> _contextCache;
        private static readonly TimeSpan CacheExpiration = TimeSpan.FromMinutes(15);

        /// <summary>
        /// Constructor for MarketDataEnrichmentService
        /// </summary>
        public MarketDataEnrichmentService(
            IStockDataCacheService stockDataCacheService,
            ILogger<MarketDataEnrichmentService> logger)
        {
            _stockDataCacheService = stockDataCacheService ?? throw new ArgumentNullException(nameof(stockDataCacheService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _contextCache = new ConcurrentDictionary<string, CachedHistoricalContext>(StringComparer.OrdinalIgnoreCase);
        }

        /// <inheritdoc/>
        public async Task<string> GetHistoricalContextAsync(string symbol, int days = 60)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return string.Empty;
            }

            try
            {
                symbol = symbol.ToUpperInvariant().Trim();
                
                // Check cache first
                var cacheKey = $"{symbol}_{days}";
                if (_contextCache.TryGetValue(cacheKey, out var cached) && !cached.IsExpired)
                {
                    _logger.LogDebug("Retrieved historical context for {Symbol} from cache", symbol);
                    return cached.Context;
                }

                _logger.LogInformation("Fetching historical context for {Symbol} ({Days} days)", symbol, days);

                // Fetch historical data
                var historicalData = await GetHistoricalDataAsync(symbol, days);
                
                if (historicalData == null || historicalData.Count == 0)
                {
                    _logger.LogWarning("No historical data found for {Symbol}", symbol);
                    return string.Empty;
                }

                // Build the historical context
                var context = BuildHistoricalContext(symbol, historicalData);

                // Cache the result
                _contextCache[cacheKey] = new CachedHistoricalContext(context);

                return context;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching historical context for {Symbol}", symbol);
                return string.Empty;
            }
        }

        /// <inheritdoc/>
        public async Task<string> GetHistoricalContextSummaryAsync(string symbol)
        {
            // Use a shorter timeframe for summary
            return await GetHistoricalContextAsync(symbol, 30);
        }

        /// <inheritdoc/>
        public void ClearCacheForSymbol(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return;
            }

            symbol = symbol.ToUpperInvariant().Trim();
            
            // Remove all cache entries for this symbol
            var keysToRemove = _contextCache.Keys.Where(k => k.StartsWith(symbol + "_")).ToList();
            foreach (var key in keysToRemove)
            {
                _contextCache.TryRemove(key, out _);
            }

            _logger.LogInformation("Cleared historical context cache for {Symbol}", symbol);
        }

        /// <inheritdoc/>
        public void ClearAllCache()
        {
            _contextCache.Clear();
            _logger.LogInformation("Cleared all historical context cache");
        }

        /// <summary>
        /// Fetches historical price data for a symbol
        /// </summary>
        private async Task<List<HistoricalPrice>> GetHistoricalDataAsync(string symbol, int days)
        {
            try
            {
                // Determine appropriate timeframe based on days requested
                string timeframe = days switch
                {
                    <= 30 => "1mo",
                    <= 90 => "3mo",
                    <= 180 => "6mo",
                    _ => "1y"
                };

                var data = await _stockDataCacheService.GetStockDataAsync(symbol, timeframe, "1d", forceRefresh: false);
                
                // Filter to requested number of days
                if (data != null && data.Count > days)
                {
                    data = data.OrderByDescending(d => d.Date).Take(days).OrderBy(d => d.Date).ToList();
                }

                return data ?? new List<HistoricalPrice>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching historical data for {Symbol}", symbol);
                return new List<HistoricalPrice>();
            }
        }

        /// <summary>
        /// Builds a formatted historical context string from price data
        /// </summary>
        private string BuildHistoricalContext(string symbol, List<HistoricalPrice> data)
        {
            var sb = new StringBuilder();
            
            sb.AppendLine($"Historical Data Context for {symbol}:");
            sb.AppendLine($"- Data Period: {data.First().Date:yyyy-MM-dd} to {data.Last().Date:yyyy-MM-dd} ({data.Count} days)");

            // Current and recent prices
            var latestPrice = data.Last().Close;
            var previousClose = data.Count > 1 ? data[data.Count - 2].Close : latestPrice;
            var dailyChange = latestPrice - previousClose;
            var dailyChangePercent = previousClose != 0 ? (dailyChange / previousClose) * 100 : 0;

            sb.AppendLine($"- Latest Close: ${latestPrice:F2}");
            sb.AppendLine($"- Daily Change: {(dailyChange >= 0 ? "+" : "")}{dailyChange:F2} ({(dailyChangePercent >= 0 ? "+" : "")}{dailyChangePercent:F2}%)");

            // Moving Averages
            var ma5 = CalculateMovingAverage(data, 5);
            var ma20 = CalculateMovingAverage(data, 20);
            var ma50 = CalculateMovingAverage(data, 50);

            sb.AppendLine();
            sb.AppendLine("Moving Averages:");
            if (ma5.HasValue)
            {
                var ma5Diff = ((latestPrice - ma5.Value) / ma5.Value) * 100;
                sb.AppendLine($"- 5-day MA: ${ma5.Value:F2} (Price is {(ma5Diff >= 0 ? "+" : "")}{ma5Diff:F2}% relative)");
            }
            if (ma20.HasValue)
            {
                var ma20Diff = ((latestPrice - ma20.Value) / ma20.Value) * 100;
                sb.AppendLine($"- 20-day MA: ${ma20.Value:F2} (Price is {(ma20Diff >= 0 ? "+" : "")}{ma20Diff:F2}% relative)");
            }
            if (ma50.HasValue)
            {
                var ma50Diff = ((latestPrice - ma50.Value) / ma50.Value) * 100;
                sb.AppendLine($"- 50-day MA: ${ma50.Value:F2} (Price is {(ma50Diff >= 0 ? "+" : "")}{ma50Diff:F2}% relative)");
            }

            // Price Range
            var highestPrice = data.Max(d => d.High);
            var lowestPrice = data.Min(d => d.Low);
            var priceRange = highestPrice - lowestPrice;
            var positionInRange = priceRange != 0 ? ((latestPrice - lowestPrice) / priceRange) * 100 : 50;

            sb.AppendLine();
            sb.AppendLine("Price Range:");
            sb.AppendLine($"- Period High: ${highestPrice:F2}");
            sb.AppendLine($"- Period Low: ${lowestPrice:F2}");
            sb.AppendLine($"- Price Range: ${priceRange:F2}");
            sb.AppendLine($"- Current Position in Range: {positionInRange:F1}% (0%=Low, 100%=High)");

            // Volatility
            var volatility = CalculateVolatility(data);
            var atr = CalculateATR(data, 14);

            sb.AppendLine();
            sb.AppendLine("Volatility Metrics:");
            sb.AppendLine($"- Historical Volatility (Std Dev of Returns): {volatility:F2}%");
            if (atr.HasValue)
            {
                sb.AppendLine($"- Average True Range (14-day): ${atr.Value:F2}");
            }

            // Volume Analysis
            var avgVolume = data.Average(d => (double)d.Volume);
            var recentAvgVolume = data.TakeLast(5).Average(d => (double)d.Volume);
            var volumeRatio = avgVolume != 0 ? (recentAvgVolume / avgVolume) * 100 : 100;

            sb.AppendLine();
            sb.AppendLine("Volume Analysis:");
            sb.AppendLine($"- Average Daily Volume: {FormatVolume(avgVolume)}");
            sb.AppendLine($"- Recent 5-day Avg Volume: {FormatVolume(recentAvgVolume)}");
            sb.AppendLine($"- Volume Trend: {(volumeRatio > 110 ? "Above Average" : volumeRatio < 90 ? "Below Average" : "Normal")} ({volumeRatio:F1}% of average)");

            // Trend Analysis
            var trend = AnalyzeTrend(data, ma5, ma20, ma50);
            sb.AppendLine();
            sb.AppendLine($"Trend Analysis: {trend}");

            return sb.ToString();
        }

        /// <summary>
        /// Calculates simple moving average
        /// </summary>
        private double? CalculateMovingAverage(List<HistoricalPrice> data, int period)
        {
            if (data.Count < period)
            {
                return null;
            }

            return data.TakeLast(period).Average(d => d.Close);
        }

        /// <summary>
        /// Calculates historical volatility (standard deviation of daily returns)
        /// </summary>
        private double CalculateVolatility(List<HistoricalPrice> data)
        {
            if (data.Count < 2)
            {
                return 0;
            }

            var returns = new List<double>();
            for (int i = 1; i < data.Count; i++)
            {
                if (data[i - 1].Close != 0)
                {
                    var dailyReturn = (data[i].Close - data[i - 1].Close) / data[i - 1].Close * 100;
                    returns.Add(dailyReturn);
                }
            }

            if (returns.Count < 2)
            {
                return 0;
            }

            var mean = returns.Average();
            var sumOfSquares = returns.Sum(r => Math.Pow(r - mean, 2));
            // Use sample standard deviation (n-1) for unbiased estimation
            return Math.Sqrt(sumOfSquares / (returns.Count - 1));
        }

        /// <summary>
        /// Calculates Average True Range (ATR)
        /// </summary>
        private double? CalculateATR(List<HistoricalPrice> data, int period)
        {
            if (data.Count < period + 1)
            {
                return null;
            }

            var trueRanges = new List<double>();
            for (int i = 1; i < data.Count; i++)
            {
                var high = data[i].High;
                var low = data[i].Low;
                var prevClose = data[i - 1].Close;

                var tr = Math.Max(
                    high - low,
                    Math.Max(
                        Math.Abs(high - prevClose),
                        Math.Abs(low - prevClose)
                    )
                );
                trueRanges.Add(tr);
            }

            return trueRanges.TakeLast(period).Average();
        }

        /// <summary>
        /// Analyzes the overall trend based on price and moving averages
        /// </summary>
        private string AnalyzeTrend(List<HistoricalPrice> data, double? ma5, double? ma20, double? ma50)
        {
            var latestPrice = data.Last().Close;
            var trendSignals = new List<string>();

            // Price vs MAs
            if (ma5.HasValue && ma20.HasValue)
            {
                if (latestPrice > ma5.Value && latestPrice > ma20.Value)
                {
                    trendSignals.Add("Price above short-term MAs (bullish)");
                }
                else if (latestPrice < ma5.Value && latestPrice < ma20.Value)
                {
                    trendSignals.Add("Price below short-term MAs (bearish)");
                }
            }

            // MA crossovers
            if (ma5.HasValue && ma20.HasValue)
            {
                if (ma5.Value > ma20.Value)
                {
                    trendSignals.Add("5-day MA above 20-day MA (golden cross pattern)");
                }
                else
                {
                    trendSignals.Add("5-day MA below 20-day MA (death cross pattern)");
                }
            }

            // Long-term trend
            if (ma50.HasValue)
            {
                if (latestPrice > ma50.Value)
                {
                    trendSignals.Add("Price above 50-day MA (long-term uptrend)");
                }
                else
                {
                    trendSignals.Add("Price below 50-day MA (long-term downtrend)");
                }
            }

            // Recent momentum
            if (data.Count >= 5)
            {
                var recentPrices = data.TakeLast(5).Select(d => d.Close).ToList();
                var momentum = recentPrices.Last() - recentPrices.First();
                var momentumPercent = recentPrices.First() != 0 ? (momentum / recentPrices.First()) * 100 : 0;

                if (momentumPercent > 2)
                {
                    trendSignals.Add($"Strong recent momentum (+{momentumPercent:F1}% in 5 days)");
                }
                else if (momentumPercent < -2)
                {
                    trendSignals.Add($"Weak recent momentum ({momentumPercent:F1}% in 5 days)");
                }
                else
                {
                    trendSignals.Add($"Sideways momentum ({momentumPercent:F1}% in 5 days)");
                }
            }

            return string.Join("; ", trendSignals);
        }

        /// <summary>
        /// Formats volume for display
        /// </summary>
        private string FormatVolume(double volume)
        {
            if (volume >= 1_000_000_000)
            {
                return $"{volume / 1_000_000_000:F2}B";
            }
            if (volume >= 1_000_000)
            {
                return $"{volume / 1_000_000:F2}M";
            }
            if (volume >= 1_000)
            {
                return $"{volume / 1_000:F2}K";
            }
            return $"{volume:F0}";
        }

        /// <summary>
        /// Cached historical context with expiration
        /// </summary>
        private class CachedHistoricalContext
        {
            public string Context { get; }
            public DateTime CachedAt { get; }
            public bool IsExpired => DateTime.UtcNow - CachedAt > CacheExpiration;

            public CachedHistoricalContext(string context)
            {
                Context = context;
                CachedAt = DateTime.UtcNow;
            }
        }
    }
}
