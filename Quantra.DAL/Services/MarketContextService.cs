using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to fetch and manage market context data for TFT model.
    /// Provides S&amp;P 500, VIX, Treasury yield, sector ETF, and market breadth data.
    /// Data is cached for 15 minutes during market hours to reduce API calls.
    /// </summary>
    public class MarketContextService
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;

        // In-memory cache for market context data with 15-minute expiry
        private static readonly Dictionary<string, (object Data, DateTime Timestamp)> _marketContextCache
            = new Dictionary<string, (object, DateTime)>();
        private static readonly object _cacheLock = new object();
        private const int CacheExpirationMinutes = 15;

        public MarketContextService(AlphaVantageService alphaVantageService, LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _loggingService = loggingService;
        }

        /// <summary>
        /// Get current S&amp;P 500 (SPY) price and 1-day return.
        /// </summary>
        /// <returns>Tuple of (price, returnPct)</returns>
        public async Task<(double price, double returnPct)> GetSP500DataAsync()
        {
            const string cacheKey = "SP500_Data";
            
            // Check cache first
            var cached = GetCachedValue<(double, double)>(cacheKey);
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                var quote = await _alphaVantageService.GetQuoteDataAsync("SPY");
                var historicalData = await _alphaVantageService.GetDailyData("SPY", "compact");

                double currentPrice = quote?.Price ?? 0;
                double previousClose = historicalData?
                    .OrderByDescending(h => h.Date)
                    .Skip(1)
                    .FirstOrDefault()?.Close ?? currentPrice;

                double returnPct = previousClose > 0 ? (currentPrice - previousClose) / previousClose : 0;

                var result = (currentPrice, returnPct);
                CacheValue(cacheKey, result);

                return result;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get S&P 500 data: {ex.Message}");
                return (0, 0);
            }
        }

        /// <summary>
        /// Get VIX (volatility index) current level.
        /// </summary>
        /// <returns>Current VIX value</returns>
        public async Task<double> GetVIXAsync()
        {
            const string cacheKey = "VIX_Data";
            
            // Check cache first
            var cached = GetCachedValue<double>(cacheKey);
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                // VIX is handled by AlphaVantageService symbol normalization (^VIX)
                var quote = await _alphaVantageService.GetQuoteDataAsync("VIX");
                double vix = quote?.Price ?? 0;

                CacheValue(cacheKey, vix);
                return vix;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get VIX data: {ex.Message}");
                return 0;
            }
        }

        /// <summary>
        /// Get 10-Year Treasury Yield.
        /// Uses Alpha Vantage TREASURY_YIELD function.
        /// </summary>
        /// <returns>10-Year Treasury Yield percentage</returns>
        public async Task<double> GetTreasuryYield10YAsync()
        {
            const string cacheKey = "TreasuryYield_10Y";
            
            // Check cache first
            var cached = GetCachedValue<double>(cacheKey);
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                double treasuryYield = await _alphaVantageService.GetTreasuryYieldAsync("10year");
                CacheValue(cacheKey, treasuryYield);
                return treasuryYield;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get 10-Year Treasury Yield: {ex.Message}");
                return 0;
            }
        }

        /// <summary>
        /// Get sector ETF price for context.
        /// </summary>
        /// <param name="sector">Sector name (e.g., "Technology", "Healthcare")</param>
        /// <returns>Sector ETF price</returns>
        public async Task<double> GetSectorETFPriceAsync(string sector)
        {
            if (string.IsNullOrWhiteSpace(sector))
            {
                return 0;
            }

            string cacheKey = $"SectorETF_{sector}";
            
            // Check cache first
            var cached = GetCachedValue<double>(cacheKey);
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                string etfSymbol = MapSectorToETF(sector);
                var quote = await _alphaVantageService.GetQuoteDataAsync(etfSymbol);
                double price = quote?.Price ?? 0;

                CacheValue(cacheKey, price);
                return price;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get sector ETF price for {sector}: {ex.Message}");
                return 0;
            }
        }

        /// <summary>
        /// Get sector ETF return (1-day percentage change) for context.
        /// </summary>
        /// <param name="sector">Sector name (e.g., "Technology", "Healthcare")</param>
        /// <returns>Tuple of (price, returnPct)</returns>
        public async Task<(double price, double returnPct)> GetSectorETFDataAsync(string sector)
        {
            if (string.IsNullOrWhiteSpace(sector))
            {
                return (0, 0);
            }

            string cacheKey = $"SectorETFData_{sector}";
            
            // Check cache first
            var cached = GetCachedValue<(double, double)>(cacheKey);
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                string etfSymbol = MapSectorToETF(sector);
                var quote = await _alphaVantageService.GetQuoteDataAsync(etfSymbol);
                var historicalData = await _alphaVantageService.GetDailyData(etfSymbol, "compact");

                double currentPrice = quote?.Price ?? 0;
                double previousClose = historicalData?
                    .OrderByDescending(h => h.Date)
                    .Skip(1)
                    .FirstOrDefault()?.Close ?? currentPrice;

                double returnPct = previousClose > 0 ? (currentPrice - previousClose) / previousClose : 0;

                var result = (currentPrice, returnPct);
                CacheValue(cacheKey, result);

                return result;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get sector ETF data for {sector}: {ex.Message}");
                return (0, 0);
            }
        }

        /// <summary>
        /// Maps sector name to corresponding sector ETF symbol.
        /// </summary>
        /// <param name="sector">Sector name</param>
        /// <returns>ETF symbol for the sector</returns>
        public string MapSectorToETF(string sector)
        {
            if (string.IsNullOrWhiteSpace(sector))
            {
                return "SPY"; // Default to S&P 500
            }

            // Normalize sector name for matching
            string normalizedSector = sector.ToUpperInvariant().Trim();

            return normalizedSector switch
            {
                "TECHNOLOGY" or "INFORMATION TECHNOLOGY" or "TECH" => "XLK",
                "HEALTHCARE" or "HEALTH CARE" => "XLV",
                "FINANCIAL SERVICES" or "FINANCIALS" or "FINANCIAL" => "XLF",
                "CONSUMER DISCRETIONARY" or "CONSUMER CYCLICAL" => "XLY",
                "CONSUMER STAPLES" or "CONSUMER DEFENSIVE" => "XLP",
                "ENERGY" => "XLE",
                "UTILITIES" => "XLU",
                "REAL ESTATE" => "XLRE",
                "MATERIALS" or "BASIC MATERIALS" => "XLB",
                "INDUSTRIALS" or "INDUSTRIAL" => "XLI",
                "COMMUNICATION SERVICES" or "TELECOMMUNICATIONS" or "TELECOM" => "XLC",
                _ => "SPY" // Default to S&P 500 if sector unknown
            };
        }

        /// <summary>
        /// Get sector name from sector code.
        /// Inverse of AlphaVantageService.GetSectorCode().
        /// </summary>
        /// <param name="sectorCode">Numeric sector code</param>
        /// <returns>Sector name</returns>
        public string GetSectorName(int sectorCode)
        {
            return sectorCode switch
            {
                0 => "Technology",
                1 => "Healthcare",
                2 => "Financial",
                3 => "Consumer Discretionary",
                4 => "Consumer Staples",
                5 => "Industrials",
                6 => "Energy",
                7 => "Materials",
                8 => "Real Estate",
                9 => "Utilities",
                10 => "Communication Services",
                _ => "Unknown"
            };
        }

        /// <summary>
        /// Get market breadth (advance/decline ratio).
        /// Calculates from market internals using top movers data.
        /// </summary>
        /// <returns>Market breadth ratio (&gt;1 = more advancing than declining)</returns>
        public async Task<double> GetMarketBreadthAsync()
        {
            const string cacheKey = "MarketBreadth";
            
            // Check cache first
            var cached = GetCachedValue<double>(cacheKey);
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                // Get top movers to estimate market breadth
                var topMovers = await _alphaVantageService.GetTopMoversAsync();
                
                if (topMovers == null)
                {
                    return 1.0; // Neutral default
                }

                // Count advancing vs declining from top movers data
                int advancingCount = topMovers.TopGainers?.Count ?? 0;
                int decliningCount = topMovers.TopLosers?.Count ?? 0;

                // Calculate breadth ratio
                double breadth;
                if (decliningCount > 0)
                {
                    breadth = (double)advancingCount / decliningCount;
                }
                else if (advancingCount > 0)
                {
                    breadth = 2.0; // Strong bullish if no decliners
                }
                else
                {
                    breadth = 1.0; // Neutral if no data
                }

                CacheValue(cacheKey, breadth);
                return breadth;
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get market breadth: {ex.Message}");
                return 1.0; // Neutral default
            }
        }

        /// <summary>
        /// Get volatility regime classification based on VIX level.
        /// </summary>
        /// <param name="vix">Current VIX value</param>
        /// <returns>Volatility regime code (0=Low, 1=Normal, 2=Elevated, 3=High)</returns>
        public int GetVolatilityRegime(double vix)
        {
            return vix switch
            {
                < 15 => 0,  // Low volatility
                < 20 => 1,  // Normal volatility
                < 30 => 2,  // Elevated volatility
                _ => 3      // High volatility
            };
        }

        /// <summary>
        /// Get volatility regime description based on VIX level.
        /// </summary>
        /// <param name="vix">Current VIX value</param>
        /// <returns>Volatility regime description</returns>
        public string GetVolatilityRegimeDescription(double vix)
        {
            return vix switch
            {
                < 15 => "Low Volatility",
                < 20 => "Normal",
                < 30 => "Elevated Volatility",
                _ => "High Volatility"
            };
        }

        /// <summary>
        /// Calculate relative strength of a stock vs its sector.
        /// </summary>
        /// <param name="stockPrice">Current stock price</param>
        /// <param name="sectorETFPrice">Current sector ETF price</param>
        /// <returns>Relative strength ratio</returns>
        public double CalculateRelativeStrengthVsSector(double stockPrice, double sectorETFPrice)
        {
            if (stockPrice <= 0 || sectorETFPrice <= 0)
            {
                return 1.0; // Neutral if prices are invalid
            }

            return stockPrice / sectorETFPrice;
        }

        /// <summary>
        /// Get all market context data in a single call.
        /// Useful for building the complete market context indicators dictionary.
        /// </summary>
        /// <returns>Dictionary of market context indicators</returns>
        public async Task<Dictionary<string, double>> GetAllMarketContextAsync()
        {
            var indicators = new Dictionary<string, double>();

            try
            {
                // S&P 500 context
                var (sp500Price, sp500Return) = await GetSP500DataAsync();
                indicators["SP500_Price"] = sp500Price;
                indicators["SP500_Return"] = sp500Return;
                indicators["SP500_Direction"] = sp500Return > 0 ? 1.0 : -1.0;

                // VIX (volatility regime)
                double vix = await GetVIXAsync();
                indicators["VIX"] = vix;
                indicators["VolatilityRegime"] = GetVolatilityRegime(vix);

                // Treasury yield (interest rate environment)
                double treasuryYield = await GetTreasuryYield10YAsync();
                indicators["TreasuryYield_10Y"] = treasuryYield;

                // Market breadth
                double marketBreadth = await GetMarketBreadthAsync();
                indicators["MarketBreadth"] = marketBreadth;
                indicators["IsBullishBreadth"] = marketBreadth > 1.0 ? 1.0 : 0.0;

                _loggingService?.Log("Info", $"Market context: SPY={sp500Price:F2} ({sp500Return:P2}), VIX={vix:F2}, 10Y={treasuryYield:F2}%");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get complete market context: {ex.Message}");
            }

            return indicators;
        }

        /// <summary>
        /// Get sector-specific market context for a stock.
        /// </summary>
        /// <param name="sectorCode">Sector code from company overview</param>
        /// <param name="currentStockPrice">Current stock price</param>
        /// <returns>Dictionary of sector-specific indicators</returns>
        public async Task<Dictionary<string, double>> GetSectorContextAsync(int sectorCode, double currentStockPrice)
        {
            var indicators = new Dictionary<string, double>();

            try
            {
                string sector = GetSectorName(sectorCode);
                if (sector == "Unknown")
                {
                    return indicators;
                }

                // Get sector ETF data
                var (sectorETFPrice, sectorETFReturn) = await GetSectorETFDataAsync(sector);
                indicators["SectorETF_Price"] = sectorETFPrice;
                indicators["SectorETF_Return"] = sectorETFReturn;

                // Calculate relative strength vs sector
                if (currentStockPrice > 0 && sectorETFPrice > 0)
                {
                    indicators["RelativeStrengthVsSector"] = CalculateRelativeStrengthVsSector(currentStockPrice, sectorETFPrice);
                }

                _loggingService?.Log("Debug", $"Sector context for {sector}: ETF Price={sectorETFPrice:F2}, Return={sectorETFReturn:P2}");
            }
            catch (Exception ex)
            {
                _loggingService?.Log("Warning", $"Failed to get sector context: {ex.Message}");
            }

            return indicators;
        }

        #region Cache Helper Methods

        /// <summary>
        /// Get cached value if available and not expired.
        /// </summary>
        private T? GetCachedValue<T>(string key) where T : struct
        {
            lock (_cacheLock)
            {
                if (_marketContextCache.TryGetValue(key, out var cacheEntry))
                {
                    if (DateTime.UtcNow - cacheEntry.Timestamp < TimeSpan.FromMinutes(CacheExpirationMinutes))
                    {
                        _loggingService?.Log("Debug", $"Using cached market context for {key}");
                        return (T)cacheEntry.Data;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Cache a value with current timestamp.
        /// </summary>
        private void CacheValue<T>(string key, T value)
        {
            lock (_cacheLock)
            {
                _marketContextCache[key] = (value, DateTime.UtcNow);
            }
        }

        /// <summary>
        /// Clear all cached market context data.
        /// </summary>
        public void ClearCache()
        {
            lock (_cacheLock)
            {
                _marketContextCache.Clear();
                _loggingService?.Log("Info", "Market context cache cleared");
            }
        }

        /// <summary>
        /// Clear cached data for a specific key.
        /// </summary>
        /// <param name="key">Cache key to clear</param>
        public void ClearCache(string key)
        {
            if (string.IsNullOrEmpty(key))
            {
                return;
            }

            lock (_cacheLock)
            {
                if (_marketContextCache.Remove(key))
                {
                    _loggingService?.Log("Debug", $"Cleared market context cache for {key}");
                }
            }
        }

        #endregion
    }
}
