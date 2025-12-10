using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Quantra.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Quantra.DAL.Services;

namespace Quantra.DAL.Services
{
    public class AlphaVantageService : IAlphaVantageService
    {
        private readonly HttpClient _client;
        private readonly string _apiKey;
        private readonly SemaphoreSlim _apiSemaphore;
        private readonly IUserSettingsService _userSettingsService;
        private readonly LoggingService _loggingService;

        // Standard API rate limits
        private const int StandardApiCallsPerMinute = 75;
        private const int PremiumApiCallsPerMinute = 600; // Premium tier rate limit (can be adjusted based on plan)
        private const string StockCacheKey = "stock_symbols_cache";

        // Market cap thresholds for categorization (in dollars)
        private const decimal SmallCapMaxThreshold = 2_000_000_000m;        // $2 billion
        private const decimal MidCapMaxThreshold = 10_000_000_000m;         // $10 billion
        private const decimal LargeCapMaxThreshold = 200_000_000_000m;      // $200 billion

        // Current rate limit - will be determined based on API key type one day
        private int _maxApiCallsPerMinute;

        public static int ApiCallLimit => Instance?._maxApiCallsPerMinute ?? StandardApiCallsPerMinute;

        // Property to check if using premium API
        public bool IsPremiumKey => IsPremiumApiKey(_apiKey);

        // Singleton pattern for easy access
        private static AlphaVantageService Instance { get; set; }

        // Cache for fundamental data with timestamps
        private readonly Dictionary<string, (double Value, DateTime Timestamp)> _fundamentalDataCache = new Dictionary<string, (double, DateTime)>();
        private readonly object _cacheLock = new object();

        // Cache for company overview data with 7-day expiry for TFT static metadata features
        private readonly Dictionary<string, (CompanyOverview Overview, DateTime Timestamp)> _companyOverviewCache = new Dictionary<string, (CompanyOverview, DateTime)>();
        private readonly object _companyOverviewCacheLock = new object();
        private const int CompanyOverviewCacheDays = 7;

        public AlphaVantageService(IUserSettingsService userSettingsService, LoggingService loggingService)
        {
            _userSettingsService = userSettingsService;
            _loggingService = loggingService;
            _client = new HttpClient
            {
                BaseAddress = new Uri("https://www.alphavantage.co/")
            };
            _apiKey = GetApiKey();
            _apiSemaphore = new SemaphoreSlim(1, 1);

            // Load API rate limit from user settings
            var settings = _userSettingsService.GetUserSettings();
            _maxApiCallsPerMinute = settings?.AlphaVantageApiCallsPerMinute ?? StandardApiCallsPerMinute;

            Instance = this;
        }

        /// <summary>
        /// Updates the API rate limit from current user settings
        /// </summary>
        public void UpdateApiRateLimitFromSettings()
        {
            var settings = _userSettingsService.GetUserSettings();
            _maxApiCallsPerMinute = settings?.AlphaVantageApiCallsPerMinute ?? StandardApiCallsPerMinute;
        }

        /// <summary>
        /// Determines if the API key is a premium key
        /// </summary>
        /// <param name="apiKey">Alpha Vantage API key to check</param>
        /// <returns>True if premium, false otherwise</returns>
        private bool IsPremiumApiKey(string apiKey)
        {
            // This is a placeholder - implement your actual detection logic
            // For example, you might have a configuration setting, or check against a list of premium keys
            // For now, we'll just check if the API key has a "PREMIUM_" prefix
            return !string.IsNullOrEmpty(apiKey) &&
                   (apiKey.StartsWith("PREMIUM_") ||
                    Environment.GetEnvironmentVariable("ALPHA_VANTAGE_PREMIUM") == "true");
        }

        public int GetCurrentDbApiCallCount()
        {
            return GetAlphaVantageApiUsageCount(DateTime.UtcNow);
        }

        public int GetAlphaVantageApiUsageCount(DateTime utcNow)
        {
            // TODO: Implement API usage tracking if needed
            return 0;
        }

        public void LogApiUsage()
        {
            LogApiUsage(null, null);
        }

        public void LogApiUsage(string endpoint, string parameters)
        {
            //DatabaseMonolith.LogAlphaVantageApiUsage(endpoint, parameters);
        }

        public async Task<T> SendWithSlidingWindowAsync<T>(string functionName, Dictionary<string, string> parameters)
        {
            await WaitForApiLimit();

            var paramString = string.Join("&", parameters.Select(kv => $"{kv.Key}={kv.Value}"));
            var endpoint = $"query?function={functionName}&{paramString}&apikey={_apiKey}";
            await LogApiCall(functionName, paramString);

            var response = await _client.GetAsync(endpoint);
            var content = await response.Content.ReadAsStringAsync();

            // If T is string, return the raw content directly
            if (typeof(T) == typeof(string))
            {
                object result = content;
                return (T)result;
            }

            // Defensive: If the response is not valid JSON for T, return default or throw a more helpful error
            try
            {
                return JsonConvert.DeserializeObject<T>(content);
            }
            catch (JsonException ex)
            {
                // Optionally log the error and the content for debugging
                //DatabaseMonolith.Log("Error", $"Failed to deserialize AlphaVantage response for {functionName}", $"Content: {content}\nException: {ex}");
                // Otherwise, return default
                return default;
            }
        }

        /// <summary>
        /// Normalizes symbol for AlphaVantage API calls, handling special cases like VIX
        /// </summary>
        /// <param name="symbol">The symbol to normalize</param>
        /// <returns>The normalized symbol for API calls</returns>
        private string NormalizeSymbol(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return symbol;

            // Handle VIX special case - AlphaVantage expects ^VIX format
            if (symbol.Equals("VIX", StringComparison.OrdinalIgnoreCase))
                return "^VIX";

            return symbol;
        }

        public async Task<QuoteData> GetQuoteDataAsync(string symbol)
        {
            try
            {
                // Set global loading state for API calls
                GlobalLoadingStateService.SetLoadingState(true);

                // Normalize symbol for API call
                string normalizedSymbol = NormalizeSymbol(symbol);

                await WaitForApiLimit();
                var endpoint = $"query?function=GLOBAL_QUOTE&symbol={normalizedSymbol}&apikey={_apiKey}";
                await LogApiCall("GLOBAL_QUOTE", normalizedSymbol);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    if (data["Global Quote"] is JObject quote)
                    {
                        QuoteData quoteData = new QuoteData
                        {
                            Symbol = quote["01. symbol"]?.ToString() ?? "",
                            Price = TryParseDouble(quote["05. price"]),
                            Change = TryParseDouble(quote["09. change"]),
                            ChangePercent = TryParsePercentage(quote["10. change percent"]),
                            DayHigh = TryParseDouble(quote["03. high"]),
                            DayLow = TryParseDouble(quote["04. low"]),
                            Volume = TryParseDouble(quote["06. volume"]),
                            Date = TryParseDateTime(quote["07. latest trading day"]),
                            LastUpdated = DateTime.Now,
                            LastAccessed = DateTime.Now,
                            MarketCap = 0 // Will be populated separately if needed
                        };

                        // Fetch RSI and P/E Ratio for grid display
                        try
                        {
                            quoteData.RSI = await GetRSI(quoteData.Symbol);
                        }
                        catch
                        {
                            quoteData.RSI = 0; // Default value if RSI fetch fails
                        }

                        try
                        {
                            double? peRatio = await GetPERatioAsync(quoteData.Symbol);
                            quoteData.PERatio = peRatio ?? 0; // Default value if P/E fetch fails
                        }
                        catch
                        {
                            quoteData.PERatio = 0; // Default value if P/E fetch fails
                        }

                        return quoteData;
                    }
                }

                return null;
            }
            finally
            {
                // Clear global loading state when API call completes
                GlobalLoadingStateService.SetLoadingState(false);
            }
        }

        public async Task<double> GetQuoteData(string symbol, string interval = "1min")
        {
            var quote = await GetQuoteDataAsync(symbol);
            return quote?.Price ?? 0;
        }

        public async Task<List<string>> GetAllStockSymbols()
        {
            // TODO: Implement caching via UserSettingsService if needed
            // var cachedSymbols = _userSettingsService.GetUserPreference(StockCacheKey, null);

            await WaitForApiLimit();
            var endpoint = $"query?function=LISTING_STATUS&apikey={_apiKey}";
            await LogApiCall("LISTING_STATUS", null);

            var response = await _client.GetAsync(endpoint);
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var symbols = content.Split('\n')
                    .Skip(1) // Skip header row
                    .Where(line => !string.IsNullOrWhiteSpace(line))
                    .Select(line => line.Split(',')[0])
                    .ToList();

                // Add VIX to the symbol list since it's not included in regular listings
                if (!symbols.Contains("VIX"))
                {
                    symbols.Add("VIX");
                }

                // Cache the symbols
                CacheSymbols(symbols);
                return symbols;
            }

            // Return VIX as a fallback if API fails
            return new List<string> { "VIX" };
        }

        public void CacheSymbols(List<string> symbols)
        {
            if (symbols == null || !symbols.Any())
                return;

            try
            {
                // Store in UserPreferences table with timestamp
                string symbolsJson = JsonConvert.SerializeObject(symbols);
                _userSettingsService.SaveUserPreference(StockCacheKey, symbolsJson);
                _loggingService.Log("Info", $"Cached {symbols.Count} symbols to database");
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Failed to cache symbols");
            }
        }

        public async Task<double> GetRSI(string symbol, string interval = "1min")
        {
            // Check cache first
            var cached = GetCachedFundamentalData(symbol, $"RSI_{interval}", 1); // 1 hour cache for RSI
            if (cached.HasValue)
                return cached.Value;

            // Calculate RSI internally using historical data from database first
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 3)
                    return 50; // Need at least 3 data points for any meaningful RSI calculation

                var closingPrices = historicalData.Select(h => h.Close).ToList();

                // Use adaptive period based on available data
                // Standard RSI uses 14 periods, but we can calculate with fewer if needed
                int rsiPeriod = Math.Min(14, closingPrices.Count - 1);
                var rsiValues = CalculateRSIInternal(closingPrices, rsiPeriod);

                var latestRsi = rsiValues.LastOrDefault(r => !double.IsNaN(r));
                var result = double.IsNaN(latestRsi) ? 50 : latestRsi;

                // Cache the result
                CacheFundamentalData(symbol, $"RSI_{interval}", result);
                return result;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate RSI for {symbol}", ex.ToString());
                return 50; // Neutral default
            }
        }

        public async Task<double> GetLatestADX(string symbol, string interval = "1min")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 30) // Need enough data for ADX calculation
                    return 25; // Neutral default when insufficient data

                var highs = historicalData.Select(h => h.High).ToList();
                var lows = historicalData.Select(h => h.Low).ToList();
                var closes = historicalData.Select(h => h.Close).ToList();

                var adxValues = CalculateADXInternal(highs, lows, closes, 14);
                var latestAdx = adxValues.LastOrDefault(a => !double.IsNaN(a));
                return double.IsNaN(latestAdx) ? 25 : latestAdx;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate ADX for {symbol}", ex.ToString());
                return 25; // Neutral default
            }
        }

        public async Task<double> GetATR(string symbol, string interval = "1min")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 15) // Need enough data for ATR calculation
                    return 1.0; // Default value when insufficient data

                var highs = historicalData.Select(h => h.High).ToList();
                var lows = historicalData.Select(h => h.Low).ToList();
                var closes = historicalData.Select(h => h.Close).ToList();

                var atrValues = CalculateATRInternal(highs, lows, closes, 14);
                var latestAtr = atrValues.LastOrDefault(a => !double.IsNaN(a));
                return double.IsNaN(latestAtr) ? 1.0 : latestAtr;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate ATR for {symbol}", ex.ToString());
                return 1.0; // Default value
            }
        }

        public async Task<double> GetMomentumScore(string symbol, string interval = "1min")
        {
            // Calculate basic momentum using price change from cached data
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 12)
                    return 0; // Neutral default

                var closingPrices = historicalData.Select(h => h.Close).ToList();

                // Simple momentum calculation: (current - previous) / previous * 100
                var current = closingPrices.Last();
                var previous = closingPrices[closingPrices.Count - 11]; // 10 periods ago

                if (previous == 0)
                    return 0;

                return (current - previous) / previous * 100;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate momentum for {symbol}", ex.ToString());
                return 0; // Neutral default
            }
        }

        public async Task<(double StochK, double StochD)> GetSTOCH(string symbol, string interval = "1min")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 20) // Need enough data for Stochastic calculation
                    return (50, 50); // Neutral default when insufficient data

                var highs = historicalData.Select(h => h.High).ToList();
                var lows = historicalData.Select(h => h.Low).ToList();
                var closes = historicalData.Select(h => h.Close).ToList();

                var (stochK, stochD) = CalculateStochasticInternal(highs, lows, closes, 14, 3, 3);
                var latestK = stochK.LastOrDefault(k => !double.IsNaN(k));
                var latestD = stochD.LastOrDefault(d => !double.IsNaN(d));

                return (double.IsNaN(latestK) ? 50 : latestK, double.IsNaN(latestD) ? 50 : latestD);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate Stochastic for {symbol}", ex.ToString());
                return (50, 50); // Neutral default
            }
        }

        public async Task<double> GetCCI(string symbol, string interval = "1min")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 20) // Need enough data for CCI calculation
                    return 0; // Neutral default when insufficient data

                var highs = historicalData.Select(h => h.High).ToList();
                var lows = historicalData.Select(h => h.Low).ToList();
                var closes = historicalData.Select(h => h.Close).ToList();

                var cciValues = CalculateCCIInternal(highs, lows, closes, 14);
                var latestCci = cciValues.LastOrDefault(c => !double.IsNaN(c));
                return double.IsNaN(latestCci) ? 0 : latestCci;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate CCI for {symbol}", ex.ToString());
                return 0; // Neutral default
            }
        }

        public async Task<List<string>> GetMostVolatileStocksAsync()
        {
            // TODO: Implement caching if needed

            await WaitForApiLimit();
            var endpoint = $"query?function=TOP_GAINERS_LOSERS&apikey={_apiKey}";
            await LogApiCall("TOP_GAINERS_LOSERS", null);

            var response = await _client.GetAsync(endpoint);
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var data = JObject.Parse(content);

                var volatileStocks = new List<string>();

                if (data["top_gainers"] is JArray gainers)
                    volatileStocks.AddRange(gainers.Select(g => g["ticker"]?.ToString()).Where(ticker => !string.IsNullOrEmpty(ticker)));

                if (data["top_losers"] is JArray losers)
                    volatileStocks.AddRange(losers.Select(l => l["ticker"]?.ToString()).Where(ticker => !string.IsNullOrEmpty(ticker)));

                // TODO: Cache the volatile stocks if needed
                return volatileStocks;
            }

            return new List<string>();
        }

        public async Task<List<StockIndicator>> GetIndicatorsAsync(string symbol)
        {
            var indicators = new List<StockIndicator>();

            try
            {
                // Get current market price
                var price = await GetQuoteData(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "Price",
                    Value = price.ToString("F2"),
                    Description = "Current market price"
                });

                // Get RSI
                var rsi = await GetRSI(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "RSI",
                    Value = rsi.ToString("F2"),
                    Description = "Relative Strength Index"
                });

                // Get ADX
                var adx = await GetLatestADX(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "ADX",
                    Value = adx.ToString("F2"),
                    Description = "Average Directional Index"
                });

                // Get ATR
                var atr = await GetATR(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "ATR",
                    Value = atr.ToString("F2"),
                    Description = "Average True Range"
                });

                // Get Momentum Score
                var momentum = await GetMomentumScore(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "MomentumScore",
                    Value = momentum.ToString("F2"),
                    Description = "Overall Momentum Score"
                });

                // Get CCI
                var cci = await GetCCI(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "CCI",
                    Value = cci.ToString("F2"),
                    Description = "Commodity Channel Index"
                });

                // Get STOCH
                var stoch = await GetSTOCH(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "StochK",
                    Value = stoch.StochK.ToString("F2"),
                    Description = "Stochastic Oscillator %K"
                });
                indicators.Add(new StockIndicator
                {
                    Name = "StochD",
                    Value = stoch.StochD.ToString("F2"),
                    Description = "Stochastic Oscillator %D"
                });

                // Get Ultimate Oscillator
                var uo = await GetUltimateOscillator(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "UltimateOscillator",
                    Value = uo.ToString("F2"),
                    Description = "Ultimate Oscillator"
                });

                // Get On-Balance Volume
                var obv = await GetOBV(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "OBV",
                    Value = obv.ToString("F0"),
                    Description = "On-Balance Volume"
                });

                // Get Money Flow Index
                var mfi = await GetMFI(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "MFI",
                    Value = mfi.ToString("F2"),
                    Description = "Money Flow Index"
                });

                // Get VWAP
                var vwap = await GetVWAP(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "VWAP",
                    Value = vwap.ToString("F2"),
                    Description = "Volume Weighted Average Price"
                });

                // Get MACD
                var macdResult = await GetMACD(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "MACD",
                    Value = macdResult.Macd.ToString("F4"),
                    Description = "MACD Value"
                });
                indicators.Add(new StockIndicator
                {
                    Name = "MACD_Signal",
                    Value = macdResult.MacdSignal.ToString("F4"),
                    Description = "MACD Signal Line"
                });
                indicators.Add(new StockIndicator
                {
                    Name = "MACD_Hist",
                    Value = macdResult.MacdHist.ToString("F4"),
                    Description = "MACD Histogram"
                });

                // Get P/E Ratio (OVERVIEW)
                var peRatio = await GetPERatioAsync(symbol);
                indicators.Add(new StockIndicator
                {
                    Name = "PERatio",
                    Value = peRatio?.ToString("F2") ?? "N/A",
                    Description = "Price to Earnings Ratio (P/E)"
                });
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to get indicators", ex.ToString());
            }

            return indicators;
        }

        public async Task<double> GetUltimateOscillator(string symbol, string interval = "1min")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 30) // Need enough data for Ultimate Oscillator calculation
                    return 50; // Neutral default when insufficient data

                var highs = historicalData.Select(h => h.High).ToList();
                var lows = historicalData.Select(h => h.Low).ToList();
                var closes = historicalData.Select(h => h.Close).ToList();

                var uoValues = CalculateUltimateOscillatorInternal(highs, lows, closes, 7, 14, 28);
                var latestUo = uoValues.LastOrDefault(u => !double.IsNaN(u));
                return double.IsNaN(latestUo) ? 50 : latestUo;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate Ultimate Oscillator for {symbol}", ex.ToString());
                return 50; // Neutral default
            }
        }

        public async Task<Dictionary<string, double>> GetAllTechnicalIndicatorsAsync(string symbol)
        {
            var indicators = new Dictionary<string, double>();
            var technicalIndicators = await GetIndicatorsAsync(symbol);

            foreach (var indicator in technicalIndicators)
            {
                indicators[indicator.Name] = double.Parse(indicator.Value);
            }

            return indicators;
        }

        /// <summary>
        /// Calculate OBV using historical prices
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <param name="interval">Data interval</param>
        /// <returns>OBV value</returns>
        public async Task<double> GetOBV(string symbol, string interval = "1day")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 2)
                    return 0;

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                double obv = 0;
                for (int i = 1; i < historicalData.Count; i++)
                {
                    var currentClose = historicalData[i].Close;
                    var previousClose = historicalData[i - 1].Close;
                    var currentVolume = historicalData[i].Volume;

                    if (currentClose > previousClose)
                        obv += currentVolume;
                    else if (currentClose < previousClose)
                        obv -= currentVolume;
                    // Price unchanged - OBV remains the same
                }

                return obv;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate OBV for {symbol}", ex.ToString());
                return 0;
            }
        }

        /// <summary>
        /// Calculate Money Flow Index (MFI) using historical prices
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <param name="interval">Data interval</param>
        /// <returns>MFI value</returns>
        public async Task<double> GetMFI(string symbol, string interval = "1day")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 14)
                    return 50; // Default value

                // Sort by date to ensure chronological order
                historicalData = historicalData.OrderBy(h => h.Date).ToList();

                // Take the last 14 periods
                var periods = historicalData.Skip(Math.Max(0, historicalData.Count - 14)).ToList();

                double positiveMoneyFlow = 0;
                double negativeMoneyFlow = 0;

                for (int i = 1; i < periods.Count; i++)
                {
                    // Calculate typical price
                    double currentTypicalPrice = (periods[i].High + periods[i].Low + periods[i].Close) / 3;
                    double prevTypicalPrice = (periods[i - 1].High + periods[i - 1].Low + periods[i - 1].Close) / 3;

                    // Calculate raw money flow
                    double rawMoneyFlow = currentTypicalPrice * periods[i].Volume;

                    // Add to positive/negative money flow
                    if (currentTypicalPrice > prevTypicalPrice)
                        positiveMoneyFlow += rawMoneyFlow;
                    else if (currentTypicalPrice < prevTypicalPrice)
                        negativeMoneyFlow += rawMoneyFlow;
                }

                // Calculate money flow ratio
                double moneyFlowRatio = negativeMoneyFlow == 0 ? 100 : positiveMoneyFlow / negativeMoneyFlow;

                // Calculate MFI
                double mfi = 100 - 100 / (1 + moneyFlowRatio);

                return mfi;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate MFI for {symbol}", ex.ToString());
                return 50; // Default value
            }
        }

        public async Task<List<double>> GetHistoricalClosingPricesAsync(string symbol, int count)
        {
            await WaitForApiLimit();
            var endpoint = $"query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=compact&apikey={_apiKey}";
            await LogApiCall("TIME_SERIES_DAILY", symbol);

            var response = await _client.GetAsync(endpoint);
            var content = await response.Content.ReadAsStringAsync();
            var data = JObject.Parse(content);

            var prices = new List<double>();
            if (data["Time Series (Daily)"] is JObject timeSeries)
            {
                prices = timeSeries.Properties()
                    .Take(count)
                    .Select(p => TryParseDouble(p.Value["4. close"]))
                    .Where(price => price > 0) // Filter out invalid prices
                    .ToList();
            }

            return prices;
        }

        private async Task WaitForApiLimit()
        {
            await _apiSemaphore.WaitAsync();
            try
            {
                var recentCalls = GetCurrentDbApiCallCount();
                if (recentCalls >= _maxApiCallsPerMinute)
                {
                    // Wait until enough time has passed
                    await Task.Delay(TimeSpan.FromSeconds(61));
                }
            }
            finally
            {
                _apiSemaphore.Release();
            }
        }

        private async Task LogApiCall(string endpoint, string parameters)
        {
            await _apiSemaphore.WaitAsync();
            try
            {
                //DatabaseMonolith.LogAlphaVantageApiUsage(endpoint, parameters);
            }
            finally
            {
                _apiSemaphore.Release();
            }
        }

        public static string GetApiKey()
        {
            // Get API key from database using the Utilities helper
            return Quantra.DAL.Utilities.Utilities.GetAlphaVantageApiKey();
        }

        /// <summary>
        /// Gets forex historical data using Alpha Vantage Premium API
        /// </summary>
        /// <param name="fromSymbol">From currency symbol</param>
        /// <param name="toSymbol">To currency symbol</param>
        /// <param name="interval">Data interval (e.g., 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)</param>
        /// <returns>List of historical prices</returns>
        public async Task<List<HistoricalPrice>> GetForexHistoricalData(string fromSymbol, string toSymbol, string interval = "daily")
        {
            await WaitForApiLimit();
            string function;

            if (interval.EndsWith("min"))
            {
                function = "FX_INTRADAY";
            }
            else if (interval == "daily" || interval == "1d")
            {
                function = "FX_DAILY";
            }
            else if (interval == "weekly" || interval == "1wk")
            {
                function = "FX_WEEKLY";
            }
            else if (interval == "monthly" || interval == "1mo")
            {
                function = "FX_MONTHLY";
            }
            else
            {
                function = "FX_DAILY";
            }

            var parameters = new Dictionary<string, string>
            {
                { "from_symbol", fromSymbol },
                { "to_symbol", toSymbol },
                { "outputsize", "full" },
                { "apikey", _apiKey }
            };

            if (function == "FX_INTRADAY")
            {
                parameters.Add("interval", interval);
            }

            var responseString = await SendWithSlidingWindowAsync<string>(function, parameters);
            return ParseForexResponse(responseString, function);
        }

        /// <summary>
        /// Gets cryptocurrency historical data using Alpha Vantage Premium API
        /// </summary>
        /// <param name="symbol">Cryptocurrency symbol</param>
        /// <param name="market">Market (e.g., USD, EUR, CNY)</param>
        /// <param name="interval">Data interval (e.g., 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly)</param>
        /// <returns>List of historical prices</returns>
        public async Task<List<HistoricalPrice>> GetCryptoHistoricalData(string symbol, string market = "USD", string interval = "daily")
        {
            await WaitForApiLimit();
            string function;

            if (interval.EndsWith("min"))
            {
                function = "CRYPTO_INTRADAY";
            }
            else if (interval == "daily" || interval == "1d")
            {
                function = "DIGITAL_CURRENCY_DAILY";
            }
            else if (interval == "weekly" || interval == "1wk")
            {
                function = "DIGITAL_CURRENCY_WEEKLY";
            }
            else if (interval == "monthly" || interval == "1mo")
            {
                function = "DIGITAL_CURRENCY_MONTHLY";
            }
            else
            {
                function = "DIGITAL_CURRENCY_DAILY";
            }

            var parameters = new Dictionary<string, string>
            {
                { "symbol", symbol },
                { "market", market },
                { "outputsize", "full" },
                { "apikey", _apiKey }
            };

            if (function == "CRYPTO_INTRADAY")
            {
                parameters.Add("interval", interval);
            }

            var responseString = await SendWithSlidingWindowAsync<string>(function, parameters);
            return ParseCryptoResponse(responseString, function);
        }

        /// <summary>
        /// Gets historical data from database cache first, then falls back to API if insufficient data
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Data interval</param>
        /// <returns>List of historical prices from cache or API</returns>
        private async Task<List<HistoricalPrice>> GetCachedHistoricalDataFirst(string symbol, string interval)
        {
            try
            {
                // First, try to get data from database cache
                var cachedData = await GetCachedHistoricalPrices(symbol, interval);

                // If we have sufficient cached data (at least 50 data points for reliable calculations), use it
                if (cachedData.Count >= 50)
                {
                    //DatabaseMonolith.Log("Info", $"Using cached historical data for {symbol} - {cachedData.Count} data points");
                    return cachedData;
                }

                // If insufficient cached data, fall back to API
                //DatabaseMonolith.Log("Info", $"Insufficient cached data for {symbol} ({cachedData.Count} points), fetching from API");
                return await GetExtendedHistoricalData(symbol, interval, "compact");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error in GetCachedHistoricalDataFirst for {symbol}", ex.ToString());
                // Fall back to API on any error
                return await GetExtendedHistoricalData(symbol, interval, "compact");
            }
        }

        /// <summary>
        /// Gets cached historical price data from database
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Data interval</param>
        /// <returns>List of cached historical prices</returns>
        private async Task<List<HistoricalPrice>> GetCachedHistoricalPrices(string symbol, string interval)
        {
            // Convert interval to timeRange format expected by database
            string timeRange = interval switch
            {
                "1min" => "1day", // For minute data, get 1 day worth
                "5min" => "5day", // For 5min data, get 5 days worth
                "15min" => "1week", // For 15min data, get 1 week worth
                "30min" => "2week", // For 30min data, get 2 weeks worth
                "1hour" => "1month", // For hourly data, get 1 month worth
                "daily" => "3month", // For daily data, get 3 months worth
                _ => "1month"
            };

            try
            {
                // Try to get cached data from database
                // Note: GetStockDataWithTimestamp was removed - caching not implemented
                // Return empty list to fallback to API call
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to get cached historical prices for {symbol}", ex.ToString());
            }

            return new List<HistoricalPrice>();
        }

        /// <summary>
        /// Gets intraday price data using the TIME_SERIES_INTRADAY Alpha Vantage API endpoint
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Intraday interval: 1min, 5min, 15min, 30min, or 60min</param>
        /// <param name="outputSize">Output size: compact (latest 100 data points) or full (trailing 30 days)</param>
        /// <param name="dataType">Response format: json or csv</param>
        /// <returns>List of intraday historical prices</returns>
        public async Task<List<HistoricalPrice>> GetIntradayData(string symbol, string interval = "5min", string outputSize = "compact", string dataType = "json")
        {
            await WaitForApiLimit();

            // Normalize symbol for API call
            string normalizedSymbol = NormalizeSymbol(symbol);

            // Validate interval - AlphaVantage TIME_SERIES_INTRADAY supports: 1min, 5min, 15min, 30min, 60min
            var validIntervals = new[] { "1min", "5min", "15min", "30min", "60min" };
            if (!validIntervals.Contains(interval))
            {
                _loggingService.Log("Warning", $"Invalid intraday interval '{interval}'. Defaulting to 5min.");
                interval = "5min";
            }

            var parameters = new Dictionary<string, string>
            {
                { "symbol", normalizedSymbol },
                { "interval", interval },
                { "outputsize", outputSize },
                { "datatype", dataType },
                { "apikey", _apiKey }
            };

            await LogApiCall("TIME_SERIES_INTRADAY", $"{normalizedSymbol}&interval={interval}");
            var responseString = await SendWithSlidingWindowAsync<string>("TIME_SERIES_INTRADAY", parameters);
            return ParseAlphaVantageResponse(responseString, "TIME_SERIES_INTRADAY");
        }

        /// <summary>
        /// Gets daily price data using the TIME_SERIES_DAILY Alpha Vantage API endpoint (non-adjusted)
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="outputSize">Output size: compact (latest 100 data points) or full (20+ years)</param>
        /// <param name="dataType">Response format: json or csv</param>
        /// <returns>List of daily historical prices (non-adjusted)</returns>
        public async Task<List<HistoricalPrice>> GetDailyData(string symbol, string outputSize = "full", string dataType = "json")
        {
            await WaitForApiLimit();

            string normalizedSymbol = NormalizeSymbol(symbol);

            var parameters = new Dictionary<string, string>
            {
                { "symbol", normalizedSymbol },
                { "outputsize", outputSize },
                { "datatype", dataType },
                { "apikey", _apiKey }
            };

            await LogApiCall("TIME_SERIES_DAILY", normalizedSymbol);
            var responseString = await SendWithSlidingWindowAsync<string>("TIME_SERIES_DAILY", parameters);
            return ParseAlphaVantageResponse(responseString, "TIME_SERIES_DAILY");
        }

        /// <summary>
        /// Gets weekly price data using the TIME_SERIES_WEEKLY Alpha Vantage API endpoint (non-adjusted)
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="dataType">Response format: json or csv</param>
        /// <returns>List of weekly historical prices (non-adjusted)</returns>
        public async Task<List<HistoricalPrice>> GetWeeklyData(string symbol, string dataType = "json")
        {
            await WaitForApiLimit();

            string normalizedSymbol = NormalizeSymbol(symbol);

            var parameters = new Dictionary<string, string>
            {
                { "symbol", normalizedSymbol },
                { "datatype", dataType },
                { "apikey", _apiKey }
            };

            await LogApiCall("TIME_SERIES_WEEKLY", normalizedSymbol);
            var responseString = await SendWithSlidingWindowAsync<string>("TIME_SERIES_WEEKLY", parameters);
            return ParseAlphaVantageResponse(responseString, "TIME_SERIES_WEEKLY");
        }

        /// <summary>
        /// Gets monthly price data using the TIME_SERIES_MONTHLY Alpha Vantage API endpoint (non-adjusted)
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="dataType">Response format: json or csv</param>
        /// <returns>List of monthly historical prices (non-adjusted)</returns>
        public async Task<List<HistoricalPrice>> GetMonthlyData(string symbol, string dataType = "json")
        {
            await WaitForApiLimit();

            string normalizedSymbol = NormalizeSymbol(symbol);

            var parameters = new Dictionary<string, string>
            {
                { "symbol", normalizedSymbol },
                { "datatype", dataType },
                { "apikey", _apiKey }
            };

            await LogApiCall("TIME_SERIES_MONTHLY", normalizedSymbol);
            var responseString = await SendWithSlidingWindowAsync<string>("TIME_SERIES_MONTHLY", parameters);
            return ParseAlphaVantageResponse(responseString, "TIME_SERIES_MONTHLY");
        }

        /// <summary>
        /// Gets extended historical data with adjusted prices for more accurate backtesting
        /// Supports both adjusted and non-adjusted endpoints based on the useAdjusted parameter
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Data interval: supports intraday (1min, 5min, 15min, 30min, 60min), daily, weekly, monthly, or their aliases (1d, 1wk, 1mo)</param>
        /// <param name="outputSize">Output size (compact/full)</param>
        /// <param name="dataType">Type of data (json/csv)</param>
        /// <param name="useAdjusted">Whether to use adjusted prices (default: true for daily/weekly/monthly, always false for intraday)</param>
        /// <returns>List of historical prices</returns>
        public async Task<List<HistoricalPrice>> GetExtendedHistoricalData(string symbol, string interval = "daily", string outputSize = "full", string dataType = "json", bool useAdjusted = true)
        {
            await WaitForApiLimit();

            // Normalize symbol for API call
            string normalizedSymbol = NormalizeSymbol(symbol);

            string function;
            string avInterval = "";

            // Handle intraday intervals (1min, 5min, 15min, 30min, 60min)
            if (interval.EndsWith("min"))
            {
                function = "TIME_SERIES_INTRADAY";
                avInterval = interval;
            }
            else if (interval == "daily" || interval == "1d")
            {
                function = useAdjusted ? "TIME_SERIES_DAILY_ADJUSTED" : "TIME_SERIES_DAILY";
            }
            else if (interval == "weekly" || interval == "1wk")
            {
                function = useAdjusted ? "TIME_SERIES_WEEKLY_ADJUSTED" : "TIME_SERIES_WEEKLY";
            }
            else if (interval == "monthly" || interval == "1mo")
            {
                function = useAdjusted ? "TIME_SERIES_MONTHLY_ADJUSTED" : "TIME_SERIES_MONTHLY";
            }
            else
            {
                function = useAdjusted ? "TIME_SERIES_DAILY_ADJUSTED" : "TIME_SERIES_DAILY";
            }

            var parameters = new Dictionary<string, string>
            {
                { "symbol", normalizedSymbol },
                { "outputsize", outputSize },
                { "datatype", dataType },
                { "apikey", _apiKey }
            };

            if (function == "TIME_SERIES_INTRADAY")
            {
                parameters.Add("interval", avInterval);
            }

            var logDetails = string.IsNullOrEmpty(avInterval) ? normalizedSymbol : $"{normalizedSymbol}&interval={avInterval}";
            await LogApiCall(function, logDetails);
            var responseString = await SendWithSlidingWindowAsync<string>(function, parameters);
            return ParseAlphaVantageResponse(responseString, function);
        }

        /// <summary>
        /// Parses Alpha Vantage forex response into HistoricalPrice list
        /// </summary>
        private List<HistoricalPrice> ParseForexResponse(string jsonResponse, string function)
        {
            var result = new List<HistoricalPrice>();

            try
            {
                var jsonObject = JObject.Parse(jsonResponse);

                string timeSeriesKey = function switch
                {
                    "FX_INTRADAY" => jsonObject.Properties().FirstOrDefault(p => p.Name.StartsWith("Time Series FX"))?.Name,
                    "FX_DAILY" => "Time Series FX (Daily)",
                    "FX_WEEKLY" => "Time Series FX (Weekly)",
                    "FX_MONTHLY" => "Time Series FX (Monthly)",
                    _ => null
                };

                if (timeSeriesKey == null || !jsonObject.ContainsKey(timeSeriesKey))
                    return result;

                var timeSeries = jsonObject[timeSeriesKey] as JObject;
                if (timeSeries == null)
                    return result;

                foreach (var item in timeSeries)
                {
                    var dateStr = item.Key;
                    var data = item.Value;

                    if (DateTime.TryParse(dateStr, out DateTime date))
                    {
                        double ParseDouble(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return double.TryParse(token.ToString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double val) ? val : 0;
                        }

                        result.Add(new HistoricalPrice
                        {
                            Date = date,
                            Open = ParseDouble("1. open"),
                            High = ParseDouble("2. high"),
                            Low = ParseDouble("3. low"),
                            Close = ParseDouble("4. close"),
                            Volume = 0, // Forex doesn't typically include volume
                            AdjClose = ParseDouble("4. close") // No adjusted close for forex
                        });
                    }
                }

                // Sort by date ascending
                result = result.OrderBy(h => h.Date).ToList();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to parse forex response", ex.ToString());
            }

            return result;
        }

        /// <summary>
        /// Parses Alpha Vantage crypto response into HistoricalPrice list
        /// </summary>
        private List<HistoricalPrice> ParseCryptoResponse(string jsonResponse, string function)
        {
            var result = new List<HistoricalPrice>();

            try
            {
                var jsonObject = JObject.Parse(jsonResponse);

                string timeSeriesKey = function switch
                {
                    "CRYPTO_INTRADAY" => jsonObject.Properties().FirstOrDefault(p => p.Name.StartsWith("Time Series Crypto"))?.Name,
                    "DIGITAL_CURRENCY_DAILY" => "Time Series (Digital Currency Daily)",
                    "DIGITAL_CURRENCY_WEEKLY" => "Time Series (Digital Currency Weekly)",
                    "DIGITAL_CURRENCY_MONTHLY" => "Time Series (Digital Currency Monthly)",
                    _ => null
                };

                if (timeSeriesKey == null || !jsonObject.ContainsKey(timeSeriesKey))
                    return result;

                var timeSeries = jsonObject[timeSeriesKey] as JObject;
                if (timeSeries == null)
                    return result;

                foreach (var item in timeSeries)
                {
                    var dateStr = item.Key;
                    var data = item.Value;

                    if (DateTime.TryParse(dateStr, out DateTime date))
                    {
                        double ParseDouble(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return double.TryParse(token.ToString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double val) ? val : 0;
                        }

                        long ParseLong(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return long.TryParse(token.ToString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out long val) ? val : 0;
                        }

                        // For crypto, we use the USD values if available (premium API provides market-specific values)
                        result.Add(new HistoricalPrice
                        {
                            Date = date,
                            Open = ParseDouble("1a. open (USD)") != 0 ? ParseDouble("1a. open (USD)") : ParseDouble("1. open"),
                            High = ParseDouble("2a. high (USD)") != 0 ? ParseDouble("2a. high (USD)") : ParseDouble("2. high"),
                            Low = ParseDouble("3a. low (USD)") != 0 ? ParseDouble("3a. low (USD)") : ParseDouble("3. low"),
                            Close = ParseDouble("4a. close (USD)") != 0 ? ParseDouble("4a. close (USD)") : ParseDouble("4. close"),
                            Volume = ParseLong("5. volume") != 0 ? ParseLong("5. volume") : ParseLong("6. market cap (USD)"),
                            AdjClose = ParseDouble("4a. close (USD)") != 0 ? ParseDouble("4a. close (USD)") : ParseDouble("4. close") // Crypto doesn't have adjusted close
                        });
                    }
                }

                // Sort by date ascending
                result = result.OrderBy(h => h.Date).ToList();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to parse crypto response", ex.ToString());
            }

            return result;
        }

        /// <summary>
        /// Parses the Alpha Vantage API response and converts it to a list of HistoricalPrice objects
        /// </summary>
        private List<HistoricalPrice> ParseAlphaVantageResponse(string jsonResponse, string function)
        {
            var result = new List<HistoricalPrice>();
            try
            {
                if (string.IsNullOrWhiteSpace(jsonResponse))
                {
                    _loggingService?.Log("Warning", $"Empty response received for {function}");
                    return result;
                }

                var jsonObject = JObject.Parse(jsonResponse);

                // Check for API error messages
                if (jsonObject["Error Message"] != null)
                {
                    var errorMsg = jsonObject["Error Message"].ToString();
                    _loggingService?.Log("Error", $"Alpha Vantage API Error for {function}: {errorMsg}");
                    return result;
                }

                // Check for rate limit message
                if (jsonObject["Note"] != null)
                {
                    var note = jsonObject["Note"].ToString();
                    _loggingService?.Log("Warning", $"Alpha Vantage API Note for {function}: {note}");
                    return result;
                }

                // Check for information message (often indicates invalid parameters)
                if (jsonObject["Information"] != null)
                {
                    var info = jsonObject["Information"].ToString();
                    _loggingService?.Log("Warning", $"Alpha Vantage API Information for {function}: {info}");
                    return result;
                }

                // Determine the correct time series key
                string timeSeriesKey = function switch
                {
                    "TIME_SERIES_INTRADAY" => jsonObject.Properties().FirstOrDefault(p => p.Name.StartsWith("Time Series"))?.Name,
                    "TIME_SERIES_DAILY_ADJUSTED" => "Time Series (Daily)",
                    "TIME_SERIES_WEEKLY_ADJUSTED" => "Weekly Adjusted Time Series",
                    "TIME_SERIES_MONTHLY_ADJUSTED" => "Monthly Adjusted Time Series",
                    "TIME_SERIES_DAILY" => "Time Series (Daily)",
                    "TIME_SERIES_WEEKLY" => "Weekly Time Series",
                    "TIME_SERIES_MONTHLY" => "Monthly Time Series",
                    _ => null
                };

                if (timeSeriesKey == null)
                {
                    _loggingService?.Log("Warning", $"Unknown time series key for function {function}");
                    return result;
                }

                if (!jsonObject.ContainsKey(timeSeriesKey))
                {
                    _loggingService?.Log("Warning", $"Response does not contain expected key '{timeSeriesKey}' for {function}");
                    return result;
                }

                var timeSeries = jsonObject[timeSeriesKey] as JObject;
                if (timeSeries == null)
                {
                    _loggingService?.Log("Warning", $"Time series data is null or not a valid object for {function}");
                    return result;
                }

                if (timeSeries.Count == 0)
                {
                    _loggingService?.Log("Info", $"No time series data points found for {function}");
                    return result;
                }

                foreach (var item in timeSeries)
                {
                    var dateStr = item.Key;
                    var data = item.Value;

                    if (data == null)
                    {
                        _loggingService?.Log("Warning", $"Null data point for date {dateStr} in {function}");
                        continue;
                    }

                    if (DateTime.TryParse(dateStr, out DateTime date))
                    {
                        double ParseDouble(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return double.TryParse(token.ToString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double val) ? val : 0;
                        }

                        long ParseLong(string key)
                        {
                            var token = data[key];
                            if (token == null) return 0;
                            return long.TryParse(token.ToString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out long val) ? val : 0;
                        }

                        result.Add(new HistoricalPrice
                        {
                            Date = date,
                            Open = ParseDouble("1. open"),
                            High = ParseDouble("2. high"),
                            Low = ParseDouble("3. low"),
                            Close = ParseDouble("4. close"),
                            Volume = ParseLong("6. volume") != 0 ? ParseLong("6. volume") : ParseLong("5. volume"),
                            AdjClose = data["5. adjusted close"] != null ? ParseDouble("5. adjusted close") : ParseDouble("4. close")
                        });
                    }
                    else
                    {
                        _loggingService?.Log("Warning", $"Failed to parse date '{dateStr}' in {function}");
                    }
                }

                // Sort by date ascending
                result = result.OrderBy(h => h.Date).ToList();

                _loggingService?.Log("Info", $"Successfully parsed {result.Count} data points for {function}");
            }
            catch (JsonReaderException jsonEx)
            {
                _loggingService?.Log("Error", $"JSON parsing error for {function}: {jsonEx.Message}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to parse Alpha Vantage response for {function}");
            }
            return result;
        }

        // Get VWAP using historical data calculation
        public async Task<double> GetVWAP(string symbol, string interval = "15min")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count == 0)
                    return 0;

                double cumulativeTPV = 0; // Typical Price * Volume
                long cumulativeVolume = 0;

                foreach (var bar in historicalData)
                {
                    double typicalPrice = (bar.High + bar.Low + bar.Close) / 3;
                    cumulativeTPV += typicalPrice * bar.Volume;
                    cumulativeVolume += bar.Volume;
                }

                if (cumulativeVolume == 0)
                    return historicalData.Last().Close;

                return cumulativeTPV / cumulativeVolume;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate VWAP for {symbol}", ex.ToString());
                return 0;
            }
        }

        // Get MACD using historical data calculation
        public async Task<(double Macd, double MacdSignal, double MacdHist)> GetMACD(string symbol, string interval = "daily", string seriesType = "close")
        {
            try
            {
                var historicalData = await GetCachedHistoricalDataFirst(symbol, interval);
                if (historicalData.Count < 35)
                    return (0, 0, 0);

                var prices = historicalData.Select(h => h.Close).ToList();

                // Calculate MACD with standard settings (12, 26, 9)
                var ema12 = CalculateEMAInternal(prices, 12);
                var ema26 = CalculateEMAInternal(prices, 26);

                if (ema12.Count == 0 || ema26.Count == 0)
                    return (0, 0, 0);

                // Calculate MACD line
                var macdLine = new List<double>();
                for (int i = 0; i < Math.Min(ema12.Count, ema26.Count); i++)
                {
                    macdLine.Add(ema12[i] - ema26[i]);
                }

                // Calculate signal line (9-day EMA of MACD line)
                var signalLine = CalculateEMAInternal(macdLine, 9);

                if (macdLine.Count > 0 && signalLine.Count > 0)
                {
                    double macd = macdLine.Last();
                    double signal = signalLine.Last();
                    double histogram = macd - signal;
                    return (macd, signal, histogram);
                }

                return (0, 0, 0);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to calculate MACD for {symbol}", ex.ToString());
                return (0, 0, 0);
            }
        }

        /// <summary>
        /// Gets the P/E ratio for a stock using the Alpha Vantage OVERVIEW endpoint
        /// </summary>
        public async Task<double?> GetPERatioAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            // Check cache first
            var cached = GetCachedFundamentalData(symbol, "PE_RATIO", 4); // 4 hour cache for P/E ratio
            if (cached.HasValue)
                return cached.Value;

            // Fetch from API
            await WaitForApiLimit();
            var endpoint = $"query?function=OVERVIEW&symbol={symbol}&apikey={_apiKey}";
            await LogApiCall("OVERVIEW", symbol);

            var response = await _client.GetAsync(endpoint);
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var data = JObject.Parse(content);
                if (data["PERatio"] != null && double.TryParse(data["PERatio"].ToString(), out double peRatio))
                {
                    // Cache the result
                    CacheFundamentalData(symbol, "PE_RATIO", peRatio);
                    return peRatio;
                }
            }

            return null;
        }

        #region Private Calculation Methods

        private static List<double> CalculateADXInternal(List<double> highs, List<double> lows, List<double> closes, int period = 14)
        {
            var result = new List<double>();
            int length = Math.Min(Math.Min(highs.Count, lows.Count), closes.Count);

            if (length < period + 1)
            {
                for (int i = 0; i < length; i++)
                    result.Add(double.NaN);
                return result;
            }

            // Calculate True Range and Directional Movement
            var trueRanges = new List<double>();
            var plusDMs = new List<double>();
            var minusDMs = new List<double>();

            for (int i = 1; i < length; i++)
            {
                // True Range
                double tr1 = highs[i] - lows[i];
                double tr2 = Math.Abs(highs[i] - closes[i - 1]);
                double tr3 = Math.Abs(lows[i] - closes[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);

                // Directional Movement
                double highDiff = highs[i] - highs[i - 1];
                double lowDiff = lows[i - 1] - lows[i];

                double plusDM = highDiff > lowDiff && highDiff > 0 ? highDiff : 0;
                double minusDM = lowDiff > highDiff && lowDiff > 0 ? lowDiff : 0;

                plusDMs.Add(plusDM);
                minusDMs.Add(minusDM);
            }

            // Smooth the values using Wilder's smoothing (EMA-like)
            var smoothedTRs = new List<double>();
            var smoothedPlusDMs = new List<double>();
            var smoothedMinusDMs = new List<double>();

            if (trueRanges.Count >= period)
            {
                // First smoothed value is SMA
                double trSum = 0, plusDMSum = 0, minusDMSum = 0;
                for (int i = 0; i < period; i++)
                {
                    trSum += trueRanges[i];
                    plusDMSum += plusDMs[i];
                    minusDMSum += minusDMs[i];
                }

                smoothedTRs.Add(trSum / period);
                smoothedPlusDMs.Add(plusDMSum / period);
                smoothedMinusDMs.Add(minusDMSum / period);

                // Subsequent values use Wilder's smoothing
                for (int i = period; i < trueRanges.Count; i++)
                {
                    smoothedTRs.Add((smoothedTRs.Last() * (period - 1) + trueRanges[i]) / period);
                    smoothedPlusDMs.Add((smoothedPlusDMs.Last() * (period - 1) + plusDMs[i]) / period);
                    smoothedMinusDMs.Add((smoothedMinusDMs.Last() * (period - 1) + minusDMs[i]) / period);
                }
            }

            // Calculate +DI and -DI
            var plusDIs = new List<double>();
            var minusDIs = new List<double>();

            for (int i = 0; i < smoothedTRs.Count; i++)
            {
                double plusDI = smoothedTRs[i] == 0 ? 0 : smoothedPlusDMs[i] / smoothedTRs[i] * 100;
                double minusDI = smoothedTRs[i] == 0 ? 0 : smoothedMinusDMs[i] / smoothedTRs[i] * 100;

                plusDIs.Add(plusDI);
                minusDIs.Add(minusDI);
            }

            // Calculate DX
            var dxValues = new List<double>();
            for (int i = 0; i < plusDIs.Count; i++)
            {
                double diSum = plusDIs[i] + minusDIs[i];
                double dx = diSum == 0 ? 0 : Math.Abs(plusDIs[i] - minusDIs[i]) / diSum * 100;
                dxValues.Add(dx);
            }

            // Calculate ADX (EMA of DX)
            var adxValues = CalculateEMAInternal(dxValues, period);

            // Pad with NaN for initial periods
            for (int i = 0; i < period; i++)
                result.Add(double.NaN);

            result.AddRange(adxValues);
            return result;
        }

        private static List<double> CalculateATRInternal(List<double> highs, List<double> lows, List<double> closes, int period = 14)
        {
            var result = new List<double>();
            int length = Math.Min(Math.Min(highs.Count, lows.Count), closes.Count);

            if (length < 2)
            {
                for (int i = 0; i < length; i++)
                    result.Add(double.NaN);
                return result;
            }

            // Calculate True Range for each period
            var trueRanges = new List<double>();

            // First period - just high-low
            result.Add(highs[0] - lows[0]);

            for (int i = 1; i < length; i++)
            {
                double tr1 = highs[i] - lows[i];
                double tr2 = Math.Abs(highs[i] - closes[i - 1]);
                double tr3 = Math.Abs(lows[i] - closes[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);
            }

            // Calculate ATR using EMA of True Range
            if (trueRanges.Count >= period)
            {
                var atrValues = CalculateEMAInternal(trueRanges, period);
                result.AddRange(atrValues);
            }
            else
            {
                // Not enough data for full ATR calculation
                for (int i = 1; i < length; i++)
                    result.Add(double.NaN);
            }

            return result;
        }

        private static (List<double> K, List<double> D) CalculateStochasticInternal(List<double> highs, List<double> lows, List<double> closes, int kPeriod, int kSmoothing, int dPeriod)
        {
            var result = (K: new List<double>(), D: new List<double>());
            int length = Math.Min(Math.Min(highs.Count, lows.Count), closes.Count);

            // Calculate %K for each point
            var rawK = new List<double>();
            for (int i = 0; i < length; i++)
            {
                if (i < kPeriod - 1)
                {
                    rawK.Add(double.NaN);
                    continue;
                }

                // Find highest high and lowest low over period
                var highestHigh = double.MinValue;
                var lowestLow = double.MaxValue;

                for (int j = i - kPeriod + 1; j <= i; j++)
                {
                    highestHigh = Math.Max(highestHigh, highs[j]);
                    lowestLow = Math.Min(lowestLow, lows[j]);
                }

                // Calculate raw %K
                double currentClose = closes[i];
                double stochK = highestHigh == lowestLow ? 50 : (currentClose - lowestLow) / (highestHigh - lowestLow) * 100;
                rawK.Add(stochK);
            }

            // Calculate smoothed %K using SMA
            var smoothedK = kSmoothing > 1 ? CalculateSMAInternal(rawK, kSmoothing) : rawK;
            result.K = smoothedK;

            // Calculate %D (SMA of %K)
            result.D = CalculateSMAInternal(smoothedK, dPeriod);

            return result;
        }

        private static List<double> CalculateCCIInternal(List<double> highs, List<double> lows, List<double> closes, int period)
        {
            var result = new List<double>();
            int length = Math.Min(Math.Min(highs.Count, lows.Count), closes.Count);

            // Calculate typical prices: (H+L+C)/3
            var typicalPrices = new List<double>();
            for (int i = 0; i < length; i++)
            {
                typicalPrices.Add((highs[i] + lows[i] + closes[i]) / 3);
            }

            // Calculate CCI
            for (int i = 0; i < length; i++)
            {
                if (i < period - 1)
                {
                    result.Add(double.NaN);
                    continue;
                }

                // Calculate SMA of typical prices
                var sma = 0.0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    sma += typicalPrices[j];
                }
                sma /= period;

                // Calculate mean deviation
                var meanDev = 0.0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    meanDev += Math.Abs(typicalPrices[j] - sma);
                }
                meanDev /= period;

                // Calculate CCI
                double cci = meanDev == 0 ? 0 : (typicalPrices[i] - sma) / (0.015 * meanDev);
                result.Add(cci);
            }

            return result;
        }

        private static List<double> CalculateUltimateOscillatorInternal(List<double> highs, List<double> lows, List<double> closes, int period1 = 7, int period2 = 14, int period3 = 28)
        {
            var result = new List<double>();
            int length = Math.Min(Math.Min(highs.Count, lows.Count), closes.Count);

            if (length < period3 + 1)
            {
                for (int i = 0; i < length; i++)
                    result.Add(double.NaN);
                return result;
            }

            // Calculate Buying Pressure (BP) and True Range (TR)
            var buyingPressures = new List<double>();
            var trueRanges = new List<double>();

            for (int i = 1; i < length; i++)
            {
                // Buying Pressure = Close - Min(Low, Previous Close)
                double minLow = Math.Min(lows[i], closes[i - 1]);
                double bp = closes[i] - minLow;
                buyingPressures.Add(bp);

                // True Range
                double tr1 = highs[i] - lows[i];
                double tr2 = Math.Abs(highs[i] - closes[i - 1]);
                double tr3 = Math.Abs(lows[i] - closes[i - 1]);
                double tr = Math.Max(tr1, Math.Max(tr2, tr3));
                trueRanges.Add(tr);
            }

            // Calculate Ultimate Oscillator
            for (int i = period3 - 1; i < buyingPressures.Count; i++)
            {
                // Sum BP and TR for each period
                double bp1Sum = 0, tr1Sum = 0;
                double bp2Sum = 0, tr2Sum = 0;
                double bp3Sum = 0, tr3Sum = 0;

                // Period 1
                for (int j = i - period1 + 1; j <= i; j++)
                {
                    bp1Sum += buyingPressures[j];
                    tr1Sum += trueRanges[j];
                }

                // Period 2
                for (int j = i - period2 + 1; j <= i; j++)
                {
                    bp2Sum += buyingPressures[j];
                    tr2Sum += trueRanges[j];
                }

                // Period 3
                for (int j = i - period3 + 1; j <= i; j++)
                {
                    bp3Sum += buyingPressures[j];
                    tr3Sum += trueRanges[j];
                }

                // Calculate averages
                double avg1 = tr1Sum == 0 ? 0 : bp1Sum / tr1Sum;
                double avg2 = tr2Sum == 0 ? 0 : bp2Sum / tr2Sum;
                double avg3 = tr3Sum == 0 ? 0 : bp3Sum / tr3Sum;

                // Ultimate Oscillator formula
                double uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7;
                result.Add(uo);
            }

            // Pad with NaN for initial periods
            for (int i = 0; i < period3; i++)
                result.Insert(0, double.NaN);

            return result;
        }

        private static List<double> CalculateSMAInternal(List<double> values, int period)
        {
            var result = new List<double>();

            for (int i = 0; i < values.Count; i++)
            {
                if (i < period - 1)
                {
                    result.Add(double.NaN);
                    continue;
                }

                double sum = 0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    if (double.IsNaN(values[j]))
                    {
                        sum = double.NaN;
                        break;
                    }
                    sum += values[j];
                }

                result.Add(double.IsNaN(sum) ? double.NaN : sum / period);
            }

            return result;
        }

        private static List<double> CalculateRSIInternal(List<double> prices, int period)
        {
            var result = new List<double>();

            if (prices.Count <= period)
            {
                for (int i = 0; i < prices.Count; i++)
                {
                    result.Add(double.NaN);
                }
                return result;
            }

            var priceChanges = new List<double>();
            for (int i = 1; i < prices.Count; i++)
            {
                priceChanges.Add(prices[i] - prices[i - 1]);
            }

            var gains = new List<double>();
            var losses = new List<double>();

            for (int i = 0; i < priceChanges.Count; i++)
            {
                double change = priceChanges[i];
                gains.Add(change > 0 ? change : 0);
                losses.Add(change < 0 ? Math.Abs(change) : 0);

                if (i < period - 1)
                {
                    result.Add(double.NaN);
                }
                else
                {
                    double avgGain = gains.Skip(i - period + 1).Take(period).Average();
                    double avgLoss = losses.Skip(i - period + 1).Take(period).Average();

                    double rs = avgLoss == 0 ? 100 : avgGain / avgLoss;
                    double rsi = 100 - 100 / (1 + rs);
                    result.Add(rsi);
                }
            }

            return result;
        }

        private static List<double> CalculateEMAInternal(List<double> prices, int period)
        {
            var result = new List<double>();

            if (prices.Count == 0)
                return result;

            double multiplier = 2.0 / (period + 1);

            // First value is just the price
            result.Add(prices[0]);

            for (int i = 1; i < prices.Count; i++)
            {
                double ema = prices[i] * multiplier + result[i - 1] * (1 - multiplier);
                result.Add(ema);
            }

            return result;
        }

        #endregion

        #region Helper Methods for Null-Safe Parsing

        /// <summary>
        /// Safely parse a JSON token to double, returning 0 if null or invalid
        /// </summary>
        private static double TryParseDouble(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return 0.0;

            if (double.TryParse(token.ToString(), out double result))
                return result;

            return 0.0;
        }

        /// <summary>
        /// Safely parse a JSON token as percentage, returning 0 if null or invalid
        /// </summary>
        private static double TryParsePercentage(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return 0.0;

            var value = token.ToString();
            if (string.IsNullOrEmpty(value))
                return 0.0;

            // Remove percentage sign if present
            value = value.TrimEnd('%');

            if (double.TryParse(value, out double result))
                return result;

            return 0.0;
        }

        /// <summary>
        /// Safely parse a JSON token to DateTime, returning current date if null or invalid
        /// </summary>
        private static DateTime TryParseDateTime(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return DateTime.Now;

            if (DateTime.TryParse(token.ToString(), out DateTime result))
                return result;

            return DateTime.Now;
        }

        #endregion

        /// <summary>
        /// Gets historical indicator data calculated from cached historical price data
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="indicatorType">Type of indicator to calculate (RSI, MACD, ADX, ROC, BB_Width, etc.)</param>
        /// <returns>List of historical indicator values</returns>
        public async Task<List<double>> GetHistoricalIndicatorData(string symbol, string indicatorType)
        {
            try
            {
                // Get real historical price data from database cache first
                var historicalData = await GetCachedHistoricalPrices(symbol, "daily");

                if (historicalData.Count < 10)
                {
                    _loggingService.Log("Warning", $"Insufficient historical data for {symbol} ({historicalData.Count} points), returning empty list");
                    return new List<double>();
                }

                List<double> result = new List<double>();

                // Calculate real indicator values based on historical price data
                switch (indicatorType.ToUpperInvariant())
                {
                    case "RSI":
                        var closingPrices = historicalData.Select(h => h.Close).ToList();
                        var rsiValues = CalculateRSIInternal(closingPrices, Math.Min(14, closingPrices.Count - 1));
                        result = rsiValues.Where(v => !double.IsNaN(v)).ToList();
                        break;

                    case "MACD":
                        var prices = historicalData.Select(h => h.Close).ToList();
                        var (macdLine, signalLine, histogram) = CalculateMACD(prices, 12, 26, 9);

                        // Return both MACD line and signal line values
                        result.AddRange(macdLine.Where(v => !double.IsNaN(v)));
                        result.AddRange(signalLine.Where(v => !double.IsNaN(v)));
                        break;

                    case "VOLUME":
                        result = historicalData.Select(h => (double)h.Volume).ToList();
                        break;

                    case "ADX":
                        var highs = historicalData.Select(h => h.High).ToList();
                        var lows = historicalData.Select(h => h.Low).ToList();
                        var closes = historicalData.Select(h => h.Close).ToList();
                        var adxValues = CalculateADXInternal(highs, lows, closes, Math.Min(14, closes.Count / 2));
                        result = adxValues.Where(v => !double.IsNaN(v)).ToList();
                        break;

                    case "ROC":
                        var rocPrices = historicalData.Select(h => h.Close).ToList();
                        var rocValues = CalculateROC(rocPrices, Math.Min(10, rocPrices.Count / 2));
                        result = rocValues.Where(v => !double.IsNaN(v)).ToList();
                        break;

                    case "BB_WIDTH":
                        // Bollinger Bands Width calculation
                        var bbPrices = historicalData.Select(h => h.Close).ToList();
                        var (upper, middle, lower) = CalculateBollingerBands(bbPrices, Math.Min(20, bbPrices.Count / 2), 2.0);

                        for (int i = 0; i < upper.Count && i < lower.Count; i++)
                        {
                            if (!double.IsNaN(upper[i]) && !double.IsNaN(lower[i]) && middle[i] != 0)
                            {
                                double width = (upper[i] - lower[i]) / middle[i] * 100; // Width as percentage
                                result.Add(width);
                            }
                        }
                        break;

                    default:
                        _loggingService.Log("Warning", $"Unknown indicator type: {indicatorType}. Returning empty list");
                        return new List<double>();
                }

                _loggingService.Log("Info", $"Calculated {result.Count} real {indicatorType} values for {symbol} from {historicalData.Count} historical data points");
                return result;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error calculating real historical data for {indicatorType} on {symbol}");
                return new List<double>(); // Return empty list on error
            }
        }

        /// <summary>
        /// Helper method to calculate Bollinger Bands for internal use
        /// </summary>
        private (List<double> upper, List<double> middle, List<double> lower) CalculateBollingerBands(List<double> prices, int period, double stdDevMultiplier)
        {
            var result = (Upper: new List<double>(), Middle: new List<double>(), Lower: new List<double>());

            // Calculate Simple Moving Average (SMA)
            var sma = CalculateSMAInternal(prices, period);
            result.Middle = sma;

            // Calculate standard deviation for each window
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period - 1)
                {
                    // Not enough data for full window
                    result.Upper.Add(double.NaN);
                    result.Lower.Add(double.NaN);
                    continue;
                }

                // Get window of prices for calculating std dev
                var window = prices.Skip(i - period + 1).Take(period).ToList();
                var mean = sma[i];
                var stdDev = Math.Sqrt(window.Average(v => Math.Pow(v - mean, 2)));

                // Calculate upper and lower bands
                result.Upper.Add(mean + stdDevMultiplier * stdDev);
                result.Lower.Add(mean - stdDevMultiplier * stdDev);
            }

            return result;
        }

        /// <summary>
        /// Helper method to calculate MACD for internal use
        /// </summary>
        private (List<double> MacdLine, List<double> SignalLine, List<double> Histogram) CalculateMACD(List<double> prices, int fastPeriod, int slowPeriod, int signalPeriod)
        {
            var result = (MacdLine: new List<double>(), SignalLine: new List<double>(), Histogram: new List<double>());

            // Calculate EMAs
            var fastEMA = CalculateEMAInternal(prices, fastPeriod);
            var slowEMA = CalculateEMAInternal(prices, slowPeriod);

            // Calculate MACD line
            var macdLine = new List<double>();
            for (int i = 0; i < prices.Count; i++)
            {
                if (i < slowPeriod - 1)
                {
                    // Not enough data for slow EMA
                    macdLine.Add(double.NaN);
                }
                else
                {
                    // MACD = Fast EMA - Slow EMA
                    macdLine.Add(fastEMA[i] - slowEMA[i]);
                }
            }

            // Calculate signal line (EMA of MACD line)
            var signalLine = CalculateEMAInternal(macdLine, signalPeriod);

            // Calculate histogram (MACD - Signal)
            var histogram = new List<double>();
            for (int i = 0; i < macdLine.Count; i++)
            {
                if (i < slowPeriod + signalPeriod - 2)
                {
                    // Not enough data for signal line
                    histogram.Add(double.NaN);
                }
                else
                {
                    // Histogram = MACD - Signal
                    histogram.Add(macdLine[i] - signalLine[i]);
                }
            }

            result.MacdLine = macdLine;
            result.SignalLine = signalLine;
            result.Histogram = histogram;

            return result;
        }

        /// <summary>
        /// Helper method to calculate Rate of Change for internal use
        /// </summary>
        private List<double> CalculateROC(List<double> prices, int period = 10)
        {
            var result = new List<double>();

            for (int i = 0; i < prices.Count; i++)
            {
                if (i < period)
                {
                    // Not enough data for ROC calculation
                    result.Add(double.NaN);
                }
                else
                {
                    // ROC = ((current - previous) / previous) * 100
                    double current = prices[i];
                    double previous = prices[i - period];

                    if (previous == 0)
                    {
                        result.Add(0);
                    }
                    else
                    {
                        double roc = (current - previous) / previous * 100;
                        result.Add(roc);
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Gets cached fundamental data for a symbol if available and not expired
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="dataType">Type of fundamental data (e.g., "PE_RATIO", "RSI", "VWAP", "MACD")</param>
        /// <param name="maxCacheAgeHours">Maximum age of cached data in hours (default 2 hours)</param>
        /// <returns>Cached value if available and valid, null otherwise</returns>
        public double? GetCachedFundamentalData(string symbol, string dataType, double maxCacheAgeHours = 2.0)
        {
            if (string.IsNullOrEmpty(symbol) || string.IsNullOrEmpty(dataType))
                return null;

            var cacheKey = $"{symbol}_{dataType}";

            lock (_cacheLock)
            {
                if (_fundamentalDataCache.TryGetValue(cacheKey, out var cachedData))
                {
                    var cacheAge = DateTime.Now - cachedData.Timestamp;

                    if (cacheAge.TotalHours <= maxCacheAgeHours)
                    {
                        _loggingService.Log("Info", $"Retrieved cached {dataType} for {symbol} (age: {cacheAge.TotalMinutes:F1} minutes)");
                        return cachedData.Value;
                    }
                    else
                    {
                        // Cache expired, remove it
                        _fundamentalDataCache.Remove(cacheKey);
                        _loggingService.Log("Info", $"Cache expired for {dataType} on {symbol} (age: {cacheAge.TotalHours:F1} hours)");
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Stores fundamental data in cache with current timestamp
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="dataType">Type of fundamental data</param>
        /// <param name="value">Value to cache</param>
        private void CacheFundamentalData(string symbol, string dataType, double value)
        {
            if (string.IsNullOrEmpty(symbol) || string.IsNullOrEmpty(dataType))
                return;

            var cacheKey = $"{symbol}_{dataType}";

            lock (_cacheLock)
            {
                _fundamentalDataCache[cacheKey] = (value, DateTime.Now);
                _loggingService.Log("Info", $"Cached {dataType} for {symbol}: {value}");
            }
        }

        /// <summary>
        /// Clears all cached fundamental data for a specific symbol
        /// </summary>
        /// <param name="symbol">Stock symbol to clear cache for</param>
        public void ClearCachedFundamentalData(string symbol)
        {
            if (string.IsNullOrEmpty(symbol))
                return;

            lock (_cacheLock)
            {
                var keysToRemove = _fundamentalDataCache.Keys
                .Where(k => k.StartsWith($"{symbol}_"))
                        .ToList();

                foreach (var key in keysToRemove)
                {
                    _fundamentalDataCache.Remove(key);
                }

                if (keysToRemove.Count > 0)
                {
                    _loggingService.Log("Info", $"Cleared {keysToRemove.Count} cached fundamental data entries for {symbol}");
                }
            }
        }

        /// <summary>
        /// Clears all expired fundamental data from cache
        /// </summary>
        /// <param name="maxAgeHours">Maximum age in hours</param>
        public void ClearExpiredFundamentalData(double maxAgeHours = 24.0)
        {
            lock (_cacheLock)
            {
                var expiredKeys = _fundamentalDataCache
                .Where(kvp => (DateTime.Now - kvp.Value.Timestamp).TotalHours > maxAgeHours)
              .Select(kvp => kvp.Key)
               .ToList();

                foreach (var key in expiredKeys)
                {
                    _fundamentalDataCache.Remove(key);
                }

                if (expiredKeys.Count > 0)
                {
                    _loggingService.Log("Info", $"Cleared {expiredKeys.Count} expired fundamental data cache entries");
                }
            }
        }

        /// <summary>
        /// Searches for symbols using the Alpha Vantage SYMBOL_SEARCH endpoint
        /// </summary>
        /// <param name="keywords">Keywords to search for (e.g., "BA", "Boeing", "AAPL")</param>
        /// <returns>List of matching symbols with details</returns>
        public async Task<List<SymbolSearchResult>> SearchSymbolsAsync(string keywords)
        {
            if (string.IsNullOrWhiteSpace(keywords))
                return new List<SymbolSearchResult>();

            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=SYMBOL_SEARCH&keywords={Uri.EscapeDataString(keywords)}&apikey={_apiKey}";
                await LogApiCall("SYMBOL_SEARCH", keywords);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    var results = new List<SymbolSearchResult>();

                    if (data["bestMatches"] is JArray bestMatches)
                    {
                        foreach (var match in bestMatches)
                        {
                            results.Add(new SymbolSearchResult
                            {
                                Symbol = match["1. symbol"]?.ToString() ?? "",
                                Name = match["2. name"]?.ToString() ?? "",
                                Type = match["3. type"]?.ToString() ?? "",
                                Region = match["4. region"]?.ToString() ?? "",
                                MarketOpen = match["5. marketOpen"]?.ToString() ?? "",
                                MarketClose = match["6. marketClose"]?.ToString() ?? "",
                                Timezone = match["7. timezone"]?.ToString() ?? "",
                                Currency = match["8. currency"]?.ToString() ?? "",
                                MatchScore = TryParseDouble(match["9. matchScore"])
                            });
                        }
                    }

                    _loggingService.Log("Info", $"Symbol search for '{keywords}' returned {results.Count} results");
                    return results;
                }

                return new List<SymbolSearchResult>();
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error searching symbols for '{keywords}'");
                return new List<SymbolSearchResult>();
            }
        }

        #region Fundamental Data API Methods

        /// <summary>
        /// Gets company overview data from Alpha Vantage OVERVIEW endpoint with 7-day caching
        /// Used for TFT static metadata features (Sector, MarketCap, Exchange)
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>CompanyOverview object with fundamental data</returns>
        public async Task<CompanyOverview> GetCompanyOverviewAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            // Check cache first (7-day expiry for static metadata)
            lock (_companyOverviewCacheLock)
            {
                if (_companyOverviewCache.TryGetValue(symbol.ToUpperInvariant(), out var cached))
                {
                    var cacheAge = DateTime.Now - cached.Timestamp;
                    if (cacheAge.TotalDays <= CompanyOverviewCacheDays)
                    {
                        _loggingService.Log("Info", $"Using cached company overview for {symbol} (age: {cacheAge.TotalHours:F1} hours)");
                        return cached.Overview;
                    }
                    else
                    {
                        // Cache expired, remove it
                        _companyOverviewCache.Remove(symbol.ToUpperInvariant());
                        _loggingService.Log("Info", $"Cache expired for company overview {symbol} (age: {cacheAge.TotalDays:F1} days)");
                    }
                }
            }

            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=OVERVIEW&symbol={Uri.EscapeDataString(symbol)}&apikey={_apiKey}";
                await LogApiCall("OVERVIEW", symbol);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    if (data["Symbol"] == null || string.IsNullOrEmpty(data["Symbol"].ToString()))
                    {
                        _loggingService.Log("Warning", $"No overview data found for {symbol}");
                        return null;
                    }

                    var overview = new CompanyOverview
                    {
                        Symbol = data["Symbol"]?.ToString(),
                        Name = data["Name"]?.ToString(),
                        Description = data["Description"]?.ToString(),
                        Exchange = data["Exchange"]?.ToString(),
                        Currency = data["Currency"]?.ToString(),
                        Country = data["Country"]?.ToString(),
                        Sector = data["Sector"]?.ToString(),
                        Industry = data["Industry"]?.ToString(),
                        Address = data["Address"]?.ToString(),
                        FiscalYearEnd = data["FiscalYearEnd"]?.ToString(),
                        MarketCapitalization = TryParseDecimal(data["MarketCapitalization"]),
                        EBITDA = TryParseDecimal(data["EBITDA"]),
                        PERatio = TryParseDecimal(data["PERatio"]),
                        PEGRatio = TryParseDecimal(data["PEGRatio"]),
                        BookValue = TryParseDecimal(data["BookValue"]),
                        DividendPerShare = TryParseDecimal(data["DividendPerShare"]),
                        DividendYield = TryParseDecimal(data["DividendYield"]),
                        EPS = TryParseDecimal(data["EPS"]),
                        RevenuePerShareTTM = TryParseDecimal(data["RevenuePerShareTTM"]),
                        ProfitMargin = TryParseDecimal(data["ProfitMargin"]),
                        OperatingMarginTTM = TryParseDecimal(data["OperatingMarginTTM"]),
                        ReturnOnAssetsTTM = TryParseDecimal(data["ReturnOnAssetsTTM"]),
                        ReturnOnEquityTTM = TryParseDecimal(data["ReturnOnEquityTTM"]),
                        RevenueTTM = TryParseDecimal(data["RevenueTTM"]),
                        GrossProfitTTM = TryParseDecimal(data["GrossProfitTTM"]),
                        DilutedEPSTTM = TryParseDecimal(data["DilutedEPSTTM"]),
                        QuarterlyEarningsGrowthYOY = TryParseDecimal(data["QuarterlyEarningsGrowthYOY"]),
                        QuarterlyRevenueGrowthYOY = TryParseDecimal(data["QuarterlyRevenueGrowthYOY"]),
                        AnalystTargetPrice = TryParseDecimal(data["AnalystTargetPrice"]),
                        TrailingPE = TryParseDecimal(data["TrailingPE"]),
                        ForwardPE = TryParseDecimal(data["ForwardPE"]),
                        PriceToSalesRatioTTM = TryParseDecimal(data["PriceToSalesRatioTTM"]),
                        PriceToBookRatio = TryParseDecimal(data["PriceToBookRatio"]),
                        EVToRevenue = TryParseDecimal(data["EVToRevenue"]),
                        EVToEBITDA = TryParseDecimal(data["EVToEBITDA"]),
                        Beta = TryParseDecimal(data["Beta"]),
                        Week52High = TryParseDecimal(data["52WeekHigh"]),
                        Week52Low = TryParseDecimal(data["52WeekLow"]),
                        Day50MovingAverage = TryParseDecimal(data["50DayMovingAverage"]),
                        Day200MovingAverage = TryParseDecimal(data["200DayMovingAverage"]),
                        SharesOutstanding = TryParseLong(data["SharesOutstanding"]),
                        DividendDate = data["DividendDate"]?.ToString(),
                        ExDividendDate = data["ExDividendDate"]?.ToString(),
                        LastUpdated = DateTime.Now
                    };

                    // Cache the overview for 7 days
                    lock (_companyOverviewCacheLock)
                    {
                        _companyOverviewCache[symbol.ToUpperInvariant()] = (overview, DateTime.Now);
                    }

                    _loggingService.Log("Info", $"Retrieved and cached company overview for {symbol}");
                    return overview;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error getting company overview for '{symbol}'");
                return null;
            }
        }

        /// <summary>
        /// Gets sector code as numerical value for ML/TFT model
        /// Maps sector names to numeric codes for static metadata features
        /// </summary>
        /// <param name="sector">Sector name from company overview</param>
        /// <returns>Numeric code: Technology=0, Healthcare=1, Financial=2, Consumer Cyclical=3, 
        /// Consumer Defensive=4, Industrial=5, Energy=6, Basic Materials=7, Real Estate=8, 
        /// Utilities=9, Communication Services=10, Unknown=-1</returns>
        public int GetSectorCode(string sector)
        {
            if (string.IsNullOrWhiteSpace(sector))
                return -1;

            // Normalize sector name for matching
            var normalizedSector = sector.ToUpperInvariant().Trim();

            return normalizedSector switch
            {
                "TECHNOLOGY" or "INFORMATION TECHNOLOGY" or "TECH" => 0,
                "HEALTHCARE" or "HEALTH CARE" => 1,
                "FINANCIAL SERVICES" or "FINANCIALS" or "FINANCIAL" => 2,
                "CONSUMER CYCLICAL" or "CONSUMER DISCRETIONARY" => 3,
                "CONSUMER DEFENSIVE" or "CONSUMER STAPLES" => 4,
                "INDUSTRIALS" or "INDUSTRIAL" => 5,
                "ENERGY" => 6,
                "BASIC MATERIALS" or "MATERIALS" => 7,
                "REAL ESTATE" => 8,
                "UTILITIES" => 9,
                "COMMUNICATION SERVICES" or "TELECOMMUNICATIONS" or "TELECOM" => 10,
                _ => -1 // Unknown sector
            };
        }

        /// <summary>
        /// Gets market cap category as numerical value for ML/TFT model
        /// Categories based on standard market cap classifications
        /// </summary>
        /// <param name="marketCap">Market capitalization in dollars</param>
        /// <returns>Numeric code: Small-cap (less than 2B)=0, Mid-cap (2B-10B)=1, 
        /// Large-cap (10B-200B)=2, Mega-cap (greater than 200B)=3, Unknown=-1</returns>
        public int GetMarketCapCategory(decimal? marketCap)
        {
            if (!marketCap.HasValue || marketCap.Value <= 0)
                return -1;

            var value = marketCap.Value;

            // Use class-level constants for thresholds
            if (value < SmallCapMaxThreshold)
                return 0; // Small-cap
            if (value < MidCapMaxThreshold)
                return 1; // Mid-cap
            if (value < LargeCapMaxThreshold)
                return 2; // Large-cap

            return 3; // Mega-cap
        }

        /// <summary>
        /// Gets market cap category as numerical value for ML/TFT model (long overload)
        /// </summary>
        /// <param name="marketCap">Market capitalization in dollars as long</param>
        /// <returns>Numeric code for market cap category</returns>
        public int GetMarketCapCategory(long marketCap)
        {
            return GetMarketCapCategory((decimal)marketCap);
        }

        /// <summary>
        /// Gets exchange code as numerical value for ML/TFT model
        /// Maps exchange names to numeric codes for static metadata features
        /// </summary>
        /// <param name="exchange">Exchange name from company overview</param>
        /// <returns>Numeric code: NYSE=0, NASDAQ=1, AMEX=2, Other=3, Unknown=-1</returns>
        public int GetExchangeCode(string exchange)
        {
            if (string.IsNullOrWhiteSpace(exchange))
                return -1;

            // Normalize exchange name for matching
            var normalizedExchange = exchange.ToUpperInvariant().Trim();

            // Check exact matches first
            return normalizedExchange switch
            {
                "NYSE" or "NEW YORK STOCK EXCHANGE" => 0,
                "NASDAQ" or "NASDAQ GLOBAL SELECT" or "NASDAQ GLOBAL MARKET" or "NASDAQ CAPITAL MARKET" => 1,
                "AMEX" or "NYSE AMERICAN" or "NYSE MKT" or "AMERICAN STOCK EXCHANGE" => 2,
                "BATS" or "IEX" or "CBOE" or "ARCA" or "NYSE ARCA" => 3, // Other US exchanges
                _ => GetExchangeCodeByPartialMatch(normalizedExchange)
            };
        }

        /// <summary>
        /// Helper method to determine exchange code by partial string matching
        /// Used as fallback when exact match is not found
        /// </summary>
        private static int GetExchangeCodeByPartialMatch(string normalizedExchange)
        {
            if (normalizedExchange.Contains("NYSE"))
                return 0;
            if (normalizedExchange.Contains("NASDAQ"))
                return 1;
            if (normalizedExchange.Contains("AMEX"))
                return 2;
            
            return 3; // Other/Unknown exchange
        }

        /// <summary>
        /// Clears expired company overview cache entries
        /// </summary>
        public void ClearExpiredCompanyOverviewCache()
        {
            lock (_companyOverviewCacheLock)
            {
                var expiredKeys = _companyOverviewCache
                    .Where(kvp => (DateTime.Now - kvp.Value.Timestamp).TotalDays > CompanyOverviewCacheDays)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in expiredKeys)
                {
                    _companyOverviewCache.Remove(key);
                }

                if (expiredKeys.Count > 0)
                {
                    _loggingService.Log("Info", $"Cleared {expiredKeys.Count} expired company overview cache entries");
                }
            }
        }

        /// <summary>
        /// Gets income statement data from Alpha Vantage INCOME_STATEMENT endpoint
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>IncomeStatement object with annual and quarterly reports</returns>
        public async Task<IncomeStatement> GetIncomeStatementAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=INCOME_STATEMENT&symbol={Uri.EscapeDataString(symbol)}&apikey={_apiKey}";
                await LogApiCall("INCOME_STATEMENT", symbol);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    if (data["symbol"] == null && data["Symbol"] == null)
                    {
                        _loggingService.Log("Warning", $"No income statement data found for {symbol}");
                        return null;
                    }

                    var incomeStatement = new IncomeStatement
                    {
                        Symbol = data["symbol"]?.ToString() ?? symbol,
                        LastUpdated = DateTime.Now
                    };

                    // Parse annual reports
                    if (data["annualReports"] is JArray annualReports)
                    {
                        foreach (var report in annualReports)
                        {
                            incomeStatement.AnnualReports.Add(ParseIncomeStatementReport(report));
                        }
                    }

                    // Parse quarterly reports
                    if (data["quarterlyReports"] is JArray quarterlyReports)
                    {
                        foreach (var report in quarterlyReports)
                        {
                            incomeStatement.QuarterlyReports.Add(ParseIncomeStatementReport(report));
                        }
                    }

                    _loggingService.Log("Info", $"Retrieved income statement for {symbol}: {incomeStatement.AnnualReports.Count} annual, {incomeStatement.QuarterlyReports.Count} quarterly reports");
                    return incomeStatement;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error getting income statement for '{symbol}'");
                return null;
            }
        }

        /// <summary>
        /// Gets balance sheet data from Alpha Vantage BALANCE_SHEET endpoint
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>BalanceSheet object with annual and quarterly reports</returns>
        public async Task<BalanceSheet> GetBalanceSheetAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=BALANCE_SHEET&symbol={Uri.EscapeDataString(symbol)}&apikey={_apiKey}";
                await LogApiCall("BALANCE_SHEET", symbol);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    if (data["symbol"] == null && data["Symbol"] == null)
                    {
                        _loggingService.Log("Warning", $"No balance sheet data found for {symbol}");
                        return null;
                    }

                    var balanceSheet = new BalanceSheet
                    {
                        Symbol = data["symbol"]?.ToString() ?? symbol,
                        LastUpdated = DateTime.Now
                    };

                    // Parse annual reports
                    if (data["annualReports"] is JArray annualReports)
                    {
                        foreach (var report in annualReports)
                        {
                            balanceSheet.AnnualReports.Add(ParseBalanceSheetReport(report));
                        }
                    }

                    // Parse quarterly reports
                    if (data["quarterlyReports"] is JArray quarterlyReports)
                    {
                        foreach (var report in quarterlyReports)
                        {
                            balanceSheet.QuarterlyReports.Add(ParseBalanceSheetReport(report));
                        }
                    }

                    _loggingService.Log("Info", $"Retrieved balance sheet for {symbol}: {balanceSheet.AnnualReports.Count} annual, {balanceSheet.QuarterlyReports.Count} quarterly reports");
                    return balanceSheet;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error getting balance sheet for '{symbol}'");
                return null;
            }
        }

        /// <summary>
        /// Gets cash flow data from Alpha Vantage CASH_FLOW endpoint
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>CashFlowStatement object with annual and quarterly reports</returns>
        public async Task<CashFlowStatement> GetCashFlowAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=CASH_FLOW&symbol={Uri.EscapeDataString(symbol)}&apikey={_apiKey}";
                await LogApiCall("CASH_FLOW", symbol);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    if (data["symbol"] == null && data["Symbol"] == null)
                    {
                        _loggingService.Log("Warning", $"No cash flow data found for {symbol}");
                        return null;
                    }

                    var cashFlow = new CashFlowStatement
                    {
                        Symbol = data["symbol"]?.ToString() ?? symbol,
                        LastUpdated = DateTime.Now
                    };

                    // Parse annual reports
                    if (data["annualReports"] is JArray annualReports)
                    {
                        foreach (var report in annualReports)
                        {
                            cashFlow.AnnualReports.Add(ParseCashFlowReport(report));
                        }
                    }

                    // Parse quarterly reports
                    if (data["quarterlyReports"] is JArray quarterlyReports)
                    {
                        foreach (var report in quarterlyReports)
                        {
                            cashFlow.QuarterlyReports.Add(ParseCashFlowReport(report));
                        }
                    }

                    _loggingService.Log("Info", $"Retrieved cash flow for {symbol}: {cashFlow.AnnualReports.Count} annual, {cashFlow.QuarterlyReports.Count} quarterly reports");
                    return cashFlow;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error getting cash flow for '{symbol}'");
                return null;
            }
        }

        /// <summary>
        /// Gets earnings data from Alpha Vantage EARNINGS endpoint
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>EarningsData object with annual and quarterly earnings</returns>
        public async Task<EarningsData> GetEarningsAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                return null;

            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=EARNINGS&symbol={Uri.EscapeDataString(symbol)}&apikey={_apiKey}";
                await LogApiCall("EARNINGS", symbol);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    if (data["symbol"] == null && data["Symbol"] == null)
                    {
                        _loggingService.Log("Warning", $"No earnings data found for {symbol}");
                        return null;
                    }

                    var earnings = new EarningsData
                    {
                        Symbol = data["symbol"]?.ToString() ?? symbol,
                        LastUpdated = DateTime.Now
                    };

                    // Parse annual earnings
                    if (data["annualEarnings"] is JArray annualEarnings)
                    {
                        foreach (var report in annualEarnings)
                        {
                            earnings.AnnualEarnings.Add(new AnnualEarningsReport
                            {
                                FiscalDateEnding = report["fiscalDateEnding"]?.ToString(),
                                ReportedEPS = TryParseDecimal(report["reportedEPS"])
                            });
                        }
                    }

                    // Parse quarterly earnings
                    if (data["quarterlyEarnings"] is JArray quarterlyEarnings)
                    {
                        foreach (var report in quarterlyEarnings)
                        {
                            earnings.QuarterlyEarnings.Add(new QuarterlyEarningsReport
                            {
                                FiscalDateEnding = report["fiscalDateEnding"]?.ToString(),
                                ReportedDate = report["reportedDate"]?.ToString(),
                                ReportedEPS = TryParseDecimal(report["reportedEPS"]),
                                EstimatedEPS = TryParseDecimal(report["estimatedEPS"]),
                                Surprise = TryParseDecimal(report["surprise"]),
                                SurprisePercentage = TryParseDecimal(report["surprisePercentage"])
                            });
                        }
                    }

                    _loggingService.Log("Info", $"Retrieved earnings for {symbol}: {earnings.AnnualEarnings.Count} annual, {earnings.QuarterlyEarnings.Count} quarterly reports");
                    return earnings;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error getting earnings for '{symbol}'");
                return null;
            }
        }

        #region Fundamental Data Helper Methods

        private IncomeStatementReport ParseIncomeStatementReport(JToken report)
        {
            return new IncomeStatementReport
            {
                FiscalDateEnding = report["fiscalDateEnding"]?.ToString(),
                ReportedCurrency = report["reportedCurrency"]?.ToString(),
                GrossProfit = TryParseDecimal(report["grossProfit"]),
                TotalRevenue = TryParseDecimal(report["totalRevenue"]),
                CostOfRevenue = TryParseDecimal(report["costOfRevenue"]),
                CostOfGoodsAndServicesSold = TryParseDecimal(report["costofGoodsAndServicesSold"]),
                OperatingIncome = TryParseDecimal(report["operatingIncome"]),
                SellingGeneralAndAdministrative = TryParseDecimal(report["sellingGeneralAndAdministrative"]),
                ResearchAndDevelopment = TryParseDecimal(report["researchAndDevelopment"]),
                OperatingExpenses = TryParseDecimal(report["operatingExpenses"]),
                InvestmentIncomeNet = TryParseDecimal(report["investmentIncomeNet"]),
                NetInterestIncome = TryParseDecimal(report["netInterestIncome"]),
                InterestIncome = TryParseDecimal(report["interestIncome"]),
                InterestExpense = TryParseDecimal(report["interestExpense"]),
                NonInterestIncome = TryParseDecimal(report["nonInterestIncome"]),
                OtherNonOperatingIncome = TryParseDecimal(report["otherNonOperatingIncome"]),
                Depreciation = TryParseDecimal(report["depreciation"]),
                DepreciationAndAmortization = TryParseDecimal(report["depreciationAndAmortization"]),
                IncomeBeforeTax = TryParseDecimal(report["incomeBeforeTax"]),
                IncomeTaxExpense = TryParseDecimal(report["incomeTaxExpense"]),
                InterestAndDebtExpense = TryParseDecimal(report["interestAndDebtExpense"]),
                NetIncomeFromContinuingOperations = TryParseDecimal(report["netIncomeFromContinuingOperations"]),
                ComprehensiveIncomeNetOfTax = TryParseDecimal(report["comprehensiveIncomeNetOfTax"]),
                EBIT = TryParseDecimal(report["ebit"]),
                EBITDA = TryParseDecimal(report["ebitda"]),
                NetIncome = TryParseDecimal(report["netIncome"])
            };
        }

        private BalanceSheetReport ParseBalanceSheetReport(JToken report)
        {
            return new BalanceSheetReport
            {
                FiscalDateEnding = report["fiscalDateEnding"]?.ToString(),
                ReportedCurrency = report["reportedCurrency"]?.ToString(),
                TotalAssets = TryParseDecimal(report["totalAssets"]),
                TotalCurrentAssets = TryParseDecimal(report["totalCurrentAssets"]),
                CashAndCashEquivalentsAtCarryingValue = TryParseDecimal(report["cashAndCashEquivalentsAtCarryingValue"]),
                CashAndShortTermInvestments = TryParseDecimal(report["cashAndShortTermInvestments"]),
                Inventory = TryParseDecimal(report["inventory"]),
                CurrentNetReceivables = TryParseDecimal(report["currentNetReceivables"]),
                TotalNonCurrentAssets = TryParseDecimal(report["totalNonCurrentAssets"]),
                PropertyPlantEquipment = TryParseDecimal(report["propertyPlantEquipment"]),
                AccumulatedDepreciationAmortizationPPE = TryParseDecimal(report["accumulatedDepreciationAmortizationPPE"]),
                IntangibleAssets = TryParseDecimal(report["intangibleAssets"]),
                IntangibleAssetsExcludingGoodwill = TryParseDecimal(report["intangibleAssetsExcludingGoodwill"]),
                Goodwill = TryParseDecimal(report["goodwill"]),
                Investments = TryParseDecimal(report["investments"]),
                LongTermInvestments = TryParseDecimal(report["longTermInvestments"]),
                ShortTermInvestments = TryParseDecimal(report["shortTermInvestments"]),
                OtherCurrentAssets = TryParseDecimal(report["otherCurrentAssets"]),
                OtherNonCurrentAssets = TryParseDecimal(report["otherNonCurrrentAssets"]),
                TotalLiabilities = TryParseDecimal(report["totalLiabilities"]),
                TotalCurrentLiabilities = TryParseDecimal(report["totalCurrentLiabilities"]),
                CurrentAccountsPayable = TryParseDecimal(report["currentAccountsPayable"]),
                DeferredRevenue = TryParseDecimal(report["deferredRevenue"]),
                CurrentDebt = TryParseDecimal(report["currentDebt"]),
                ShortTermDebt = TryParseDecimal(report["shortTermDebt"]),
                TotalNonCurrentLiabilities = TryParseDecimal(report["totalNonCurrentLiabilities"]),
                CapitalLeaseObligations = TryParseDecimal(report["capitalLeaseObligations"]),
                LongTermDebt = TryParseDecimal(report["longTermDebt"]),
                CurrentLongTermDebt = TryParseDecimal(report["currentLongTermDebt"]),
                LongTermDebtNoncurrent = TryParseDecimal(report["longTermDebtNoncurrent"]),
                ShortLongTermDebtTotal = TryParseDecimal(report["shortLongTermDebtTotal"]),
                OtherCurrentLiabilities = TryParseDecimal(report["otherCurrentLiabilities"]),
                OtherNonCurrentLiabilities = TryParseDecimal(report["otherNonCurrentLiabilities"]),
                TotalShareholderEquity = TryParseDecimal(report["totalShareholderEquity"]),
                TreasuryStock = TryParseDecimal(report["treasuryStock"]),
                RetainedEarnings = TryParseDecimal(report["retainedEarnings"]),
                CommonStock = TryParseDecimal(report["commonStock"]),
                CommonStockSharesOutstanding = TryParseDecimal(report["commonStockSharesOutstanding"])
            };
        }

        private CashFlowReport ParseCashFlowReport(JToken report)
        {
            return new CashFlowReport
            {
                FiscalDateEnding = report["fiscalDateEnding"]?.ToString(),
                ReportedCurrency = report["reportedCurrency"]?.ToString(),
                OperatingCashflow = TryParseDecimal(report["operatingCashflow"]),
                PaymentsForOperatingActivities = TryParseDecimal(report["paymentsForOperatingActivities"]),
                ProceedsFromOperatingActivities = TryParseDecimal(report["proceedsFromOperatingActivities"]),
                ChangeInOperatingLiabilities = TryParseDecimal(report["changeInOperatingLiabilities"]),
                ChangeInOperatingAssets = TryParseDecimal(report["changeInOperatingAssets"]),
                DepreciationDepletionAndAmortization = TryParseDecimal(report["depreciationDepletionAndAmortization"]),
                CapitalExpenditures = TryParseDecimal(report["capitalExpenditures"]),
                ChangeInReceivables = TryParseDecimal(report["changeInReceivables"]),
                ChangeInInventory = TryParseDecimal(report["changeInInventory"]),
                ProfitLoss = TryParseDecimal(report["profitLoss"]),
                CashflowFromInvestment = TryParseDecimal(report["cashflowFromInvestment"]),
                CashflowFromFinancing = TryParseDecimal(report["cashflowFromFinancing"]),
                ProceedsFromRepaymentsOfShortTermDebt = TryParseDecimal(report["proceedsFromRepaymentsOfShortTermDebt"]),
                PaymentsForRepurchaseOfCommonStock = TryParseDecimal(report["paymentsForRepurchaseOfCommonStock"]),
                PaymentsForRepurchaseOfEquity = TryParseDecimal(report["paymentsForRepurchaseOfEquity"]),
                PaymentsForRepurchaseOfPreferredStock = TryParseDecimal(report["paymentsForRepurchaseOfPreferredStock"]),
                DividendPayout = TryParseDecimal(report["dividendPayout"]),
                DividendPayoutCommonStock = TryParseDecimal(report["dividendPayoutCommonStock"]),
                DividendPayoutPreferredStock = TryParseDecimal(report["dividendPayoutPreferredStock"]),
                ProceedsFromIssuanceOfCommonStock = TryParseDecimal(report["proceedsFromIssuanceOfCommonStock"]),
                ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet = TryParseDecimal(report["proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet"]),
                ProceedsFromIssuanceOfPreferredStock = TryParseDecimal(report["proceedsFromIssuanceOfPreferredStock"]),
                ProceedsFromRepurchaseOfEquity = TryParseDecimal(report["proceedsFromRepurchaseOfEquity"]),
                ProceedsFromSaleOfTreasuryStock = TryParseDecimal(report["proceedsFromSaleOfTreasuryStock"]),
                ChangeInCashAndCashEquivalents = TryParseDecimal(report["changeInCashAndCashEquivalents"]),
                ChangeInExchangeRate = TryParseDecimal(report["changeInExchangeRate"]),
                NetIncome = TryParseDecimal(report["netIncome"])
            };
        }

        /// <summary>
        /// Safely parse a JSON token to decimal, returning null if null or invalid
        /// </summary>
        private static decimal? TryParseDecimal(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return null;

            var value = token.ToString();
            if (string.IsNullOrEmpty(value) || value == "None" || value == "-")
                return null;

            if (decimal.TryParse(value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out decimal result))
                return result;

            return null;
        }

        /// <summary>
        /// Safely parse a JSON token to long, returning null if null or invalid
        /// </summary>
        private static long? TryParseLong(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return null;

            var value = token.ToString();
            if (string.IsNullOrEmpty(value) || value == "None" || value == "-")
                return null;

            if (long.TryParse(value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out long result))
                return result;

            return null;
        }

        #endregion

        #endregion

        #region Intelligence and News API Methods

        /// <summary>
        /// Gets news sentiment data from Alpha Vantage NEWS_SENTIMENT endpoint
        /// </summary>
        /// <param name="tickers">Optional comma-separated list of stock symbols (e.g., "AAPL,MSFT")</param>
        /// <param name="topics">Optional comma-separated list of topics (e.g., "technology,earnings")</param>
        /// <param name="timeFrom">Optional start time in YYYYMMDDTHHMM format</param>
        /// <param name="timeTo">Optional end time in YYYYMMDDTHHMM format</param>
        /// <param name="sort">Sort order: LATEST, EARLIEST, RELEVANCE (default: LATEST)</param>
        /// <param name="limit">Number of results (1-1000, default: 50)</param>
        /// <returns>NewsSentimentResponse with list of news items</returns>
        public async Task<NewsSentimentResponse> GetNewsSentimentAsync(
            string tickers = null,
            string topics = null,
            string timeFrom = null,
            string timeTo = null,
            string sort = "LATEST",
            int limit = 50)
        {
            try
            {
                await WaitForApiLimit();

                var queryParams = new List<string> { $"function=NEWS_SENTIMENT", $"apikey={_apiKey}" };

                if (!string.IsNullOrEmpty(tickers))
                    queryParams.Add($"tickers={Uri.EscapeDataString(tickers)}");

                if (!string.IsNullOrEmpty(topics))
                    queryParams.Add($"topics={Uri.EscapeDataString(topics)}");

                if (!string.IsNullOrEmpty(timeFrom))
                    queryParams.Add($"time_from={timeFrom}");

                if (!string.IsNullOrEmpty(timeTo))
                    queryParams.Add($"time_to={timeTo}");

                if (!string.IsNullOrEmpty(sort))
                    queryParams.Add($"sort={sort}");

                queryParams.Add($"limit={Math.Min(1000, Math.Max(1, limit))}");

                var endpoint = $"query?{string.Join("&", queryParams)}";
                await LogApiCall("NEWS_SENTIMENT", tickers ?? "all");

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    var result = new NewsSentimentResponse
                    {
                        ItemsCount = (int?)data["items"] ?? 0,
                        SentimentScoreDefinition = data["sentiment_score_definition"]?.ToString(),
                        RelevanceScoreDefinition = data["relevance_score_definition"]?.ToString()
                    };

                    if (data["feed"] is JArray feed)
                    {
                        foreach (var item in feed)
                        {
                            var newsItem = new NewsSentimentItem
                            {
                                Title = item["title"]?.ToString(),
                                Url = item["url"]?.ToString(),
                                Summary = item["summary"]?.ToString(),
                                BannerImage = item["banner_image"]?.ToString(),
                                Source = item["source"]?.ToString(),
                                CategoryWithinSource = item["category_within_source"]?.ToString(),
                                OverallSentimentScore = TryParseDouble(item["overall_sentiment_score"]),
                                OverallSentimentLabel = item["overall_sentiment_label"]?.ToString()
                            };

                            // Parse time_published
                            if (item["time_published"] != null)
                            {
                                var timeStr = item["time_published"].ToString();
                                if (DateTime.TryParseExact(timeStr, "yyyyMMddTHHmmss",
                                    System.Globalization.CultureInfo.InvariantCulture,
                                    System.Globalization.DateTimeStyles.None, out DateTime publishedTime))
                                {
                                    newsItem.TimePublished = publishedTime;
                                }
                            }

                            // Parse authors
                            if (item["authors"] is JArray authors)
                            {
                                foreach (var author in authors)
                                {
                                    newsItem.Authors.Add(author.ToString());
                                }
                            }

                            // Parse topics
                            if (item["topics"] is JArray topics_array)
                            {
                                foreach (var topic in topics_array)
                                {
                                    newsItem.Topics.Add(new TopicInfo
                                    {
                                        Topic = topic["topic"]?.ToString(),
                                        RelevanceScore = TryParseDouble(topic["relevance_score"])
                                    });
                                }
                            }

                            // Parse ticker sentiment
                            if (item["ticker_sentiment"] is JArray tickerSentiments)
                            {
                                foreach (var ts in tickerSentiments)
                                {
                                    newsItem.TickerSentiments.Add(new TickerSentiment
                                    {
                                        Ticker = ts["ticker"]?.ToString(),
                                        RelevanceScore = TryParseDouble(ts["relevance_score"]),
                                        SentimentScore = TryParseDouble(ts["ticker_sentiment_score"]),
                                        SentimentLabel = ts["ticker_sentiment_label"]?.ToString()
                                    });
                                }
                            }

                            result.Feed.Add(newsItem);
                        }
                    }

                    _loggingService.Log("Info", $"Retrieved {result.Feed.Count} news sentiment items");
                    return result;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error getting news sentiment data");
                return null;
            }
        }

        /// <summary>
        /// Gets top gainers, losers, and most actively traded stocks from Alpha Vantage TOP_GAINERS_LOSERS endpoint
        /// </summary>
        /// <returns>TopMoversResponse with lists of gainers, losers, and most active stocks</returns>
        public async Task<TopMoversResponse> GetTopMoversAsync()
        {
            try
            {
                await WaitForApiLimit();
                var endpoint = $"query?function=TOP_GAINERS_LOSERS&apikey={_apiKey}";
                await LogApiCall("TOP_GAINERS_LOSERS", null);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    var result = new TopMoversResponse
                    {
                        Metadata = data["metadata"]?.ToString(),
                        LastUpdated = DateTime.Now
                    };

                    // Parse top gainers
                    if (data["top_gainers"] is JArray gainers)
                    {
                        foreach (var item in gainers)
                        {
                            result.TopGainers.Add(ParseMarketMover(item, MarketMoverCategory.Gainer));
                        }
                    }

                    // Parse top losers
                    if (data["top_losers"] is JArray losers)
                    {
                        foreach (var item in losers)
                        {
                            result.TopLosers.Add(ParseMarketMover(item, MarketMoverCategory.Loser));
                        }
                    }

                    // Parse most actively traded
                    if (data["most_actively_traded"] is JArray mostActive)
                    {
                        foreach (var item in mostActive)
                        {
                            result.MostActivelyTraded.Add(ParseMarketMover(item, MarketMoverCategory.MostActive));
                        }
                    }

                    _loggingService.Log("Info", $"Retrieved top movers: {result.TopGainers.Count} gainers, {result.TopLosers.Count} losers, {result.MostActivelyTraded.Count} most active");
                    return result;
                }

                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error getting top movers data");
                return null;
            }
        }

        private MarketMover ParseMarketMover(JToken item, MarketMoverCategory category)
        {
            return new MarketMover
            {
                Ticker = item["ticker"]?.ToString(),
                Price = TryParseDouble(item["price"]),
                ChangeAmount = TryParseDouble(item["change_amount"]),
                ChangePercentage = TryParsePercentage(item["change_percentage"]),
                Volume = (long)TryParseDouble(item["volume"]),
                Category = category
            };
        }

        /// <summary>
        /// Gets insider transactions from Alpha Vantage INSIDER_TRANSACTIONS endpoint
        /// </summary>
        /// <param name="symbol">Stock ticker symbol (optional - if null, returns all transactions)</param>
        /// <param name="dateFrom">Optional start date for filtering transactions</param>
        /// <param name="dateTo">Optional end date for filtering transactions</param>
        /// <returns>InsiderTransactionsResponse with list of insider transactions</returns>
        public async Task<InsiderTransactionsResponse> GetInsiderTransactionsAsync(
            string symbol = null, 
            DateTime? dateFrom = null, 
            DateTime? dateTo = null)
        {
            try
            {
                await WaitForApiLimit();
                
                // Build query parameters
                var queryParams = new List<string>
                {
                    "function=INSIDER_TRANSACTIONS",
                    $"apikey={_apiKey}"
                };

                // Add symbol if provided
                if (!string.IsNullOrWhiteSpace(symbol))
                {
                    queryParams.Add($"symbol={Uri.EscapeDataString(symbol)}");
                }

                var endpoint = $"query?{string.Join("&", queryParams)}";
                await LogApiCall("INSIDER_TRANSACTIONS", symbol ?? "all");

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    var result = new InsiderTransactionsResponse
                    {
                        Symbol = symbol ?? "ALL"
                    };

                    if (data["data"] is JArray transactions)
                    {
                        foreach (var item in transactions)
                        {
                            // Parse dates first for filtering
                            DateTime? transactionDate = null;
                            if (DateTime.TryParse(item["transaction_date"]?.ToString(), out DateTime parsedTransactionDate))
                            {
                                transactionDate = parsedTransactionDate;
                            }

                            DateTime? filingDate = null;
                            if (DateTime.TryParse(item["filing_date"]?.ToString(), out DateTime parsedFilingDate))
                            {
                                filingDate = parsedFilingDate;
                            }

                            // Apply date range filtering if specified
                            if (dateFrom.HasValue && transactionDate.HasValue && transactionDate.Value < dateFrom.Value)
                                continue;

                            if (dateTo.HasValue && transactionDate.HasValue && transactionDate.Value > dateTo.Value)
                                continue;

                            var transaction = new InsiderTransactionData
                            {
                                Symbol = item["symbol"]?.ToString() ?? symbol ?? "N/A",
                                OwnerName = item["owner_name"]?.ToString(),
                                OwnerCik = item["owner_cik"]?.ToString(),
                                OwnerTitle = item["owner_title"]?.ToString(),
                                SecurityType = item["security_type"]?.ToString(),
                                TransactionCode = item["transaction_code"]?.ToString(),
                                SharesTraded = (int)TryParseDouble(item["shares"]),
                                PricePerShare = TryParseDouble(item["share_price"]),
                                SharesOwnedFollowing = (int)TryParseDouble(item["shares_owned_following"]),
                                AcquisitionOrDisposal = item["acquisition_or_disposal"]?.ToString(),
                                FilingDate = filingDate ?? DateTime.MinValue,
                                TransactionDate = transactionDate ?? DateTime.MinValue
                            };

                            result.Transactions.Add(transaction);
                        }
                    }

                    var logMessage = string.IsNullOrWhiteSpace(symbol) 
                        ? $"Retrieved {result.Transactions.Count} insider transactions across all symbols"
                        : $"Retrieved {result.Transactions.Count} insider transactions for {symbol}";
                    
                    if (dateFrom.HasValue || dateTo.HasValue)
                    {
                        logMessage += $" (date range: {dateFrom?.ToString("yyyy-MM-dd") ?? "any"} to {dateTo?.ToString("yyyy-MM-dd") ?? "any"})";
                    }

                    _loggingService.Log("Info", logMessage);
                    return result;
                }

                return null;
            }
            catch (Exception ex)
            {
                var errorContext = string.IsNullOrWhiteSpace(symbol) ? "all symbols" : symbol;
                _loggingService.LogErrorWithContext(ex, $"Error getting insider transactions for {errorContext}");
                return null;
            }
        }

        /// <summary>
        /// Gets the latest insider transactions across all symbols (convenience method)
        /// </summary>
        /// <param name="limit">Maximum number of transactions to return (default: 100)</param>
        /// <returns>InsiderTransactionsResponse with list of latest insider transactions sorted by transaction date</returns>
        public async Task<InsiderTransactionsResponse> GetLatestInsiderTransactionsAsync(int limit = 100)
        {
            try
            {
                // Get transactions without symbol filter
                var result = await GetInsiderTransactionsAsync(symbol: null, dateFrom: null, dateTo: null);

                if (result != null && result.Transactions.Count > 0)
                {
                    // Sort by transaction date descending (most recent first) and limit
                    result.Transactions = result.Transactions
                        .OrderByDescending(t => t.TransactionDate)
                        .Take(limit)
                        .ToList();

                    _loggingService.Log("Info", $"Retrieved {result.Transactions.Count} latest insider transactions (limited to {limit})");
                }

                return result;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error getting latest insider transactions");
                return null;
            }
        }

        #endregion

        #region Analytics API Methods

        /// <summary>
        /// Get fixed window analytics from Alpha Vantage ANALYTICS_FIXED_WINDOW endpoint
        /// Calculates statistical metrics across a fixed date range for multiple symbols
        /// </summary>
        /// <param name="symbols">Comma-separated list of symbols (e.g., "AAPL,SPY,QQQ")</param>
        /// <param name="startDate">Start date for the analysis window</param>
        /// <param name="interval">Time interval (DAILY, WEEKLY, MONTHLY)</param>
        /// <param name="calculations">Comma-separated list of calculations to perform</param>
        /// <param name="ohlcType">Price type to use (close, open, high, low)</param>
        /// <returns>Analytics result with calculated metrics</returns>
        public async Task<AnalyticsFixedWindowResult> GetAnalyticsFixedWindowAsync(
            string symbols,
            DateTime startDate,
            string interval = "DAILY",
            string calculations = "MEAN_VALUE,STDDEV,CORRELATION",
            string ohlcType = "close")
        {
            try
            {
                if (string.IsNullOrWhiteSpace(symbols))
                    throw new ArgumentException("Symbols cannot be null or empty", nameof(symbols));

                await WaitForApiLimit();

                // Format date as YYYY-MM-DD
                string rangeParam = startDate.ToString("yyyy-MM-dd");

                // Add annualized parameter to STDDEV if not already present
                if (calculations.Contains("STDDEV") && !calculations.Contains("annualized"))
                {
                    calculations = calculations.Replace("STDDEV", "STDDEV,STDDEV(annualized=True)");
                }

                // Build the API endpoint
                var endpoint = $"query?function=ANALYTICS_FIXED_WINDOW" +
                    $"&SYMBOLS={Uri.EscapeDataString(symbols)}" +
                    $"&RANGE={rangeParam}" +
                    $"&INTERVAL={interval}" +
                    $"&OHLC={ohlcType}" +
                    $"&CALCULATIONS={Uri.EscapeDataString(calculations)}" +
                    $"&apikey={_apiKey}";

                await LogApiCall("ANALYTICS_FIXED_WINDOW", $"{symbols}, {rangeParam}");

                var response = await _client.GetAsync(endpoint);
                var content = await response.Content.ReadAsStringAsync();

                if (!response.IsSuccessStatusCode)
                {
                    _loggingService.Log("Error", $"Analytics API request failed with status {response.StatusCode}: {content}");
                    return null;
                }

                // Parse the response
                var result = ParseAnalyticsFixedWindowResponse(content);

                if (result != null && result.IsValid)
                {
                    _loggingService.Log("Info", $"Retrieved analytics for {symbols} from {rangeParam}: " +
                        $"{result.MeanValues?.Count ?? 0} symbols analyzed");
                }
                else if (result?.Metadata?.HasError == true)
                {
                    _loggingService.Log("Warning", $"Analytics API returned error/note: {result.Metadata.Error ?? result.Metadata.Note}");
                }

                return result;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error getting analytics for '{symbols}'");
                return null;
            }
        }

        /// <summary>
        /// Parse ANALYTICS_FIXED_WINDOW API response
        /// </summary>
        private AnalyticsFixedWindowResult ParseAnalyticsFixedWindowResponse(string jsonResponse)
        {
            var result = new AnalyticsFixedWindowResult();

            try
            {
                var json = JObject.Parse(jsonResponse);

                // Check for errors or rate limit messages
                if (json["Error Message"] != null)
                {
                    result.Metadata.Error = json["Error Message"].ToString();
                    _loggingService.Log("Error", $"Analytics API Error: {result.Metadata.Error}");
                    return result;
                }

                if (json["Note"] != null)
                {
                    result.Metadata.Note = json["Note"].ToString();
                    _loggingService.Log("Warning", $"Analytics API Note: {result.Metadata.Note}");
                    return result;
                }

                if (json["Information"] != null)
                {
                    result.Metadata.Information = json["Information"].ToString();
                }

                // Parse the main analytics data
                var analytics = json["ANALYTICS_FIXED_WINDOW"];
                if (analytics == null)
                {
                    _loggingService.Log("Warning", "No ANALYTICS_FIXED_WINDOW data in response");
                    return result;
                }

                // Parse metadata
                result.Symbols = analytics["symbol"]?.ToString();
                result.Range = analytics["range"]?.ToString();
                result.Interval = analytics["interval"]?.ToString();
                result.OhlcType = analytics["ohlc"]?.ToString() ?? "close";
                result.Metadata.ResponseTime = DateTime.Now;

                // Parse calculations
                var calculationsNode = analytics["calculations"];
                if (calculationsNode != null)
                {
                    // Parse MEAN_VALUE
                    if (calculationsNode["MEAN_VALUE"] != null)
                    {
                        result.MeanValues = ParseSymbolValues(calculationsNode["MEAN_VALUE"]);
                    }

                    // Parse STDDEV (regular)
                    if (calculationsNode["STDDEV"] != null)
                    {
                        result.StdDev = ParseSymbolValues(calculationsNode["STDDEV"]);
                    }

                    // Parse STDDEV (annualized)
                    var annualizedStdDevKey = calculationsNode.Children<JProperty>()
                        .FirstOrDefault(p => p.Name.Contains("STDDEV") && p.Name.Contains("annualized"))?.Name;

                    if (annualizedStdDevKey != null)
                    {
                        result.AnnualizedStdDev = ParseSymbolValues(calculationsNode[annualizedStdDevKey]);
                    }

                    // Parse VARIANCE
                    if (calculationsNode["VARIANCE"] != null)
                    {
                        result.Variance = ParseSymbolValues(calculationsNode["VARIANCE"]);
                    }

                    // Parse VARIANCE (annualized)
                    var annualizedVarianceKey = calculationsNode.Children<JProperty>()
                        .FirstOrDefault(p => p.Name.Contains("VARIANCE") && p.Name.Contains("annualized"))?.Name;

                    if (annualizedVarianceKey != null)
                    {
                        result.AnnualizedVariance = ParseSymbolValues(calculationsNode[annualizedVarianceKey]);
                    }

                    // Parse CORRELATION matrix
                    if (calculationsNode["CORRELATION"] != null)
                    {
                        result.CorrelationMatrix = ParseCorrelationMatrix(calculationsNode["CORRELATION"]);
                    }

                    // Parse COVARIANCE matrix
                    if (calculationsNode["COVARIANCE"] != null)
                    {
                        result.CovarianceMatrix = ParseCorrelationMatrix(calculationsNode["COVARIANCE"]);
                    }
                }

                return result;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error parsing analytics response");
                result.Metadata.Error = $"Parse error: {ex.Message}";
                return result;
            }
        }

        /// <summary>
        /// Parse symbol->value dictionary from JSON
        /// </summary>
        private Dictionary<string, double> ParseSymbolValues(JToken token)
        {
            var result = new Dictionary<string, double>();

            try
            {
                foreach (var property in token.Children<JProperty>())
                {
                    if (double.TryParse(property.Value.ToString(), out double value))
                    {
                        result[property.Name] = value;
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error parsing symbol values");
            }

            return result;
        }

        /// <summary>
        /// Parse correlation or covariance matrix from JSON
        /// </summary>
        private Dictionary<string, Dictionary<string, double>> ParseCorrelationMatrix(JToken token)
        {
            var matrix = new Dictionary<string, Dictionary<string, double>>();

            try
            {
                foreach (var symbolProperty in token.Children<JProperty>())
                {
                    var symbolName = symbolProperty.Name;
                    var correlations = new Dictionary<string, double>();

                    foreach (var correlationProperty in symbolProperty.Value.Children<JProperty>())
                    {
                        if (double.TryParse(correlationProperty.Value.ToString(), out double value))
                        {
                            correlations[correlationProperty.Name] = value;
                        }
                    }

                    matrix[symbolName] = correlations;
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error parsing correlation matrix");
            }

            return matrix;
        }

        #endregion

        #region Earnings Calendar (TFT Known Future Inputs)

        /// <summary>
        /// Gets earnings calendar data from Alpha Vantage EARNINGS endpoint.
        /// Returns historical and future earnings dates for a symbol.
        /// Used for TFT model known future inputs.
        /// </summary>
        /// <param name="symbol">Stock ticker symbol</param>
        /// <returns>List of earnings calendar data including report dates and EPS estimates</returns>
        public async Task<List<EarningsCalendarData>> GetEarningsCalendarAsync(string symbol)
        {
            if (string.IsNullOrWhiteSpace(symbol))
            {
                return new List<EarningsCalendarData>();
            }

            symbol = symbol.ToUpperInvariant();

            try
            {
                _loggingService.Log("Info", $"Fetching earnings calendar for {symbol} from Alpha Vantage");

                // Use EARNINGS function which returns both historical and future earnings
                // API: https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={key}
                string url = $"query?function=EARNINGS&symbol={symbol}&apikey={_apiKey}";

                await _apiSemaphore.WaitAsync();
                try
                {
                    var response = await _client.GetStringAsync(url);
                    LogApiUsage("EARNINGS", symbol);

                    if (string.IsNullOrEmpty(response))
                    {
                        _loggingService.Log("Warning", $"Empty response for earnings calendar for {symbol}");
                        return new List<EarningsCalendarData>();
                    }

                    var json = JObject.Parse(response);
                    var earningsList = new List<EarningsCalendarData>();

                    // Parse quarterly earnings from the response
                    var quarterlyEarnings = json["quarterlyEarnings"] as JArray;
                    if (quarterlyEarnings != null)
                    {
                        foreach (var earning in quarterlyEarnings)
                        {
                            DateTime reportDate;
                            string reportedDateStr = earning["reportedDate"]?.ToString();
                            string fiscalDateEnding = earning["fiscalDateEnding"]?.ToString();
                            
                            // Try to parse report date, fallback to fiscal date ending
                            if (!DateTime.TryParse(reportedDateStr, out reportDate))
                            {
                                if (!DateTime.TryParse(fiscalDateEnding, out reportDate))
                                {
                                    continue; // Skip if we can't determine the date
                                }
                            }

                            decimal? estimatedEps = null;
                            decimal? reportedEps = null;
                            decimal? surprisePercentage = null;

                            if (decimal.TryParse(earning["estimatedEPS"]?.ToString(), out decimal estEps))
                            {
                                estimatedEps = estEps;
                            }
                            if (decimal.TryParse(earning["reportedEPS"]?.ToString(), out decimal repEps))
                            {
                                reportedEps = repEps;
                            }
                            if (decimal.TryParse(earning["surprisePercentage"]?.ToString(), out decimal surprise))
                            {
                                surprisePercentage = surprise;
                            }

                            // Construct fiscal quarter string (e.g., "Q1 2024")
                            string fiscalQuarter = null;
                            if (!string.IsNullOrEmpty(fiscalDateEnding) && DateTime.TryParse(fiscalDateEnding, out DateTime fiscalDate))
                            {
                                int quarter = ((fiscalDate.Month - 1) / 3) + 1;
                                fiscalQuarter = $"Q{quarter} {fiscalDate.Year}";
                            }

                            earningsList.Add(new EarningsCalendarData
                            {
                                Symbol = symbol,
                                FiscalDateEnding = fiscalDateEnding,
                                ReportDate = reportDate,
                                FiscalQuarter = fiscalQuarter,
                                EstimatedEPS = estimatedEps,
                                ReportedEPS = reportedEps,
                                SurprisePercentage = surprisePercentage
                            });
                        }
                    }

                    _loggingService.Log("Info", $"Parsed {earningsList.Count} earnings records for {symbol}");
                    return earningsList;
                }
                finally
                {
                    _apiSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to fetch earnings calendar for {symbol}");
                return new List<EarningsCalendarData>();
            }
        }

        #endregion

        #region Economic Indicators (Market Context for TFT)

        /// <summary>
        /// Gets Treasury Yield data from Alpha Vantage TREASURY_YIELD endpoint.
        /// Used for TFT model market context features.
        /// </summary>
        /// <param name="maturity">Treasury maturity: "3month", "2year", "5year", "7year", "10year", "30year"</param>
        /// <param name="interval">Data interval: "daily", "weekly", "monthly"</param>
        /// <returns>Latest treasury yield percentage</returns>
        public async Task<double> GetTreasuryYieldAsync(string maturity = "10year", string interval = "daily")
        {
            // Validate maturity parameter
            var validMaturities = new[] { "3month", "2year", "5year", "7year", "10year", "30year" };
            if (!validMaturities.Contains(maturity.ToLowerInvariant()))
            {
                _loggingService?.Log("Warning", $"Invalid treasury maturity '{maturity}', defaulting to 10year");
                maturity = "10year";
            }

            // Validate interval parameter
            var validIntervals = new[] { "daily", "weekly", "monthly" };
            if (!validIntervals.Contains(interval.ToLowerInvariant()))
            {
                _loggingService?.Log("Warning", $"Invalid treasury interval '{interval}', defaulting to daily");
                interval = "daily";
            }

            // Check cache first
            var cacheKey = $"TreasuryYield_{maturity}";
            var cached = GetCachedFundamentalData(cacheKey, maturity, 4); // 4 hour cache
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                await WaitForApiLimit();
                
                var endpoint = $"query?function=TREASURY_YIELD&interval={interval}&maturity={maturity}&apikey={_apiKey}";
                await LogApiCall("TREASURY_YIELD", $"{maturity},{interval}");

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    // Parse the data array to get the latest yield
                    if (data["data"] is JArray dataArray && dataArray.Count > 0)
                    {
                        var latestEntry = dataArray[0];
                        if (latestEntry["value"] != null && 
                            double.TryParse(latestEntry["value"].ToString(), 
                                System.Globalization.NumberStyles.Any, 
                                System.Globalization.CultureInfo.InvariantCulture, 
                                out double yieldValue))
                        {
                            // Cache the result
                            CacheFundamentalData(cacheKey, maturity, yieldValue);
                            
                            _loggingService?.Log("Info", $"Retrieved {maturity} Treasury Yield: {yieldValue:F2}%");
                            return yieldValue;
                        }
                    }
                    
                    _loggingService?.Log("Warning", $"No treasury yield data found for {maturity}");
                }

                return 0;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to fetch treasury yield for {maturity}");
                return 0;
            }
        }

        /// <summary>
        /// Gets Federal Funds Rate from Alpha Vantage FEDERAL_FUNDS_RATE endpoint.
        /// Used for TFT model market context features.
        /// </summary>
        /// <param name="interval">Data interval: "daily", "weekly", "monthly"</param>
        /// <returns>Latest federal funds rate percentage</returns>
        public async Task<double> GetFederalFundsRateAsync(string interval = "daily")
        {
            // Validate interval parameter
            var validIntervals = new[] { "daily", "weekly", "monthly" };
            if (!validIntervals.Contains(interval.ToLowerInvariant()))
            {
                _loggingService?.Log("Warning", $"Invalid fed funds interval '{interval}', defaulting to daily");
                interval = "daily";
            }

            // Check cache first
            var cached = GetCachedFundamentalData("FedFundsRate", "rate", 4); // 4 hour cache
            if (cached.HasValue)
            {
                return cached.Value;
            }

            try
            {
                await WaitForApiLimit();
                
                var endpoint = $"query?function=FEDERAL_FUNDS_RATE&interval={interval}&apikey={_apiKey}";
                await LogApiCall("FEDERAL_FUNDS_RATE", interval);

                var response = await _client.GetAsync(endpoint);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JObject.Parse(content);

                    // Parse the data array to get the latest rate
                    if (data["data"] is JArray dataArray && dataArray.Count > 0)
                    {
                        var latestEntry = dataArray[0];
                        if (latestEntry["value"] != null && 
                            double.TryParse(latestEntry["value"].ToString(), 
                                System.Globalization.NumberStyles.Any, 
                                System.Globalization.CultureInfo.InvariantCulture, 
                                out double rateValue))
                        {
                            // Cache the result
                            CacheFundamentalData("FedFundsRate", "rate", rateValue);
                            
                            _loggingService?.Log("Info", $"Retrieved Federal Funds Rate: {rateValue:F2}%");
                            return rateValue;
                        }
                    }
                    
                    _loggingService?.Log("Warning", "No federal funds rate data found");
                }

                return 0;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to fetch federal funds rate");
                return 0;
            }
        }

        #endregion
    }
}
