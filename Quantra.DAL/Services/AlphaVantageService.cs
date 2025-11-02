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

        // Standard API rate limits
        private const int StandardApiCallsPerMinute = 75;
        private const int PremiumApiCallsPerMinute = 600; // Premium tier rate limit (can be adjusted based on plan)
        private const string StockCacheKey = "stock_symbols_cache";

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

        public AlphaVantageService(IUserSettingsService userSettingsService)
        {
 _userSettingsService = userSettingsService;
    _client = new HttpClient
         {
      BaseAddress = new Uri("https://www.alphavantage.co/")
 };
            _apiKey = GetApiKey();
       _apiSemaphore = new SemaphoreSlim(1, 1);
          _maxApiCallsPerMinute = StandardApiCallsPerMinute;

 Instance = this;
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
                LoggingService.Log("Info", $"Cached {symbols.Count} symbols to database");
            }
            catch (Exception ex)
            {
                LoggingService.LogErrorWithContext(ex, "Failed to cache symbols");
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
                    double prevTypicalPrice = (periods[i-1].High + periods[i-1].Low + periods[i-1].Close) / 3;

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
            // Get API key from configuration or environment
            return Environment.GetEnvironmentVariable("ALPHA_VANTAGE_API_KEY") 
                ?? "686FIILJC6K24MAS"; // TODO: remove me
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
                LoggingService.Log("Error", $"Failed to get cached historical prices for {symbol}", ex.ToString());
            }

            return new List<HistoricalPrice>();
        }

        /// <summary>
        /// Gets extended historical data with adjusted prices for more accurate backtesting
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="interval">Data interval</param>
        /// <param name="outputSize">Output size (compact/full)</param>
        /// <param name="dataType">Type of data (json/csv)</param>
        /// <returns>List of historical prices with adjusted values</returns>
        public async Task<List<HistoricalPrice>> GetExtendedHistoricalData(string symbol, string interval = "daily", string outputSize = "full", string dataType = "json")
        {
            await WaitForApiLimit();
            
            // Normalize symbol for API call
            string normalizedSymbol = NormalizeSymbol(symbol);
            
            string function;
            string avInterval = "";
            
            if (interval.EndsWith("min"))
            {
                function = "TIME_SERIES_INTRADAY";
                avInterval = interval;
            }
            else if (interval == "daily" || interval == "1d")
            {
                function = "TIME_SERIES_DAILY_ADJUSTED";
            }
            else if (interval == "weekly" || interval == "1wk") 
            {
                function = "TIME_SERIES_WEEKLY_ADJUSTED";
            }
            else if (interval == "monthly" || interval == "1mo")
            {
                function = "TIME_SERIES_MONTHLY_ADJUSTED";
            }
            else
            {
                function = "TIME_SERIES_DAILY_ADJUSTED";
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
                var jsonObject = JObject.Parse(jsonResponse);

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
                }

                // Sort by date ascending
                result = result.OrderBy(h => h.Date).ToList();
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to parse Alpha Vantage response", ex.ToString());
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
          LoggingService.Log("Warning", $"Insufficient historical data for {symbol} ({historicalData.Count} points), returning empty list");
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
       LoggingService.Log("Warning", $"Unknown indicator type: {indicatorType}. Returning empty list");
     return new List<double>();
          }

      LoggingService.Log("Info", $"Calculated {result.Count} real {indicatorType} values for {symbol} from {historicalData.Count} historical data points");
       return result;
    }
            catch (Exception ex)
     {
     LoggingService.LogErrorWithContext(ex, $"Error calculating real historical data for {indicatorType} on {symbol}");
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
    LoggingService.Log("Info", $"Retrieved cached {dataType} for {symbol} (age: {cacheAge.TotalMinutes:F1} minutes)");
    return cachedData.Value;
               }
      else
   {
          // Cache expired, remove it
      _fundamentalDataCache.Remove(cacheKey);
         LoggingService.Log("Info", $"Cache expired for {dataType} on {symbol} (age: {cacheAge.TotalHours:F1} hours)");
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
                LoggingService.Log("Info", $"Cached {dataType} for {symbol}: {value}");
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
     LoggingService.Log("Info", $"Cleared {keysToRemove.Count} cached fundamental data entries for {symbol}");
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
            LoggingService.Log("Info", $"Cleared {expiredKeys.Count} expired fundamental data cache entries");
}
    }
 }
    }
}
