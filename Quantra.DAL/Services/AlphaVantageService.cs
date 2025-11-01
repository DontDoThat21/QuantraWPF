using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Threading;
using Newtonsoft.Json.Linq;
using System.IO;
using Quantra.Models;
using System.Net;
using Quantra.DAL.Services.Interfaces;
using Microsoft.Data.SqlClient;

namespace Quantra.DAL.Services
{
    public class AlphaVantageService : IAlphaVantageService
    {
        private readonly HttpClient _client;
        private readonly string _apiKey;
        private readonly SemaphoreSlim _apiSemaphore;
        private readonly ISettingsService _settingsService;
        
        // Standard API rate limits
        private const int StandardApiCallsPerMinute = 75;
        private const int PremiumApiCallsPerMinute = 600; // Premium tier rate limit (can be adjusted based on plan)
        
        // Current rate limit - will be determined based on API key type one day
        private int _maxApiCallsPerMinute;
        
        public static int ApiCallLimit => Instance?._maxApiCallsPerMinute ?? StandardApiCallsPerMinute;
        
        // Property to check if using premium API
        public bool IsPremiumKey => IsPremiumApiKey(_apiKey);
        
        // Singleton pattern for easy access
        private static AlphaVantageService Instance { get; set; }

        public AlphaVantageService(ISettingsService settingsService)
        {
            _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
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
            return DatabaseMonolith.GetAlphaVantageApiUsageCount(DateTime.UtcNow);
        }

        public void LogApiUsage()
        {
            LogApiUsage(null, null);
        }

        public void LogApiUsage(string endpoint, string parameters)
        {
            DatabaseMonolith.LogAlphaVantageApiUsage(endpoint, parameters);
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
                return JsonSerializer.Deserialize<T>(content);
            }
            catch (JsonException ex)
            {
                // Optionally log the error and the content for debugging
                DatabaseMonolith.Log("Error", $"Failed to deserialize AlphaVantage response for {functionName}", $"Content: {content}\nException: {ex}");
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
            // First check cache
            var cachedSymbols = DatabaseMonolith.GetCachedSymbols();
            if (cachedSymbols != null && cachedSymbols.Any())
            {
                // Ensure VIX is included in cached symbols
                if (!cachedSymbols.Contains("VIX"))
                {
                    cachedSymbols.Add("VIX");
                }
                return cachedSymbols;
            }

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
                DatabaseMonolith.CacheSymbols(symbols);
                return symbols;
            }

            // Return VIX as a fallback if API fails
            return new List<string> { "VIX" };
        }

        public async Task<double> GetRSI(string symbol, string interval = "1min")
        {
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
                return double.IsNaN(latestRsi) ? 50 : latestRsi;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to calculate RSI for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate ADX for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate ATR for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate momentum for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate Stochastic for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate CCI for {symbol}", ex.ToString());
                return 0; // Neutral default
            }
        }

        public async Task<List<string>> GetMostVolatileStocksAsync()
        {
            // Check cache first
            var cachedVolatileStocks = DatabaseMonolith.GetCachedVolatileStocks();
            if (cachedVolatileStocks != null && cachedVolatileStocks.Any())
                return cachedVolatileStocks;

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

                // Cache the volatile stocks
                DatabaseMonolith.CacheVolatileStocks(volatileStocks);
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
                DatabaseMonolith.Log("Error", "Failed to get indicators", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate Ultimate Oscillator for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate OBV for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate MFI for {symbol}", ex.ToString());
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
                DatabaseMonolith.LogAlphaVantageApiUsage(endpoint, parameters);
            }
            finally
            {
                _apiSemaphore.Release();
            }
        }

        public string GetApiKey()
        {
            // Try to get API key from settings service first
            try
            {
                var defaultProfile = _settingsService?.GetDefaultSettingsProfile();
                if (defaultProfile != null && !string.IsNullOrWhiteSpace(defaultProfile.AlphaVantageApiKey))
                {
                    return defaultProfile.AlphaVantageApiKey;
                }
            }
            catch (Exception ex)
            {
                // Log the error but continue to environment variable fallback
                DatabaseMonolith.Log("Warning", "Failed to retrieve API key from settings service", ex.Message);
            }
            
            // Get API key from environment variable as fallback
            var apiKey = Environment.GetEnvironmentVariable("ALPHA_VANTAGE_API_KEY");
            if (!string.IsNullOrWhiteSpace(apiKey))
            {
                return apiKey;
            }
            
            // Log warning if no API key is configured
            DatabaseMonolith.Log("Warning", "No Alpha Vantage API key configured. API calls will fail.", 
                "Please configure AlphaVantageApiKey in settings or set ALPHA_VANTAGE_API_KEY environment variable");
            
            // Return empty string instead of hardcoded key
            return string.Empty;
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
                    DatabaseMonolith.Log("Info", $"Using cached historical data for {symbol} - {cachedData.Count} data points");
                    return cachedData;
                }
                
                // If insufficient cached data, fall back to API
                DatabaseMonolith.Log("Info", $"Insufficient cached data for {symbol} ({cachedData.Count} points), fetching from API");
                return await GetExtendedHistoricalData(symbol, interval, "compact");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error in GetCachedHistoricalDataFirst for {symbol}", ex.ToString());
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
                var (stockData, timestamp) = await DatabaseMonolith.GetStockDataWithTimestamp(symbol, timeRange);
                
                if (stockData?.Dates != null && stockData.Dates.Count > 0)
                {
                    var historicalPrices = new List<HistoricalPrice>();
                    
                    for (int i = 0; i < stockData.Dates.Count && i < stockData.Prices.Count; i++)
                    {
                        // Use Close prices for High/Low/Open if not available
                        var price = stockData.Prices[i];
                        var volume = stockData.Volumes != null && i < stockData.Volumes.Count ? (long)stockData.Volumes[i] : 1000;
                        
                        historicalPrices.Add(new HistoricalPrice
                        {
                            Date = stockData.Dates[i],
                            Open = price,
                            High = price * 1.01, // Approximate high as 1% above close
                            Low = price * 0.99,  // Approximate low as 1% below close
                            Close = price,
                            Volume = volume,
                            AdjClose = price // what is adjusted close used for?
                        });
                    }
                    
                    // Sort by date to ensure chronological order
                    return historicalPrices.OrderBy(h => h.Date).ToList();
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to get cached historical prices for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", "Failed to parse forex response", ex.ToString());
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
                DatabaseMonolith.Log("Error", "Failed to parse crypto response", ex.ToString());
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
                DatabaseMonolith.Log("Error", "Failed to parse Alpha Vantage response", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate VWAP for {symbol}", ex.ToString());
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
                DatabaseMonolith.Log("Error", $"Failed to calculate MACD for {symbol}", ex.ToString());
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
            var cachedPeRatio = DatabaseMonolith.GetCachedFundamentalData(symbol, "PE_RATIO", 24);
            if (cachedPeRatio.HasValue)
            {
                return cachedPeRatio.Value;
            }

            // If not in cache, fetch from API
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
                    DatabaseMonolith.CacheFundamentalData(symbol, "PE_RATIO", peRatio);
                    return peRatio;
                }
            }

            // Cache null result to avoid repeated API calls for invalid symbols
            DatabaseMonolith.CacheFundamentalData(symbol, "PE_RATIO", null);
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
            
            // Calculate smoothed versions using EMA
            var smoothedTRs = CalculateEMAInternal(trueRanges, period);
            var smoothedPlusDMs = CalculateEMAInternal(plusDMs, period);
            var smoothedMinusDMs = CalculateEMAInternal(minusDMs, period);
            
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

        #region Order History Management

        /// <summary>
        /// Adds an order to the order history database
        /// </summary>
        /// <param name="order">The order to add to history</param>
        /// <remarks>
        /// This method currently uses a separate SQL Server connection.
        /// Once DatabaseMonolith is fully migrated to SQL Server, this should use
        /// DatabaseMonolith's connection infrastructure for consistency.
        /// </remarks>
        public void AddOrderToHistory(OrderModel order)
        {
            if (order == null)
            {
                Log("Error", "Cannot add null order to history");
                return;
            }

            try
            {
                // Get connection string from DatabaseMonolith or configuration
                string connectionString = GetSqlServerConnectionString();
                
                using (var connection = new SqlConnection(connectionString))
                {
                    connection.Open();

                    using (var command = new SqlCommand())
                    {
                        command.Connection = connection;
                        command.CommandText = @"
                            INSERT INTO OrderHistory (
                                Symbol, OrderType, Quantity, Price, StopLoss, TakeProfit, 
                                IsPaperTrade, Status, PredictionSource, Timestamp
                            )
                            VALUES (
                                @Symbol, @OrderType, @Quantity, @Price, @StopLoss, @TakeProfit, 
                                @IsPaperTrade, @Status, @PredictionSource, @Timestamp
                            );";

                        command.Parameters.AddWithValue("@Symbol", order.Symbol);
                        command.Parameters.AddWithValue("@OrderType", order.OrderType);
                        command.Parameters.AddWithValue("@Quantity", order.Quantity);
                        command.Parameters.AddWithValue("@Price", order.Price);
                        command.Parameters.AddWithValue("@StopLoss", order.StopLoss);
                        command.Parameters.AddWithValue("@TakeProfit", order.TakeProfit);
                        
                        // Use SqlDbType.Bit for boolean values in SQL Server
                        var isPaperTradeParam = command.Parameters.Add("@IsPaperTrade", System.Data.SqlDbType.Bit);
                        isPaperTradeParam.Value = order.IsPaperTrade;
                        
                        command.Parameters.AddWithValue("@Status", order.Status);
                        
                        // Use DBNull.Value for null strings to properly represent NULL in database
                        command.Parameters.AddWithValue("@PredictionSource", 
                            (object)order.PredictionSource ?? DBNull.Value);
                        
                        command.Parameters.AddWithValue("@Timestamp", order.Timestamp);

                        command.ExecuteNonQuery();
                    }
                }

                Log("Info", $"Order added to history: {order.Symbol} {order.OrderType} {order.Quantity} @ {order.Price:C2}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.LogErrorWithContext(ex, $"Failed to add order to history: {order.Symbol}");
            }
        }

        
        /// <summary>
        /// Log a message (convenience method that delegates to DatabaseMonolith)
        /// </summary>
        private void Log(string level, string message, string details = null)
        {
            DatabaseMonolith.Log(level, message, details);
        }

        /// <summary>
        /// Gets the SQL Server connection string from DatabaseMonolith or configuration
        /// </summary>
        /// <returns>SQL Server connection string</returns>
        private string GetSqlServerConnectionString()
        {
            // TODO: This should be centralized in DatabaseMonolith or a configuration service
            // For now, this provides a basic SQL Server connection string from environment variables
            // Once DatabaseMonolith is fully migrated to SQL Server, use its connection infrastructure
            
            string server = Environment.GetEnvironmentVariable("SQL_SERVER");
            string database = Environment.GetEnvironmentVariable("SQL_DATABASE");
            string userId = Environment.GetEnvironmentVariable("SQL_USER");
            string password = Environment.GetEnvironmentVariable("SQL_PASSWORD");
            
            // Validate required configuration
            if (string.IsNullOrWhiteSpace(server))
            {
                DatabaseMonolith.Log("Warning", "SQL_SERVER environment variable not set, using default 'localhost'");
                server = "localhost";
            }
            
            if (string.IsNullOrWhiteSpace(database))
            {
                DatabaseMonolith.Log("Warning", "SQL_DATABASE environment variable not set, using default 'Quantra'");
                database = "Quantra";
            }
            
            // Use integrated security if no credentials provided
            if (string.IsNullOrWhiteSpace(userId) || string.IsNullOrWhiteSpace(password))
            {
                return $"Server={server};Database={database};Integrated Security=true;TrustServerCertificate=true;";
            }
            else
            {
                return $"Server={server};Database={database};User Id={userId};Password={password};TrustServerCertificate=true;";
            }
        }

        #endregion
    }
}
