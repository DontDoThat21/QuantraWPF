using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Quantra.Models;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for fetching and managing options data from Alpha Vantage
    /// Supports real-time and historical options chains
    /// </summary>
    public class OptionsDataService
    {
        private readonly HttpClient _client;
        private readonly string _apiKey;
        private readonly SemaphoreSlim _apiSemaphore;
        private readonly LoggingService _loggingService;
        private readonly IAlphaVantageService _alphaVantageService;

        // Cache for options data with 5-15 minute expiration
        private readonly Dictionary<string, (List<OptionData> Data, DateTime Timestamp)> _optionsCache 
            = new Dictionary<string, (List<OptionData>, DateTime)>();
        private readonly object _cacheLock = new object();
        private const int CacheExpirationMinutes = 10;

        public OptionsDataService(
            IAlphaVantageService alphaVantageService,
            LoggingService loggingService)
        {
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
            
            _client = new HttpClient
            {
                BaseAddress = new Uri("https://www.alphavantage.co/")
            };
            _apiKey = AlphaVantageService.GetApiKey();
            _apiSemaphore = new SemaphoreSlim(1, 1);
        }

        /// <summary>
        /// Gets the full options chain for a symbol
        /// </summary>
        /// <param name="symbol">Underlying symbol (e.g., AAPL)</param>
        /// <param name="expiration">Optional: Filter by specific expiration date</param>
        /// <param name="includeGreeks">Whether to include Greeks calculations (default: true)</param>
        /// <returns>List of option contracts</returns>
        public async Task<List<OptionData>> GetOptionsChainAsync(
            string symbol, 
            DateTime? expiration = null, 
            bool includeGreeks = true)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));

            // Check cache first
            var cacheKey = $"{symbol}_{expiration?.ToString("yyyyMMdd") ?? "all"}";
            lock (_cacheLock)
            {
                if (_optionsCache.TryGetValue(cacheKey, out var cached))
                {
                    var age = DateTime.Now - cached.Timestamp;
                    if (age.TotalMinutes <= CacheExpirationMinutes)
                    {
                        _loggingService.Log("Info", $"Using cached options chain for {symbol} (age: {age.TotalMinutes:F1} min)");
                        return cached.Data;
                    }
                    else
                    {
                        _optionsCache.Remove(cacheKey);
                    }
                }
            }

            try
            {
                _loggingService.Log("Info", $"Fetching options chain for {symbol}");

                // Alpha Vantage REALTIME_OPTIONS endpoint
                await _apiSemaphore.WaitAsync();
                try
                {
                    var endpoint = $"query?function=REALTIME_OPTIONS&symbol={symbol}&apikey={_apiKey}";
                    
                    if (expiration.HasValue)
                    {
                        endpoint += $"&date={expiration.Value:yyyy-MM-dd}";
                    }

                    _alphaVantageService.LogApiUsage("REALTIME_OPTIONS", symbol);

                    var response = await _client.GetAsync(endpoint);
                    if (!response.IsSuccessStatusCode)
                    {
                        _loggingService.Log("Error", $"Options API returned status {response.StatusCode}");
                        return new List<OptionData>();
                    }

                    var content = await response.Content.ReadAsStringAsync();
                    var options = ParseOptionsChainResponse(content, symbol, includeGreeks);

                    // Cache the result
                    lock (_cacheLock)
                    {
                        _optionsCache[cacheKey] = (options, DateTime.Now);
                    }

                    _loggingService.Log("Info", $"Retrieved {options.Count} option contracts for {symbol}");
                    return options;
                }
                finally
                {
                    _apiSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error fetching options chain for {symbol}");
                return new List<OptionData>();
            }
        }

        /// <summary>
        /// Gets a specific option contract by contract ID
        /// </summary>
        /// <param name="contractId">Option contract identifier</param>
        /// <param name="includeGreeks">Whether to include Greeks calculations</param>
        /// <returns>Option contract data</returns>
        public async Task<OptionData> GetOptionContractAsync(string contractId, bool includeGreeks = true)
        {
            if (string.IsNullOrWhiteSpace(contractId))
                throw new ArgumentException("Contract ID cannot be null or empty", nameof(contractId));

            try
            {
                // Parse contract ID to extract symbol
                // Format: SYMBOL{YYMMDD}{C/P}{Strike} (e.g., AAPL240315C00150000)
                var symbol = ExtractSymbolFromContractId(contractId);
                
                var chain = await GetOptionsChainAsync(symbol, null, includeGreeks);
                return chain.FirstOrDefault(o => o.OptionSymbol == contractId);
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error fetching option contract {contractId}");
                return null;
            }
        }

        /// <summary>
        /// Gets historical options data for a specific date
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="date">Historical date</param>
        /// <returns>Historical options chain</returns>
        public async Task<List<OptionData>> GetHistoricalOptionsAsync(string symbol, DateTime date)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));

            try
            {
                _loggingService.Log("Info", $"Fetching historical options for {symbol} on {date:yyyy-MM-dd}");

                await _apiSemaphore.WaitAsync();
                try
                {
                    // Alpha Vantage HISTORICAL_OPTIONS endpoint
                    var endpoint = $"query?function=HISTORICAL_OPTIONS&symbol={symbol}&date={date:yyyy-MM-dd}&apikey={_apiKey}";
                    
                    _alphaVantageService.LogApiUsage("HISTORICAL_OPTIONS", $"{symbol}_{date:yyyy-MM-dd}");

                    var response = await _client.GetAsync(endpoint);
                    if (!response.IsSuccessStatusCode)
                    {
                        _loggingService.Log("Error", $"Historical options API returned status {response.StatusCode}");
                        return new List<OptionData>();
                    }

                    var content = await response.Content.ReadAsStringAsync();
                    var options = ParseOptionsChainResponse(content, symbol, true);

                    _loggingService.Log("Info", $"Retrieved {options.Count} historical option contracts for {symbol}");
                    return options;
                }
                finally
                {
                    _apiSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Error fetching historical options for {symbol} on {date:yyyy-MM-dd}");
                return new List<OptionData>();
            }
        }

        /// <summary>
        /// Gets available expiration dates for a symbol
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <returns>List of expiration dates</returns>
        public async Task<List<DateTime>> GetExpirationDatesAsync(string symbol)
        {
            var chain = await GetOptionsChainAsync(symbol, null, false);
            return chain
                .Select(o => o.ExpirationDate)
                .Distinct()
                .OrderBy(d => d)
                .ToList();
        }

        /// <summary>
        /// Gets available strike prices for a specific expiration
        /// </summary>
        /// <param name="symbol">Underlying symbol</param>
        /// <param name="expiration">Expiration date</param>
        /// <param name="optionType">CALL or PUT</param>
        /// <returns>List of strike prices</returns>
        public async Task<List<double>> GetStrikePricesAsync(
            string symbol, 
            DateTime expiration, 
            string optionType)
        {
            var chain = await GetOptionsChainAsync(symbol, expiration, false);
            return chain
                .Where(o => o.OptionType.Equals(optionType, StringComparison.OrdinalIgnoreCase))
                .Select(o => o.StrikePrice)
                .Distinct()
                .OrderBy(s => s)
                .ToList();
        }

        /// <summary>
        /// Clears expired cache entries
        /// </summary>
        public void ClearExpiredCache()
        {
            lock (_cacheLock)
            {
                var expiredKeys = _optionsCache
                    .Where(kvp => (DateTime.Now - kvp.Value.Timestamp).TotalMinutes > CacheExpirationMinutes)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in expiredKeys)
                {
                    _optionsCache.Remove(key);
                }

                if (expiredKeys.Count > 0)
                {
                    _loggingService.Log("Info", $"Cleared {expiredKeys.Count} expired options cache entries");
                }
            }
        }

        #region Private Helper Methods

        /// <summary>
        /// Parses Alpha Vantage options chain JSON response
        /// </summary>
        private List<OptionData> ParseOptionsChainResponse(string jsonResponse, string symbol, bool includeGreeks)
        {
            var options = new List<OptionData>();

            try
            {
                var json = JObject.Parse(jsonResponse);

                // Check for errors
                if (json["Error Message"] != null || json["Note"] != null || json["Information"] != null)
                {
                    var error = json["Error Message"]?.ToString() ?? 
                               json["Note"]?.ToString() ?? 
                               json["Information"]?.ToString();
                    _loggingService.Log("Warning", $"Options API message: {error}");
                    return options;
                }

                // Parse data array
                if (json["data"] is JArray dataArray)
                {
                    foreach (var item in dataArray)
                    {
                        var option = new OptionData
                        {
                            UnderlyingSymbol = symbol,
                            StrikePrice = TryParseDouble(item["strike"]),
                            OptionType = item["type"]?.ToString()?.ToUpper() ?? "CALL",
                            Bid = TryParseDouble(item["bid"]),
                            Ask = TryParseDouble(item["ask"]),
                            LastPrice = TryParseDouble(item["last"]),
                            Volume = TryParseLong(item["volume"]),
                            OpenInterest = TryParseLong(item["open_interest"]),
                            ImpliedVolatility = TryParseDouble(item["implied_volatility"]),
                            FetchTimestamp = DateTime.Now
                        };

                        // Parse expiration date
                        if (DateTime.TryParse(item["expiration"]?.ToString(), out DateTime expDate))
                        {
                            option.ExpirationDate = expDate;
                        }

                        // Parse Greeks if available from API
                        if (includeGreeks)
                        {
                            option.Delta = TryParseDouble(item["delta"]);
                            option.Gamma = TryParseDouble(item["gamma"]);
                            option.Theta = TryParseDouble(item["theta"]);
                            option.Vega = TryParseDouble(item["vega"]);
                            option.Rho = TryParseDouble(item["rho"]);
                        }

                        options.Add(option);
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error parsing options chain response");
            }

            return options;
        }

        /// <summary>
        /// Extracts underlying symbol from option contract ID
        /// </summary>
        private string ExtractSymbolFromContractId(string contractId)
        {
            // Parse format: SYMBOL{YYMMDD}{C/P}{Strike}
            // Find the first digit which marks the start of the date
            for (int i = 0; i < contractId.Length; i++)
            {
                if (char.IsDigit(contractId[i]))
                {
                    return contractId.Substring(0, i);
                }
            }

            return contractId;
        }

        private static double TryParseDouble(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return 0.0;

            if (double.TryParse(token.ToString(), 
                System.Globalization.NumberStyles.Any, 
                System.Globalization.CultureInfo.InvariantCulture, 
                out double result))
            {
                return result;
            }

            return 0.0;
        }

        private static long TryParseLong(JToken token)
        {
            if (token == null || token.Type == JTokenType.Null)
                return 0;

            if (long.TryParse(token.ToString(), 
                System.Globalization.NumberStyles.Any, 
                System.Globalization.CultureInfo.InvariantCulture, 
                out long result))
            {
                return result;
            }

            return 0;
        }

        #endregion
    }
}
