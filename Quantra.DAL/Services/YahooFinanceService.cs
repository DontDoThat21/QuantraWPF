using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Net.Http.Headers;
using Quantra.DAL.Services;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for fetching financial data from Yahoo Finance
    /// </summary>
    public class YahooFinanceService
    {
        private readonly HttpClient _client;
        private readonly LoggingService _loggingService;

        public YahooFinanceService(LoggingService loggingService)
        {
            _loggingService = loggingService ?? throw new ArgumentNullException(nameof(loggingService));
            
            // Use HttpClientHandler to automatically decompress gzip responses
            var handler = new HttpClientHandler
            {
                AutomaticDecompression = System.Net.DecompressionMethods.GZip | System.Net.DecompressionMethods.Deflate
            };
            
            _client = new HttpClient(handler)
            {
                Timeout = TimeSpan.FromSeconds(10)
            };
            
            // Set proper headers to look like a real browser
            _client.DefaultRequestHeaders.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36");
            _client.DefaultRequestHeaders.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8");
            _client.DefaultRequestHeaders.Add("Accept-Language", "en-US,en;q=0.9");
            _client.DefaultRequestHeaders.Add("Accept-Encoding", "gzip, deflate");
            _client.DefaultRequestHeaders.CacheControl = new CacheControlHeaderValue { NoCache = true };
        }

        /// <summary>
        /// Gets the current VIX value from Yahoo Finance
        /// </summary>
        /// <returns>VIX value as double, or null if unable to fetch</returns>
        public async Task<double?> GetVixValueAsync()
        {
            try
            {
                var url = "https://finance.yahoo.com/quote/%5EVIX/";
                _loggingService.Log("Info", $"Fetching VIX data from Yahoo Finance: {url}");

                var response = await _client.GetAsync(url);
                if (!response.IsSuccessStatusCode)
                {
                    _loggingService.Log("Warning", $"Yahoo Finance request failed with status: {response.StatusCode}");
                    return null;
                }

                var html = await response.Content.ReadAsStringAsync();
                
                // Log first 500 characters for debugging
                _loggingService.Log("Debug", $"HTML response preview: {html.Substring(0, Math.Min(500, html.Length))}");

                // Try multiple patterns to find the VIX value
                // Pattern 1: Look for span with data-testid="qsp-price"
                var pattern1 = @"<span[^>]*data-testid=""qsp-price""[^>]*>([0-9]+\.?[0-9]*)<";
                var match1 = Regex.Match(html, pattern1);
                
                if (match1.Success && match1.Groups.Count > 1)
                {
                    var valueStr = match1.Groups[1].Value;
                    if (double.TryParse(valueStr, System.Globalization.NumberStyles.Any, 
                        System.Globalization.CultureInfo.InvariantCulture, out double vixValue))
                    {
                        _loggingService.Log("Info", $"Successfully parsed VIX value (pattern 1 - qsp-price): {vixValue}");
                        return vixValue;
                    }
                }

                // Pattern 2: Look for span with class containing "yf-ipw1h0" and "base"
                var pattern2 = @"<span[^>]*class=""[^""]*yf-ipw1h0[^""]*base[^""]*""[^>]*>([0-9]+\.?[0-9]*)<";
                var match2 = Regex.Match(html, pattern2);
                
                if (match2.Success && match2.Groups.Count > 1)
                {
                    var valueStr = match2.Groups[1].Value;
                    if (double.TryParse(valueStr, System.Globalization.NumberStyles.Any, 
                        System.Globalization.CultureInfo.InvariantCulture, out double vixValue))
                    {
                        _loggingService.Log("Info", $"Successfully parsed VIX value (pattern 2 - yf-ipw1h0): {vixValue}");
                        return vixValue;
                    }
                }

                // Pattern 3: Look for price in fin-streamer with data-symbol
                var pattern3 = @"<fin-streamer[^>]*data-symbol=""[\^]?VIX""[^>]*data-value=""([0-9]+\.?[0-9]*)""";
                var match3 = Regex.Match(html, pattern3);
                
                if (match3.Success && match3.Groups.Count > 1)
                {
                    var valueStr = match3.Groups[1].Value;
                    if (double.TryParse(valueStr, System.Globalization.NumberStyles.Any, 
                        System.Globalization.CultureInfo.InvariantCulture, out double vixValue))
                    {
                        _loggingService.Log("Info", $"Successfully parsed VIX value (pattern 3 - fin-streamer): {vixValue}");
                        return vixValue;
                    }
                }

                // Pattern 4: data-testid="qsp-price" with flexible content
                var pattern4 = @"data-testid=""qsp-price""[^>]*>([0-9]+\.?[0-9]*)<";
                var match4 = Regex.Match(html, pattern4);
                
                if (match4.Success && match4.Groups.Count > 1)
                {
                    var valueStr = match4.Groups[1].Value;
                    if (double.TryParse(valueStr, System.Globalization.NumberStyles.Any, 
                        System.Globalization.CultureInfo.InvariantCulture, out double vixValue))
                    {
                        _loggingService.Log("Info", $"Successfully parsed VIX value (pattern 4 - qsp-price): {vixValue}");
                        return vixValue;
                    }
                }

                _loggingService.Log("Warning", "Could not parse VIX value from Yahoo Finance response - no pattern matched");
                return null;
            }
            catch (TaskCanceledException)
            {
                _loggingService.Log("Warning", "Yahoo Finance request timed out");
                return null;
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error fetching VIX from Yahoo Finance");
                return null;
            }
        }

        /// <summary>
        /// Gets the current VIX value with caching support
        /// </summary>
        /// <param name="cacheMaxAgeMinutes">Maximum age of cached value in minutes</param>
        /// <returns>VIX value as double, or null if unable to fetch</returns>
        public async Task<(double? Value, DateTime? LastUpdate)> GetVixValueWithCacheAsync(int cacheMaxAgeMinutes = 5)
        {
            try
            {
                var vixValue = await GetVixValueAsync();
                var timestamp = vixValue.HasValue ? DateTime.Now : (DateTime?)null;
                return (vixValue, timestamp);
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Error fetching VIX with cache");
                return (null, null);
            }
        }

        public void Dispose()
        {
            _client?.Dispose();
        }
    }
}
