using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using Quantra.DAL.Services.Interfaces;
using Microsoft.Extensions.Configuration;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to check connectivity with external APIs used by Quantra.
    /// Monitors Alpha Vantage API, News Sentiment API, and other critical endpoints.
    /// </summary>
    public class ApiConnectivityService : IApiConnectivityService, IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly Dictionary<string, DateTime?> _lastSuccessfulConnections;
        private readonly List<ApiEndpoint> _endpoints;
        private readonly string _alphaVantageApiKey;
        private bool _disposed;

        public ApiConnectivityService(IConfiguration configuration = null)
        {
            _httpClient = new HttpClient();
            _httpClient.Timeout = TimeSpan.FromSeconds(10); // Increased timeout for reliability
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "Quantra-Financial-Platform/1.0");

            _lastSuccessfulConnections = new Dictionary<string, DateTime?>();

            // Get API key from configuration or use demo key as fallback
            _alphaVantageApiKey = configuration?["AlphaVantage:ApiKey"] 
                ?? Environment.GetEnvironmentVariable("ALPHAVANTAGE_API_KEY") 
                ?? "demo";

            // Define endpoints to monitor
            // Using GLOBAL_QUOTE as it's lightweight and counts against daily quota
            _endpoints = new List<ApiEndpoint>
            {
                new ApiEndpoint
                {
                    Name = "Alpha Vantage - Market Data",
                    Url = $"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={_alphaVantageApiKey}",
                    Description = "Real-time and historical stock data"
                },
                new ApiEndpoint
                {
                    Name = "Alpha Vantage - News Sentiment",
                    Url = $"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&limit=1&apikey={_alphaVantageApiKey}",
                    Description = "Market news and sentiment analysis"
                },
                new ApiEndpoint
                {
                    Name = "Alpha Vantage - Technical Indicators",
                    Url = $"https://www.alphavantage.co/query?function=SMA&symbol=IBM&interval=daily&time_period=10&series_type=close&apikey={_alphaVantageApiKey}",
                    Description = "Technical analysis indicators"
                }
            };

            // Initialize the last successful connections
            foreach (var endpoint in _endpoints)
            {
                _lastSuccessfulConnections[endpoint.Name] = null;
            }
        }

        /// <summary>
        /// Checks connectivity to all monitored APIs and returns status for the one with issues (if any).
        /// Tests Alpha Vantage endpoints to ensure data can be fetched for trading operations.
        /// </summary>
        /// <returns>Status of API connectivity with details about any failures</returns>
        public async Task<ApiConnectivityStatus> CheckConnectivityAsync()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ApiConnectivityService));

            foreach (var endpoint in _endpoints)
            {
                try
                {
                    var start = DateTime.Now;
                    var response = await _httpClient.GetAsync(endpoint.Url).ConfigureAwait(false);
                    var elapsed = (DateTime.Now - start).TotalMilliseconds;

                    if (response.IsSuccessStatusCode)
                    {
                        // Validate response content to ensure it's not an error message
                        var content = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                        
                        // Alpha Vantage returns 200 OK even for rate limit errors
                        if (content.Contains("\"Error Message\"") || content.Contains("rate limit"))
                        {
                            return new ApiConnectivityStatus
                            {
                                IsConnected = false,
                                ApiName = endpoint.Name,
                                StatusMessage = "API rate limit exceeded or invalid request",
                                LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                                ResponseTimeMs = elapsed,
                                Description = endpoint.Description
                            };
                        }

                        // Check for "Note" field which indicates API key issues
                        if (content.Contains("\"Note\"") && content.Contains("premium"))
                        {
                            return new ApiConnectivityStatus
                            {
                                IsConnected = false,
                                ApiName = endpoint.Name,
                                StatusMessage = "API key requires premium subscription for this feature",
                                LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                                ResponseTimeMs = elapsed,
                                Description = endpoint.Description
                            };
                        }

                        // Update the last successful connection time
                        _lastSuccessfulConnections[endpoint.Name] = DateTime.Now;
                        continue; // This endpoint is fine, check the next one
                    }
                    else
                    {
                        // Return the status with details about the failing endpoint
                        return new ApiConnectivityStatus
                        {
                            IsConnected = false,
                            ApiName = endpoint.Name,
                            StatusMessage = $"HTTP {(int)response.StatusCode}: {response.ReasonPhrase}",
                            LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                            ResponseTimeMs = elapsed,
                            Description = endpoint.Description
                        };
                    }
                }
                catch (TaskCanceledException)
                {
                    // Timeout occurred
                    return new ApiConnectivityStatus
                    {
                        IsConnected = false,
                        ApiName = endpoint.Name,
                        StatusMessage = "Request timed out - API may be unavailable",
                        LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                        ResponseTimeMs = null,
                        Description = endpoint.Description
                    };
                }
                catch (HttpRequestException ex)
                {
                    // Network connectivity issue
                    return new ApiConnectivityStatus
                    {
                        IsConnected = false,
                        ApiName = endpoint.Name,
                        StatusMessage = $"Network error: {ex.Message}",
                        LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                        ResponseTimeMs = null,
                        Description = endpoint.Description
                    };
                }
                catch (Exception ex)
                {
                    // Return the status with details about the failing endpoint
                    return new ApiConnectivityStatus
                    {
                        IsConnected = false,
                        ApiName = endpoint.Name,
                        StatusMessage = $"Unexpected error: {ex.Message}",
                        LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                        ResponseTimeMs = null,
                        Description = endpoint.Description
                    };
                }
            }

            // If we get here, all endpoints are working
            return new ApiConnectivityStatus
            {
                IsConnected = true,
                ApiName = "All APIs",
                StatusMessage = "All monitored APIs are responding normally",
                LastSuccessfulConnection = DateTime.Now,
                ResponseTimeMs = 0,
                Description = "All API endpoints operational"
            };
        }

        /// <summary>
        /// Gets the last successful connection time for a specific API endpoint.
        /// </summary>
        /// <param name="apiName">Name of the API endpoint</param>
        /// <returns>DateTime of last successful connection, or null if never connected</returns>
        public DateTime? GetLastSuccessfulConnection(string apiName)
        {
            return _lastSuccessfulConnections.TryGetValue(apiName, out var lastConnection) 
                ? lastConnection 
                : null;
        }

        /// <summary>
        /// Gets all monitored API endpoints.
        /// </summary>
        /// <returns>List of monitored endpoints</returns>
        public IReadOnlyList<string> GetMonitoredEndpoints()
        {
            var endpointNames = new List<string>();
            foreach (var endpoint in _endpoints)
            {
                endpointNames.Add(endpoint.Name);
            }
            return endpointNames.AsReadOnly();
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            _httpClient?.Dispose();
            _disposed = true;
        }

        private class ApiEndpoint
        {
            public string Name { get; set; }
            public string Url { get; set; }
            public string Description { get; set; }
        }
    }
}