using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using Quantra.Services.Interfaces;

namespace Quantra.Services
{
    /// <summary>
    /// Service to check connectivity with external APIs used by Quantra
    /// </summary>
    public class ApiConnectivityService : IApiConnectivityService
    {
        private readonly HttpClient _httpClient;
        private readonly Dictionary<string, DateTime?> _lastSuccessfulConnections;
        private readonly List<ApiEndpoint> _endpoints;

        public ApiConnectivityService()
        {
            _httpClient = new HttpClient();
            _httpClient.Timeout = TimeSpan.FromSeconds(5); // Short timeout to detect issues quickly
            
            _lastSuccessfulConnections = new Dictionary<string, DateTime?>();
            
            // Define endpoints to monitor
            _endpoints = new List<ApiEndpoint>
            {
                new ApiEndpoint 
                { 
                    Name = "Alpha Vantage API", 
                    Url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo"
                },
                new ApiEndpoint
                {
                    Name = "Market News API",
                    Url = "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo"
                }
                // Add other APIs as needed
            };
            
            // Initialize the last successful connections
            foreach (var endpoint in _endpoints)
            {
                _lastSuccessfulConnections[endpoint.Name] = null;
            }
        }

        /// <summary>
        /// Checks connectivity to all monitored APIs and returns status for the one with issues (if any)
        /// </summary>
        public async Task<ApiConnectivityStatus> CheckConnectivityAsync()
        {
            foreach (var endpoint in _endpoints)
            {
                try
                {
                    var start = DateTime.Now;
                    var response = await _httpClient.GetAsync(endpoint.Url);
                    var elapsed = (DateTime.Now - start).TotalMilliseconds;
                    
                    if (response.IsSuccessStatusCode)
                    {
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
                            StatusMessage = $"HTTP Status: {response.StatusCode}",
                            LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                            ResponseTimeMs = elapsed
                        };
                    }
                }
                catch (Exception ex)
                {
                    // Return the status with details about the failing endpoint
                    return new ApiConnectivityStatus
                    {
                        IsConnected = false,
                        ApiName = endpoint.Name,
                        StatusMessage = ex.Message,
                        LastSuccessfulConnection = _lastSuccessfulConnections[endpoint.Name],
                        ResponseTimeMs = null
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
                ResponseTimeMs = 0
            };
        }
        
        private class ApiEndpoint
        {
            public string Name { get; set; }
            public string Url { get; set; }
        }
    }
}