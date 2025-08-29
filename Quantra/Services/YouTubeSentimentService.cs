using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Quantra.Configuration;
using Quantra.Configuration.Models;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Services.Interfaces;

namespace Quantra.Services
{
    /// <summary>
    /// Service for analyzing sentiment from YouTube videos, particularly Bloomberg 24/7 live streams.
    /// Integrates audio extraction, transcription, and financial sentiment analysis.
    /// </summary>
    public class YouTubeSentimentService : ISocialMediaSentimentService
    {
        private readonly ILogger<YouTubeSentimentService> _logger;
        private readonly ApiConfig _apiConfig;
        private readonly SentimentAnalysisConfig _sentimentConfig;
        private readonly string _pythonScriptPath;
        private readonly Dictionary<string, SentimentCacheItem> _sentimentCache = new Dictionary<string, SentimentCacheItem>();

        /// <summary>
        /// Constructor for YouTubeSentimentService
        /// </summary>
        public YouTubeSentimentService(
            ILogger<YouTubeSentimentService> logger,
            IConfigurationManager configManager)
        {
            _logger = logger;
            _apiConfig = configManager.GetSection<ApiConfig>("ApiConfig");
            _sentimentConfig = configManager.GetSection<SentimentAnalysisConfig>("SentimentAnalysisConfig");
            _pythonScriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "youtube_sentiment_analysis.py");
        }

        /// <summary>
        /// High-level method: fetches YouTube content and returns average sentiment for a symbol.
        /// </summary>
        public async Task<double> GetSymbolSentimentAsync(string symbol)
        {
            try
            {
                // Check cache first
                var cacheKey = $"youtube_sentiment_{symbol}";
                if (_sentimentCache.TryGetValue(cacheKey, out var cachedItem) && 
                    DateTime.Now - cachedItem.Timestamp < TimeSpan.FromMinutes(30)) // Default 30 minutes cache
                {
                    _logger.LogInformation($"Returning cached sentiment for {symbol}: {cachedItem.Sentiment}");
                    return cachedItem.Sentiment;
                }

                return new double();

                // why is this commented lmao
                // if this idea useless? bloomberg free is so npc anyways..
                // Get YouTube URLs for the symbol (Bloomberg streams, company channels, etc.)
                //var urls = await GetRecentContentAsync(symbol, 5);
                //if (urls.Count == 0)
                //{
                //    _logger.LogWarning($"No YouTube content found for symbol: {symbol}");
                //    return 0.0;
                //}

                // Analyze sentiment from the URLs
                //var sentiment = await AnalyzeSentimentAsync(urls);
                
                // Cache the result
                //_sentimentCache[cacheKey] = new SentimentCacheItem
                //{
                //    Sentiment = sentiment,
                //    Timestamp = DateTime.Now
                //};
                //
                //_logger.LogInformation($"YouTube sentiment analysis completed for {symbol}: {sentiment}");
                //return sentiment;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting YouTube sentiment for symbol: {symbol}");
                return 0.0;
            }
        }

        /// <summary>
        /// Analyzes sentiment from a list of YouTube URLs.
        /// </summary>
        public async Task<double> AnalyzeSentimentAsync(List<string> urls)
        {
            if (urls == null || urls.Count == 0)
            {
                return 0.0;
            }

            return await ResilienceHelper.RetryAsync(async () =>
            {
                // Prepare data for Python script
                var urlData = urls.ConvertAll(url => new
                {
                    url = url,
                    context = "Bloomberg financial news and market analysis"
                });

                var jsonInput = JsonSerializer.Serialize(urlData);
                
                // Call Python script
                var sentiment = await CallPythonSentimentAnalysis(jsonInput);
                
                _logger.LogInformation($"YouTube sentiment analysis completed for {urls.Count} URLs: {sentiment}");
                return sentiment;
            });
        }

        /// <summary>
        /// Gets detailed sentiment data for a symbol by source.
        /// </summary>
        public async Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol)
        {
            var result = new Dictionary<string, double>();
            
            try
            {
                // Analyze different types of YouTube content
                var bloombergUrls = await GetBloombergContentAsync(symbol);
                var companyUrls = await GetCompanyContentAsync(symbol);
                var newsUrls = await GetNewsContentAsync(symbol);

                if (bloombergUrls.Count > 0)
                {
                    result["Bloomberg_YouTube"] = await AnalyzeSentimentAsync(bloombergUrls);
                }

                if (companyUrls.Count > 0)
                {
                    result["Company_YouTube"] = await AnalyzeSentimentAsync(companyUrls);
                }

                if (newsUrls.Count > 0)
                {
                    result["News_YouTube"] = await AnalyzeSentimentAsync(newsUrls);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting detailed YouTube sentiment for symbol: {symbol}");
            }

            return result;
        }

        /// <summary>
        /// Gets recent YouTube content URLs for a symbol
        /// </summary>
        public async Task<List<string>> FetchRecentContentAsync(string symbol, int count = 10)
        {
            var urls = new List<string>();
            
            try
            {
                // Combine different sources
                urls.AddRange(await GetBloombergContentAsync(symbol));
                urls.AddRange(await GetCompanyContentAsync(symbol));
                urls.AddRange(await GetNewsContentAsync(symbol));

                // Limit to requested count
                if (urls.Count > count)
                {
                    urls = urls.GetRange(0, count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error fetching YouTube content for symbol: {symbol}");
            }

            return urls;
        }

        /// <summary>
        /// Analyzes sentiment from a single YouTube URL
        /// </summary>
        public async Task<double> AnalyzeYouTubeUrlSentimentAsync(string url, string context = "Bloomberg financial news")
        {
            try
            {
                var data = new { url = url, context = context };
                var jsonInput = JsonSerializer.Serialize(data);
                
                return await CallPythonSentimentAnalysis(jsonInput);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing sentiment for YouTube URL: {url}");
                return 0.0;
            }
        }

        /// <summary>
        /// Calls the Python script for YouTube sentiment analysis
        /// </summary>
        private async Task<double> CallPythonSentimentAnalysis(string jsonInput)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = $"\"{_pythonScriptPath}\"",
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                // Set OpenAI API key as environment variable
                if (!string.IsNullOrEmpty(_apiConfig.OpenAI.ApiKey))
                {
                    psi.EnvironmentVariables["OPENAI_API_KEY"] = _apiConfig.OpenAI.ApiKey;
                }

                using var process = new Process { StartInfo = psi };
                process.Start();

                // Send input
                await process.StandardInput.WriteAsync(jsonInput);
                await process.StandardInput.FlushAsync();
                process.StandardInput.Close();

                // Read output
                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();

                await process.WaitForExitAsync();

                if (process.ExitCode != 0)
                {
                    _logger.LogError($"Python script failed with exit code {process.ExitCode}: {error}");
                    return 0.0;
                }

                // Parse result
                var result = JsonSerializer.Deserialize<JsonElement>(output);
                
                if (result.ValueKind == JsonValueKind.Array)
                {
                    // Multiple results - calculate average
                    var sentiments = new List<double>();
                    foreach (var item in result.EnumerateArray())
                    {
                        if (item.TryGetProperty("sentiment_score", out var scoreProperty) && 
                            scoreProperty.TryGetDouble(out var score))
                        {
                            sentiments.Add(score);
                        }
                    }
                    return sentiments.Count > 0 ? sentiments.Average() : 0.0;
                }
                else if (result.TryGetProperty("sentiment_score", out var sentimentProperty) && 
                         sentimentProperty.TryGetDouble(out var sentiment))
                {
                    return sentiment;
                }

                return 0.0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calling Python sentiment analysis script");
                return 0.0;
            }
        }

        /// <summary>
        /// Gets Bloomberg-specific YouTube content URLs for a symbol
        /// </summary>
        private async Task<List<string>> GetBloombergContentAsync(string symbol)
        {
            // In a real implementation, this would search for Bloomberg videos about the symbol
            // For now, return some Bloomberg live streams and channels
            var urls = new List<string>();

            // Bloomberg live streams and popular financial content
            var bloombergUrls = new[]
            {
                "https://www.youtube.com/watch?v=dp8PhLsUcFE", // Bloomberg TV Live
                "https://www.youtube.com/watch?v=Ga3maNZ0x0w", // Bloomberg Markets
                "https://www.youtube.com/c/Bloomberg" // Bloomberg channel
            };

            // Add relevant Bloomberg URLs (this could be enhanced with actual YouTube search)
            urls.AddRange(bloombergUrls);

            return urls;
        }

        /// <summary>
        /// Gets company-specific YouTube content URLs for a symbol
        /// </summary>
        private async Task<List<string>> GetCompanyContentAsync(string symbol)
        {
            var urls = new List<string>();
            
            // In a real implementation, this would search for the company's official YouTube channel
            // and recent videos about the company
            
            return urls;
        }

        /// <summary>
        /// Gets news-related YouTube content URLs for a symbol
        /// </summary>
        private async Task<List<string>> GetNewsContentAsync(string symbol)
        {
            var urls = new List<string>();
            
            // In a real implementation, this would search for financial news videos about the symbol
            
            return urls;
        }

        /// <summary>
        /// Cache item for sentiment data
        /// </summary>
        private class SentimentCacheItem
        {
            public double Sentiment { get; set; }
            public DateTime Timestamp { get; set; }
        }
    }
}