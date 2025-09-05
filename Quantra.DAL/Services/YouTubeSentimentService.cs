using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.CrossCutting.Logging;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for analyzing sentiment from YouTube videos, particularly Bloomberg 24/7 live streams.
    /// Integrates audio extraction, transcription, and financial sentiment analysis.
    /// </summary>
    public class YouTubeSentimentService : ISocialMediaSentimentService
    {
        private readonly ILogger _logger;
        private readonly string _pythonScriptPath;
        private readonly Dictionary<string, SentimentCacheItem> _sentimentCache = new Dictionary<string, SentimentCacheItem>();

        /// <summary>
        /// Constructor for YouTubeSentimentService
        /// </summary>
        /// <remarks>
        /// Accepts loosely-typed parameters to avoid cross-project hard references.
        /// If a logger is not provided, falls back to the application's logging system.
        /// </remarks>
        public YouTubeSentimentService(object logger = null, object configManager = null)
        {
            // Prefer provided logger if it matches our logging abstraction, otherwise get a default one
            _logger = logger as ILogger ?? Log.ForType<YouTubeSentimentService>();

            // Python script default path under app base directory
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
                    _logger.Information("Returning cached sentiment for {Symbol}: {Sentiment}", symbol, cachedItem.Sentiment);
                    return cachedItem.Sentiment;
                }

                // For now, fetch recent content and analyze it
                var urls = await FetchRecentContentAsync(symbol, 5);
                if (urls.Count == 0)
                {
                    _logger.Warning("No YouTube content found for symbol: {Symbol}", symbol);
                    return 0.0;
                }

                var sentiment = await AnalyzeSentimentAsync(urls);

                // Cache the result
                _sentimentCache[cacheKey] = new SentimentCacheItem
                {
                    Sentiment = sentiment,
                    Timestamp = DateTime.Now
                };

                _logger.Information("YouTube sentiment analysis completed for {Symbol}: {Sentiment}", symbol, sentiment);
                return sentiment;
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error getting YouTube sentiment for symbol: {Symbol}", symbol);
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
                    url,
                    context = "Bloomberg financial news and market analysis"
                });

                var jsonInput = JsonSerializer.Serialize(urlData);

                // Call Python script
                var sentiment = await CallPythonSentimentAnalysis(jsonInput);

                _logger.Information("YouTube sentiment analysis completed for {Count} URLs: {Sentiment}", urls.Count, sentiment);
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
                _logger.Error(ex, "Error getting detailed YouTube sentiment for symbol: {Symbol}", symbol);
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
                _logger.Error(ex, "Error fetching YouTube content for symbol: {Symbol}", symbol);
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
                var data = new { url, context };
                var jsonInput = JsonSerializer.Serialize(data);

                return await CallPythonSentimentAnalysis(jsonInput);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error analyzing sentiment for YouTube URL: {Url}", url);
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

                // Ensure OpenAI API key is available via environment variable if set
                var openAiApiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
                if (!string.IsNullOrWhiteSpace(openAiApiKey))
                {
                    psi.EnvironmentVariables["OPENAI_API_KEY"] = openAiApiKey;
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
                    _logger.Error("Python script failed with exit code {ExitCode}: {Error}", process.ExitCode, error);
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
                else if (result.ValueKind == JsonValueKind.Object &&
                         result.TryGetProperty("sentiment_score", out var sentimentProperty) &&
                         sentimentProperty.TryGetDouble(out var sentiment))
                {
                    return sentiment;
                }

                return 0.0;
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error calling Python sentiment analysis script");
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