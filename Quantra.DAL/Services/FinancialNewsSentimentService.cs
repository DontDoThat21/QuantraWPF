using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Concurrent;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.DAL.Utilities; // Added for Utilities.GetNewsApiKey

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to fetch financial news articles and perform sentiment analysis using a Python script (GPU-accelerated).
    /// Supports multiple news sources like Bloomberg, CNBC, Wall Street Journal, etc.
    /// </summary>
    public class FinancialNewsSentimentService : ISocialMediaSentimentService
    {
        private readonly string pythonScriptPath = "python/sentiment_analysis.py"; // Reuse same Python script as other services
        private readonly HttpClient _client;
        private readonly string _newsApiKey;
        private readonly UserSettings _userSettings;
        
        // Cache for recent news articles to reduce API calls
        private readonly ConcurrentDictionary<string, (List<NewsArticle> Articles, DateTime Timestamp)> _articlesCache = 
            new ConcurrentDictionary<string, (List<NewsArticle>, DateTime)>();
        
        // Financial news sources with configuration
        private readonly List<NewsSourceConfig> _newsSourceConfigs;
        
        // Default news sources if no configuration is provided
        private readonly List<string> _defaultNewsSources = new List<string> { 
            "bloomberg.com",
            "cnbc.com",
            "wsj.com",
            "reuters.com",
            "marketwatch.com",
            "finance.yahoo.com",
            "ft.com"
        };

        public FinancialNewsSentimentService(UserSettings userSettings = null)
        {
            _client = new HttpClient();
            _newsApiKey = GetNewsApiKey();
            _userSettings = userSettings ?? new UserSettings();
            
            // Initialize news source configurations
            _newsSourceConfigs = InitializeNewsSourceConfigs();
        }
        
        /// <summary>
        /// Initialize news source configurations based on settings
        /// </summary>
        private List<NewsSourceConfig> InitializeNewsSourceConfigs()
        {
            var configs = new List<NewsSourceConfig>();
            
            // Bloomberg
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "bloomberg.com", 
                Name = "Bloomberg", 
                Weight = 1.5,  // Higher weight due to high credibility
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("bloomberg.com") != true || 
                            _userSettings.EnabledNewsSources["bloomberg.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "trading", "economy", "finance" }
            });
            
            // CNBC
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "cnbc.com", 
                Name = "CNBC", 
                Weight = 1.2,
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("cnbc.com") != true || 
                            _userSettings.EnabledNewsSources["cnbc.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "trading", "investment", "finance" }
            });
            
            // Wall Street Journal
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "wsj.com", 
                Name = "Wall Street Journal", 
                Weight = 1.4,
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("wsj.com") != true || 
                            _userSettings.EnabledNewsSources["wsj.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "trading", "economy", "finance" }
            });
            
            // Reuters
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "reuters.com", 
                Name = "Reuters", 
                Weight = 1.3,
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("reuters.com") != true || 
                            _userSettings.EnabledNewsSources["reuters.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "trading", "economy", "business" }
            });
            
            // MarketWatch
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "marketwatch.com", 
                Name = "MarketWatch", 
                Weight = 1.1,
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("marketwatch.com") != true || 
                            _userSettings.EnabledNewsSources["marketwatch.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "investing", "trade", "finance" }
            });
            
            // Yahoo Finance
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "finance.yahoo.com", 
                Name = "Yahoo Finance", 
                Weight = 1.0,
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("finance.yahoo.com") != true || 
                            _userSettings.EnabledNewsSources["finance.yahoo.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "investing", "trade", "finance" }
            });
            
            // Financial Times
            configs.Add(new NewsSourceConfig 
            { 
                Domain = "ft.com", 
                Name = "Financial Times", 
                Weight = 1.4,
                IsEnabled = _userSettings.EnabledNewsSources?.ContainsKey("ft.com") != true || 
                            _userSettings.EnabledNewsSources["ft.com"],
                RelevanceKeywords = new List<string> { "market", "stocks", "trading", "economy", "finance" }
            });
            
            return configs;
        }

        /// <summary>
        /// Gets the News API key from the settings file or environment variables
        /// </summary>
        private string GetNewsApiKey()
        {
            try
            {
                // First try environment variable
                string key = Environment.GetEnvironmentVariable("NEWS_API_KEY");
                if (!string.IsNullOrEmpty(key))
                    return key;
                
                // Then try from DAL utilities (local settings file fallback)
                key = global::Quantra.DAL.Utilities.Utilities.GetNewsApiKey();
                if (!string.IsNullOrWhiteSpace(key))
                    return key;
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to get News API key", ex.ToString());
            }

            // Fallback
            return string.Empty;
        }

        /// <summary>
        /// Fetches recent news articles for a stock symbol.
        /// </summary>
        public async Task<List<string>> FetchRecentContentAsync(string symbol, int count = 10)
        {
            var newsArticles = await FetchNewsArticlesAsync(symbol, count);
            return newsArticles.Select(article => article.GetCombinedContent()).ToList();
        }

        /// <summary>
        /// Fetches recent news articles as NewsArticle objects for a stock symbol.
        /// </summary>
        public async Task<List<NewsArticle>> FetchNewsArticlesAsync(string symbol, int count = 10)
        {
            // Check cache first
            string cacheKey = $"news_{symbol.ToLower()}";
            if (_articlesCache.TryGetValue(cacheKey, out var cachedData))
            {
                // If cache is still valid (not older than configured interval)
                var cacheAgeMinutes = (DateTime.Now - cachedData.Timestamp).TotalMinutes;
                if (cacheAgeMinutes < _userSettings.NewsArticleRefreshIntervalMinutes)
                {
                    return cachedData.Articles.Take(count).ToList();
                }
            }
            
            var articles = new List<NewsArticle>();
            
            try
            {
                // Get enabled news sources
                var enabledSources = _newsSourceConfigs
                    .Where(config => config.IsEnabled)
                    .Select(config => config.Domain)
                    .ToList();
                
                if (enabledSources.Count == 0)
                {
                    // Fallback to default if no sources are enabled
                    enabledSources = _defaultNewsSources;
                }
                
                // Use NewsAPI.org to get recent financial news
                string sourcesParam = string.Join(",", enabledSources);
                int apiCount = Math.Min(100, count * 2); // Request more than needed to filter for relevance
                string url = $"https://newsapi.org/v2/everything?q={Uri.EscapeDataString(symbol)}&domains={sourcesParam}&sortBy=publishedAt&apiKey={_newsApiKey}&pageSize={apiCount}";
                
                var response = await _client.GetAsync(url);
                if (!response.IsSuccessStatusCode)
                {
                    DatabaseMonolith.Log("Warning", $"News API call failed for {symbol}: {response.StatusCode}");
                    return articles;
                }
                
                var json = await response.Content.ReadAsStringAsync();
                using var doc = JsonDocument.Parse(json);
                
                if (doc.RootElement.TryGetProperty("articles", out var articlesElement))
                {
                    foreach (var article in articlesElement.EnumerateArray())
                    {
                        var newsArticle = new NewsArticle();
                        
                        // Parse article data
                        if (article.TryGetProperty("title", out var titleProp))
                            newsArticle.Title = titleProp.GetString() ?? string.Empty;
                            
                        if (article.TryGetProperty("description", out var descProp))
                            newsArticle.Description = descProp.GetString() ?? string.Empty;
                            
                        if (article.TryGetProperty("content", out var contentProp))
                            newsArticle.Content = contentProp.GetString() ?? string.Empty;
                            
                        if (article.TryGetProperty("url", out var urlProp))
                            newsArticle.Url = urlProp.GetString() ?? string.Empty;
                            
                        if (article.TryGetProperty("publishedAt", out var pubDateProp))
                        {
                            if (DateTime.TryParse(pubDateProp.GetString(), out DateTime pubDate))
                                newsArticle.PublishedAt = pubDate;
                            else
                                newsArticle.PublishedAt = DateTime.Now;
                        }
                        
                        // Extract source information
                        if (article.TryGetProperty("source", out var sourceProp))
                        {
                            if (sourceProp.TryGetProperty("name", out var sourceNameProp))
                                newsArticle.SourceName = sourceNameProp.GetString() ?? string.Empty;
                        }
                        
                        // Determine source domain from URL
                        if (!string.IsNullOrEmpty(newsArticle.Url))
                        {
                            try
                            {
                                Uri uri = new Uri(newsArticle.Url);
                                newsArticle.SourceDomain = uri.Host;
                                
                                // Match to configured sources
                                var sourceConfig = _newsSourceConfigs.FirstOrDefault(
                                    s => uri.Host.Contains(s.Domain) || s.Domain != null && uri.Host.Contains(s.Domain));
                                
                                if (sourceConfig != null)
                                {
                                    newsArticle.SourceName = sourceConfig.Name;
                                    newsArticle.SourceDomain = sourceConfig.Domain;
                                }
                            }
                            catch
                            {
                                // If URL parsing fails, use empty domain
                                newsArticle.SourceDomain = string.Empty;
                            }
                        }
                        
                        // Calculate relevance score (based on presence of stock symbol and relevant keywords)
                        newsArticle.RelevanceScore = CalculateRelevanceScore(newsArticle, symbol);
                        
                        // Add if it passes minimum relevance threshold
                        if (newsArticle.RelevanceScore >= 0.3 && !string.IsNullOrWhiteSpace(newsArticle.GetCombinedContent()))
                        {
                            // Check for duplicates before adding
                            bool isDuplicate = articles.Any(a => a.IsSimilarTo(newsArticle));
                            if (!isDuplicate)
                            {
                                articles.Add(newsArticle);
                            }
                        }
                    }
                }
                
                // Sort by relevance and publication date
                articles = articles
                    .OrderByDescending(a => a.RelevanceScore * 0.7 + (1.0 - (DateTime.Now - a.PublishedAt).TotalDays / 30.0) * 0.3)
                    .Take(count)
                    .ToList();
                
                // Cache the results
                _articlesCache[cacheKey] = (articles, DateTime.Now);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Error fetching news articles for {symbol}", ex.ToString());
            }
            
            return articles;
        }
        
        /// <summary>
        /// Calculate relevance score for an article relative to a stock symbol
        /// </summary>
        private double CalculateRelevanceScore(NewsArticle article, string symbol)
        {
            if (article == null || string.IsNullOrEmpty(symbol))
                return 0.0;
                
            double score = 0.0;
            string content = article.GetCombinedContent().ToLowerInvariant();
            string stockSymbol = symbol.ToLowerInvariant();
            
            // Direct symbol mention is highly relevant
            if (content.Contains(stockSymbol))
                score += 0.6;
                
            // Company name mentions if different from symbol
            if (stockSymbol.Length > 2 && content.Contains(stockSymbol + " stock"))
                score += 0.2;
                
            // Check for relevant keywords from the article's source configuration
            var sourceConfig = _newsSourceConfigs.FirstOrDefault(s => s.Domain == article.SourceDomain);
            if (sourceConfig != null)
            {
                foreach (var keyword in sourceConfig.RelevanceKeywords)
                {
                    if (content.Contains(keyword.ToLowerInvariant()))
                        score += 0.05;
                }
            }
            
            // Recent articles are more relevant
            TimeSpan age = DateTime.Now - article.PublishedAt;
            if (age.TotalDays < 1)
                score += 0.15;
            else if (age.TotalDays < 3)
                score += 0.1;
                
            // Cap at 1.0
            return Math.Min(1.0, score);
        }

        /// <summary>
        /// Calls the Python sentiment analysis script and returns the average sentiment score.
        /// </summary>
        public async Task<double> AnalyzeSentimentAsync(List<string> articles)
        {
            if (articles == null || articles.Count == 0)
                return 0.0;
                
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"{pythonScriptPath}",
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                
                using var process = new Process { StartInfo = psi };
                process.Start();
                
                // Send articles as JSON to stdin (same approach as other sentiment services)
                var json = JsonSerializer.Serialize(articles);
                await process.StandardInput.WriteLineAsync(json);
                process.StandardInput.Close();
                
                // Read output (expecting a single float value)
                string output = await process.StandardOutput.ReadLineAsync();
                string error = await process.StandardError.ReadToEndAsync();
                
                // Set a generous timeout for larger batches of articles
                int timeout = Math.Max(5000, articles.Count * 500); // Base timeout + 500ms per article
                if (!process.WaitForExit(timeout)) 
                {
                    DatabaseMonolith.Log("Warning", "Python sentiment script did not exit within the timeout period. Terminating process.");
                    process.Kill(); // Forcefully terminate the process
                }
                
                if (!string.IsNullOrWhiteSpace(error))
                {
                    DatabaseMonolith.Log("Warning", $"Python sentiment script stderr: {error}");
                }
                
                if (double.TryParse(output, out double sentiment))
                    return sentiment;
                else
                    DatabaseMonolith.Log("Warning", $"Python sentiment script returned non-numeric output: {output}");
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Error running Python sentiment analysis script for news articles", ex.ToString());
            }
            
            return 0.0;
        }

        /// <summary>
        /// High-level method: fetches news articles and returns average sentiment for a symbol.
        /// </summary>
        public async Task<double> GetSymbolSentimentAsync(string symbol)
        {
            int articleCount = _userSettings?.MaxNewsArticlesPerSymbol ?? 15;
            var articles = await FetchRecentContentAsync(symbol, articleCount);
            return await AnalyzeSentimentAsync(articles);
        }
        
        /// <summary>
        /// Gets sentiment data for a symbol across multiple news sources.
        /// </summary>
        /// <returns>Dictionary with news source names as keys and sentiment scores as values.</returns>
        public async Task<Dictionary<string, double>> GetDetailedNewsSourceSentimentAsync(string symbol)
        {
            var result = new Dictionary<string, double>();
            
            // Get all news articles first
            var newsArticles = await FetchNewsArticlesAsync(symbol, _userSettings?.MaxNewsArticlesPerSymbol ?? 15);
            
            // Group by source domain
            var articlesBySource = newsArticles.GroupBy(a => a.SourceDomain);
            
            foreach (var sourceGroup in articlesBySource)
            {
                string sourceDomain = sourceGroup.Key;
                if (string.IsNullOrEmpty(sourceDomain))
                    continue;
                    
                try
                {
                    // Get all articles for this source
                    var sourceArticles = sourceGroup.Select(a => a.GetCombinedContent()).ToList();
                    
                    if (sourceArticles.Count > 0)
                    {
                        double sentiment = await AnalyzeSentimentAsync(sourceArticles);
                        result[sourceDomain] = sentiment;
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Warning", $"Error getting sentiment for {symbol} from {sourceDomain}", ex.ToString());
                }
            }
            
            return result;
        }
        
        /// <summary>
        /// Gets detailed sentiment data for a symbol by source.
        /// </summary>
        public async Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol)
        {
            // This method is implemented to satisfy the interface. For news, we just call the specific method.
            return await GetDetailedNewsSourceSentimentAsync(symbol);
        }
        
        /// <summary>
        /// Gets detailed sentiment data with news articles
        /// </summary>
        public async Task<(Dictionary<string, double> Sentiment, List<NewsArticle> Articles)> GetDetailedNewsAnalysisAsync(string symbol)
        {
            var sentiment = new Dictionary<string, double>();
            var newsArticles = await FetchNewsArticlesAsync(symbol, _userSettings?.MaxNewsArticlesPerSymbol ?? 15);
            
            // Group by source domain for sentiment analysis
            var articlesBySource = newsArticles.GroupBy(a => a.SourceDomain);
            foreach (var sourceGroup in articlesBySource)
            {
                string sourceDomain = sourceGroup.Key;
                if (string.IsNullOrEmpty(sourceDomain))
                    continue;
                    
                try
                {
                    var sourceArticles = sourceGroup.Select(a => a.GetCombinedContent()).ToList();
                    if (sourceArticles.Count > 0)
                    {
                        double sentimentScore = await AnalyzeSentimentAsync(sourceArticles);
                        sentiment[sourceDomain] = sentimentScore;
                        
                        // Also update sentiment scores in each article
                        foreach (var article in sourceGroup)
                        {
                            article.SentimentScore = sentimentScore; // For simplicity, use same score for all articles from source
                        }
                    }
                }
                catch (Exception ex)
                {
                    DatabaseMonolith.Log("Warning", $"Error analyzing sentiment for {symbol} from {sourceDomain}", ex.ToString());
                }
            }
            
            return (sentiment, newsArticles);
        }
        
        /// <summary>
        /// Clears the news cache for a symbol or all symbols
        /// </summary>
        public void ClearCache(string symbol = null)
        {
            if (string.IsNullOrEmpty(symbol))
            {
                _articlesCache.Clear();
            }
            else
            {
                string cacheKey = $"news_{symbol.ToLower()}";
                _articlesCache.TryRemove(cacheKey, out _);
            }
        }
    }
}