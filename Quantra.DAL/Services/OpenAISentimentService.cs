using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Quantra.CrossCutting.Logging;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service to perform sentiment analysis using OpenAI's GPT models.
    /// Provides deeper, more context-aware sentiment analysis compared to traditional models.
    /// </summary>
    public class OpenAISentimentService : ISocialMediaSentimentService
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger _logger;
        private readonly dynamic _apiConfig; // dynamic to avoid hard project refs for DI flexibility
        private readonly dynamic _sentimentConfig; // dynamic to avoid hard project refs for DI flexibility
        private readonly Dictionary<string, SentimentCacheItem> _sentimentCache = new Dictionary<string, SentimentCacheItem>();
        
        /// <summary>
        /// Flexible constructor for DI with optional logger and configuration manager
        /// </summary>
        /// <remarks>
        /// Accepts loosely-typed parameters to avoid cross-project hard references and DI mismatches.
        /// </remarks>
        public OpenAISentimentService(object logger = null, object configManager = null)
        {
            _logger = logger as ILogger ?? Log.ForType<OpenAISentimentService>();
            _httpClient = new HttpClient();

            // Build lightweight config objects from provided configuration manager (if available)
            _apiConfig = BuildApiConfig(configManager);
            _sentimentConfig = BuildSentimentConfig(configManager);
            
            // Set up the HTTP client
            string baseUrl = _apiConfig?.OpenAI?.BaseUrl ?? "https://api.openai.com";
            string apiKey = _apiConfig?.OpenAI?.ApiKey ?? Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty;
            int timeoutSeconds = _apiConfig?.OpenAI?.DefaultTimeout ?? 30;
            
            _httpClient.BaseAddress = new Uri(baseUrl);
            if (!string.IsNullOrWhiteSpace(apiKey))
            {
                _httpClient.DefaultRequestHeaders.Remove("Authorization");
                _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
            }
            _httpClient.Timeout = TimeSpan.FromSeconds(Math.Max(1, timeoutSeconds));
        }
        
        /// <summary>
        /// High-level method: fetches content and returns average sentiment for a symbol.
        /// </summary>
        public async Task<double> GetSymbolSentimentAsync(string symbol)
        {
            _logger.Information("Analyzing sentiment for {Symbol} using OpenAI", symbol);
            
            // Check cache first
            if (IsCacheValid(symbol))
            {
                _logger.Information("Using cached sentiment for {Symbol}", symbol);
                return _sentimentCache[symbol].Score;
            }
            
            // Fetch recent content for this symbol
            var content = await FetchRecentContentAsync(symbol);
            
            // Analyze sentiment
            var sentiment = await AnalyzeSentimentAsync(content);
            
            // Cache the result
            _sentimentCache[symbol] = new SentimentCacheItem
            {
                Score = sentiment,
                Timestamp = DateTime.UtcNow
            };
            
            return sentiment;
        }
        
        /// <summary>
        /// Analyzes sentiment from a list of text content using OpenAI.
        /// </summary>
        public async Task<double> AnalyzeSentimentAsync(List<string> textContent)
        {
            if (textContent == null || textContent.Count == 0)
            {
                _logger.Warning("No content provided for sentiment analysis");
                return 0.0;
            }
            
            try
            {
                // Combine the texts into one document for analysis
                // Limited to prevent exceeding token limits
                string combinedText = string.Join("\n---\n", textContent.GetRange(0, Math.Min(5, textContent.Count)));
                
                // Create the prompt for sentiment analysis
                string prompt = GenerateSentimentPrompt(combinedText);
                
                // Call the OpenAI API using the ResilienceHelper
                var response = await ResilienceHelper.ExternalApiCallAsync("OpenAI", async () => 
                {
                    var requestBody = new
                    {
                        model = _apiConfig?.OpenAI?.Model ?? "gpt-4o-mini",
                        messages = new[]
                        {
                            new { role = "system", content = "You are a financial sentiment analyst specialized in stock market analysis." },
                            new { role = "user", content = prompt }
                        },
                        temperature = _apiConfig?.OpenAI?.Temperature ?? 0.2,
                        max_tokens = _sentimentConfig?.OpenAI?.MaxTokens ?? 500
                    };
                    
                    var content = new StringContent(
                        JsonSerializer.Serialize(requestBody),
                        Encoding.UTF8,
                        "application/json"
                    );
                    
                    var httpResponse = await _httpClient.PostAsync("/v1/chat/completions", content);
                    httpResponse.EnsureSuccessStatusCode();
                    
                    var responseString = await httpResponse.Content.ReadAsStringAsync();
                    return JsonSerializer.Deserialize<OpenAIResponse>(responseString, new JsonSerializerOptions 
                    { 
                        PropertyNameCaseInsensitive = true 
                    });
                });
                
                // Parse the sentiment score from the response
                return ParseSentimentScore(response.Choices[0].Message.Content);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error analyzing sentiment with OpenAI");
                return 0.0;
            }
        }
        
        /// <summary>
        /// Gets detailed sentiment data for a symbol by source.
        /// </summary>
        public async Task<Dictionary<string, double>> GetDetailedSourceSentimentAsync(string symbol)
        {
            Dictionary<string, double> results = new Dictionary<string, double>
            {
                { "news", 0.0 },
                { "social_media", 0.0 },
                { "earnings_calls", 0.0 }
            };
            
            try
            {
                // Get news content
                var newsContent = await FetchRecentContentAsync(symbol, "news", 3);
                if (newsContent.Count > 0)
                {
                    results["news"] = await AnalyzeSentimentWithContext(newsContent, "news", symbol);
                }
                
                // Get social media content
                var socialContent = await FetchRecentContentAsync(symbol, "social", 5);
                if (socialContent.Count > 0)
                {
                    results["social_media"] = await AnalyzeSentimentWithContext(socialContent, "social_media", symbol);
                }
                
                // Get earnings calls content
                var earningsContent = await FetchRecentContentAsync(symbol, "earnings", 1);
                if (earningsContent.Count > 0)
                {
                    results["earnings_calls"] = await AnalyzeSentimentWithContext(earningsContent, "earnings_calls", symbol);
                }
                
                return results;
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error getting detailed source sentiment for {Symbol}", symbol);
                return results;
            }
        }
        
        /// <summary>
        /// Gets recent content (articles, posts, etc.) for a symbol
        /// </summary>
        public async Task<List<string>> FetchRecentContentAsync(string symbol, int count = 10)
        {
            // This implementation will delegate to other services to fetch content
            // Using dependency injection would be better, but for simplicity we'll use direct instantiation
            var results = new List<string>();
            
            try
            {
                // This method would typically gather content from various services
                // For now, we'll just return some placeholder content
                results.Add($"Recent news for {symbol}: The company reported strong earnings this quarter.");
                results.Add($"Analyst opinions on {symbol} have been mixed, with some expressing concerns about valuation.");
                results.Add($"A post on social media mentioned that {symbol} might be releasing a new product soon.");
                
                // In a real implementation, this would gather actual content from APIs
                
                return results.GetRange(0, Math.Min(count, results.Count));
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error fetching content for {Symbol}", symbol);
                return new List<string>();
            }
        }
        
        /// <summary>
        /// Fetches recent content from a specific source type
        /// </summary>
        private async Task<List<string>> FetchRecentContentAsync(string symbol, string sourceType, int count)
        {
            // This would be implemented to fetch from specific sources based on the sourceType
            // For now, return placeholders
            var results = new List<string>();
            
            switch (sourceType.ToLower())
            {
                case "news":
                    results.Add($"NEWS: {symbol} quarterly results exceed expectations");
                    results.Add($"NEWS: Analysts upgrade {symbol} rating to Buy");
                    break;
                case "social":
                    results.Add($"SOCIAL: I think {symbol} is going to do well this quarter!");
                    results.Add($"SOCIAL: Just bought more shares of {symbol}");
                    results.Add($"SOCIAL: Not sure about {symbol}'s latest product launch");
                    break;
                case "earnings":
                    results.Add($"EARNINGS CALL: {symbol} CEO: 'We're seeing strong growth in our core segments...'");
                    break;
            }
            
            return results.GetRange(0, Math.Min(count, results.Count));
        }
        
        /// <summary>
        /// Analyzes sentiment with context-specific prompts
        /// </summary>
        private async Task<double> AnalyzeSentimentWithContext(List<string> content, string sourceType, string symbol)
        {
            if (!(_sentimentConfig?.OpenAI?.UseContextAwarePrompts ?? true))
            {
                // If context-aware prompts are disabled, use the standard analysis
                return await AnalyzeSentimentAsync(content);
            }
            
            try
            {
                string systemPrompt, userPrompt;
                
                switch (sourceType.ToLower())
                {
                    case "news":
                        systemPrompt = "You are a financial analyst specialized in interpreting news sentiment for stocks.";
                        userPrompt = $"Analyze the sentiment in these news articles about {symbol}. Is it bullish or bearish? Provide a sentiment score between -1 (extremely bearish) and 1 (extremely bullish).\n\n{string.Join("\n---\n", content)}";
                        break;
                    case "social_media":
                        systemPrompt = "You are a social media sentiment analyst focusing on stock market discussions.";
                        userPrompt = $"Analyze these social media posts about {symbol}. What's the overall sentiment? Provide a sentiment score between -1 (extremely negative) and 1 (extremely positive).\n\n{string.Join("\n---\n", content)}";
                        break;
                    case "earnings_calls":
                        systemPrompt = "You are a financial analyst specializing in earnings call interpretation.";
                        userPrompt = $"Analyze this earnings call transcript for {symbol}. Is the tone bullish or bearish? Provide a sentiment score between -1 (extremely bearish) and 1 (extremely bullish).\n\n{string.Join("\n", content)}";
                        break;
                    default:
                        return await AnalyzeSentimentAsync(content);
                }
                
                // Call the OpenAI API with context-specific prompts
                var response = await ResilienceHelper.ExternalApiCallAsync("OpenAI", async () =>
                {
                    var requestBody = new
                    {
                        model = _apiConfig?.OpenAI?.Model ?? "gpt-4o-mini",
                        messages = new[]
                        {
                            new { role = "system", content = systemPrompt },
                            new { role = "user", content = userPrompt }
                        },
                        temperature = _apiConfig?.OpenAI?.Temperature ?? 0.2,
                        max_tokens = _sentimentConfig?.OpenAI?.MaxTokens ?? 500
                    };
                    
                    var content = new StringContent(
                        JsonSerializer.Serialize(requestBody),
                        Encoding.UTF8,
                        "application/json"
                    );
                    
                    var httpResponse = await _httpClient.PostAsync("/v1/chat/completions", content);
                    httpResponse.EnsureSuccessStatusCode();
                    
                    var responseString = await httpResponse.Content.ReadAsStringAsync();
                    return JsonSerializer.Deserialize<OpenAIResponse>(responseString, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                });
                
                return ParseSentimentScore(response.Choices[0].Message.Content);
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error in context-aware sentiment analysis for {SourceType}", sourceType);
                return 0.0;
            }
        }
        
        /// <summary>
        /// Generates a prompt for sentiment analysis
        /// </summary>
        private string GenerateSentimentPrompt(string content)
        {
            return $"Analyze the sentiment in the following text related to financial markets and stocks. " +
                   $"Is the sentiment bullish or bearish? " +
                   $"Provide a sentiment score between -1.0 (extremely bearish) and 1.0 (extremely bullish), " +
                   $"with 0 being neutral. Return only the numeric score at the end of your analysis.\n\n" +
                   $"Text: {content}\n\n" +
                   $"Analysis:";
        }
        
        /// <summary>
        /// Parses a sentiment score from an OpenAI response
        /// </summary>
        private double ParseSentimentScore(string responseContent)
        {
            try
            {
                // Try to find a numeric score in the response
                // Pattern to match: a number between -1 and 1, possibly with decimal points
                var scoreString = System.Text.RegularExpressions.Regex.Match(
                    responseContent, 
                    @"[-+]?\d*\.\d+|\d+"
                ).Value;
                
                if (!string.IsNullOrEmpty(scoreString) && double.TryParse(scoreString, out double score))
                {
                    // Ensure the score is within the -1 to 1 range
                    return Math.Max(-1.0, Math.Min(1.0, score));
                }
                
                // If no explicit score, analyze the text for sentiment words
                int positiveCount = CountOccurrences(responseContent.ToLower(), new[] { "bullish", "positive", "optimistic", "strong", "growth", "upside" });
                int negativeCount = CountOccurrences(responseContent.ToLower(), new[] { "bearish", "negative", "pessimistic", "weak", "decline", "downside" });
                
                if (positiveCount > negativeCount)
                {
                    return 0.5; // Default positive
                }
                else if (negativeCount > positiveCount)
                {
                    return -0.5; // Default negative
                }
                
                return 0.0; // Neutral
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Error parsing sentiment score from OpenAI response");
                return 0.0;
            }
        }
        
        /// <summary>
        /// Counts occurrences of any string in a list within a text
        /// </summary>
        private int CountOccurrences(string text, string[] words)
        {
            int count = 0;
            foreach (var word in words)
            {
                int index = 0;
                while ((index = text.IndexOf(word, index)) != -1)
                {
                    count++;
                    index += word.Length;
                }
            }
            return count;
        }
        
        /// <summary>
        /// Checks if cache for a symbol is still valid
        /// </summary>
        private bool IsCacheValid(string symbol)
        {
            if (!_sentimentCache.ContainsKey(symbol))
            {
                return false;
            }
            
            var cacheItem = _sentimentCache[symbol];
            var expiryTime = TimeSpan.FromMinutes(_sentimentConfig?.OpenAI?.CacheExpiryMinutes ?? 30);
            
            return DateTime.UtcNow - cacheItem.Timestamp < expiryTime;
        }
        
        /// <summary>
        /// Build minimal API config object from configuration manager
        /// </summary>
        private dynamic BuildApiConfig(object configManager)
        {
            dynamic root = new ExpandoObject();
            dynamic openAi = new ExpandoObject();
            openAi.BaseUrl = GetConfigValue(configManager, "ApiConfig:OpenAI:BaseUrl", "https://api.openai.com");
            openAi.ApiKey = GetConfigValue(configManager, "ApiConfig:OpenAI:ApiKey", Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? string.Empty);
            openAi.Model = GetConfigValue(configManager, "ApiConfig:OpenAI:Model", "gpt-4o-mini");
            openAi.Temperature = GetConfigValue(configManager, "ApiConfig:OpenAI:Temperature", 0.2);
            openAi.DefaultTimeout = GetConfigValue(configManager, "ApiConfig:OpenAI:DefaultTimeout", 30);
            root.OpenAI = openAi;
            return root;
        }
        
        /// <summary>
        /// Build minimal Sentiment config object from configuration manager
        /// </summary>
        private dynamic BuildSentimentConfig(object configManager)
        {
            dynamic root = new ExpandoObject();
            dynamic openAi = new ExpandoObject();
            openAi.MaxTokens = GetConfigValue(configManager, "SentimentAnalysisConfig:OpenAI:MaxTokens", 500);
            openAi.CacheExpiryMinutes = GetConfigValue(configManager, "SentimentAnalysisConfig:OpenAI:CacheExpiryMinutes", 30);
            openAi.UseContextAwarePrompts = GetConfigValue(configManager, "SentimentAnalysisConfig:OpenAI:UseContextAwarePrompts", true);
            root.OpenAI = openAi;
            return root;
        }
        
        /// <summary>
        /// Attempts to get a configuration value via reflection from a custom IConfigurationManager; falls back to defaults
        /// </summary>
        private T GetConfigValue<T>(object configManager, string key, T defaultValue)
        {
            try
            {
                if (configManager == null)
                {
                    return defaultValue;
                }
                var type = configManager.GetType();
                // Try generic GetValue<T>(string key, T defaultValue)
                var method = type.GetMethod("GetValue");
                if (method != null && method.IsGenericMethod)
                {
                    var generic = method.MakeGenericMethod(typeof(T));
                    var result = generic.Invoke(configManager, new object[] { key, defaultValue });
                    if (result is T typed)
                    {
                        return typed;
                    }
                }
                // Try indexer style: configManager[key]
                var indexer = type.GetProperty("Item", new[] { typeof(string) });
                if (indexer != null)
                {
                    var value = indexer.GetValue(configManager, new object[] { key });
                    if (value is T t)
                        return t;
                    if (value != null)
                    {
                        return (T)Convert.ChangeType(value, typeof(T));
                    }
                }
            }
            catch
            {
                // Ignore and fall back
            }
            return defaultValue;
        }
        
        /// <summary>
        /// Cache item for storing sentiment scores
        /// </summary>
        private class SentimentCacheItem
        {
            public double Score { get; set; }
            public DateTime Timestamp { get; set; }
        }
    }
}