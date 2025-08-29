using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Primitives;
using System.ComponentModel;
using Quantra.Configuration;
using Quantra.Configuration.Models;
using Quantra.Services;

namespace Quantra.Tests.Services
{
    /// <summary>
    /// Tests for YouTube sentiment analysis service
    /// </summary>
    public static class YouTubeSentimentServiceTests
    {
        /// <summary>
        /// Test basic YouTube sentiment analysis functionality
        /// </summary>
        public static async Task<Dictionary<string, object>> RunBasicTestAsync()
        {
            var results = new Dictionary<string, object>();
            
            try
            {
                // Create mock configuration manager
                var configManager = new MockConfigurationManager();
                
                // Create logger (using console logger for testing)
                var logger = new MockLogger<YouTubeSentimentService>();
                
                // Create service
                var youTubeService = new YouTubeSentimentService(logger, configManager);
                
                // Test sentiment analysis with mock URLs
                var testUrls = new List<string>
                {
                    "https://www.youtube.com/watch?v=dp8PhLsUcFE", // Bloomberg TV Live
                    "https://www.youtube.com/watch?v=Ga3maNZ0x0w"  // Bloomberg Markets
                };
                
                // Test analyzing sentiment from URLs
                var sentiment = await youTubeService.AnalyzeSentimentAsync(testUrls);
                results["AnalyzeSentimentAsync"] = sentiment.ToString("F4");
                
                // Test getting symbol sentiment
                var symbolSentiment = await youTubeService.GetSymbolSentimentAsync("AAPL");
                results["GetSymbolSentimentAsync"] = symbolSentiment.ToString("F4");
                
                // Test getting detailed source sentiment
                var detailedSentiment = await youTubeService.GetDetailedSourceSentimentAsync("AAPL");
                results["DetailedSourcesCount"] = detailedSentiment.Count;
                
                // Test fetching recent content
                var recentContent = await youTubeService.FetchRecentContentAsync("AAPL", 5);
                results["RecentContentCount"] = recentContent.Count;
                
                results["TestSuccess"] = true;
                results["TestTime"] = DateTime.Now.ToString();
            }
            catch (Exception ex)
            {
                results["TestSuccess"] = false;
                results["Error"] = ex.Message;
                results["TestTime"] = DateTime.Now.ToString();
            }
            
            return results;
        }
        
        /// <summary>
        /// Test YouTube URL sentiment analysis
        /// </summary>
        public static async Task<Dictionary<string, object>> RunUrlAnalysisTestAsync()
        {
            var results = new Dictionary<string, object>();
            
            try
            {
                // Create mock configuration manager
                var configManager = new MockConfigurationManager();
                
                // Create logger
                var logger = new MockLogger<YouTubeSentimentService>();
                
                // Create service
                var youTubeService = new YouTubeSentimentService(logger, configManager);
                
                // Test single URL analysis (this should use fallback mode)
                var testUrl = "https://www.youtube.com/watch?v=dp8PhLsUcFE";
                var sentiment = await youTubeService.AnalyzeYouTubeUrlSentimentAsync(testUrl, "Bloomberg financial news");
                
                results["SingleUrlSentiment"] = sentiment.ToString("F4");
                results["TestSuccess"] = true;
                results["TestTime"] = DateTime.Now.ToString();
            }
            catch (Exception ex)
            {
                results["TestSuccess"] = false;
                results["Error"] = ex.Message;
                results["TestTime"] = DateTime.Now.ToString();
            }
            
            return results;
        }
    }
    
    /// <summary>
    /// Mock configuration manager for testing
    /// </summary>
    public class MockConfigurationManager : Quantra.Configuration.IConfigurationManager
    {
        private readonly Dictionary<string, string> _values = new();
        private readonly Dictionary<string, object> _sections = new();
        
        public event EventHandler<ConfigurationChangedEventArgs> ConfigurationChanged;
        
        public Microsoft.Extensions.Configuration.IConfiguration RawConfiguration => null; // Not used in tests
        
        public T GetSection<T>(string sectionPath) where T : class, new()
        {
            if (typeof(T) == typeof(ApiConfig))
            {
                var apiConfig = new ApiConfig
                {
                    OpenAI = new OpenAiApiConfig
                    {
                        ApiKey = "mock-api-key",
                        BaseUrl = "https://api.openai.com/v1",
                        DefaultTimeout = 30
                    }
                };
                return (T)(object)apiConfig;
            }
            
            if (typeof(T) == typeof(SentimentAnalysisConfig))
            {
                var sentimentConfig = new SentimentAnalysisConfig
                {
                    OpenAI = new OpenAiSentimentConfig()
                };
                return (T)(object)sentimentConfig;
            }
            
            return new T();
        }

        public T GetValue<T>(string key, T defaultValue = default)
        {
            if (_values.TryGetValue(key, out var value))
            {
                try
                {
                    return (T)Convert.ChangeType(value, typeof(T));
                }
                catch
                {
                    return defaultValue;
                }
            }
            return defaultValue;
        }

        public void SetValue<T>(string key, T value, bool persist = true)
        {
            var oldValue = _values.TryGetValue(key, out var existing) ? existing : null;
            _values[key] = value?.ToString() ?? string.Empty;
            
            if (persist)
            {
                ConfigurationChanged?.Invoke(this, new ConfigurationChangedEventArgs(key, oldValue, value));
            }
        }

        public Task SaveChangesAsync()
        {
            return Task.CompletedTask;
        }

        public Task ReloadAsync()
        {
            return Task.CompletedTask;
        }

        public void RegisterChangeNotifications<T>(string sectionPath, T instance) where T : class, INotifyPropertyChanged
        {
            // Mock implementation - do nothing
        }

        public string BackupConfiguration()
        {
            return "mock-backup-path";
        }

        public bool RestoreConfigurationFromBackup(string backupPath)
        {
            return true;
        }

        public void ResetToDefaults(string sectionPath = null)
        {
            if (sectionPath == null)
            {
                _values.Clear();
                _sections.Clear();
            }
        }

        // Explicit interface implementations for Microsoft.Extensions.Configuration.IConfigurationManager
        System.Collections.Generic.IEnumerable<Microsoft.Extensions.Configuration.IConfigurationSection> Microsoft.Extensions.Configuration.IConfiguration.GetChildren() => new List<Microsoft.Extensions.Configuration.IConfigurationSection>();
        Microsoft.Extensions.Primitives.IChangeToken Microsoft.Extensions.Configuration.IConfiguration.GetReloadToken() => null;
        Microsoft.Extensions.Configuration.IConfigurationSection Microsoft.Extensions.Configuration.IConfiguration.GetSection(string key) => null;
        string Microsoft.Extensions.Configuration.IConfiguration.this[string key] { get => null; set { } }
        System.Collections.Generic.IDictionary<string, object> Microsoft.Extensions.Configuration.IConfigurationBuilder.Properties => new Dictionary<string, object>();
        System.Collections.Generic.IList<Microsoft.Extensions.Configuration.IConfigurationSource> Microsoft.Extensions.Configuration.IConfigurationBuilder.Sources => new List<Microsoft.Extensions.Configuration.IConfigurationSource>();
        Microsoft.Extensions.Configuration.IConfigurationBuilder Microsoft.Extensions.Configuration.IConfigurationBuilder.Add(Microsoft.Extensions.Configuration.IConfigurationSource source) => this;
        Microsoft.Extensions.Configuration.IConfigurationRoot Microsoft.Extensions.Configuration.IConfigurationBuilder.Build() => null;
    }

    /// <summary>
    /// Mock change token for testing
    /// </summary>
    public class MockChangeToken : IChangeToken
    {
        public bool HasChanged => false;
        public bool ActiveChangeCallbacks => false;
        public IDisposable RegisterChangeCallback(Action<object> callback, object state) => new MockDisposable();
    }

    /// <summary>
    /// Mock disposable for testing
    /// </summary>
    public class MockDisposable : IDisposable
    {
        public void Dispose() { }
    }

    /// <summary>
    /// Mock configuration section for testing
    /// </summary>
    public class MockConfigurationSection : IConfigurationSection
    {
        private readonly Dictionary<string, string> _values;
        
        public MockConfigurationSection(string key, Dictionary<string, string> values)
        {
            Key = key;
            Path = key;
            _values = values;
        }

        public string this[string key] 
        { 
            get => _values.TryGetValue($"{Path}:{key}", out var value) ? value : null;
            set => _values[$"{Path}:{key}"] = value;
        }

        public string Key { get; }
        public string Path { get; }
        public string Value 
        { 
            get => _values.TryGetValue(Path, out var value) ? value : null;
            set => _values[Path] = value;
        }

        public IEnumerable<IConfigurationSection> GetChildren() => new List<IConfigurationSection>();
        public IChangeToken GetReloadToken() => new MockChangeToken();
        public IConfigurationSection GetSection(string key) => new MockConfigurationSection($"{Path}:{key}", _values);
    }

    /// <summary>
    /// Mock configuration root for testing
    /// </summary>
    public class MockConfigurationRoot : IConfigurationRoot
    {
        private readonly Dictionary<string, string> _values;
        
        public MockConfigurationRoot(Dictionary<string, string> values)
        {
            _values = values;
        }

        public string this[string key] 
        { 
            get => _values.TryGetValue(key, out var value) ? value : null;
            set => _values[key] = value;
        }

        public IEnumerable<IConfigurationProvider> Providers => new List<IConfigurationProvider>();

        public IEnumerable<IConfigurationSection> GetChildren() => new List<IConfigurationSection>();
        public IChangeToken GetReloadToken() => new MockChangeToken();
        public IConfigurationSection GetSection(string key) => new MockConfigurationSection(key, _values);
        public void Reload() { }
    }

    /// <summary>
    /// Mock logger for testing
    /// </summary>
    public class MockLogger<T> : ILogger<T>
    {
        public IDisposable BeginScope<TState>(TState state) => new MockDisposable();
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            // Mock implementation - do nothing or write to console if needed
            Console.WriteLine($"[{logLevel}] {formatter(state, exception)}");
        }
    }
}