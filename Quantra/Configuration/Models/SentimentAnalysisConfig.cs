using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Quantra.Configuration.Validation;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// Sentiment analysis configuration
    /// </summary>
    public class SentimentAnalysisConfig : ConfigModelBase
    {
        /// <summary>
        /// News sentiment configuration
        /// </summary>
        public NewsSentimentConfig News
        {
            get => Get<NewsSentimentConfig>(new NewsSentimentConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// Analyst ratings configuration
        /// </summary>
        public AnalystRatingsConfig AnalystRatings
        {
            get => Get<AnalystRatingsConfig>(new AnalystRatingsConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// Insider trading configuration
        /// </summary>
        public InsiderTradingConfig InsiderTrading
        {
            get => Get<InsiderTradingConfig>(new InsiderTradingConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// OpenAI sentiment analysis configuration
        /// </summary>
        public OpenAiSentimentConfig OpenAI
        {
            get => Get<OpenAiSentimentConfig>(new OpenAiSentimentConfig());
            set => Set(value);
        }
    }
    
    /// <summary>
    /// News sentiment configuration
    /// </summary>
    public class NewsSentimentConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable news sentiment analysis
        /// </summary>
        public bool EnableNewsSentimentAnalysis
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// News article refresh interval in minutes
        /// </summary>
        [ConfigurationRange(5, 1440)]
        public int NewsArticleRefreshIntervalMinutes
        {
            get => Get(30);
            set => Set(value);
        }
        
        /// <summary>
        /// Maximum news articles per symbol
        /// </summary>
        [ConfigurationRange(1, 100)]
        public int MaxNewsArticlesPerSymbol
        {
            get => Get(15);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable news source filtering
        /// </summary>
        public bool EnableNewsSourceFiltering
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Enabled news sources
        /// </summary>
        public Dictionary<string, bool> EnabledNewsSources
        {
            get => Get<Dictionary<string, bool>>(new Dictionary<string, bool>
            {
                { "bloomberg.com", true },
                { "cnbc.com", true },
                { "wsj.com", true },
                { "reuters.com", true },
                { "marketwatch.com", true },
                { "finance.yahoo.com", true },
                { "ft.com", true }
            });
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Analyst ratings configuration
    /// </summary>
    public class AnalystRatingsConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable analyst ratings
        /// </summary>
        public bool EnableAnalystRatings
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Ratings cache expiry in hours
        /// </summary>
        [ConfigurationRange(1, 240)]
        public int? RatingsCacheExpiryHours
        {
            get => Get<int?>(24);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable rating change alerts
        /// </summary>
        public bool EnableRatingChangeAlerts
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable consensus change alerts
        /// </summary>
        public bool EnableConsensusChangeAlerts
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Analyst rating sentiment weight
        /// </summary>
        [ConfigurationRange(0.1, 10.0)]
        public double AnalystRatingSentimentWeight
        {
            get => Get(2.0);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Insider trading configuration
    /// </summary>
    public class InsiderTradingConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable insider trading analysis
        /// </summary>
        public bool EnableInsiderTradingAnalysis
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Insider data refresh interval in minutes
        /// </summary>
        [ConfigurationRange(15, 1440)]
        public int InsiderDataRefreshIntervalMinutes
        {
            get => Get(120);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable insider trading alerts
        /// </summary>
        public bool EnableInsiderTradingAlerts
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Track notable insiders
        /// </summary>
        public bool TrackNotableInsiders
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Insider trading sentiment weight
        /// </summary>
        [ConfigurationRange(0.1, 10.0)]
        public double InsiderTradingSentimentWeight
        {
            get => Get(2.5);
            set => Set(value);
        }
        
        /// <summary>
        /// Highlight CEO transactions
        /// </summary>
        public bool HighlightCEOTransactions
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Highlight options activity
        /// </summary>
        public bool HighlightOptionsActivity
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable insider transaction notifications
        /// </summary>
        public bool EnableInsiderTransactionNotifications
        {
            get => Get(true);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// OpenAI sentiment analysis configuration
    /// </summary>
    public class OpenAiSentimentConfig : ConfigModelBase
    {
        /// <summary>
        /// Enable OpenAI sentiment analysis
        /// </summary>
        public bool EnableOpenAiSentimentAnalysis
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Cache expiry time for OpenAI sentiment analysis results in minutes
        /// </summary>
        [ConfigurationRange(5, 1440)]
        public int CacheExpiryMinutes
        {
            get => Get(30);
            set => Set(value);
        }
        
        /// <summary>
        /// Use OpenAI for enhanced prediction explanations
        /// </summary>
        public bool EnableEnhancedPredictionExplanations
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Maximum tokens to use in OpenAI requests
        /// </summary>
        [ConfigurationRange(100, 8000)]
        public int MaxTokens
        {
            get => Get(1000);
            set => Set(value);
        }
        
        /// <summary>
        /// OpenAI sentiment weight in combined sentiment score
        /// </summary>
        [ConfigurationRange(0.1, 10.0)]
        public double OpenAiSentimentWeight
        {
            get => Get(3.0); // Higher weight than other sentiment sources
            set => Set(value);
        }
        
        /// <summary>
        /// Use context-aware prompts for specific data sources
        /// </summary>
        public bool UseContextAwarePrompts
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable detailed sentiment score breakdown
        /// </summary>
        public bool EnableDetailedSentimentBreakdown
        {
            get => Get(true);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable event detection in analyzed content
        /// </summary>
        public bool EnableEventDetection
        {
            get => Get(true);
            set => Set(value);
        }
    }
}