using System.ComponentModel.DataAnnotations;
using Quantra.Configuration.Validation;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// Top-level application configuration
    /// </summary>
    public class AppConfig : ConfigModelBase
    {
        /// <summary>
        /// Application section
        /// </summary>
        public ApplicationConfig Application
        {
            get => Get<ApplicationConfig>(new ApplicationConfig());
            set => Set(value);
        }

        /// <summary>
        /// API configuration
        /// </summary>
        public ApiConfig Api
        {
            get => Get<ApiConfig>(new ApiConfig());
            set => Set(value);
        }

        /// <summary>
        /// Cache configuration
        /// </summary>
        public CacheConfig Cache
        {
            get => Get<CacheConfig>(new CacheConfig());
            set => Set(value);
        }

        /// <summary>
        /// UI configuration
        /// </summary>
        public UIConfig UI
        {
            get => Get<UIConfig>(new UIConfig());
            set => Set(value);
        }

        /// <summary>
        /// Notifications configuration
        /// </summary>
        public NotificationConfig Notifications
        {
            get => Get<NotificationConfig>(new NotificationConfig());
            set => Set(value);
        }

        /// <summary>
        /// Trading configuration
        /// </summary>
        public TradingConfig Trading
        {
            get => Get<TradingConfig>(new TradingConfig());
            set => Set(value);
        }

        /// <summary>
        /// Sentiment analysis configuration
        /// </summary>
        public SentimentAnalysisConfig SentimentAnalysis
        {
            get => Get<SentimentAnalysisConfig>(new SentimentAnalysisConfig());
            set => Set(value);
        }
    }

    /// <summary>
    /// Application section configuration
    /// </summary>
    public class ApplicationConfig : ConfigModelBase
    {
        /// <summary>
        /// Application name
        /// </summary>
        [Required]
        public string Name
        {
            get => Get("Quantra");
            set => Set(value);
        }

        /// <summary>
        /// Application version
        /// </summary>
        [Required]
        public string Version
        {
            get => Get("1.0.0");
            set => Set(value);
        }

        /// <summary>
        /// Application environment (Development, Staging, Production)
        /// </summary>
        [Required]
        public string Environment
        {
            get => Get("Production");
            set => Set(value);
        }
    }
}