using System.ComponentModel.DataAnnotations;
using Quantra.Configuration.Validation;

namespace Quantra.Configuration.Models
{
    /// <summary>
    /// API configuration
    /// </summary>
    public class ApiConfig : ConfigModelBase
    {
        /// <summary>
        /// Alpha Vantage API configuration
        /// </summary>
        public AlphaVantageApiConfig AlphaVantage
        {
            get => Get<AlphaVantageApiConfig>(new AlphaVantageApiConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// News API configuration
        /// </summary>
        public NewsApiConfig News
        {
            get => Get<NewsApiConfig>(new NewsApiConfig());
            set => Set(value);
        }
        
        /// <summary>
        /// OpenAI API configuration
        /// </summary>
        public OpenAiApiConfig OpenAI
        {
            get => Get<OpenAiApiConfig>(new OpenAiApiConfig());
            set => Set(value);
        }
    }
    
    /// <summary>
    /// Alpha Vantage API configuration
    /// </summary>
    public class AlphaVantageApiConfig : ConfigModelBase
    {
        /// <summary>
        /// API key
        /// </summary>
        [RequiredInEnvironment("Production")]
        [DataType(DataType.Password)]
        public string ApiKey
        {
            get => Get(string.Empty);
            set => Set(value);
        }
        
        /// <summary>
        /// Base URL
        /// </summary>
        [Required]
        [Url]
        public string BaseUrl
        {
            get => Get("https://www.alphavantage.co/query");
            set => Set(value);
        }
        
        /// <summary>
        /// Default timeout in seconds
        /// </summary>
        [ConfigurationRange(5, 300)]
        public int DefaultTimeout
        {
            get => Get(30);
            set => Set(value);
        }
        
        /// <summary>
        /// Enable API modal checks
        /// </summary>
        public bool EnableApiModalChecks
        {
            get => Get(true);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// News API configuration
    /// </summary>
    public class NewsApiConfig : ConfigModelBase
    {
        /// <summary>
        /// API key
        /// </summary>
        [DataType(DataType.Password)]
        public string ApiKey
        {
            get => Get(string.Empty);
            set => Set(value);
        }
        
        /// <summary>
        /// Base URL
        /// </summary>
        [Required]
        [Url]
        public string BaseUrl
        {
            get => Get("https://newsapi.org/v2");
            set => Set(value);
        }
        
        /// <summary>
        /// Default timeout in seconds
        /// </summary>
        [ConfigurationRange(5, 300)]
        public int DefaultTimeout
        {
            get => Get(30);
            set => Set(value);
        }
    }
    
    /// <summary>
    /// OpenAI API configuration
    /// </summary>
    public class OpenAiApiConfig : ConfigModelBase
    {
        /// <summary>
        /// API key
        /// </summary>
        [DataType(DataType.Password)]
        public string ApiKey
        {
            get => Get(string.Empty);
            set => Set(value);
        }
        
        /// <summary>
        /// Base URL
        /// </summary>
        [Required]
        [Url]
        public string BaseUrl
        {
            get => Get("https://api.openai.com/v1");
            set => Set(value);
        }
        
        /// <summary>
        /// Model to use (e.g., gpt-4, gpt-3.5-turbo)
        /// </summary>
        [Required]
        public string Model
        {
            get => Get("gpt-3.5-turbo");
            set => Set(value);
        }
        
        /// <summary>
        /// Default timeout in seconds
        /// </summary>
        [ConfigurationRange(5, 300)]
        public int DefaultTimeout
        {
            get => Get(60);
            set => Set(value);
        }
        
        /// <summary>
        /// Temperature for model responses (0.0-1.0)
        /// </summary>
        [ConfigurationRange(0.0, 1.0)]
        public double Temperature
        {
            get => Get(0.3); // Low temperature for more consistent/deterministic responses
            set => Set(value);
        }
    }
}