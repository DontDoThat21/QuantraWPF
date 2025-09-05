using System; // Added for Exception reference
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Quantra.Configuration;
using Quantra.Configuration.Models;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels;
using Quantra.Controls;
using IConfigurationManager = Quantra.Configuration.IConfigurationManager;
using ConfigurationManager = Quantra.Configuration.ConfigurationManager;
using Quantra.DAL.Services; // Added for concrete service registrations

namespace Quantra.Extensions
{
    /// <summary>
    /// Extension methods for service collection
    /// </summary>
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Add Quantra services to the service collection
        /// </summary>
        /// <param name="services">The service collection to add services to</param>
        /// <returns>The service collection with Quantra services added</returns>
        public static IServiceCollection AddQuantraServices(this IServiceCollection services)
        {
            // Register configuration management
            services.AddSingleton<IConfigurationManager>(sp => 
                new ConfigurationManager(sp.GetRequiredService<IConfiguration>()));
                
            // Register configuration models as singletons
            services.AddSingleton(sp => sp.GetRequiredService<IConfigurationManager>().GetSection<AppConfig>(""));
            services.AddSingleton(sp => sp.GetRequiredService<IConfigurationManager>().GetSection<ApiConfig>("Api"));
            services.AddSingleton(sp => sp.GetRequiredService<IConfigurationManager>().GetSection<TradingConfig>("Trading"));
            
            // Register database configuration bridge
            services.AddSingleton<DatabaseConfigBridge>();
            
            // Audio and notification services depend on UserSettings from DatabaseMonolith, so construct via factories
            services.AddSingleton<IAudioService>(sp => new AudioService(DatabaseMonolith.GetUserSettings()));
            services.AddSingleton<ISettingsService, SettingsService>();
            services.AddSingleton<INotificationService>(sp =>
            {
                var userSettings = DatabaseMonolith.GetUserSettings();
                var audio = sp.GetRequiredService<IAudioService>();
                var settings = sp.GetRequiredService<ISettingsService>();
                return new NotificationService(userSettings, audio, settings);
            });

            // Core services
            services.AddSingleton<ITechnicalIndicatorService, TechnicalIndicatorService>();
            services.AddSingleton<IAlphaVantageService, AlphaVantageService>();
            services.AddSingleton<IEmailService, EmailService>();
            services.AddSingleton<ISmsService, SmsService>();
            services.AddSingleton<ITradingService, TradingService>();
            services.AddSingleton<IStockDataCacheService, StockDataCacheService>();
            
            // System Health Monitoring Services
            services.AddSingleton<IApiConnectivityService, ApiConnectivityService>();
            services.AddSingleton<RealTimeInferenceService>();
            services.AddSingleton<SystemHealthMonitorService>();

            // Register sentiment services and OpenAI helpers
            // Prefer OpenAI-backed implementation for ISocialMediaSentimentService
            services.AddSingleton<ISocialMediaSentimentService>(sp =>
            {
                var configMgr = sp.GetService<IConfigurationManager>();
                // Pass the configuration manager into the OpenAISentimentService (constructor accepts object)
                return new OpenAISentimentService(logger: null, configManager: configMgr);
            });

            // Register prediction enhancement service via factory to satisfy its constructor
            services.AddSingleton<OpenAIPredictionEnhancementService>(sp =>
            {
                var sentiment = sp.GetRequiredService<ISocialMediaSentimentService>();
                var configMgr = sp.GetService<IConfigurationManager>();
                return new OpenAIPredictionEnhancementService(sentiment, configMgr, null);
            });
            
            // Register ViewModels
            services.AddTransient<PredictionAnalysisViewModel>();
            
            // Register Views
            services.AddTransient<PredictionAnalysisControl>();
            
            return services;
        }
    }
}