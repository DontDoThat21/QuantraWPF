using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Quantra.Configuration;
using Quantra.Configuration.Models;
using Quantra.Models;
using Quantra.Services;
using Quantra.Services.Interfaces;
using Quantra.ViewModels;
using Quantra.Controls;
using IConfigurationManager = Quantra.Configuration.IConfigurationManager;
using ConfigurationManager = Quantra.Configuration.ConfigurationManager;

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
            
            // Register UserSettings from database with error handling for DI
            services.AddSingleton<UserSettings>(sp => 
            {
                try
                {
                    return DatabaseMonolith.GetUserSettings();
                }
                catch (Exception)
                {
                    // If database is not available during DI initialization,
                    // return a default UserSettings instance to prevent DI failures
                    return new UserSettings();
                }
            });
            
            // Register audio service
            services.AddSingleton<IAudioService, AudioService>();
                
            // Register services
            services.AddSingleton<ISettingsService, SettingsService>();
            services.AddSingleton<INotificationService, NotificationService>();
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
            
            // Register ViewModels
            services.AddTransient<PredictionAnalysisViewModel>();
            
            // Register Views
            services.AddTransient<PredictionAnalysisControl>();
            
            return services;
        }
    }
}