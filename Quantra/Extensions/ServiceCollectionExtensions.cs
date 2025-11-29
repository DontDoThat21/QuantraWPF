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
using Quantra.DAL.Data;
using Microsoft.EntityFrameworkCore;

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
            // Register DbContext
            services.AddDbContext<QuantraDbContext>(options =>
            {
                options.UseSqlServer(ConnectionHelper.ConnectionString, sqlServerOptions =>
                {
                    sqlServerOptions.CommandTimeout(30);
                });

#if DEBUG
                options.EnableSensitiveDataLogging();
                options.EnableDetailedErrors();
#endif
            });

            // Register database initialization service
            services.AddSingleton<IDatabaseInitializationService, DatabaseInitializationService>();

            // Register configuration management
            services.AddSingleton<IConfigurationManager>(sp =>
                new ConfigurationManager(sp.GetRequiredService<IConfiguration>()));

            // Register configuration models as singletons
            services.AddSingleton(sp => sp.GetRequiredService<IConfigurationManager>().GetSection<AppConfig>(""));
            services.AddSingleton(sp => sp.GetRequiredService<IConfigurationManager>().GetSection<ApiConfig>("Api"));
            services.AddSingleton(sp => sp.GetRequiredService<IConfigurationManager>().GetSection<TradingConfig>("Trading"));

            // Register database configuration bridge
            services.AddSingleton<DatabaseConfigBridge>();

            // Register concrete types first, then interfaces pointing to the same instances
            services.AddSingleton<SettingsService>();
            services.AddSingleton<ISettingsService>(sp => sp.GetRequiredService<SettingsService>());

            services.AddSingleton<UserSettingsService>();
            services.AddSingleton<IUserSettingsService>(sp => sp.GetRequiredService<UserSettingsService>());

            // Audio and notification services depend on UserSettings from DatabaseMonolith, so construct via factories
            services.AddSingleton<IAudioService>(sp => new AudioService(sp.GetRequiredService<IUserSettingsService>().GetUserSettings()));

            // Register NotificationService as both concrete type and interface
            services.AddSingleton<NotificationService>(sp =>
            {
                var userSettings = sp.GetRequiredService<IUserSettingsService>().GetUserSettings();
                var audio = sp.GetRequiredService<IAudioService>();
                var settings = sp.GetRequiredService<ISettingsService>();
                return new NotificationService(userSettings, audio, settings);
            });
            services.AddSingleton<INotificationService>(sp => sp.GetRequiredService<NotificationService>());

            // Core services - Register concrete types first, then interfaces pointing to same instances
            services.AddSingleton<TechnicalIndicatorService>();
            services.AddSingleton<ITechnicalIndicatorService>(sp => sp.GetRequiredService<TechnicalIndicatorService>());

            services.AddSingleton<AlphaVantageService>();
            services.AddSingleton<IAlphaVantageService>(sp => sp.GetRequiredService<AlphaVantageService>());

            services.AddSingleton<HistoricalDataService>();
            services.AddSingleton<IHistoricalDataService>(sp => sp.GetRequiredService<HistoricalDataService>());

            // Register EmailService as both concrete type and interface
            services.AddSingleton<EmailService>();
            services.AddSingleton<IEmailService>(sp => sp.GetRequiredService<EmailService>());

            // Register SmsService as both concrete type and interface
            services.AddSingleton<SmsService>();
            services.AddSingleton<ISmsService>(sp => sp.GetRequiredService<SmsService>());

            // Register TradingService as both concrete type and interface
            services.AddSingleton<TradingService>();
            services.AddSingleton<ITradingService>(sp => sp.GetRequiredService<TradingService>());

            // Register StockDataCacheService as both concrete type and interface
            services.AddSingleton<StockDataCacheService>();
            services.AddSingleton<IStockDataCacheService>(sp => sp.GetRequiredService<StockDataCacheService>());

            // Logging service
            services.AddSingleton<LoggingService>();

            // Register TradeRecordService as both concrete type and interface
            services.AddSingleton<TradeRecordService>();
            services.AddSingleton<ITradeRecordService>(sp => sp.GetRequiredService<TradeRecordService>());

            // Register TransactionService as both concrete type and interface
            services.AddScoped<TransactionService>();
            services.AddScoped<ITransactionService>(sp => sp.GetRequiredService<TransactionService>());

            // System Health Monitoring Services
            services.AddSingleton<IApiConnectivityService, ApiConnectivityService>();
            services.AddSingleton<RealTimeInferenceService>();
            services.AddSingleton<StockSymbolCacheService>();
            services.AddSingleton<SystemHealthMonitorService>();

            // Register PredictionCacheService
            services.AddSingleton<PredictionCacheService>();

            // Register CacheManagementService (MarketChat story 10)
            services.AddSingleton<CacheManagementService>(sp =>
            {
                var loggingService = sp.GetRequiredService<LoggingService>();
                var predictionCacheService = sp.GetService<PredictionCacheService>();
                return new CacheManagementService(loggingService, predictionCacheService);
            });
            services.AddSingleton<ICacheManagementService>(sp => sp.GetRequiredService<CacheManagementService>());

            // Register sentiment services and OpenAI helpers
            // Prefer OpenAI-backed implementation for ISocialMediaSentimentService
            services.AddSingleton<ISocialMediaSentimentService>(sp =>
            {
                var configMgr = sp.GetService<IConfigurationManager>();
                // Pass the configuration manager into the OpenAISentimentService (constructor accepts object)
                return new OpenAISentimentService(logger: null, configManager: configMgr);
            });

            // Register TwitterSentimentService for Twitter sentiment analysis
            services.AddSingleton<TwitterSentimentService>();

            // Register FinancialNewsSentimentService for financial news analysis (separate from main social media)
            services.AddSingleton<FinancialNewsSentimentService>(sp =>
            {
                var userSettings = sp.GetRequiredService<IUserSettingsService>().GetUserSettings();
                return new FinancialNewsSentimentService(userSettings);
            });

            // Register EarningsTranscriptService
            services.AddSingleton<EarningsTranscriptService>();
            services.AddSingleton<IEarningsTranscriptService>(sp => sp.GetRequiredService<EarningsTranscriptService>());

            // Register AnalystRatingService
            services.AddSingleton<AnalystRatingService>(sp =>
            {
                var userSettings = sp.GetRequiredService<IUserSettingsService>().GetUserSettings();
                var alertPublisher = sp.GetService<IAlertPublisher>();
                var loggingService = sp.GetRequiredService<LoggingService>();
                return new AnalystRatingService(userSettings, alertPublisher, loggingService);
            });
            services.AddSingleton<IAnalystRatingService>(sp => sp.GetRequiredService<AnalystRatingService>());

            // Register InsiderTradingService
            services.AddSingleton<InsiderTradingService>(sp =>
            {
                var userSettings = sp.GetRequiredService<IUserSettingsService>().GetUserSettings();
                return new InsiderTradingService(userSettings);
            });
            services.AddSingleton<IInsiderTradingService>(sp => sp.GetRequiredService<InsiderTradingService>());

            // Register AnalystAlertService
            services.AddSingleton<AnalystAlertService>(sp =>
            {
                var analystRatingService = sp.GetRequiredService<IAnalystRatingService>();
                return new AnalystAlertService(analystRatingService);
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
            services.AddTransient<LoginWindowViewModel>();
            services.AddTransient<CreateTabWindowViewModel>();
            services.AddTransient<AlertsControlViewModel>();
            services.AddTransient<MoveControlWindowViewModel>();
            services.AddTransient<TradingRulesControlViewModel>();
            services.AddTransient<TransactionsViewModel>();
            services.AddTransient<ResizeControlWindowViewModel>();
            services.AddTransient<SentimentDashboardControlViewModel>(sp =>
            {
                var userSettings = sp.GetRequiredService<IUserSettingsService>().GetUserSettings();
                var userSettingsService = sp.GetRequiredService<UserSettingsService>();
                var loggingService = sp.GetRequiredService<LoggingService>();
                var analystRatingService = sp.GetRequiredService<IAnalystRatingService>();
                var insiderTradingService = sp.GetRequiredService<IInsiderTradingService>();
                return new SentimentDashboardControlViewModel(
                    userSettings,
                    userSettingsService,
                    loggingService,
                    analystRatingService,
                    insiderTradingService);
            });
            services.AddTransient<StockExplorerViewModel>();
            services.AddTransient<BacktestResultsViewModel>();

            // Register Views
            services.AddTransient<PredictionAnalysisControl>();

            return services;
        }
    }
}