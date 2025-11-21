using System;
using Quantra.Models;
using Quantra.Configuration.Models;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using Quantra.DAL.Services;

namespace Quantra.Configuration
{
    /// <summary>
    /// Bridge between new configuration system and DatabaseMonolith
    /// </summary>
    /// todo: remove me
    public class DatabaseConfigBridge : IDisposable
    {
        private readonly IConfigurationManager _configManager;
        private readonly IConfiguration _configuration;
        private readonly AppConfig _appConfig;
        private readonly UserSettingsService _userSettingsService;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="configManager">The configuration manager</param>
        /// <param name="configuration">The raw configuration</param>
        public DatabaseConfigBridge(IConfigurationManager configManager, IConfiguration configuration, UserSettingsService userSettingsService)
        {
            _configManager = configManager ?? throw new ArgumentNullException(nameof(configManager));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            // Get top-level configuration - use empty string for root level
            _appConfig = _configManager.GetSection<AppConfig>("");

            // Register for configuration changes
            _configManager.ConfigurationChanged += OnConfigurationChanged;

            // Initial sync from config to database
            SyncConfigToDatabase();
        }

        /// <summary>
        /// Synchronize configuration to database
        /// </summary>
        public void SyncConfigToDatabase()
        {
            try
            {
                // Create user settings from configuration
                var settings = CreateUserSettingsFromConfig();

                // Save to database
                DatabaseMonolith.SaveUserSettings(settings);

                //DatabaseMonolith.Log("Info", "Configuration synchronized to database");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to synchronize configuration to database", ex.ToString());
            }
        }

        /// <summary>
        /// Synchronize database to configuration
        /// </summary>
        public void SyncDatabaseToConfig()
        {
            try
            {
                // Get user settings from database
                var settings = _userSettingsService.GetUserSettings();

                // Update config values (but don't persist yet)
                UpdateConfigFromUserSettings(settings, false);

                // Persist all changes at once
                _configManager.SaveChangesAsync().GetAwaiter().GetResult();

                //DatabaseMonolith.Log("Info", "Database configuration synchronized to config system");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to synchronize database to configuration", ex.ToString());
            }
        }

        /// <summary>
        /// Create UserSettings from config
        /// </summary>
        /// <returns>UserSettings object populated from config</returns>
        private UserSettings CreateUserSettingsFromConfig()
        {
            var settings = new UserSettings();
            
            // API settings
            settings.EnableApiModalChecks = _appConfig.Api.AlphaVantage.EnableApiModalChecks;
            settings.ApiTimeoutSeconds = _appConfig.Api.AlphaVantage.DefaultTimeout;
            
            // Cache settings
            settings.CacheDurationMinutes = _appConfig.Cache.CacheDurationMinutes;
            settings.EnableHistoricalDataCache = _appConfig.Cache.EnableHistoricalDataCache;
            
            // UI settings
            settings.EnableDarkMode = _appConfig.UI.EnableDarkMode;
            settings.ChartUpdateIntervalSeconds = _appConfig.UI.ChartUpdateIntervalSeconds;
            settings.DefaultGridRows = _appConfig.UI.DefaultGridRows;
            settings.DefaultGridColumns = _appConfig.UI.DefaultGridColumns;
            settings.GridBorderColor = _appConfig.UI.GridBorderColor;
            
            // Notification settings
            settings.EnablePriceAlerts = _appConfig.Notifications.EnablePriceAlerts;
            settings.EnableTradeNotifications = _appConfig.Notifications.EnableTradeNotifications;
            
            // Email settings
            settings.AlertEmail = _appConfig.Notifications.Email.DefaultRecipient;
            settings.EnableEmailAlerts = _appConfig.Notifications.Email.EnableEmailAlerts;
            settings.EnableStandardAlertEmails = _appConfig.Notifications.Email.EnableStandardAlertEmails;
            settings.EnableOpportunityAlertEmails = _appConfig.Notifications.Email.EnableOpportunityAlertEmails;
            settings.EnablePredictionAlertEmails = _appConfig.Notifications.Email.EnablePredictionAlertEmails;
            settings.EnableGlobalAlertEmails = _appConfig.Notifications.Email.EnableGlobalAlertEmails;
            settings.EnableSystemHealthAlertEmails = _appConfig.Notifications.Email.EnableSystemHealthAlertEmails;
            
            // SMS settings
            settings.AlertPhoneNumber = _appConfig.Notifications.SMS.DefaultRecipient;
            settings.EnableSmsAlerts = _appConfig.Notifications.SMS.EnableSmsAlerts;
            settings.EnableStandardAlertSms = _appConfig.Notifications.SMS.EnableStandardAlertSms;
            settings.EnableOpportunityAlertSms = _appConfig.Notifications.SMS.EnableOpportunityAlertSms;
            settings.EnablePredictionAlertSms = _appConfig.Notifications.SMS.EnablePredictionAlertSms;
            settings.EnableGlobalAlertSms = _appConfig.Notifications.SMS.EnableGlobalAlertSms;
            
            // Push notification settings
            settings.PushNotificationUserId = _appConfig.Notifications.Push.UserId;
            settings.EnablePushNotifications = _appConfig.Notifications.Push.EnablePushNotifications;
            settings.EnableStandardAlertPushNotifications = _appConfig.Notifications.Push.EnableStandardAlertPushNotifications;
            settings.EnableOpportunityAlertPushNotifications = _appConfig.Notifications.Push.EnableOpportunityAlertPushNotifications;
            settings.EnablePredictionAlertPushNotifications = _appConfig.Notifications.Push.EnablePredictionAlertPushNotifications;
            settings.EnableGlobalAlertPushNotifications = _appConfig.Notifications.Push.EnableGlobalAlertPushNotifications;
            settings.EnableTechnicalIndicatorAlertPushNotifications = _appConfig.Notifications.Push.EnableTechnicalIndicatorAlertPushNotifications;
            settings.EnableSentimentShiftAlertPushNotifications = _appConfig.Notifications.Push.EnableSentimentShiftAlertPushNotifications;
            settings.EnableSystemHealthAlertPushNotifications = _appConfig.Notifications.Push.EnableSystemHealthAlertPushNotifications;
            settings.EnableTradeExecutionPushNotifications = _appConfig.Notifications.Push.EnableTradeExecutionPushNotifications;
            
            // Sound settings
            settings.EnableAlertSounds = _appConfig.Notifications.Sound.EnableAlertSounds;
            settings.DefaultAlertSound = _appConfig.Notifications.Sound.DefaultAlertSound;
            settings.DefaultOpportunitySound = _appConfig.Notifications.Sound.DefaultOpportunitySound;
            settings.DefaultPredictionSound = _appConfig.Notifications.Sound.DefaultPredictionSound;
            settings.DefaultTechnicalIndicatorSound = _appConfig.Notifications.Sound.DefaultTechnicalIndicatorSound;
            settings.AlertVolume = _appConfig.Notifications.Sound.AlertVolume;
            
            // Visual indicator settings
            settings.EnableVisualIndicators = _appConfig.Notifications.Visual.EnableVisualIndicators;
            settings.DefaultVisualIndicatorType = _appConfig.Notifications.Visual.DefaultVisualIndicatorType;
            settings.DefaultVisualIndicatorColor = _appConfig.Notifications.Visual.DefaultVisualIndicatorColor;
            settings.VisualIndicatorDuration = _appConfig.Notifications.Visual.VisualIndicatorDuration;
            
            // Trading settings
            settings.EnablePaperTrading = _appConfig.Trading.EnablePaperTrading;
            settings.RiskLevel = _appConfig.Trading.RiskLevel;
            settings.AccountSize = _appConfig.Trading.AccountSize;
            settings.BaseRiskPercentage = _appConfig.Trading.BaseRiskPercentage;
            settings.PositionSizingMethod = _appConfig.Trading.PositionSizingMethod;
            settings.MaxPositionSizePercent = _appConfig.Trading.MaxPositionSizePercent;
            settings.FixedTradeAmount = _appConfig.Trading.FixedTradeAmount;
            settings.UseVolatilityBasedSizing = _appConfig.Trading.UseVolatilityBasedSizing;
            settings.ATRMultiple = _appConfig.Trading.ATRMultiple;
            settings.UseKellyCriterion = _appConfig.Trading.UseKellyCriterion;
            settings.HistoricalWinRate = _appConfig.Trading.HistoricalWinRate;
            settings.HistoricalRewardRiskRatio = _appConfig.Trading.HistoricalRewardRiskRatio;
            settings.KellyFractionMultiplier = _appConfig.Trading.KellyFractionMultiplier;
            
            // News sentiment analysis settings
            settings.EnableNewsSentimentAnalysis = _appConfig.SentimentAnalysis.News.EnableNewsSentimentAnalysis;
            settings.NewsArticleRefreshIntervalMinutes = _appConfig.SentimentAnalysis.News.NewsArticleRefreshIntervalMinutes;
            settings.MaxNewsArticlesPerSymbol = _appConfig.SentimentAnalysis.News.MaxNewsArticlesPerSymbol;
            settings.EnableNewsSourceFiltering = _appConfig.SentimentAnalysis.News.EnableNewsSourceFiltering;
            settings.EnabledNewsSources = _appConfig.SentimentAnalysis.News.EnabledNewsSources;
            
            // Analyst rating settings
            settings.EnableAnalystRatings = _appConfig.SentimentAnalysis.AnalystRatings.EnableAnalystRatings;
            settings.RatingsCacheExpiryHours = _appConfig.SentimentAnalysis.AnalystRatings.RatingsCacheExpiryHours;
            settings.EnableRatingChangeAlerts = _appConfig.SentimentAnalysis.AnalystRatings.EnableRatingChangeAlerts;
            settings.EnableConsensusChangeAlerts = _appConfig.SentimentAnalysis.AnalystRatings.EnableConsensusChangeAlerts;
            settings.AnalystRatingSentimentWeight = _appConfig.SentimentAnalysis.AnalystRatings.AnalystRatingSentimentWeight;
            
            // Insider trading settings
            settings.EnableInsiderTradingAnalysis = _appConfig.SentimentAnalysis.InsiderTrading.EnableInsiderTradingAnalysis;
            settings.InsiderDataRefreshIntervalMinutes = _appConfig.SentimentAnalysis.InsiderTrading.InsiderDataRefreshIntervalMinutes;
            settings.EnableInsiderTradingAlerts = _appConfig.SentimentAnalysis.InsiderTrading.EnableInsiderTradingAlerts;
            settings.TrackNotableInsiders = _appConfig.SentimentAnalysis.InsiderTrading.TrackNotableInsiders;
            settings.InsiderTradingSentimentWeight = _appConfig.SentimentAnalysis.InsiderTrading.InsiderTradingSentimentWeight;
            settings.HighlightCEOTransactions = _appConfig.SentimentAnalysis.InsiderTrading.HighlightCEOTransactions;
            settings.HighlightOptionsActivity = _appConfig.SentimentAnalysis.InsiderTrading.HighlightOptionsActivity;
            settings.EnableInsiderTransactionNotifications = _appConfig.SentimentAnalysis.InsiderTrading.EnableInsiderTransactionNotifications;
            
            return settings;
        }

        /// <summary>
        /// Update configuration from UserSettings
        /// </summary>
        /// <param name="settings">The UserSettings to copy from</param>
        /// <param name="persist">Whether to persist changes immediately</param>
        private void UpdateConfigFromUserSettings(UserSettings settings, bool persist = true)
        {
            // API settings
            _configManager.SetValue("Api:AlphaVantage:EnableApiModalChecks", settings.EnableApiModalChecks, false);
            _configManager.SetValue("Api:AlphaVantage:DefaultTimeout", settings.ApiTimeoutSeconds, false);
            
            // Cache settings
            _configManager.SetValue("Cache:CacheDurationMinutes", settings.CacheDurationMinutes, false);
            _configManager.SetValue("Cache:EnableHistoricalDataCache", settings.EnableHistoricalDataCache, false);
            
            // UI settings
            _configManager.SetValue("UI:EnableDarkMode", settings.EnableDarkMode, false);
            _configManager.SetValue("UI:ChartUpdateIntervalSeconds", settings.ChartUpdateIntervalSeconds, false);
            _configManager.SetValue("UI:DefaultGridRows", settings.DefaultGridRows, false);
            _configManager.SetValue("UI:DefaultGridColumns", settings.DefaultGridColumns, false);
            _configManager.SetValue("UI:GridBorderColor", settings.GridBorderColor, false);
            
            // Notification settings
            _configManager.SetValue("Notifications:EnablePriceAlerts", settings.EnablePriceAlerts, false);
            _configManager.SetValue("Notifications:EnableTradeNotifications", settings.EnableTradeNotifications, false);
            
            // Email settings
            _configManager.SetValue("Notifications:Email:DefaultRecipient", settings.AlertEmail, false);
            _configManager.SetValue("Notifications:Email:EnableEmailAlerts", settings.EnableEmailAlerts, false);
            _configManager.SetValue("Notifications:Email:EnableStandardAlertEmails", settings.EnableStandardAlertEmails, false);
            _configManager.SetValue("Notifications:Email:EnableOpportunityAlertEmails", settings.EnableOpportunityAlertEmails, false);
            _configManager.SetValue("Notifications:Email:EnablePredictionAlertEmails", settings.EnablePredictionAlertEmails, false);
            _configManager.SetValue("Notifications:Email:EnableGlobalAlertEmails", settings.EnableGlobalAlertEmails, false);
            _configManager.SetValue("Notifications:Email:EnableSystemHealthAlertEmails", settings.EnableSystemHealthAlertEmails, false);
            
            // SMS settings
            _configManager.SetValue("Notifications:SMS:DefaultRecipient", settings.AlertPhoneNumber, false);
            _configManager.SetValue("Notifications:SMS:EnableSmsAlerts", settings.EnableSmsAlerts, false);
            _configManager.SetValue("Notifications:SMS:EnableStandardAlertSms", settings.EnableStandardAlertSms, false);
            _configManager.SetValue("Notifications:SMS:EnableOpportunityAlertSms", settings.EnableOpportunityAlertSms, false);
            _configManager.SetValue("Notifications:SMS:EnablePredictionAlertSms", settings.EnablePredictionAlertSms, false);
            _configManager.SetValue("Notifications:SMS:EnableGlobalAlertSms", settings.EnableGlobalAlertSms, false);
            
            // Push notification settings
            _configManager.SetValue("Notifications:Push:UserId", settings.PushNotificationUserId, false);
            _configManager.SetValue("Notifications:Push:EnablePushNotifications", settings.EnablePushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableStandardAlertPushNotifications", settings.EnableStandardAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableOpportunityAlertPushNotifications", settings.EnableOpportunityAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnablePredictionAlertPushNotifications", settings.EnablePredictionAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableGlobalAlertPushNotifications", settings.EnableGlobalAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableTechnicalIndicatorAlertPushNotifications", settings.EnableTechnicalIndicatorAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableSentimentShiftAlertPushNotifications", settings.EnableSentimentShiftAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableSystemHealthAlertPushNotifications", settings.EnableSystemHealthAlertPushNotifications, false);
            _configManager.SetValue("Notifications:Push:EnableTradeExecutionPushNotifications", settings.EnableTradeExecutionPushNotifications, false);
            
            // Sound settings
            _configManager.SetValue("Notifications:Sound:EnableAlertSounds", settings.EnableAlertSounds, false);
            _configManager.SetValue("Notifications:Sound:DefaultAlertSound", settings.DefaultAlertSound, false);
            _configManager.SetValue("Notifications:Sound:DefaultOpportunitySound", settings.DefaultOpportunitySound, false);
            _configManager.SetValue("Notifications:Sound:DefaultPredictionSound", settings.DefaultPredictionSound, false);
            _configManager.SetValue("Notifications:Sound:DefaultTechnicalIndicatorSound", settings.DefaultTechnicalIndicatorSound, false);
            _configManager.SetValue("Notifications:Sound:AlertVolume", settings.AlertVolume, false);
            
            // Visual indicator settings
            _configManager.SetValue("Notifications:Visual:EnableVisualIndicators", settings.EnableVisualIndicators, false);
            _configManager.SetValue("Notifications:Visual:DefaultVisualIndicatorType", settings.DefaultVisualIndicatorType, false);
            _configManager.SetValue("Notifications:Visual:DefaultVisualIndicatorColor", settings.DefaultVisualIndicatorColor, false);
            _configManager.SetValue("Notifications:Visual:VisualIndicatorDuration", settings.VisualIndicatorDuration, false);
            
            // Trading settings
            _configManager.SetValue("Trading:EnablePaperTrading", settings.EnablePaperTrading, false);
            _configManager.SetValue("Trading:RiskLevel", settings.RiskLevel, false);
            _configManager.SetValue("Trading:AccountSize", settings.AccountSize, false);
            _configManager.SetValue("Trading:BaseRiskPercentage", settings.BaseRiskPercentage, false);
            _configManager.SetValue("Trading:PositionSizingMethod", settings.PositionSizingMethod, false);
            _configManager.SetValue("Trading:MaxPositionSizePercent", settings.MaxPositionSizePercent, false);
            _configManager.SetValue("Trading:FixedTradeAmount", settings.FixedTradeAmount, false);
            _configManager.SetValue("Trading:UseVolatilityBasedSizing", settings.UseVolatilityBasedSizing, false);
            _configManager.SetValue("Trading:ATRMultiple", settings.ATRMultiple, false);
            _configManager.SetValue("Trading:UseKellyCriterion", settings.UseKellyCriterion, false);
            _configManager.SetValue("Trading:HistoricalWinRate", settings.HistoricalWinRate, false);
            _configManager.SetValue("Trading:HistoricalRewardRiskRatio", settings.HistoricalRewardRiskRatio, false);
            _configManager.SetValue("Trading:KellyFractionMultiplier", settings.KellyFractionMultiplier, false);
            
            // News sentiment analysis settings
            _configManager.SetValue("SentimentAnalysis:News:EnableNewsSentimentAnalysis", settings.EnableNewsSentimentAnalysis, false);
            _configManager.SetValue("SentimentAnalysis:News:NewsArticleRefreshIntervalMinutes", settings.NewsArticleRefreshIntervalMinutes, false);
            _configManager.SetValue("SentimentAnalysis:News:MaxNewsArticlesPerSymbol", settings.MaxNewsArticlesPerSymbol, false);
            _configManager.SetValue("SentimentAnalysis:News:EnableNewsSourceFiltering", settings.EnableNewsSourceFiltering, false);
            _configManager.SetValue("SentimentAnalysis:News:EnabledNewsSources", settings.EnabledNewsSources, false);
            
            // Analyst rating settings
            _configManager.SetValue("SentimentAnalysis:AnalystRatings:EnableAnalystRatings", settings.EnableAnalystRatings, false);
            _configManager.SetValue("SentimentAnalysis:AnalystRatings:RatingsCacheExpiryHours", settings.RatingsCacheExpiryHours, false);
            _configManager.SetValue("SentimentAnalysis:AnalystRatings:EnableRatingChangeAlerts", settings.EnableRatingChangeAlerts, false);
            _configManager.SetValue("SentimentAnalysis:AnalystRatings:EnableConsensusChangeAlerts", settings.EnableConsensusChangeAlerts, false);
            _configManager.SetValue("SentimentAnalysis:AnalystRatings:AnalystRatingSentimentWeight", settings.AnalystRatingSentimentWeight, false);
            
            // Insider trading settings
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:EnableInsiderTradingAnalysis", settings.EnableInsiderTradingAnalysis, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:InsiderDataRefreshIntervalMinutes", settings.InsiderDataRefreshIntervalMinutes, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:EnableInsiderTradingAlerts", settings.EnableInsiderTradingAlerts, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:TrackNotableInsiders", settings.TrackNotableInsiders, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:InsiderTradingSentimentWeight", settings.InsiderTradingSentimentWeight, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:HighlightCEOTransactions", settings.HighlightCEOTransactions, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:HighlightOptionsActivity", settings.HighlightOptionsActivity, false);
            _configManager.SetValue("SentimentAnalysis:InsiderTrading:EnableInsiderTransactionNotifications", settings.EnableInsiderTransactionNotifications, false);
            
            // If persist is true, save all changes
            if (persist)
            {
                _configManager.SaveChangesAsync().GetAwaiter().GetResult();
            }
        }

        /// <summary>
        /// Configuration change handler
        /// </summary>
        /// <param name="sender">Sender object</param>
        /// <param name="e">Event arguments</param>
        private void OnConfigurationChanged(object sender, ConfigurationChangedEventArgs e)
        {
            // Sync configuration changes to database
            SyncConfigToDatabase();
        }

        /// <summary>
        /// Dispose resources
        /// </summary>
        public void Dispose()
        {
            // Unregister from configuration changes
            if (_configManager != null)
            {
                _configManager.ConfigurationChanged -= OnConfigurationChanged;
            }
        }
    }
}