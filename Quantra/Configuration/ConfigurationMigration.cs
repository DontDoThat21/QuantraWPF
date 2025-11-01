using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Quantra.Configuration
{
    /// <summary>
    /// Handles migration of configuration data
    /// </summary>
    public static class ConfigurationMigration
    {
        /// <summary>
        /// Migrate settings from legacy sources to new configuration system
        /// </summary>
        /// <param name="configManager">The configuration manager</param>
        /// <returns>Task for the migration operation</returns>
        /// todo REMOVE ME
        public static async Task MigrateFromLegacySources(IConfigurationManager configManager)
        {
            // Log start of migration
            //DatabaseMonolith.Log("Info", "Starting configuration migration from legacy sources");
            
            try
            {
                // Check if user settings file already exists - if so, migration already done
                var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                var userSettingsPath = Path.Combine(appDataPath, "Quantra", "usersettings.json");
                
                if (File.Exists(userSettingsPath))
                {
                    // Already migrated
                    //DatabaseMonolith.Log("Info", "Configuration already migrated, skipping");
                    return;
                }
                
                // Get settings from database
                var settings = DatabaseMonolith.GetUserSettings();
                
                // Create root object for JSON
                var root = new JObject();
                
                // ---- API Settings ----
                var api = new JObject();
                var alphaVantage = new JObject();
                alphaVantage["EnableApiModalChecks"] = settings.EnableApiModalChecks;
                alphaVantage["DefaultTimeout"] = settings.ApiTimeoutSeconds;
                api["AlphaVantage"] = alphaVantage;
                root["Api"] = api;
                
                // ---- Cache Settings ----
                var cache = new JObject();
                cache["EnableHistoricalDataCache"] = settings.EnableHistoricalDataCache;
                cache["CacheDurationMinutes"] = settings.CacheDurationMinutes;
                root["Cache"] = cache;
                
                // ---- UI Settings ----
                var ui = new JObject();
                ui["EnableDarkMode"] = settings.EnableDarkMode;
                ui["ChartUpdateIntervalSeconds"] = settings.ChartUpdateIntervalSeconds;
                ui["DefaultGridRows"] = settings.DefaultGridRows;
                ui["DefaultGridColumns"] = settings.DefaultGridColumns;
                ui["GridBorderColor"] = settings.GridBorderColor;
                root["UI"] = ui;
                
                // ---- Notification Settings ----
                var notifications = new JObject();
                notifications["EnablePriceAlerts"] = settings.EnablePriceAlerts;
                notifications["EnableTradeNotifications"] = settings.EnableTradeNotifications;
                
                // Email settings
                var email = new JObject();
                email["DefaultRecipient"] = settings.AlertEmail;
                email["EnableEmailAlerts"] = settings.EnableEmailAlerts;
                email["EnableStandardAlertEmails"] = settings.EnableStandardAlertEmails;
                email["EnableOpportunityAlertEmails"] = settings.EnableOpportunityAlertEmails;
                email["EnablePredictionAlertEmails"] = settings.EnablePredictionAlertEmails;
                email["EnableGlobalAlertEmails"] = settings.EnableGlobalAlertEmails;
                email["EnableSystemHealthAlertEmails"] = settings.EnableSystemHealthAlertEmails;
                notifications["Email"] = email;
                
                // SMS settings
                var sms = new JObject();
                sms["DefaultRecipient"] = settings.AlertPhoneNumber;
                sms["EnableSmsAlerts"] = settings.EnableSmsAlerts;
                sms["EnableStandardAlertSms"] = settings.EnableStandardAlertSms;
                sms["EnableOpportunityAlertSms"] = settings.EnableOpportunityAlertSms;
                sms["EnablePredictionAlertSms"] = settings.EnablePredictionAlertSms;
                sms["EnableGlobalAlertSms"] = settings.EnableGlobalAlertSms;
                notifications["SMS"] = sms;
                
                // Push notification settings
                var push = new JObject();
                push["UserId"] = settings.PushNotificationUserId;
                push["EnablePushNotifications"] = settings.EnablePushNotifications;
                push["EnableStandardAlertPushNotifications"] = settings.EnableStandardAlertPushNotifications;
                push["EnableOpportunityAlertPushNotifications"] = settings.EnableOpportunityAlertPushNotifications;
                push["EnablePredictionAlertPushNotifications"] = settings.EnablePredictionAlertPushNotifications;
                push["EnableGlobalAlertPushNotifications"] = settings.EnableGlobalAlertPushNotifications;
                push["EnableTechnicalIndicatorAlertPushNotifications"] = settings.EnableTechnicalIndicatorAlertPushNotifications;
                push["EnableSentimentShiftAlertPushNotifications"] = settings.EnableSentimentShiftAlertPushNotifications;
                push["EnableSystemHealthAlertPushNotifications"] = settings.EnableSystemHealthAlertPushNotifications;
                push["EnableTradeExecutionPushNotifications"] = settings.EnableTradeExecutionPushNotifications;
                notifications["Push"] = push;
                
                // Sound settings
                var sound = new JObject();
                sound["EnableAlertSounds"] = settings.EnableAlertSounds;
                sound["DefaultAlertSound"] = settings.DefaultAlertSound;
                sound["DefaultOpportunitySound"] = settings.DefaultOpportunitySound;
                sound["DefaultPredictionSound"] = settings.DefaultPredictionSound;
                sound["DefaultTechnicalIndicatorSound"] = settings.DefaultTechnicalIndicatorSound;
                sound["AlertVolume"] = settings.AlertVolume;
                notifications["Sound"] = sound;
                
                // Visual indicator settings
                var visual = new JObject();
                visual["EnableVisualIndicators"] = settings.EnableVisualIndicators;
                visual["DefaultVisualIndicatorType"] = settings.DefaultVisualIndicatorType;
                visual["DefaultVisualIndicatorColor"] = settings.DefaultVisualIndicatorColor;
                visual["VisualIndicatorDuration"] = settings.VisualIndicatorDuration;
                notifications["Visual"] = visual;
                
                root["Notifications"] = notifications;
                
                // ---- Trading Settings ----
                var trading = new JObject();
                trading["EnablePaperTrading"] = settings.EnablePaperTrading;
                trading["RiskLevel"] = settings.RiskLevel;
                trading["AccountSize"] = settings.AccountSize;
                trading["BaseRiskPercentage"] = settings.BaseRiskPercentage;
                trading["PositionSizingMethod"] = settings.PositionSizingMethod;
                trading["MaxPositionSizePercent"] = settings.MaxPositionSizePercent;
                trading["FixedTradeAmount"] = settings.FixedTradeAmount;
                trading["UseVolatilityBasedSizing"] = settings.UseVolatilityBasedSizing;
                trading["ATRMultiple"] = settings.ATRMultiple;
                trading["UseKellyCriterion"] = settings.UseKellyCriterion;
                trading["HistoricalWinRate"] = settings.HistoricalWinRate;
                trading["HistoricalRewardRiskRatio"] = settings.HistoricalRewardRiskRatio;
                trading["KellyFractionMultiplier"] = settings.KellyFractionMultiplier;
                root["Trading"] = trading;
                
                // ---- Sentiment Analysis Settings ----
                var sentimentAnalysis = new JObject();
                
                // News sentiment settings
                var news = new JObject();
                news["EnableNewsSentimentAnalysis"] = settings.EnableNewsSentimentAnalysis;
                news["NewsArticleRefreshIntervalMinutes"] = settings.NewsArticleRefreshIntervalMinutes;
                news["MaxNewsArticlesPerSymbol"] = settings.MaxNewsArticlesPerSymbol;
                news["EnableNewsSourceFiltering"] = settings.EnableNewsSourceFiltering;
                news["EnabledNewsSources"] = JToken.FromObject(settings.EnabledNewsSources);
                sentimentAnalysis["News"] = news;
                
                // Analyst ratings settings
                var analystRatings = new JObject();
                analystRatings["EnableAnalystRatings"] = settings.EnableAnalystRatings;
                analystRatings["RatingsCacheExpiryHours"] = settings.RatingsCacheExpiryHours;
                analystRatings["EnableRatingChangeAlerts"] = settings.EnableRatingChangeAlerts;
                analystRatings["EnableConsensusChangeAlerts"] = settings.EnableConsensusChangeAlerts;
                analystRatings["AnalystRatingSentimentWeight"] = settings.AnalystRatingSentimentWeight;
                sentimentAnalysis["AnalystRatings"] = analystRatings;
                
                // Insider trading settings
                var insiderTrading = new JObject();
                insiderTrading["EnableInsiderTradingAnalysis"] = settings.EnableInsiderTradingAnalysis;
                insiderTrading["InsiderDataRefreshIntervalMinutes"] = settings.InsiderDataRefreshIntervalMinutes;
                insiderTrading["EnableInsiderTradingAlerts"] = settings.EnableInsiderTradingAlerts;
                insiderTrading["TrackNotableInsiders"] = settings.TrackNotableInsiders;
                insiderTrading["InsiderTradingSentimentWeight"] = settings.InsiderTradingSentimentWeight;
                insiderTrading["HighlightCEOTransactions"] = settings.HighlightCEOTransactions;
                insiderTrading["HighlightOptionsActivity"] = settings.HighlightOptionsActivity;
                insiderTrading["EnableInsiderTransactionNotifications"] = settings.EnableInsiderTransactionNotifications;
                sentimentAnalysis["InsiderTrading"] = insiderTrading;
                
                root["SentimentAnalysis"] = sentimentAnalysis;
                
                // Create directory if it doesn't exist
                var userSettingsDir = Path.GetDirectoryName(userSettingsPath);
                if (!Directory.Exists(userSettingsDir))
                {
                    Directory.CreateDirectory(userSettingsDir!);
                }
                
                // Save to user settings file
                using (var writer = new StreamWriter(userSettingsPath))
                {
                    await writer.WriteAsync(root.ToString(Formatting.Indented));
                }
                
                // Reload the configuration
                await configManager.ReloadAsync();
                
                // Log successful migration
                //DatabaseMonolith.Log("Info", "Successfully migrated configuration from legacy sources");
            }
            catch (Exception ex)
            {
                // Log error
                //DatabaseMonolith.Log("Error", "Error migrating configuration from legacy sources", ex.ToString());
            }
        }
    }
}