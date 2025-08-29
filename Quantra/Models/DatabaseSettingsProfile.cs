using System;
using Quantra;

namespace Quantra.Models
{
    public class DatabaseSettingsProfile
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public bool IsDefault { get; set; }
        public DateTime CreatedDate { get; set; }
        public DateTime ModifiedDate { get; set; }
        
        // Settings properties
        public bool EnableApiModalChecks { get; set; }
        public int ApiTimeoutSeconds { get; set; }
        public int CacheDurationMinutes { get; set; }
        public bool EnableHistoricalDataCache { get; set; }
        public bool EnableDarkMode { get; set; }
        public int ChartUpdateIntervalSeconds { get; set; }
        public int DefaultGridRows { get; set; }
        public int DefaultGridColumns { get; set; }
        public string GridBorderColor { get; set; }
        public bool EnablePriceAlerts { get; set; }
        public bool EnableTradeNotifications { get; set; }
        public bool EnablePaperTrading { get; set; }
        public string RiskLevel { get; set; }
        
        // Email settings
        public string AlertEmail { get; set; } = "tylortrub@gmail.com";
        public bool EnableEmailAlerts { get; set; } = false;
        public bool EnableStandardAlertEmails { get; set; } = false;
        public bool EnableOpportunityAlertEmails { get; set; } = false;
        public bool EnablePredictionAlertEmails { get; set; } = false;
        public bool EnableGlobalAlertEmails { get; set; } = false;
        public bool EnableSystemHealthAlertEmails { get; set; } = false;
        
        // SMS settings
        public string AlertPhoneNumber { get; set; } = "";
        public bool EnableSmsAlerts { get; set; } = false;
        public bool EnableStandardAlertSms { get; set; } = false;
        public bool EnableOpportunityAlertSms { get; set; } = false;
        public bool EnablePredictionAlertSms { get; set; } = false;
        public bool EnableGlobalAlertSms { get; set; } = false;
        
        // Push notification settings
        public string PushNotificationUserId { get; set; } = "";
        public bool EnablePushNotifications { get; set; } = false;
        public bool EnableStandardAlertPushNotifications { get; set; } = false;
        public bool EnableOpportunityAlertPushNotifications { get; set; } = false;
        public bool EnablePredictionAlertPushNotifications { get; set; } = false;
        public bool EnableGlobalAlertPushNotifications { get; set; } = false;
        public bool EnableTechnicalIndicatorAlertPushNotifications { get; set; } = false;
        public bool EnableSentimentShiftAlertPushNotifications { get; set; } = false;
        public bool EnableSystemHealthAlertPushNotifications { get; set; } = false;
        public bool EnableTradeExecutionPushNotifications { get; set; } = false;
        
        // Window state settings
        public bool RememberWindowState { get; set; } = true;
        public int LastWindowState { get; set; } = 0; // 0:Normal, 1:Minimized, 2:Maximized
        
        // VIX monitoring settings
        public bool EnableVixMonitoring { get; set; } = true;
        
        // Create a new profile from a UserSettings object
        public static DatabaseSettingsProfile FromUserSettings(UserSettings settings, string name, string description, bool isDefault)
        {
            return new DatabaseSettingsProfile
            {
                Name = name,
                Description = description,
                IsDefault = isDefault,
                CreatedDate = DateTime.Now,
                ModifiedDate = DateTime.Now,
                EnableApiModalChecks = settings.EnableApiModalChecks,
                ApiTimeoutSeconds = settings.ApiTimeoutSeconds,
                CacheDurationMinutes = settings.CacheDurationMinutes,
                EnableHistoricalDataCache = settings.EnableHistoricalDataCache,
                EnableDarkMode = settings.EnableDarkMode,
                ChartUpdateIntervalSeconds = settings.ChartUpdateIntervalSeconds,
                DefaultGridRows = settings.DefaultGridRows,
                DefaultGridColumns = settings.DefaultGridColumns,
                GridBorderColor = settings.GridBorderColor,
                EnablePriceAlerts = settings.EnablePriceAlerts,
                EnableTradeNotifications = settings.EnableTradeNotifications,
                EnablePaperTrading = settings.EnablePaperTrading,
                RiskLevel = settings.RiskLevel,
                AlertEmail = settings.AlertEmail,
                EnableEmailAlerts = settings.EnableEmailAlerts,
                EnableStandardAlertEmails = settings.EnableStandardAlertEmails,
                EnableOpportunityAlertEmails = settings.EnableOpportunityAlertEmails,
                EnablePredictionAlertEmails = settings.EnablePredictionAlertEmails,
                EnableGlobalAlertEmails = settings.EnableGlobalAlertEmails,
                EnableSystemHealthAlertEmails = settings.EnableSystemHealthAlertEmails,
                AlertPhoneNumber = settings.AlertPhoneNumber,
                EnableSmsAlerts = settings.EnableSmsAlerts,
                EnableStandardAlertSms = settings.EnableStandardAlertSms,
                EnableOpportunityAlertSms = settings.EnableOpportunityAlertSms,
                EnablePredictionAlertSms = settings.EnablePredictionAlertSms,
                EnableGlobalAlertSms = settings.EnableGlobalAlertSms,
                // Push notification settings
                PushNotificationUserId = settings.PushNotificationUserId,
                EnablePushNotifications = settings.EnablePushNotifications,
                EnableStandardAlertPushNotifications = settings.EnableStandardAlertPushNotifications,
                EnableOpportunityAlertPushNotifications = settings.EnableOpportunityAlertPushNotifications,
                EnablePredictionAlertPushNotifications = settings.EnablePredictionAlertPushNotifications,
                EnableGlobalAlertPushNotifications = settings.EnableGlobalAlertPushNotifications,
                EnableTechnicalIndicatorAlertPushNotifications = settings.EnableTechnicalIndicatorAlertPushNotifications,
                EnableSentimentShiftAlertPushNotifications = settings.EnableSentimentShiftAlertPushNotifications,
                EnableSystemHealthAlertPushNotifications = settings.EnableSystemHealthAlertPushNotifications,
                EnableTradeExecutionPushNotifications = settings.EnableTradeExecutionPushNotifications,
                // Window state settings
                RememberWindowState = settings.RememberWindowState,
                LastWindowState = settings.LastWindowState,
                // VIX monitoring settings
                EnableVixMonitoring = settings.EnableVixMonitoring
            };
        }
        
        // Convert this profile to UserSettings
        public UserSettings ToUserSettings()
        {
            return new UserSettings
            {
                EnableApiModalChecks = this.EnableApiModalChecks,
                ApiTimeoutSeconds = this.ApiTimeoutSeconds,
                CacheDurationMinutes = this.CacheDurationMinutes,
                EnableHistoricalDataCache = this.EnableHistoricalDataCache,
                EnableDarkMode = this.EnableDarkMode,
                ChartUpdateIntervalSeconds = this.ChartUpdateIntervalSeconds,
                DefaultGridRows = this.DefaultGridRows,
                DefaultGridColumns = this.DefaultGridColumns,
                GridBorderColor = this.GridBorderColor,
                EnablePriceAlerts = this.EnablePriceAlerts,
                EnableTradeNotifications = this.EnableTradeNotifications,
                EnablePaperTrading = this.EnablePaperTrading,
                RiskLevel = this.RiskLevel,
                AlertEmail = this.AlertEmail,
                EnableEmailAlerts = this.EnableEmailAlerts,
                EnableStandardAlertEmails = this.EnableStandardAlertEmails,
                EnableOpportunityAlertEmails = this.EnableOpportunityAlertEmails,
                EnablePredictionAlertEmails = this.EnablePredictionAlertEmails,
                EnableGlobalAlertEmails = this.EnableGlobalAlertEmails,
                EnableSystemHealthAlertEmails = this.EnableSystemHealthAlertEmails,
                AlertPhoneNumber = this.AlertPhoneNumber,
                EnableSmsAlerts = this.EnableSmsAlerts,
                EnableStandardAlertSms = this.EnableStandardAlertSms,
                EnableOpportunityAlertSms = this.EnableOpportunityAlertSms,
                EnablePredictionAlertSms = this.EnablePredictionAlertSms,
                EnableGlobalAlertSms = this.EnableGlobalAlertSms,
                // Push notification settings
                PushNotificationUserId = this.PushNotificationUserId,
                EnablePushNotifications = this.EnablePushNotifications,
                EnableStandardAlertPushNotifications = this.EnableStandardAlertPushNotifications,
                EnableOpportunityAlertPushNotifications = this.EnableOpportunityAlertPushNotifications,
                EnablePredictionAlertPushNotifications = this.EnablePredictionAlertPushNotifications,
                EnableGlobalAlertPushNotifications = this.EnableGlobalAlertPushNotifications,
                EnableTechnicalIndicatorAlertPushNotifications = this.EnableTechnicalIndicatorAlertPushNotifications,
                EnableSentimentShiftAlertPushNotifications = this.EnableSentimentShiftAlertPushNotifications,
                EnableSystemHealthAlertPushNotifications = this.EnableSystemHealthAlertPushNotifications,
                EnableTradeExecutionPushNotifications = this.EnableTradeExecutionPushNotifications,
                // Window state settings
                RememberWindowState = this.RememberWindowState,
                LastWindowState = this.LastWindowState,
                // VIX monitoring settings
                EnableVixMonitoring = this.EnableVixMonitoring,
            };
        }
        
        // Clone the profile
        public DatabaseSettingsProfile Clone(string newName = null)
        {
            return new DatabaseSettingsProfile
            {
                Name = newName ?? $"{this.Name} (Copy)",
                Description = this.Description,
                IsDefault = false, // Clone is never default
                CreatedDate = DateTime.Now,
                ModifiedDate = DateTime.Now,
                EnableApiModalChecks = this.EnableApiModalChecks,
                ApiTimeoutSeconds = this.ApiTimeoutSeconds,
                CacheDurationMinutes = this.CacheDurationMinutes,
                EnableHistoricalDataCache = this.EnableHistoricalDataCache,
                EnableDarkMode = this.EnableDarkMode,
                ChartUpdateIntervalSeconds = this.ChartUpdateIntervalSeconds,
                DefaultGridRows = this.DefaultGridRows,
                DefaultGridColumns = this.DefaultGridColumns,
                GridBorderColor = this.GridBorderColor,
                EnablePriceAlerts = this.EnablePriceAlerts,
                EnableTradeNotifications = this.EnableTradeNotifications,
                EnablePaperTrading = this.EnablePaperTrading,
                RiskLevel = this.RiskLevel,
                AlertEmail = this.AlertEmail,
                EnableEmailAlerts = this.EnableEmailAlerts,
                EnableStandardAlertEmails = this.EnableStandardAlertEmails,
                EnableOpportunityAlertEmails = this.EnableOpportunityAlertEmails,
                EnablePredictionAlertEmails = this.EnablePredictionAlertEmails,
                EnableGlobalAlertEmails = this.EnableGlobalAlertEmails,
                EnableSystemHealthAlertEmails = this.EnableSystemHealthAlertEmails,
                AlertPhoneNumber = this.AlertPhoneNumber,
                EnableSmsAlerts = this.EnableSmsAlerts,
                EnableStandardAlertSms = this.EnableStandardAlertSms,
                EnableOpportunityAlertSms = this.EnableOpportunityAlertSms,
                EnablePredictionAlertSms = this.EnablePredictionAlertSms,
                EnableGlobalAlertSms = this.EnableGlobalAlertSms,
                // Push notification settings
                PushNotificationUserId = this.PushNotificationUserId,
                EnablePushNotifications = this.EnablePushNotifications,
                EnableStandardAlertPushNotifications = this.EnableStandardAlertPushNotifications,
                EnableOpportunityAlertPushNotifications = this.EnableOpportunityAlertPushNotifications,
                EnablePredictionAlertPushNotifications = this.EnablePredictionAlertPushNotifications,
                EnableGlobalAlertPushNotifications = this.EnableGlobalAlertPushNotifications,
                EnableTechnicalIndicatorAlertPushNotifications = this.EnableTechnicalIndicatorAlertPushNotifications,
                EnableSentimentShiftAlertPushNotifications = this.EnableSentimentShiftAlertPushNotifications,
                EnableSystemHealthAlertPushNotifications = this.EnableSystemHealthAlertPushNotifications,
                EnableTradeExecutionPushNotifications = this.EnableTradeExecutionPushNotifications,
                // Window state settings
                RememberWindowState = this.RememberWindowState,
                LastWindowState = this.LastWindowState,
                // VIX monitoring settings
                EnableVixMonitoring = this.EnableVixMonitoring
            };
        }

        // Create a profile with default settings
        public static DatabaseSettingsProfile CreateDefault(string name, string description = null, bool isDefault = false)
        {
            return new DatabaseSettingsProfile
            {
                Name = name,
                Description = description ?? "Default system settings",
                IsDefault = isDefault,
                CreatedDate = DateTime.Now,
                ModifiedDate = DateTime.Now,
                // Default settings values
                EnableApiModalChecks = true,
                ApiTimeoutSeconds = 30,
                CacheDurationMinutes = 15,
                EnableHistoricalDataCache = true,
                EnableDarkMode = true,
                ChartUpdateIntervalSeconds = 2,
                DefaultGridRows = 4,
                DefaultGridColumns = 4,
                GridBorderColor = "#FF00FFFF", // Cyan
                EnablePriceAlerts = true,
                EnableTradeNotifications = true,
                EnablePaperTrading = true,
                RiskLevel = "Low",
                AlertEmail = "tylortrub@gmail.com",
                EnableEmailAlerts = false,
                EnableStandardAlertEmails = false,
                EnableOpportunityAlertEmails = false,
                EnablePredictionAlertEmails = false,
                EnableGlobalAlertEmails = false,
                EnableSystemHealthAlertEmails = false,
                AlertPhoneNumber = "",
                EnableSmsAlerts = false,
                EnableStandardAlertSms = false,
                EnableOpportunityAlertSms = false,
                EnablePredictionAlertSms = false,
                EnableGlobalAlertSms = false,
                // Push notification settings
                PushNotificationUserId = "",
                EnablePushNotifications = false,
                EnableStandardAlertPushNotifications = false,
                EnableOpportunityAlertPushNotifications = false,
                EnablePredictionAlertPushNotifications = false,
                EnableGlobalAlertPushNotifications = false,
                EnableTechnicalIndicatorAlertPushNotifications = false,
                EnableSentimentShiftAlertPushNotifications = false,
                EnableSystemHealthAlertPushNotifications = false,
                EnableTradeExecutionPushNotifications = false,
                // Window state settings
                RememberWindowState = true,
                LastWindowState = 0, // Default to Normal window state
                // VIX monitoring settings
                EnableVixMonitoring = true
            };
        }
    }
}
