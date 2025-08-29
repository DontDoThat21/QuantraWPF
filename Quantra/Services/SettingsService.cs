using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Threading.Tasks;
using Quantra.Models;
using Quantra.CrossCutting.ErrorHandling;
using Quantra.Services.Interfaces;

namespace Quantra.Services
{
    public class SettingsService : ISettingsService
    {
        // Ensure the settings profiles table exists
        public void EnsureSettingsProfilesTable()
        {
            ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    var createTableQuery = @"
                        CREATE TABLE IF NOT EXISTS SettingsProfiles (
                        Id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Name TEXT NOT NULL UNIQUE,
                        Description TEXT,
                        IsDefault INTEGER DEFAULT 0,
                        CreatedDate DATETIME NOT NULL,
                        ModifiedDate DATETIME NOT NULL,
                        EnableApiModalChecks INTEGER,
                        ApiTimeoutSeconds INTEGER,
                        CacheDurationMinutes INTEGER,
                        EnableHistoricalDataCache INTEGER,
                        EnableDarkMode INTEGER,
                        ChartUpdateIntervalSeconds INTEGER,
                        DefaultGridRows INTEGER,
                        DefaultGridColumns INTEGER,
                        GridBorderColor TEXT,
                        EnablePriceAlerts INTEGER,
                        EnableTradeNotifications INTEGER,
                        EnablePaperTrading INTEGER,
                        RiskLevel TEXT,
                        AlertEmail TEXT,
                        EnableEmailAlerts INTEGER DEFAULT 0,
                        EnableStandardAlertEmails INTEGER DEFAULT 0,
                        EnableOpportunityAlertEmails INTEGER DEFAULT 0,
                        EnablePredictionAlertEmails INTEGER DEFAULT 0,
                        EnableGlobalAlertEmails INTEGER DEFAULT 0,
                        EnableSystemHealthAlertEmails INTEGER DEFAULT 0,
                        AlertPhoneNumber TEXT DEFAULT '',
                        EnableSmsAlerts INTEGER DEFAULT 0,
                        EnableStandardAlertSms INTEGER DEFAULT 0,
                        EnableOpportunityAlertSms INTEGER DEFAULT 0,
                        EnablePredictionAlertSms INTEGER DEFAULT 0,
                        EnableGlobalAlertSms INTEGER DEFAULT 0,
                        PushNotificationUserId TEXT DEFAULT '',
                        EnablePushNotifications INTEGER DEFAULT 0,
                        EnableStandardAlertPushNotifications INTEGER DEFAULT 0,
                        EnableOpportunityAlertPushNotifications INTEGER DEFAULT 0,
                        EnablePredictionAlertPushNotifications INTEGER DEFAULT 0,
                        EnableGlobalAlertPushNotifications INTEGER DEFAULT 0,
                        EnableTechnicalIndicatorAlertPushNotifications INTEGER DEFAULT 0,
                        EnableSentimentShiftAlertPushNotifications INTEGER DEFAULT 0,
                        EnableSystemHealthAlertPushNotifications INTEGER DEFAULT 0,
                        EnableTradeExecutionPushNotifications INTEGER DEFAULT 0,
                        AlphaVantageApiKey TEXT DEFAULT ''
                    )";
                
                    using (var command = new SQLiteCommand(createTableQuery, connection))
                    {
                        command.ExecuteNonQuery();
                    }
                
                    // Check and add missing columns if needed
                    string[] emailColumns = new[] {
                        "EnableEmailAlerts",
                        "EnableStandardAlertEmails",
                        "EnableOpportunityAlertEmails",
                        "EnablePredictionAlertEmails",
                        "EnableGlobalAlertEmails",
                        "EnableSystemHealthAlertEmails"
                    };
                    string[] smsColumns = new[] {
                        "AlertPhoneNumber",
                        "EnableSmsAlerts",
                        "EnableStandardAlertSms",
                        "EnableOpportunityAlertSms",
                        "EnablePredictionAlertSms",
                        "EnableGlobalAlertSms"
                    };
                    string[] pushColumns = new[] {
                        "PushNotificationUserId",
                        "EnablePushNotifications",
                        "EnableStandardAlertPushNotifications",
                        "EnableOpportunityAlertPushNotifications",
                        "EnablePredictionAlertPushNotifications",
                        "EnableGlobalAlertPushNotifications",
                        "EnableTechnicalIndicatorAlertPushNotifications",
                        "EnableSentimentShiftAlertPushNotifications",
                        "EnableSystemHealthAlertPushNotifications",
                        "EnableTradeExecutionPushNotifications"
                    };
                    using (var command = new SQLiteCommand("PRAGMA table_info(SettingsProfiles)", connection))
                    using (var reader = command.ExecuteReader())
                    {
                        var existingColumns = new HashSet<string>();
                        while (reader.Read())
                        {
                            existingColumns.Add(reader["name"].ToString());
                        }
                        foreach (var col in emailColumns)
                        {
                            if (!existingColumns.Contains(col))
                            {
                                using (var addCmd = new SQLiteCommand($"ALTER TABLE SettingsProfiles ADD COLUMN {col} INTEGER DEFAULT 0", connection))
                                {
                                    addCmd.ExecuteNonQuery();
                                    DatabaseMonolith.Log("Info", $"Added {col} column to SettingsProfiles table");
                                }
                            }
                        }
                    
                        // Check and add SMS columns
                        foreach (var col in smsColumns)
                        {
                            if (!existingColumns.Contains(col))
                            {
                                string defaultValue = col == "AlertPhoneNumber" ? "DEFAULT ''" : "DEFAULT 0";
                                using (var addCmd = new SQLiteCommand($"ALTER TABLE SettingsProfiles ADD COLUMN {col} {(col == "AlertPhoneNumber" ? "TEXT" : "INTEGER")} {defaultValue}", connection))
                                {
                                    addCmd.ExecuteNonQuery();
                                    DatabaseMonolith.Log("Info", $"Added {col} column to SettingsProfiles table");
                                }
                            }
                        }
                    
                        // Check and add push notification columns
                        foreach (var col in pushColumns)
                        {
                            if (!existingColumns.Contains(col))
                            {
                                string defaultValue = col == "PushNotificationUserId" ? "DEFAULT ''" : "DEFAULT 0";
                                using (var addCmd = new SQLiteCommand($"ALTER TABLE SettingsProfiles ADD COLUMN {col} {(col == "PushNotificationUserId" ? "TEXT" : "INTEGER")} {defaultValue}", connection))
                                {
                                    addCmd.ExecuteNonQuery();
                                    DatabaseMonolith.Log("Info", $"Added {col} column to SettingsProfiles table");
                                }
                            }
                        }
                    }
                
                    // Check if the AlertEmail column exists in the table
                    bool hasAlertEmail = false;
                    using (var command = new SQLiteCommand("PRAGMA table_info(SettingsProfiles)", connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                if (reader["name"].ToString() == "AlertEmail")
                                {
                                    hasAlertEmail = true;
                                    break;
                                }
                            }
                        }
                    }
                
                    // If the AlertEmail column doesn't exist, add it
                    if (!hasAlertEmail)
                    {
                        using (var command = new SQLiteCommand("ALTER TABLE SettingsProfiles ADD COLUMN AlertEmail TEXT DEFAULT 'tylortrub@gmail.com'", connection))
                        {
                            command.ExecuteNonQuery();
                            DatabaseMonolith.Log("Info", "Added AlertEmail column to SettingsProfiles table");
                        }
                    }

                    // Check if the EnableVixMonitoring column exists in the table
                    bool hasEnableVixMonitoring = false;
                    using (var command = new SQLiteCommand("PRAGMA table_info(SettingsProfiles)", connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                if (reader["name"].ToString() == "EnableVixMonitoring")
                                {
                                    hasEnableVixMonitoring = true;
                                    break;
                                }
                            }
                        }
                    }

                    // If the EnableVixMonitoring column doesn't exist, add it
                    if (!hasEnableVixMonitoring)
                    {
                        using (var command = new SQLiteCommand("ALTER TABLE SettingsProfiles ADD COLUMN EnableVixMonitoring INTEGER DEFAULT 1", connection))
                        {
                            command.ExecuteNonQuery();
                            DatabaseMonolith.Log("Info", "Added EnableVixMonitoring column to SettingsProfiles table");
                        }
                    }

                    // Check if the AlphaVantageApiKey column exists in the table
                    bool hasAlphaVantageApiKey = false;
                    using (var command = new SQLiteCommand("PRAGMA table_info(SettingsProfiles)", connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                if (reader["name"].ToString() == "AlphaVantageApiKey")
                                {
                                    hasAlphaVantageApiKey = true;
                                    break;
                                }
                            }
                        }
                    }

                    // If the AlphaVantageApiKey column doesn't exist, add it
                    if (!hasAlphaVantageApiKey)
                    {
                        using (var command = new SQLiteCommand("ALTER TABLE SettingsProfiles ADD COLUMN AlphaVantageApiKey TEXT DEFAULT ''", connection))
                        {
                            command.ExecuteNonQuery();
                            DatabaseMonolith.Log("Info", "Added AlphaVantageApiKey column to SettingsProfiles table");
                        }
                    }
                
                    // Check if any profiles exist, if not create a default one
                    using (var command = new SQLiteCommand("SELECT COUNT(*) FROM SettingsProfiles", connection))
                    {
                        var count = Convert.ToInt32(command.ExecuteScalar());
                        if (count == 0)
                        {
                            // Create a default profile with hardcoded values to avoid any potential recursion
                            var defaultProfile = new DatabaseSettingsProfile
                            {
                                Name = "Default",
                                Description = "Default system settings",
                                IsDefault = true,
                                CreatedDate = DateTime.Now,
                                ModifiedDate = DateTime.Now,
                                EnableApiModalChecks = true,
                                ApiTimeoutSeconds = 30,
                                CacheDurationMinutes = 15,
                                EnableHistoricalDataCache = true,
                                EnableDarkMode = true,
                                ChartUpdateIntervalSeconds = 2,
                                DefaultGridRows = 4,
                                DefaultGridColumns = 4,
                                GridBorderColor = "#FF00FFFF",
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
                                EnableSystemHealthAlertEmails = false
                            };
                            CreateSettingsProfile(defaultProfile);
                        
                            DatabaseMonolith.Log("Info", "Created default settings profile");
                        }
                    }
                }
            }, RetryOptions.ForCriticalOperation());
        }
        
        // Create a new settings profile
        public int CreateSettingsProfile(DatabaseSettingsProfile profile)
        {
            return ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                
                    // If this is set as default, clear other defaults first
                    if (profile.IsDefault)
                    {
                        using (var clearDefaultCmd = new SQLiteCommand("UPDATE SettingsProfiles SET IsDefault = 0", connection))
                        {
                            clearDefaultCmd.ExecuteNonQuery();
                        }
                    }
                
                    // Insert the new profile
                    var insertQuery = @"
                        INSERT INTO SettingsProfiles (
                            Name, Description, IsDefault, CreatedDate, ModifiedDate,
                            EnableApiModalChecks, ApiTimeoutSeconds, CacheDurationMinutes,
                            EnableHistoricalDataCache, EnableDarkMode, ChartUpdateIntervalSeconds,
                            DefaultGridRows, DefaultGridColumns, GridBorderColor,
                            EnablePriceAlerts, EnableTradeNotifications, EnablePaperTrading, RiskLevel,
                            AlertEmail, EnableEmailAlerts, EnableStandardAlertEmails, EnableOpportunityAlertEmails, EnablePredictionAlertEmails, EnableGlobalAlertEmails, EnableSystemHealthAlertEmails,
                            AlertPhoneNumber, EnableSmsAlerts, EnableStandardAlertSms, EnableOpportunityAlertSms, EnablePredictionAlertSms, EnableGlobalAlertSms,
                            PushNotificationUserId, EnablePushNotifications, EnableStandardAlertPushNotifications, EnableOpportunityAlertPushNotifications, 
                            EnablePredictionAlertPushNotifications, EnableGlobalAlertPushNotifications, EnableTechnicalIndicatorAlertPushNotifications, 
                            EnableSentimentShiftAlertPushNotifications, EnableSystemHealthAlertPushNotifications, EnableTradeExecutionPushNotifications,
                            EnableVixMonitoring, AlphaVantageApiKey
                        ) VALUES (
                            @Name, @Description, @IsDefault, @CreatedDate, @ModifiedDate,
                            @EnableApiModalChecks, @ApiTimeoutSeconds, @CacheDurationMinutes,
                            @EnableHistoricalDataCache, @EnableDarkMode, @ChartUpdateIntervalSeconds,
                            @DefaultGridRows, @DefaultGridColumns, @GridBorderColor,
                            @EnablePriceAlerts, @EnableTradeNotifications, @EnablePaperTrading, @RiskLevel,
                            @AlertEmail, @EnableEmailAlerts, @EnableStandardAlertEmails, @EnableOpportunityAlertEmails, @EnablePredictionAlertEmails, @EnableGlobalAlertEmails, @EnableSystemHealthAlertEmails,
                            @AlertPhoneNumber, @EnableSmsAlerts, @EnableStandardAlertSms, @EnableOpportunityAlertSms, @EnablePredictionAlertSms, @EnableGlobalAlertSms,
                            @PushNotificationUserId, @EnablePushNotifications, @EnableStandardAlertPushNotifications, @EnableOpportunityAlertPushNotifications,
                            @EnablePredictionAlertPushNotifications, @EnableGlobalAlertPushNotifications, @EnableTechnicalIndicatorAlertPushNotifications,
                            @EnableSentimentShiftAlertPushNotifications, @EnableSystemHealthAlertPushNotifications, @EnableTradeExecutionPushNotifications,
                            @EnableVixMonitoring, @AlphaVantageApiKey
                        )";
                
                    using (var command = new SQLiteCommand(insertQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Name", profile.Name);
                        command.Parameters.AddWithValue("@Description", profile.Description ?? "");
                        command.Parameters.AddWithValue("@IsDefault", profile.IsDefault ? 1 : 0);
                        command.Parameters.AddWithValue("@CreatedDate", profile.CreatedDate);
                        command.Parameters.AddWithValue("@ModifiedDate", profile.ModifiedDate);
                        command.Parameters.AddWithValue("@EnableApiModalChecks", profile.EnableApiModalChecks ? 1 : 0);
                        command.Parameters.AddWithValue("@ApiTimeoutSeconds", profile.ApiTimeoutSeconds);
                        command.Parameters.AddWithValue("@CacheDurationMinutes", profile.CacheDurationMinutes);
                        command.Parameters.AddWithValue("@EnableHistoricalDataCache", profile.EnableHistoricalDataCache ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableDarkMode", profile.EnableDarkMode ? 1 : 0);
                        command.Parameters.AddWithValue("@ChartUpdateIntervalSeconds", profile.ChartUpdateIntervalSeconds);
                        command.Parameters.AddWithValue("@DefaultGridRows", profile.DefaultGridRows);
                        command.Parameters.AddWithValue("@DefaultGridColumns", profile.DefaultGridColumns);
                        command.Parameters.AddWithValue("@GridBorderColor", profile.GridBorderColor);
                        command.Parameters.AddWithValue("@EnablePriceAlerts", profile.EnablePriceAlerts ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableTradeNotifications", profile.EnableTradeNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePaperTrading", profile.EnablePaperTrading ? 1 : 0);
                        command.Parameters.AddWithValue("@RiskLevel", profile.RiskLevel);
                        command.Parameters.AddWithValue("@AlertEmail", profile.AlertEmail ?? "tylortrub@gmail.com");
                        command.Parameters.AddWithValue("@EnableEmailAlerts", profile.EnableEmailAlerts ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableStandardAlertEmails", profile.EnableStandardAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableOpportunityAlertEmails", profile.EnableOpportunityAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePredictionAlertEmails", profile.EnablePredictionAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableGlobalAlertEmails", profile.EnableGlobalAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableSystemHealthAlertEmails", profile.EnableSystemHealthAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@AlertPhoneNumber", profile.AlertPhoneNumber ?? "");
                        command.Parameters.AddWithValue("@EnableSmsAlerts", profile.EnableSmsAlerts ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableStandardAlertSms", profile.EnableStandardAlertSms ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableOpportunityAlertSms", profile.EnableOpportunityAlertSms ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePredictionAlertSms", profile.EnablePredictionAlertSms ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableGlobalAlertSms", profile.EnableGlobalAlertSms ? 1 : 0);
                    
                        // Push notification parameters
                        command.Parameters.AddWithValue("@PushNotificationUserId", profile.PushNotificationUserId ?? "");
                        command.Parameters.AddWithValue("@EnablePushNotifications", profile.EnablePushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableStandardAlertPushNotifications", profile.EnableStandardAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableOpportunityAlertPushNotifications", profile.EnableOpportunityAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePredictionAlertPushNotifications", profile.EnablePredictionAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableGlobalAlertPushNotifications", profile.EnableGlobalAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableTechnicalIndicatorAlertPushNotifications", profile.EnableTechnicalIndicatorAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableSentimentShiftAlertPushNotifications", profile.EnableSentimentShiftAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableSystemHealthAlertPushNotifications", profile.EnableSystemHealthAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableTradeExecutionPushNotifications", profile.EnableTradeExecutionPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableVixMonitoring", profile.EnableVixMonitoring ? 1 : 0);
                        command.Parameters.AddWithValue("@AlphaVantageApiKey", profile.AlphaVantageApiKey ?? "");
                    }
                
                    // Get the ID of the inserted profile
                    using (var idCommand = new SQLiteCommand("SELECT last_insert_rowid()", connection))
                    {
                        return Convert.ToInt32(idCommand.ExecuteScalar());
                    }
                }
            }, RetryOptions.ForCriticalOperation());
        }
        
        // Update an existing settings profile
        public bool UpdateSettingsProfile(DatabaseSettingsProfile profile)
        {
            return ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                
                    // If this is set as default, clear other defaults first
                    if (profile.IsDefault)
                    {
                        using (var clearDefaultCmd = new SQLiteCommand("UPDATE SettingsProfiles SET IsDefault = 0", connection))
                        {
                            clearDefaultCmd.ExecuteNonQuery();
                        }
                    }
                
                    // Update the profile
                    var updateQuery = @"
                        UPDATE SettingsProfiles SET
                            Name = @Name,
                            Description = @Description,
                            IsDefault = @IsDefault,
                            ModifiedDate = @ModifiedDate,
                            EnableApiModalChecks = @EnableApiModalChecks,
                            ApiTimeoutSeconds = @ApiTimeoutSeconds,
                            CacheDurationMinutes = @CacheDurationMinutes,
                            EnableHistoricalDataCache = @EnableHistoricalDataCache,
                            EnableDarkMode = @EnableDarkMode,
                            ChartUpdateIntervalSeconds = @ChartUpdateIntervalSeconds,
                            DefaultGridRows = @DefaultGridRows,
                            DefaultGridColumns = @DefaultGridColumns,
                            GridBorderColor = @GridBorderColor,
                            EnablePriceAlerts = @EnablePriceAlerts,
                            EnableTradeNotifications = @EnableTradeNotifications,
                            EnablePaperTrading = @EnablePaperTrading,
                            RiskLevel = @RiskLevel,
                            AlertEmail = @AlertEmail,
                            EnableEmailAlerts = @EnableEmailAlerts,
                            EnableStandardAlertEmails = @EnableStandardAlertEmails,
                            EnableOpportunityAlertEmails = @EnableOpportunityAlertEmails,
                            EnablePredictionAlertEmails = @EnablePredictionAlertEmails,
                            EnableGlobalAlertEmails = @EnableGlobalAlertEmails,
                            EnableSystemHealthAlertEmails = @EnableSystemHealthAlertEmails,
                            AlertPhoneNumber = @AlertPhoneNumber,
                            EnableSmsAlerts = @EnableSmsAlerts,
                            EnableStandardAlertSms = @EnableStandardAlertSms,
                            EnableOpportunityAlertSms = @EnableOpportunityAlertSms,
                            EnablePredictionAlertSms = @EnablePredictionAlertSms,
                            EnableGlobalAlertSms = @EnableGlobalAlertSms,
                            PushNotificationUserId = @PushNotificationUserId,
                            EnablePushNotifications = @EnablePushNotifications,
                            EnableStandardAlertPushNotifications = @EnableStandardAlertPushNotifications,
                            EnableOpportunityAlertPushNotifications = @EnableOpportunityAlertPushNotifications,
                            EnablePredictionAlertPushNotifications = @EnablePredictionAlertPushNotifications,
                            EnableGlobalAlertPushNotifications = @EnableGlobalAlertPushNotifications,
                            EnableTechnicalIndicatorAlertPushNotifications = @EnableTechnicalIndicatorAlertPushNotifications,
                            EnableSentimentShiftAlertPushNotifications = @EnableSentimentShiftAlertPushNotifications,
                            EnableSystemHealthAlertPushNotifications = @EnableSystemHealthAlertPushNotifications,
                            EnableTradeExecutionPushNotifications = @EnableTradeExecutionPushNotifications,
                            EnableVixMonitoring = @EnableVixMonitoring,
                            AlphaVantageApiKey = @AlphaVantageApiKey
                        WHERE Id = @Id";
                
                    using (var command = new SQLiteCommand(updateQuery, connection))
                    {
                        profile.ModifiedDate = DateTime.Now;
                    
                        command.Parameters.AddWithValue("@Id", profile.Id);
                        command.Parameters.AddWithValue("@Name", profile.Name);
                        command.Parameters.AddWithValue("@Description", profile.Description ?? "");
                        command.Parameters.AddWithValue("@IsDefault", profile.IsDefault ? 1 : 0);
                        command.Parameters.AddWithValue("@ModifiedDate", profile.ModifiedDate);
                        command.Parameters.AddWithValue("@EnableApiModalChecks", profile.EnableApiModalChecks ? 1 : 0);
                        command.Parameters.AddWithValue("@ApiTimeoutSeconds", profile.ApiTimeoutSeconds);
                        command.Parameters.AddWithValue("@CacheDurationMinutes", profile.CacheDurationMinutes);
                        command.Parameters.AddWithValue("@EnableHistoricalDataCache", profile.EnableHistoricalDataCache ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableDarkMode", profile.EnableDarkMode ? 1 : 0);
                        command.Parameters.AddWithValue("@ChartUpdateIntervalSeconds", profile.ChartUpdateIntervalSeconds);
                        command.Parameters.AddWithValue("@DefaultGridRows", profile.DefaultGridRows);
                        command.Parameters.AddWithValue("@DefaultGridColumns", profile.DefaultGridColumns);
                        command.Parameters.AddWithValue("@GridBorderColor", profile.GridBorderColor);
                        command.Parameters.AddWithValue("@EnablePriceAlerts", profile.EnablePriceAlerts ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableTradeNotifications", profile.EnableTradeNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePaperTrading", profile.EnablePaperTrading ? 1 : 0);
                        command.Parameters.AddWithValue("@RiskLevel", profile.RiskLevel);
                        command.Parameters.AddWithValue("@AlertEmail", profile.AlertEmail ?? "tylortrub@gmail.com");
                        command.Parameters.AddWithValue("@EnableEmailAlerts", profile.EnableEmailAlerts ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableStandardAlertEmails", profile.EnableStandardAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableOpportunityAlertEmails", profile.EnableOpportunityAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePredictionAlertEmails", profile.EnablePredictionAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableGlobalAlertEmails", profile.EnableGlobalAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableSystemHealthAlertEmails", profile.EnableSystemHealthAlertEmails ? 1 : 0);
                        command.Parameters.AddWithValue("@AlertPhoneNumber", profile.AlertPhoneNumber ?? "");
                        command.Parameters.AddWithValue("@EnableSmsAlerts", profile.EnableSmsAlerts ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableStandardAlertSms", profile.EnableStandardAlertSms ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableOpportunityAlertSms", profile.EnableOpportunityAlertSms ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePredictionAlertSms", profile.EnablePredictionAlertSms ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableGlobalAlertSms", profile.EnableGlobalAlertSms ? 1 : 0);
                    
                        // Push notification parameters
                        command.Parameters.AddWithValue("@PushNotificationUserId", profile.PushNotificationUserId ?? "");
                        command.Parameters.AddWithValue("@EnablePushNotifications", profile.EnablePushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableStandardAlertPushNotifications", profile.EnableStandardAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableOpportunityAlertPushNotifications", profile.EnableOpportunityAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnablePredictionAlertPushNotifications", profile.EnablePredictionAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableGlobalAlertPushNotifications", profile.EnableGlobalAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableTechnicalIndicatorAlertPushNotifications", profile.EnableTechnicalIndicatorAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableSentimentShiftAlertPushNotifications", profile.EnableSentimentShiftAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableSystemHealthAlertPushNotifications", profile.EnableSystemHealthAlertPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableTradeExecutionPushNotifications", profile.EnableTradeExecutionPushNotifications ? 1 : 0);
                        command.Parameters.AddWithValue("@EnableVixMonitoring", profile.EnableVixMonitoring ? 1 : 0);
                        command.Parameters.AddWithValue("@AlphaVantageApiKey", profile.AlphaVantageApiKey ?? "");
                    
                        int rowsAffected = command.ExecuteNonQuery();
                        return rowsAffected > 0;
                    }
                }
            }, RetryOptions.ForCriticalOperation());
        }
        
        // Delete a settings profile
        public bool DeleteSettingsProfile(int profileId)
        {
            return ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                
                    // Check if this is the default profile
                    bool isDefault = false;
                    using (var checkCommand = new SQLiteCommand("SELECT IsDefault FROM SettingsProfiles WHERE Id = @Id", connection))
                    {
                        checkCommand.Parameters.AddWithValue("@Id", profileId);
                        var result = checkCommand.ExecuteScalar();
                        if (result != null)
                            isDefault = Convert.ToBoolean(Convert.ToInt32(result));
                    }
                
                    // Don't allow deleting the default profile if it's the only one
                    if (isDefault)
                    {
                        using (var countCommand = new SQLiteCommand("SELECT COUNT(*) FROM SettingsProfiles", connection))
                        {
                            int count = Convert.ToInt32(countCommand.ExecuteScalar());
                            if (count <= 1)
                                return false; // Can't delete the only profile
                        }
                    }
                
                    // Delete the profile
                    using (var deleteCommand = new SQLiteCommand("DELETE FROM SettingsProfiles WHERE Id = @Id", connection))
                    {
                        deleteCommand.Parameters.AddWithValue("@Id", profileId);
                        int rowsAffected = deleteCommand.ExecuteNonQuery();
                    
                        // If we deleted the default profile, set another one as default
                        if (isDefault && rowsAffected > 0)
                        {
                            using (var newDefaultCommand = new SQLiteCommand(
                                "UPDATE SettingsProfiles SET IsDefault = 1 WHERE Id = (SELECT Id FROM SettingsProfiles LIMIT 1)", 
                                connection))
                            {
                                newDefaultCommand.ExecuteNonQuery();
                            }
                        }
                    
                        return rowsAffected > 0;
                    }
                }
            }, RetryOptions.ForCriticalOperation());
        }
        
        // Get a specific settings profile
        public DatabaseSettingsProfile GetSettingsProfile(int profileId)
        {
            return ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    var selectQuery = "SELECT * FROM SettingsProfiles WHERE Id = @Id";
                    using (var command = new SQLiteCommand(selectQuery, connection))
                    {
                        command.Parameters.AddWithValue("@Id", profileId);
                        using (var reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                return ReadProfileFromReader(reader);
                            }
                        }
                    }
                
                    return null;
                }
            }, RetryOptions.ForUserFacingOperation());
        }
        
        // Get the default settings profile
        public DatabaseSettingsProfile GetDefaultSettingsProfile()
        {
            return ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                
                    // First check if the table exists
                    using (var tableCheckCmd = new SQLiteCommand(
                        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='SettingsProfiles'", 
                        connection))
                    {
                        if (Convert.ToInt32(tableCheckCmd.ExecuteScalar()) == 0)
                        {
                            // Table doesn't exist, create it first
                            EnsureSettingsProfilesTable();
                        }
                    }
                
                    var selectQuery = "SELECT * FROM SettingsProfiles WHERE IsDefault = 1 LIMIT 1";
                    using (var command = new SQLiteCommand(selectQuery, connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                return ReadProfileFromReader(reader);
                            }
                        }
                    }
                
                    // If no default profile exists, get the first one
                    selectQuery = "SELECT * FROM SettingsProfiles LIMIT 1";
                    using (var command = new SQLiteCommand(selectQuery, connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            if (reader.Read())
                            {
                                return ReadProfileFromReader(reader);
                            }
                        }
                    }
                
                    // If still no profile, create a default one
                    EnsureSettingsProfiles();
                    
                    // Try one more time after ensuring profiles exist
                    selectQuery = "SELECT * FROM SettingsProfiles WHERE IsDefault = 1 LIMIT 1";
                    using (var command2 = new SQLiteCommand(selectQuery, connection))
                    {
                        using (var reader2 = command2.ExecuteReader())
                        {
                            if (reader2.Read())
                            {
                                return ReadProfileFromReader(reader2);
                            }
                        }
                    }
                    
                    // If still nothing, return null to avoid infinite recursion
                    return null;
                }
            }, RetryOptions.ForUserFacingOperation());
        }
        
        // Get the default settings profile asynchronously
        public Task<DatabaseSettingsProfile> GetDefaultSettingsProfileAsync()
        {
            return Task.FromResult(GetDefaultSettingsProfile());
        }
        
        // Get all settings profiles
        public List<DatabaseSettingsProfile> GetAllSettingsProfiles()
        {
            return ResilienceHelper.Retry(() =>
            {
                var profiles = new List<DatabaseSettingsProfile>();
                
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                
                    // First check if the table exists
                    using (var tableCheckCmd = new SQLiteCommand(
                        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='SettingsProfiles'", 
                        connection))
                    {
                        if (Convert.ToInt32(tableCheckCmd.ExecuteScalar()) == 0)
                        {
                            // Table doesn't exist, create it first
                            EnsureSettingsProfilesTable();
                        }
                    }
                
                    var selectQuery = "SELECT * FROM SettingsProfiles ORDER BY IsDefault DESC, Name ASC";
                    using (var command = new SQLiteCommand(selectQuery, connection))
                    {
                        using (var reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                profiles.Add(ReadProfileFromReader(reader));
                            }
                        }
                    }
                
                    return profiles;
                }
            }, RetryOptions.ForUserFacingOperation());
        }
        
        // Helper method to read a profile from a database reader
        private static DatabaseSettingsProfile ReadProfileFromReader(SQLiteDataReader reader)
        {
            // Defensive: check for column existence before accessing
            string[] columns = new string[reader.FieldCount];
            for (int i = 0; i < reader.FieldCount; i++)
                columns[i] = reader.GetName(i);
            bool Has(string col) => Array.IndexOf(columns, col) >= 0;

            return new DatabaseSettingsProfile
            {
                Id = Convert.ToInt32(reader["Id"]),
                Name = reader["Name"].ToString(),
                Description = Has("Description") ? reader["Description"]?.ToString() ?? "" : "",
                IsDefault = Has("IsDefault") ? Convert.ToBoolean(Convert.ToInt32(reader["IsDefault"])) : false,
                CreatedDate = Has("CreatedDate") ? Convert.ToDateTime(reader["CreatedDate"]) : DateTime.Now,
                ModifiedDate = Has("ModifiedDate") ? Convert.ToDateTime(reader["ModifiedDate"]) : DateTime.Now,
                EnableApiModalChecks = Has("EnableApiModalChecks") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableApiModalChecks"])) : false,
                ApiTimeoutSeconds = Has("ApiTimeoutSeconds") ? Convert.ToInt32(reader["ApiTimeoutSeconds"]) : 30,
                CacheDurationMinutes = Has("CacheDurationMinutes") ? Convert.ToInt32(reader["CacheDurationMinutes"]) : 15,
                EnableHistoricalDataCache = Has("EnableHistoricalDataCache") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableHistoricalDataCache"])) : false,
                EnableDarkMode = Has("EnableDarkMode") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableDarkMode"])) : false,
                ChartUpdateIntervalSeconds = Has("ChartUpdateIntervalSeconds") ? Convert.ToInt32(reader["ChartUpdateIntervalSeconds"]) : 2,
                DefaultGridRows = Has("DefaultGridRows") ? Convert.ToInt32(reader["DefaultGridRows"]) : 4,
                DefaultGridColumns = Has("DefaultGridColumns") ? Convert.ToInt32(reader["DefaultGridColumns"]) : 4,
                GridBorderColor = Has("GridBorderColor") ? reader["GridBorderColor"]?.ToString() ?? "#FF00FFFF" : "#FF00FFFF",
                EnablePriceAlerts = Has("EnablePriceAlerts") ? Convert.ToBoolean(Convert.ToInt32(reader["EnablePriceAlerts"])) : false,
                EnableTradeNotifications = Has("EnableTradeNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableTradeNotifications"])) : false,
                EnablePaperTrading = Has("EnablePaperTrading") ? Convert.ToBoolean(Convert.ToInt32(reader["EnablePaperTrading"])) : false,
                RiskLevel = Has("RiskLevel") ? reader["RiskLevel"]?.ToString() ?? "Low" : "Low",
                AlertEmail = Has("AlertEmail") ? reader["AlertEmail"]?.ToString() ?? "tylortrub@gmail.com" : "tylortrub@gmail.com",
                EnableEmailAlerts = Has("EnableEmailAlerts") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableEmailAlerts"])) : false,
                EnableStandardAlertEmails = Has("EnableStandardAlertEmails") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableStandardAlertEmails"])) : false,
                EnableOpportunityAlertEmails = Has("EnableOpportunityAlertEmails") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableOpportunityAlertEmails"])) : false,
                EnablePredictionAlertEmails = Has("EnablePredictionAlertEmails") ? Convert.ToBoolean(Convert.ToInt32(reader["EnablePredictionAlertEmails"])) : false,
                EnableGlobalAlertEmails = Has("EnableGlobalAlertEmails") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableGlobalAlertEmails"])) : false,
                EnableSystemHealthAlertEmails = Has("EnableSystemHealthAlertEmails") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableSystemHealthAlertEmails"])) : false,
                AlertPhoneNumber = Has("AlertPhoneNumber") ? reader["AlertPhoneNumber"]?.ToString() ?? "" : "",
                EnableSmsAlerts = Has("EnableSmsAlerts") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableSmsAlerts"])) : false,
                EnableStandardAlertSms = Has("EnableStandardAlertSms") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableStandardAlertSms"])) : false,
                EnableOpportunityAlertSms = Has("EnableOpportunityAlertSms") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableOpportunityAlertSms"])) : false,
                EnablePredictionAlertSms = Has("EnablePredictionAlertSms") ? Convert.ToBoolean(Convert.ToInt32(reader["EnablePredictionAlertSms"])) : false,
                EnableGlobalAlertSms = Has("EnableGlobalAlertSms") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableGlobalAlertSms"])) : false,
                // Push notification settings
                PushNotificationUserId = Has("PushNotificationUserId") ? reader["PushNotificationUserId"]?.ToString() ?? "" : "",
                EnablePushNotifications = Has("EnablePushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnablePushNotifications"])) : false,
                EnableStandardAlertPushNotifications = Has("EnableStandardAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableStandardAlertPushNotifications"])) : false,
                EnableOpportunityAlertPushNotifications = Has("EnableOpportunityAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableOpportunityAlertPushNotifications"])) : false,
                EnablePredictionAlertPushNotifications = Has("EnablePredictionAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnablePredictionAlertPushNotifications"])) : false,
                EnableGlobalAlertPushNotifications = Has("EnableGlobalAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableGlobalAlertPushNotifications"])) : false,
                EnableTechnicalIndicatorAlertPushNotifications = Has("EnableTechnicalIndicatorAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableTechnicalIndicatorAlertPushNotifications"])) : false,
                EnableSentimentShiftAlertPushNotifications = Has("EnableSentimentShiftAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableSentimentShiftAlertPushNotifications"])) : false,
                EnableSystemHealthAlertPushNotifications = Has("EnableSystemHealthAlertPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableSystemHealthAlertPushNotifications"])) : false,
                EnableTradeExecutionPushNotifications = Has("EnableTradeExecutionPushNotifications") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableTradeExecutionPushNotifications"])) : false,
                EnableVixMonitoring = Has("EnableVixMonitoring") ? Convert.ToBoolean(Convert.ToInt32(reader["EnableVixMonitoring"])) : true
            };
        }
        
        // Ensure at least one settings profile exists
        public void EnsureSettingsProfiles()
        {
            ResilienceHelper.Retry(() =>
            {
                // First ensure the table exists
                EnsureSettingsProfilesTable();
                
                // Then check if any profiles exist
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    using (var command = new SQLiteCommand("SELECT COUNT(*) FROM SettingsProfiles", connection))
                    {
                        var count = Convert.ToInt32(command.ExecuteScalar());
                        if (count == 0)
                        {
                            // Create a default profile from current settings
                            var defaultProfile = new DatabaseSettingsProfile
                            {
                                Name = "Default",
                                Description = "Default system settings",
                                IsDefault = true,
                                CreatedDate = DateTime.Now,
                                ModifiedDate = DateTime.Now,
                                EnableApiModalChecks = true,
                                ApiTimeoutSeconds = 30,
                                CacheDurationMinutes = 15,
                                EnableHistoricalDataCache = true,
                                EnableDarkMode = true,
                                ChartUpdateIntervalSeconds = 2,
                                DefaultGridRows = 4,
                                DefaultGridColumns = 4,
                                GridBorderColor = "#FF00FFFF",
                                EnablePriceAlerts = true,
                                EnableTradeNotifications = true,
                                EnablePaperTrading = true,
                                RiskLevel = "Low",
                                AlertEmail = "tylortrub@gmail.com",
                                EnableEmailAlerts = true,
                                EnableStandardAlertEmails = true,
                                EnableOpportunityAlertEmails = true,
                                EnablePredictionAlertEmails = true,
                                EnableGlobalAlertEmails = true,
                                EnableSystemHealthAlertEmails = true,
                                EnableVixMonitoring = true
                            };
                            CreateSettingsProfile(defaultProfile);
                        
                            DatabaseMonolith.Log("Info", "Created default settings profile");
                        }
                    }
                }
            }, RetryOptions.ForCriticalOperation());
        }
        
        // Set a profile as the default
        public bool SetProfileAsDefault(int profileId)
        {
            return ResilienceHelper.Retry(() =>
            {
                using (var connection = DatabaseMonolith.GetConnection())
                {
                    connection.Open();
                    
                    // Clear existing defaults
                    using (var clearCommand = new SQLiteCommand("UPDATE SettingsProfiles SET IsDefault = 0", connection))
                    {
                        clearCommand.ExecuteNonQuery();
                    }
                
                    // Set new default
                    using (var defaultCommand = new SQLiteCommand("UPDATE SettingsProfiles SET IsDefault = 1 WHERE Id = @Id", connection))
                    {
                        defaultCommand.Parameters.AddWithValue("@Id", profileId);
                        int rowsAffected = defaultCommand.ExecuteNonQuery();
                        return rowsAffected > 0;
                    }
                }
            }, RetryOptions.ForCriticalOperation());
        }
    }
}
