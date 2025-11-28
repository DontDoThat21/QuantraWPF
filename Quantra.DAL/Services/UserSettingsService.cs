using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Models;
using Newtonsoft.Json;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Quantra.Utilities;

namespace Quantra.DAL.Services
{
    public class UserSettingsService : IUserSettingsService
    {
        private readonly QuantraDbContext _dbContext;
        private readonly LoggingService _loggingService;
        private const string USER_SETTINGS_KEY = "UserSettings";
        private const int COMPRESSION_THRESHOLD = 1000; // Compress strings larger than 1KB

        public UserSettingsService(QuantraDbContext dbContext, LoggingService loggingService)
        {
            _dbContext = dbContext;
            _loggingService = loggingService;
        }

        /// <summary>
        /// Saves user settings to database using UserPreferences table
        /// </summary>
        public void SaveUserSettings(UserSettings settings)
        {
            if (settings == null)
            {
                throw new ArgumentNullException(nameof(settings));
            }

            try
            {
                // Serialize UserSettings to JSON
                var json = JsonConvert.SerializeObject(settings);

                // Check if settings already exist
                var existingPreference = _dbContext.UserPreferences.Find(USER_SETTINGS_KEY);

                if (existingPreference != null)
                {
                    existingPreference.Value = json;
                    existingPreference.LastUpdated = DateTime.Now;
                }
                else
                {
                    _dbContext.UserPreferences.Add(new UserPreference
                    {
                        Key = USER_SETTINGS_KEY,
                        Value = json,
                        LastUpdated = DateTime.Now
                    });
                }

                _dbContext.SaveChanges();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to save user settings", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Gets user settings from database. Returns default settings if none exist.
        /// </summary>
        public UserSettings GetUserSettings()
        {
            try
            {
                var preference = _dbContext.UserPreferences.Find(USER_SETTINGS_KEY);

                if (preference != null && !string.IsNullOrWhiteSpace(preference.Value))
                {
                    return JsonConvert.DeserializeObject<UserSettings>(preference.Value);
                }

                // Return default settings if none exist
                return new UserSettings();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to retrieve user settings, returning defaults", ex.ToString());
                return new UserSettings();
            }
        }

        /// <summary>
        /// Gets user preference by key with automatic decompression
        /// </summary>
        public string GetUserPreference(string key, string defaultValue = null)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                throw new ArgumentException("Key cannot be null or whitespace", nameof(key));
            }

            try
            {
                var preference = _dbContext.UserPreferences.Find(key);
                if (preference?.Value == null)
                    return defaultValue;

                // Automatically decompress if the value was compressed
                if (CompressionHelper.IsCompressed(preference.Value))
                {
                    return CompressionHelper.DecompressString(preference.Value);
                }

                return preference.Value;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to retrieve user preference: {key}", ex.ToString());
                return defaultValue;
            }
        }

        /// <summary>
        /// Saves user preference with automatic compression for large values
        /// </summary>
        public void SaveUserPreference(string key, string value)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                throw new ArgumentException("Key cannot be null or whitespace", nameof(key));
            }

            try
            {
                // Automatically compress large values to avoid truncation
                string valueToStore = value;
                if (!string.IsNullOrEmpty(value) && value.Length > COMPRESSION_THRESHOLD)
                {
                    valueToStore = CompressionHelper.CompressString(value);
                    _loggingService.Log("Info", $"Compressed preference '{key}' from {value.Length} to {valueToStore.Length} characters");
                }

                var existingPreference = _dbContext.UserPreferences.Find(key);

                if (existingPreference != null)
                {
                    existingPreference.Value = valueToStore;
                    existingPreference.LastUpdated = DateTime.Now;
                }
                else
                {
                    _dbContext.UserPreferences.Add(new UserPreference
                    {
                        Key = key,
                        Value = valueToStore,
                        LastUpdated = DateTime.Now
                    });
                }

                _dbContext.SaveChanges();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to save user preference: {key}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Gets remembered accounts using UserCredentials table
        /// </summary>
        public Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts()
        {
            try
            {
                var credentials = _dbContext.UserCredentials.ToList();

                var accounts = new Dictionary<string, (string Username, string Password, string Pin)>();

                foreach (var cred in credentials)
                {
                    var key = $"{cred.Username}_{cred.Pin}";
                    accounts[key] = (cred.Username, cred.Password, cred.Pin);
                }

                return accounts;
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Failed to retrieve remembered accounts", ex.ToString());
                return new Dictionary<string, (string Username, string Password, string Pin)>();
            }
        }

        /// <summary>
        /// Remembers an account by storing it in UserCredentials table
        /// </summary>
        public void RememberAccount(string username, string password, string pin)
        {
            if (string.IsNullOrWhiteSpace(username))
            {
                throw new ArgumentException("Username cannot be null or whitespace", nameof(username));
            }

            try
            {
                // Check if account already exists
                var existing = _dbContext.UserCredentials
                    .FirstOrDefault(c => c.Username == username && c.Pin == pin);

                if (existing != null)
                {
                    // Update existing credentials
                    existing.Password = password;
                    existing.LastLoginDate = DateTime.Now;
                }
                else
                {
                    // Add new credentials
                    _dbContext.UserCredentials.Add(new UserCredential
                    {
                        Username = username,
                        Password = password,
                        Pin = pin ?? string.Empty,
                        LastLoginDate = DateTime.Now
                    });
                }

                _dbContext.SaveChanges();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to remember account: {username}", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Get the active benchmark configuration
        /// </summary>
        public (string type, string id) GetActiveBenchmark()
        {
            var settings = GetUserSettings();
            return (settings.ActiveBenchmarkType ?? "SPY", settings.ActiveBenchmarkId);
        }

        /// <summary>
        /// Set the active benchmark (standard benchmark like SPY, QQQ, etc.)
        /// </summary>
        public void SetActiveBenchmark(string benchmarkType)
        {
            var settings = GetUserSettings();
            settings.ActiveBenchmarkType = benchmarkType;
            settings.ActiveBenchmarkId = null; // Clear custom benchmark ID
            SaveUserSettings(settings);
        }

        /// <summary>
        /// Set the active custom benchmark
        /// </summary>
        public void SetActiveCustomBenchmark(string customBenchmarkId)
        {
            var settings = GetUserSettings();
            settings.ActiveBenchmarkType = "CUSTOM";
            settings.ActiveBenchmarkId = customBenchmarkId;
            SaveUserSettings(settings);
        }

        /// <summary>
        /// Save the window state (maximized/normal) for restoration on next startup
        /// </summary>
        public void SaveWindowState(System.Windows.WindowState windowState)
        {
            var settings = GetUserSettings();
            if (settings.RememberWindowState)
            {
                settings.LastWindowState = (int)windowState;
                SaveUserSettings(settings);
            }
        }

        /// <summary>
        /// Get the saved window state for restoration
        /// </summary>
        public System.Windows.WindowState? GetSavedWindowState()
        {
            var settings = GetUserSettings();
            if (settings.RememberWindowState)
            {
                return (System.Windows.WindowState)settings.LastWindowState;
            }
            return null;
        }

        /// <summary>
        /// Saves DataGrid configuration for a specific control in a tab.
        /// This method stores the DataGrid settings (width, height, column widths) in the UserAppSettings table.
        /// Multiple control configurations can be stored per tab.
        /// </summary>
        /// <param name="tabName">The name of the tab containing the control</param>
        /// <param name="controlName">The name of the control (e.g., "StockDataGrid")</param>
        /// <param name="settings">The DataGrid settings to save</param>
        public void SaveDataGridConfig(string tabName, string controlName, DataGridSettings settings)
        {
            if (string.IsNullOrWhiteSpace(tabName))
            {
                throw new ArgumentException("Tab name cannot be null or whitespace", nameof(tabName));
            }

            if (string.IsNullOrWhiteSpace(controlName))
            {
                throw new ArgumentException("Control name cannot be null or whitespace", nameof(controlName));
            }

            if (settings == null)
            {
                throw new ArgumentNullException(nameof(settings));
            }

            try
            {
                // Find existing UserAppSetting for this tab
                var userAppSetting = _dbContext.UserAppSettings
                    .FirstOrDefault(u => u.TabName == tabName);

                Dictionary<string, DataGridSettings> allConfigs;

                if (userAppSetting != null && !string.IsNullOrEmpty(userAppSetting.DataGridConfig))
                {
                    try
                    {
                        allConfigs = JsonConvert.DeserializeObject<Dictionary<string, DataGridSettings>>(userAppSetting.DataGridConfig)
                                   ?? new Dictionary<string, DataGridSettings>();
                    }
                    catch (Exception ex)
                    {
                        _loggingService.Log("Warning", $"Failed to parse existing DataGridConfig for tab '{tabName}'", ex.ToString());
                        allConfigs = new Dictionary<string, DataGridSettings>();
                    }
                }
                else
                {
                    allConfigs = new Dictionary<string, DataGridSettings>();
                }

                // Update the specific control's settings
                allConfigs[controlName] = settings;

                // Serialize back to JSON
                var updatedConfigJson = JsonConvert.SerializeObject(allConfigs);

                if (userAppSetting != null)
                {
                    // Update existing record
                    userAppSetting.DataGridConfig = updatedConfigJson;
                    _dbContext.SaveChanges();
                }
                else
                {
                    // Don't create tab entries for StockExplorer fallback names (auto-generated GUIDs)
                    // These are temporary names used for DataGrid settings persistence only
                    if (!tabName.StartsWith("StockExplorer_"))
                    {
                        // Insert new record if tab doesn't exist and it's not an auto-generated name
                        _dbContext.UserAppSettings.Add(new UserAppSetting
                        {
                            TabName = tabName,
                            TabOrder = 0,
                            DataGridConfig = updatedConfigJson,
                            GridRows = 4,
                            GridColumns = 4
                        });
                        _dbContext.SaveChanges();
                    }
                    else
                    {
                        // For StockExplorer fallback names, just log that settings were not saved to avoid tab creation
                        _loggingService.Log("Info", $"Skipped creating tab entry for auto-generated StockExplorer name: {tabName}");
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to save DataGrid config for tab '{tabName}', control '{controlName}'", ex.ToString());
                throw;
            }
        }

        /// <summary>
        /// Loads DataGrid configuration for a specific control in a tab.
        /// Returns default settings if no configuration is found.
        /// </summary>
        /// <param name="tabName">The name of the tab containing the control</param>
        /// <param name="controlName">The name of the control (e.g., "StockDataGrid")</param>
        /// <returns>DataGrid settings for the specified control, or default settings if not found</returns>
        public DataGridSettings LoadDataGridConfig(string tabName, string controlName)
        {
            if (string.IsNullOrWhiteSpace(tabName))
            {
                return new DataGridSettings();
            }

            if (string.IsNullOrWhiteSpace(controlName))
            {
                return new DataGridSettings();
            }

            try
            {
                var userAppSetting = _dbContext.UserAppSettings
                    .FirstOrDefault(u => u.TabName == tabName);

                if (userAppSetting == null || string.IsNullOrEmpty(userAppSetting.DataGridConfig))
                {
                    return new DataGridSettings();
                }

                try
                {
                    // Parse the JSON to get all DataGrid configs for this tab
                    var allConfigs = JsonConvert.DeserializeObject<Dictionary<string, DataGridSettings>>(userAppSetting.DataGridConfig);
                    if (allConfigs != null && allConfigs.ContainsKey(controlName))
                    {
                        return allConfigs[controlName];
                    }
                }
                catch (Exception ex)
                {
                    _loggingService.Log("Warning", $"Failed to parse DataGridConfig for tab '{tabName}', control '{controlName}'", ex.ToString());
                }

                return new DataGridSettings();
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Failed to load DataGrid config for tab '{tabName}', control '{controlName}'", ex.ToString());
                return new DataGridSettings();
            }
        }
    }
}
