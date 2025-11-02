using System;
using System.Collections.Generic;
using System.Linq;
using Quantra.Models;
using Newtonsoft.Json;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.DAL.Services
{
    public class UserSettingsService : IUserSettingsService
    {
        private readonly QuantraDbContext _dbContext;
        private const string USER_SETTINGS_KEY = "UserSettings";

        public UserSettingsService(QuantraDbContext dbContext)
        {
            _dbContext = dbContext;
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
                LoggingService.Log("Error", "Failed to save user settings", ex.ToString());
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
                LoggingService.Log("Error", "Failed to retrieve user settings, returning defaults", ex.ToString());
                return new UserSettings();
            }
        }

        /// <summary>
        /// Gets user preference by key
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
                return preference?.Value ?? defaultValue;
            }
            catch (Exception ex)
            {
                LoggingService.Log("Error", $"Failed to retrieve user preference: {key}", ex.ToString());
                return defaultValue;
            }
        }

        /// <summary>
        /// Saves user preference
        /// </summary>
        public void SaveUserPreference(string key, string value)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                throw new ArgumentException("Key cannot be null or whitespace", nameof(key));
            }

            try
            {
                var existingPreference = _dbContext.UserPreferences.Find(key);

                if (existingPreference != null)
                {
                    existingPreference.Value = value;
                    existingPreference.LastUpdated = DateTime.Now;
                }
                else
                {
                    _dbContext.UserPreferences.Add(new UserPreference
                    {
                        Key = key,
                        Value = value,
                        LastUpdated = DateTime.Now
                    });
                }

                _dbContext.SaveChanges();
            }
            catch (Exception ex)
            {
                LoggingService.Log("Error", $"Failed to save user preference: {key}", ex.ToString());
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
                LoggingService.Log("Error", "Failed to retrieve remembered accounts", ex.ToString());
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
                LoggingService.Log("Error", $"Failed to remember account: {username}", ex.ToString());
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
    }
}
