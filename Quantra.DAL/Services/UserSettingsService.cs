using System;
using System.Collections.Generic;
//using System.Data.SQLite;
using Quantra.Models;
using Newtonsoft.Json;

namespace Quantra.DAL.Services
{
    public class UserSettingsService : IUserSettingsService
    {
        public void SaveUserSettings(UserSettings settings)
        {
            DatabaseMonolith.SaveUserSettings(settings);
        }

        public UserSettings GetUserSettings()
        {
            return DatabaseMonolith.GetUserSettings();
        }

        public string GetUserPreference(string key, string defaultValue = null)
        {
            return DatabaseMonolith.GetUserPreference(key, defaultValue);
        }

        public void SaveUserPreference(string key, string value)
        {
            DatabaseMonolith.SaveUserPreference(key, value);
        }

        public Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts()
        {
            return DatabaseMonolith.GetRememberedAccounts();
        }

        public void RememberAccount(string username, string password, string pin)
        {
            DatabaseMonolith.RememberAccount(username, password, pin);
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
