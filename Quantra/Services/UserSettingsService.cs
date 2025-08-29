using System;
using System.Collections.Generic;
using System.Data.SQLite;
using Quantra.Models;
using Newtonsoft.Json;

namespace Quantra.Services
{
    public static class UserSettingsService
    {
        public static void SaveUserSettings(UserSettings settings)
        {
            DatabaseMonolith.SaveUserSettings(settings);
        }

        public static UserSettings GetUserSettings()
        {
            return DatabaseMonolith.GetUserSettings();
        }

        public static string GetUserPreference(string key, string defaultValue = null)
        {
            return DatabaseMonolith.GetUserPreference(key, defaultValue);
        }

        public static void SaveUserPreference(string key, string value)
        {
            DatabaseMonolith.SaveUserPreference(key, value);
        }

        public static Dictionary<string, (string Username, string Password, string Pin)> GetRememberedAccounts()
        {
            return DatabaseMonolith.GetRememberedAccounts();
        }

        public static void RememberAccount(string username, string password, string pin)
        {
            DatabaseMonolith.RememberAccount(username, password, pin);
        }

        /// <summary>
        /// Get the active benchmark configuration
        /// </summary>
        public static (string type, string id) GetActiveBenchmark()
        {
            var settings = GetUserSettings();
            return (settings.ActiveBenchmarkType ?? "SPY", settings.ActiveBenchmarkId);
        }

        /// <summary>
        /// Set the active benchmark (standard benchmark like SPY, QQQ, etc.)
        /// </summary>
        public static void SetActiveBenchmark(string benchmarkType)
        {
            var settings = GetUserSettings();
            settings.ActiveBenchmarkType = benchmarkType;
            settings.ActiveBenchmarkId = null; // Clear custom benchmark ID
            SaveUserSettings(settings);
        }

        /// <summary>
        /// Set the active custom benchmark
        /// </summary>
        public static void SetActiveCustomBenchmark(string customBenchmarkId)
        {
            var settings = GetUserSettings();
            settings.ActiveBenchmarkType = "CUSTOM";
            settings.ActiveBenchmarkId = customBenchmarkId;
            SaveUserSettings(settings);
        }

        /// <summary>
        /// Save the window state (maximized/normal) for restoration on next startup
        /// </summary>
        public static void SaveWindowState(System.Windows.WindowState windowState)
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
        public static System.Windows.WindowState? GetSavedWindowState()
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
