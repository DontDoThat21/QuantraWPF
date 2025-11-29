using System.IO;
using System.Text.Json;

namespace Quantra.Helpers
{
    public static class ApiKeyHelper
    {
        private const string SettingsFile = "alphaVantageSettings.json";
        private const string NewsApiKeyProperty = "NewsApiKey";
        private const string OpenAiApiKeyProperty = "OpenAiApiKey";

        /// <summary>
        /// Gets the Alpha Vantage API key from the database (default settings profile).
        /// Delegates to DAL Utilities for centralized caching.
        /// </summary>
        /// <returns>Alpha Vantage API key or empty string if not found</returns>
        public static string GetAlphaVantageApiKey()
        {
            // Delegate to DAL Utilities for centralized caching
            return Quantra.DAL.Utilities.Utilities.GetAlphaVantageApiKey();
        }

        /// <summary>
        /// Clears the cached API key, forcing a fresh database lookup on next call.
        /// </summary>
        public static void ClearApiKeyCache()
        {
            // Delegate to DAL Utilities for centralized cache management
            Quantra.DAL.Utilities.Utilities.ClearApiKeyCache();
        }

        public static string GetNewsApiKey()
        {
            if (!File.Exists(SettingsFile))
                return string.Empty;

            var json = File.ReadAllText(SettingsFile);
            using (var doc = JsonDocument.Parse(json))
            {
                if (doc.RootElement.TryGetProperty(NewsApiKeyProperty, out var apiKeyElement))
                {
                    return apiKeyElement.GetString() ?? string.Empty;
                }
            }
            return string.Empty;
        }

        public static string GetOpenAiApiKey()
        {
            if (!File.Exists(SettingsFile))
                return string.Empty;

            var json = File.ReadAllText(SettingsFile);
            using (var doc = JsonDocument.Parse(json))
            {
                if (doc.RootElement.TryGetProperty(OpenAiApiKeyProperty, out var apiKeyElement))
                {
                    return apiKeyElement.GetString() ?? string.Empty;
                }
            }
            return string.Empty;
        }

    }
}
