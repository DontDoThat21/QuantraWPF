using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Quantra.Helpers
{
    public static class ApiKeyHelper
    {
        private const string SettingsFile = "alphaVantageSettings.json";
        private const string NewsApiKeyProperty = "NewsApiKey";
        private const string OpenAiApiKeyProperty = "OpenAiApiKey";

        // Cache for Alpha Vantage API key to avoid repeated database calls
        private static string _cachedAlphaVantageApiKey;
        private static DateTime _cacheExpiration = DateTime.MinValue;
        private static readonly object _cacheLock = new object();
        private const int CacheExpirationMinutes = 5;

        /// <summary>
        /// Gets the Alpha Vantage API key from the database (default settings profile).
        /// Uses caching to avoid repeated database calls.
        /// </summary>
        /// <returns>Alpha Vantage API key or empty string if not found</returns>
        public static string GetAlphaVantageApiKey()
        {
            lock (_cacheLock)
            {
                // Return cached value if still valid
                if (!string.IsNullOrEmpty(_cachedAlphaVantageApiKey) && DateTime.UtcNow < _cacheExpiration)
                {
                    return _cachedAlphaVantageApiKey;
                }

                try
                {
                    var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
                    optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);

                    using (var context = new QuantraDbContext(optionsBuilder.Options))
                    {
                        // Ensure database is created
                        context.Database.EnsureCreated();

                        // Get the default profile (IsDefault = true), or fall back to first profile
                        var defaultProfile = context.SettingsProfiles
                            .AsNoTracking()
                            .FirstOrDefault(p => p.IsDefault);

                        if (defaultProfile == null)
                        {
                            defaultProfile = context.SettingsProfiles
                                .AsNoTracking()
                                .FirstOrDefault();
                        }

                        if (defaultProfile != null && !string.IsNullOrWhiteSpace(defaultProfile.AlphaVantageApiKey))
                        {
                            _cachedAlphaVantageApiKey = defaultProfile.AlphaVantageApiKey;
                            _cacheExpiration = DateTime.UtcNow.AddMinutes(CacheExpirationMinutes);
                            return _cachedAlphaVantageApiKey;
                        }
                    }
                }
                catch (Microsoft.Data.SqlClient.SqlException)
                {
                    // Database connection error - return empty
                }
                catch (InvalidOperationException)
                {
                    // EF Core operation error - return empty
                }

                return string.Empty;
            }
        }

        /// <summary>
        /// Clears the cached API key, forcing a fresh database lookup on next call.
        /// </summary>
        public static void ClearApiKeyCache()
        {
            lock (_cacheLock)
            {
                _cachedAlphaVantageApiKey = null;
                _cacheExpiration = DateTime.MinValue;
            }
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
