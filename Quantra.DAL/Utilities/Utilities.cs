using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Quantra.DAL.Utilities
{
    // Local DAL Utilities for API key retrieval from database
    public static class Utilities
    {
        private const string SettingsFile = "alphaVantageSettings.json";
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

        public static string GetOpenAiApiKey()
        {
            // Prefer environment variables
            var envKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
                        ?? Environment.GetEnvironmentVariable("CHATGPT_API_KEY");
            if (!string.IsNullOrWhiteSpace(envKey))
            {
                return envKey;
            }

            // Fallback to settings file
            try
            {
                if (File.Exists(SettingsFile))
                {
                    var json = File.ReadAllText(SettingsFile);
                    using var doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty(OpenAiApiKeyProperty, out var apiKeyElement))
                    {
                        var key = apiKeyElement.GetString();
                        if (!string.IsNullOrWhiteSpace(key))
                        {
                            return key;
                        }
                    }
                }
            }
            catch
            {
                // Ignore and fallback
            }

            return string.Empty;
        }

        // New: Retrieve News API key (e.g., for newsapi.org)
        public static string GetNewsApiKey()
        {
            // Environment variables first
            var envKey = Environment.GetEnvironmentVariable("NEWS_API_KEY")
                        ?? Environment.GetEnvironmentVariable("NEWSAPI_API_KEY");
            if (!string.IsNullOrWhiteSpace(envKey))
            {
                return envKey;
            }

            // Common settings files to probe
            var candidateFiles = new[]
            {
                "newsSettings.json",
                "NewsSettings.json",
                SettingsFile, // fallback to existing settings file, in case key is colocated
                "appsettings.json"
            };

            foreach (var file in candidateFiles)
            {
                try
                {
                    if (!File.Exists(file))
                        continue;

                    var json = File.ReadAllText(file);
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;

                    // Try common flat property names
                    if (TryReadStringProperty(root, "NewsApiKey", out var key) && !string.IsNullOrWhiteSpace(key))
                        return key;
                    if (TryReadStringProperty(root, "ApiKey", out key) && !string.IsNullOrWhiteSpace(key))
                        return key;
                    if (TryReadStringProperty(root, "NEWS_API_KEY", out key) && !string.IsNullOrWhiteSpace(key))
                        return key;

                    // Try nested object: NewsApi: { ApiKey: "..." }
                    if (root.TryGetProperty("NewsApi", out var newsApiObj))
                    {
                        if (TryReadStringProperty(newsApiObj, "ApiKey", out key) && !string.IsNullOrWhiteSpace(key))
                            return key;
                        if (TryReadStringProperty(newsApiObj, "Key", out key) && !string.IsNullOrWhiteSpace(key))
                            return key;
                    }
                }
                catch
                {
                    // Ignore parse errors and continue
                }
            }

            return string.Empty;
        }

        private static bool TryReadStringProperty(JsonElement element, string propertyName, out string value)
        {
            value = null;
            if (element.ValueKind == JsonValueKind.Object && element.TryGetProperty(propertyName, out var prop))
            {
                value = prop.GetString();
                return true;
            }
            return false;
        }
    }
}
