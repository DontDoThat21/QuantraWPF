using System;
using System.IO;
using System.Text.Json;

namespace Quantra.DAL.Utilities
{
    // Local DAL Utilities to avoid cross-project dependency for API key retrieval
    public static class Utilities
    {
        private const string SettingsFile = "alphaVantageSettings.json";
        private const string AlphaVantageApiKeyProperty = "AlphaVantageApiKey";
        private const string OpenAiApiKeyProperty = "OpenAiApiKey";

        public static string GetAlphaVantageApiKey()
        {
            // Prefer environment variable
            var envKey = Environment.GetEnvironmentVariable("ALPHA_VANTAGE_API_KEY");
            if (!string.IsNullOrWhiteSpace(envKey))
            {
                return envKey;
            }

            // Fallback to local settings file
            try
            {
                if (File.Exists(SettingsFile))
                {
                    var json = File.ReadAllText(SettingsFile);
                    using var doc = JsonDocument.Parse(json);
                    if (doc.RootElement.TryGetProperty(AlphaVantageApiKeyProperty, out var apiKeyElement))
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
                // Swallow and fallback below
            }

            // Last resort
            return string.Empty;
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
