using System.IO;
using System.Text.Json;

namespace Quantra.Helpers
{
    public static class ApiKeyHelper
    {
        private const string SettingsFile = "alphaVantageSettings.json";
        private const string ApiKeyProperty = "AlphaVantageApiKey";
        private const string NewsApiKeyProperty = "NewsApiKey";
        private const string OpenAiApiKeyProperty = "OpenAiApiKey";


        public static string GetAlphaVantageApiKey()
        {
            if (!File.Exists(SettingsFile))
                //if(!File.Exists)
                throw new FileNotFoundException($"Settings file '{SettingsFile}' not found.");

            var json = File.ReadAllText(SettingsFile);
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty(ApiKeyProperty, out var apiKeyElement))
            {
                return apiKeyElement.GetString();
            }
            throw new KeyNotFoundException($"'{ApiKeyProperty}' not found in settings file.");
        }

        public static string GetNewsApiKey()
        {
            if (!File.Exists(SettingsFile))
                throw new FileNotFoundException($"Settings file '{SettingsFile}' not found.");

            var json = File.ReadAllText(SettingsFile);
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty(NewsApiKeyProperty, out var apiKeyElement))
            {
                return apiKeyElement.GetString();
            }
            throw new KeyNotFoundException($"'{NewsApiKeyProperty}' not found in settings file.");
        }

        public static string GetOpenAiApiKey()
        {
            if (!File.Exists(SettingsFile))
                throw new FileNotFoundException($"Settings file '{SettingsFile}' not found.");

            var json = File.ReadAllText(SettingsFile);
            using var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty(OpenAiApiKeyProperty, out var apiKeyElement))
            {
                return apiKeyElement.GetString();
            }
            throw new KeyNotFoundException($"'{OpenAiApiKeyProperty}' not found in settings file.");
        }

    }
}
