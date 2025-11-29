using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
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


        /// <summary>
        /// Gets the Alpha Vantage API key from the database (default settings profile).
        /// </summary>
        /// <returns>Alpha Vantage API key or empty string if not found</returns>
        public static string GetAlphaVantageApiKey()
        {
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
                        return defaultProfile.AlphaVantageApiKey;
                    }
                }
            }
            catch
            {
                // Swallow and return empty
            }

            return string.Empty;
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
