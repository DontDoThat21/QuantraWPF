using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Quantra.Utilities;
using System;
using System.Collections.Generic;
using System.IO;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for handling API configuration setup and migration from legacy configuration files.
    /// This service centralizes API key management and configuration migration logic.
    /// </summary>
    public class ApiConfigurationService : IApiConfigurationService
    {
        private IConfiguration _configuration;
        private LoggingService _loggingService;
        private const string LEGACY_SETTINGS_FILE = "alphaVantageSettings.json";

        /// <summary>
        /// Gets the Alpha Vantage API key from configuration or legacy settings
        /// </summary>
        public string AlphaVantageApiKey { get; private set; }

        /// <summary>
        /// Gets the News API key from configuration or legacy settings
        /// </summary>
        public string NewsApiKey { get; private set; }

        /// <summary>
        /// Gets the OpenAI API key from configuration or legacy settings
        /// </summary>
        public string OpenAiApiKey { get; private set; }

        /// <summary>
        /// Initializes a new instance of the ApiConfigurationService
        /// </summary>
        /// <param name="configuration">IConfiguration instance containing app settings</param>
        public ApiConfigurationService(IConfiguration configuration, LoggingService loggingService)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _loggingService = new LoggingService();
            LoadApiKeys();
        }

        /// <summary>
        /// Loads API keys from configuration and legacy files.
        /// Priority: Environment Variables > Configuration > Legacy JSON File
        /// </summary>
        private void LoadApiKeys()
        {
            // Load Alpha Vantage API Key
            AlphaVantageApiKey = GetApiKey("AlphaVantageApiKey", "ALPHA_VANTAGE_API_KEY");

            // Load News API Key
            NewsApiKey = GetApiKey("NewsApiKey", "NEWS_API_KEY");

            // Load OpenAI API Key
            OpenAiApiKey = GetApiKey("OpenAiApiKey", "OPENAI_API_KEY", "CHATGPT_API_KEY");

            // Migrate legacy settings if they exist
            MigrateLegacySettings();
        }

        /// <summary>
        /// Gets an API key from multiple sources with fallback logic
        /// </summary>
        /// <param name="configKey">Configuration key to look for</param>
        /// <param name="envVars">Environment variable names to check (in priority order)</param>
        /// <returns>API key value or empty string if not found</returns>
        private string GetApiKey(string configKey, params string[] envVars)
        {
            // Priority 1: Check environment variables
            foreach (var envVar in envVars)
            {
                var envValue = Environment.GetEnvironmentVariable(envVar);
                if (!string.IsNullOrWhiteSpace(envValue))
                {
                    _loggingService.Log("Info", $"Loaded {configKey} from environment variable {envVar}");
                    return envValue;
                }
            }

            // Priority 2: Check IConfiguration (appsettings.json, user secrets, etc.)
            var configValue = _configuration[$"Api:{configKey}"];
            if (!string.IsNullOrWhiteSpace(configValue))
            {
                _loggingService.Log("Info", $"Loaded {configKey} from configuration");
                return configValue;
            }

            // Priority 3: Check legacy settings file
            var legacyValue = GetLegacyApiKey(configKey);
            if (!string.IsNullOrWhiteSpace(legacyValue))
            {
                _loggingService.Log("Warning", $"Loaded {configKey} from legacy settings file. Consider migrating to appsettings.json or environment variables.");
                return legacyValue;
            }

            _loggingService.Log("Warning", $"{configKey} not found in any configuration source");
            return string.Empty;
        }

        /// <summary>
        /// Retrieves API key from legacy JSON settings file
        /// </summary>
        /// <param name="keyName">Name of the key to retrieve</param>
        /// <returns>API key value or null if not found</returns>
        private string GetLegacyApiKey(string keyName)
        {
            if (!File.Exists(LEGACY_SETTINGS_FILE))
                return null;

            try
            {
                var json = File.ReadAllText(LEGACY_SETTINGS_FILE);
                var settings = JsonConvert.DeserializeObject<Dictionary<string, string>>(json);

                if (settings != null && settings.TryGetValue(keyName, out var value))
                {
                    return value;
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", $"Error reading legacy settings file for {keyName}", ex.ToString());
            }

            return null;
        }

        /// <summary>
        /// Migrates settings from legacy alphaVantageSettings.json to the new configuration system.
        /// This method handles backward compatibility during the transition period.
        /// </summary>
        private void MigrateLegacySettings()
        {
            if (!File.Exists(LEGACY_SETTINGS_FILE))
                return;

            try
            {
                var legacySettings = JsonConvert.DeserializeObject<Dictionary<string, string>>(
                    File.ReadAllText(LEGACY_SETTINGS_FILE));

                if (legacySettings == null || legacySettings.Count == 0)
                    return;

                bool migrated = false;

                // Migrate Alpha Vantage API Key if not already in configuration
                if (legacySettings.TryGetValue("AlphaVantageApiKey", out var alphaVantageKey) 
                    && !string.IsNullOrWhiteSpace(alphaVantageKey)
                    && string.IsNullOrWhiteSpace(AlphaVantageApiKey))
                {
                    AlphaVantageApiKey = alphaVantageKey;
                    migrated = true;
                }

                // Migrate News API Key if present
                if (legacySettings.TryGetValue("NewsApiKey", out var newsKey) 
                    && !string.IsNullOrWhiteSpace(newsKey)
                    && string.IsNullOrWhiteSpace(NewsApiKey))
                {
                    NewsApiKey = newsKey;
                    migrated = true;
                }

                // Migrate OpenAI API Key if present
                if (legacySettings.TryGetValue("OpenAiApiKey", out var openAiKey) 
                    && !string.IsNullOrWhiteSpace(openAiKey)
                    && string.IsNullOrWhiteSpace(OpenAiApiKey))
                {
                    OpenAiApiKey = openAiKey;
                    migrated = true;
                }

                if (migrated)
                {
                    _loggingService.Log("Info", 
                        $"Successfully migrated API keys from {LEGACY_SETTINGS_FILE}. " +
                        "Consider adding these to appsettings.json or environment variables for better security.");
                }
            }
            catch (Exception ex)
            {
                _loggingService.Log("Error", "Error migrating legacy Alpha Vantage settings", ex.ToString());
            }
        }

        /// <summary>
        /// Validates that required API keys are configured
        /// </summary>
        /// <returns>True if all required API keys are present, false otherwise</returns>
        public bool ValidateApiKeys()
        {
            bool isValid = true;

            if (string.IsNullOrWhiteSpace(AlphaVantageApiKey))
            {
                _loggingService.Log("Warning", "Alpha Vantage API key is not configured. Market data features will be limited.");
                isValid = false;
            }

            // News and OpenAI keys are optional, so we just log warnings
            if (string.IsNullOrWhiteSpace(NewsApiKey))
            {
                _loggingService.Log("Info", "News API key is not configured. News sentiment features will be unavailable.");
            }

            if (string.IsNullOrWhiteSpace(OpenAiApiKey))
            {
                _loggingService.Log("Info", "OpenAI API key is not configured. AI-powered analysis features will be unavailable.");
            }

            return isValid;
        }

        /// <summary>
        /// Refreshes API keys from all sources
        /// </summary>
        public void RefreshApiKeys()
        {
            LoadApiKeys();
            _loggingService.Log("Info", "API keys refreshed from configuration sources");
        }

        /// <summary>
        /// Sets the application configuration for database operations.
        /// </summary>
        /// <param name="configuration">IConfiguration instance containing app settings</param>
        /// <remarks>
        /// This method configures the database layer with application settings and handles
        /// migration from legacy configuration files (alphaVantageSettings.json) to the
        /// new configuration system. API keys and other sensitive settings are loaded
        /// and stored securely.
        /// 
        /// Should be called during application startup before other database operations.
        /// </remarks>
        /// <example>
        /// <code>
        /// var config = new ConfigurationBuilder()
        ///     .AddJsonFile("appsettings.json")
        ///     .Build();
        /// DatabaseMonolith.SetConfiguration(config);
        /// </code>
        /// </example>
        public void SetConfiguration(IConfiguration configuration)
        {
            _configuration = configuration;

            // Copy any settings from alphaVantageSettings.json to main config if they exist
            if (File.Exists("alphaVantageSettings.json"))
            {
                try
                {
                    var alphaVantageSettings = JsonConvert.DeserializeObject<Dictionary<string, string>>(
                        File.ReadAllText("alphaVantageSettings.json"));

                    if (alphaVantageSettings.TryGetValue("AlphaVantageApiKey", out var apiKey) && !string.IsNullOrWhiteSpace(apiKey))
                    {
                        // This is just temporary until we fully migrate to the new config system
                        // We'll store this in memory only for now
                        AlphaVantageApiKey = apiKey;
                    }
                }
                catch (Exception ex)
                {
                    _loggingService.Log("Error", "Error loading Alpha Vantage settings", ex.ToString());
                }
            }
        }
    }
}
