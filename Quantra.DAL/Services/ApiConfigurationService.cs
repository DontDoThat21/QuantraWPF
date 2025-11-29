using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using Quantra.DAL.Data;
using Quantra.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for handling API configuration setup and retrieval from database.
    /// This service centralizes API key management by loading from the default settings profile in the database.
    /// </summary>
    public class ApiConfigurationService : IApiConfigurationService
    {
        private IConfiguration _configuration;
        private LoggingService _loggingService;

        /// <summary>
        /// Gets the Alpha Vantage API key from the default settings profile in the database
        /// </summary>
        public string AlphaVantageApiKey { get; private set; }

        /// <summary>
        /// Gets the News API key from configuration
        /// </summary>
        public string NewsApiKey { get; private set; }

        /// <summary>
        /// Gets the OpenAI API key from configuration
        /// </summary>
        public string OpenAiApiKey { get; private set; }

        /// <summary>
        /// Initializes a new instance of the ApiConfigurationService
        /// </summary>
        /// <param name="configuration">IConfiguration instance containing app settings</param>
        /// <param name="loggingService">LoggingService instance for logging</param>
        public ApiConfigurationService(IConfiguration configuration, LoggingService loggingService)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _loggingService = loggingService ?? new LoggingService();
            LoadApiKeys();
        }

        /// <summary>
        /// Loads API keys from the database (default settings profile) for AlphaVantage,
        /// and from configuration for other API keys.
        /// </summary>
        private void LoadApiKeys()
        {
            // Load Alpha Vantage API Key from database (default profile)
            AlphaVantageApiKey = GetAlphaVantageApiKeyFromDatabase();

            // Load News API Key from configuration
            NewsApiKey = GetApiKeyFromConfiguration("NewsApiKey", "NEWS_API_KEY");

            // Load OpenAI API Key from configuration
            OpenAiApiKey = GetApiKeyFromConfiguration("OpenAiApiKey", "OPENAI_API_KEY", "CHATGPT_API_KEY");
        }

        /// <summary>
        /// Gets the Alpha Vantage API key from the database using EntityFramework.
        /// Loads from the default settings profile (IsDefault = true).
        /// </summary>
        /// <returns>Alpha Vantage API key or empty string if not found</returns>
        private string GetAlphaVantageApiKeyFromDatabase()
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
                        _loggingService.Log("Info", "Loaded AlphaVantageApiKey from database (default settings profile)");
                        return defaultProfile.AlphaVantageApiKey;
                    }
                }

                _loggingService.Log("Warning", "AlphaVantageApiKey not found in database settings profile");
                return string.Empty;
            }
            catch (Microsoft.Data.SqlClient.SqlException ex)
            {
                _loggingService.Log("Error", "Database connection error loading AlphaVantageApiKey", ex.ToString());
                return string.Empty;
            }
            catch (InvalidOperationException ex)
            {
                _loggingService.Log("Error", "EF Core operation error loading AlphaVantageApiKey", ex.ToString());
                return string.Empty;
            }
        }

        /// <summary>
        /// Gets an API key from configuration sources (IConfiguration).
        /// </summary>
        /// <param name="configKey">Configuration key to look for</param>
        /// <param name="envVars">Environment variable names to check (in priority order)</param>
        /// <returns>API key value or empty string if not found</returns>
        private string GetApiKeyFromConfiguration(string configKey, params string[] envVars)
        {
            // Priority 1: Check IConfiguration (appsettings.json, user secrets, etc.)
            var configValue = _configuration[$"Api:{configKey}"];
            if (!string.IsNullOrWhiteSpace(configValue))
            {
                _loggingService.Log("Info", $"Loaded {configKey} from configuration");
                return configValue;
            }

            _loggingService.Log("Info", $"{configKey} not found in configuration source");
            return string.Empty;
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
        /// Sets the application configuration and reloads API keys.
        /// </summary>
        /// <param name="configuration">IConfiguration instance containing app settings</param>
        /// <remarks>
        /// This method configures the service with application settings.
        /// API keys are loaded from the database (default settings profile).
        /// 
        /// Should be called during application startup before other database operations.
        /// </remarks>
        public void SetConfiguration(IConfiguration configuration)
        {
            _configuration = configuration;
            
            // Reload API keys from database
            LoadApiKeys();
        }
    }
}
