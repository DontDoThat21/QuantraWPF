using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Quantra.DAL.Models;
using Quantra.DAL.Services.Interfaces;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing training configurations.
    /// Handles saving, loading, and managing multiple training presets.
    /// </summary>
    public class TrainingConfigurationService
    {
        private readonly LoggingService _loggingService;
        private readonly string _configDirectory;
        private const string CONFIG_FILE_EXTENSION = ".trainconfig.json";

        public TrainingConfigurationService(LoggingService loggingService)
        {
            _loggingService = loggingService;

            // Store configurations in AppData
            var appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            _configDirectory = Path.Combine(appData, "Quantra", "TrainingConfigurations");

            // Ensure directory exists
            Directory.CreateDirectory(_configDirectory);

            // Create default configurations if they don't exist
            EnsureDefaultConfigurations();
        }

        /// <summary>
        /// Ensure default configurations exist
        /// </summary>
        private void EnsureDefaultConfigurations()
        {
            try
            {
                var existingConfigs = GetAllConfigurations();

                if (!existingConfigs.Any(c => c.ConfigurationName == "Default"))
                {
                    SaveConfiguration(TrainingConfiguration.CreateDefault());
                }
                if (!existingConfigs.Any(c => c.ConfigurationName == "Fast Training"))
                {
                    SaveConfiguration(TrainingConfiguration.CreateFastTraining());
                }
                if (!existingConfigs.Any(c => c.ConfigurationName == "High Accuracy"))
                {
                    SaveConfiguration(TrainingConfiguration.CreateHighAccuracy());
                }
                if (!existingConfigs.Any(c => c.ConfigurationName == "TFT Optimized"))
                {
                    SaveConfiguration(TrainingConfiguration.CreateTFTOptimized());
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to create default training configurations");
            }
        }

        /// <summary>
        /// Save a training configuration
        /// </summary>
        public bool SaveConfiguration(TrainingConfiguration config)
        {
            try
            {
                // Validate configuration
                var errors = config.Validate();
                if (errors.Any())
                {
                    _loggingService?.Log("Error", $"Configuration validation failed: {string.Join(", ", errors)}");
                    return false;
                }

                config.LastModifiedDate = DateTime.Now;

                var fileName = GetSafeFileName(config.ConfigurationName) + CONFIG_FILE_EXTENSION;
                var filePath = Path.Combine(_configDirectory, fileName);

                var json = config.ToJson();
                File.WriteAllText(filePath, json);

                _loggingService?.Log("Info", $"Saved training configuration: {config.ConfigurationName}");
                return true;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to save configuration: {config.ConfigurationName}");
                return false;
            }
        }

        /// <summary>
        /// Load a training configuration by name
        /// </summary>
        public TrainingConfiguration LoadConfiguration(string configurationName)
        {
            try
            {
                var fileName = GetSafeFileName(configurationName) + CONFIG_FILE_EXTENSION;
                var filePath = Path.Combine(_configDirectory, fileName);

                if (!File.Exists(filePath))
                {
                    _loggingService?.Log("Warning", $"Configuration not found: {configurationName}");
                    return null;
                }

                var json = File.ReadAllText(filePath);
                var config = TrainingConfiguration.FromJson(json);

                _loggingService?.Log("Info", $"Loaded training configuration: {configurationName}");
                return config;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to load configuration: {configurationName}");
                return null;
            }
        }

        /// <summary>
        /// Get all available configurations
        /// </summary>
        public List<TrainingConfiguration> GetAllConfigurations()
        {
            var configurations = new List<TrainingConfiguration>();

            try
            {
                var configFiles = Directory.GetFiles(_configDirectory, "*" + CONFIG_FILE_EXTENSION);

                foreach (var file in configFiles)
                {
                    try
                    {
                        var json = File.ReadAllText(file);
                        var config = TrainingConfiguration.FromJson(json);
                        configurations.Add(config);
                    }
                    catch (Exception ex)
                    {
                        _loggingService?.LogErrorWithContext(ex, $"Failed to load configuration file: {file}");
                    }
                }

                // Sort by name
                configurations = configurations.OrderBy(c => c.ConfigurationName).ToList();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to get all configurations");
            }

            return configurations;
        }

        /// <summary>
        /// Delete a configuration
        /// </summary>
        public bool DeleteConfiguration(string configurationName)
        {
            try
            {
                // Don't allow deleting default configurations
                if (configurationName == "Default" || configurationName == "Fast Training" ||
                    configurationName == "High Accuracy" || configurationName == "TFT Optimized")
                {
                    _loggingService?.Log("Warning", $"Cannot delete built-in configuration: {configurationName}");
                    return false;
                }

                var fileName = GetSafeFileName(configurationName) + CONFIG_FILE_EXTENSION;
                var filePath = Path.Combine(_configDirectory, fileName);

                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                    _loggingService?.Log("Info", $"Deleted training configuration: {configurationName}");
                    return true;
                }

                return false;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to delete configuration: {configurationName}");
                return false;
            }
        }

        /// <summary>
        /// Export configuration to file
        /// </summary>
        public bool ExportConfiguration(TrainingConfiguration config, string filePath)
        {
            try
            {
                var json = config.ToJson();
                File.WriteAllText(filePath, json);
                _loggingService?.Log("Info", $"Exported configuration to: {filePath}");
                return true;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to export configuration to: {filePath}");
                return false;
            }
        }

        /// <summary>
        /// Import configuration from file
        /// </summary>
        public TrainingConfiguration ImportConfiguration(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    _loggingService?.Log("Error", $"Import file not found: {filePath}");
                    return null;
                }

                var json = File.ReadAllText(filePath);
                var config = TrainingConfiguration.FromJson(json);

                _loggingService?.Log("Info", $"Imported configuration from: {filePath}");
                return config;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to import configuration from: {filePath}");
                return null;
            }
        }

        /// <summary>
        /// Get safe filename from configuration name
        /// </summary>
        private string GetSafeFileName(string name)
        {
            var invalidChars = Path.GetInvalidFileNameChars();
            var safeName = string.Join("_", name.Split(invalidChars));
            return safeName;
        }

        /// <summary>
        /// Get the last used configuration or default
        /// </summary>
        public TrainingConfiguration GetLastUsedOrDefault()
        {
            try
            {
                // For now, just return default
                // Could be enhanced to store last used in settings
                return LoadConfiguration("Default") ?? TrainingConfiguration.CreateDefault();
            }
            catch
            {
                return TrainingConfiguration.CreateDefault();
            }
        }
    }
}
