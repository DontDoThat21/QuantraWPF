using System;
using System.Collections.Generic;
using Quantra.Models;
using Quantra.Repositories;
using Quantra;

namespace Quantra.Services
{
    public class IndicatorSettingsService
    {
        private readonly IndicatorSettingsRepository repository = new IndicatorSettingsRepository();

        // Initialize the service by ensuring table exists
        public static void InitializeService()
        {
            try
            {
                IndicatorSettingsRepository.EnsureIndicatorSettingsTableExists();
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to initialize IndicatorSettingsService", ex.ToString());
            }
        }

        // Save a single indicator setting
        public static void SaveIndicatorSetting(int controlId, string indicatorName, bool isEnabled)
        {
            try
            {
                var setting = new IndicatorSettingsModel(controlId, indicatorName, isEnabled);
                IndicatorSettingsRepository.SaveIndicatorSetting(setting);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to save indicator setting: {indicatorName}", ex.ToString());
            }
        }

        // Save multiple indicator settings at once
        public static void SaveIndicatorSettings(int controlId, Dictionary<string, bool> indicators)
        {
            try
            {
                var settings = new List<IndicatorSettingsModel>();

                foreach (var indicator in indicators)
                {
                    settings.Add(new IndicatorSettingsModel(controlId, indicator.Key, indicator.Value));
                }

                IndicatorSettingsRepository.SaveIndicatorSettings(settings);
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", "Failed to save multiple indicator settings", ex.ToString());
            }
        }

        // Get all settings for a control and return as dictionary
        public static Dictionary<string, bool> GetIndicatorSettingsForControl(string controlId)
        {
            var indicatorDictionary = new Dictionary<string, bool>();

            try
            {
                var settings = IndicatorSettingsRepository.GetIndicatorSettingsForControl(controlId);

                foreach (var setting in settings)
                {
                    indicatorDictionary[setting.IndicatorName] = setting.IsEnabled;
                }
                
                // Add default value for Breadth Thrust if not present
                if (!indicatorDictionary.ContainsKey("BreadthThrust"))
                {
                    indicatorDictionary["BreadthThrust"] = false;
                }
            }
            catch (Exception ex)
            {
                DatabaseMonolith.Log("Error", $"Failed to retrieve indicator settings for control {controlId}", ex.ToString());
            }

            return indicatorDictionary;
        }

        public IndicatorSettingsModel GetSettingsForControl(int controlId)
        {
            var settings = repository.GetByControlId(controlId);
            
            // If no settings exist yet, initialize with defaults
            if (settings == null)
            {
                settings = new IndicatorSettingsModel
                {
                    ControlId = controlId,
                    UseVwap = true,
                    UseMacd = true,
                    UseRsi = true,
                    UseBollinger = true,
                    UseMa = true,
                    UseVolume = true,
                    UseBreadthThrust = false
                };
            }
            return settings;
        }

        public void SaveOrUpdateSettingsForControl(IndicatorSettingsModel settings)
        {
            if (repository.Exists(settings.ControlId))
            {
                repository.Update(settings);
            }
            else
            {
                repository.Save(settings);
            }
        }
    }
}
