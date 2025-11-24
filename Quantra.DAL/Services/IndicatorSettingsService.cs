using System;
using System.Collections.Generic;
using Quantra.Models;
using Quantra.Repositories;
using Quantra.DAL.Data;

namespace Quantra.DAL.Services
{
    public class IndicatorSettingsService
    {
        private readonly IndicatorSettingsRepository _repository;

        public IndicatorSettingsService(QuantraDbContext context)
        {
            _repository = new IndicatorSettingsRepository(context);
        }

        // Save a single indicator setting
        public void SaveIndicatorSetting(int controlId, string indicatorName, bool isEnabled)
        {
            try
            {
                var setting = new IndicatorSettingsModel(controlId, indicatorName, isEnabled);
                _repository.SaveIndicatorSetting(setting);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save indicator setting: {indicatorName}: {ex.Message}");
                throw;
            }
        }

        // Save multiple indicator settings at once
        public void SaveIndicatorSettings(int controlId, Dictionary<string, bool> indicators)
        {
            try
            {
                var settings = new List<IndicatorSettingsModel>();

                foreach (var indicator in indicators)
                {
                    settings.Add(new IndicatorSettingsModel(controlId, indicator.Key, indicator.Value));
                }

                _repository.SaveIndicatorSettings(settings);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save multiple indicator settings: {ex.Message}");
                throw;
            }
        }

        // Get all settings for a control and return as dictionary
        public Dictionary<string, bool> GetIndicatorSettingsForControl(string controlId)
        {
            var indicatorDictionary = new Dictionary<string, bool>();

            try
            {
                var settings = _repository.GetIndicatorSettingsForControl(controlId);

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
                Console.WriteLine($"Failed to retrieve indicator settings for control {controlId}: {ex.Message}");
            }

            return indicatorDictionary;
        }

        public IndicatorSettingsModel GetSettingsForControl(int controlId)
        {
            var settings = _repository.GetByControlId(controlId);

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
            if (_repository.Exists(settings.ControlId))
            {
                _repository.Update(settings);
            }
            else
            {
                _repository.Save(settings);
            }
        }
    }
}
