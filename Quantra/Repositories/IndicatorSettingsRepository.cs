using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using Quantra.Models;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;

namespace Quantra.Repositories
{
    /// <summary>
    /// DEPRECATED: This file is maintained for backward compatibility only.
    /// Use Quantra.DAL.Repositories.IndicatorSettingsRepository with dependency injection instead.
    /// This repository has been migrated to use Entity Framework Core.
    /// </summary>
    [Obsolete("Use Quantra.DAL.Repositories.IndicatorSettingsRepository with QuantraDbContext via dependency injection")]
    public class IndicatorSettingsRepository
    {
        private readonly QuantraDbContext _context;

        public IndicatorSettingsRepository(QuantraDbContext context)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
        }

        // Save or update a single indicator setting using EF Core
        public void SaveIndicatorSetting(IndicatorSettingsModel setting)
        {
            try
            {
                // Check if entity already exists
                var existingEntity = _context.IndicatorSettings
            .FirstOrDefault(i => i.ControlId == setting.ControlId &&
          i.IndicatorName == setting.IndicatorName);

                if (existingEntity != null)
                {
                    // Update existing
                    existingEntity.IsEnabled = setting.IsEnabled;
                    existingEntity.LastUpdated = DateTime.Now;
                    _context.IndicatorSettings.Update(existingEntity);
                }
                else
                {
                    // Add new
                    var entity = new IndicatorSettingsEntity
                    {
                        ControlId = setting.ControlId,
                        IndicatorName = setting.IndicatorName,
                        IsEnabled = setting.IsEnabled,
                        LastUpdated = DateTime.Now
                    };
                    _context.IndicatorSettings.Add(entity);
                }

                _context.SaveChanges();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save indicator setting for {setting.IndicatorName}: {ex.Message}");
                throw;
            }
        }

        // Save multiple indicator settings at once (atomically) using EF Core
        public void SaveIndicatorSettings(List<IndicatorSettingsModel> settings)
        {
            try
            {
                foreach (var setting in settings)
                {
                    var existingEntity = _context.IndicatorSettings
                  .FirstOrDefault(i => i.ControlId == setting.ControlId &&
                             i.IndicatorName == setting.IndicatorName);

                    if (existingEntity != null)
                    {
                        existingEntity.IsEnabled = setting.IsEnabled;
                        existingEntity.LastUpdated = DateTime.Now;
                        _context.IndicatorSettings.Update(existingEntity);
                    }
                    else
                    {
                        var entity = new IndicatorSettingsEntity
                        {
                            ControlId = setting.ControlId,
                            IndicatorName = setting.IndicatorName,
                            IsEnabled = setting.IsEnabled,
                            LastUpdated = DateTime.Now
                        };
                        _context.IndicatorSettings.Add(entity);
                    }
                }

                _context.SaveChanges();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save multiple indicator settings: {ex.Message}");
                throw;
            }
        }

        // Get all indicator settings for a specific control using EF Core
        public List<IndicatorSettingsModel> GetIndicatorSettingsForControl(string controlId)
        {
            try
            {
                if (!int.TryParse(controlId, out int controlIdInt))
                {
                    return new List<IndicatorSettingsModel>();
                }

                return _context.IndicatorSettings
                          .AsNoTracking()
           .Where(i => i.ControlId == controlIdInt)
            .Select(i => new IndicatorSettingsModel
            {
                Id = i.Id,
                ControlId = i.ControlId,
                IndicatorName = i.IndicatorName,
                IsEnabled = i.IsEnabled,
                LastUpdated = i.LastUpdated
            })
             .ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to retrieve indicator settings for control {controlId}: {ex.Message}");
                throw;
            }
        }

        public bool Exists(int controlId)
        {
            try
            {
                return _context.IndicatorSettings
                         .AsNoTracking()
                         .Any(i => i.ControlId == controlId);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to check existence of indicator settings for control {controlId}: {ex.Message}");
                throw;
            }
        }

        public IndicatorSettingsModel GetByControlId(int controlId)
        {
            try
            {
                var entities = _context.IndicatorSettings
     .AsNoTracking()
     .Where(i => i.ControlId == controlId)
                .ToList();

                if (!entities.Any())
                {
                    return null;
                }

                var settings = new IndicatorSettingsModel
                {
                    ControlId = controlId,
                    Id = entities.First().Id
                };

                // Map indicators to properties
                foreach (var entity in entities)
                {
                    switch (entity.IndicatorName)
                    {
                        case "VWAP":
                            settings.UseVwap = entity.IsEnabled;
                            break;
                        case "MACD":
                            settings.UseMacd = entity.IsEnabled;
                            break;
                        case "RSI":
                            settings.UseRsi = entity.IsEnabled;
                            break;
                        case "Bollinger":
                            settings.UseBollinger = entity.IsEnabled;
                            break;
                        case "MA":
                            settings.UseMa = entity.IsEnabled;
                            break;
                        case "Volume":
                            settings.UseVolume = entity.IsEnabled;
                            break;
                        case "BreadthThrust":
                            settings.UseBreadthThrust = entity.IsEnabled;
                            break;
                    }
                }

                return settings;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to retrieve indicator settings for control {controlId}: {ex.Message}");
                throw;
            }
        }

        public void Save(IndicatorSettingsModel settings)
        {
            try
            {
                // Save each indicator setting individually
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "VWAP", settings.UseVwap));
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "MACD", settings.UseMacd));
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "RSI", settings.UseRsi));
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "Bollinger", settings.UseBollinger));
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "MA", settings.UseMa));
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "Volume", settings.UseVolume));
                SaveIndicatorSetting(new IndicatorSettingsModel(settings.ControlId, "BreadthThrust", settings.UseBreadthThrust));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save indicator settings for control {settings.ControlId}: {ex.Message}");
                throw;
            }
        }

        public void Update(IndicatorSettingsModel settings)
        {
            try
            {
                // Since SaveIndicatorSetting uses upsert logic, we can reuse the same Save method
                Save(settings);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to update indicator settings for control {settings.ControlId}: {ex.Message}");
                throw;
            }
        }
    }
}
