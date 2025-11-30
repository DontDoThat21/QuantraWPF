using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Quantra.DAL.Data;
using Quantra.DAL.Data.Entities;
using Newtonsoft.Json;

namespace Quantra.DAL.Services
{
    /// <summary>
    /// Service for managing stock configurations - predefined sets of symbols for batch loading
    /// </summary>
    public class StockConfigurationService
    {
        private readonly LoggingService _loggingService;

        public StockConfigurationService(LoggingService loggingService)
        {
            _loggingService = loggingService;
        }

        private DbContextOptions<QuantraDbContext> CreateOptions()
        {
            var optionsBuilder = new DbContextOptionsBuilder<QuantraDbContext>();
            optionsBuilder.UseSqlServer(ConnectionHelper.ConnectionString);
            return optionsBuilder.Options;
        }

        /// <summary>
        /// Creates a new stock configuration
        /// </summary>
        /// <param name="name">Name of the configuration</param>
        /// <param name="description">Description of the configuration</param>
        /// <param name="symbols">List of stock symbols</param>
        /// <returns>Created configuration entity</returns>
        public StockConfigurationEntity CreateConfiguration(string name, string description, List<string> symbols)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Configuration name cannot be empty", nameof(name));
            }

            if (symbols == null || !symbols.Any())
            {
                throw new ArgumentException("Configuration must contain at least one symbol", nameof(symbols));
            }

            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    // Check if name already exists
                    var existing = context.StockConfigurations.FirstOrDefault(c => c.Name == name);
                    if (existing != null)
                    {
                        throw new InvalidOperationException($"A configuration with name '{name}' already exists");
                    }

                    var entity = new StockConfigurationEntity
                    {
                        Name = name,
                        Description = description ?? string.Empty,
                        Symbols = JsonConvert.SerializeObject(symbols),
                        CreatedAt = DateTime.Now,
                        IsDefault = false
                    };

                    context.StockConfigurations.Add(entity);
                    context.SaveChanges();

                    _loggingService.Log("Info", $"Created stock configuration '{name}' with {symbols.Count} symbols");
                    return entity;
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to create stock configuration '{name}'");
                throw;
            }
        }

        /// <summary>
        /// Updates an existing stock configuration
        /// </summary>
        public StockConfigurationEntity UpdateConfiguration(int id, string name, string description, List<string> symbols)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                throw new ArgumentException("Configuration name cannot be empty", nameof(name));
            }

            if (symbols == null || !symbols.Any())
            {
                throw new ArgumentException("Configuration must contain at least one symbol", nameof(symbols));
            }

            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    var entity = context.StockConfigurations.FirstOrDefault(c => c.Id == id);
                    if (entity == null)
                    {
                        throw new InvalidOperationException($"Configuration with ID {id} not found");
                    }

                    // Check if new name conflicts with another configuration
                    var nameConflict = context.StockConfigurations.FirstOrDefault(c => c.Name == name && c.Id != id);
                    if (nameConflict != null)
                    {
                        throw new InvalidOperationException($"A configuration with name '{name}' already exists");
                    }

                    entity.Name = name;
                    entity.Description = description ?? string.Empty;
                    entity.Symbols = JsonConvert.SerializeObject(symbols);

                    context.SaveChanges();

                    _loggingService.Log("Info", $"Updated stock configuration '{name}' with {symbols.Count} symbols");
                    return entity;
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to update stock configuration ID {id}");
                throw;
            }
        }

        /// <summary>
        /// Deletes a stock configuration
        /// </summary>
        public bool DeleteConfiguration(int id)
        {
            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    var entity = context.StockConfigurations.FirstOrDefault(c => c.Id == id);
                    if (entity == null)
                    {
                        _loggingService.Log("Warning", $"Configuration with ID {id} not found for deletion");
                        return false;
                    }

                    context.StockConfigurations.Remove(entity);
                    context.SaveChanges();

                    _loggingService.Log("Info", $"Deleted stock configuration '{entity.Name}'");
                    return true;
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to delete stock configuration ID {id}");
                return false;
            }
        }

        /// <summary>
        /// Gets all stock configurations
        /// </summary>
        public List<StockConfigurationEntity> GetAllConfigurations()
        {
            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    return context.StockConfigurations
                        .AsNoTracking()
                        .OrderBy(c => c.Name)
                        .ToList();
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Failed to retrieve stock configurations");
                return new List<StockConfigurationEntity>();
            }
        }

        /// <summary>
        /// Gets a specific stock configuration by ID
        /// </summary>
        public StockConfigurationEntity GetConfiguration(int id)
        {
            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    return context.StockConfigurations
                        .AsNoTracking()
                        .FirstOrDefault(c => c.Id == id);
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to retrieve stock configuration ID {id}");
                return null;
            }
        }

        /// <summary>
        /// Gets symbols from a stock configuration
        /// </summary>
        public List<string> GetConfigurationSymbols(int id)
        {
            try
            {
                var entity = GetConfiguration(id);
                if (entity == null)
                {
                    return new List<string>();
                }

                return JsonConvert.DeserializeObject<List<string>>(entity.Symbols) ?? new List<string>();
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to parse symbols for configuration ID {id}");
                return new List<string>();
            }
        }

        /// <summary>
        /// Sets a configuration as the default
        /// </summary>
        public bool SetDefaultConfiguration(int id)
        {
            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    // Clear existing default
                    var existingDefault = context.StockConfigurations.Where(c => c.IsDefault).ToList();
                    foreach (var config in existingDefault)
                    {
                        config.IsDefault = false;
                    }

                    // Set new default
                    var entity = context.StockConfigurations.FirstOrDefault(c => c.Id == id);
                    if (entity == null)
                    {
                        return false;
                    }

                    entity.IsDefault = true;
                    context.SaveChanges();

                    _loggingService.Log("Info", $"Set '{entity.Name}' as default stock configuration");
                    return true;
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to set default configuration ID {id}");
                return false;
            }
        }

        /// <summary>
        /// Gets the default stock configuration
        /// </summary>
        public StockConfigurationEntity GetDefaultConfiguration()
        {
            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    return context.StockConfigurations
                        .AsNoTracking()
                        .FirstOrDefault(c => c.IsDefault);
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, "Failed to retrieve default stock configuration");
                return null;
            }
        }

        /// <summary>
        /// Updates the LastUsed timestamp for a configuration
        /// </summary>
        public void UpdateLastUsed(int id)
        {
            try
            {
                using (var context = new QuantraDbContext(CreateOptions()))
                {
                    var entity = context.StockConfigurations.FirstOrDefault(c => c.Id == id);
                    if (entity != null)
                    {
                        entity.LastUsed = DateTime.Now;
                        context.SaveChanges();
                    }
                }
            }
            catch (Exception ex)
            {
                _loggingService.LogErrorWithContext(ex, $"Failed to update last used for configuration ID {id}");
            }
        }

        /// <summary>
        /// Gets the symbol count for a configuration
        /// </summary>
        public int GetSymbolCount(int id)
        {
            var symbols = GetConfigurationSymbols(id);
            return symbols.Count;
        }
    }
}
