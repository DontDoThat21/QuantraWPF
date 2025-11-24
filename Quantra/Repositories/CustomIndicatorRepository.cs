using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Quantra.Models;

namespace Quantra.Repositories
{
    /// <summary>
    /// Repository for storing and retrieving custom indicator definitions
    /// </summary>
    public class CustomIndicatorRepository
    {
        private readonly string _indicatorsDirectory;
        private readonly Dictionary<string, CustomIndicatorDefinition> _cache;
        private static readonly JsonSerializerOptions _options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public CustomIndicatorRepository()
        {
            // Store indicator definitions in the application's data directory
            var appDataPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "Quantra");

            _indicatorsDirectory = Path.Combine(appDataPath, "CustomIndicators");
            _cache = new Dictionary<string, CustomIndicatorDefinition>();

            // Ensure the directory exists
            if (!Directory.Exists(_indicatorsDirectory))
            {
                Directory.CreateDirectory(_indicatorsDirectory);
            }
        }

        /// <summary>
        /// Get a custom indicator definition by ID
        /// </summary>
        /// <param name="id">The indicator ID</param>
        /// <returns>The indicator definition or null if not found</returns>
        public async Task<CustomIndicatorDefinition> GetIndicatorAsync(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentNullException(nameof(id));

            if (_cache.TryGetValue(id, out var cachedDefinition))
                return cachedDefinition;

            var filePath = GetFilePath(id);
            if (!File.Exists(filePath))
                return null;

            try
            {
                using var stream = File.OpenRead(filePath);
                var definition = await JsonSerializer.DeserializeAsync<CustomIndicatorDefinition>(
                    stream, _options);

                if (definition != null)
                {
                    _cache[id] = definition;
                    return definition;
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to load indicator definition {id}", ex.ToString());
            }

            return null;
        }

        /// <summary>
        /// Save a custom indicator definition
        /// </summary>
        /// <param name="definition">The indicator definition to save</param>
        /// <returns>True if successful</returns>
        public async Task<bool> SaveIndicatorAsync(CustomIndicatorDefinition definition)
        {
            if (definition == null)
                throw new ArgumentNullException(nameof(definition));

            if (string.IsNullOrWhiteSpace(definition.Id))
                definition.Id = Guid.NewGuid().ToString();

            definition.ModifiedDate = DateTime.Now;

            var filePath = GetFilePath(definition.Id);
            try
            {
                using var stream = File.Create(filePath);
                await JsonSerializer.SerializeAsync(stream, definition, _options);

                // Update cache
                _cache[definition.Id] = definition;
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to save indicator definition {definition.Id}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Delete a custom indicator definition
        /// </summary>
        /// <param name="id">The indicator ID to delete</param>
        /// <returns>True if successful</returns>
        public bool DeleteIndicator(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentNullException(nameof(id));

            var filePath = GetFilePath(id);
            try
            {
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                    _cache.Remove(id);
                    return true;
                }
                return false;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Failed to delete indicator definition {id}", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Get all custom indicator definitions
        /// </summary>
        /// <returns>List of all custom indicator definitions</returns>
        public async Task<List<CustomIndicatorDefinition>> GetAllIndicatorsAsync()
        {
            var indicators = new List<CustomIndicatorDefinition>();

            try
            {
                // Load all indicator definition files
                var files = Directory.GetFiles(_indicatorsDirectory, "*.json");

                foreach (var file in files)
                {
                    try
                    {
                        using var stream = File.OpenRead(file);
                        var definition = await JsonSerializer.DeserializeAsync<CustomIndicatorDefinition>(
                            stream, _options);

                        if (definition != null)
                        {
                            indicators.Add(definition);
                            _cache[definition.Id] = definition;
                        }
                    }
                    catch (Exception ex)
                    {
                        //DatabaseMonolith.Log("Error", $"Failed to load indicator definition file {file}", ex.ToString());
                    }
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to list indicator definition files", ex.ToString());
            }

            return indicators;
        }

        /// <summary>
        /// Search for indicator definitions matching certain criteria
        /// </summary>
        /// <param name="searchTerm">Text to search in name and description</param>
        /// <param name="category">Optional category filter</param>
        /// <returns>List of matching indicator definitions</returns>
        public async Task<List<CustomIndicatorDefinition>> SearchIndicatorsAsync(string searchTerm, string category = null)
        {
            var allIndicators = await GetAllIndicatorsAsync();

            // Filter by search term and category if provided
            return allIndicators
                .Where(i => (string.IsNullOrWhiteSpace(searchTerm) ||
                             i.Name.Contains(searchTerm, StringComparison.OrdinalIgnoreCase) ||
                             i.Description.Contains(searchTerm, StringComparison.OrdinalIgnoreCase)) &&
                            (string.IsNullOrWhiteSpace(category) ||
                             i.Category.Equals(category, StringComparison.OrdinalIgnoreCase)))
                .ToList();
        }

        /// <summary>
        /// Get the file path for an indicator definition
        /// </summary>
        /// <param name="id">The indicator ID</param>
        /// <returns>The full file path</returns>
        private string GetFilePath(string id)
        {
            return Path.Combine(_indicatorsDirectory, $"{id}.json");
        }

        /// <summary>
        /// Check if an indicator with the given ID exists
        /// </summary>
        /// <param name="id">The indicator ID to check</param>
        /// <returns>True if the indicator exists</returns>
        public bool IndicatorExists(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                return false;

            return File.Exists(GetFilePath(id));
        }
    }
}