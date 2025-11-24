using System;
using System.Collections.Generic;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Primitives;

namespace Quantra.Configuration.Providers
{
    /// <summary>
    /// Environment variables configuration provider for sensitive configuration values
    /// </summary>
    public class EnvironmentConfigProvider : IConfigurationProvider, IDisposable
    {
        private const string PREFIX = "QUANTRA_";
        private readonly Dictionary<string, string> _data = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        /// <summary>
        /// Constructor
        /// </summary>
        public EnvironmentConfigProvider()
        {
            LoadData();
        }

        /// <summary>
        /// Get a configuration value with key
        /// </summary>
        /// <param name="key">The key to get the value for</param>
        /// <param name="value">The value if found</param>
        /// <returns>Whether the key exists</returns>
        public bool TryGet(string key, out string value)
        {
            return _data.TryGetValue(key, out value);
        }

        /// <summary>
        /// Set a configuration value
        /// </summary>
        /// <param name="key">The key to set</param>
        /// <param name="value">The value to set</param>
        public void Set(string key, string value)
        {
            _data[key] = value;
        }

        /// <summary>
        /// Get the children of a given key
        /// </summary>
        /// <param name="earlierKeys">Earlier key segments</param>
        /// <param name="parentPath">The parent path</param>
        /// <returns>The child keys</returns>
        public IEnumerable<string> GetChildKeys(IEnumerable<string> earlierKeys, string? parentPath)
        {
            var prefix = parentPath == null ? string.Empty : parentPath + ConfigurationPath.KeyDelimiter;
            var results = new List<string>();

            foreach (var pair in _data)
            {
                if (pair.Key.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    var key = pair.Key.Substring(prefix.Length);
                    var keySegment = key;

                    // If the key contains a delimiter, only take the first segment
                    var delimiterIndex = key.IndexOf(ConfigurationPath.KeyDelimiter, StringComparison.OrdinalIgnoreCase);
                    if (delimiterIndex >= 0)
                    {
                        keySegment = key.Substring(0, delimiterIndex);
                    }

                    if (!results.Contains(keySegment))
                    {
                        results.Add(keySegment);
                    }
                }
            }

            results.AddRange(earlierKeys);
            results.Sort(StringComparer.OrdinalIgnoreCase);
            return results;
        }

        /// <summary>
        /// Reload the configuration data
        /// </summary>
        public void Load()
        {
            LoadData();
        }

        /// <summary>
        /// Get a change token that signals when the configuration changes
        /// </summary>
        /// <returns>A change token</returns>
        public IChangeToken GetReloadToken()
        {
            return new ConfigurationReloadToken();
        }

        /// <summary>
        /// Load data from environment variables
        /// </summary>
        private void LoadData()
        {
            _data.Clear();

            foreach (var variable in Environment.GetEnvironmentVariables().Keys)
            {
                var key = variable.ToString();

                // Only process our prefixed environment variables
                if (key.StartsWith(PREFIX, StringComparison.OrdinalIgnoreCase))
                {
                    // Remove prefix and normalize
                    var normalizedKey = key.Substring(PREFIX.Length)
                        .Replace("__", ConfigurationPath.KeyDelimiter)
                        .Replace("_", ConfigurationPath.KeyDelimiter);

                    // Store in dictionary
                    var value = Environment.GetEnvironmentVariable(key);
                    if (!string.IsNullOrEmpty(value))
                    {
                        _data[normalizedKey] = value;
                    }
                }
            }
        }

        /// <summary>
        /// Dispose the provider
        /// </summary>
        public void Dispose()
        {
            // Nothing to dispose
        }
    }

    /// <summary>
    /// Configuration source for environment variables
    /// </summary>
    public class EnvironmentConfigSource : IConfigurationSource
    {
        /// <summary>
        /// Build the configuration provider
        /// </summary>
        /// <param name="builder">The configuration builder</param>
        /// <returns>The configuration provider</returns>
        public IConfigurationProvider Build(IConfigurationBuilder builder)
        {
            return new EnvironmentConfigProvider();
        }
    }

    /// <summary>
    /// Extension methods for configuration builder
    /// </summary>
    public static class EnvironmentConfigExtensions
    {
        /// <summary>
        /// Add Quantra environment variables configuration source
        /// </summary>
        /// <param name="builder">The configuration builder</param>
        /// <returns>The configuration builder</returns>
        public static IConfigurationBuilder AddQuantraEnvironmentVariables(this IConfigurationBuilder builder)
        {
            return builder.Add(new EnvironmentConfigSource());
        }
    }
}