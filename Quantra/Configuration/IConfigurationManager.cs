using System;
using Microsoft.Extensions.Configuration;
using System.ComponentModel;
using System.Threading.Tasks;

namespace Quantra.Configuration
{
    /// <summary>
    /// Interface for the configuration manager, extending Microsoft's IConfigurationManager
    /// </summary>
    public interface IConfigurationManager : Microsoft.Extensions.Configuration.IConfigurationManager
    {
        /// <summary>
        /// Event that fires when a configuration value changes
        /// </summary>
        event EventHandler<ConfigurationChangedEventArgs> ConfigurationChanged;

        /// <summary>
        /// Get the raw configuration object
        /// </summary>
        IConfiguration RawConfiguration { get; }

        /// <summary>
        /// Get a typed configuration section
        /// </summary>
        /// <typeparam name="T">The type to bind to</typeparam>
        /// <param name="sectionPath">The section path in the configuration</param>
        /// <returns>The typed configuration object</returns>
        T GetSection<T>(string sectionPath) where T : class, new();

        /// <summary>
        /// Get a specific configuration value
        /// </summary>
        /// <typeparam name="T">The type to convert the value to</typeparam>
        /// <param name="key">The configuration key</param>
        /// <param name="defaultValue">Default value to return if the key doesn't exist</param>
        /// <returns>The configuration value</returns>
        T GetValue<T>(string key, T defaultValue = default);

        /// <summary>
        /// Set a configuration value
        /// </summary>
        /// <typeparam name="T">The type of value to set</typeparam>
        /// <param name="key">The configuration key</param>
        /// <param name="value">The value to set</param>
        /// <param name="persist">Whether to persist the change immediately</param>
        void SetValue<T>(string key, T value, bool persist = true);

        /// <summary>
        /// Save all pending configuration changes
        /// </summary>
        /// <returns>Task representing the save operation</returns>
        Task SaveChangesAsync();

        /// <summary>
        /// Reload the configuration from all sources
        /// </summary>
        /// <returns>Task representing the reload operation</returns>
        Task ReloadAsync();

        /// <summary>
        /// Register a configuration object for automatic change notifications
        /// </summary>
        /// <param name="sectionPath">The section path to register for</param>
        /// <param name="instance">The instance to update on changes</param>
        void RegisterChangeNotifications<T>(string sectionPath, T instance) where T : class, INotifyPropertyChanged;

        /// <summary>
        /// Create a backup of the current configuration
        /// </summary>
        /// <returns>The path to the backup file</returns>
        string BackupConfiguration();

        /// <summary>
        /// Restore configuration from a backup file
        /// </summary>
        /// <param name="backupPath">Path to the backup file</param>
        /// <returns>True if successful, false otherwise</returns>
        bool RestoreConfigurationFromBackup(string backupPath);

        /// <summary>
        /// Reset to default configuration
        /// </summary>
        /// <param name="sectionPath">Optional section path to reset, null for entire configuration</param>
        void ResetToDefaults(string sectionPath = null);
    }

    /// <summary>
    /// Event arguments for configuration changes
    /// </summary>
    public class ConfigurationChangedEventArgs : EventArgs
    {
        /// <summary>
        /// The key that changed
        /// </summary>
        public string Key { get; }

        /// <summary>
        /// The old value
        /// </summary>
        public object OldValue { get; }

        /// <summary>
        /// The new value
        /// </summary>
        public object NewValue { get; }

        /// <summary>
        /// Constructor
        /// </summary>
        public ConfigurationChangedEventArgs(string key, object oldValue, object newValue)
        {
            Key = key;
            OldValue = oldValue;
            NewValue = newValue;
        }
    }
}