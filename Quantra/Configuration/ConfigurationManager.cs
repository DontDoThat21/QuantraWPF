using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Primitives;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Quantra.Configuration.Models;

namespace Quantra.Configuration
{
    /// <summary>
    /// Implementation of the configuration manager
    /// </summary>
    public class ConfigurationManager : IConfigurationManager, IDisposable
    {
        private const string DEFAULT_USER_SETTINGS_FILE = "usersettings.json";
        private const string BACKUP_FOLDER = "ConfigBackups";

        private readonly IConfiguration _configuration;
        private readonly IConfigurationBuilder _configurationBuilder;
        private readonly ConcurrentDictionary<string, object> _cachedSections = new ConcurrentDictionary<string, object>();
        private readonly ConcurrentDictionary<string, object> _pendingChanges = new ConcurrentDictionary<string, object>();
        private readonly ConcurrentDictionary<string, List<WeakReference>> _changeNotifications = new ConcurrentDictionary<string, List<WeakReference>>();
        private readonly ReaderWriterLockSlim _configLock = new ReaderWriterLockSlim();
        private readonly string _userSettingsFilePath;
        private readonly string _backupFolderPath;
        private IDisposable _changeTokenRegistration;

        /// <summary>
        /// Event that fires when a configuration value changes
        /// </summary>
        public event EventHandler<ConfigurationChangedEventArgs> ConfigurationChanged;

        /// <summary>
        /// Get the raw configuration object
        /// </summary>
        public IConfiguration RawConfiguration => _configuration;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="configuration">The configuration to wrap</param>
        public ConfigurationManager(IConfiguration configuration)
        {
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            // Initialize configuration builder
            _configurationBuilder = new ConfigurationBuilder();

            // If configuration is a root, copy its sources to the builder
            if (configuration is IConfigurationRoot configRoot)
            {
                foreach (var provider in configRoot.Providers)
                {
                    // We need to handle this specially because we can't directly access sources
                    // This will be used only when rebuilding configuration
                }
            }

            // Set up file paths
            var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            var appFolder = Path.Combine(appDataPath, "Quantra");

            // Ensure app folder exists
            if (!Directory.Exists(appFolder))
                Directory.CreateDirectory(appFolder);

            _userSettingsFilePath = Path.Combine(appFolder, DEFAULT_USER_SETTINGS_FILE);
            _backupFolderPath = Path.Combine(appFolder, BACKUP_FOLDER);

            // Ensure backup folder exists
            if (!Directory.Exists(_backupFolderPath))
                Directory.CreateDirectory(_backupFolderPath);

            // Register for config changes
            ChangeToken.OnChange(
                () => _configuration.GetReloadToken(),
                () => OnConfigurationChanged());
        }

        #region IConfigurationBuilder Implementation

        /// <summary>
        /// Gets the properties used to create an <see cref="IConfiguration"/>.
        /// </summary>
        public IDictionary<string, object> Properties => _configurationBuilder.Properties;

        /// <summary>
        /// Gets the sources used to build configuration.
        /// </summary>
        public IList<IConfigurationSource> Sources => _configurationBuilder.Sources;

        /// <summary>
        /// Adds a configuration source to the builder.
        /// </summary>
        /// <param name="source">The source to add.</param>
        /// <returns>The builder.</returns>
        public IConfigurationBuilder Add(IConfigurationSource source)
        {
            return _configurationBuilder.Add(source);
        }

        /// <summary>
        /// Builds an <see cref="IConfiguration"/> with the sources that were added.
        /// </summary>
        /// <returns>The built configuration.</returns>
        public IConfigurationRoot Build()
        {
            return _configurationBuilder.Build();
        }

        #endregion

        #region IConfiguration Implementation

        /// <summary>
        /// Gets a configuration value.
        /// </summary>
        /// <param name="key">The key of the configuration value.</param>
        /// <returns>The configuration value, or null if not found.</returns>
        public string this[string key]
        {
            get => _configuration[key];
            set => SetValue(key, value);
        }

        /// <summary>
        /// Gets the immediate children of this configuration element.
        /// </summary>
        /// <returns>The children.</returns>
        public IEnumerable<IConfigurationSection> GetChildren()
        {
            return _configuration.GetChildren();
        }

        /// <summary>
        /// Returns a <see cref="IChangeToken"/> that can be used to observe when this configuration is reloaded.
        /// </summary>
        /// <returns>A change token.</returns>
        public IChangeToken GetReloadToken()
        {
            return _configuration.GetReloadToken();
        }

        /// <summary>
        /// Gets a configuration section by key (IConfiguration implementation).
        /// </summary>
        /// <param name="key">The key of the section.</param>
        /// <returns>The configuration section.</returns>
        public IConfigurationSection GetSection(string key)
        {
            return _configuration.GetSection(key);
        }

        #endregion

        /// <summary>
        /// Finalizer
        /// </summary>
        ~ConfigurationManager()
        {
            Dispose(false);
        }

        /// <summary>
        /// Dispose method
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// Protected dispose method
        /// </summary>
        /// <param name="disposing">Whether to dispose managed resources</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                _changeTokenRegistration?.Dispose();
                _configLock.Dispose();
            }
        }

        /// <summary>
        /// Get a typed configuration section
        /// </summary>
        /// <typeparam name="T">The type to bind to</typeparam>
        /// <param name="sectionPath">The section path in the configuration</param>
        /// <returns>The typed configuration object</returns>
        public T GetSection<T>(string sectionPath) where T : class, new()
        {
            if (sectionPath == null)
                throw new ArgumentNullException(nameof(sectionPath));

            // Handle empty string as root level section - use special cache key
            string cacheKey = string.IsNullOrEmpty(sectionPath) ? "__ROOT__" : sectionPath;

            // Get from cache if available
            if (_cachedSections.TryGetValue(cacheKey, out var cached) && cached is T typedCached)
                return typedCached;

            _configLock.EnterReadLock();
            try
            {
                IConfigurationSection section;

                // For empty/null sectionPath, treat as root configuration
                if (string.IsNullOrEmpty(sectionPath))
                {
                    // For root configuration, we'll bind directly from _configuration
                    var instance = new T();
                    _configuration.Bind(instance);
                    _cachedSections[cacheKey] = instance;
                    return instance;
                }
                else
                {
                    section = _configuration.GetSection(sectionPath);

                    // Check if section exists
                    if (!section.Exists())
                    {
                        // If section doesn't exist, create new default
                        var defaultInstance = new T();
                        _cachedSections[cacheKey] = defaultInstance;
                        return defaultInstance;
                    }

                    // Bind section to new instance
                    var instance = new T();
                    section.Bind(instance);

                    // Cache the bound instance
                    _cachedSections[cacheKey] = instance;

                    return instance;
                }
            }
            finally
            {
                _configLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Get a specific configuration value
        /// </summary>
        /// <typeparam name="T">The type to convert the value to</typeparam>
        /// <param name="key">The configuration key</param>
        /// <param name="defaultValue">Default value to return if the key doesn't exist</param>
        /// <returns>The configuration value</returns>
        public T GetValue<T>(string key, T defaultValue = default)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentNullException(nameof(key));

            _configLock.EnterReadLock();
            try
            {
                return _configuration.GetValue<T>(key, defaultValue);
            }
            finally
            {
                _configLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Set a configuration value
        /// </summary>
        /// <typeparam name="T">The type of value to set</typeparam>
        /// <param name="key">The configuration key</param>
        /// <param name="value">The value to set</param>
        /// <param name="persist">Whether to persist the change immediately</param>
        public void SetValue<T>(string key, T value, bool persist = true)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentNullException(nameof(key));

            // Get old value for change notification
            var oldValue = GetValue<T>(key);

            _configLock.EnterWriteLock();
            try
            {
                // Store in the pending changes dictionary
                _pendingChanges[key] = value;

                // Clear any cached sections that contain this key
                foreach (var cachedKey in _cachedSections.Keys.ToList())
                {
                    if (key.StartsWith(cachedKey) || cachedKey.StartsWith(key))
                    {
                        _cachedSections.TryRemove(cachedKey, out _);
                    }
                }

                // Persist if requested
                if (persist)
                {
                    SaveChangesAsync().GetAwaiter().GetResult();
                }

                // Fire change notification
                OnConfigurationValueChanged(key, oldValue, value);
            }
            finally
            {
                _configLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Save all pending configuration changes
        /// </summary>
        /// <returns>Task representing the save operation</returns>
        public async Task SaveChangesAsync()
        {
            if (_pendingChanges.Count == 0)
                return;

            _configLock.EnterWriteLock();
            try
            {
                // Get current user settings or create new if it doesn't exist
                JObject userSettings = null;

                // Check if file exists and load it
                if (File.Exists(_userSettingsFilePath))
                {
                    var json = await File.ReadAllTextAsync(_userSettingsFilePath);
                    userSettings = JObject.Parse(json);
                }
                else
                {
                    userSettings = new JObject();
                }

                // Apply all pending changes
                foreach (var change in _pendingChanges)
                {
                    SetJsonValue(userSettings, change.Key, JToken.FromObject(change.Value));
                }

                // Clear pending changes
                _pendingChanges.Clear();

                // Save to file
                var serialized = userSettings.ToString(Formatting.Indented);
                await File.WriteAllTextAsync(_userSettingsFilePath, serialized);
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to save configuration changes", ex.ToString());
                throw;
            }
            finally
            {
                _configLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Set a value in a JSON object using hierarchical path
        /// </summary>
        /// <param name="rootObject">The root JSON object</param>
        /// <param name="path">The hierarchical path</param>
        /// <param name="value">The value to set</param>
        private void SetJsonValue(JObject rootObject, string path, JToken value)
        {
            if (rootObject == null)
                throw new ArgumentNullException(nameof(rootObject));

            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentNullException(nameof(path));

            // Split path into segments
            var segments = path.Split(':');

            JObject current = rootObject;

            // Navigate to the parent object
            for (int i = 0; i < segments.Length - 1; i++)
            {
                var segment = segments[i];

                if (!current.ContainsKey(segment))
                {
                    // Create new object if it doesn't exist
                    var newObject = new JObject();
                    current[segment] = newObject;
                    current = newObject;
                }
                else
                {
                    var token = current[segment];

                    // Convert to object if not already
                    if (token.Type != JTokenType.Object)
                    {
                        var newObject = new JObject();
                        current[segment] = newObject;
                        current = newObject;
                    }
                    else
                    {
                        current = (JObject)token;
                    }
                }
            }

            // Set value on the last segment
            current[segments.Last()] = value;
        }

        /// <summary>
        /// Reload the configuration from all sources
        /// </summary>
        /// <returns>Task representing the reload operation</returns>
        public Task ReloadAsync()
        {
            _configLock.EnterWriteLock();
            try
            {
                // Clear caches
                _cachedSections.Clear();

                // Force reload of configuration
                if (_configuration is IConfigurationRoot configRoot)
                {
                    configRoot.Reload();
                }

                return Task.CompletedTask;
            }
            finally
            {
                _configLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Handler for configuration changes
        /// </summary>
        private void OnConfigurationChanged()
        {
            _configLock.EnterWriteLock();
            try
            {
                // Clear cached sections
                _cachedSections.Clear();

                // Notify subscribers of changes
                NotifyChangeSubscribers();
            }
            finally
            {
                _configLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Handler for specific configuration value changes
        /// </summary>
        /// <param name="key">The key that changed</param>
        /// <param name="oldValue">The old value</param>
        /// <param name="newValue">The new value</param>
        private void OnConfigurationValueChanged(string key, object oldValue, object newValue)
        {
            var args = new ConfigurationChangedEventArgs(key, oldValue, newValue);
            ConfigurationChanged?.Invoke(this, args);

            // Notify subscribers for this specific path
            NotifyChangeSubscribers(key);
        }

        /// <summary>
        /// Register a configuration object for automatic change notifications
        /// </summary>
        /// <param name="sectionPath">The section path to register for</param>
        /// <param name="instance">The instance to update on changes</param>
        public void RegisterChangeNotifications<T>(string sectionPath, T instance) where T : class, INotifyPropertyChanged
        {
            if (sectionPath == null)
                throw new ArgumentNullException(nameof(sectionPath));

            if (instance == null)
                throw new ArgumentNullException(nameof(instance));

            // Handle empty string as root level section - use special cache key for consistency
            string cacheKey = string.IsNullOrEmpty(sectionPath) ? "__ROOT__" : sectionPath;

            _configLock.EnterWriteLock();
            try
            {
                // Get or create notification list for this section
                if (!_changeNotifications.TryGetValue(cacheKey, out var notifications))
                {
                    notifications = new List<WeakReference>();
                    _changeNotifications[cacheKey] = notifications;
                }

                // Add weak reference to the instance
                notifications.Add(new WeakReference(instance));
            }
            finally
            {
                _configLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Notify change subscribers
        /// </summary>
        /// <param name="changedPath">Optional specific path that changed</param>
        private void NotifyChangeSubscribers(string changedPath = null)
        {
            _configLock.EnterReadLock();
            try
            {
                foreach (var kvp in _changeNotifications)
                {
                    // Skip if we have a specific path and it's not related to this subscription
                    if (!string.IsNullOrEmpty(changedPath) &&
                        !changedPath.StartsWith(kvp.Key) &&
                        !kvp.Key.StartsWith(changedPath))
                        continue;

                    var path = kvp.Key;
                    var references = kvp.Value;

                    // Clean up dead references while notifying live ones
                    var deadRefs = new List<WeakReference>();

                    foreach (var weakRef in references)
                    {
                        if (!weakRef.IsAlive)
                        {
                            deadRefs.Add(weakRef);
                            continue;
                        }

                        var target = weakRef.Target;
                        if (target != null)
                        {
                            // Rebind the section to update the target
                            var section = _configuration.GetSection(path);

                            // Use reflection to get all property setters
                            var properties = target.GetType().GetProperties()
                                .Where(p => p.CanWrite && p.CanRead)
                                .ToList();

                            // For each property, try to get the value from config
                            foreach (var prop in properties)
                            {
                                var configValue = section[prop.Name];
                                if (configValue != null)
                                {
                                    try
                                    {
                                        var typedValue = Convert.ChangeType(configValue, prop.PropertyType);
                                        prop.SetValue(target, typedValue);
                                    }
                                    catch (Exception)
                                    {
                                        // Skip this property if conversion fails
                                    }
                                }
                            }
                        }
                        else
                        {
                            deadRefs.Add(weakRef);
                        }
                    }

                    // Remove dead references
                    if (deadRefs.Count > 0)
                    {
                        _configLock.ExitReadLock();
                        _configLock.EnterWriteLock();
                        try
                        {
                            foreach (var deadRef in deadRefs)
                            {
                                kvp.Value.Remove(deadRef);
                            }
                        }
                        finally
                        {
                            _configLock.ExitWriteLock();
                            _configLock.EnterReadLock();
                        }
                    }
                }
            }
            finally
            {
                _configLock.ExitReadLock();
            }
        }

        /// <summary>
        /// Create a backup of the current configuration
        /// </summary>
        /// <returns>The path to the backup file</returns>
        public string BackupConfiguration()
        {
            try
            {
                // Create backup filename with timestamp
                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var backupFilename = $"config_backup_{timestamp}.json";
                var backupPath = Path.Combine(_backupFolderPath, backupFilename);

                // Ensure we have user settings to backup
                if (File.Exists(_userSettingsFilePath))
                {
                    File.Copy(_userSettingsFilePath, backupPath, true);
                    //DatabaseMonolith.Log("Info", $"Configuration backup created at {backupPath}");
                    return backupPath;
                }

                // If no user settings yet, create an empty backup
                File.WriteAllText(backupPath, "{}");
                return backupPath;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to create configuration backup", ex.ToString());
                return null;
            }
        }

        /// <summary>
        /// Restore configuration from a backup file
        /// </summary>
        /// <param name="backupPath">Path to the backup file</param>
        /// <returns>True if successful, false otherwise</returns>
        public bool RestoreConfigurationFromBackup(string backupPath)
        {
            try
            {
                if (!File.Exists(backupPath))
                {
                    //DatabaseMonolith.Log("Error", $"Configuration backup file not found: {backupPath}");
                    return false;
                }

                // Create a backup of the current configuration before restoring
                BackupConfiguration();

                // Restore from backup
                File.Copy(backupPath, _userSettingsFilePath, true);

                // Reload configuration
                ReloadAsync().GetAwaiter().GetResult();

                //DatabaseMonolith.Log("Info", $"Configuration restored from backup: {backupPath}");
                return true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to restore configuration from backup", ex.ToString());
                return false;
            }
        }

        /// <summary>
        /// Reset to default configuration
        /// </summary>
        /// <param name="sectionPath">Optional section path to reset, null for entire configuration</param>
        public void ResetToDefaults(string sectionPath = null)
        {
            try
            {
                _configLock.EnterWriteLock();
                try
                {
                    if (string.IsNullOrEmpty(sectionPath))
                    {
                        // Reset entire configuration
                        if (File.Exists(_userSettingsFilePath))
                        {
                            // Create a backup before resetting
                            BackupConfiguration();

                            // Delete user settings file
                            File.Delete(_userSettingsFilePath);
                        }
                    }
                    else
                    {
                        // Reset specific section
                        if (File.Exists(_userSettingsFilePath))
                        {
                            // Create a backup before resetting
                            BackupConfiguration();

                            // Load user settings
                            var json = File.ReadAllText(_userSettingsFilePath);
                            var userSettings = JObject.Parse(json);

                            // Remove the section
                            var segments = sectionPath.Split(':');
                            JToken current = userSettings;

                            for (int i = 0; i < segments.Length - 1; i++)
                            {
                                if (!(current is JObject currentObj))
                                    break;

                                if (!currentObj.ContainsKey(segments[i]))
                                    break;

                                current = currentObj[segments[i]];
                            }

                            if (current is JObject currentObj2 && currentObj2.ContainsKey(segments.Last()))
                            {
                                currentObj2.Remove(segments.Last());

                                // Save updated settings
                                var serialized = userSettings.ToString(Formatting.Indented);
                                File.WriteAllText(_userSettingsFilePath, serialized);
                            }
                        }
                    }

                    // Clear caches
                    _cachedSections.Clear();
                    _pendingChanges.Clear();

                    // Force configuration reload
                    ReloadAsync().GetAwaiter().GetResult();

                    //DatabaseMonolith.Log("Info", $"Configuration reset to defaults{(string.IsNullOrEmpty(sectionPath) ? "" : $" for section {sectionPath}")}");
                }
                finally
                {
                    _configLock.ExitWriteLock();
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to reset configuration to defaults", ex.ToString());
                throw;
            }
        }
    }
}