using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using Microsoft.Extensions.Configuration;
using Quantra.ViewModels.Base;
using Quantra.Configuration;
using Quantra.Configuration.Models;
using Quantra.Commands;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for configuration control
    /// </summary>
    public class ConfigurationViewModel : ViewModelBase
    {
        private readonly Quantra.Configuration.IConfigurationManager _configManager;
        private readonly DatabaseConfigBridge _configBridge;
        private readonly AppConfig _appConfig;
        private bool _isDirty = false;
        private int _selectedSettingsTabIndex; // stores selected tab index for the UI
        
        /// <summary>
        /// Currently selected tab index in Configuration view
        /// </summary>
        public int SelectedSettingsTabIndex
        {
            get => _selectedSettingsTabIndex;
            set 
            {
                if (SetProperty(ref _selectedSettingsTabIndex, value))
                {
                    //DatabaseMonolith.Log("Info", $"ViewModel SelectedSettingsTabIndex changed to: {value}");
                }
            }
        }
        
        /// <summary>
        /// API configuration
        /// </summary>
        public ApiConfig Api 
        { 
            get => _appConfig.Api;
            set
            {
                if (_appConfig.Api != value)
                {
                    _appConfig.Api = value;
                    OnPropertyChanged();
                    IsDirty = true;
                }
            }
        }
        
        /// <summary>
        /// Cache configuration
        /// </summary>
        public CacheConfig Cache
        {
            get => _appConfig.Cache;
            set
            {
                if (_appConfig.Cache != value)
                {
                    _appConfig.Cache = value;
                    OnPropertyChanged();
                    IsDirty = true;
                }
            }
        }
        
        /// <summary>
        /// UI configuration
        /// </summary>
        public UIConfig UI
        {
            get => _appConfig.UI;
            set
            {
                if (_appConfig.UI != value)
                {
                    _appConfig.UI = value;
                    OnPropertyChanged();
                    IsDirty = true;
                }
            }
        }
        
        /// <summary>
        /// Notification configuration
        /// </summary>
        public NotificationConfig Notifications
        {
            get => _appConfig.Notifications;
            set
            {
                if (_appConfig.Notifications != value)
                {
                    _appConfig.Notifications = value;
                    OnPropertyChanged();
                    IsDirty = true;
                }
            }
        }
        
        /// <summary>
        /// Trading configuration
        /// </summary>
        public TradingConfig Trading
        {
            get => _appConfig.Trading;
            set
            {
                if (_appConfig.Trading != value)
                {
                    _appConfig.Trading = value;
                    OnPropertyChanged();
                    IsDirty = true;
                }
            }
        }
        
        /// <summary>
        /// Sentiment analysis configuration
        /// </summary>
        public SentimentAnalysisConfig SentimentAnalysis
        {
            get => _appConfig.SentimentAnalysis;
            set
            {
                if (_appConfig.SentimentAnalysis != value)
                {
                    _appConfig.SentimentAnalysis = value;
                    OnPropertyChanged();
                    IsDirty = true;
                }
            }
        }
        
        /// <summary>
        /// Flag indicating whether there are unsaved changes
        /// </summary>
        public bool IsDirty
        {
            get => _isDirty;
            set => SetProperty(ref _isDirty, value);
        }
        
        /// <summary>
        /// Command to save configuration changes
        /// </summary>
        public ICommand SaveCommand { get; }
        
        /// <summary>
        /// Command to revert changes
        /// </summary>
        public ICommand RevertCommand { get; }
        
        /// <summary>
        /// Command to reset section to defaults
        /// </summary>
        public ICommand ResetSectionCommand { get; }
        
        /// <summary>
        /// Command to backup configuration
        /// </summary>
        public ICommand BackupCommand { get; }
        
        /// <summary>
        /// Command to restore configuration
        /// </summary>
        public ICommand RestoreCommand { get; }
        
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="configManager">Configuration manager</param>
        /// <param name="configBridge">Database configuration bridge</param>
        public ConfigurationViewModel(Quantra.Configuration.IConfigurationManager configManager, DatabaseConfigBridge configBridge)
        {
            _configManager = configManager ?? throw new ArgumentNullException(nameof(configManager));
            _configBridge = configBridge ?? throw new ArgumentNullException(nameof(configBridge));
            
            // Get app configuration
            _appConfig = configManager.GetSection<AppConfig>("");
            
            // Define commands
            SaveCommand = new RelayCommand(_ => Save(), _ => CanSave());
            RevertCommand = new RelayCommand(_ => Revert(), _ => CanRevert());
            ResetSectionCommand = new RelayCommand<string>(ResetSection);
            BackupCommand = new RelayCommand(_ => Backup());
            RestoreCommand = new RelayCommand<string>(Restore);
            
            // Listen for property changes to mark as dirty
            PropertyChanged += (sender, args) =>
            {
                // Changing the selected tab should not mark the config dirty
                if (args.PropertyName != nameof(IsDirty) && args.PropertyName != nameof(SelectedSettingsTabIndex))
                {
                    IsDirty = true;
                }
            };
            
            // Register for config changes
            _configManager.ConfigurationChanged += (sender, args) =>
            {
                // Refresh UI when configuration changes externally
                OnPropertyChanged(string.Empty);
            };
        }
        
        /// <summary>
        /// Save changes
        /// </summary>
        private void Save()
        {
            try
            {
                _configManager.SaveChangesAsync().GetAwaiter().GetResult();
                _configBridge.SyncConfigToDatabase();
                IsDirty = false;
                
                // Log successful save
                //DatabaseMonolith.Log("Info", "Configuration saved successfully");
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to save configuration", ex.ToString());
            }
        }
        
        /// <summary>
        /// Whether save can be executed
        /// </summary>
        /// <returns>True if can save</returns>
        private bool CanSave()
        {
            return IsDirty;
        }
        
        /// <summary>
        /// Revert changes
        /// </summary>
        private void Revert()
        {
            try
            {
                // Reload configuration
                _configManager.ReloadAsync().GetAwaiter().GetResult();
                
                // Refresh all properties
                OnPropertyChanged(string.Empty);
                
                IsDirty = false;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to revert configuration changes", ex.ToString());
            }
        }
        
        /// <summary>
        /// Whether revert can be executed
        /// </summary>
        /// <returns>True if can revert</returns>
        private bool CanRevert()
        {
            return IsDirty;
        }
        
        /// <summary>
        /// Reset section to defaults
        /// </summary>
        /// <param name="sectionPath">The section path</param>
        private void ResetSection(string sectionPath)
        {
            try
            {
                // Reset section to defaults
                _configManager.ResetToDefaults(sectionPath);
                
                // Refresh properties
                OnPropertyChanged(string.Empty);
                
                IsDirty = false;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to reset configuration section", ex.ToString());
            }
        }
        
        /// <summary>
        /// Backup configuration
        /// </summary>
        private void Backup()
        {
            try
            {
                string backupPath = _configManager.BackupConfiguration();
                if (!string.IsNullOrEmpty(backupPath))
                {
                    //DatabaseMonolith.Log("Info", $"Configuration backed up to: {backupPath}");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to backup configuration", ex.ToString());
            }
        }
        
        /// <summary>
        /// Restore configuration from backup
        /// </summary>
        /// <param name="backupPath">Path to backup file</param>
        private void Restore(string backupPath)
        {
            try
            {
                if (_configManager.RestoreConfigurationFromBackup(backupPath))
                {
                    // Refresh all properties
                    OnPropertyChanged(string.Empty);
                    
                    IsDirty = false;
                    
                    //DatabaseMonolith.Log("Info", "Configuration restored from backup");
                }
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", "Failed to restore configuration from backup", ex.ToString());
            }
        }
    }
}