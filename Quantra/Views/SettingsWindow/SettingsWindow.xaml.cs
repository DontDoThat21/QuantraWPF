using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using Quantra.DAL.Services;

namespace Quantra
{
    public partial class SettingsWindow : Window
    {
        private List<DatabaseSettingsProfile> _profiles;
        private DatabaseSettingsProfile _selectedProfile;
        private SettingsService _settingsService;
        private bool _isLoading = false;
        private DispatcherTimer _saveTimer;
        private readonly System.Windows.Media.BrushConverter _brushConverter = new System.Windows.Media.BrushConverter();

        // Parameterless constructor for XAML and legacy code compatibility
        public SettingsWindow()
        {
            // Get SettingsService from DI container
            _settingsService = App.ServiceProvider?.GetService<SettingsService>()
                ?? App.ServiceProvider?.GetService<ISettingsService>() as SettingsService;

            if (_settingsService == null)
            {
                throw new InvalidOperationException("SettingsService is not registered in the DI container or ServiceProvider is not initialized.");
            }

            InitializeWindow();
        }

        // Constructor for dependency injection
        public SettingsWindow(SettingsService settingsService)
        {
            _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
            InitializeWindow();
        }

        // Constructor for dependency injection with interface
        public SettingsWindow(ISettingsService settingsService)
        {
            _settingsService = settingsService as SettingsService ??
                throw new ArgumentException("SettingsService implementation required", nameof(settingsService));
            InitializeWindow();
        }

        private void InitializeWindow()
        {
            InitializeComponent();
            InitializeSaveTimer();
            // Use async loading for profiles
            _ = LoadProfilesAsync();
            AttachEventHandlers();
            Closing += SettingsWindow_Closing;
        }

        private void SettingsWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            // Save any pending changes before closing
            if (_saveTimer.IsEnabled)
            {
                _saveTimer.Stop();
                SaveCurrentSettings();
            }

            // Clean up the timer
        }

        private void InitializeSaveTimer()
        {
            _saveTimer = new DispatcherTimer();
            _saveTimer.Interval = TimeSpan.FromMilliseconds(1000); // 1 second debounce
            _saveTimer.Tick += SaveTimer_Tick;
        }

        private void AttachEventHandlers()
        {
            // CheckBox event handlers
            EnableApiModalChecks.Checked += OnSettingChanged;
            EnableApiModalChecks.Unchecked += OnSettingChanged;
            EnableHistoricalDataCache.Checked += OnSettingChanged;
            EnableHistoricalDataCache.Unchecked += OnSettingChanged;
            EnableDarkMode.Checked += OnSettingChanged;
            EnableDarkMode.Unchecked += OnSettingChanged;
            RememberWindowState.Checked += OnSettingChanged;
            RememberWindowState.Unchecked += OnSettingChanged;
            EnableVixMonitoring.Checked += OnSettingChanged;
            EnableVixMonitoring.Unchecked += OnSettingChanged;
            EnablePriceAlerts.Checked += OnSettingChanged;
            EnablePriceAlerts.Unchecked += OnSettingChanged;
            EnableTradeNotifications.Checked += OnSettingChanged;
            EnableTradeNotifications.Unchecked += OnSettingChanged;
            EnablePaperTrading.Checked += OnSettingChanged;
            EnablePaperTrading.Unchecked += OnSettingChanged;
            EnableEmailAlerts.Checked += OnSettingChanged;
            EnableEmailAlerts.Unchecked += OnSettingChanged;
            EnableStandardAlertEmails.Checked += OnSettingChanged;
            EnableStandardAlertEmails.Unchecked += OnSettingChanged;
            EnableOpportunityAlertEmails.Checked += OnSettingChanged;
            EnableOpportunityAlertEmails.Unchecked += OnSettingChanged;
            EnablePredictionAlertEmails.Checked += OnSettingChanged;
            EnablePredictionAlertEmails.Unchecked += OnSettingChanged;
            EnableGlobalAlertEmails.Checked += OnSettingChanged;
            EnableGlobalAlertEmails.Unchecked += OnSettingChanged;

            // TextBox event handlers
            ApiTimeoutTextBox.TextChanged += OnSettingChanged;
            CacheDurationTextBox.TextChanged += OnSettingChanged;
            ChartUpdateIntervalTextBox.TextChanged += OnSettingChanged;
            DefaultGridRowsTextBox.TextChanged += OnSettingChanged;
            DefaultGridColumnsTextBox.TextChanged += OnSettingChanged;
            AlertEmailTextBox.TextChanged += OnSettingChanged;

            // PasswordBox event handlers
            // ComboBox event handlers
            RiskLevelComboBox.SelectionChanged += OnSettingChanged;
            GridBorderColorComboBox.SelectionChanged += OnSettingChanged;
            AlphaVantageApiPlanComboBox.SelectionChanged += OnSettingChanged;
            AlphaVantageDataEntitlementComboBox.SelectionChanged += OnSettingChanged;
        }

        private async Task LoadProfilesAsync()
        {
            _isLoading = true;
            _profiles = _settingsService.GetAllSettingsProfiles();
            ProfilesListView.ItemsSource = _profiles;
            if (_profiles.Count > 0)
            {
                var defaultProfile = await _settingsService.GetDefaultSettingsProfileAsync();
                if (defaultProfile == null)
                    defaultProfile = _profiles[0];
                ProfilesListView.SelectedItem = defaultProfile;
                LoadProfileToUI(defaultProfile);
            }
            _isLoading = false;
        }

        private void LoadProfileToUI(DatabaseSettingsProfile profile)
        {
            if (profile == null) return;
            _selectedProfile = profile;
            EnableApiModalChecks.IsChecked = profile.EnableApiModalChecks;
            ApiTimeoutTextBox.Text = profile.ApiTimeoutSeconds.ToString();
            CacheDurationTextBox.Text = profile.CacheDurationMinutes.ToString();
            EnableHistoricalDataCache.IsChecked = profile.EnableHistoricalDataCache;
            EnableDarkMode.IsChecked = profile.EnableDarkMode;
            RememberWindowState.IsChecked = profile.RememberWindowState;
            ChartUpdateIntervalTextBox.Text = profile.ChartUpdateIntervalSeconds.ToString();
            DefaultGridRowsTextBox.Text = profile.DefaultGridRows.ToString();
            DefaultGridColumnsTextBox.Text = profile.DefaultGridColumns.ToString();
            EnablePriceAlerts.IsChecked = profile.EnablePriceAlerts;
            EnableTradeNotifications.IsChecked = profile.EnableTradeNotifications;
            EnablePaperTrading.IsChecked = profile.EnablePaperTrading;
            RiskLevelComboBox.SelectedItem = RiskLevelComboBox.Items.Cast<ComboBoxItem>().FirstOrDefault(i => (string)i.Content == profile.RiskLevel);
            // Set grid border color
            foreach (ComboBoxItem item in GridBorderColorComboBox.Items)
            {
                if ((string)item.Tag == profile.GridBorderColor)
                {
                    GridBorderColorComboBox.SelectedItem = item;
                    ColorPreview.Fill = (System.Windows.Media.Brush)new System.Windows.Media.BrushConverter().ConvertFromString(profile.GridBorderColor);
                    break;
                }
            }
            // Email alert settings
            EnableEmailAlerts.IsChecked = profile.EnableEmailAlerts;
            EnableStandardAlertEmails.IsChecked = profile.EnableStandardAlertEmails;
            EnableOpportunityAlertEmails.IsChecked = profile.EnableOpportunityAlertEmails;
            EnablePredictionAlertEmails.IsChecked = profile.EnablePredictionAlertEmails;
            EnableGlobalAlertEmails.IsChecked = profile.EnableGlobalAlertEmails;
            AlertEmailTextBox.Text = profile.AlertEmail;
            // VIX monitoring setting
            EnableVixMonitoring.IsChecked = profile.EnableVixMonitoring;
            // Alpha Vantage API Key setting
            AlphaVantageApiKeyBox.Password = profile.AlphaVantageApiKey ?? "";
            // Alpha Vantage API Plan setting
            foreach (ComboBoxItem item in AlphaVantageApiPlanComboBox.Items)
            {
                if (int.TryParse((string)item.Tag, out int callsPerMinute) && callsPerMinute == profile.AlphaVantageApiCallsPerMinute)
                {
                    AlphaVantageApiPlanComboBox.SelectedItem = item;
                    break;
                }
            }
            // Alpha Vantage Data Entitlement setting
            foreach (ComboBoxItem item in AlphaVantageDataEntitlementComboBox.Items)
            {
                if ((string)item.Tag == (profile.AlphaVantageDataEntitlement ?? "none"))
                {
                    AlphaVantageDataEntitlementComboBox.SelectedItem = item;
                    break;
                }
            }
        }

        private void ProfilesListView_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isLoading) return;
            if (ProfilesListView.SelectedItem is DatabaseSettingsProfile profile)
            {
                LoadProfileToUI(profile);
            }
        }

        private void NewProfileButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Generate a unique profile name
                string newProfileName;
                int profileNumber = _profiles.Count + 1;
                do
                {
                    newProfileName = $"Profile {profileNumber}";
                    profileNumber++;
                } while (_profiles.Any(p => p.Name == newProfileName));

                var newProfile = DatabaseSettingsProfile.CreateDefault(newProfileName);
                newProfile.Name = newProfileName;
                newProfile.Description = "Custom profile";
                newProfile.IsDefault = false;
                newProfile.UserId = AuthenticationService.CurrentUserId;
                newProfile.CreatedDate = DateTime.Now;
                newProfile.ModifiedDate = DateTime.Now;
                
                int id = _settingsService.CreateSettingsProfile(newProfile);
                _ = LoadProfilesAsync();
                ProfilesListView.SelectedItem = _profiles.FirstOrDefault(p => p.Id == id);
            }
            catch (InvalidOperationException ex)
            {
                MessageBox.Show(ex.Message, "Cannot Create Profile", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to create profile: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void RenameProfileButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
            
            var input = Microsoft.VisualBasic.Interaction.InputBox("Enter new profile name:", "Rename Profile", _selectedProfile.Name);
            if (!string.IsNullOrWhiteSpace(input))
            {
                // Check if the new name conflicts with existing profiles
                if (_profiles.Any(p => p.Id != _selectedProfile.Id && p.Name == input))
                {
                    MessageBox.Show($"A profile with the name '{input}' already exists. Please choose a different name.", 
                                  "Duplicate Profile Name", MessageBoxButton.OK, MessageBoxImage.Warning);
                    return;
                }
                
                try
                {
                    _selectedProfile.Name = input;
                    _settingsService.UpdateSettingsProfile(_selectedProfile);
                    _ = LoadProfilesAsync();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Failed to rename profile: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void DeleteProfileButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
            if (MessageBox.Show($"Delete profile '{_selectedProfile.Name}'?", "Confirm", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
            {
                _settingsService.DeleteSettingsProfile(_selectedProfile.Id);
                _ = LoadProfilesAsync();
            }
        }

        private void SetAsDefaultButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
            _settingsService.SetProfileAsDefault(_selectedProfile.Id);
            _ = LoadProfilesAsync();
        }

        private void SaveTimer_Tick(object sender, EventArgs e)
        {
            _saveTimer.Stop();
            SaveCurrentSettings();
        }

        private void OnSettingChanged(object sender, RoutedEventArgs e)
        {
            if (_isLoading || _selectedProfile == null) return;

            // Reset the timer to debounce rapid changes
            _saveTimer.Stop();
            _saveTimer.Start();
        }

        private void OnSettingChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isLoading || _selectedProfile == null) return;

            // Reset the timer to debounce rapid changes
            _saveTimer.Stop();
            _saveTimer.Start();
        }

        private void OnSettingChanged(object sender, TextChangedEventArgs e)
        {
            if (_isLoading || _selectedProfile == null) return;

            // Reset the timer to debounce rapid changes
            _saveTimer.Stop();
            _saveTimer.Start();
        }

        private void SaveCurrentSettings()
        {
            if (_selectedProfile == null) return;

            try
            {
                // Validate and update profile from UI
                _selectedProfile.EnableApiModalChecks = EnableApiModalChecks.IsChecked == true;
                _selectedProfile.ApiTimeoutSeconds = int.TryParse(ApiTimeoutTextBox.Text, out int apiTimeout) ? apiTimeout : 30;
                _selectedProfile.CacheDurationMinutes = int.TryParse(CacheDurationTextBox.Text, out int cacheMin) ? cacheMin : 15;
                _selectedProfile.EnableHistoricalDataCache = EnableHistoricalDataCache.IsChecked == true;
                _selectedProfile.EnableDarkMode = EnableDarkMode.IsChecked == true;
                _selectedProfile.RememberWindowState = RememberWindowState.IsChecked == true;
                _selectedProfile.ChartUpdateIntervalSeconds = int.TryParse(ChartUpdateIntervalTextBox.Text, out int chartInt) ? chartInt : 2;
                _selectedProfile.DefaultGridRows = int.TryParse(DefaultGridRowsTextBox.Text, out int gridRows) ? gridRows : 4;
                _selectedProfile.DefaultGridColumns = int.TryParse(DefaultGridColumnsTextBox.Text, out int gridCols) ? gridCols : 4;
                _selectedProfile.EnablePriceAlerts = EnablePriceAlerts.IsChecked == true;
                _selectedProfile.EnableTradeNotifications = EnableTradeNotifications.IsChecked == true;
                _selectedProfile.EnablePaperTrading = EnablePaperTrading.IsChecked == true;
                if (RiskLevelComboBox.SelectedItem is ComboBoxItem riskItem)
                    _selectedProfile.RiskLevel = (string)riskItem.Content;
                if (GridBorderColorComboBox.SelectedItem is ComboBoxItem colorItem)
                    _selectedProfile.GridBorderColor = (string)colorItem.Tag;
                // Email alert settings
                _selectedProfile.EnableEmailAlerts = EnableEmailAlerts.IsChecked == true;
                _selectedProfile.EnableStandardAlertEmails = EnableStandardAlertEmails.IsChecked == true;
                _selectedProfile.EnableOpportunityAlertEmails = EnableOpportunityAlertEmails.IsChecked == true;
                _selectedProfile.EnablePredictionAlertEmails = EnablePredictionAlertEmails.IsChecked == true;
                _selectedProfile.EnableGlobalAlertEmails = EnableGlobalAlertEmails.IsChecked == true;
                _selectedProfile.AlertEmail = AlertEmailTextBox.Text.Trim();
                // VIX monitoring setting
                _selectedProfile.EnableVixMonitoring = EnableVixMonitoring.IsChecked == true;
                // Alpha Vantage API Key setting - saved to database only (no environment variable)
                _selectedProfile.AlphaVantageApiKey = AlphaVantageApiKeyBox.Password.Trim();
                // Alpha Vantage API Plan setting
                if (AlphaVantageApiPlanComboBox.SelectedItem is ComboBoxItem apiPlanItem)
                {
                    if (int.TryParse((string)apiPlanItem.Tag, out int callsPerMinute))
                    {
                        _selectedProfile.AlphaVantageApiCallsPerMinute = callsPerMinute;
                    }
                }
                // Alpha Vantage Data Entitlement setting
                if (AlphaVantageDataEntitlementComboBox.SelectedItem is ComboBoxItem dataEntitlementItem)
                {
                    _selectedProfile.AlphaVantageDataEntitlement = (string)dataEntitlementItem.Tag ?? "none";
                }

                _selectedProfile.ModifiedDate = DateTime.Now;
                _settingsService.UpdateSettingsProfile(_selectedProfile);

                // Clear API key cache to ensure updated key is used
                Quantra.DAL.Utilities.Utilities.ClearApiKeyCache();

                // Update AlphaVantageService rate limit if it's initialized
                var alphaVantageService = App.ServiceProvider?.GetService<Quantra.DAL.Services.AlphaVantageService>();
                alphaVantageService?.UpdateApiRateLimitFromSettings();

                // Immediately refresh the SharedTitleBar API usage display to show the new limit
                Task.Run(async () =>
                {
                    await Task.Delay(100); // Small delay to ensure settings are fully committed
                    SharedTitleBar.RefreshApiUsageDisplayStatic();
                });
            }
            catch (Exception ex)
            {
                // Silently log the error - don't show message boxes for passive saving
                System.Diagnostics.Debug.WriteLine($"Error saving settings: {ex.Message}");
            }
        }

        private void GridBorderColorComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isLoading) return;
            if (GridBorderColorComboBox.SelectedItem is ComboBoxItem selectedItem)
            {
                var colorValue = (string)selectedItem.Tag;
                if (!string.IsNullOrEmpty(colorValue))
                {
                    ColorPreview.Fill = (System.Windows.Media.Brush)new System.Windows.Media.BrushConverter().ConvertFromString(colorValue);
                }
            }

            // Trigger auto-save
            OnSettingChanged(sender, e);
        }

        private void AlphaVantageApiKeyBox_PasswordChanged(object sender, RoutedEventArgs e)
        {
            if (_isLoading) return;
            // Trigger auto-save
            OnSettingChanged(sender, e);
        }

        private void SaveProfileButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null)
            {
                MessageBox.Show("No profile selected.", "Save Profile", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // Stop any pending auto-save timer
            if (_saveTimer.IsEnabled)
            {
                _saveTimer.Stop();
            }

            // Save immediately
            SaveCurrentSettings();

            // Show confirmation
            MessageBox.Show($"Profile '{_selectedProfile.Name}' has been saved successfully.", "Save Profile", MessageBoxButton.OK, MessageBoxImage.Information);

            // Refresh the profiles list to show updated modification date
            _ = LoadProfilesAsync();
        }

        private void SaveMenuItem_Click(object sender, RoutedEventArgs e)
        {
            // Same functionality as button click
            SaveProfileButton_Click(sender, e);
        }

        private void AlphaVantageApiPlanComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isLoading) return;
            // Trigger auto-save
            OnSettingChanged(sender, e);
        }

        private void AlphaVantageDataEntitlementComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (_isLoading) return;
            // Trigger auto-save
            OnSettingChanged(sender, e);
        }
    }
}

