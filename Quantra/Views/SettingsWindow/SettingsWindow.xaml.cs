using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Quantra.Models;
using Quantra.Services;

namespace Quantra
{
    public partial class SettingsWindow : Window
    {
        private List<DatabaseSettingsProfile> _profiles;
        private DatabaseSettingsProfile _selectedProfile;
        private bool _isLoading = false;

        public SettingsWindow()
        {
            InitializeComponent();
            // No need to manually bind named controls, WPF does this automatically
            LoadProfiles();
        }

        private void LoadProfiles()
        {
            _isLoading = true;
            _profiles = SettingsService.GetAllSettingsProfiles();
            ProfilesListView.ItemsSource = _profiles;
            if (_profiles.Count > 0)
            {
                var defaultProfile = _profiles.FirstOrDefault(p => p.IsDefault) ?? _profiles[0];
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
            var newProfile = DatabaseSettingsProfile.CreateDefault($"Profile {_profiles.Count + 1}");
            newProfile.Name = $"Profile {_profiles.Count + 1}";
            newProfile.Description = "Custom profile";
            newProfile.IsDefault = false;
            newProfile.CreatedDate = DateTime.Now;
            newProfile.ModifiedDate = DateTime.Now;
            int id = SettingsService.CreateSettingsProfile(newProfile);
            LoadProfiles();
            ProfilesListView.SelectedItem = _profiles.FirstOrDefault(p => p.Id == id);
        }

        private void RenameProfileButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
            var input = Microsoft.VisualBasic.Interaction.InputBox("Enter new profile name:", "Rename Profile", _selectedProfile.Name);
            if (!string.IsNullOrWhiteSpace(input))
            {
                _selectedProfile.Name = input;
                SettingsService.UpdateSettingsProfile(_selectedProfile);
                LoadProfiles();
            }
        }

        private void DeleteProfileButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
            if (MessageBox.Show($"Delete profile '{_selectedProfile.Name}'?", "Confirm", MessageBoxButton.YesNo, MessageBoxImage.Warning) == MessageBoxResult.Yes)
            {
                SettingsService.DeleteSettingsProfile(_selectedProfile.Id);
                LoadProfiles();
            }
        }

        private void SetAsDefaultButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
            SettingsService.SetProfileAsDefault(_selectedProfile.Id);
            LoadProfiles();
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedProfile == null) return;
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
            SettingsService.UpdateSettingsProfile(_selectedProfile);
            MessageBox.Show("Settings saved.", "Settings", MessageBoxButton.OK, MessageBoxImage.Information);
            LoadProfiles();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}

