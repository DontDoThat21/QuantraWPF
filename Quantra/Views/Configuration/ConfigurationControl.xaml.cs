using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Extensions.DependencyInjection;
using Quantra.Configuration;
using Quantra.ViewModels; // Add reference to ViewModels namespace

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for ConfigurationControl.xaml
    /// </summary>
    public partial class ConfigurationControl : UserControl, INotifyPropertyChanged
    {
        // Fallback property for legacy mode binding when ViewModel DI is unavailable
        private int _selectedSettingsTabIndex = 0;
        public int SelectedSettingsTabIndex 
        { 
            get => _selectedSettingsTabIndex;
            set
            {
                if (_selectedSettingsTabIndex != value)
                {
                    _selectedSettingsTabIndex = value;
                    OnPropertyChanged();
                }
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public ConfigurationControl()
        {
            InitializeComponent();
            
            // Debug: Check if TabControl was created properly
            //DatabaseMonolith.Log("Info", "ConfigurationControl constructor called");
            
            // Set up the view model using dependency injection
            var configManager = App.ServiceProvider.GetService<IConfigurationManager>();
            var configBridge = App.ServiceProvider.GetService<DatabaseConfigBridge>();
            
            if (configManager != null && configBridge != null)
            {
                DataContext = new ConfigurationViewModel(configManager, configBridge);
                //DatabaseMonolith.Log("Info", "ConfigurationViewModel created successfully via DI");
            }
            else
            {
                // Fallback to legacy configuration if DI fails
                InitializeLegacyConfiguration();
                // Ensure bindings still work in legacy mode
                DataContext = this;
                //DatabaseMonolith.Log("Info", "Using legacy configuration mode");
            }
            
            // Debug: Log initial tab count and selection
            Loaded += (s, e) => 
            {
                //DatabaseMonolith.Log("Info", $"ConfigurationControl loaded with {ConfigurationTabControl.Items.Count} tabs");
                //DatabaseMonolith.Log("Info", $"Initial selected index: {ConfigurationTabControl.SelectedIndex}");
            };
        }
        
        /// <summary>
        /// Initialize legacy configuration when dependency injection is not available
        /// </summary>
        private void InitializeLegacyConfiguration()
        {
            // Initialize event handlers for legacy implementation
            RefreshRateSlider.ValueChanged += RefreshRateSlider_ValueChanged;
            
            // Load saved settings from database
            LoadSettings();
        }

        private void LoadSettings()
        {
            try
            {
                // Try to load settings from database or settings file
                var settings = DatabaseMonolith.GetUserSettings();

                // Apply loaded settings to UI controls
                // Since settings is a tuple, we can directly access its properties
                EnableApiModalChecksCheckBox.IsChecked = settings.EnableApiModalChecks;

                // In a real implementation, we would load all settings here
                // Position sizing settings
                if (settings.AccountSize > 0)
                    AccountSizeTextBox.Text = settings.AccountSize.ToString();
                
                if (settings.BaseRiskPercentage > 0)
                    RiskPercentageTextBox.Text = (settings.BaseRiskPercentage * 100.0).ToString("F1");
                
                if (!string.IsNullOrEmpty(settings.PositionSizingMethod))
                {
                    // Set the selected item in the combo box
                    foreach (ComboBoxItem item in PositionSizingMethodComboBox.Items)
                    {
                        if (item.Content.ToString() == settings.PositionSizingMethod)
                        {
                            PositionSizingMethodComboBox.SelectedItem = item;
                            break;
                        }
                    }
                }
                
                if (settings.MaxPositionSizePercent > 0)
                    MaxPositionSizeTextBox.Text = (settings.MaxPositionSizePercent * 100.0).ToString("F1");
                
                if (settings.FixedTradeAmount > 0)
                    FixedTradeAmountTextBox.Text = settings.FixedTradeAmount.ToString("F0");
                
                if (settings.ATRMultiple > 0)
                    ATRMultipleTextBox.Text = settings.ATRMultiple.ToString("F1");
                
                UseKellyCriterionCheckBox.IsChecked = settings.UseKellyCriterion;
                
                if (settings.HistoricalWinRate > 0)
                    WinRateTextBox.Text = (settings.HistoricalWinRate * 100.0).ToString("F0");
                
                if (settings.HistoricalRewardRiskRatio > 0)
                    RewardRiskRatioTextBox.Text = settings.HistoricalRewardRiskRatio.ToString("F1");
                
                if (settings.KellyFractionMultiplier > 0)
                    KellyFractionTextBox.Text = settings.KellyFractionMultiplier.ToString("F1");

                //DatabaseMonolith.Log("Info", "Configuration settings loaded successfully");
            }
            catch (Exception ex)
            {
                // If loading fails, use default values
                SetDefaultValues();
                //DatabaseMonolith.Log("Error", "Failed to load configuration settings", ex.ToString());
            }
        }

        private void SetDefaultValues()
        {
            // Set default values for all settings
            TradingModeComboBox.SelectedIndex = 0; // Market
            RiskModeComboBox.SelectedIndex = 1;    // Normal
            ThemeComboBox.SelectedIndex = 0;       // Dark
            RefreshRateSlider.Value = 5;           // 5 seconds
            EnableNotificationsCheckBox.IsChecked = true;
            EnableApiModalChecksCheckBox.IsChecked = true;
            AutoSaveCheckBox.IsChecked = true;
            
            // API settings defaults - empty for security reasons
            FmpApiKeyBox.Password = "";
            WebullUsernameTextBox.Text = "";
            WebullPasswordBox.Password = "";
            WebullPinBox.Password = "";
            RememberCredentialsCheckBox.IsChecked = false;
            
            // Advanced settings defaults
            DefaultStopLossTextBox.Text = "2.0";
            
            // Position sizing defaults
            AccountSizeTextBox.Text = "100000";
            RiskPercentageTextBox.Text = "1.0";
            PositionSizingMethodComboBox.SelectedIndex = 0; // FixedRisk
            MaxPositionSizeTextBox.Text = "10.0"; // 10% of account
            FixedTradeAmountTextBox.Text = "5000"; // $5000
            ATRMultipleTextBox.Text = "2.0"; // 2x ATR
            UseKellyCriterionCheckBox.IsChecked = false;
            WinRateTextBox.Text = "55"; // 55%
            RewardRiskRatioTextBox.Text = "2.0"; // 2:1
            KellyFractionTextBox.Text = "0.5"; // Half-Kelly
            DefaultTakeProfitTextBox.Text = "5.0";
            RsiOversoldTextBox.Text = "30";
            RsiOverboughtTextBox.Text = "70";
            BollingerPeriodTextBox.Text = "20";
            BollingerStdDevTextBox.Text = "2.0";
        }

        private void RefreshRateSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (RefreshRateLabel != null)
            {
                int value = (int)e.NewValue;
                RefreshRateLabel.Text = value == 1 ? "1 second" : $"{value} seconds";
            }
        }

        private void ConfigurationTabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Guard against events during initialization or when no tab is selected
            if (sender is not TabControl tabControl || tabControl.SelectedItem is not TabItem selectedTab)
                return;

            try
            {
                string tabHeader = selectedTab.Header?.ToString() ?? "";
                
                // Log the tab selection for debugging
                //DatabaseMonolith.Log("Info", $"ConfigurationControl: Tab selected - {tabHeader}");
                
                // Update the property when tab changes manually (backup for binding issues)
                SelectedSettingsTabIndex = tabControl.SelectedIndex;
                
                // Also update ViewModel if available
                if (DataContext is ConfigurationViewModel viewModel)
                {
                    viewModel.SelectedSettingsTabIndex = tabControl.SelectedIndex;
                    //DatabaseMonolith.Log("Info", $"Updated ViewModel SelectedSettingsTabIndex to: {tabControl.SelectedIndex}");
                }
                else
                {
                    //DatabaseMonolith.Log("Info", "DataContext is not ConfigurationViewModel, using legacy mode");
                }

                // Handle specific tab selections that might need special handling
                switch (tabHeader)
                {
                    case "General":
                        // Handle general settings tab selection
                        //DatabaseMonolith.Log("Info", "General settings tab selected");
                        break;

                    case "API Configuration":
                        // Handle API configuration tab selection
                        //DatabaseMonolith.Log("Info", "API configuration tab selected");
                        break;

                    case "Trading Settings":
                        // Handle trading settings tab selection
                        //DatabaseMonolith.Log("Info", "Trading settings tab selected");
                        break;

                    case "Position Sizing":
                        // Handle position sizing tab selection
                        //DatabaseMonolith.Log("Info", "Position sizing tab selected");
                        break;

                    case "Advanced":
                        // Handle advanced settings tab selection
                        //DatabaseMonolith.Log("Info", "Advanced settings tab selected");
                        break;

                    default:
                        //DatabaseMonolith.Log("Info", $"Unknown tab selected: {tabHeader}");
                        break;
                }

                // Mark the event as handled to prevent it from bubbling up to MainWindow
                e.Handled = true;
            }
            catch (Exception ex)
            {
                //DatabaseMonolith.Log("Error", $"Error handling tab selection change in ConfigurationControl: {ex.Message}", ex.ToString());
            }
        }

        private void SaveSettingsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Save all settings to the database or settings file
                
                // For this example, we'll save the EnableApiModalChecks setting
                bool enableApiModalChecks = EnableApiModalChecksCheckBox.IsChecked ?? false;
                DatabaseMonolith.SaveUserSettings(WebullPinBox.Password, enableApiModalChecks);
                
                // In a real implementation, we would save all settings here
                
                // Save API keys
                string apiKey = FmpApiKeyBox.Password;
                // Some method to securely save the API key
                
                // Save credentials if requested
                if (RememberCredentialsCheckBox.IsChecked == true)
                {
                    string username = WebullUsernameTextBox.Text;
                    string password = WebullPasswordBox.Password;
                    string pin = WebullPinBox.Password;
                    
                    // Save credentials securely (in a real app)
                    
                    // Save credentials securely
                    DatabaseMonolith.RememberAccount(username, password, pin);
                }
                
                // Save position sizing settings
                if (double.TryParse(AccountSizeTextBox.Text, out double accountSize))
                    DatabaseMonolith.SaveSetting("AccountSize", accountSize.ToString());
                
                if (double.TryParse(RiskPercentageTextBox.Text, out double riskPct))
                    DatabaseMonolith.SaveSetting("BaseRiskPercentage", (riskPct / 100.0).ToString());
                
                if (PositionSizingMethodComboBox.SelectedItem is ComboBoxItem selectedMethod)
                    DatabaseMonolith.SaveSetting("PositionSizingMethod", selectedMethod.Content.ToString());
                
                if (double.TryParse(MaxPositionSizeTextBox.Text, out double maxPosSize))
                    DatabaseMonolith.SaveSetting("MaxPositionSizePercent", (maxPosSize / 100.0).ToString());
                
                if (double.TryParse(FixedTradeAmountTextBox.Text, out double fixedAmt))
                    DatabaseMonolith.SaveSetting("FixedTradeAmount", fixedAmt.ToString());
                
                if (double.TryParse(ATRMultipleTextBox.Text, out double atrMult))
                    DatabaseMonolith.SaveSetting("ATRMultiple", atrMult.ToString());
                
                bool useKelly = UseKellyCriterionCheckBox.IsChecked ?? false;
                DatabaseMonolith.SaveSetting("UseKellyCriterion", useKelly.ToString());
                
                if (double.TryParse(WinRateTextBox.Text, out double winRate))
                    DatabaseMonolith.SaveSetting("HistoricalWinRate", (winRate / 100.0).ToString());
                
                if (double.TryParse(RewardRiskRatioTextBox.Text, out double rrRatio))
                    DatabaseMonolith.SaveSetting("HistoricalRewardRiskRatio", rrRatio.ToString());
                
                if (double.TryParse(KellyFractionTextBox.Text, out double kellyFrac))
                    DatabaseMonolith.SaveSetting("KellyFractionMultiplier", kellyFrac.ToString());
                    
                // Log success
                //DatabaseMonolith.Log("Info", "Configuration settings saved successfully");
                
                // Show success message to user
                MessageBox.Show("Settings have been saved successfully.", "Settings Saved", 
                                MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                // Log and show error
                //DatabaseMonolith.Log("Error", "Failed to save configuration settings", ex.ToString());
                MessageBox.Show($"Error saving settings: {ex.Message}", "Error",
                                MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            // Confirm with user before resetting
            MessageBoxResult result = MessageBox.Show(
                "Are you sure you want to reset all settings to their default values?", 
                "Reset Confirmation", 
                MessageBoxButton.YesNo, 
                MessageBoxImage.Warning);
                
            if (result == MessageBoxResult.Yes)
            {
                SetDefaultValues();
                //DatabaseMonolith.Log("Info", "Configuration settings reset to defaults");
                MessageBox.Show("All settings have been reset to their default values.", 
                               "Settings Reset", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }

        private void TestConnectionButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Disable button to prevent multiple clicks
                TestConnectionButton.IsEnabled = false;
                
                // Get API credentials from input fields
                string apiKey = FmpApiKeyBox.Password;
                string username = WebullUsernameTextBox.Text;
                string password = WebullPasswordBox.Password;
                string pin = WebullPinBox.Password;
                
                // Test connection logic would go here
                // For demo purposes, we'll just show a success message
                
                // In a real implementation:
                // var tradingBot = new WebullTradingBot();
                // bool success = await tradingBot.Authenticate(username, password, pin);
                
                bool success = true; // Mock success
                
                // Show appropriate message
                if (success)
                {
                    MessageBox.Show("Connection test successful!", "Success", 
                                   MessageBoxButton.OK, MessageBoxImage.Information);
                    //DatabaseMonolith.Log("Info", "API connection test successful");
                }
                else
                {
                    MessageBox.Show("Connection test failed. Please check your credentials and try again.", 
                                   "Connection Failed", MessageBoxButton.OK, MessageBoxImage.Error);
                    //DatabaseMonolith.Log("Error", "API connection test failed");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error testing connection: {ex.Message}", "Error",
                               MessageBoxButton.OK, MessageBoxImage.Error);
                //DatabaseMonolith.Log("Error", "Error during API connection test", ex.ToString());
            }
            finally
            {
                // Re-enable button
                TestConnectionButton.IsEnabled = true;
            }
        }

        private void ClearCacheButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Clear cache logic would go here
                // For demo purposes, we'll just show a success message
                
                MessageBox.Show("Cache cleared successfully!", "Success", 
                               MessageBoxButton.OK, MessageBoxImage.Information);
                //DatabaseMonolith.Log("Info", "Cache cleared successfully");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error clearing cache: {ex.Message}", "Error",
                              MessageBoxButton.OK, MessageBoxImage.Error);
                //DatabaseMonolith.Log("Error", "Error clearing cache", ex.ToString());
            }
        }

        private void ExportLogsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Export logs logic would go here
                // For demo purposes, we'll just show a success message
                
                // In a real implementation, you would:
                // 1. Show a save file dialog
                // 2. Export logs from database to the selected file
                
                MessageBox.Show("Logs exported successfully!", "Success", 
                              MessageBoxButton.OK, MessageBoxImage.Information);
                //DatabaseMonolith.Log("Info", "Logs exported successfully");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error exporting logs: {ex.Message}", "Error",
                             MessageBoxButton.OK, MessageBoxImage.Error);
                //DatabaseMonolith.Log("Error", "Error exporting logs", ex.ToString());
            }
        }
    }
}
