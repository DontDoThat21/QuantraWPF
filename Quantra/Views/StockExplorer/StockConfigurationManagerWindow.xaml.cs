using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using Quantra.DAL.Data.Entities;
using Quantra.DAL.Services;
using Quantra.Views.Shared;

namespace Quantra.Views.StockExplorer
{
    /// <summary>
    /// View model for stock configuration list display
    /// </summary>
    public class StockConfigurationViewModel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public List<string> Symbols { get; set; }
        public string SymbolCountText => $"{Symbols?.Count ?? 0} symbols";
        public bool IsDefault { get; set; }
    }

    /// <summary>
    /// Interaction logic for StockConfigurationManagerWindow.xaml
    /// </summary>
    public partial class StockConfigurationManagerWindow : Window
    {
        private readonly StockConfigurationService _configService;
        private List<StockConfigurationViewModel> _configurations;
        private StockConfigurationViewModel _selectedConfiguration;
        private bool _isNewConfiguration;

        /// <summary>
        /// Gets the selected configuration ID after the dialog is closed
        /// </summary>
        public int? SelectedConfigurationId { get; private set; }

        /// <summary>
        /// Gets the symbols from the selected configuration
        /// </summary>
        public List<string> SelectedSymbols { get; private set; }

        /// <summary>
        /// Gets whether a configuration was selected (vs. cancelled)
        /// </summary>
        public bool ConfigurationSelected { get; private set; }

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public StockConfigurationManagerWindow()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Initializes a new instance of the StockConfigurationManagerWindow class.
        /// </summary>
        /// <param name="configService">The stock configuration service</param>
        public StockConfigurationManagerWindow(StockConfigurationService configService)
        {
            InitializeComponent();
            _configService = configService;
            
            LoadConfigurations();
            ClearEditor();
        }

        /// <summary>
        /// Shows the configuration manager dialog.
        /// </summary>
        /// <param name="configService">The stock configuration service</param>
        /// <param name="owner">The owner window</param>
        /// <returns>The selected symbols or null if cancelled</returns>
        public static List<string> ShowAndGetSymbols(StockConfigurationService configService, Window owner = null)
        {
            var dialog = new StockConfigurationManagerWindow(configService);
            
            if (owner != null)
            {
                dialog.Owner = owner;
            }
            else if (Application.Current.MainWindow != null)
            {
                dialog.Owner = Application.Current.MainWindow;
            }
            
            dialog.ShowDialog();
            
            return dialog.ConfigurationSelected ? dialog.SelectedSymbols : null;
        }

        private void LoadConfigurations()
        {
            try
            {
                var entities = _configService.GetAllConfigurations();
                _configurations = entities.Select(e => new StockConfigurationViewModel
                {
                    Id = e.Id,
                    Name = e.Name,
                    Description = e.Description,
                    Symbols = _configService.GetConfigurationSymbols(e.Id),
                    IsDefault = e.IsDefault
                }).ToList();

                ConfigurationsListBox.ItemsSource = _configurations;
                
                UpdateStatus($"Loaded {_configurations.Count} configurations");
            }
            catch (Exception ex)
            {
                CustomModal.ShowError($"Failed to load configurations: {ex.Message}", "Error", this);
            }
        }

        private void ClearEditor()
        {
            _selectedConfiguration = null;
            _isNewConfiguration = false;
            NameTextBox.Text = string.Empty;
            DescriptionTextBox.Text = string.Empty;
            SymbolsTextBox.Text = string.Empty;
            UpdateSymbolCount();
        }

        private void ConfigurationsListBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            _selectedConfiguration = ConfigurationsListBox.SelectedItem as StockConfigurationViewModel;
            _isNewConfiguration = false;

            if (_selectedConfiguration != null)
            {
                NameTextBox.Text = _selectedConfiguration.Name;
                DescriptionTextBox.Text = _selectedConfiguration.Description;
                SymbolsTextBox.Text = string.Join(Environment.NewLine, _selectedConfiguration.Symbols);
                UpdateSymbolCount();
            }
        }

        private void NewButton_Click(object sender, RoutedEventArgs e)
        {
            ConfigurationsListBox.SelectedItem = null;
            ClearEditor();
            _isNewConfiguration = true;
            NameTextBox.Focus();
            UpdateStatus("Creating new configuration...");
        }

        private void DeleteButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedConfiguration == null)
            {
                CustomModal.ShowWarning("Please select a configuration to delete.", "No Selection", this);
                return;
            }

            var confirmed = ConfirmationModal.Show(
                $"Are you sure you want to delete the configuration '{_selectedConfiguration.Name}'?",
                "Confirm Delete",
                this);

            if (confirmed)
            {
                try
                {
                    if (_configService.DeleteConfiguration(_selectedConfiguration.Id))
                    {
                        LoadConfigurations();
                        ClearEditor();
                        UpdateStatus("Configuration deleted");
                    }
                    else
                    {
                        CustomModal.ShowError("Failed to delete configuration.", "Error", this);
                    }
                }
                catch (Exception ex)
                {
                    CustomModal.ShowError($"Failed to delete configuration: {ex.Message}", "Error", this);
                }
            }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            var name = NameTextBox.Text?.Trim();
            var description = DescriptionTextBox.Text?.Trim();
            var symbolsText = SymbolsTextBox.Text?.Trim();

            if (string.IsNullOrEmpty(name))
            {
                CustomModal.ShowWarning("Please enter a configuration name.", "Validation Error", this);
                NameTextBox.Focus();
                return;
            }

            var symbols = ParseSymbols(symbolsText);
            if (!symbols.Any())
            {
                CustomModal.ShowWarning("Please enter at least one symbol.", "Validation Error", this);
                SymbolsTextBox.Focus();
                return;
            }

            try
            {
                if (_isNewConfiguration || _selectedConfiguration == null)
                {
                    // Create new configuration
                    _configService.CreateConfiguration(name, description, symbols);
                    UpdateStatus($"Created configuration '{name}' with {symbols.Count} symbols");
                }
                else
                {
                    // Update existing configuration
                    _configService.UpdateConfiguration(_selectedConfiguration.Id, name, description, symbols);
                    UpdateStatus($"Updated configuration '{name}' with {symbols.Count} symbols");
                }

                LoadConfigurations();
                
                // Select the saved configuration
                var savedConfig = _configurations.FirstOrDefault(c => c.Name == name);
                if (savedConfig != null)
                {
                    ConfigurationsListBox.SelectedItem = savedConfig;
                }
            }
            catch (InvalidOperationException ex)
            {
                CustomModal.ShowWarning(ex.Message, "Validation Error", this);
            }
            catch (Exception ex)
            {
                CustomModal.ShowError($"Failed to save configuration: {ex.Message}", "Error", this);
            }
        }

        private void SelectButton_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedConfiguration == null)
            {
                CustomModal.ShowWarning("Please select a configuration to load.", "No Selection", this);
                return;
            }

            SelectedConfigurationId = _selectedConfiguration.Id;
            SelectedSymbols = _selectedConfiguration.Symbols;
            ConfigurationSelected = true;

            // Update last used timestamp
            _configService.UpdateLastUsed(_selectedConfiguration.Id);

            Close();
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            ConfigurationSelected = false;
            Close();
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            ConfigurationSelected = false;
            Close();
        }

        private void Header_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                ConfigurationSelected = false;
                Close();
                e.Handled = true;
            }
        }

        private void SymbolsTextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
        {
            UpdateSymbolCount();
        }

        private void UpdateSymbolCount()
        {
            var symbols = ParseSymbols(SymbolsTextBox.Text);
            SymbolCountTextBlock.Text = $"{symbols.Count} symbols";
        }

        private List<string> ParseSymbols(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return new List<string>();
            }

            return text
                .Split(new[] { '\r', '\n', ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => s.Trim().ToUpper())
                .Where(s => !string.IsNullOrEmpty(s))
                .Distinct()
                .ToList();
        }

        private void UpdateStatus(string message)
        {
            StatusTextBlock.Text = message;
        }
    }
}
