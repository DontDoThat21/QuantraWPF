using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Input;
using Quantra.DAL.Enums;
using Quantra.DAL.Services;

namespace Quantra.Views.StockExplorer
{
    /// <summary>
    /// Interaction logic for LoadHistoricalsModeWindow.xaml
    /// </summary>
    public partial class LoadHistoricalsModeWindow : Window
    {
        private readonly StockConfigurationService _configService;
        private List<string> _selectedConfigurationSymbols;

        /// <summary>
        /// Gets the selected load mode
        /// </summary>
        public HistoricalsLoadMode SelectedMode { get; private set; }

        /// <summary>
        /// Gets the symbols to load (for StockConfiguration mode)
        /// </summary>
        public List<string> SymbolsToLoad { get; private set; }

        /// <summary>
        /// Gets whether the user confirmed the selection
        /// </summary>
        public bool Confirmed { get; private set; }

        /// <summary>
        /// Gets or sets the count of loaded symbols (for display purposes)
        /// </summary>
        public int LoadedSymbolCount { get; set; }

        /// <summary>
        /// Gets or sets the total symbol count (for display purposes)
        /// </summary>
        public int TotalSymbolCount { get; set; }

        /// <summary>
        /// Parameterless constructor for XAML designer support
        /// </summary>
        public LoadHistoricalsModeWindow()
        {
            InitializeComponent();
            SetupEventHandlers();
        }

        /// <summary>
        /// Initializes a new instance of the LoadHistoricalsModeWindow class.
        /// </summary>
        /// <param name="configService">The stock configuration service</param>
        /// <param name="loadedCount">Number of symbols already loaded</param>
        /// <param name="totalCount">Total number of symbols available</param>
        public LoadHistoricalsModeWindow(StockConfigurationService configService, int loadedCount = 0, int totalCount = 12459)
        {
            InitializeComponent();
            _configService = configService;
            LoadedSymbolCount = loadedCount;
            TotalSymbolCount = totalCount;
            
            SetupEventHandlers();
            UpdateDescriptions();
        }

        /// <summary>
        /// Shows the load mode selection dialog.
        /// </summary>
        /// <param name="configService">The stock configuration service</param>
        /// <param name="loadedCount">Number of symbols already loaded</param>
        /// <param name="totalCount">Total number of symbols available</param>
        /// <param name="owner">The owner window</param>
        /// <returns>Tuple of (SelectedMode, SymbolsToLoad) or null if cancelled</returns>
        public static (HistoricalsLoadMode Mode, List<string> Symbols)? Show(
            StockConfigurationService configService, 
            int loadedCount = 0, 
            int totalCount = 12459,
            Window owner = null)
        {
            var dialog = new LoadHistoricalsModeWindow(configService, loadedCount, totalCount);
            
            if (owner != null)
            {
                dialog.Owner = owner;
            }
            else if (Application.Current.MainWindow != null)
            {
                dialog.Owner = Application.Current.MainWindow;
            }
            
            dialog.ShowDialog();
            
            if (dialog.Confirmed)
            {
                return (dialog.SelectedMode, dialog.SymbolsToLoad);
            }
            
            return null;
        }

        private void SetupEventHandlers()
        {
            // Show/hide configuration panel based on selection
            AllSymbolsRadio.Checked += (s, e) => ConfigurationPanel.Visibility = Visibility.Collapsed;
            NonLoadedOnlyRadio.Checked += (s, e) => ConfigurationPanel.Visibility = Visibility.Collapsed;
            StockConfigurationRadio.Checked += (s, e) => ConfigurationPanel.Visibility = Visibility.Visible;
        }

        private void UpdateDescriptions()
        {
            var nonLoadedCount = TotalSymbolCount - LoadedSymbolCount;
            NonLoadedDescription.Text = $"Load historical data only for {nonLoadedCount:N0} symbols without cached data (out of {TotalSymbolCount:N0} total)";
        }

        private void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            if (AllSymbolsRadio.IsChecked == true)
            {
                SelectedMode = HistoricalsLoadMode.AllSymbols;
                SymbolsToLoad = null;
                Confirmed = true;
                Close();
            }
            else if (NonLoadedOnlyRadio.IsChecked == true)
            {
                SelectedMode = HistoricalsLoadMode.NonLoadedOnly;
                SymbolsToLoad = null;
                Confirmed = true;
                Close();
            }
            else if (StockConfigurationRadio.IsChecked == true)
            {
                if (_selectedConfigurationSymbols == null || _selectedConfigurationSymbols.Count == 0)
                {
                    // Open the configuration manager to select a configuration
                    OpenConfigurationManager();
                }
                else
                {
                    SelectedMode = HistoricalsLoadMode.StockConfiguration;
                    SymbolsToLoad = _selectedConfigurationSymbols;
                    Confirmed = true;
                    Close();
                }
            }
        }

        private void ManageConfigurationsButton_Click(object sender, RoutedEventArgs e)
        {
            OpenConfigurationManager();
        }

        private void OpenConfigurationManager()
        {
            if (_configService == null)
            {
                Shared.CustomModal.ShowError("Configuration service not available.", "Error", this);
                return;
            }

            var symbols = StockConfigurationManagerWindow.ShowAndGetSymbols(_configService, this);
            
            if (symbols != null && symbols.Count > 0)
            {
                _selectedConfigurationSymbols = symbols;
                ManageConfigurationsButton.Content = $"Selected: {symbols.Count} symbols";
                
                // If user selected symbols, auto-confirm the dialog
                SelectedMode = HistoricalsLoadMode.StockConfiguration;
                SymbolsToLoad = _selectedConfigurationSymbols;
                Confirmed = true;
                Close();
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            Confirmed = false;
            Close();
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Confirmed = false;
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
                Confirmed = false;
                Close();
                e.Handled = true;
            }
            else if (e.Key == Key.Enter)
            {
                LoadButton_Click(LoadButton, new RoutedEventArgs());
                e.Handled = true;
            }
        }
    }
}
