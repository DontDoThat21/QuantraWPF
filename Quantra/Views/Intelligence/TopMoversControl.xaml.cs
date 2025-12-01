using System;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Media;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.Intelligence
{
    /// <summary>
    /// Converter to get appropriate color for price changes
    /// </summary>
    public class ChangeColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double change)
            {
                if (change > 0)
                    return new SolidColorBrush(Color.FromRgb(80, 224, 112)); // Green
                if (change < 0)
                    return new SolidColorBrush(Color.FromRgb(255, 107, 107)); // Red
            }
            return new SolidColorBrush(Color.FromRgb(170, 170, 170)); // Gray for zero/neutral
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Converter to get sign prefix for changes
    /// </summary>
    public class ChangeSignConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double change)
            {
                if (change > 0)
                    return "+";
                if (change < 0)
                    return ""; // Negative sign is included in the number
            }
            return "";
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Interaction logic for TopMoversControl.xaml
    /// </summary>
    public partial class TopMoversControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private TopMoversResponse _topMovers;
        private bool _hasLoadedInitially;
        private int _selectedTabIndex = 0; // Track selected tab to preserve selection

        /// <summary>
        /// Event raised when a symbol is double-clicked for navigation
        /// </summary>
        public event EventHandler<string> SymbolSelected;

        public TopMoversControl()
        {
            InitializeComponent();

            // Get services from DI if available
            try
            {
                _alphaVantageService = App.ServiceProvider?.GetService(typeof(AlphaVantageService)) as AlphaVantageService;
                _loggingService = App.ServiceProvider?.GetService(typeof(LoggingService)) as LoggingService;
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Service initialization error: {ex.Message}";
            }

            // Auto-load data when control is loaded
            Loaded += TopMoversControl_Loaded;
            
            // Handle tab selection changes to preserve selection
            MoversTabControl.SelectionChanged += MoversTabControl_SelectionChanged;
        }

        private void MoversTabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            // Store the selected tab index to preserve it during data updates
            if (MoversTabControl.SelectedIndex >= 0)
            {
                _selectedTabIndex = MoversTabControl.SelectedIndex;
            }
        }

        private async void TopMoversControl_Loaded(object sender, RoutedEventArgs e)
        {
            // Only load once on initial load
            if (!_hasLoadedInitially)
            {
                _hasLoadedInitially = true;
                try
                {
                    await LoadTopMovers();
                }
                catch (Exception ex)
                {
                    StatusText.Text = $"Error loading data: {ex.Message}";
                    _loggingService?.LogErrorWithContext(ex, "Error in TopMoversControl_Loaded");
                }
            }
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadTopMovers();
        }

        private async System.Threading.Tasks.Task LoadTopMovers()
        {
            if (_alphaVantageService == null)
            {
                StatusText.Text = "Alpha Vantage service not available.";
                return;
            }

            try
            {
                LoadingIndicator.Visibility = Visibility.Visible;
                RefreshButton.IsEnabled = false;
                StatusText.Text = "Loading market movers data...";

                // Store current tab selection before updating data
                var currentTabIndex = MoversTabControl.SelectedIndex;
                
                _topMovers = await _alphaVantageService.GetTopMoversAsync();

                if (_topMovers != null)
                {
                    GainersGrid.ItemsSource = _topMovers.TopGainers;
                    LosersGrid.ItemsSource = _topMovers.TopLosers;
                    MostActiveGrid.ItemsSource = _topMovers.MostActivelyTraded;

                    StatusText.Text = $"Last updated: {_topMovers.LastUpdated:g} | " +
                                     $"{_topMovers.TopGainers.Count} gainers, " +
                                     $"{_topMovers.TopLosers.Count} losers, " +
                                     $"{_topMovers.MostActivelyTraded.Count} most active";

                    _loggingService?.Log("Info", $"Loaded top movers: {_topMovers.TopGainers.Count} gainers, {_topMovers.TopLosers.Count} losers");
                    
                    // Restore tab selection after data update
                    // Use Dispatcher to ensure UI has finished updating before setting selection
                    Dispatcher.InvokeAsync(() =>
                    {
                        if (currentTabIndex >= 0 && currentTabIndex < MoversTabControl.Items.Count)
                        {
                            MoversTabControl.SelectedIndex = currentTabIndex;
                        }
                        else if (_selectedTabIndex >= 0 && _selectedTabIndex < MoversTabControl.Items.Count)
                        {
                            MoversTabControl.SelectedIndex = _selectedTabIndex;
                        }
                    }, System.Windows.Threading.DispatcherPriority.Loaded);
                }
                else
                {
                    GainersGrid.ItemsSource = null;
                    LosersGrid.ItemsSource = null;
                    MostActiveGrid.ItemsSource = null;
                    StatusText.Text = "No data available. Please try again later.";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, "Error loading top movers");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                RefreshButton.IsEnabled = true;
            }
        }

        private void DataGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (sender is DataGrid grid && grid.SelectedItem is MarketMover mover)
            {
                SymbolSelected?.Invoke(this, mover.Ticker);
                _loggingService?.Log("Info", $"Symbol selected from top movers: {mover.Ticker}");
            }
        }

        /// <summary>
        /// Public method to refresh data programmatically
        /// </summary>
        public async System.Threading.Tasks.Task RefreshAsync()
        {
            await LoadTopMovers();
        }
    }
}
