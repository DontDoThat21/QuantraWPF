using MaterialDesignThemes.Wpf;
using Quantra.ViewModels;
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Quantra.DAL.TradingEngine.Orders;
using System.Windows.Input;
using Quantra.DAL.Services;
using System.Windows.Threading;
using System.Linq;
using Quantra.Models;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for PaperTradingControl.xaml
    /// </summary>
    public partial class PaperTradingControl : UserControl
    {
        private PaperTradingViewModel _viewModel;
        private IAlphaVantageService _alphaVantageService;
        private DispatcherTimer _symbolSearchTimer;
        private DispatcherTimer _notificationTimer;
        private string _lastSearchText = string.Empty;
        private QuoteData _currentQuoteData;

        public PaperTradingControl(IAlphaVantageService alphaVantageService)
        {
            InitializeComponent();

            // Inject AlphaVantage service
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));

            _viewModel = new PaperTradingViewModel();
            DataContext = _viewModel;

            // Initialize symbol search timer
            InitializeSymbolSearchTimer();

            // Initialize notification timer
            InitializeNotificationTimer();

            // Load data when control is loaded
            Loaded += (s, e) =>
            {
                _viewModel.Initialize();
            };

            // Clean up when control is unloaded
            Unloaded += (s, e) =>
            {
                _notificationTimer.Stop();
                _viewModel.Dispose();
                _symbolSearchTimer?.Stop();
            };
        }

        private void InitializeSymbolSearchTimer()
        {
            _symbolSearchTimer = new DispatcherTimer();
            _symbolSearchTimer.Interval = TimeSpan.FromMilliseconds(500);
            _symbolSearchTimer.Tick += SymbolSearchTimer_Tick;
        }

        private void InitializeNotificationTimer()
        {
            _notificationTimer = new DispatcherTimer();
            _notificationTimer.Interval = TimeSpan.FromSeconds(3);
            _notificationTimer.Tick += NotificationTimer_Tick;
        }

        private void NotificationTimer_Tick(object sender, EventArgs e)
        {
            _notificationTimer.Stop();
            NotificationPanel.Visibility = Visibility.Collapsed;
        }

        private void StartStopButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _viewModel.ToggleEngine();
            }
            catch (Exception ex)
            {
                ShowNotification($"Error: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private async void BuyButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (ValidateOrderEntry())
                {
                    await _viewModel.PlaceOrderAsync(OrderSide.Buy);
                    ShowNotification($"Buy order placed for {_viewModel.OrderSymbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                    ClearOrderEntry();
                }
            }
            catch (Exception ex)
            {
                ShowNotification($"Error placing order: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private async void SellButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (ValidateOrderEntry())
                {
                    await _viewModel.PlaceOrderAsync(OrderSide.Sell);
                    ShowNotification($"Sell order placed for {_viewModel.OrderSymbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                    ClearOrderEntry();
                }
            }
            catch (Exception ex)
            {
                ShowNotification($"Error placing order: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private void CancelOrderButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (sender is Button button && button.Tag is Guid orderId)
                {
                    if (_viewModel.CancelOrder(orderId))
                    {
                        ShowNotification("Order cancelled", PackIconKind.CheckCircle, Colors.LimeGreen);
                    }
                    else
                    {
                        ShowNotification("Could not cancel order", PackIconKind.AlertCircle, Colors.Orange);
                    }
                }
            }
            catch (Exception ex)
            {
                ShowNotification($"Error cancelling order: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private void ResetAccountButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var result = MessageBox.Show(
                    "Are you sure you want to reset your paper trading account? This will close all positions and reset your balance to $100,000.",
                    "Reset Account",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Question);

                if (result == MessageBoxResult.Yes)
                {
                    _viewModel.ResetAccount();
                    ShowNotification("Account reset to $100,000", PackIconKind.Refresh, Colors.SkyBlue);
                }
            }
            catch (Exception ex)
            {
                ShowNotification($"Error resetting account: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _viewModel.Refresh();
                ShowNotification("Data refreshed", PackIconKind.Refresh, Colors.SkyBlue);
            }
            catch (Exception ex)
            {
                ShowNotification($"Error refreshing: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private void ShowAllOrdersCheckBox_Changed(object sender, RoutedEventArgs e)
        {
            _viewModel.RefreshOrders();
        }

        private bool ValidateOrderEntry()
        {
            if (string.IsNullOrWhiteSpace(_viewModel.OrderSymbol))
            {
                ShowNotification("Please enter a symbol", PackIconKind.AlertCircle, Colors.Orange);
                return false;
            }

            if (_viewModel.OrderQuantity <= 0)
            {
                ShowNotification("Please enter a valid quantity", PackIconKind.AlertCircle, Colors.Orange);
                return false;
            }

            // Validate limit price for limit orders
            if (_viewModel.SelectedOrderType?.ToString()?.Contains("Limit") == true && _viewModel.LimitPrice <= 0)
            {
                ShowNotification("Please enter a valid limit price", PackIconKind.AlertCircle, Colors.Orange);
                return false;
            }

            // Validate stop price for stop orders
            if (_viewModel.SelectedOrderType?.ToString()?.Contains("Stop") == true && _viewModel.StopPrice <= 0)
            {
                ShowNotification("Please enter a valid stop price", PackIconKind.AlertCircle, Colors.Orange);
                return false;
            }

            return true;
        }

        private void ClearOrderEntry()
        {
            _viewModel.OrderSymbol = string.Empty;
            _viewModel.OrderQuantity = 100;
            _viewModel.LimitPrice = 0;
            _viewModel.StopPrice = 0;
        }

        private void ShowNotification(string message, PackIconKind icon, Color iconColor)
        {
            // Update notification properties in the view model
            _viewModel.NotificationText = message;
            _viewModel.NotificationIcon = icon;
            _viewModel.NotificationIconColor = new SolidColorBrush(iconColor);
            _viewModel.NotificationBorderBrush = new SolidColorBrush(Color.FromArgb(100, iconColor.R, iconColor.G, iconColor.B));

            // Show the notification
            NotificationPanel.Visibility = Visibility.Visible;

            // Restart the class-level timer to hide the notification after a delay
            _notificationTimer.Stop();
            _notificationTimer.Start();
        }

        /// <summary>
        /// Method to force refresh the control
        /// </summary>
        public void ForceRefresh()
        {
            _viewModel?.Refresh();
        }

        #region Symbol Search Event Handlers

        private void SymbolSearchTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            _symbolSearchTimer?.Stop();
            _symbolSearchTimer?.Start();
        }

        private async void SymbolSearchTimer_Tick(object sender, EventArgs e)
        {
            _symbolSearchTimer?.Stop();

            var searchText = SymbolSearchTextBox?.Text?.Trim();
            if (string.IsNullOrEmpty(searchText) || searchText == _lastSearchText || searchText.Length < 1)
                return;

            _lastSearchText = searchText;

            try
            {
                if (SearchLoadingText != null)
                    SearchLoadingText.Visibility = Visibility.Visible;

                var results = await _alphaVantageService.SearchSymbolsAsync(searchText);

                if (SymbolSearchListBox != null)
                    SymbolSearchListBox.ItemsSource = results;

                if (SearchLoadingText != null)
                    SearchLoadingText.Visibility = Visibility.Collapsed;

                if (results.Count > 0 && SymbolSearchPopup != null)
                {
                    SymbolSearchPopup.IsOpen = true;
                }
            }
            catch (Exception ex)
            {
                if (SearchLoadingText != null)
                    SearchLoadingText.Visibility = Visibility.Collapsed;
                ShowNotification($"Error searching symbols: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private async void SymbolSearchListBox_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (SymbolSearchListBox?.SelectedItem is SymbolSearchResult selectedResult)
            {
                await SelectSymbolAsync(selectedResult.Symbol);
                SymbolSearchPopup.IsOpen = false;
            }
        }

        private async void SymbolSearchListBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter && SymbolSearchListBox?.SelectedItem is SymbolSearchResult selectedResult)
            {
                await SelectSymbolAsync(selectedResult.Symbol);
                SymbolSearchPopup.IsOpen = false;
                e.Handled = true;
            }
        }

        private void SymbolSearchTextBox_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Down && SymbolSearchPopup?.IsOpen == true)
            {
                SymbolSearchListBox?.Focus();
                if (SymbolSearchListBox != null && SymbolSearchListBox.Items.Count > 0)
                {
                    SymbolSearchListBox.SelectedIndex = 0;
                }
                e.Handled = true;
            }
            else if (e.Key == Key.Enter)
            {
                var searchText = SymbolSearchTextBox?.Text?.Trim().ToUpper();
                if (!string.IsNullOrEmpty(searchText))
                {
                    SelectSymbolAsync(searchText);
                    e.Handled = true;
                }
            }
        }

        private void SymbolSearchTextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            // Optionally trigger search when focused
        }

        private void SymbolSearchTextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            // Delay closing to allow selection
            Dispatcher.BeginInvoke(new Action(() =>
            {
                if (SymbolSearchPopup != null && !SymbolSearchListBox.IsMouseOver)
                {
                    SymbolSearchPopup.IsOpen = false;
                }
            }), DispatcherPriority.Background);
        }

        private async System.Threading.Tasks.Task SelectSymbolAsync(string symbol)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(symbol))
                    return;

                // Update the text box
                _viewModel.OrderSymbol = symbol.ToUpper();

                // Fetch quote data from Global Quote API
                _currentQuoteData = await _alphaVantageService.GetQuoteDataAsync(symbol);

                if (_currentQuoteData != null)
                {
                    // Update UI with quote data
                    UpdateQuoteDataDisplay(_currentQuoteData);
                    QuoteDataPanel.Visibility = Visibility.Visible;
                }
                else
                {
                    ShowNotification($"Could not fetch quote data for {symbol}", PackIconKind.AlertCircle, Colors.Orange);
                    QuoteDataPanel.Visibility = Visibility.Collapsed;
                }
            }
            catch (Exception ex)
            {
                ShowNotification($"Error loading symbol: {ex.Message}", PackIconKind.Error, Colors.Red);
                QuoteDataPanel.Visibility = Visibility.Collapsed;
            }
        }

        private void UpdateQuoteDataDisplay(QuoteData quoteData)
        {
            if (quoteData == null)
                return;

            // Update price
            QuotePriceText.Text = $"${quoteData.Price:F2}";

            // Update change with color
            var changeColor = quoteData.Change >= 0 ? Colors.LimeGreen : Colors.Red;
            var changeSymbol = quoteData.Change >= 0 ? "+" : "";
            QuoteChangeText.Text = $"{changeSymbol}{quoteData.Change:F2} ({changeSymbol}{quoteData.ChangePercent:F2}%)";
            QuoteChangeText.Foreground = new SolidColorBrush(changeColor);

            // Update day range
            QuoteDayRangeText.Text = $"${quoteData.DayLow:F2} - ${quoteData.DayHigh:F2}";

            // Update volume
            QuoteVolumeText.Text = quoteData.Volume >= 1000000
                ? $"{quoteData.Volume / 1000000:F2}M"
                : $"{quoteData.Volume:N0}";
        }

        #endregion
    }
}
