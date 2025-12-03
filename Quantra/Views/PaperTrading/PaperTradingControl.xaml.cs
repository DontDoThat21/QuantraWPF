using MaterialDesignThemes.Wpf;
using Quantra.ViewModels;
using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Quantra.DAL.TradingEngine.Orders;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for PaperTradingControl.xaml
    /// </summary>
    public partial class PaperTradingControl : UserControl
    {
        private PaperTradingViewModel _viewModel;

        public PaperTradingControl()
        {
            InitializeComponent();

            _viewModel = new PaperTradingViewModel();
            DataContext = _viewModel;

            // Load data when control is loaded
            Loaded += (s, e) =>
            {
                _viewModel.Initialize();
            };

            // Clean up when control is unloaded
            Unloaded += (s, e) =>
            {
                _viewModel.Dispose();
            };
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

        private void BuyButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (ValidateOrderEntry())
                {
                    _viewModel.PlaceOrder(OrderSide.Buy);
                    ShowNotification($"Buy order placed for {_viewModel.OrderSymbol}", PackIconKind.CheckCircle, Colors.LimeGreen);
                    ClearOrderEntry();
                }
            }
            catch (Exception ex)
            {
                ShowNotification($"Error placing order: {ex.Message}", PackIconKind.Error, Colors.Red);
            }
        }

        private void SellButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (ValidateOrderEntry())
                {
                    _viewModel.PlaceOrder(OrderSide.Sell);
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

            // Create and start a timer to hide the notification after a delay
            var timer = new System.Windows.Threading.DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(3)
            };
            timer.Tick += (s, args) =>
            {
                NotificationPanel.Visibility = Visibility.Collapsed;
                timer.Stop();
            };
            timer.Start();
        }

        /// <summary>
        /// Method to force refresh the control
        /// </summary>
        public void ForceRefresh()
        {
            _viewModel?.Refresh();
        }
    }
}
