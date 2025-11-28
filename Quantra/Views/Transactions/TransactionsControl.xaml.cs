using MaterialDesignThemes.Wpf;
using Microsoft.Extensions.DependencyInjection;
using Quantra.Models;
using Quantra.DAL.Services.Interfaces;
using Quantra.ViewModels;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for TransactionsControl.xaml
    /// </summary>
    public partial class TransactionsControl : UserControl
    {
        private TransactionsViewModel _viewModel;
        private IServiceScope _scope;

        public TransactionsControl()
        {
            InitializeComponent();
            
            // Create a scope for scoped services
            _scope = App.ServiceProvider.CreateScope();
            var scopedProvider = _scope.ServiceProvider;

            _viewModel = scopedProvider.GetRequiredService<TransactionsViewModel>();

            DataContext = _viewModel;

            // Load data when control is loaded
            Loaded += (s, e) => {
                _viewModel.LoadTransactions();
            };
            
            // Clean up scope when control is unloaded
            Unloaded += (s, e) => {
                _scope?.Dispose();
            };
        }

        private void TransactionsGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (_viewModel.SelectedTransaction != null)
            {
                // Open transaction details window or show details in a dialog
                MessageBox.Show($"Transaction Details for {_viewModel.SelectedTransaction.Symbol} " +
                    $"({_viewModel.SelectedTransaction.TransactionType}) at {_viewModel.SelectedTransaction.ExecutionTime}",
                    "Transaction Details", MessageBoxButton.OK, MessageBoxImage.Information);
                
                // In a real implementation, you might open a details window:
                // var detailsWindow = new TransactionDetailsWindow(_viewModel.SelectedTransaction);
                // detailsWindow.Owner = Window.GetWindow(this);
                // detailsWindow.ShowDialog();
            }
        }

        private void SearchTextBox_KeyUp(object sender, KeyEventArgs e)
        {
            // Apply search filter when Enter is pressed
            if (e.Key == Key.Enter)
            {
                _viewModel.Search(SearchTextBox.Text);
                ShowNotification("Search applied.", PackIconKind.Magnify, Colors.SkyBlue);
            }
        }

        private void SearchButton_Click(object sender, RoutedEventArgs e)
        {
            _viewModel.Search(SearchTextBox.Text);
            ShowNotification("Search applied.", PackIconKind.Magnify, Colors.SkyBlue);
        }

        private void ApplyFiltersButton_Click(object sender, RoutedEventArgs e)
        {
            _viewModel.ApplyFilters();
            ShowNotification("Filters applied.", PackIconKind.Filter, Colors.SkyBlue);
        }

        private void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            _viewModel.LoadTransactions();
            ShowNotification("Transaction data refreshed.", PackIconKind.Refresh, Colors.LimeGreen);
        }

        private void ExportButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _viewModel.ExportData();
                ShowNotification(_viewModel.NotificationText, PackIconKind.FileExport, Colors.LimeGreen);
            }
            catch (Exception ex)
            {
                ShowNotification($"Export failed: {ex.Message}", PackIconKind.Error, Colors.Red);
                //DatabaseMonolith.Log("Error", "Failed to export transaction data", ex.ToString());
            }
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

        // Method to force refresh
        public void ForceRefresh()
        {
            _viewModel?.LoadTransactions();
        }
    }
}
