using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
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
    /// Converter to get appropriate color for transaction types
    /// </summary>
    public class TransactionTypeColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            var code = value as string;
            if (string.IsNullOrEmpty(code))
                return new SolidColorBrush(Color.FromRgb(62, 62, 86)); // Default gray

            return code switch
            {
                "P" => new SolidColorBrush(Color.FromRgb(32, 192, 64)), // Purchase - Green
                "S" => new SolidColorBrush(Color.FromRgb(192, 32, 32)), // Sale - Red
                "M" or "X" => new SolidColorBrush(Color.FromRgb(100, 149, 237)), // Exercise - Blue
                "A" or "G" => new SolidColorBrush(Color.FromRgb(255, 204, 0)), // Award - Yellow
                "D" => new SolidColorBrush(Color.FromRgb(255, 140, 0)), // Sale to Issuer - Orange
                "F" => new SolidColorBrush(Color.FromRgb(138, 43, 226)), // Tax Payment - Purple
                _ => new SolidColorBrush(Color.FromRgb(62, 62, 86)) // Default gray
            };
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Simple bool to visibility converter
    /// </summary>
    public class BoolToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is bool boolValue)
            {
                return boolValue ? Visibility.Visible : Visibility.Collapsed;
            }
            return Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Interaction logic for InsiderTransactionsControl.xaml
    /// </summary>
    public partial class InsiderTransactionsControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private List<InsiderTransactionData> _allTransactions;
        private List<InsiderTransactionData> _filteredTransactions;

        public InsiderTransactionsControl()
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

            // Set default date range (last 90 days)
            EndDatePicker.SelectedDate = DateTime.Today;
            StartDatePicker.SelectedDate = DateTime.Today.AddDays(-90);
        }

        private async void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadInsiderTransactions();
        }

        private async void SymbolTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await LoadInsiderTransactions();
            }
        }

        private async System.Threading.Tasks.Task LoadInsiderTransactions()
        {
            var symbol = SymbolTextBox.Text?.Trim().ToUpper();
            if (string.IsNullOrWhiteSpace(symbol))
            {
                StatusText.Text = "Please enter a valid stock symbol.";
                return;
            }

            if (_alphaVantageService == null)
            {
                StatusText.Text = "Alpha Vantage service not available.";
                return;
            }

            try
            {
                LoadingIndicator.Visibility = Visibility.Visible;
                LoadButton.IsEnabled = false;
                StatusText.Text = $"Loading insider transactions for {symbol}...";

                var response = await _alphaVantageService.GetInsiderTransactionsAsync(symbol);

                if (response != null && response.Transactions.Count > 0)
                {
                    _allTransactions = response.Transactions;
                    ApplyFilters();
                    UpdateSummary();
                    StatusText.Text = $"Loaded {_allTransactions.Count} transactions for {symbol}";
                    _loggingService?.Log("Info", $"Loaded {_allTransactions.Count} insider transactions for {symbol}");
                }
                else
                {
                    _allTransactions = null;
                    _filteredTransactions = null;
                    TransactionsGrid.ItemsSource = null;
                    TotalBuysText.Text = "";
                    TotalSellsText.Text = "";
                    StatusText.Text = $"No insider transactions found for {symbol}.";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, $"Error loading insider transactions for {symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        private void TransactionTypeFilter_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            ApplyFilters();
        }

        private void DateFilter_Changed(object sender, SelectionChangedEventArgs e)
        {
            ApplyFilters();
        }

        private void ClearFilterButton_Click(object sender, RoutedEventArgs e)
        {
            TransactionTypeFilter.SelectedIndex = 0;
            EndDatePicker.SelectedDate = DateTime.Today;
            StartDatePicker.SelectedDate = DateTime.Today.AddDays(-90);
            ApplyFilters();
        }

        private void ApplyFilters()
        {
            if (TransactionsGrid == null)
                return;

            if (_allTransactions == null || _allTransactions.Count == 0)
            {
                _filteredTransactions = new List<InsiderTransactionData>();
                TransactionsGrid.ItemsSource = _filteredTransactions;
                return;
            }

            _filteredTransactions = _allTransactions.ToList();

            // Apply transaction type filter
            if (TransactionTypeFilter.SelectedItem is ComboBoxItem selectedType)
            {
                var filter = selectedType.Content?.ToString();
                _filteredTransactions = filter switch
                {
                    "Purchases" => _filteredTransactions.Where(t => t.TransactionCode == "P").ToList(),
                    "Sales" => _filteredTransactions.Where(t => t.TransactionCode == "S").ToList(),
                    "Option Exercises" => _filteredTransactions.Where(t => t.TransactionCode == "M" || t.TransactionCode == "X").ToList(),
                    "Awards/Grants" => _filteredTransactions.Where(t => t.TransactionCode == "A" || t.TransactionCode == "G").ToList(),
                    _ => _filteredTransactions
                };
            }

            // Apply date range filter
            if (StartDatePicker.SelectedDate.HasValue)
            {
                var startDate = StartDatePicker.SelectedDate.Value;
                _filteredTransactions = _filteredTransactions.Where(t => t.TransactionDate >= startDate).ToList();
            }

            if (EndDatePicker.SelectedDate.HasValue)
            {
                var endDate = EndDatePicker.SelectedDate.Value.AddDays(1); // Include the end date
                _filteredTransactions = _filteredTransactions.Where(t => t.TransactionDate < endDate).ToList();
            }

            // Sort by filing date descending
            _filteredTransactions = _filteredTransactions.OrderByDescending(t => t.FilingDate).ToList();

            TransactionsGrid.ItemsSource = _filteredTransactions;
            UpdateSummary();
        }

        private void UpdateSummary()
        {
            if (_filteredTransactions == null || _filteredTransactions.Count == 0)
            {
                TotalBuysText.Text = "";
                TotalSellsText.Text = "";
                return;
            }

            var buys = _filteredTransactions.Where(t => t.IsBuy).ToList();
            var sells = _filteredTransactions.Where(t => t.IsSell).ToList();

            double totalBuyValue = buys.Sum(t => t.TotalValue);
            double totalSellValue = sells.Sum(t => t.TotalValue);

            TotalBuysText.Text = $"Buys: {buys.Count} (${totalBuyValue:N0})";
            TotalSellsText.Text = $"Sells: {sells.Count} (${totalSellValue:N0})";
        }

        /// <summary>
        /// Public method to load data for a specific symbol programmatically
        /// </summary>
        public async System.Threading.Tasks.Task LoadSymbolAsync(string symbol)
        {
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                SymbolTextBox.Text = symbol.ToUpper();
                await LoadInsiderTransactions();
            }
        }
    }
}
