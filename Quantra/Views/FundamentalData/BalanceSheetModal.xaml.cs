using System;
using System.Windows;
using System.Windows.Input;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.FundamentalData
{
    /// <summary>
    /// Interaction logic for BalanceSheetModal.xaml
    /// </summary>
    public partial class BalanceSheetModal : Window
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private readonly string _symbol;
        private BalanceSheet _balanceSheet;

        public BalanceSheetModal(AlphaVantageService alphaVantageService, LoggingService loggingService, string symbol)
        {
            InitializeComponent();
            _alphaVantageService = alphaVantageService;
            _loggingService = loggingService;
            _symbol = symbol;

            TitleText.Text = $"Balance Sheet - {symbol}";
            
            this.Loaded += BalanceSheetModal_Loaded;
        }

        private async void BalanceSheetModal_Loaded(object sender, RoutedEventArgs e)
        {
            await LoadBalanceSheet();
        }

        private async System.Threading.Tasks.Task LoadBalanceSheet()
        {
            if (_alphaVantageService == null || string.IsNullOrEmpty(_symbol))
            {
                StatusText.Text = "Error: Service or symbol not available.";
                return;
            }

            try
            {
                LoadingIndicator.Visibility = Visibility.Visible;
                StatusText.Text = $"Loading balance sheet for {_symbol}...";

                _balanceSheet = await _alphaVantageService.GetBalanceSheetAsync(_symbol);

                if (_balanceSheet != null)
                {
                    UpdateDataGrid();
                    StatusText.Text = $"Last updated: {_balanceSheet.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded balance sheet for {_symbol}");
                }
                else
                {
                    StatusText.Text = $"No balance sheet data found for {_symbol}.";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, $"Error loading balance sheet for {_symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
            }
        }

        private void UpdateDataGrid()
        {
            if (_balanceSheet == null) return;

            var reports = QuarterlyRadio.IsChecked == true 
                ? _balanceSheet.QuarterlyReports 
                : _balanceSheet.AnnualReports;

            BalanceSheetGrid.ItemsSource = reports;
        }

        private void ReportTypeChanged(object sender, RoutedEventArgs e)
        {
            UpdateDataGrid();
        }

        private void TitleBar_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ClickCount == 2)
            {
                this.WindowState = this.WindowState == System.Windows.WindowState.Maximized 
                    ? System.Windows.WindowState.Normal 
                    : System.Windows.WindowState.Maximized;
            }
            else
            {
                this.DragMove();
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}
