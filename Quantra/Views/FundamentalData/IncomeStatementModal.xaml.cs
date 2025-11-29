using System;
using System.Windows;
using System.Windows.Input;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.FundamentalData
{
    /// <summary>
    /// Interaction logic for IncomeStatementModal.xaml
    /// </summary>
    public partial class IncomeStatementModal : Window
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private readonly string _symbol;
        private IncomeStatement _incomeStatement;

        public IncomeStatementModal(AlphaVantageService alphaVantageService, LoggingService loggingService, string symbol)
        {
            InitializeComponent();
            _alphaVantageService = alphaVantageService;
            _loggingService = loggingService;
            _symbol = symbol;

            TitleText.Text = $"Income Statement - {symbol}";
            
            this.Loaded += IncomeStatementModal_Loaded;
        }

        private async void IncomeStatementModal_Loaded(object sender, RoutedEventArgs e)
        {
            await LoadIncomeStatement();
        }

        private async System.Threading.Tasks.Task LoadIncomeStatement()
        {
            if (_alphaVantageService == null || string.IsNullOrEmpty(_symbol))
            {
                StatusText.Text = "Error: Service or symbol not available.";
                return;
            }

            try
            {
                LoadingIndicator.Visibility = Visibility.Visible;
                StatusText.Text = $"Loading income statement for {_symbol}...";

                _incomeStatement = await _alphaVantageService.GetIncomeStatementAsync(_symbol);

                if (_incomeStatement != null)
                {
                    UpdateDataGrid();
                    StatusText.Text = $"Last updated: {_incomeStatement.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded income statement for {_symbol}");
                }
                else
                {
                    StatusText.Text = $"No income statement data found for {_symbol}.";
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, $"Error loading income statement for {_symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
            }
        }

        private void UpdateDataGrid()
        {
            if (_incomeStatement == null) return;

            var reports = QuarterlyRadio.IsChecked == true 
                ? _incomeStatement.QuarterlyReports 
                : _incomeStatement.AnnualReports;

            IncomeStatementGrid.ItemsSource = reports;
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
