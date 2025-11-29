using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.FundamentalData
{
    /// <summary>
    /// Interaction logic for CashFlowControl.xaml
    /// </summary>
    public partial class CashFlowControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private CashFlowStatement _cashFlowStatement;

        public CashFlowControl()
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
        }

        private async void LoadButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadCashFlowData();
        }

        private async void SymbolTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await LoadCashFlowData();
            }
        }

        private async System.Threading.Tasks.Task LoadCashFlowData()
        {
            var symbol = SymbolTextBox.Text?.Trim().ToUpper();
            if (string.IsNullOrWhiteSpace(symbol))
            {
                StatusText.Text = "Please enter a valid symbol.";
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
                StatusText.Text = $"Loading cash flow data for {symbol}...";

                _cashFlowStatement = await _alphaVantageService.GetCashFlowAsync(symbol);

                if (_cashFlowStatement != null)
                {
                    UpdateDataGrid();
                    UpdateSummaryCards();
                    SummaryCardsGrid.Visibility = Visibility.Visible;
                    StatusText.Text = $"Last updated: {_cashFlowStatement.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded cash flow for {symbol}");
                }
                else
                {
                    StatusText.Text = $"No cash flow data found for {symbol}.";
                    SummaryCardsGrid.Visibility = Visibility.Collapsed;
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, $"Error loading cash flow for {symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        private void UpdateDataGrid()
        {
            if (_cashFlowStatement == null) return;

            var reports = QuarterlyRadio.IsChecked == true 
                ? _cashFlowStatement.QuarterlyReports 
                : _cashFlowStatement.AnnualReports;

            CashFlowGrid.ItemsSource = reports;
        }

        private void UpdateSummaryCards()
        {
            if (_cashFlowStatement == null) return;

            var reports = QuarterlyRadio.IsChecked == true 
                ? _cashFlowStatement.QuarterlyReports 
                : _cashFlowStatement.AnnualReports;

            if (reports != null && reports.Count > 0)
            {
                var latestReport = reports[0];
                OperatingCashFlowText.Text = FormatCurrency(latestReport.OperatingCashflow);
                InvestingCashFlowText.Text = FormatCurrency(latestReport.CashflowFromInvestment);
                FinancingCashFlowText.Text = FormatCurrency(latestReport.CashflowFromFinancing);
                FreeCashFlowText.Text = FormatCurrency(latestReport.FreeCashFlow);
            }
        }

        private string FormatCurrency(decimal? value)
        {
            if (!value.HasValue) return "N/A";
            
            var absValue = Math.Abs(value.Value);
            var isNegative = value.Value < 0;
            string formatted;
            
            if (absValue >= 1_000_000_000)
                formatted = $"{absValue / 1_000_000_000:F2}B";
            else if (absValue >= 1_000_000)
                formatted = $"{absValue / 1_000_000:F2}M";
            else
                formatted = $"{absValue:N0}";
            
            return isNegative ? $"-${formatted}" : $"${formatted}";
        }

        private void ReportTypeChanged(object sender, RoutedEventArgs e)
        {
            UpdateDataGrid();
            UpdateSummaryCards();
        }

        /// <summary>
        /// Public method to load data for a specific symbol programmatically
        /// </summary>
        public async System.Threading.Tasks.Task LoadSymbolAsync(string symbol)
        {
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                SymbolTextBox.Text = symbol.ToUpper();
                await LoadCashFlowData();
            }
        }
    }
}
