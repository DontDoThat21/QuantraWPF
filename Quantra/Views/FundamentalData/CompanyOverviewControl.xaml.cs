using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Quantra.DAL.Services;
using Quantra.Models;
using Quantra.ViewModels;

namespace Quantra.Views.FundamentalData
{
    /// <summary>
    /// Interaction logic for CompanyOverviewControl.xaml
    /// </summary>
    public partial class CompanyOverviewControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private CompanyOverview _currentOverview;

        public CompanyOverviewControl()
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
            await LoadCompanyData();
        }

        private async void SymbolTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await LoadCompanyData();
            }
        }

        private async System.Threading.Tasks.Task LoadCompanyData()
        {
            var symbol = SymbolTextBox.Text?.Trim().ToUpper();
            if (string.IsNullOrWhiteSpace(symbol))
            {
                ErrorTextBlock.Text = "Please enter a valid symbol.";
                ErrorTextBlock.Visibility = Visibility.Visible;
                return;
            }

            if (_alphaVantageService == null)
            {
                ErrorTextBlock.Text = "Alpha Vantage service not available.";
                ErrorTextBlock.Visibility = Visibility.Visible;
                return;
            }

            try
            {
                // Show loading state
                LoadingIndicator.Visibility = Visibility.Visible;
                ErrorTextBlock.Visibility = Visibility.Collapsed;
                ContentGrid.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = false;
                StatusText.Text = $"Loading data for {symbol}...";

                // Fetch company overview
                var overview = await _alphaVantageService.GetCompanyOverviewAsync(symbol);

                if (overview != null)
                {
                    _currentOverview = overview;
                    PopulateData(overview);
                    ContentGrid.Visibility = Visibility.Visible;
                    StatusText.Text = $"Last updated: {overview.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded company overview for {symbol}");
                }
                else
                {
                    ErrorTextBlock.Text = $"No data found for symbol '{symbol}'. Please check the symbol and try again.";
                    ErrorTextBlock.Visibility = Visibility.Visible;
                    StatusText.Text = "No data found.";
                }
            }
            catch (Exception ex)
            {
                ErrorTextBlock.Text = $"Error loading data: {ex.Message}";
                ErrorTextBlock.Visibility = Visibility.Visible;
                StatusText.Text = "Error occurred.";
                _loggingService?.LogErrorWithContext(ex, $"Error loading company overview for {symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        private void PopulateData(CompanyOverview overview)
        {
            // Company Information
            CompanyNameText.Text = overview.Name ?? "N/A";
            CompanySymbolText.Text = overview.Symbol ?? "N/A";
            ExchangeText.Text = overview.Exchange ?? "N/A";
            SectorText.Text = overview.Sector ?? "N/A";
            IndustryText.Text = overview.Industry ?? "N/A";
            DescriptionText.Text = overview.Description ?? "No description available.";

            // Valuation Metrics
            MarketCapText.Text = overview.FormattedMarketCap;
            PERatioText.Text = FormatDecimal(overview.PERatio, "F2");
            PEGRatioText.Text = FormatDecimal(overview.PEGRatio, "F2");
            BookValueText.Text = FormatCurrency(overview.BookValue);
            DividendYieldText.Text = FormatPercentage(overview.DividendYield);
            EPSText.Text = FormatCurrency(overview.EPS);
            Week52HighText.Text = FormatCurrency(overview.Week52High);
            Week52LowText.Text = FormatCurrency(overview.Week52Low);

            // Additional Metrics
            BetaText.Text = FormatDecimal(overview.Beta, "F2");
            ProfitMarginText.Text = FormatPercentage(overview.ProfitMargin);
            ROEText.Text = FormatPercentage(overview.ReturnOnEquityTTM);
            AnalystTargetText.Text = FormatCurrency(overview.AnalystTargetPrice);
        }

        private string FormatDecimal(decimal? value, string format)
        {
            return value.HasValue ? value.Value.ToString(format) : "N/A";
        }

        private string FormatCurrency(decimal? value)
        {
            return value.HasValue ? $"${value.Value:F2}" : "N/A";
        }

        private string FormatPercentage(decimal? value)
        {
            if (!value.HasValue) return "N/A";
            // If value is already in decimal form (e.g., 0.15 for 15%), multiply by 100
            var percentValue = Math.Abs(value.Value) < 1 ? value.Value * 100 : value.Value;
            return $"{percentValue:F2}%";
        }

        private async void IncomeStatementButton_Click(object sender, RoutedEventArgs e)
        {
            if (_currentOverview == null || string.IsNullOrEmpty(_currentOverview.Symbol))
            {
                MessageBox.Show("Please load a company first.", "No Company Selected", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                var incomeStatementModal = new IncomeStatementModal(_alphaVantageService, _loggingService, _currentOverview.Symbol);
                incomeStatementModal.Owner = Window.GetWindow(this);
                incomeStatementModal.ShowDialog();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error opening income statement modal for {_currentOverview.Symbol}");
                MessageBox.Show($"Error opening income statement: {ex.Message}", "Error", 
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private async void BalanceSheetButton_Click(object sender, RoutedEventArgs e)
        {
            if (_currentOverview == null || string.IsNullOrEmpty(_currentOverview.Symbol))
            {
                MessageBox.Show("Please load a company first.", "No Company Selected", 
                    MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            try
            {
                var balanceSheetModal = new BalanceSheetModal(_alphaVantageService, _loggingService, _currentOverview.Symbol);
                balanceSheetModal.Owner = Window.GetWindow(this);
                balanceSheetModal.ShowDialog();
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Error opening balance sheet modal for {_currentOverview.Symbol}");
                MessageBox.Show($"Error opening balance sheet: {ex.Message}", "Error", 
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Public method to load data for a specific symbol programmatically
        /// </summary>
        public async System.Threading.Tasks.Task LoadSymbolAsync(string symbol)
        {
            if (!string.IsNullOrWhiteSpace(symbol))
            {
                SymbolTextBox.Text = symbol.ToUpper();
                await LoadCompanyData();
            }
        }
    }
}
