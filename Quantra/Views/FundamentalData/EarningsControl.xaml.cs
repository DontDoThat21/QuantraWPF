using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Input;
using LiveCharts;
using LiveCharts.Wpf;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Views.FundamentalData
{
    /// <summary>
    /// Value converter to determine if a value is positive
    /// </summary>
    public class PositiveValueConverter : IValueConverter
    {
        public static PositiveValueConverter Instance { get; } = new PositiveValueConverter();

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is decimal decimalValue)
                return decimalValue >= 0;
            if (value is double doubleValue)
                return doubleValue >= 0;
            return true;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Interaction logic for EarningsControl.xaml
    /// </summary>
    public partial class EarningsControl : UserControl
    {
        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private EarningsData _earningsData;

        public EarningsControl()
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
            await LoadEarningsData();
        }

        private async void SymbolTextBox_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                await LoadEarningsData();
            }
        }

        private async System.Threading.Tasks.Task LoadEarningsData()
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
                StatusText.Text = $"Loading earnings data for {symbol}...";

                _earningsData = await _alphaVantageService.GetEarningsAsync(symbol);

                if (_earningsData != null)
                {
                    PopulateData();
                    UpcomingEarningsCard.Visibility = Visibility.Visible;
                    StatusText.Text = $"Last updated: {_earningsData.LastUpdated:g}";
                    _loggingService?.Log("Info", $"Loaded earnings for {symbol}");
                }
                else
                {
                    StatusText.Text = $"No earnings data found for {symbol}.";
                    UpcomingEarningsCard.Visibility = Visibility.Collapsed;
                }
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
                _loggingService?.LogErrorWithContext(ex, $"Error loading earnings for {symbol}");
            }
            finally
            {
                LoadingIndicator.Visibility = Visibility.Collapsed;
                LoadButton.IsEnabled = true;
            }
        }

        private void PopulateData()
        {
            if (_earningsData == null) return;

            // Populate upcoming earnings
            UpcomingEarningsText.Text = _earningsData.UpcomingEarningsDate;

            // Populate latest EPS
            if (_earningsData.QuarterlyEarnings?.Count > 0)
            {
                var latestEps = _earningsData.QuarterlyEarnings[0].ReportedEPS;
                LatestEPSText.Text = latestEps.HasValue ? $"${latestEps.Value:F2}" : "N/A";
            }

            // Populate quarterly earnings grid
            QuarterlyEarningsGrid.ItemsSource = _earningsData.QuarterlyEarnings;

            // Populate annual earnings grid
            AnnualEarningsGrid.ItemsSource = _earningsData.AnnualEarnings;

            // Populate EPS trend chart
            PopulateEPSChart();
        }

        private void PopulateEPSChart()
        {
            if (_earningsData?.QuarterlyEarnings == null || _earningsData.QuarterlyEarnings.Count == 0)
            {
                EPSChart.Series = null;
                return;
            }

            try
            {
                // Get the last 12 quarters for the chart (most recent first, so reverse for chronological order)
                var quarterlyData = _earningsData.QuarterlyEarnings
                    .Where(q => q.ReportedEPS.HasValue)
                    .Take(12)
                    .Reverse()
                    .ToList();

                if (quarterlyData.Count == 0) return;

                var reportedValues = new ChartValues<double>();
                var estimatedValues = new ChartValues<double>();
                var labels = new List<string>();

                foreach (var quarter in quarterlyData)
                {
                    reportedValues.Add((double)(quarter.ReportedEPS ?? 0));
                    estimatedValues.Add((double)(quarter.EstimatedEPS ?? 0));
                    
                    // Format the date label
                    if (DateTime.TryParse(quarter.FiscalDateEnding, out var date))
                    {
                        labels.Add(date.ToString("MMM yy"));
                    }
                    else
                    {
                        labels.Add(quarter.FiscalDateEnding ?? "");
                    }
                }

                EPSChart.Series = new SeriesCollection
                {
                    new LineSeries
                    {
                        Title = "Reported EPS",
                        Values = reportedValues,
                        Stroke = System.Windows.Media.Brushes.Cyan,
                        Fill = System.Windows.Media.Brushes.Transparent,
                        PointGeometry = DefaultGeometries.Circle,
                        PointGeometrySize = 8
                    },
                    new LineSeries
                    {
                        Title = "Estimated EPS",
                        Values = estimatedValues,
                        Stroke = System.Windows.Media.Brushes.Orange,
                        Fill = System.Windows.Media.Brushes.Transparent,
                        PointGeometry = DefaultGeometries.Square,
                        PointGeometrySize = 6,
                        StrokeDashArray = new System.Windows.Media.DoubleCollection { 4, 2 }
                    }
                };

                // Update X axis labels
                if (EPSChart.AxisX != null && EPSChart.AxisX.Count > 0)
                {
                    EPSChart.AxisX[0].Labels = labels;
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Error populating EPS chart");
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
                await LoadEarningsData();
            }
        }
    }
}
