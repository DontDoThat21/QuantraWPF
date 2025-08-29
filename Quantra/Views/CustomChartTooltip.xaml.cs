using LiveCharts;
using LiveCharts.Defaults;
using LiveCharts.Wpf;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Quantra.Controls
{
    /// <summary>
    /// Interaction logic for CustomChartTooltip.xaml
    /// </summary>
    public partial class CustomChartTooltip : UserControl, IChartTooltip
    {
        // Collection of indicator values to display
        public ObservableCollection<IndicatorValueItem> IndicatorItemsSource { get; set; } = new ObservableCollection<IndicatorValueItem>();
        
        private TooltipData _data;
        public TooltipData Data 
        { 
            get => _data; 
            set 
            {
                _data = value;
                OnPropertyChanged(nameof(Data));
            }
        }
        
        private TooltipSelectionMode? _selectionMode;
        public TooltipSelectionMode? SelectionMode 
        { 
            get => _selectionMode; 
            set 
            {
                _selectionMode = value;
                OnPropertyChanged(nameof(SelectionMode));
            }
        }

        public CustomChartTooltip()
        {
            InitializeComponent();
            IndicatorValues.ItemsSource = IndicatorItemsSource;
        }

        public event PropertyChangedEventHandler? PropertyChanged;

        private void OnPropertyChanged(string propertyName) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));

        public void OnSeriesHover(ChartPoint chartPoint)
        {
            // Set datetime using improved extraction logic
            DateTimeText.Text = ExtractAndFormatDateTime(chartPoint);

            // Handle different series types
            if (chartPoint.SeriesView is CandleSeries)
            {
                // Show OHLC panel for candlesticks
                OHLCPanel.Visibility = Visibility.Visible;

                var ohlc = (OhlcPoint)chartPoint.Instance;
                OpenText.Text = ohlc.Open.ToString("F2");
                HighText.Text = ohlc.High.ToString("F2");
                LowText.Text = ohlc.Low.ToString("F2");
                CloseText.Text = ohlc.Close.ToString("F2");
                
                // Set colors based on whether it's an up or down candle
                var bullish = ohlc.Close >= ohlc.Open;
                CloseText.Foreground = bullish ? new SolidColorBrush(Color.FromRgb(0x90, 0xEE, 0x90)) : new SolidColorBrush(Color.FromRgb(0xF0, 0x80, 0x80));
            }
            else
            {
                // Hide OHLC panel for other series types
                OHLCPanel.Visibility = Visibility.Collapsed;
            }

            // Clear previous indicator values
            IndicatorItemsSource.Clear();

            // Add value for current series
            IndicatorItemsSource.Add(new IndicatorValueItem
            {
                Name = chartPoint.SeriesView.Title + ":",
                Value = FormatValue(chartPoint.SeriesView, chartPoint.Y),
                Color = GetBrush(chartPoint.SeriesView)
            });
        }

        public void OnSeriesHover(IDictionary<ChartPoint, ChartPoint[]> points)
        {
            // Get the first point to determine DateTime
            var firstPoint = points.Keys.FirstOrDefault();
            if (firstPoint != null)
            {
                // Set datetime using improved extraction logic
                DateTimeText.Text = ExtractAndFormatDateTime(firstPoint);
            }

            // Check if any CandleSeries is present
            bool hasCandleSeries = false;
            foreach (var kvp in points)
            {
                if (kvp.Key.SeriesView is CandleSeries)
                {
                    hasCandleSeries = true;
                    var ohlc = (OhlcPoint)kvp.Key.Instance;
                    OpenText.Text = ohlc.Open.ToString("F2");
                    HighText.Text = ohlc.High.ToString("F2");
                    LowText.Text = ohlc.Low.ToString("F2");
                    CloseText.Text = ohlc.Close.ToString("F2");
                    
                    // Set colors based on whether it's an up or down candle
                    var bullish = ohlc.Close >= ohlc.Open;
                    CloseText.Foreground = bullish ? new SolidColorBrush(Color.FromRgb(0x90, 0xEE, 0x90)) : new SolidColorBrush(Color.FromRgb(0xF0, 0x80, 0x80));
                }
            }

            // Show/hide OHLC panel
            OHLCPanel.Visibility = hasCandleSeries ? Visibility.Visible : Visibility.Collapsed;

            // Clear previous indicator values
            IndicatorItemsSource.Clear();

            // Add values for all series
            foreach (var kvp in points)
            {
                var point = kvp.Key;
                // Skip adding OHLC values if already displaying in panel
                if (point.SeriesView is CandleSeries)
                    continue;

                IndicatorItemsSource.Add(new IndicatorValueItem
                {
                    Name = point.SeriesView.Title + ":",
                    Value = FormatValue(point.SeriesView, point.Y),
                    Color = GetBrush(point.SeriesView)
                });
            }
        }

        private string ExtractAndFormatDateTime(ChartPoint chartPoint)
        {
            // Try multiple approaches to extract DateTime from chart point
            DateTime? dateTime = null;

            // Approach 1: Check if Instance is DateTime
            if (chartPoint.Instance is DateTime dt)
            {
                dateTime = dt;
            }
            // Approach 2: Check if Key is DateTime (use type check and safe cast)
            else if (chartPoint.Key != null && chartPoint.Key.GetType() == typeof(DateTime))
            {
                // Only cast if the type is exactly DateTime
                object keyObj = chartPoint.Key;
                if (keyObj is DateTime keyDateTime)
                {
                    dateTime = keyDateTime;
                }
            }
            // Approach 3: For OHLC data, try to extract from the data point
            else if (chartPoint.Instance is OhlcPoint ohlcPoint)
            {
                // Check if the OHLC point has a DateTime property or if we can derive it from X value
                if (chartPoint.X is double xValue)
                {
                    try
                    {
                        // Try to convert X value to DateTime (common in financial charts)
                        dateTime = DateTime.FromOADate(xValue);
                    }
                    catch
                    {
                        // If conversion fails, continue to fallback
                    }
                }
            }
            // Approach 4: Try to parse string representation (but skip if int)
            else if (chartPoint.Key != null && !(chartPoint.Key is int))
            {
                if (DateTime.TryParse(chartPoint.Key.ToString(), out DateTime parsedDate))
                {
                    dateTime = parsedDate;
                }
            }

            // Format the DateTime if we found one
            if (dateTime.HasValue)
            {
                return dateTime.Value.ToString("MMM dd, yyyy - HH:mm");
            }

            // Fallback for non-DateTime data
            if (chartPoint.Key is int intKey)
            {
                return $"Point: {intKey}";
            }
            else if (chartPoint.Key != null)
            {
                return $"Point: {chartPoint.Key}";
            }
            else
            {
                return "No date available";
            }
        }

        private string FormatValue(object seriesView, double value)
        {
            // Defensive: try to get Title property if possible
            string title = (seriesView as dynamic)?.Title as string ?? string.Empty;
            if (title.Contains("RSI") || title.Contains("Stoch"))
            {
                return value.ToString("F2");
            }
            else if (title.Contains("Volume"))
            {
                // Format volume with K, M, B suffixes
                return FormatLargeNumber(value);
            }
            else if (title.Contains("Bollinger") || title.Contains("Upper") || title.Contains("Lower") || title.Contains("Band"))
            {
                // Format Bollinger Bands with 2 decimal places (price format)
                return value.ToString("F2");
            }
            else
            {
                return value.ToString("F2");
            }
        }

        private Brush GetBrush(object seriesView)
        {
            // Extract color from series
            if (seriesView is LineSeries lineSeries)
            {
                return lineSeries.Stroke;
            }
            else if (seriesView is CandleSeries candleSeries)
            {
                return candleSeries.Stroke;
            }
            else if (seriesView is ColumnSeries columnSeries)
            {
                return columnSeries.Fill;
            }

            // Default color
            return Brushes.White;
        }

        private string FormatLargeNumber(double value)
        {
            if (value >= 1_000_000_000)
            {
                return (value / 1_000_000_000).ToString("F2") + "B";
            }
            else if (value >= 1_000_000)
            {
                return (value / 1_000_000).ToString("F2") + "M";
            }
            else if (value >= 1_000)
            {
                return (value / 1_000).ToString("F2") + "K";
            }
            else
            {
                return value.ToString("F0");
            }
        }
    }

    public class IndicatorValueItem
    {
        public string Name { get; set; }
        public string Value { get; set; }
        public Brush Color { get; set; } = Brushes.White;
    }
}