using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Defaults;
using Quantra.Models;
using Quantra.DAL.Models;

namespace Quantra.Controls
{
    /// <summary>
    /// Candlestick chart implementation for multi-horizon forecasting
    /// Displays OHLCV data with TFT prediction bands
    /// </summary>
    public partial class PredictionAnalysis : UserControl
    {
        // Candlestick chart properties
        private SeriesCollection _candlestickSeriesCollection;
        private SeriesCollection _volumeSeriesCollection;
        private List<string> _chartDateLabels;
        private bool _isChartVisible = false;

        /// <summary>
        /// Gets or sets the candlestick series collection for binding
        /// </summary>
        public SeriesCollection CandlestickSeriesCollection
        {
            get => _candlestickSeriesCollection;
            set
            {
                _candlestickSeriesCollection = value;
                OnPropertyChanged(nameof(CandlestickSeriesCollection));
            }
        }

        /// <summary>
        /// Gets or sets the volume series collection for binding
        /// </summary>
        public SeriesCollection VolumeSeriesCollection
        {
            get => _volumeSeriesCollection;
            set
            {
                _volumeSeriesCollection = value;
                OnPropertyChanged(nameof(VolumeSeriesCollection));
            }
        }

        /// <summary>
        /// Gets or sets the chart date labels for X-axis
        /// </summary>
        public List<string> ChartDateLabels
        {
            get => _chartDateLabels;
            set
            {
                _chartDateLabels = value;
                OnPropertyChanged(nameof(ChartDateLabels));
            }
        }

        /// <summary>
        /// Gets or sets whether the chart is visible
        /// </summary>
        public bool IsChartVisible
        {
            get => _isChartVisible;
            set
            {
                _isChartVisible = value;
                OnPropertyChanged(nameof(IsChartVisible));
            }
        }

        /// <summary>
        /// Volume formatter for Y-axis labels
        /// </summary>
        public Func<double, string> VolumeFormatter => value =>
        {
            if (value >= 1_000_000_000)
                return $"{value / 1_000_000_000:F1}B";
            if (value >= 1_000_000)
                return $"{value / 1_000_000:F1}M";
            if (value >= 1_000)
                return $"{value / 1_000:F1}K";
            return value.ToString("F0");
        };

        /// <summary>
        /// Initialize the candlestick chart components
        /// </summary>
        private void InitializeCandlestickChart()
        {
            _candlestickSeriesCollection = new SeriesCollection();
            _volumeSeriesCollection = new SeriesCollection();
            _chartDateLabels = new List<string>();
            _isChartVisible = false;
        }

        /// <summary>
        /// Update the candlestick chart with OHLCV data and TFT predictions
        /// </summary>
        /// <param name="symbol">Stock symbol</param>
        /// <param name="historicalData">Historical OHLCV data</param>
        /// <param name="tftResult">TFT prediction result with multi-horizon forecasts</param>
        public void UpdateCandlestickChart(string symbol, List<HistoricalPrice> historicalData, TFTPredictionResult tftResult = null)
        {
            try
            {
                if (historicalData == null || historicalData.Count == 0)
                {
                    _loggingService?.Log("Warning", "No historical data available for candlestick chart");
                    IsChartVisible = false;
                    return;
                }

                // Sort data by date
                var sortedData = historicalData.OrderBy(h => h.Date).ToList();

                // Create new series collections
                var candlestickSeries = new SeriesCollection();
                var volumeSeries = new SeriesCollection();
                var dateLabels = new List<string>();

                // 1. Add Historical Candlesticks
                var historicalCandles = new ChartValues<OhlcPoint>();
                foreach (var data in sortedData)
                {
                    historicalCandles.Add(new OhlcPoint(data.Open, data.High, data.Low, data.Close));
                    dateLabels.Add(data.Date.ToString("MM/dd"));
                }

                var candleSeries = new CandleSeries
                {
                    Title = $"{symbol} - Historical",
                    Values = historicalCandles,
                    MaxColumnWidth = 10,
                    IncreaseBrush = new SolidColorBrush(Color.FromRgb(0x90, 0xEE, 0x90)), // Light green
                    DecreaseBrush = new SolidColorBrush(Color.FromRgb(0xF0, 0x80, 0x80))  // Light red
                };
                candlestickSeries.Add(candleSeries);

                // 2. Add TFT Predictions (if available)
                if (tftResult != null && tftResult.Horizons != null && tftResult.Horizons.Count > 0)
                {
                    // Get the last date and price from historical data
                    var lastDate = sortedData.Last().Date;
                    var lastPrice = sortedData.Last().Close;

                    // Extract multi-horizon predictions
                    var horizonKeys = new[] { "5d", "10d", "20d", "30d" };
                    var predictionDates = new List<DateTime>();
                    var medianPrices = new List<double>();
                    var upperBounds = new List<double>();
                    var lowerBounds = new List<double>();

                    foreach (var key in horizonKeys)
                    {
                        if (tftResult.Horizons.TryGetValue(key, out var horizonData))
                        {
                            int daysAhead = int.Parse(key.Replace("d", ""));
                            var predictionDate = lastDate.AddDays(daysAhead);
                            
                            predictionDates.Add(predictionDate);
                            medianPrices.Add(horizonData.MedianPrice);
                            upperBounds.Add(horizonData.UpperBound);
                            lowerBounds.Add(horizonData.LowerBound);
                            
                            dateLabels.Add(predictionDate.ToString("MM/dd") + "*");
                        }
                    }

                    // Add median prediction line
                    var predictionLine = new ChartValues<double>();
                    predictionLine.AddRange(Enumerable.Repeat(double.NaN, sortedData.Count)); // Historical portion = NaN
                    predictionLine.Add(lastPrice); // Connect to last historical price
                    predictionLine.AddRange(medianPrices);

                    var medianSeries = new LineSeries
                    {
                        Title = "TFT Median Forecast",
                        Values = predictionLine,
                        Stroke = new SolidColorBrush(Color.FromRgb(0x1E, 0x90, 0xFF)), // Dodger blue
                        Fill = Brushes.Transparent,
                        StrokeThickness = 3,
                        PointGeometry = DefaultGeometries.Circle,
                        PointGeometrySize = 8,
                        LineSmoothness = 0.3
                    };
                    candlestickSeries.Add(medianSeries);

                    // Add upper confidence band
                    var upperBandValues = new ChartValues<double>();
                    upperBandValues.AddRange(Enumerable.Repeat(double.NaN, sortedData.Count));
                    upperBandValues.Add(lastPrice);
                    upperBandValues.AddRange(upperBounds);

                    var upperBandSeries = new LineSeries
                    {
                        Title = "Upper 90% CI",
                        Values = upperBandValues,
                        Stroke = new SolidColorBrush(Color.FromRgb(0xFF, 0xA5, 0x00)), // Orange
                        Fill = Brushes.Transparent,
                        StrokeThickness = 2,
                        StrokeDashArray = new System.Windows.Media.DoubleCollection(new[] { 4.0, 2.0 }),
                        PointGeometry = null,
                        LineSmoothness = 0.3
                    };
                    candlestickSeries.Add(upperBandSeries);

                    // Add lower confidence band
                    var lowerBandValues = new ChartValues<double>();
                    lowerBandValues.AddRange(Enumerable.Repeat(double.NaN, sortedData.Count));
                    lowerBandValues.Add(lastPrice);
                    lowerBandValues.AddRange(lowerBounds);

                    var lowerBandSeries = new LineSeries
                    {
                        Title = "Lower 10% CI",
                        Values = lowerBandValues,
                        Stroke = new SolidColorBrush(Color.FromRgb(0xFF, 0xA5, 0x00)), // Orange
                        Fill = Brushes.Transparent,
                        StrokeThickness = 2,
                        StrokeDashArray = new System.Windows.Media.DoubleCollection(new[] { 4.0, 2.0 }),
                        PointGeometry = null,
                        LineSmoothness = 0.3
                    };
                    candlestickSeries.Add(lowerBandSeries);

                    // Add prediction candles (visualize future OHLC estimates)
                    var futureCandles = new ChartValues<OhlcPoint>();
                    for (int i = 0; i < medianPrices.Count; i++)
                    {
                        double median = medianPrices[i];
                        double upper = upperBounds[i];
                        double lower = lowerBounds[i];
                        
                        // Estimate OHLC from median and bounds
                        double open = i == 0 ? lastPrice : medianPrices[i - 1];
                        double close = median;
                        double high = upper;
                        double low = lower;

                        futureCandles.Add(new OhlcPoint(open, high, low, close));
                    }

                    var futureCandleSeries = new CandleSeries
                    {
                        Title = $"{symbol} - Forecast",
                        Values = futureCandles,
                        MaxColumnWidth = 10,
                        IncreaseBrush = new SolidColorBrush(Color.FromArgb(128, 0x90, 0xEE, 0x90)), // Semi-transparent green
                        DecreaseBrush = new SolidColorBrush(Color.FromArgb(128, 0xF0, 0x80, 0x80)), // Semi-transparent red
                        ScalesYAt = 0  // Same Y-axis as historical
                    };
                    
                    // Offset future candles to appear after historical data
                    for (int i = 0; i < sortedData.Count; i++)
                    {
                        futureCandleSeries.Values.Insert(0, new OhlcPoint(double.NaN, double.NaN, double.NaN, double.NaN));
                    }
                    candlestickSeries.Add(futureCandleSeries);
                }

                // 3. Add Volume Bars
                var volumeValues = new ChartValues<double>();
                var volumeColors = new List<Brush>();
                
                for (int i = 0; i < sortedData.Count; i++)
                {
                    volumeValues.Add(sortedData[i].Volume);
                    
                    // Color based on price movement
                    if (i > 0)
                    {
                        bool isIncrease = sortedData[i].Close >= sortedData[i - 1].Close;
                        volumeColors.Add(isIncrease 
                            ? new SolidColorBrush(Color.FromRgb(0x90, 0xEE, 0x90)) 
                            : new SolidColorBrush(Color.FromRgb(0xF0, 0x80, 0x80)));
                    }
                    else
                    {
                        volumeColors.Add(new SolidColorBrush(Color.FromRgb(0x80, 0x80, 0x80))); // Gray for first bar
                    }
                }

                var volumeColumnSeries = new ColumnSeries
                {
                    Title = "Volume",
                    Values = volumeValues,
                    Fill = new SolidColorBrush(Color.FromRgb(0x60, 0x60, 0x80)), // Darker gray-blue
                    MaxColumnWidth = 15
                };
                volumeSeries.Add(volumeColumnSeries);

                // Update properties
                CandlestickSeriesCollection = candlestickSeries;
                VolumeSeriesCollection = volumeSeries;
                ChartDateLabels = dateLabels;
                IsChartVisible = true;

                // Update chart symbol text if control exists
                var chartSymbolText = this.FindName("ChartSymbolText") as TextBlock;
                if (chartSymbolText != null)
                {
                    chartSymbolText.Text = $" - {symbol}";
                }

                _loggingService?.Log("Info", $"Updated candlestick chart for {symbol} with {sortedData.Count} historical bars" + 
                    (tftResult != null ? $" and {tftResult.Horizons?.Count ?? 0} forecast horizons" : ""));
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to update candlestick chart");
                IsChartVisible = false;
            }
        }

        /// <summary>
        /// Clear the candlestick chart
        /// </summary>
        public void ClearCandlestickChart()
        {
            CandlestickSeriesCollection?.Clear();
            VolumeSeriesCollection?.Clear();
            ChartDateLabels?.Clear();
            IsChartVisible = false;

            var chartSymbolText = this.FindName("ChartSymbolText") as TextBlock;
            if (chartSymbolText != null)
            {
                chartSymbolText.Text = "";
            }
        }
    }
}
