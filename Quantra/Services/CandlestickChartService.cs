using System;
using System.Collections.Generic;
using System.Linq;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Defaults;
using Quantra.Models;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Quantra.Services
{
    /// <summary>
    /// Service responsible for transforming historical price data into chart series.
    /// Separates chart rendering logic from UI and data access concerns.
    /// Enhanced with after-hours indicators, gap markers, volume heatmap, and average volume.
    /// </summary>
    public class CandlestickChartService
    {
        private readonly CandlestickChartColorScheme _colorScheme;
        private const double AFTER_HOURS_OPACITY = 0.6; // Dimmed for after-hours
        private const double VOLUME_HEATMAP_INTENSITY = 2.0; // Multiplier for high volume

        public CandlestickChartService(CandlestickChartColorScheme colorScheme = null)
        {
            _colorScheme = colorScheme ?? new CandlestickChartColorScheme();
        }

        /// <summary>
        /// Creates candlestick chart series from historical price data
        /// </summary>
        public (SeriesCollection CandlestickSeries, SeriesCollection VolumeSeries, List<string> TimeLabels) 
            CreateChartSeries(List<HistoricalPrice> historicalData, string symbol, int maxCandles = 0)
        {
            if (historicalData == null || historicalData.Count == 0)
            {
                return (new SeriesCollection(), new SeriesCollection(), new List<string>());
            }

            // Sort by date ascending
            var sortedData = historicalData.OrderBy(h => h.Date).ToList();

            // Limit to max candles if specified
            if (maxCandles > 0 && sortedData.Count > maxCandles)
            {
                sortedData = sortedData.Skip(sortedData.Count - maxCandles).ToList();
            }

            // Create chart data - separate regular and after-hours
            var regularCandleValues = new ChartValues<OhlcPoint>();
            var afterHoursCandleValues = new ChartValues<OhlcPoint>();
            var upVolumes = new ChartValues<double>();
            var downVolumes = new ChartValues<double>();
            var timeLabels = new List<string>();
            var gapIndices = new List<int>();
            var regularIndices = new List<int>();
            var afterHoursIndices = new List<int>();

            DateTime? previousDate = null;

            for (int i = 0; i < sortedData.Count; i++)
            {
                var candle = sortedData[i];
                var ohlcPoint = new OhlcPoint(candle.Open, candle.High, candle.Low, candle.Close);

                // Separate regular hours from after-hours
                if (IsAfterHours(candle.Date))
                {
                    afterHoursCandleValues.Add(ohlcPoint);
                    afterHoursIndices.Add(i);
                    regularCandleValues.Add(new OhlcPoint(0, 0, 0, 0)); // Placeholder
                }
                else
                {
                    regularCandleValues.Add(ohlcPoint);
                    regularIndices.Add(i);
                    afterHoursCandleValues.Add(new OhlcPoint(0, 0, 0, 0)); // Placeholder
                }

                // Separate volumes by price action
                bool isUp = candle.Close >= candle.Open;
                if (isUp)
                {
                    upVolumes.Add(candle.Volume);
                    downVolumes.Add(0);
                }
                else
                {
                    upVolumes.Add(0);
                    downVolumes.Add(candle.Volume);
                }

                // Enhanced time labels with date when day changes and after-hours indicator
                string timeLabel = FormatTimeLabel(candle.Date, previousDate);
                if (IsAfterHours(candle.Date))
                {
                    timeLabel += " AH";
                }
                timeLabels.Add(timeLabel);

                // Detect gaps
                if (previousDate != null && IsGap(candle.Date, previousDate.Value, GetIntervalMinutes(sortedData)))
                {
                    gapIndices.Add(i);
                }

                previousDate = candle.Date;
            }

            // Calculate average volume for comparison
            var avgVolume = sortedData.Average(d => d.Volume);

            // Create series collections
            var candlestickSeries = CreateCandlestickSeries(symbol, sortedData, 
                regularCandleValues, afterHoursCandleValues, 
                regularIndices, afterHoursIndices, gapIndices);
            var volumeSeries = CreateVolumeSeries(sortedData, upVolumes, downVolumes, avgVolume);

            return (candlestickSeries, volumeSeries, timeLabels);
        }

        /// <summary>
        /// Creates candlestick series with separate regular and after-hours series plus gap markers
        /// </summary>
        private SeriesCollection CreateCandlestickSeries(
            string symbol,
            List<HistoricalPrice> data,
            ChartValues<OhlcPoint> regularCandleValues,
            ChartValues<OhlcPoint> afterHoursCandleValues,
            List<int> regularIndices,
            List<int> afterHoursIndices,
            List<int> gapIndices)
        {
            var series = new SeriesCollection();
            
            // Regular hours candlestick series (brighter)
            var regularSeries = new CandleSeries
            {
                Title = $"{symbol} (Regular)",
                Values = regularCandleValues,
                MaxColumnWidth = 20,
                IncreaseBrush = _colorScheme.CandleUpBrush,
                DecreaseBrush = _colorScheme.CandleDownBrush,
                LabelPoint = point => CreateCandleTooltip(point, data, gapIndices, false)
            };
            
            series.Add(regularSeries);
            
            // After-hours candlestick series (dimmed for distinction)
            if (afterHoursIndices.Count > 0)
            {
                var afterHoursSeries = new CandleSeries
                {
                    Title = $"{symbol} (After-Hours)",
                    Values = afterHoursCandleValues,
                    MaxColumnWidth = 20,
                    IncreaseBrush = _colorScheme.CandleAfterHoursUpBrush,
                    DecreaseBrush = _colorScheme.CandleAfterHoursDownBrush,
                    LabelPoint = point => CreateCandleTooltip(point, data, gapIndices, true)
                };
                
                series.Add(afterHoursSeries);
            }
            
            // Add gap markers as scatter points (diamonds)
            if (gapIndices.Count > 0)
            {
                var gapMarkerValues = new ChartValues<ScatterPoint>();
                foreach (var gapIndex in gapIndices)
                {
                    if (gapIndex >= 0 && gapIndex < data.Count)
                    {
                        // Place marker at the high of the candle where gap occurs
                        gapMarkerValues.Add(new ScatterPoint(gapIndex, data[gapIndex].High * 1.02)); // Slightly above
                    }
                }
                
                if (gapMarkerValues.Count > 0)
                {
                    var gapMarkerSeries = new ScatterSeries
                    {
                        Title = "Market Gaps",
                        Values = gapMarkerValues,
                        MinPointShapeDiameter = 15,
                        MaxPointShapeDiameter = 15,
                        Fill = _colorScheme.GapMarkerBrush,
                        Stroke = Brushes.Black,
                        StrokeThickness = 1,
                        LabelPoint = point =>
                        {
                            var gapIndex = (int)point.X;
                            if (gapIndex >= 0 && gapIndex < data.Count)
                            {
                                return $"Market Gap\n{data[gapIndex].Date:MM/dd HH:mm}";
                            }
                            return "Gap";
                        }
                    };
                    
                    series.Add(gapMarkerSeries);
                }
            }

            return series;
        }

        /// <summary>
        /// Creates volume series with dynamic coloring, heatmap intensity, and average volume line
        /// </summary>
        private SeriesCollection CreateVolumeSeries(
            List<HistoricalPrice> data,
            ChartValues<double> upVolumes,
            ChartValues<double> downVolumes,
            double avgVolume)
        {
            // Apply heatmap intensity based on volume relative to average
            var upVolumesBrushes = new List<Brush>();
            var downVolumesBrushes = new List<Brush>();
            
            for (int i = 0; i < data.Count; i++)
            {
                var volumeRatio = data[i].Volume / avgVolume;
                var intensity = Math.Min(volumeRatio / VOLUME_HEATMAP_INTENSITY, 1.0);
                
                // Adjust alpha based on intensity (higher volume = more opaque)
                byte alpha = (byte)(64 + (intensity * 191)); // Range: 64-255
                
                bool isUp = data[i].Close >= data[i].Open;
                if (isUp)
                {
                    upVolumesBrushes.Add(new SolidColorBrush(Color.FromArgb(alpha, 0x20, 0xC0, 0x40)));
                    downVolumesBrushes.Add(Brushes.Transparent);
                }
                else
                {
                    upVolumesBrushes.Add(Brushes.Transparent);
                    downVolumesBrushes.Add(new SolidColorBrush(Color.FromArgb(alpha, 0xC0, 0x20, 0x20)));
                }
            }
            
            var volumeSeries = new SeriesCollection
            {
                new ColumnSeries
                {
                    Title = "Buy Volume",
                    Values = upVolumes,
                    Fill = _colorScheme.VolumeUpBrush,
                    MaxColumnWidth = 20,
                    LabelPoint = point => CreateVolumeTooltip(point, data, avgVolume, "Buying Pressure")
                },
                new ColumnSeries
                {
                    Title = "Sell Volume",
                    Values = downVolumes,
                    Fill = _colorScheme.VolumeDownBrush,
                    MaxColumnWidth = 20,
                    LabelPoint = point => CreateVolumeTooltip(point, data, avgVolume, "Selling Pressure")
                }
            };
            
            // Add average volume line
            var avgVolumeLine = new ChartValues<double>();
            for (int i = 0; i < data.Count; i++)
            {
                avgVolumeLine.Add(avgVolume);
            }
            
            volumeSeries.Add(new LineSeries
            {
                Title = "Avg Volume",
                Values = avgVolumeLine,
                Stroke = _colorScheme.AvgVolumeBrush,
                Fill = Brushes.Transparent,
                StrokeThickness = 2,
                StrokeDashArray = new System.Windows.Media.DoubleCollection(new[] { 3.0, 3.0 }),
                PointGeometry = null,
                LabelPoint = point => $"Average Volume: {FormatVolume(avgVolume)}"
            });

            return volumeSeries;
        }

        /// <summary>
        /// Formats time label based on date changes
        /// </summary>
        private string FormatTimeLabel(DateTime current, DateTime? previous)
        {
            if (previous == null || previous.Value.Date != current.Date)
            {
                return current.ToString("MM/dd\nHH:mm");
            }
            return current.ToString("HH:mm");
        }

        /// <summary>
        /// Detects if there is a significant gap between two dates
        /// </summary>
        private bool IsGap(DateTime current, DateTime previous, int expectedIntervalMinutes)
        {
            var actualInterval = (current - previous).TotalMinutes;
            return actualInterval > expectedIntervalMinutes * 2;
        }

        /// <summary>
        /// Gets expected interval in minutes from data
        /// </summary>
        private int GetIntervalMinutes(List<HistoricalPrice> data)
        {
            if (data.Count < 2) return 5;

            var intervals = new List<double>();
            for (int i = 1; i < Math.Min(10, data.Count); i++)
            {
                intervals.Add((data[i].Date - data[i - 1].Date).TotalMinutes);
            }

            return (int)intervals.Average();
        }

        /// <summary>
        /// Creates tooltip for candlestick
        /// </summary>
        private string CreateCandleTooltip(ChartPoint point, List<HistoricalPrice> data, List<int> gapIndices, bool isAfterHours)
        {
            var ohlc = (OhlcPoint)point.Instance;
            var index = (int)point.X;

            // Skip placeholder candles (all zeros)
            if (ohlc.Open == 0 && ohlc.High == 0 && ohlc.Low == 0 && ohlc.Close == 0)
            {
                return string.Empty;
            }

            if (index >= 0 && index < data.Count)
            {
                var candle = data[index];
                var change = candle.Close - candle.Open;
                var changePercent = candle.Open != 0 ? (change / candle.Open) * 100 : 0;
                var direction = change >= 0 ? "?" : "?";
                var gapIndicator = gapIndices.Contains(index) ? " [GAP]" : "";
                var afterHoursIndicator = IsAfterHours(candle.Date) ? " [AFTER-HOURS]" : "";

                return $"{candle.Date:MM/dd HH:mm}{gapIndicator}{afterHoursIndicator}\n" +
                       $"Open:  ${ohlc.Open:F2}\n" +
                       $"High:  ${ohlc.High:F2}\n" +
                       $"Low:   ${ohlc.Low:F2}\n" +
                       $"Close: ${ohlc.Close:F2}\n" +
                       $"Volume: {FormatVolume(candle.Volume)}\n" +
                       $"Change: {direction} ${Math.Abs(change):F2} ({changePercent:+0.00;-0.00;0.00}%)";
            }

            return string.Empty;
        }

        /// <summary>
        /// Creates tooltip for volume bar with average comparison
        /// </summary>
        private string CreateVolumeTooltip(ChartPoint point, List<HistoricalPrice> data, double avgVolume, string pressureType)
        {
            if (point.Y > 0)
            {
                var index = (int)point.X;
                if (index >= 0 && index < data.Count)
                {
                    var volumeRatio = data[index].Volume / avgVolume;
                    var comparison = volumeRatio > 1.0 ? $"+{(volumeRatio - 1.0) * 100:F0}%" : $"{(volumeRatio - 1.0) * 100:F0}%";
                    var intensity = volumeRatio > 1.5 ? "High" : volumeRatio > 1.0 ? "Above Avg" : volumeRatio > 0.5 ? "Below Avg" : "Low";
                    
                    return $"{data[index].Date:HH:mm}\n" +
                           $"Volume: {FormatVolume(point.Y)}\n" +
                           $"vs Avg: {comparison} ({intensity})\n" +
                           $"({pressureType})";
                }
            }
            return string.Empty;
        }

        /// <summary>
        /// Formats volume for display
        /// </summary>
        private string FormatVolume(double volume)
        {
            if (volume >= 1_000_000_000)
                return $"{volume / 1_000_000_000:F1}B";
            if (volume >= 1_000_000)
                return $"{volume / 1_000_000:F1}M";
            if (volume >= 1_000)
                return $"{volume / 1_000:F1}K";
            return volume.ToString("F0");
        }

        /// <summary>
        /// Checks if a date is in after-hours trading
        /// </summary>
        private bool IsAfterHours(DateTime date)
        {
            var hour = date.Hour;
            return hour < 9 || (hour == 9 && date.Minute < 30) || hour >= 16;
        }
    }

    /// <summary>
    /// Color scheme for candlestick chart
    /// </summary>
    public class CandlestickChartColorScheme
    {
        public Brush CandleUpBrush { get; set; } = new SolidColorBrush(Color.FromRgb(0x20, 0xC0, 0x40));
        public Brush CandleDownBrush { get; set; } = new SolidColorBrush(Color.FromRgb(0xC0, 0x20, 0x20));
        public Brush CandleAfterHoursUpBrush { get; set; } = new SolidColorBrush(Color.FromArgb(153, 0x20, 0xC0, 0x40)); // 60% opacity
        public Brush CandleAfterHoursDownBrush { get; set; } = new SolidColorBrush(Color.FromArgb(153, 0xC0, 0x20, 0x20)); // 60% opacity
        public Brush VolumeUpBrush { get; set; } = new SolidColorBrush(Color.FromArgb(128, 0x20, 0xC0, 0x40));
        public Brush VolumeDownBrush { get; set; } = new SolidColorBrush(Color.FromArgb(128, 0xC0, 0x20, 0x20));
        public Brush AvgVolumeBrush { get; set; } = new SolidColorBrush(Color.FromRgb(0xFF, 0xA5, 0x00)); // Orange
        public Brush GapMarkerBrush { get; set; } = new SolidColorBrush(Color.FromRgb(0xFF, 0xFF, 0x00)); // Yellow
    }
}
