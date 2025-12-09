using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Threading;
using LiveCharts;
using LiveCharts.Wpf;
using LiveCharts.Defaults;
using Quantra.DAL.Services;
using Quantra.Models;

namespace Quantra.Controls
{
    /// <summary>
    /// Candlestick Chart Modal Window for Real-Time Stock Data Visualization
    /// </summary>
    public partial class CandlestickChartModal : Window, INotifyPropertyChanged
    {
        #region Fields

        private readonly AlphaVantageService _alphaVantageService;
        private readonly LoggingService _loggingService;
        private readonly string _symbol;
        private DispatcherTimer _refreshTimer;
        private CancellationTokenSource _cancellationTokenSource;
        
        private const int REFRESH_INTERVAL_SECONDS = 15; // 15 seconds to respect API limits
        private const int API_RATE_LIMIT_CALLS = 5; // 5 calls per minute for free tier
        private const int CACHE_DURATION_SECONDS = 10; // Cache data for 10 seconds
        
        private SeriesCollection _candlestickSeries;
        private SeriesCollection _volumeSeries;
        private List<string> _timeLabels;
        private bool _isAutoRefreshEnabled = true;
        private bool _isLoading = false;
        private bool _isDataLoaded = false;
        private bool _isNoData = false;
        private string _currentInterval = "5min";
        private DateTime _lastUpdateTime = DateTime.MinValue;
        private double _lastPrice = 0;
        private double _priceChange = 0;
        private double _priceChangePercent = 0;
        private int _maxCandles = 100;
        
        // Caching
        private List<HistoricalPrice> _cachedData;
        private DateTime _cacheTimestamp = DateTime.MinValue;
        private string _cachedInterval;
        
        // Zoom/Pan
        private double? _xAxisMin;
        private double? _xAxisMax;
        private double? _yAxisMin;
        private double? _yAxisMax;
        private double _zoomLevel = 1.0;

        #endregion

        #region Properties

        public event PropertyChangedEventHandler PropertyChanged;

        public string Symbol => _symbol;

        public string WindowTitle => $"Real-Time Candlestick Chart - {_symbol}";

        public SeriesCollection CandlestickSeries
        {
            get => _candlestickSeries;
            set
            {
                _candlestickSeries = value;
                OnPropertyChanged(nameof(CandlestickSeries));
            }
        }

        public SeriesCollection VolumeSeries
        {
            get => _volumeSeries;
            set
            {
                _volumeSeries = value;
                OnPropertyChanged(nameof(VolumeSeries));
            }
        }

        public List<string> TimeLabels
        {
            get => _timeLabels;
            set
            {
                _timeLabels = value;
                OnPropertyChanged(nameof(TimeLabels));
            }
        }

        public bool IsAutoRefreshEnabled
        {
            get => _isAutoRefreshEnabled;
            set
            {
                _isAutoRefreshEnabled = value;
                OnPropertyChanged(nameof(IsAutoRefreshEnabled));
                OnPropertyChanged(nameof(AutoRefreshText));
                
                if (value)
                    StartAutoRefresh();
                else
                    StopAutoRefresh();
            }
        }

        public string AutoRefreshText => IsAutoRefreshEnabled ? "ON" : "OFF";

        public bool IsLoading
        {
            get => _isLoading;
            set
            {
                _isLoading = value;
                OnPropertyChanged(nameof(IsLoading));
            }
        }

        public bool IsDataLoaded
        {
            get => _isDataLoaded;
            set
            {
                _isDataLoaded = value;
                OnPropertyChanged(nameof(IsDataLoaded));
            }
        }

        public bool IsNoData
        {
            get => _isNoData;
            set
            {
                _isNoData = value;
                OnPropertyChanged(nameof(IsNoData));
            }
        }

        public double LastPrice
        {
            get => _lastPrice;
            set
            {
                _lastPrice = value;
                OnPropertyChanged(nameof(LastPrice));
            }
        }

        public double PriceChange
        {
            get => _priceChange;
            set
            {
                _priceChange = value;
                OnPropertyChanged(nameof(PriceChange));
            }
        }

        public double PriceChangePercent
        {
            get => _priceChangePercent;
            set
            {
                _priceChangePercent = value;
                OnPropertyChanged(nameof(PriceChangePercent));
            }
        }

        public Brush PriceChangeColor => PriceChange >= 0 
            ? new SolidColorBrush(Color.FromRgb(0x20, 0xC0, 0x40)) // Green
            : new SolidColorBrush(Color.FromRgb(0xC0, 0x20, 0x20)); // Red

        public string LastUpdateText => _lastUpdateTime == DateTime.MinValue 
            ? "Never updated" 
            : $"Last update: {_lastUpdateTime:HH:mm:ss}";

        public string StatusText => IsLoading 
            ? "Loading data..." 
            : $"Showing {_timeLabels?.Count ?? 0} candles";

        public string ApiUsageText => $"API Rate Limit: {API_RATE_LIMIT_CALLS}/min";

        public Func<double, string> PriceFormatter => value => $"${value:F2}";

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

        public bool IsCacheValid => _cachedData != null && 
                                    _cachedInterval == _currentInterval && 
                                    (DateTime.Now - _cacheTimestamp).TotalSeconds < CACHE_DURATION_SECONDS;

        public string CacheStatusText => IsCacheValid 
            ? $"Cached ({(int)(CACHE_DURATION_SECONDS - (DateTime.Now - _cacheTimestamp).TotalSeconds)}s)" 
            : "Live";

        public double? XAxisMin
        {
            get => _xAxisMin;
            set
            {
                _xAxisMin = value;
                OnPropertyChanged(nameof(XAxisMin));
            }
        }

        public double? XAxisMax
        {
            get => _xAxisMax;
            set
            {
                _xAxisMax = value;
                OnPropertyChanged(nameof(XAxisMax));
            }
        }

        public double? YAxisMin
        {
            get => _yAxisMin;
            set
            {
                _yAxisMin = value;
                OnPropertyChanged(nameof(YAxisMin));
            }
        }

        public double? YAxisMax
        {
            get => _yAxisMax;
            set
            {
                _yAxisMax = value;
                OnPropertyChanged(nameof(YAxisMax));
            }
        }

        #endregion

        #region Constructor

        public CandlestickChartModal(string symbol, AlphaVantageService alphaVantageService, LoggingService loggingService)
        {
            // CRITICAL: Assign services BEFORE InitializeComponent to prevent null reference
            // InitializeComponent triggers XAML binding events (like IntervalComboBox_SelectionChanged)
            _symbol = symbol;
            _alphaVantageService = alphaVantageService ?? throw new ArgumentNullException(nameof(alphaVantageService));
            _loggingService = loggingService;
            _cancellationTokenSource = new CancellationTokenSource();
            
            // Now initialize XAML (this may trigger event handlers that need _alphaVantageService)
            InitializeComponent();
            
            DataContext = this;
            
            InitializeChart();
            
            // Start with initial data load
            Loaded += async (s, e) => await LoadCandlestickDataAsync();
        }

        #endregion

        #region Initialization

        private void InitializeChart()
        {
            _candlestickSeries = new SeriesCollection();
            _volumeSeries = new SeriesCollection();
            _timeLabels = new List<string>();
            
            IsLoading = false;
            IsDataLoaded = false;
            IsNoData = false;
            
            _loggingService?.Log("Info", $"Initialized candlestick chart modal for {_symbol}");
        }

        #endregion

        #region Data Loading

        private async Task LoadCandlestickDataAsync(bool forceRefresh = false)
        {
            // Check cache first
            if (!forceRefresh && IsCacheValid)
            {
                _loggingService?.Log("Info", $"Using cached data for {_symbol}");
                await UpdateChartAsync(_cachedData).ConfigureAwait(false);
                
                await Dispatcher.InvokeAsync(() =>
                {
                    OnPropertyChanged(nameof(IsCacheValid));
                    OnPropertyChanged(nameof(CacheStatusText));
                });
                return;
            }
            
            await Dispatcher.InvokeAsync(() =>
            {
                IsLoading = true;
                IsDataLoaded = false;
                IsNoData = false;
            });
            
            try
            {
                _loggingService?.Log("Info", $"Loading candlestick data for {_symbol} with interval {_currentInterval}");
                
                // Get intraday data from Alpha Vantage (run on background thread)
                var historicalData = await Task.Run(async () => 
                    await _alphaVantageService.GetIntradayData(
                        _symbol, 
                        _currentInterval, 
                        "compact", 
                        "json").ConfigureAwait(false)
                ).ConfigureAwait(false);
                
                if (historicalData == null || historicalData.Count == 0)
                {
                    _loggingService?.Log("Warning", $"No candlestick data available for {_symbol}");
                    await Dispatcher.InvokeAsync(() =>
                    {
                        IsNoData = true;
                        IsLoading = false;
                    });
                    return;
                }
                
                // Cache the data
                _cachedData = historicalData;
                _cacheTimestamp = DateTime.Now;
                _cachedInterval = _currentInterval;
                
                // Update chart on UI thread
                await UpdateChartAsync(historicalData).ConfigureAwait(false);
                
                await Dispatcher.InvokeAsync(() =>
                {
                    _lastUpdateTime = DateTime.Now;
                    IsDataLoaded = true;
                    IsNoData = false;
                    
                    OnPropertyChanged(nameof(LastUpdateText));
                    OnPropertyChanged(nameof(StatusText));
                    OnPropertyChanged(nameof(PriceChangeColor));
                    OnPropertyChanged(nameof(IsCacheValid));
                    OnPropertyChanged(nameof(CacheStatusText));
                });
                
                _loggingService?.Log("Info", $"Successfully loaded {historicalData.Count} candles for {_symbol}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to load candlestick data for {_symbol}");
                await Dispatcher.InvokeAsync(() => IsNoData = true);
            }
            finally
            {
                await Dispatcher.InvokeAsync(() => IsLoading = false);
            }
        }

        private async Task UpdateChartAsync(List<HistoricalPrice> historicalData)
        {
            // Process data on background thread
            var (sortedData, candleValues, volumeValues, labels, volumeColors, gaps) = await Task.Run(() =>
            {
                // Sort by date ascending
                var sorted = historicalData.OrderBy(h => h.Date).ToList();
                
                // Limit to configured max candles for performance
                if (_maxCandles > 0 && sorted.Count > _maxCandles)
                {
                    sorted = sorted.Skip(sorted.Count - _maxCandles).ToList();
                }
                
                // Create chart data with enhanced features
                var candles = new ChartValues<OhlcPoint>();
                var volumes = new ChartValues<double>();
                var timeLabels = new List<string>();
                var volColors = new List<Brush>();
                var gapIndices = new List<int>();
                
                DateTime? previousDate = null;
                
                for (int i = 0; i < sorted.Count; i++)
                {
                    var candle = sorted[i];
                    
                    // Add OHLCV data
                    candles.Add(new OhlcPoint(candle.Open, candle.High, candle.Low, candle.Close));
                    volumes.Add(candle.Volume);
                    
                    // Enhanced time labels with date when day changes
                    string timeLabel;
                    if (previousDate == null || previousDate.Value.Date != candle.Date.Date)
                    {
                        // Show date and time when day changes
                        timeLabel = candle.Date.ToString("MM/dd\nHH:mm");
                    }
                    else
                    {
                        // Just show time for same day
                        timeLabel = candle.Date.ToString("HH:mm");
                    }
                    timeLabels.Add(timeLabel);
                    
                    // Dynamic volume coloring based on price action
                    bool isPriceUp = candle.Close >= candle.Open;
                    var volumeColor = isPriceUp 
                        ? new SolidColorBrush(Color.FromArgb(128, 0x20, 0xC0, 0x40)) // Green (buying pressure)
                        : new SolidColorBrush(Color.FromArgb(128, 0xC0, 0x20, 0x20)); // Red (selling pressure)
                    volColors.Add(volumeColor);
                    
                    // Detect gaps (significant time jumps between candles)
                    if (previousDate != null)
                    {
                        var expectedInterval = GetExpectedIntervalMinutes(_currentInterval);
                        var actualInterval = (candle.Date - previousDate.Value).TotalMinutes;
                        
                        // If gap is more than 2x the expected interval, mark it
                        if (actualInterval > expectedInterval * 2)
                        {
                            gapIndices.Add(i);
                        }
                    }
                    
                    previousDate = candle.Date;
                }
                
                return (sorted, candles, volumes, timeLabels, volColors, gapIndices);
            }).ConfigureAwait(false);
            
            // Update UI on UI thread
            await Dispatcher.InvokeAsync(() =>
            {
                UpdateChartWithData(sortedData, candleValues, volumeValues, labels, volumeColors, gaps);
                UpdatePriceInfo(sortedData);
            });
        }
        
        private int GetExpectedIntervalMinutes(string interval)
        {
            return interval switch
            {
                "1min" => 1,
                "5min" => 5,
                "15min" => 15,
                "30min" => 30,
                "60min" => 60,
                _ => 5
            };
        }

        private void UpdateChartWithData(List<HistoricalPrice> data, ChartValues<OhlcPoint> candleValues, ChartValues<double> volumeValues, List<string> labels, List<Brush> volumeColors, List<int> gapIndices)
        {
            // Create candlestick series with tooltips
            var candleSeries = new CandleSeries
            {
                Title = _symbol,
                Values = candleValues,
                MaxColumnWidth = 20,
                IncreaseBrush = new SolidColorBrush(Color.FromRgb(0x20, 0xC0, 0x40)), // Green
                DecreaseBrush = new SolidColorBrush(Color.FromRgb(0xC0, 0x20, 0x20)),  // Red
                LabelPoint = point =>
                {
                    // Enhanced tooltip with OHLCV details
                    var ohlc = (OhlcPoint)point.Instance;
                    var index = (int)point.X;
                    if (index >= 0 && index < data.Count)
                    {
                        var candle = data[index];
                        var change = candle.Close - candle.Open;
                        var changePercent = candle.Open != 0 ? (change / candle.Open) * 100 : 0;
                        var direction = change >= 0 ? "?" : "?";
                        
                        // Check if this is a gap candle
                        var gapIndicator = gapIndices.Contains(index) ? " [GAP]" : "";
                        
                        // Check if after-hours (before 9:30 AM or after 4:00 PM ET)
                        var hour = candle.Date.Hour;
                        var afterHoursIndicator = (hour < 9 || (hour == 9 && candle.Date.Minute < 30) || hour >= 16) 
                            ? " [AH]" : "";
                        
                        return $"{candle.Date:MM/dd HH:mm}{gapIndicator}{afterHoursIndicator}\n" +
                               $"Open:  ${ohlc.Open:F2}\n" +
                               $"High:  ${ohlc.High:F2}\n" +
                               $"Low:   ${ohlc.Low:F2}\n" +
                               $"Close: ${ohlc.Close:F2}\n" +
                               $"Volume: {VolumeFormatter(candle.Volume)}\n" +
                               $"Change: {direction} ${Math.Abs(change):F2} ({changePercent:+0.00;-0.00;0.00}%)";
                    }
                    return string.Empty;
                }
            };
            
            CandlestickSeries = new SeriesCollection { candleSeries };
            
            // Create volume series with dynamic coloring
            var volumeSeries = new SeriesCollection();
            
            // If we have individual colors for each volume bar, create separate series
            // (LiveCharts limitation: can't color individual bars in one series easily)
            // For simplicity, we'll use two series: one for up volumes, one for down volumes
            var upVolumes = new ChartValues<double>();
            var downVolumes = new ChartValues<double>();
            
            for (int i = 0; i < volumeValues.Count; i++)
            {
                bool isUp = i < data.Count && data[i].Close >= data[i].Open;
                
                if (isUp)
                {
                    upVolumes.Add(volumeValues[i]);
                    downVolumes.Add(0);
                }
                else
                {
                    upVolumes.Add(0);
                    downVolumes.Add(volumeValues[i]);
                }
            }
            
            volumeSeries.Add(new ColumnSeries
            {
                Title = "Buy Volume",
                Values = upVolumes,
                Fill = new SolidColorBrush(Color.FromArgb(128, 0x20, 0xC0, 0x40)), // Green
                MaxColumnWidth = 20,
                LabelPoint = point => 
                {
                    if (point.Y > 0)
                    {
                        var index = (int)point.X;
                        if (index >= 0 && index < data.Count)
                        {
                            return $"{data[index].Date:HH:mm}\nVolume: {VolumeFormatter(point.Y)}\n(Buying Pressure)";
                        }
                    }
                    return string.Empty;
                }
            });
            
            volumeSeries.Add(new ColumnSeries
            {
                Title = "Sell Volume",
                Values = downVolumes,
                Fill = new SolidColorBrush(Color.FromArgb(128, 0xC0, 0x20, 0x20)), // Red
                MaxColumnWidth = 20,
                LabelPoint = point => 
                {
                    if (point.Y > 0)
                    {
                        var index = (int)point.X;
                        if (index >= 0 && index < data.Count)
                        {
                            return $"{data[index].Date:HH:mm}\nVolume: {VolumeFormatter(point.Y)}\n(Selling Pressure)";
                        }
                    }
                    return string.Empty;
                }
            });
            
            VolumeSeries = volumeSeries;
            
            TimeLabels = labels;
            
            // Reset zoom when new data is loaded
            ResetZoom();
        }

        private void UpdatePriceInfo(List<HistoricalPrice> data)
        {
            if (data.Count < 2) return;
            
            var latest = data.Last();
            var previous = data[data.Count - 2];
            
            LastPrice = latest.Close;
            PriceChange = latest.Close - previous.Close;
            PriceChangePercent = previous.Close != 0 
                ? ((latest.Close - previous.Close) / previous.Close) * 100 
                : 0;
        }

        #endregion

        #region Auto-Refresh

        private void StartAutoRefresh()
        {
            if (_refreshTimer != null)
            {
                _refreshTimer.Stop();
                _refreshTimer = null;
            }
            
            _refreshTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(REFRESH_INTERVAL_SECONDS)
            };
            _refreshTimer.Tick += async (s, e) => await LoadCandlestickDataAsync();
            _refreshTimer.Start();
            
            _loggingService?.Log("Info", $"Started auto-refresh for {_symbol} with {REFRESH_INTERVAL_SECONDS}s interval");
        }

        private void StopAutoRefresh()
        {
            if (_refreshTimer != null)
            {
                _refreshTimer.Stop();
                _refreshTimer = null;
                _loggingService?.Log("Info", $"Stopped auto-refresh for {_symbol}");
            }
        }

        #endregion

        #region Event Handlers

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadCandlestickDataAsync(forceRefresh: true);
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private async void IntervalComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            // Prevent loading data during initialization (before window is fully loaded)
            if (!IsLoaded || _alphaVantageService == null)
                return;
                
            if (IntervalComboBox.SelectedItem is System.Windows.Controls.ComboBoxItem selected)
            {
                var content = selected.Content.ToString();
                var newInterval = content.Replace(" ", "").ToLower();
                
                // Only reload if interval actually changed
                if (newInterval != _currentInterval)
                {
                    _currentInterval = newInterval;
                    _loggingService?.Log("Info", $"Changed interval to {_currentInterval} for {_symbol}");
                    
                    // Invalidate cache and reload data with new interval
                    _cachedInterval = null;
                    await LoadCandlestickDataAsync(forceRefresh: true);
                }
            }
        }

        private async void CandleLimitComboBox_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            if (!IsLoaded || _alphaVantageService == null)
                return;
                
            if (CandleLimitComboBox.SelectedItem is System.Windows.Controls.ComboBoxItem selected)
            {
                var newLimit = int.Parse(selected.Tag.ToString());
                
                if (newLimit != _maxCandles)
                {
                    _maxCandles = newLimit;
                    _loggingService?.Log("Info", $"Changed candle limit to {(_maxCandles == 0 ? "All" : _maxCandles.ToString())} for {_symbol}");
                    
                    // Reload chart with cached data if available
                    if (_cachedData != null)
                    {
                        await UpdateChartAsync(_cachedData).ConfigureAwait(false);
                    }
                }
            }
        }

        private void ZoomInButton_Click(object sender, RoutedEventArgs e)
        {
            _zoomLevel *= 0.8; // Zoom in by 20%
            ApplyZoom();
        }

        private void ZoomOutButton_Click(object sender, RoutedEventArgs e)
        {
            _zoomLevel *= 1.2; // Zoom out by 20%
            ApplyZoom();
        }

        private void ResetZoomButton_Click(object sender, RoutedEventArgs e)
        {
            ResetZoom();
        }

        private void ApplyZoom()
        {
            if (_timeLabels == null || _timeLabels.Count == 0)
                return;

            int visibleCandles = (int)(_timeLabels.Count * _zoomLevel);
            if (visibleCandles < 10) visibleCandles = 10;
            if (visibleCandles > _timeLabels.Count) visibleCandles = _timeLabels.Count;

            int startIndex = Math.Max(0, _timeLabels.Count - visibleCandles);
            int endIndex = _timeLabels.Count - 1;

            XAxisMin = startIndex;
            XAxisMax = endIndex;

            _loggingService?.Log("Info", $"Applied zoom level {_zoomLevel:F2} (showing {visibleCandles} candles)");
        }

        private void ResetZoom()
        {
            _zoomLevel = 1.0;
            XAxisMin = null;
            XAxisMax = null;
            YAxisMin = null;
            YAxisMax = null;
        }

        #endregion

        #region Cleanup

        protected override void OnClosing(CancelEventArgs e)
        {
            StopAutoRefresh();
            _cancellationTokenSource?.Cancel();
            _cancellationTokenSource?.Dispose();
            
            base.OnClosing(e);
        }

        #endregion

        #region INotifyPropertyChanged

        protected void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }
}
