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

        private async Task LoadCandlestickDataAsync()
        {
            IsLoading = true;
            IsDataLoaded = false;
            IsNoData = false;
            
            try
            {
                _loggingService?.Log("Info", $"Loading candlestick data for {_symbol} with interval {_currentInterval}");
                
                // Get intraday data from Alpha Vantage
                var historicalData = await _alphaVantageService.GetIntradayData(
                    _symbol, 
                    _currentInterval, 
                    "compact", 
                    "json");
                
                if (historicalData == null || historicalData.Count == 0)
                {
                    _loggingService?.Log("Warning", $"No candlestick data available for {_symbol}");
                    IsNoData = true;
                    IsLoading = false;
                    return;
                }
                
                // Sort by date ascending
                var sortedData = historicalData.OrderBy(h => h.Date).ToList();
                
                // Limit to last 100 candles for performance
                if (sortedData.Count > 100)
                {
                    sortedData = sortedData.Skip(sortedData.Count - 100).ToList();
                }
                
                // Update chart
                UpdateChartWithData(sortedData);
                
                // Update price information
                UpdatePriceInfo(sortedData);
                
                _lastUpdateTime = DateTime.Now;
                IsDataLoaded = true;
                IsNoData = false;
                
                OnPropertyChanged(nameof(LastUpdateText));
                OnPropertyChanged(nameof(StatusText));
                OnPropertyChanged(nameof(PriceChangeColor));
                
                _loggingService?.Log("Info", $"Successfully loaded {sortedData.Count} candles for {_symbol}");
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to load candlestick data for {_symbol}");
                IsNoData = true;
            }
            finally
            {
                IsLoading = false;
            }
        }

        private void UpdateChartWithData(List<HistoricalPrice> data)
        {
            // Create candlestick series
            var candleValues = new ChartValues<OhlcPoint>();
            var volumeValues = new ChartValues<double>();
            var labels = new List<string>();
            
            foreach (var candle in data)
            {
                candleValues.Add(new OhlcPoint(candle.Open, candle.High, candle.Low, candle.Close));
                volumeValues.Add(candle.Volume);
                labels.Add(candle.Date.ToString("HH:mm"));
            }
            
            // Update candlestick series
            CandlestickSeries = new SeriesCollection
            {
                new CandleSeries
                {
                    Title = _symbol,
                    Values = candleValues,
                    MaxColumnWidth = 20,
                    IncreaseBrush = new SolidColorBrush(Color.FromRgb(0x20, 0xC0, 0x40)), // Green
                    DecreaseBrush = new SolidColorBrush(Color.FromRgb(0xC0, 0x20, 0x20))  // Red
                }
            };
            
            // Update volume series
            VolumeSeries = new SeriesCollection
            {
                new ColumnSeries
                {
                    Title = "Volume",
                    Values = volumeValues,
                    Fill = new SolidColorBrush(Color.FromArgb(128, 0x60, 0x60, 0x80)), // Semi-transparent gray-blue
                    MaxColumnWidth = 20
                }
            };
            
            TimeLabels = labels;
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
            await LoadCandlestickDataAsync();
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
                    
                    // Reload data with new interval
                    await LoadCandlestickDataAsync();
                }
            }
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
