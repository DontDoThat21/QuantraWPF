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
using Quantra.DAL.Services;
using Quantra.DAL.Services.Interfaces;
using Quantra.Exceptions;
using Quantra.Models;
using Quantra.Services;
using Polly;
using Polly.Retry;

namespace Quantra.ViewModels
{
    /// <summary>
    /// ViewModel for the Candlestick Chart Modal following MVVM pattern.
    /// Manages all business logic and state for the chart display.
    /// </summary>
    public class CandlestickChartViewModel : INotifyPropertyChanged
    {
        #region Fields

        private readonly ICandlestickDataService _dataService;
        private readonly CandlestickChartService _chartService;
        private readonly TechnicalIndicatorService _technicalIndicatorService;
        private readonly UserSettingsService _userSettingsService;
        private readonly LoggingService _loggingService;
        private readonly AsyncRetryPolicy _retryPolicy;
        private readonly string _symbol;
        
        private DispatcherTimer _refreshTimer;
        private CancellationTokenSource _cancellationTokenSource;
        private Task _currentLoadTask;
        private readonly SemaphoreSlim _requestSemaphore = new SemaphoreSlim(1, 1);

        // Configuration
        private int _refreshIntervalSeconds = 15;
        private const int MAX_RETRY_ATTEMPTS = 3;
        private const int INITIAL_RETRY_DELAY_MS = 1000;

        // Cached data
        private List<HistoricalPrice> _cachedData;
        private DateTime _cacheTimestamp = DateTime.MinValue;
        private string _cachedInterval;

        // State properties
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
        private bool _isPaused = false;
        private double _loadingProgress = 0;
        private bool _isProgressIndeterminate = true;
        private string _loadingProgressText = "";

        // Technical Indicators
        private bool _showSMA = false;
        private bool _showEMA = false;
        private bool _showRSI = false;
        private bool _showMACD = false;
        private bool _showBollingerBands = false;
        private bool _showVWAP = false;

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
                OnPropertyChanged(nameof(PriceChangeColor));
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

        public string ApiUsageText => $"API Calls Today: {_dataService?.GetApiUsageCount() ?? 0} | Refresh: {_refreshIntervalSeconds}s";

        public bool IsPaused
        {
            get => _isPaused;
            set
            {
                _isPaused = value;
                OnPropertyChanged(nameof(IsPaused));
                OnPropertyChanged(nameof(PauseButtonText));

                if (value)
                    StopAutoRefresh();
                else
                    StartAutoRefresh();
            }
        }

        public string PauseButtonText => IsPaused ? "? Resume" : "? Pause";

        public bool ShowSMA
        {
            get => _showSMA;
            set
            {
                _showSMA = value;
                OnPropertyChanged(nameof(ShowSMA));
                UpdateIndicators();
            }
        }

        public bool ShowEMA
        {
            get => _showEMA;
            set
            {
                _showEMA = value;
                OnPropertyChanged(nameof(ShowEMA));
                UpdateIndicators();
            }
        }

        public bool ShowRSI
        {
            get => _showRSI;
            set
            {
                _showRSI = value;
                OnPropertyChanged(nameof(ShowRSI));
                UpdateIndicators();
            }
        }

        public bool ShowMACD
        {
            get => _showMACD;
            set
            {
                _showMACD = value;
                OnPropertyChanged(nameof(ShowMACD));
                UpdateIndicators();
            }
        }

        public bool ShowBollingerBands
        {
            get => _showBollingerBands;
            set
            {
                _showBollingerBands = value;
                OnPropertyChanged(nameof(ShowBollingerBands));
                UpdateIndicators();
            }
        }

        public bool ShowVWAP
        {
            get => _showVWAP;
            set
            {
                _showVWAP = value;
                OnPropertyChanged(nameof(ShowVWAP));
                UpdateIndicators();
            }
        }

        public double LoadingProgress
        {
            get => _loadingProgress;
            set
            {
                _loadingProgress = value;
                OnPropertyChanged(nameof(LoadingProgress));
            }
        }

        public bool IsProgressIndeterminate
        {
            get => _isProgressIndeterminate;
            set
            {
                _isProgressIndeterminate = value;
                OnPropertyChanged(nameof(IsProgressIndeterminate));
            }
        }

        public string LoadingProgressText
        {
            get => _loadingProgressText;
            set
            {
                _loadingProgressText = value;
                OnPropertyChanged(nameof(LoadingProgressText));
            }
        }

        public string CurrentInterval
        {
            get => _currentInterval;
            set
            {
                if (_currentInterval != value)
                {
                    _currentInterval = value;
                    OnPropertyChanged(nameof(CurrentInterval));
                    _ = LoadCandlestickDataAsync(forceRefresh: true);
                }
            }
        }

        public int MaxCandles
        {
            get => _maxCandles;
            set
            {
                if (_maxCandles != value)
                {
                    _maxCandles = value;
                    OnPropertyChanged(nameof(MaxCandles));
                    
                    // Reload chart with cached data if available
                    if (_cachedData != null)
                    {
                        UpdateChart(_cachedData);
                    }
                }
            }
        }

        public int RefreshIntervalSeconds
        {
            get => _refreshIntervalSeconds;
            set
            {
                if (_refreshIntervalSeconds != value)
                {
                    _refreshIntervalSeconds = value;
                    OnPropertyChanged(nameof(RefreshIntervalSeconds));
                    OnPropertyChanged(nameof(ApiUsageText));

                    // Restart timer with new interval
                    if (IsAutoRefreshEnabled && !IsPaused)
                    {
                        StopAutoRefresh();
                        StartAutoRefresh();
                    }
                }
            }
        }

        #endregion

        #region Constructor

        public CandlestickChartViewModel(
            string symbol,
            ICandlestickDataService dataService,
            CandlestickChartService chartService,
            TechnicalIndicatorService technicalIndicatorService,
            UserSettingsService userSettingsService,
            LoggingService loggingService)
        {
            _symbol = symbol ?? throw new ArgumentNullException(nameof(symbol));
            _dataService = dataService ?? throw new ArgumentNullException(nameof(dataService));
            _chartService = chartService ?? throw new ArgumentNullException(nameof(chartService));
            _technicalIndicatorService = technicalIndicatorService;
            _userSettingsService = userSettingsService;
            _loggingService = loggingService;

            _cancellationTokenSource = new CancellationTokenSource();

            // Initialize Polly retry policy
            _retryPolicy = Policy
                .Handle<Exception>(ex => !(ex is OperationCanceledException))
                .WaitAndRetryAsync(
                    MAX_RETRY_ATTEMPTS,
                    retryAttempt => TimeSpan.FromMilliseconds(INITIAL_RETRY_DELAY_MS * Math.Pow(2, retryAttempt - 1)),
                    onRetry: (exception, timeSpan, retryCount, context) =>
                    {
                        _loggingService?.Log("Warning", $"Retry {retryCount}/{MAX_RETRY_ATTEMPTS} for {_symbol} after {timeSpan.TotalSeconds:F1}s delay. Error: {exception.Message}");
                    });

            // Load user preferences
            LoadUserPreferences();

            // Initialize chart collections
            _candlestickSeries = new SeriesCollection();
            _volumeSeries = new SeriesCollection();
            _timeLabels = new List<string>();
        }

        #endregion

        #region Data Loading

        /// <summary>
        /// Loads candlestick data asynchronously
        /// </summary>
        public async Task LoadCandlestickDataAsync(bool forceRefresh = false)
        {
            // Request deduplication: Cancel any pending request
            if (_currentLoadTask != null && !_currentLoadTask.IsCompleted)
            {
                _loggingService?.Log("Info", $"Cancelling previous request for {_symbol}");
                _cancellationTokenSource?.Cancel();

                try
                {
                    await _currentLoadTask;
                }
                catch (OperationCanceledException)
                {
                    // Expected
                }
                catch (Exception ex)
                {
                    _loggingService?.LogErrorWithContext(ex, "Error waiting for previous request to complete");
                }
            }

            // Create new cancellation token
            _cancellationTokenSource = new CancellationTokenSource();
            var cancellationToken = _cancellationTokenSource.Token;

            // Use semaphore to ensure only one request at a time
            await _requestSemaphore.WaitAsync(cancellationToken);

            try
            {
                _currentLoadTask = LoadDataInternalAsync(forceRefresh, cancellationToken);
                await _currentLoadTask;
            }
            finally
            {
                _requestSemaphore.Release();
            }
        }

        private async Task LoadDataInternalAsync(bool forceRefresh, CancellationToken cancellationToken)
        {
            IsLoading = true;
            IsDataLoaded = false;
            IsNoData = false;
            LoadingProgress = 0;
            IsProgressIndeterminate = false;
            LoadingProgressText = "Preparing request...";

            try
            {
                _loggingService?.Log("Info", $"Loading candlestick data for {_symbol} with interval {_currentInterval}");

                LoadingProgress = 30;
                LoadingProgressText = "Fetching data...";

                // Execute with retry policy
                var historicalData = await _retryPolicy.ExecuteAsync(async (ct) =>
                {
                    ct.ThrowIfCancellationRequested();
                    return await _dataService.GetCandlestickDataAsync(_symbol, _currentInterval, forceRefresh, ct);
                }, cancellationToken);

                LoadingProgress = 60;
                LoadingProgressText = "Processing data...";

                cancellationToken.ThrowIfCancellationRequested();

                if (historicalData == null || historicalData.Count == 0)
                {
                    throw new NoDataAvailableException(_symbol, _currentInterval);
                }

                LoadingProgress = 80;
                LoadingProgressText = "Rendering chart...";

                // Store in cache
                _cachedData = historicalData;
                _cacheTimestamp = DateTime.Now;
                _cachedInterval = _currentInterval;

                // Update chart
                UpdateChart(historicalData);

                LoadingProgress = 100;
                LoadingProgressText = "Complete!";
                _lastUpdateTime = DateTime.Now;
                IsDataLoaded = true;
                IsNoData = false;

                OnPropertyChanged(nameof(LastUpdateText));
                OnPropertyChanged(nameof(StatusText));

                _loggingService?.Log("Info", $"Successfully loaded {historicalData.Count} candles for {_symbol}");
            }
            catch (NoDataAvailableException)
            {
                _loggingService?.Log("Warning", $"No candlestick data available for {_symbol}");
                IsNoData = true;
            }
            catch (OperationCanceledException)
            {
                _loggingService?.Log("Info", $"Request cancelled for {_symbol}");
                throw;
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, $"Failed to load candlestick data for {_symbol} after {MAX_RETRY_ATTEMPTS} retries");
                IsNoData = true;
            }
            finally
            {
                IsLoading = false;
            }
        }

        /// <summary>
        /// Updates the chart with new data
        /// </summary>
        private void UpdateChart(List<HistoricalPrice> data)
        {
            var (candlestickSeries, volumeSeries, timeLabels) = _chartService.CreateChartSeries(data, _symbol, _maxCandles);

            CandlestickSeries = candlestickSeries;
            VolumeSeries = volumeSeries;
            TimeLabels = timeLabels;

            UpdatePriceInfo(data);
            UpdateIndicators();
        }

        /// <summary>
        /// Updates price information
        /// </summary>
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
            if (_isPaused) return;

            if (_refreshTimer != null)
            {
                _refreshTimer.Stop();
                _refreshTimer = null;
            }

            _refreshTimer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(_refreshIntervalSeconds)
            };
            _refreshTimer.Tick += async (s, e) =>
            {
                if (!_isPaused)
                {
                    await LoadCandlestickDataAsync();
                }
            };
            _refreshTimer.Start();

            _loggingService?.Log("Info", $"Started auto-refresh for {_symbol} with {_refreshIntervalSeconds}s interval");
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

        #region Technical Indicators

        private void UpdateIndicators()
        {
            if (_cachedData == null || _cachedData.Count == 0 || CandlestickSeries == null || _technicalIndicatorService == null)
                return;

            // Implementation here would use TechnicalIndicatorService to add indicator series to the chart
            // This is a placeholder for the full implementation
        }

        #endregion

        #region Settings

        private void LoadUserPreferences()
        {
            try
            {
                var settings = _userSettingsService?.GetUserSettings();
                if (settings != null)
                {
                    _refreshIntervalSeconds = settings.ChartRefreshIntervalSeconds > 0
                        ? settings.ChartRefreshIntervalSeconds
                        : 15;

                    _loggingService?.Log("Info", $"Loaded user preference: Refresh interval = {_refreshIntervalSeconds}s");
                }
            }
            catch (Exception ex)
            {
                _loggingService?.LogErrorWithContext(ex, "Failed to load user preferences, using defaults");
                _refreshIntervalSeconds = 15;
            }
        }

        #endregion

        #region Cleanup

        public void Dispose()
        {
            StopAutoRefresh();
            _cancellationTokenSource?.Cancel();
            _cancellationTokenSource?.Dispose();
            _requestSemaphore?.Dispose();
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
